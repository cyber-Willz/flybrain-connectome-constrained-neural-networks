//! Random Fourier feature input encoding (Tancik et al., "Fourier Features
//! Let Networks Learn High Frequency Functions", 2020), layered in front of
//! a [`crate::pinn::Shpinn`]'s spectral-embedding coordinates.
//!
//! ## Why this crate didn't have it until now
//!
//! Ported from a from-scratch continuous-domain PINN system built
//! independently of this crate (`pde_pinn`, operating on real R^d
//! coordinates with exact autodiff rather than a discrete hypergraph
//! operator), whose own Burgers-equation run hit a well-documented failure
//! mode of plain tanh-MLPs: **spectral bias**, a bias toward representing
//! low-frequency functions that makes sharp, high-frequency features (a
//! near-shock in Burgers; presumably also a high-eigenvalue mode of `Delta`
//! here) disproportionately hard to fit. Fourier features address it by
//! re-expressing the raw coordinates as a bank of sinusoids at fixed,
//! spread-out frequencies *before* the network sees them, so the network
//! only has to learn coefficients on an already-spectral basis rather than
//! bending a low-frequency function into a sharp one.
//!
//! This is a genuinely portable idea even though the two systems' domains
//! are otherwise incompatible ([`crate::physics`]'s residuals need a
//! discrete graph operator, not continuous derivatives, because a
//! hypergraph's vertices aren't points in R^d with well-defined
//! derivatives between them) -- **the encoder only touches the network's
//! input side**. [`crate::pinn::Shpinn`] already accepts an arbitrary
//! input dimension, so wiring this in is: encode the existing spectral
//! embedding coordinates, feed the wider result into an unmodified
//! `Shpinn` (sized for the encoded dimension), and everything downstream
//! (the residual functions, the loss, the training loop) is untouched.
//!
//! ## Honesty about whether it actually helps
//!
//! `pde_pinn`'s own README reports a case where an aggressive frequency
//! scale looked like a clear win by training loss alone but was actually
//! badly overfit/aliased against its sparse collocation grid -- a caution
//! worth repeating here rather than assuming Fourier features are a free
//! win. `examples/burgers_spectral_demo.rs` runs the same verification
//! (independent RK4 ground truth) with and without this encoder and prints
//! both numbers rather than only the flattering one -- on this crate's
//! (small) demo hypergraph, the honest result was that it *didn't* help
//! (see that file's module doc for the measured numbers and the likely
//! reason: the plain network already fits that problem almost exactly, so
//! there was no spectral-bias bottleneck for the encoder to fix). Kept in
//! the crate anyway, tested and ready, for a problem where the bottleneck
//! is real -- see that same doc comment for what that would look like.
//!
//! The frequency matrix is fixed at construction (drawn once from a seeded
//! RNG, not trained), so there is nothing new needed on the backward side:
//! it's an ordinary (if fixed-weight) differentiable Burn tensor op, and
//! Burn's own autodiff differentiates straight through it like any other
//! layer -- unlike `pde_pinn`'s from-scratch AD, which had to hand-derive
//! and finite-difference-check a backward rule for this encoder
//! specifically.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A fixed (non-trainable) bank of random sinusoidal frequencies, applied
/// to a raw coordinate matrix to produce `[raw | sin(2*pi*B.x) |
/// cos(2*pi*B.x)]`.
pub struct FourierFeatureEncoder<B: Backend> {
    /// `[n_features, raw_dim]` frequency matrix `B`.
    freq: Tensor<B, 2>,
    raw_dim: usize,
    n_features: usize,
}

impl<B: Backend> FourierFeatureEncoder<B> {
    /// `n_features` sin/cos pairs, each frequency drawn uniformly from
    /// `[-freq_scale, freq_scale]` per raw input dimension. `freq_scale`
    /// controls the highest frequency the network can represent well --
    /// see this module's docs for the aliasing caveat on a sparse
    /// collocation set (i.e. small hypergraphs) if it's set too high.
    pub fn new(raw_dim: usize, n_features: usize, freq_scale: f64, seed: u64, device: &B::Device) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = vec![0.0f32; n_features * raw_dim];
        for x in data.iter_mut() {
            *x = rng.gen_range(-freq_scale..freq_scale) as f32;
        }
        let freq = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n_features, raw_dim]);
        FourierFeatureEncoder { freq, raw_dim, n_features }
    }

    /// Output width: the raw coordinates plus one sin and one cos per
    /// feature.
    pub fn encoded_dim(&self) -> usize {
        self.raw_dim + 2 * self.n_features
    }

    /// Encodes a `[n, raw_dim]` coordinate matrix into `[n, encoded_dim()]`:
    /// the raw coordinates kept alongside the sinusoids (a "skip
    /// connection", so the network can still represent smooth,
    /// low-frequency behavior directly), followed by `sin(2*pi*B.x)` then
    /// `cos(2*pi*B.x)`, each `[n, n_features]`.
    pub fn encode(&self, raw: Tensor<B, 2>) -> Tensor<B, 2> {
        let bx = raw.clone().matmul(self.freq.clone().transpose()); // [n, n_features]
        let two_pi_bx = bx * (2.0 * std::f64::consts::PI);
        let s = two_pi_bx.clone().sin();
        let c = two_pi_bx.cos();
        Tensor::cat(vec![raw, s, c], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn encoded_dim_and_shape_match() {
        let device = Default::default();
        let encoder = FourierFeatureEncoder::<B>::new(4, 6, 2.0, 7, &device);
        assert_eq!(encoder.encoded_dim(), 4 + 12);

        let raw = Tensor::<B, 2>::zeros([5, 4], &device) + 0.3;
        let encoded = encoder.encode(raw);
        assert_eq!(encoded.dims(), [5, 16]);
    }

    #[test]
    fn raw_coordinates_are_preserved_as_a_skip_connection() {
        let device = Default::default();
        let encoder = FourierFeatureEncoder::<B>::new(3, 2, 1.0, 1, &device);
        let raw_vals: Vec<f32> = vec![0.1, -0.2, 0.5, 0.9, 0.0, -0.4];
        let raw = Tensor::<B, 1>::from_floats(raw_vals.as_slice(), &device).reshape([2, 3]);
        let encoded = encoder.encode(raw).into_data().convert::<f32>();
        for (i, &want) in raw_vals.iter().enumerate() {
            // encoded is [n, encoded_dim] row-major; first 3 columns of
            // each row are the untouched raw coordinates.
            let row = i / 3;
            let col = i % 3;
            let got = encoded.value[row * encoder.encoded_dim() + col];
            assert!((got - want).abs() < 1e-6, "raw coordinate not preserved: got {got}, want {want}");
        }
    }

    #[test]
    fn sin_cos_features_match_hand_computation() {
        let device = Default::default();
        let encoder = FourierFeatureEncoder::<B>::new(2, 3, 4.0, 42, &device);
        let freq_data = encoder.freq.clone().into_data().convert::<f32>();

        let raw_vals = [0.3f32, -0.7f32];
        let raw = Tensor::<B, 1>::from_floats(raw_vals.as_slice(), &device).reshape([1, 2]);
        let encoded = encoder.encode(raw).into_data().convert::<f32>();

        for feat in 0..3 {
            let b0 = freq_data.value[feat * 2];
            let b1 = freq_data.value[feat * 2 + 1];
            let bx = b0 * raw_vals[0] + b1 * raw_vals[1];
            let arg = 2.0 * std::f32::consts::PI * bx;
            let want_sin = arg.sin();
            let want_cos = arg.cos();
            let got_sin = encoded.value[2 + feat];
            let got_cos = encoded.value[2 + 3 + feat];
            assert!((got_sin - want_sin).abs() < 1e-4, "sin feature {feat}: got {got_sin}, want {want_sin}");
            assert!((got_cos - want_cos).abs() < 1e-4, "cos feature {feat}: got {got_cos}, want {want_cos}");
        }
    }

    #[test]
    fn gradients_flow_through_the_encoder() {
        use burn::backend::Autodiff;
        type Be = Autodiff<NdArray<f32>>;

        let device = Default::default();
        let encoder = FourierFeatureEncoder::<Be>::new(2, 4, 3.0, 3, &device);
        let raw = Tensor::<Be, 2>::zeros([3, 2], &device) + 0.2;
        let raw = raw.require_grad();
        let encoded = encoder.encode(raw.clone());
        let loss = encoded.sum();
        let grads = loss.backward();
        let raw_grad = raw.grad(&grads).expect("raw input should have a gradient after backward");
        let data = raw_grad.into_data().convert::<f32>();
        // Should be well-defined (finite) and not identically zero -- both
        // the skip connection and the sin/cos terms contribute.
        assert!(data.value.iter().all(|v| v.is_finite()));
        assert!(data.value.iter().any(|&v| v.abs() > 1e-6));
    }
}
