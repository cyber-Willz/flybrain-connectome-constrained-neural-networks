//! Viscous Burgers' equation on a hypergraph: `du/dt + u du/dx - nu
//! d^2u/dx^2 = 0`, the standard first nonlinear PDE in the PINN literature
//! (Raissi et al. 2019's own worked example) and the simplest testbed for
//! the advection nonlinearity that also appears (per velocity component,
//! plus a pressure coupling) in [`crate::navier_stokes`].
//!
//! ## The nonlinear term has no autodiff analogue here
//!
//! A classical PINN gets `u * du/dx` by autodiffing the network twice
//! w.r.t. its spatial input and multiplying by `u` itself. A SHPINN's
//! domain is discrete, so -- exactly as [`crate::physics`] replaces
//! `d^2u/dx^2` with one application of `Delta` -- the nonlinear advective
//! term is built from [`crate::gradient`]'s discrete gradient/divergence
//! pair, using the equation's **conservative form**
//! `u du/dx = d/dx (u^2 / 2)`:
//!
//! 1. Form the flux `F(u) = u^2 / 2` at each vertex (an ordinary
//!    elementwise square, still inside Burn's autodiff graph since `u` came
//!    from the network).
//! 2. Push it onto hyperedges with the discrete gradient, `G F(u)` --
//!    [`crate::gradient::apply_grad`].
//! 3. Pull it back to vertices with the discrete divergence, `G^T (G
//!    F(u))` -- [`crate::gradient::apply_div`].
//!
//! Step 2+3 composed is `G^T G`, which [`crate::gradient`]'s tests confirm
//! equals `I - Delta` (`Delta` from [`crate::operator`]) -- so this is
//! provably the *same* discretization family the rest of this crate
//! already uses for the linear term, applied to the nonlinear flux instead
//! of to `u` itself, not an unrelated ad hoc construction. It reduces to
//! the textbook conservative finite-volume Burgers discretization
//! `(F_{i+1/2} - F_{i-1/2})/dx` when the hypergraph happens to be a path
//! graph, the same sense in which `Delta` itself reduces to the standard
//! discretization of `-d^2u/dx^2` there.
//!
//! The viscous term keeps using `Delta` directly (Burgers' diffusion is
//! linear), with the same dissipative `+nu * Delta u` sign convention
//! [`crate::physics::heat_residual`] uses.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::gradient::{apply_div, apply_grad};
use crate::operator::apply_laplacian;

/// Viscous Burgers residual: `du/dt + G^T(G(u^2/2)) + nu * Delta u`.
///
/// `laplacian` is [`crate::operator::dense_laplacian_tensor`]'s `[n, n]`
/// output, `grad` is [`crate::gradient::dense_gradient_tensor`]'s `[m, n]`
/// output (both built once per hypergraph, outside the training loop, the
/// same way every other residual in this crate takes its operators
/// pre-built). `u` is `[n, 1]` (Burgers is a scalar equation); `u_t` must
/// already be `du/dt` (see [`crate::physics::central_time_derivative`]).
pub fn burgers_residual<B: Backend>(
    laplacian: &Tensor<B, 2>,
    grad: &Tensor<B, 2>,
    u: Tensor<B, 2>,
    u_t: Tensor<B, 2>,
    nu: f64,
) -> Tensor<B, 2> {
    let flux = u.clone().powf_scalar(2.0) * 0.5;
    let edge_flux = apply_grad(grad, flux);
    let advective = apply_div(grad, edge_flux);
    let diffusive = apply_laplacian(laplacian, u) * nu;
    u_t + advective + diffusive
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;
    use spectral_hypergraph::hypergraph::SpectralHypergraph;

    type B = NdArray<f32>;

    fn small_hypergraph() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..6).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4], ids[5]], 1.0).unwrap();
        b.add_hyperedge(&[ids[0], ids[5]], 0.5).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn zero_field_gives_zero_residual() {
        use crate::gradient::dense_gradient_tensor;
        use crate::operator::dense_laplacian_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();
        let lap = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();

        let u = Tensor::<B, 2>::zeros([n, 1], &device);
        let u_t = Tensor::<B, 2>::zeros([n, 1], &device);
        let residual = burgers_residual::<B>(&lap, &grad, u, u_t, 0.1);
        let data = residual.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!(v.abs() < 1e-6, "zero field should give zero residual, got {v}");
        }
    }

    /// The advective term's construction, `G^T(G(F(u)))`, is provably
    /// `F(u) - Delta(F(u))` (see [`crate::gradient`]'s
    /// `grad_matches_laplacian_identity` test) -- check that identity holds
    /// for the actual nonlinear flux `F(u) = u^2/2` this module uses, not
    /// just for the identity field that module tests with.
    #[test]
    fn advective_term_matches_the_gt_g_equals_i_minus_laplacian_identity() {
        use crate::gradient::dense_gradient_tensor;
        use crate::operator::dense_laplacian_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();
        let lap = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();

        let u_vals: Vec<f32> = vec![0.3, -0.7, 1.2, 0.0, -0.4, 0.9];
        let u = Tensor::<B, 1>::from_floats(u_vals.as_slice(), &device).reshape([n, 1]);
        let u_t = Tensor::<B, 2>::zeros([n, 1], &device);

        let residual = burgers_residual::<B>(&lap, &grad, u.clone(), u_t, 0.0); // nu=0: isolate advective term
        let flux = u.powf_scalar(2.0) * 0.5;
        let expected_advective = flux.clone() - apply_laplacian(&lap, flux);

        let got = residual.into_data().convert::<f32>();
        let want = expected_advective.into_data().convert::<f32>();
        for (g, w) in got.value.iter().zip(want.value.iter()) {
            assert!((g - w).abs() < 1e-4, "advective term mismatch: got {g}, want {w}");
        }
    }

    #[test]
    fn shape_matches_vertex_count() {
        use crate::gradient::dense_gradient_tensor;
        use crate::operator::dense_laplacian_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();
        let lap = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let u = Tensor::<B, 2>::zeros([n, 1], &device) + 0.5;
        let u_t = Tensor::<B, 2>::zeros([n, 1], &device);
        let residual = burgers_residual::<B>(&lap, &grad, u, u_t, 0.05);
        assert_eq!(residual.dims(), [n, 1]);
    }
}
