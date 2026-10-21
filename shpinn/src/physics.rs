//! PDE residuals enforced on the hypergraph domain.
//!
//! A classical PINN penalizes `F(u, du/dx, d^2u/dx^2, du/dt, ...) = 0` at
//! collocation points, obtaining every spatial derivative via autodiff
//! through the network w.r.t. its Euclidean input coordinates. A SHPINN's
//! domain is discrete (the hypergraph), so its spatial operator is not a
//! derivative at all — it is the hypergraph's own normalized Laplacian
//! `Delta`, applied once via [`crate::operator::apply_laplacian`]. Time,
//! when a governing equation has it, is still continuous, but `du/dt` is
//! obtained via a central finite difference on the network's own output
//! ([`central_time_derivative`]) rather than autodiff, since Burn 0.13's
//! single-order autodiff can't differentiate a second time through an
//! already-captured gradient — see that function's docs for the resulting
//! semi-discrete ("method of lines") scheme this crate uses instead.
//!
//! Three canonical linear equations are provided, covering the standard
//! PINN benchmark trio (elliptic / parabolic / hyperbolic) transplanted onto
//! a hypergraph:
//!
//! * [`poisson_residual`] -- `Delta u - f = 0` (steady-state / elliptic).
//! * [`heat_residual`] -- `du/dt + alpha * Delta u = 0` (diffusion /
//!   parabolic; `Delta` is PSD, so `+alpha*Delta` is the dissipative sign).
//! * [`wave_residual`] -- `d^2u/dt^2 + c^2 * Delta u = 0` (hyperbolic).
//!
//! All three are `MSE`-ready: each returns the pointwise residual tensor
//! (same shape as `u`), which [`crate::loss`] squares and averages.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::operator::apply_laplacian;

/// Steady-state / elliptic residual: `Delta u - f`. `laplacian` is
/// [`crate::operator::dense_laplacian_tensor`]'s `[n, n]` output; `u` and
/// `f` are `[n, d]` (the network's predicted field and the known forcing
/// term, resp.). No autodiff w.r.t. any input is needed here — `u` already
/// came from the network's forward pass, so gradients w.r.t. the network's
/// weights flow back through this matmul automatically when the caller
/// calls `.backward()` on the loss built from it.
pub fn poisson_residual<B: Backend>(
    laplacian: &Tensor<B, 2>,
    u: Tensor<B, 2>,
    f: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    apply_laplacian(laplacian, u) - f.clone()
}

/// Diffusion / parabolic residual: `du/dt + alpha * Delta u`.
///
/// `u_t` must already be `du/dt` (see [`time_derivative`] for how to obtain
/// it via autodiff); this function does not differentiate anything itself,
/// it only assembles the residual out of pieces the caller computed.
pub fn heat_residual<B: Backend>(
    laplacian: &Tensor<B, 2>,
    u: Tensor<B, 2>,
    u_t: Tensor<B, 2>,
    alpha: f64,
) -> Tensor<B, 2> {
    u_t + apply_laplacian(laplacian, u) * alpha
}

/// Wave / hyperbolic residual: `d^2u/dt^2 + c^2 * Delta u`.
pub fn wave_residual<B: Backend>(
    laplacian: &Tensor<B, 2>,
    u: Tensor<B, 2>,
    u_tt: Tensor<B, 2>,
    c2: f64,
) -> Tensor<B, 2> {
    u_tt + apply_laplacian(laplacian, u) * c2
}

/// Helmholtz residual: `Delta u - k^2 * u - f` (steady-state / elliptic,
/// like [`poisson_residual`], but with a reaction term `-k^2 u` added --
/// the frequency-domain reduction of the wave equation, `d^2u/dt^2 = -c^2
/// Delta u` under a time-harmonic ansatz `u(x,t) = u(x) exp(i omega t)`
/// gives `-omega^2 u = -c^2 Delta u`, i.e. `Delta u - k^2 u = 0` with `k =
/// omega / c`.
///
/// Unlike Poisson's `Delta` (PSD, so `Delta u = f` is always uniquely
/// solvable away from the zero mode), `Delta - k^2 I` is indefinite once
/// `k^2` exceeds `Delta`'s smallest nonzero eigenvalue, and becomes exactly
/// singular whenever `k^2` lands on one of `Delta`'s eigenvalues (a
/// resonance / eigenfrequency of the hypergraph). This is a genuine
/// property of the equation, not a bug in the residual: pick `k` away from
/// `Delta`'s spectrum (checkable via
/// [`spectral_hypergraph::spectral::dense_eigen`]) for a well-posed
/// boundary-value problem, the same caveat a classical mesh-based Helmholtz
/// solver carries.
pub fn helmholtz_residual<B: Backend>(
    laplacian: &Tensor<B, 2>,
    u: Tensor<B, 2>,
    f: &Tensor<B, 2>,
    k2: f64,
) -> Tensor<B, 2> {
    apply_laplacian(laplacian, u.clone()) - u * k2 - f.clone()
}

/// Central-difference estimate of `du/dt`, from the network's own output at
/// `t + eps` and `t - eps`: `(u_plus - u_minus) / (2 * eps)`.
///
/// A "true" PINN would get this via autodiff w.r.t. a continuous time
/// input, the way it gets spatial derivatives on a Euclidean domain. Burn
/// 0.13's autodiff is single-order (differentiating a second time through a
/// gradient tensor captured by an earlier `.backward()` call is not
/// supported), so instead of a real second-order backward pass, this crate
/// uses a **semi-discrete ("method of lines") scheme**: exact in space (the
/// true hypergraph Laplacian / non-backtracking operator, applied via
/// ordinary matmul, exactly as elsewhere in this crate) and
/// finite-difference in time. `u_plus`/`u_minus` are just two more forward
/// passes of [`crate::pinn::Shpinn`] at shifted time inputs, so the whole
/// residual built from this function's output is still only ever
/// *first*-order differentiated w.r.t. the network's weights when the
/// training loop calls `.backward()` on the total loss — no double
/// backprop anywhere in this crate.
pub fn central_time_derivative<B: Backend>(u_plus: Tensor<B, 2>, u_minus: Tensor<B, 2>, eps: f64) -> Tensor<B, 2> {
    (u_plus - u_minus) / (2.0 * eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;

    #[test]
    fn poisson_residual_is_zero_for_the_zero_mode_with_zero_forcing() {
        use crate::operator::dense_laplacian_tensor;
        use spectral_hypergraph::laplacian::dense_normalized_laplacian;
        use spectral_hypergraph::spectral::dense_eigen;

        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..5).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4]], 1.0).unwrap();
        let hg = b.build().unwrap();

        type B = NdArray<f32>;
        let device = Default::default();
        let lap_t = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let lap = dense_normalized_laplacian(&hg).unwrap();
        let eig = dense_eigen(&lap);
        let n = lap.nrows();
        let zero_mode: Vec<f32> = eig.eigenvectors.column(0).iter().map(|&v| v as f32).collect();

        let u = Tensor::<B, 1>::from_floats(zero_mode.as_slice(), &device).reshape([n, 1]);
        let f = Tensor::<B, 2>::zeros([n, 1], &device);
        let residual = poisson_residual::<B>(&lap_t, u, &f);
        let data = residual.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!(v.abs() < 1e-4, "residual should vanish for the zero mode with zero forcing, got {v}");
        }
    }

    #[test]
    fn helmholtz_residual_vanishes_at_an_eigenmode_with_matching_k2() {
        // Delta v_k = lambda_k v_k, so with u = v_k, f = 0, k2 = lambda_k:
        // Delta u - k2*u - f = lambda_k*v_k - lambda_k*v_k - 0 = 0 exactly.
        use crate::operator::dense_laplacian_tensor;
        use spectral_hypergraph::laplacian::dense_normalized_laplacian;
        use spectral_hypergraph::spectral::dense_eigen;

        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..5).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4]], 1.0).unwrap();
        let hg = b.build().unwrap();

        type B = NdArray<f32>;
        let device = Default::default();
        let lap_t = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let lap = dense_normalized_laplacian(&hg).unwrap();
        let eig = dense_eigen(&lap);
        let n = lap.nrows();

        // Use a non-trivial (non-zero-mode) eigenpair, k = 2.
        let lambda_k = eig.eigenvalues[2];
        let mode: Vec<f32> = eig.eigenvectors.column(2).iter().map(|&v| v as f32).collect();

        let u = Tensor::<B, 1>::from_floats(mode.as_slice(), &device).reshape([n, 1]);
        let f = Tensor::<B, 2>::zeros([n, 1], &device);
        let residual = helmholtz_residual::<B>(&lap_t, u, &f, lambda_k);
        let data = residual.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!(v.abs() < 1e-3, "residual should vanish at a matching eigenmode, got {v}");
        }
    }

    #[test]
    fn helmholtz_residual_nonzero_when_k2_does_not_match_any_field() {
        use crate::operator::dense_laplacian_tensor;

        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..5).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4]], 1.0).unwrap();
        let hg = b.build().unwrap();

        type B = NdArray<f32>;
        let device = Default::default();
        let lap_t = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let n = hg.num_vertices();

        let u = Tensor::<B, 1>::from_floats(vec![1.0f32, 0.0, 0.0, 0.0, 0.0].as_slice(), &device).reshape([n, 1]);
        let f = Tensor::<B, 2>::zeros([n, 1], &device);
        let residual = helmholtz_residual::<B>(&lap_t, u, &f, 5.0);
        let data = residual.into_data().convert::<f32>();
        assert!(data.value.iter().any(|&v| v.abs() > 1e-2), "residual should generally be nonzero for an arbitrary field");
    }

    #[test]
    fn central_time_derivative_of_a_linear_function_recovers_its_coefficient() {
        // field(t) = 3 * t: central difference should recover slope 3
        // exactly (linear function -> zero truncation error).
        type B = NdArray<f32>;
        let device = Default::default();
        let n = 6;
        let t0 = 0.2;
        let eps = 1e-3;
        let plus: Vec<f32> = vec![3.0 * (t0 + eps) as f32; n];
        let minus: Vec<f32> = vec![3.0 * (t0 - eps) as f32; n];
        let u_plus = Tensor::<B, 1>::from_floats(plus.as_slice(), &device).reshape([n, 1]);
        let u_minus = Tensor::<B, 1>::from_floats(minus.as_slice(), &device).reshape([n, 1]);
        let dudt = central_time_derivative::<B>(u_plus, u_minus, eps);
        let data = dudt.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!((v - 3.0).abs() < 1e-3, "expected d/dt = 3.0, got {v}");
        }
    }
}
