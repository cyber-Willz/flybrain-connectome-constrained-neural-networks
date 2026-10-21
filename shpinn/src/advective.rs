//! Optional non-backtracking spatial operator (feature `nbsc-ops`).
//!
//! [`crate::physics`]'s Poisson/heat/wave residuals all use the symmetric
//! normalized hypergraph Laplacian `Delta` as the spatial operator, which is
//! the right choice for isotropic diffusion/vibration equations — but
//! `Delta` is blind to *oriented-cycle* / directional structure by
//! construction (it is symmetric; a symmetric matrix cannot encode a
//! preferred direction of flow around a cycle). `nbsc`'s whole premise is
//! that the Hashimoto / non-backtracking matrix `B` — non-symmetric,
//! spectrum genuinely complex for directed cyclic structure — captures
//! exactly what `Delta` misses. This module swaps the spatial operator in a
//! SHPINN's advection-type residual from `Delta` to `nbsc`'s learnable
//! rescaled non-backtracking recursion, for governing equations where
//! directionality is physically real (transport/advection along a
//! preferred orientation, rather than isotropic diffusion).
//!
//! This is deliberately a *separate*, optional spatial operator rather than
//! a variant of [`crate::operator::dense_laplacian_tensor`]: `B` is
//! generally non-symmetric, so it is not a drop-in replacement for `Delta`
//! in the Poisson/heat/wave residuals (whose derivation assumes a symmetric
//! PSD operator) — it is the right operator for a *different* equation
//! (advection), not a different discretization of the same one.

use nbsc::burn_layer::{dense_adjacency_tensor, degrees_tensor};
use nbsc::graph::Graph;
use nbsc::hypergraph_bridge::clique_expand;
use nbsc::spectral::estimate_spectral_radius;

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use spectral_hypergraph::hypergraph::SpectralHypergraph;

/// The order-1 rescaled non-backtracking transport operator, `T_1 X = (A /
/// rho_B) X` — exactly [`nbsc::burn_layer::NbscLayer`]'s first filter tap,
/// pulled out standalone so it can serve as a SHPINN spatial operator rather
/// than a hidden step inside a learnable graph-conv layer. `rho_B` (the
/// spectral radius of the linearized Hashimoto operator) is estimated once
/// per hypergraph via `krylov_ds` Arnoldi on the clique-expanded graph,
/// exactly the way `NbscLayerConfig::init` does it.
pub struct NonBacktrackingOperator<B: Backend> {
    adjacency: Tensor<B, 2>,
    rho_b: f64,
}

impl<B: Backend> NonBacktrackingOperator<B> {
    /// Clique-expands `hg` (via [`nbsc::hypergraph_bridge::clique_expand`])
    /// and builds the dense adjacency tensor + `rho_B` estimate on the
    /// result. `krylov_dim` and `seed` are passed straight through to
    /// [`estimate_spectral_radius`].
    pub fn build(hg: &SpectralHypergraph, krylov_dim: usize, seed: u64, device: &B::Device) -> Self {
        let g: Graph = clique_expand(hg);
        let rho_b = estimate_spectral_radius(&g, krylov_dim.min(g.n), seed).max(1e-6);
        let adjacency = dense_adjacency_tensor::<B>(&g, device);
        let _ = degrees_tensor::<B>(&g, device); // available for higher-order taps if extended later
        Self { adjacency, rho_b }
    }

    pub fn rho_b(&self) -> f64 {
        self.rho_b
    }

    /// Applies `T_1` to a `[n, f]` field tensor.
    pub fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.adjacency.clone().matmul(x) / self.rho_b
    }
}

/// Advection-type residual: `du/dt + T_1(u) - f`, the non-backtracking
/// analogue of [`crate::physics::heat_residual`]. `u_t` must already be
/// `du/dt` (see [`crate::physics::time_derivative`]).
pub fn advective_residual<B: Backend>(
    op: &NonBacktrackingOperator<B>,
    u: Tensor<B, 2>,
    u_t: Tensor<B, 2>,
    f: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    u_t + op.apply(u) - f.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;

    type B = NdArray<f32>;

    fn small_hypergraph() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..6).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4], ids[5]], 1.0).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn operator_builds_with_positive_finite_rho_b() {
        let hg = small_hypergraph();
        let device = Default::default();
        let op = NonBacktrackingOperator::<B>::build(&hg, 20, 3, &device);
        assert!(op.rho_b().is_finite() && op.rho_b() > 0.0);
    }

    #[test]
    fn apply_preserves_shape() {
        let hg = small_hypergraph();
        let device = Default::default();
        let op = NonBacktrackingOperator::<B>::build(&hg, 20, 3, &device);
        let x = Tensor::<B, 2>::zeros([6, 2], &device) + 1.0;
        let out = op.apply(x);
        assert_eq!(out.dims(), [6, 2]);
    }
}
