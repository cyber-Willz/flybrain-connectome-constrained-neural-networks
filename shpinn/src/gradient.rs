//! The vertex <-> hyperedge gradient/divergence operator pair that
//! [`crate::operator::dense_laplacian_tensor`] is secretly built from,
//! exposed here as first-class reusable operators for equations that need
//! *first-order* structure -- a nonlinear advective flux (Burgers) or a
//! genuine vertex/edge field split (Navier-Stokes) -- rather than only the
//! second-order `Delta`.
//!
//! ## Where this comes from
//!
//! [`spectral_hypergraph::laplacian::dense_normalized_laplacian`] builds
//! `Delta = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}`. Factor the middle
//! `core` term as `G^T G` with
//!
//! ```text
//! G = D_e^{-1/2} W^{1/2} H^T D_v^{-1/2}     (m x n, hyperedges x vertices)
//! ```
//!
//! Then `G^T G = D_v^{-1/2} H W^{1/2} D_e^{-1/2} D_e^{-1/2} W^{1/2} H^T
//! D_v^{-1/2} = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}`, exactly the `core`
//! term -- so `Delta = I - G^T G`, and [`grad_matches_laplacian_identity`]
//! (in this module's tests) checks that identity directly against
//! [`crate::operator::dense_laplacian_tensor`] rather than just asserting
//! it in a doc comment.
//!
//! `G` is a **discrete gradient**: it takes a vertex-indexed field `u` (`[n,
//! f]`) and returns an hyperedge-indexed field `G u` (`[m, f]`) built from
//! (normalized) differences of `u` across each hyperedge's members -- the
//! hypergraph analogue of `du/dx` living on cell faces in a finite-volume
//! scheme. `G^T` is the adjoint **discrete divergence**: it takes a
//! hyperedge-indexed field back to a vertex-indexed one. Composing them,
//! `G^T G`, is a divergence-of-a-flux operator -- the piece
//! [`crate::burgers`] uses for the nonlinear advective term, and
//! [`crate::navier_stokes`] uses (via `G` and `G^T` directly, on *separate*
//! vertex-pressure and hyperedge-flow fields) as the pressure-gradient and
//! continuity operators of a network-flow Navier-Stokes analogue.
//!
//! Held dense, exactly like [`crate::operator::dense_laplacian_tensor`]: an
//! `[m, n]` buffer, appropriate for the same hypergraph sizes this whole
//! crate targets.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use spectral_hypergraph::hypergraph::SpectralHypergraph;
use spectral_hypergraph::laplacian::{dense_incidence_matrix, hyperedge_degree_vector, vertex_degree_vector};

use crate::error::Result;

/// Builds the dense `m x n` discrete-gradient tensor `G = D_e^{-1/2}
/// W^{1/2} H^T D_v^{-1/2}` on `device`. Row `e` is hyperedge `e`'s
/// contribution; column `v` is vertex `v`'s.
pub fn dense_gradient_tensor<B: Backend>(
    hg: &SpectralHypergraph,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let h = dense_incidence_matrix(hg)?; // [n, m]
    let n = h.nrows();
    let m = h.ncols();
    let dv = vertex_degree_vector(hg)?;
    let de = hyperedge_degree_vector(hg)?;

    let mut row_major = vec![0.0f32; m * n];
    for e in hg.hyperedge_ids() {
        let scale_e = (hg.hyperedge_weight(e)? / de[e.0]).sqrt();
        for v in hg.hyperedge_members(e)? {
            let scale_v = 1.0 / dv[v.0].sqrt();
            row_major[e.0 * n + v.0] = (h[(v.0, e.0)] * scale_e * scale_v) as f32;
        }
    }
    Ok(Tensor::<B, 1>::from_floats(row_major.as_slice(), device).reshape([m, n]))
}

/// Applies the discrete gradient `G` to a `[n, f]` vertex field, returning
/// the `[m, f]` hyperedge field `G u`.
pub fn apply_grad<B: Backend>(grad: &Tensor<B, 2>, u: Tensor<B, 2>) -> Tensor<B, 2> {
    grad.clone().matmul(u)
}

/// Applies the discrete divergence `G^T` (the adjoint of [`apply_grad`]) to
/// a `[m, f]` hyperedge field, returning the `[n, f]` vertex field `G^T q`.
pub fn apply_div<B: Backend>(grad: &Tensor<B, 2>, q: Tensor<B, 2>) -> Tensor<B, 2> {
    grad.clone().transpose().matmul(q)
}

/// Builds the dense `m x m` "edge Laplacian" `G G^T` -- the natural PSD
/// diffusion operator on hyperedge-indexed fields, used by
/// [`crate::navier_stokes`] as the viscous term for an edge-indexed
/// velocity/flow field, the same role `Delta = I - G^T G` plays for
/// vertex-indexed fields.
pub fn dense_edge_laplacian_tensor<B: Backend>(grad: &Tensor<B, 2>) -> Tensor<B, 2> {
    grad.clone().matmul(grad.clone().transpose())
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
        b.add_hyperedge(&[ids[0], ids[5]], 0.5).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn shapes_match_vertex_and_hyperedge_counts() {
        let hg = small_hypergraph();
        let device = Default::default();
        let g = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        assert_eq!(g.dims(), [hg.num_hyperedges(), hg.num_vertices()]);
    }

    /// The identity this whole module exists to provide: `G^T G = I -
    /// Delta`, checked entrywise against
    /// [`crate::operator::dense_laplacian_tensor`] (itself already
    /// cross-checked against `spectral_hypergraph`'s own reference
    /// implementation in that module's tests) rather than against a second
    /// from-scratch computation of the same formula.
    #[test]
    fn grad_matches_laplacian_identity() {
        use crate::operator::dense_laplacian_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();

        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let laplacian = dense_laplacian_tensor::<B>(&hg, &device).unwrap();

        let identity = Tensor::<B, 1>::from_floats(
            (0..n * n)
                .map(|idx| if idx / n == idx % n { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([n, n]);

        let gt_g = apply_div::<B>(&grad, apply_grad::<B>(&grad, identity.clone()));
        let expected = identity - laplacian;

        let got_data = gt_g.into_data().convert::<f32>();
        let want_data = expected.into_data().convert::<f32>();
        for (got, want) in got_data.value.iter().zip(want_data.value.iter()) {
            assert!((got - want).abs() < 1e-4, "mismatch: got {got}, want {want}");
        }
    }

    #[test]
    fn edge_laplacian_is_square_and_symmetric() {
        let hg = small_hypergraph();
        let device = Default::default();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let m = hg.num_hyperedges();
        let edge_lap = dense_edge_laplacian_tensor::<B>(&grad);
        assert_eq!(edge_lap.dims(), [m, m]);

        let data = edge_lap.clone().into_data().convert::<f32>();
        for i in 0..m {
            for j in 0..m {
                let a = data.value[i * m + j];
                let b = data.value[j * m + i];
                assert!((a - b).abs() < 1e-4, "G G^T should be symmetric: ({i},{j})={a}, ({j},{i})={b}");
            }
        }
    }

    #[test]
    fn apply_grad_of_zero_field_is_zero() {
        let hg = small_hypergraph();
        let device = Default::default();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let n = hg.num_vertices();
        let m = hg.num_hyperedges();
        let zero = Tensor::<B, 2>::zeros([n, 1], &device);
        let out = apply_grad::<B>(&grad, zero);
        assert_eq!(out.dims(), [m, 1]);
        let data = out.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!(v.abs() < 1e-6);
        }
    }
}
