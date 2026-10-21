//! Incompressible Navier-Stokes on a hypergraph, in **network-flow form**:
//! velocity lives on hyperedges (one scalar "flow" per hyperedge), pressure
//! lives on vertices, and the two are coupled through the
//! [`crate::gradient`] operator pair -- the same reduction real network
//! fluid models use (water-distribution and gas-pipeline network flow,
//! 1-D blood-flow network models), all of which are themselves derived from
//! full Navier-Stokes by averaging across each conduit's cross-section and
//! keeping only the along-conduit flow as the unknown. It is that reduction
//! this module gives a hypergraph domain, not a full 2-D/3-D velocity
//! field -- a hypergraph has no ambient space for a literal velocity
//! *vector* to live in, the same gap [`crate::embedding`]'s doc comment
//! discusses for coordinates in general.
//!
//! ```text
//! momentum:   dq/dt + q |q| + nu * (G G^T) q + G p = forcing   (per hyperedge)
//! continuity: G^T q = 0                                         (per vertex)
//! ```
//!
//! * `q` (`[m, 1]`) is the flow through each hyperedge; `p` (`[n, 1]`) is
//!   the pressure at each vertex.
//! * `q |q|` is the standard quadratic inertial/friction self-interaction
//!   term used in network-flow models in place of a literal `(u . grad)u`
//!   (which needs an ambient direction a single per-edge scalar doesn't
//!   have) -- same role as `u du/dx` in [`crate::burgers`], reduced to one
//!   dimension per hyperedge instead of built from the gradient/divergence
//!   pair, since here there is exactly one flow unknown per edge rather
//!   than a vertex-indexed field to differentiate.
//! * `G G^T` (built once via [`crate::gradient::dense_edge_laplacian_tensor`])
//!   is the edge-indexed analogue of `Delta`: PSD, so `+nu * (G G^T) q`
//!   is the same dissipative sign [`crate::physics::heat_residual`] and
//!   [`crate::burgers::burgers_residual`] use for their viscous terms.
//! * `G p` ([`crate::gradient::apply_grad`]) pushes vertex pressure onto
//!   hyperedges -- pressure differences across a hyperedge drive flow
//!   through it, exactly as a pressure gradient drives flow in the
//!   continuous equation.
//! * `G^T q` ([`crate::gradient::apply_div`]) is the discrete divergence of
//!   the flow field back onto vertices: incompressibility says this must
//!   vanish everywhere -- net flow into every vertex balances net flow out,
//!   the graph analogue of `div u = 0`.
//!
//! Because `q` and `p` live on different index sets (hyperedges vs.
//! vertices), they need separate [`crate::pinn::Shpinn`] networks with
//! separate input coordinates. [`hyperedge_centroid_coords`] builds `q`'s
//! coordinates the natural way that keeps them geometrically consistent
//! with `p`'s vertex coordinates: the mean of each hyperedge's member
//! vertices' own spectral-embedding coordinates, in the same embedding
//! [`crate::embedding::laplacian_eigenmap`] already builds for `p`.

use nalgebra::DMatrix;

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use spectral_hypergraph::hypergraph::SpectralHypergraph;

use crate::embedding::SpectralEmbedding;
use crate::error::Result;
use crate::gradient::{apply_div, apply_grad};
use crate::operator::apply_laplacian; // generic matmul-by-operator, reused for G G^T too

/// Mean of each hyperedge's member vertices' spectral-embedding
/// coordinates, `[m, k]`. This is `q`'s (the hyperedge-indexed flow
/// field's) input coordinate system, built from the same embedding `p`
/// (the vertex-indexed pressure field) uses, so both networks' smoothness
/// priors come from one consistent spectral geometry.
pub fn hyperedge_centroid_coords_matrix(
    hg: &SpectralHypergraph,
    embedding: &SpectralEmbedding,
) -> Result<DMatrix<f64>> {
    let m = hg.num_hyperedges();
    let k = embedding.k();
    let mut coords = DMatrix::<f64>::zeros(m, k);
    for e in hg.hyperedge_ids() {
        let members = hg.hyperedge_members(e)?;
        let mut acc = vec![0.0f64; k];
        for v in &members {
            let row = embedding.row(v.0);
            for j in 0..k {
                acc[j] += row[j];
            }
        }
        let count = members.len().max(1) as f64;
        for j in 0..k {
            coords[(e.0, j)] = acc[j] / count;
        }
    }
    Ok(coords)
}

/// [`hyperedge_centroid_coords_matrix`], flattened to a `[m, k]` Burn
/// tensor on `device`, ready to feed into a [`crate::pinn::Shpinn`] for
/// `q`.
pub fn hyperedge_centroid_coords<B: Backend>(
    hg: &SpectralHypergraph,
    embedding: &SpectralEmbedding,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let coords = hyperedge_centroid_coords_matrix(hg, embedding)?;
    let m = coords.nrows();
    let k = coords.ncols();
    let mut data = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            data.push(coords[(i, j)] as f32);
        }
    }
    Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([m, k]))
}

/// Momentum residual: `dq/dt + q|q| + nu * (G G^T) q + G p - forcing`.
///
/// `edge_laplacian` is [`crate::gradient::dense_edge_laplacian_tensor`]'s
/// `[m, m]` output, `grad` is
/// [`crate::gradient::dense_gradient_tensor`]'s `[m, n]` output. `q` is
/// `[m, 1]`; `q_t` must already be `dq/dt` (pass zeros for a steady-state
/// problem). `p` is `[n, 1]`. `forcing` is `[m, 1]`, an external body force
/// per hyperedge (e.g. a pump); pass zeros for an unforced flow.
pub fn momentum_residual<B: Backend>(
    edge_laplacian: &Tensor<B, 2>,
    grad: &Tensor<B, 2>,
    q: Tensor<B, 2>,
    q_t: Tensor<B, 2>,
    p: Tensor<B, 2>,
    forcing: &Tensor<B, 2>,
    nu: f64,
) -> Tensor<B, 2> {
    let inertial = q.clone() * q.clone().abs();
    let viscous = apply_laplacian(edge_laplacian, q) * nu;
    let pressure_term = apply_grad(grad, p);
    q_t + inertial + viscous + pressure_term - forcing.clone()
}

/// Continuity (incompressibility) residual: `G^T q`, the discrete
/// divergence of the flow field. Should vanish at every vertex for an
/// incompressible flow; unlike [`momentum_residual`] this has no time
/// derivative or forcing term to combine it with -- it is enforced as its
/// own physics term (mean-squared, via [`crate::loss::mse`]) in the total
/// loss.
pub fn continuity_residual<B: Backend>(grad: &Tensor<B, 2>, q: Tensor<B, 2>) -> Tensor<B, 2> {
    apply_div(grad, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;

    type B = NdArray<f32>;

    fn small_hypergraph() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..7).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4]], 1.0).unwrap();
        b.add_hyperedge(&[ids[4], ids[5], ids[6]], 1.0).unwrap();
        b.add_hyperedge(&[ids[6], ids[0]], 0.5).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn centroid_coords_shape_and_averaging() {
        use crate::embedding::laplacian_eigenmap;

        let hg = small_hypergraph();
        let emb = laplacian_eigenmap(&hg, 3).unwrap();
        let coords = hyperedge_centroid_coords_matrix(&hg, &emb).unwrap();
        assert_eq!(coords.nrows(), hg.num_hyperedges());
        assert_eq!(coords.ncols(), 3);

        // Hyperedge 0 = {v0, v1, v2}: its centroid row must equal the mean
        // of those three vertices' own embedding rows.
        let r0 = emb.row(0);
        let r1 = emb.row(1);
        let r2 = emb.row(2);
        for j in 0..3 {
            let want = (r0[j] + r1[j] + r2[j]) / 3.0;
            assert!((coords[(0, j)] - want).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_flow_and_zero_pressure_gives_zero_momentum_residual() {
        use crate::gradient::{dense_edge_laplacian_tensor, dense_gradient_tensor};

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();
        let m = hg.num_hyperedges();

        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let edge_lap = dense_edge_laplacian_tensor::<B>(&grad);

        let q = Tensor::<B, 2>::zeros([m, 1], &device);
        let q_t = Tensor::<B, 2>::zeros([m, 1], &device);
        let p = Tensor::<B, 2>::zeros([n, 1], &device);
        let forcing = Tensor::<B, 2>::zeros([m, 1], &device);

        let residual = momentum_residual::<B>(&edge_lap, &grad, q, q_t, p, &forcing, 0.1);
        let data = residual.into_data().convert::<f32>();
        for &v in data.value.iter() {
            assert!(v.abs() < 1e-6, "zero fields should give zero momentum residual, got {v}");
        }
    }

    #[test]
    fn continuity_residual_shape_matches_vertex_count() {
        use crate::gradient::dense_gradient_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let n = hg.num_vertices();
        let m = hg.num_hyperedges();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        let q = Tensor::<B, 2>::zeros([m, 1], &device) + 0.3;
        let residual = continuity_residual::<B>(&grad, q);
        assert_eq!(residual.dims(), [n, 1]);
    }

    #[test]
    fn nonzero_flow_generally_violates_continuity() {
        use crate::gradient::dense_gradient_tensor;

        let hg = small_hypergraph();
        let device = Default::default();
        let m = hg.num_hyperedges();
        let grad = dense_gradient_tensor::<B>(&hg, &device).unwrap();
        // An arbitrary (non-harmonic) flow field should not, in general,
        // satisfy incompressibility -- sanity check that the residual is
        // actually sensitive to q, not trivially zero for any input.
        let q_vals: Vec<f32> = (0..m).map(|i| 0.2 + 0.1 * i as f32).collect();
        let q = Tensor::<B, 1>::from_floats(q_vals.as_slice(), &device).reshape([m, 1]);
        let residual = continuity_residual::<B>(&grad, q);
        let data = residual.into_data().convert::<f32>();
        assert!(data.value.iter().any(|&v| v.abs() > 1e-3));
    }
}
