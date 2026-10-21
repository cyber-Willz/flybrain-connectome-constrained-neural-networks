//! Spectral hypergraph embeddings: the coordinate system a SHPINN's network
//! actually sees.
//!
//! A classical PINN takes Euclidean spatial coordinates `(x, y, ...)` as
//! input. A hypergraph has no such coordinates — so this module builds an
//! analogous continuous coordinate system out of the hypergraph's own
//! spectral geometry: the [Laplacian eigenmap](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Laplacian_eigenmaps)
//! of `hg`'s **true** normalized hypergraph Laplacian
//! (Zhou/Huang/Schölkopf), i.e. the `k` eigenvectors following the trivial
//! zero mode. Two vertices that are spectrally close (connected through many
//! short, low-cardinality hyperedges) end up close together in this
//! embedding, exactly the property a PINN's smooth network needs its input
//! coordinates to have: nearby collocation points should have similar
//! solution values.
//!
//! This is deliberately built on the hypergraph's own Laplacian rather than
//! on a clique-expansion of it (the approach [`nbsc::hypergraph_bridge`]
//! uses to reuse the plain-graph NBSC/GCN pipeline) — a SHPINN wants its
//! *physics* (see [`crate::physics`]) and its *coordinates* to come from the
//! same operator, so the network's smoothness prior and the residual it is
//! penalized against are geometrically consistent.

use nalgebra::{DMatrix, DVector};

use spectral_hypergraph::hypergraph::SpectralHypergraph;
use spectral_hypergraph::laplacian::{dense_normalized_laplacian, HypergraphOperator};
use spectral_hypergraph::operator::LinearOperator;
use spectral_hypergraph::spectral::{dense_eigen, lanczos_smallest};

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::error::{Result, ShpinnError};

/// A `k`-dimensional spectral coordinate system over `hg`'s `n` vertices,
/// plus the eigenvalues (squared spatial "frequencies") those coordinates
/// carry — useful for reporting the effective bandwidth of a trained SHPINN,
/// the same way a Fourier-feature PINN reports which frequencies it used.
#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    /// `n x k` matrix of per-vertex coordinates, columns ordered by
    /// ascending eigenvalue (lowest-frequency / most-global mode first).
    pub coords: DMatrix<f64>,
    /// The `k` corresponding eigenvalues, ascending, all strictly positive
    /// (the zero mode is always dropped).
    pub eigenvalues: DVector<f64>,
}

impl SpectralEmbedding {
    pub fn n(&self) -> usize {
        self.coords.nrows()
    }

    pub fn k(&self) -> usize {
        self.coords.ncols()
    }

    /// Per-vertex coordinate row `i` as a `Vec<f64>`.
    pub fn row(&self, i: usize) -> Vec<f64> {
        self.coords.row(i).iter().copied().collect()
    }

    /// Flattens to a `[n, k]` Burn tensor (row-major, `f32`) on `device`,
    /// ready to feed into [`crate::pinn::Shpinn::forward`].
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
        let n = self.n();
        let k = self.k();
        let mut data = Vec::with_capacity(n * k);
        for i in 0..n {
            for j in 0..k {
                data.push(self.coords[(i, j)] as f32);
            }
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, k])
    }
}

/// Dense Laplacian-eigenmap embedding, `O(n^3)`: full symmetric
/// eigendecomposition of `dense_normalized_laplacian(hg)`, keeping the `k`
/// eigenvectors immediately after the zero mode. Fine up to a few thousand
/// vertices, matching [`spectral_hypergraph`]'s own guidance for
/// `dense_eigen`. Use [`laplacian_eigenmap_sparse`] beyond that.
pub fn laplacian_eigenmap(hg: &SpectralHypergraph, k: usize) -> Result<SpectralEmbedding> {
    let n = hg.num_vertices();
    if k == 0 || k >= n {
        return Err(ShpinnError::TooManyEmbeddingDims {
            requested: k,
            available: n.saturating_sub(1),
        });
    }
    let lap = dense_normalized_laplacian(hg)?;
    let eig = dense_eigen(&lap);
    let coords = eig.eigenvectors.columns(1, k).into_owned();
    let eigenvalues = eig.eigenvalues.rows(1, k).into_owned();
    Ok(SpectralEmbedding { coords, eigenvalues })
}

/// Matrix-free Laplacian-eigenmap embedding via [`lanczos_smallest`] on
/// [`HypergraphOperator`] — the path to use for hypergraphs too large for
/// `laplacian_eigenmap`'s dense `O(n^3)` eigendecomposition, exactly as
/// `spectral_hypergraph` itself recommends for `fiedler_vector`-scale
/// problems. Requests `k + 1` Ritz pairs internally (the zero mode plus the
/// `k` you asked for) and drops the first column.
pub fn laplacian_eigenmap_sparse(
    hg: &SpectralHypergraph,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<SpectralEmbedding> {
    let op = HypergraphOperator::new(hg)?;
    let n = op.dim();
    if k == 0 || k >= n {
        return Err(ShpinnError::TooManyEmbeddingDims {
            requested: k,
            available: n.saturating_sub(1),
        });
    }
    let eig = lanczos_smallest(&op, k + 1, max_iter, tol, seed)?;
    let coords = eig.eigenvectors.columns(1, k).into_owned();
    let eigenvalues = eig.eigenvalues.rows(1, k).into_owned();
    Ok(SpectralEmbedding { coords, eigenvalues })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;

    /// Two triangles bridged by one shared-pair hyperedge: connected, no
    /// isolated vertices, small enough to eigendecompose both ways and
    /// compare.
    fn bridged_triangles() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..6).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[3], ids[4], ids[5]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3]], 1.0).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn dense_embedding_has_requested_shape_and_positive_eigenvalues() {
        let hg = bridged_triangles();
        let emb = laplacian_eigenmap(&hg, 3).unwrap();
        assert_eq!(emb.n(), 6);
        assert_eq!(emb.k(), 3);
        for i in 0..3 {
            assert!(emb.eigenvalues[i] > 1e-9, "eigenvalue {i} should be strictly positive (zero mode must be dropped)");
        }
        // Ascending order.
        for i in 1..3 {
            assert!(emb.eigenvalues[i] + 1e-9 >= emb.eigenvalues[i - 1]);
        }
    }

    #[test]
    fn sparse_embedding_matches_dense_ground_truth() {
        let hg = bridged_triangles();
        let dense = laplacian_eigenmap(&hg, 2).unwrap();
        let sparse = laplacian_eigenmap_sparse(&hg, 2, 20, 1e-9, 7).unwrap();

        for i in 0..2 {
            assert!(
                (dense.eigenvalues[i] - sparse.eigenvalues[i]).abs() < 1e-6,
                "eigenvalue {i}: dense={} sparse={}",
                dense.eigenvalues[i],
                sparse.eigenvalues[i]
            );
        }
        // Eigenvectors can differ by sign; compare up to sign per column.
        for j in 0..2 {
            let dcol = dense.coords.column(j);
            let scol = sparse.coords.column(j);
            let same_sign_err: f64 = (dcol - scol).norm();
            let flip_sign_err: f64 = (dcol + scol).norm();
            assert!(
                same_sign_err.min(flip_sign_err) < 1e-4,
                "column {j} mismatch: same_sign_err={same_sign_err} flip_sign_err={flip_sign_err}"
            );
        }
    }

    #[test]
    fn requesting_too_many_dims_errors_instead_of_panicking() {
        let hg = bridged_triangles();
        let err = laplacian_eigenmap(&hg, 6).unwrap_err();
        assert!(matches!(err, ShpinnError::TooManyEmbeddingDims { .. }));
    }

    #[test]
    fn to_tensor_shape_matches_embedding() {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        let hg = bridged_triangles();
        let emb = laplacian_eigenmap(&hg, 3).unwrap();
        let device = Default::default();
        let t = emb.to_tensor::<B>(&device);
        assert_eq!(t.dims(), [6, 3]);
    }
}
