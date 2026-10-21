//! The hypergraph's normalized Laplacian, materialized as a dense Burn
//! tensor so it can sit inside Burn's autodiff graph.
//!
//! This is the SHPINN analogue of a classical PINN's spatial-derivative
//! operator: where a Euclidean PINN gets `d^2u/dx^2` via
//! `torch.autograd.grad` twice, a SHPINN's domain has no ambient space to
//! differentiate with respect to — the hypergraph *is* the domain, and its
//! own normalized Laplacian `Delta` already **is** the correct discrete
//! second-order differential operator on it (Zhou/Huang/Schölkopf's
//! construction reduces to the standard graph Laplacian discretization of
//! `-Delta_continuous` when the hypergraph is an ordinary graph). So a
//! SHPINN never needs to autodiff through space: it applies `Delta` to its
//! own output via ordinary matrix multiplication, which Burn already knows
//! how to backpropagate through, exactly the same trick
//! [`nbsc::burn_layer::dirichlet_energy_differentiable`] uses for its energy
//! regularizer — this module is that same pattern, generalized from a
//! clique-expansion Laplacian to the true hypergraph Laplacian, and exposed
//! as a first-class reusable operator rather than inlined into one loss
//! term.
//!
//! Held dense, like [`nbsc::burn_layer::dense_adjacency_tensor`]: correct
//! and simple for the hypergraph sizes a SHPINN is meant for (the network
//! itself already only has as many parameters as its MLP width — the
//! hypergraph size shows up only in this `n x n` buffer and in the `[n, *]`
//! feature tensors that flow through it). For hypergraphs too large to hold
//! `Delta` densely, use [`crate::embedding::laplacian_eigenmap_sparse`] for
//! the coordinates and fall back to a forward-only (non-differentiable)
//! residual check via [`spectral_hypergraph::laplacian::HypergraphOperator`]
//! directly, the same tradeoff `nbsc`'s own README documents for its dense
//! `NbscLayer`.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use spectral_hypergraph::hypergraph::SpectralHypergraph;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;

use crate::error::Result;

/// Builds the dense `n x n` normalized hypergraph Laplacian `Delta` as a
/// Burn tensor on `device`. Symmetric PSD with `Delta * (D_v^{1/2} * 1) = 0`
/// exactly as [`spectral_hypergraph::laplacian::dense_normalized_laplacian`]
/// documents; this function only changes the representation (`nalgebra`
/// `DMatrix<f64>` -> Burn `Tensor<B, 2>`, `f64` -> `f32`), not the math.
pub fn dense_laplacian_tensor<B: Backend>(
    hg: &SpectralHypergraph,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let lap = dense_normalized_laplacian(hg)?;
    let n = lap.nrows();
    let data: Vec<f32> = lap.iter().map(|&v| v as f32).collect();
    // `nalgebra::DMatrix` iterates column-major; transpose the flat buffer
    // back to row-major before handing it to Burn's row-major `reshape`.
    let mut row_major = vec![0.0f32; n * n];
    for col in 0..n {
        for row in 0..n {
            row_major[row * n + col] = data[col * n + row];
        }
    }
    Ok(Tensor::<B, 1>::from_floats(row_major.as_slice(), device).reshape([n, n]))
}

/// Applies `Delta` (the tensor built by [`dense_laplacian_tensor`]) to a
/// `[n, f]` feature/field tensor `u`, i.e. `Delta @ u`. Kept as a named
/// one-liner (rather than inlining `laplacian.matmul(u)` at every call site)
/// so [`crate::physics`]'s residual functions read as "the discrete spatial
/// operator applied to the field," matching how a classical PINN residual
/// reads as `laplace(u)`.
pub fn apply_laplacian<B: Backend>(laplacian: &Tensor<B, 2>, u: Tensor<B, 2>) -> Tensor<B, 2> {
    laplacian.clone().matmul(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use spectral_hypergraph::hypergraph::HypergraphBuilder;
    use spectral_hypergraph::spectral::dense_eigen;

    type B = NdArray<f32>;

    fn small_hypergraph() -> SpectralHypergraph {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..5).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
        b.add_hyperedge(&[ids[0], ids[1], ids[2]], 1.0).unwrap();
        b.add_hyperedge(&[ids[2], ids[3], ids[4]], 1.0).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn dense_tensor_matches_nalgebra_reference_entrywise() {
        let hg = small_hypergraph();
        let device = Default::default();
        let t = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let lap = dense_normalized_laplacian(&hg).unwrap();
        let n = lap.nrows();
        assert_eq!(t.dims(), [n, n]);

        let data = t.into_data().convert::<f32>();
        for i in 0..n {
            for j in 0..n {
                let want = lap[(i, j)] as f32;
                let got = data.value[i * n + j];
                assert!((want - got).abs() < 1e-5, "mismatch at ({i},{j}): want {want}, got {got}");
            }
        }
    }

    #[test]
    fn apply_laplacian_matches_matmul_and_kills_the_zero_mode() {
        let hg = small_hypergraph();
        let device = Default::default();
        let t = dense_laplacian_tensor::<B>(&hg, &device).unwrap();
        let lap = dense_normalized_laplacian(&hg).unwrap();
        let n = lap.nrows();

        // The zero mode of Delta is proportional to D_v^{1/2} * 1: applying
        // Delta to it must give (numerically) zero, the standard sanity
        // check for a correctly normalized hypergraph Laplacian.
        let eig = dense_eigen(&lap);
        let zero_mode = eig.eigenvectors.column(0).into_owned();
        assert!(eig.eigenvalues[0].abs() < 1e-9, "first eigenvalue should be ~0, got {}", eig.eigenvalues[0]);

        let data: Vec<f32> = zero_mode.iter().map(|&v| v as f32).collect();
        let u = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([n, 1]);
        let out = apply_laplacian::<B>(&t, u);
        let out_data = out.into_data().convert::<f32>();
        for &v in out_data.value.iter() {
            assert!(v.abs() < 1e-4, "Delta * zero_mode should be ~0, got {v}");
        }
    }
}
