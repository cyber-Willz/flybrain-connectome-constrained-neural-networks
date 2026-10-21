//! Combined data + physics loss, plus the boundary-mask helper used to
//! impose Dirichlet-style constraints at a chosen set of "boundary
//! vertices" — the hypergraph analogue of boundary points in a mesh-based
//! PINN.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::error::{Result, ShpinnError};

/// Mean of the elementwise square of `r`, i.e. `mean(r ⊙ r)`. Works for any
/// rank; used identically for the data term (rank 2, `[n, d]`) and the
/// physics residual term (rank 2, `[n, d]`) in [`shpinn_loss`].
pub fn mse<B: Backend, const D: usize>(r: Tensor<B, D>) -> Tensor<B, 1> {
    (r.clone() * r).mean()
}

/// Builds an `[n, 1]` mask tensor with `1.0` at each vertex index in
/// `boundary` and `0.0` elsewhere, for use with [`shpinn_loss`]'s
/// `data_mask` argument. Errors instead of panicking if an index is out of
/// range, since a stray boundary index (e.g. after re-indexing a
/// hypergraph) is exactly the kind of silent-corruption bug worth catching
/// at construction time rather than downstream in a garbled loss value.
pub fn boundary_mask<B: Backend>(
    n: usize,
    boundary: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let mut data = vec![0.0f32; n];
    for &i in boundary {
        if i >= n {
            return Err(ShpinnError::BoundaryOutOfRange { index: i, n });
        }
        data[i] = 1.0;
    }
    Ok(Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([n, 1]))
}

/// `data_weight * MSE(mask ⊙ (u - target)) + physics_weight * MSE(residual)`.
///
/// The data term only penalizes vertices where `mask` is `1.0` (typically
/// built via [`boundary_mask`]) — at every other vertex `u` is judged purely
/// by whether it satisfies the physics residual, exactly the split a
/// classical PINN makes between its boundary-condition loss and its
/// interior PDE-residual loss.
pub fn shpinn_loss<B: Backend>(
    u: Tensor<B, 2>,
    target: &Tensor<B, 2>,
    data_mask: &Tensor<B, 2>,
    residual: Tensor<B, 2>,
    data_weight: f64,
    physics_weight: f64,
) -> Tensor<B, 1> {
    let masked_diff = (u - target.clone()) * data_mask.clone();
    let data_term = mse(masked_diff) * data_weight;
    let physics_term = mse(residual) * physics_weight;
    data_term + physics_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn boundary_mask_marks_only_requested_vertices() {
        let device = Default::default();
        let mask = boundary_mask::<B>(5, &[1, 3], &device).unwrap();
        let data = mask.into_data().convert::<f32>();
        assert_eq!(data.value, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn boundary_mask_rejects_out_of_range_index() {
        let device = Default::default();
        let err = boundary_mask::<B>(3, &[5], &device).unwrap_err();
        assert!(matches!(err, ShpinnError::BoundaryOutOfRange { index: 5, n: 3 }));
    }

    #[test]
    fn masked_data_term_ignores_non_boundary_rows() {
        let device = Default::default();
        let n = 4;
        // u matches target everywhere except row 2, which is NOT a boundary
        // vertex -- the data term should still come out exactly zero.
        let u = Tensor::<B, 2>::zeros([n, 1], &device);
        let mut target_vals = vec![0.0f32; n];
        target_vals[2] = 99.0;
        let target = Tensor::<B, 1>::from_floats(target_vals.as_slice(), &device).reshape([n, 1]);
        let mask = boundary_mask::<B>(n, &[0, 1, 3], &device).unwrap();
        let residual = Tensor::<B, 2>::zeros([n, 1], &device);

        let loss = shpinn_loss::<B>(u, &target, &mask, residual, 1.0, 0.0);
        let v: f32 = loss.into_scalar();
        assert!(v.abs() < 1e-8, "row 2 is unmasked, so its mismatch must not contribute; got {v}");
    }

    #[test]
    fn physics_weight_zero_ignores_residual() {
        let device = Default::default();
        let n = 3;
        let u = Tensor::<B, 2>::zeros([n, 1], &device);
        let target = u.clone();
        let mask = boundary_mask::<B>(n, &[], &device).unwrap();
        let residual = Tensor::<B, 2>::zeros([n, 1], &device) + 1000.0;
        let loss = shpinn_loss::<B>(u, &target, &mask, residual, 1.0, 0.0);
        let v: f32 = loss.into_scalar();
        assert!(v.abs() < 1e-8, "physics_weight=0 should zero out a huge residual, got {v}");
    }
}
