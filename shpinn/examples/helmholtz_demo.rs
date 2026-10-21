//! End-to-end SHPINN demo: solves the steady-state hypergraph Helmholtz
//! equation `Delta u - k^2 u = 0` with Dirichlet boundary conditions at a
//! handful of chosen vertices -- the same boundary-value setup
//! `poisson_demo` uses, with a reaction term added
//! ([`shpinn::physics::helmholtz_residual`]).
//!
//! Ground truth: same technique as `poisson_demo` -- overwrite the
//! boundary rows of `Delta - k^2 I` with identity rows and `lu().solve()`
//! the rest directly, independent of the trained network.
//!
//! Run with: `cargo run --release --example helmholtz_demo`

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use nalgebra::{DMatrix, DVector};
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;
use spectral_hypergraph::spectral::dense_eigen;

use shpinn::embedding::laplacian_eigenmap;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::operator::dense_laplacian_tensor;
use shpinn::physics::helmholtz_residual;
use shpinn::pinn::ShpinnConfig;

type Be = Autodiff<NdArray<f32>>;

const K_EMBED: usize = 4;
const EPOCHS: usize = 6000;
const LR: f64 = 0.01;
const DATA_WEIGHT: f64 = 10.0;
const PHYSICS_WEIGHT: f64 = 1.0;

fn demo_hypergraph() -> spectral_hypergraph::SpectralHypergraph {
    let mut b = HypergraphBuilder::new();
    let ids: Vec<_> = (0..24).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
    for cluster in 0..3 {
        let base = cluster * 8;
        b.add_hyperedge(&[ids[base], ids[base + 1], ids[base + 2], ids[base + 3]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 2], ids[base + 3], ids[base + 4]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 4], ids[base + 5], ids[base + 6]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 5], ids[base + 6], ids[base + 7]], 1.0).unwrap();
    }
    b.add_hyperedge(&[ids[7], ids[8]], 1.0).unwrap();
    b.add_hyperedge(&[ids[15], ids[16]], 1.0).unwrap();
    b.add_hyperedge(&[ids[0], ids[23]], 1.0).unwrap();
    b.build().unwrap()
}

/// Solves `(Delta - k2*I) u = 0` subject to `u[i] = value` at each boundary
/// vertex, by the same row-substitution + direct solve `poisson_demo` uses.
fn solve_helmholtz_ground_truth(lap: &DMatrix<f64>, k2: f64, boundary: &[(usize, f64)]) -> DVector<f64> {
    let n = lap.nrows();
    let mut a = lap - k2 * DMatrix::<f64>::identity(n, n);
    let mut rhs = DVector::<f64>::zeros(n);
    for &(i, value) in boundary {
        for j in 0..n {
            a[(i, j)] = 0.0;
        }
        a[(i, i)] = 1.0;
        rhs[i] = value;
    }
    a.lu().solve(&rhs).expect("boundary-modified Helmholtz operator should be non-singular")
}

fn main() {
    let device = <Be as Backend>::Device::default();
    let hg = demo_hypergraph();
    let n = hg.num_vertices();

    let lap = dense_normalized_laplacian(&hg).unwrap();

    // Pick k^2 in the widest gap between consecutive nonzero eigenvalues of
    // Delta, away from resonance (see helmholtz_residual's docs) -- this
    // demo hypergraph's 3-fold cluster symmetry gives it near-degenerate
    // eigenvalue pairs, so picking the *widest* gap (rather than just the
    // first one) avoids landing next to a near-resonance where the
    // boundary-value problem is severely ill-conditioned.
    let eig = dense_eigen(&lap);
    let nonzero: Vec<f64> = eig.eigenvalues.iter().skip(1).copied().collect();
    let (mut best_lo, mut best_hi, mut best_gap) = (nonzero[0], nonzero[1], 0.0);
    for w in nonzero.windows(2) {
        let gap = w[1] - w[0];
        if gap > best_gap {
            best_gap = gap;
            best_lo = w[0];
            best_hi = w[1];
        }
    }
    let k2 = (best_lo + best_hi) / 2.0;
    println!("k^2 = {k2:.4} (widest spectral gap: {:.4} to {:.4}, gap {:.4})", best_lo, best_hi, best_gap);

    let boundary: Vec<(usize, f64)> = vec![(0, 1.0), (8, -1.0), (16, 0.5)];
    let boundary_idx: Vec<usize> = boundary.iter().map(|&(i, _)| i).collect();
    let ground_truth = solve_helmholtz_ground_truth(&lap, k2, &boundary);

    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let laplacian_tensor = dense_laplacian_tensor::<Be>(&hg, &device).unwrap();
    let forcing = Tensor::<Be, 2>::zeros([n, 1], &device);

    let mask = boundary_mask::<Be>(n, &boundary_idx, &device).unwrap();
    let interior_mask = (Tensor::<Be, 2>::zeros([n, 1], &device) + 1.0) - mask.clone();
    let mut target_vals = vec![0.0f32; n];
    for &(i, value) in &boundary {
        target_vals[i] = value as f32;
    }
    let target = Tensor::<Be, 1>::from_floats(target_vals.as_slice(), &device).reshape([n, 1]);

    let cfg = ShpinnConfig::new(K_EMBED, vec![32, 32], 1);
    let mut net = cfg.init::<Be>(&device);
    let mut optim = AdamConfig::new().init();

    for epoch in 0..EPOCHS {
        let u = net.forward(coords.clone());
        let residual = helmholtz_residual::<Be>(&laplacian_tensor, u.clone(), &forcing, k2) * interior_mask.clone();
        let loss = shpinn_loss::<Be>(u, &target, &mask, residual, DATA_WEIGHT, PHYSICS_WEIGHT);

        if epoch % 500 == 0 || epoch == EPOCHS - 1 {
            let loss_val: f32 = loss.clone().into_scalar();
            println!("epoch {epoch:5}: loss = {loss_val:.6}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
    }

    let u_final = net.forward(coords);
    let u_data = u_final.into_data().convert::<f32>();

    let mut max_abs_err = 0.0f64;
    let mut sum_sq_err = 0.0f64;
    for i in 0..n {
        let predicted = u_data.value[i] as f64;
        let truth = ground_truth[i];
        let err = (predicted - truth).abs();
        max_abs_err = max_abs_err.max(err);
        sum_sq_err += err * err;
    }
    let rmse = (sum_sq_err / n as f64).sqrt();

    println!();
    println!("Trained SHPINN vs. direct linear solve of the same Helmholtz boundary-value problem:");
    println!("  RMSE over all {n} vertices:     {rmse:.5}");
    println!("  max |error| over all vertices: {max_abs_err:.5}");
    for &(i, value) in &boundary {
        println!("  boundary vertex {i}: target={value:.3}, predicted={:.3}", u_data.value[i]);
    }

    const RMSE_TOLERANCE: f64 = 0.05;
    if rmse < RMSE_TOLERANCE {
        println!("\nPASS: SHPINN solution matches the ground-truth Helmholtz solve (RMSE < {RMSE_TOLERANCE}).");
    } else {
        println!("\nFAIL: SHPINN solution deviates from ground truth beyond tolerance (RMSE >= {RMSE_TOLERANCE}).");
        std::process::exit(1);
    }
}
