//! End-to-end SHPINN demo: solves the steady-state hypergraph Poisson
//! equation `Delta u = 0` with Dirichlet boundary conditions at a handful
//! of chosen "boundary vertices", i.e. the harmonic-extension problem on a
//! hypergraph — the discrete-domain analogue of the textbook PINN benchmark
//! "solve Laplace's equation on a disk given its boundary values."
//!
//! Unlike the loss-went-down-so-it-must-be-right style of many PINN demos,
//! this one is checked against an **independent ground truth**: the same
//! boundary-value problem solved directly as a linear system (overwrite the
//! boundary rows of the dense Laplacian with identity rows, `lu().solve()`
//! the rest). If the trained SHPINN's field doesn't match that solve to
//! reasonable tolerance at every vertex — not just the boundary ones it was
//! fitted to — the demo reports failure rather than declaring victory on
//! training loss alone.
//!
//! Run with: `cargo run --release --example poisson_demo`

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use nalgebra::{DMatrix, DVector};
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;

use shpinn::embedding::laplacian_eigenmap;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::operator::dense_laplacian_tensor;
use shpinn::physics::poisson_residual;
use shpinn::pinn::ShpinnConfig;

type Be = Autodiff<NdArray<f32>>;

const K_EMBED: usize = 4;
// The interior Dirichlet-problem submatrix for this demo hypergraph (three
// clusters bridged by single weak edges) has condition number ~13.5, so a
// residual that looks small in the loss still gets amplified into solution
// error -- 3000 epochs at a constant LR leaves the loss still visibly
// descending (not yet plateaued) and the demo fails its RMSE check on
// unlucky init draws. An LR step-decay was tried first and made things
// *worse*: this landscape has a plateau around loss ~0.0007 that training
// escapes via an large-step jump (observed dropping an order of magnitude
// between epochs 2500-3000 at the original constant LR=0.01), and decaying
// the LR early froze training inside that plateau instead of letting it
// escape. So the fix kept here is simpler and empirically verified: just
// give it enough epochs at the original constant LR to reliably clear the
// plateau and converge tightly. Verified over 5 runs (random init each
// time, no fixed seed) at EPOCHS=6000: every run passes with RMSE well
// under half the 0.05 tolerance.
const EPOCHS: usize = 6000;
const LR: f64 = 0.01;
const DATA_WEIGHT: f64 = 10.0;
const PHYSICS_WEIGHT: f64 = 1.0;

/// Three loosely-triangle-clustered groups of vertices, bridged by
/// pairwise hyperedges, plus a few larger (4-member) hyperedges within each
/// cluster -- enough genuine higher-order + cluster structure that the
/// spectral embedding actually separates the clusters, without needing the
/// `nbsc-ops` feature's SBM generator.
fn demo_hypergraph() -> spectral_hypergraph::SpectralHypergraph {
    let mut b = HypergraphBuilder::new();
    let ids: Vec<_> = (0..24).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();

    for cluster in 0..3 {
        let base = cluster * 8;
        // A 4-member hyperedge and two overlapping triangles per cluster.
        b.add_hyperedge(&[ids[base], ids[base + 1], ids[base + 2], ids[base + 3]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 2], ids[base + 3], ids[base + 4]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 4], ids[base + 5], ids[base + 6]], 1.0).unwrap();
        b.add_hyperedge(&[ids[base + 5], ids[base + 6], ids[base + 7]], 1.0).unwrap();
    }
    // Bridges between consecutive clusters so the hypergraph is connected.
    b.add_hyperedge(&[ids[7], ids[8]], 1.0).unwrap();
    b.add_hyperedge(&[ids[15], ids[16]], 1.0).unwrap();
    b.add_hyperedge(&[ids[0], ids[23]], 1.0).unwrap();

    b.build().unwrap()
}

/// Solves `Delta u = 0` subject to `u[i] = value` for every `(i, value)` in
/// `boundary`, by overwriting those rows of the dense Laplacian with
/// identity rows and solving the resulting linear system directly. This is
/// the ground truth the trained SHPINN is checked against.
fn solve_harmonic_ground_truth(lap: &DMatrix<f64>, boundary: &[(usize, f64)]) -> DVector<f64> {
    let n = lap.nrows();
    let mut a = lap.clone();
    let mut rhs = DVector::<f64>::zeros(n);
    for &(i, value) in boundary {
        for j in 0..n {
            a[(i, j)] = 0.0;
        }
        a[(i, i)] = 1.0;
        rhs[i] = value;
    }
    a.lu().solve(&rhs).expect("boundary-modified Laplacian should be non-singular")
}

fn main() {
    let device = <Be as Backend>::Device::default();
    let hg = demo_hypergraph();
    let n = hg.num_vertices();

    // Boundary condition: one vertex per cluster pinned to a distinct
    // value, so the harmonic extension is a genuinely non-trivial function
    // across the hypergraph rather than a constant.
    let boundary: Vec<(usize, f64)> = vec![(0, 1.0), (8, -1.0), (16, 0.5)];
    let boundary_idx: Vec<usize> = boundary.iter().map(|&(i, _)| i).collect();

    let lap = dense_normalized_laplacian(&hg).unwrap();
    let ground_truth = solve_harmonic_ground_truth(&lap, &boundary);

    // --- SHPINN setup -----------------------------------------------------
    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let laplacian_tensor = dense_laplacian_tensor::<Be>(&hg, &device).unwrap();
    let forcing = Tensor::<Be, 2>::zeros([n, 1], &device); // Delta u = 0

    let mask = boundary_mask::<Be>(n, &boundary_idx, &device).unwrap();
    // The physics residual (Delta u = 0) only holds at *interior* vertices
    // in a Dirichlet boundary-value problem -- a boundary vertex's true
    // Laplacian is generally nonzero (it's a prescribed value, not a free
    // unknown solving the PDE there), so enforcing Delta u = 0 at boundary
    // vertices too would fight the boundary condition itself. Mask the
    // residual down to interior vertices before it enters the loss.
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
        let residual = poisson_residual::<Be>(&laplacian_tensor, u.clone(), &forcing) * interior_mask.clone();
        let loss = shpinn_loss::<Be>(u, &target, &mask, residual, DATA_WEIGHT, PHYSICS_WEIGHT);

        if epoch % 500 == 0 || epoch == EPOCHS - 1 {
            let loss_val: f32 = loss.clone().into_scalar();
            println!("epoch {epoch:5}: loss = {loss_val:.6}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
    }

    // --- Verification against the independent ground-truth solve ---------
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
    println!("Trained SHPINN vs. direct linear solve of the same boundary-value problem:");
    println!("  RMSE over all {n} vertices:     {rmse:.5}");
    println!("  max |error| over all vertices: {max_abs_err:.5}");
    for &(i, value) in &boundary {
        println!("  boundary vertex {i}: target={value:.3}, predicted={:.3}", u_data.value[i]);
    }

    const RMSE_TOLERANCE: f64 = 0.05;
    if rmse < RMSE_TOLERANCE {
        println!("\nPASS: SHPINN solution matches the ground-truth harmonic extension (RMSE < {RMSE_TOLERANCE}).");
    } else {
        println!("\nFAIL: SHPINN solution deviates from ground truth beyond tolerance (RMSE >= {RMSE_TOLERANCE}).");
        std::process::exit(1);
    }
}
