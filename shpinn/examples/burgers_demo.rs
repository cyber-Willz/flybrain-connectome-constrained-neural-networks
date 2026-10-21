//! End-to-end SHPINN demo for viscous Burgers' equation, `du/dt + u du/dx -
//! nu d^2u/dx^2 = 0`, given an initial condition at `t = 0` -- the
//! canonical nonlinear PINN benchmark (Raissi et al. 2019), transplanted
//! onto a hypergraph via [`shpinn::burgers::burgers_residual`]'s
//! conservative-flux discretization (see that function's module docs).
//!
//! Ground truth: RK4 integration, in plain `nalgebra`, of the exact same
//! discrete ODE the physics residual enforces --
//! `du/dt = -G^T(G(u^2/2)) - nu * Delta u` -- built independently of the
//! Burn training path (its own from-scratch construction of `G` and
//! `Delta`, not a call into `shpinn::gradient`/`shpinn::operator`). This
//! checks whether the trained network actually learned to satisfy the
//! discretized PDE, the same role RK4 plays in `advection_demo` and the
//! closed-form spectral solution plays in `heat_demo` -- independent of
//! *how* the network got there, not independent of the discretization
//! itself (which is exactly what the physics loss is supposed to teach the
//! network to satisfy).
//!
//! Run with: `cargo run --release --example burgers_demo`

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use nalgebra::DVector;
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::{
    dense_incidence_matrix, dense_normalized_laplacian, hyperedge_degree_vector, vertex_degree_vector,
};

use shpinn::burgers::burgers_residual;
use shpinn::embedding::laplacian_eigenmap;
use shpinn::gradient::dense_gradient_tensor;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::operator::dense_laplacian_tensor;
use shpinn::physics::central_time_derivative;
use shpinn::pinn::ShpinnConfig;

const TIME_EPS: f64 = 1e-3;
type Be = Autodiff<NdArray<f32>>;

const K_EMBED: usize = 4;
const NU: f64 = 0.15;
const EPOCHS: usize = 5000;
const LR: f64 = 0.004;
const DATA_WEIGHT: f64 = 10.0;
const PHYSICS_WEIGHT: f64 = 1.0;
const TRAIN_TIMES: [f64; 5] = [0.0, 0.05, 0.1, 0.2, 0.3];
const CHECK_TIMES: [f64; 3] = [0.075, 0.15, 0.25];

fn small_hypergraph() -> spectral_hypergraph::SpectralHypergraph {
    let mut b = HypergraphBuilder::new();
    let ids: Vec<_> = (0..12).map(|i| b.add_vertex(format!("v{i}")).unwrap()).collect();
    b.add_hyperedge(&[ids[0], ids[1], ids[2], ids[3]], 1.0).unwrap();
    b.add_hyperedge(&[ids[3], ids[4], ids[5]], 1.0).unwrap();
    b.add_hyperedge(&[ids[5], ids[6], ids[7], ids[8]], 1.0).unwrap();
    b.add_hyperedge(&[ids[8], ids[9], ids[10], ids[11]], 1.0).unwrap();
    b.add_hyperedge(&[ids[0], ids[11]], 1.0).unwrap();
    b.build().unwrap()
}

/// Independent, from-scratch `nalgebra` build of `G` (see
/// `shpinn::gradient`'s module docs for the formula) -- deliberately not a
/// call into `shpinn::gradient::dense_gradient_tensor`, so this ground
/// truth doesn't share code with the thing it's checking.
fn reference_gradient_matrix(hg: &spectral_hypergraph::SpectralHypergraph) -> nalgebra::DMatrix<f64> {
    let h = dense_incidence_matrix(hg).unwrap(); // [n, m]
    let n = h.nrows();
    let m = h.ncols();
    let dv = vertex_degree_vector(hg).unwrap();
    let de = hyperedge_degree_vector(hg).unwrap();
    let mut g = nalgebra::DMatrix::<f64>::zeros(m, n);
    for e in hg.hyperedge_ids() {
        let scale_e = (hg.hyperedge_weight(e).unwrap() / de[e.0]).sqrt();
        for v in hg.hyperedge_members(e).unwrap() {
            let scale_v = 1.0 / dv[v.0].sqrt();
            g[(e.0, v.0)] = h[(v.0, e.0)] * scale_e * scale_v;
        }
    }
    g
}

/// `du/dt = -G^T(G(u^2/2)) - nu * Delta u`, the exact ODE
/// `burgers_residual` enforces (rearranged to isolate `du/dt`).
fn burgers_rhs(g: &nalgebra::DMatrix<f64>, lap: &nalgebra::DMatrix<f64>, u: &DVector<f64>, nu: f64) -> DVector<f64> {
    let flux: DVector<f64> = u.map(|v| 0.5 * v * v);
    let advective = g.transpose() * (g * &flux);
    -advective - nu * (lap * u)
}

fn rk4_integrate(g: &nalgebra::DMatrix<f64>, lap: &nalgebra::DMatrix<f64>, u0: &DVector<f64>, nu: f64, t_final: f64, steps: usize) -> DVector<f64> {
    let dt = t_final / steps as f64;
    let mut u = u0.clone();
    for _ in 0..steps {
        let k1 = burgers_rhs(g, lap, &u, nu);
        let k2 = burgers_rhs(g, lap, &(&u + 0.5 * dt * &k1), nu);
        let k3 = burgers_rhs(g, lap, &(&u + 0.5 * dt * &k2), nu);
        let k4 = burgers_rhs(g, lap, &(&u + dt * &k3), nu);
        u += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    u
}

fn with_time_column<B: Backend>(coords: &Tensor<B, 2>, n: usize, t: f64, device: &B::Device) -> Tensor<B, 2> {
    let t_col = Tensor::<B, 1>::from_floats(vec![t as f32; n].as_slice(), device).reshape([n, 1]);
    Tensor::cat(vec![coords.clone(), t_col], 1)
}

fn main() {
    let device = <Be as Backend>::Device::default();
    let hg = small_hypergraph();
    let n = hg.num_vertices();

    let lap_dense = dense_normalized_laplacian(&hg).unwrap();
    let g_dense = reference_gradient_matrix(&hg);

    // Initial condition: a "hot spot" bump at vertex 0, zero elsewhere --
    // same shape as heat_demo's, but now nonlinear self-steepening plus
    // viscous smoothing compete, rather than pure diffusion.
    let mut u0 = DVector::<f64>::zeros(n);
    u0[0] = 1.0;

    // Ground-truth trajectory at each check time, via RK4 from t=0.
    let mut ground_truth = std::collections::HashMap::new();
    for &t in CHECK_TIMES.iter() {
        let steps = ((t / 0.001).ceil() as usize).max(20);
        ground_truth.insert(t.to_bits(), rk4_integrate(&g_dense, &lap_dense, &u0, NU, t, steps));
    }

    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let laplacian_tensor = dense_laplacian_tensor::<Be>(&hg, &device).unwrap();
    let grad_tensor = dense_gradient_tensor::<Be>(&hg, &device).unwrap();

    let all_mask = boundary_mask::<Be>(n, &(0..n).collect::<Vec<_>>(), &device).unwrap();
    let zero_mask = boundary_mask::<Be>(n, &[], &device).unwrap();
    let u0_f32: Vec<f32> = u0.iter().map(|&v| v as f32).collect();
    let u0_target = Tensor::<Be, 1>::from_floats(u0_f32.as_slice(), &device).reshape([n, 1]);
    let zero_target = Tensor::<Be, 2>::zeros([n, 1], &device);

    let cfg = ShpinnConfig::new(K_EMBED + 1, vec![32, 32], 1);
    let mut net = cfg.init::<Be>(&device);
    let mut optim = AdamConfig::new().init();

    for epoch in 0..EPOCHS {
        let mut loss = Tensor::<Be, 1>::zeros([1], &device);

        for &t in TRAIN_TIMES.iter() {
            let input = with_time_column::<Be>(&coords, n, t, &device);
            let u = net.forward(input);
            let u_plus = net.forward(with_time_column::<Be>(&coords, n, t + TIME_EPS, &device));
            let u_minus = net.forward(with_time_column::<Be>(&coords, n, t - TIME_EPS, &device));
            let u_t = central_time_derivative::<Be>(u_plus, u_minus, TIME_EPS);
            let residual = burgers_residual::<Be>(&laplacian_tensor, &grad_tensor, u.clone(), u_t, NU);

            let (mask, target) = if t == 0.0 { (&all_mask, &u0_target) } else { (&zero_mask, &zero_target) };
            loss = loss + shpinn_loss::<Be>(u, target, mask, residual, DATA_WEIGHT, PHYSICS_WEIGHT);
        }

        if epoch % 500 == 0 || epoch == EPOCHS - 1 {
            let loss_val: f32 = loss.clone().into_scalar();
            println!("epoch {epoch:5}: loss = {loss_val:.6}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
    }

    println!();
    println!("Ground-truth field magnitude at each held-out time (sanity check against a trivial all-zero decay):");
    for &t in CHECK_TIMES.iter() {
        let truth = &ground_truth[&t.to_bits()];
        println!("  t = {t:.3}: max|u| = {:.5}, mean|u| = {:.5}", truth.abs().max(), truth.abs().sum() / truth.len() as f64);
    }
    println!();
    println!("Trained SHPINN vs. independent RK4 integration of the same discrete Burgers ODE, at held-out times:");
    let mut worst_rmse = 0.0f64;
    for &t in CHECK_TIMES.iter() {
        let input = with_time_column::<Be>(&coords, n, t, &device);
        let u_pred = net.forward(input).into_data().convert::<f32>();
        let u_true = &ground_truth[&t.to_bits()];

        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let err = u_pred.value[i] as f64 - u_true[i];
            sum_sq += err * err;
        }
        let rmse = (sum_sq / n as f64).sqrt();
        worst_rmse = worst_rmse.max(rmse);
        println!("  t = {t:.3}: RMSE = {rmse:.5}");
    }

    const RMSE_TOLERANCE: f64 = 0.08;
    if worst_rmse < RMSE_TOLERANCE {
        println!("\nPASS: SHPINN matches the independent RK4 Burgers trajectory at every held-out time (RMSE < {RMSE_TOLERANCE}).");
    } else {
        println!("\nFAIL: worst held-out RMSE {worst_rmse:.5} >= tolerance {RMSE_TOLERANCE}.");
        std::process::exit(1);
    }
}
