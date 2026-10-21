//! Directional-transport SHPINN demo (feature `nbsc-ops`): solves
//! `du/dt = -T_1(u)` where `T_1 = A / rho_B` is `nbsc`'s rescaled
//! non-backtracking (Hashimoto-spectrum-informed) operator rather than the
//! symmetric hypergraph Laplacian `Delta` — see [`shpinn::advective`]'s
//! module docs for why a genuinely directional/oriented-cycle-sensitive
//! governing equation calls for that swap.
//!
//! Ground truth here comes from direct RK4 integration of the same linear
//! ODE in `nalgebra` (not a spectral closed form, since `T_1` built from the
//! adjacency matrix is symmetric here too *as a matrix* — the
//! "non-backtracking-ness" is in how `rho_B` was estimated (via the
//! genuinely non-symmetric linearized Hashimoto operator), not in the
//! shape of `T_1` itself; RK4 makes no assumption either way, so it is the
//! right independent check regardless).
//!
//! Run with: `cargo run --release --example advection_demo --features nbsc-ops`

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use nalgebra::DVector;
use spectral_hypergraph::hypergraph::HypergraphBuilder;

use nbsc::hypergraph_bridge::clique_expand;
use nbsc::spectral::estimate_spectral_radius;

use shpinn::advective::{advective_residual, NonBacktrackingOperator};
use shpinn::embedding::laplacian_eigenmap;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::physics::central_time_derivative;
use shpinn::pinn::ShpinnConfig;

const TIME_EPS: f64 = 1e-3;

type Be = Autodiff<NdArray<f32>>;

const K_EMBED: usize = 4;
const KRYLOV_DIM: usize = 20;
const SEED: u64 = 3;
const EPOCHS: usize = 4000;
const LR: f64 = 0.005;
const DATA_WEIGHT: f64 = 10.0;
const PHYSICS_WEIGHT: f64 = 1.0;
const TRAIN_TIMES: [f64; 5] = [0.0, 0.1, 0.2, 0.35, 0.5];
const CHECK_TIMES: [f64; 3] = [0.15, 0.3, 0.45];

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

/// RK4 integration of the linear ODE `du/dt = -(adjacency / rho_b) u`,
/// entirely in `f64` `nalgebra`, independent of the Burn training path.
fn rk4_ground_truth(adjacency: &nalgebra::DMatrix<f64>, rho_b: f64, u0: &DVector<f64>, t_final: f64, steps: usize) -> DVector<f64> {
    let dt = t_final / steps as f64;
    let dudt = |u: &DVector<f64>| -(adjacency * u) / rho_b;
    let mut u = u0.clone();
    for _ in 0..steps {
        let k1 = dudt(&u);
        let k2 = dudt(&(&u + &k1 * (dt / 2.0)));
        let k3 = dudt(&(&u + &k2 * (dt / 2.0)));
        let k4 = dudt(&(&u + &k3 * dt));
        u += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
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

    let g = clique_expand(&hg);
    let rho_b = estimate_spectral_radius(&g, KRYLOV_DIM.min(g.n), SEED).max(1e-6);
    let adjacency_f64 = {
        let flat = g.dense_adjacency();
        nalgebra::DMatrix::<f64>::from_row_slice(g.n, g.n, &flat)
    };

    let mut u0 = DVector::<f64>::zeros(n);
    u0[0] = 1.0;

    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let op = NonBacktrackingOperator::<Be>::build(&hg, KRYLOV_DIM, SEED, &device);
    println!("rho_B = {:.5} (used consistently by both the Burn operator and the RK4 ground truth)", op.rho_b());

    let all_mask = boundary_mask::<Be>(n, &(0..n).collect::<Vec<_>>(), &device).unwrap();
    let zero_mask = boundary_mask::<Be>(n, &[], &device).unwrap();
    let u0_f32: Vec<f32> = u0.iter().map(|&v| v as f32).collect();
    let u0_target = Tensor::<Be, 1>::from_floats(u0_f32.as_slice(), &device).reshape([n, 1]);
    let zero_target = Tensor::<Be, 2>::zeros([n, 1], &device);
    let forcing = Tensor::<Be, 2>::zeros([n, 1], &device);

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
            let residual = advective_residual::<Be>(&op, u.clone(), u_t, &forcing);

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
    println!("Trained SHPINN vs. RK4 ground truth of the same non-backtracking-informed transport ODE:");
    let mut worst_rmse = 0.0f64;
    for &t in CHECK_TIMES.iter() {
        let input = with_time_column::<Be>(&coords, n, t, &device);
        let u_pred = net.forward(input).into_data().convert::<f32>();
        let u_true = rk4_ground_truth(&adjacency_f64, rho_b, &u0, t, 200);

        let mut sum_sq = 0.0f64;
        for i in 0..n {
            let err = u_pred.value[i] as f64 - u_true[i];
            sum_sq += err * err;
        }
        let rmse = (sum_sq / n as f64).sqrt();
        worst_rmse = worst_rmse.max(rmse);
        println!("  t = {t:.2}: RMSE = {rmse:.5}");
    }

    const RMSE_TOLERANCE: f64 = 0.08;
    if worst_rmse < RMSE_TOLERANCE {
        println!("\nPASS: SHPINN matches RK4 ground truth at every held-out time (RMSE < {RMSE_TOLERANCE}).");
    } else {
        println!("\nFAIL: worst held-out RMSE {worst_rmse:.5} >= tolerance {RMSE_TOLERANCE}.");
        std::process::exit(1);
    }
}
