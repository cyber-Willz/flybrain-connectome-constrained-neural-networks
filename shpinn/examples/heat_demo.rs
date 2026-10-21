//! End-to-end SHPINN demo for the time-dependent case: diffusion
//! (`du/dt = -alpha * Delta u`) on a hypergraph, given an initial condition
//! at `t = 0`. The network's input is `[spectral coordinates, t]`; the
//! spatial term `Delta u` is the same fixed matrix multiply the Poisson
//! demo uses, and `du/dt` is obtained via genuine Burn autodiff w.r.t. the
//! time input, using [`shpinn::physics::time_derivative`]'s "sum trick"
//! (see that function's docs for why it's valid for a row-wise network like
//! [`shpinn::pinn::Shpinn`]).
//!
//! `du/dt` itself is obtained by a central finite difference on the
//! network's own output at `t +/- TIME_EPS` (see
//! [`shpinn::physics::central_time_derivative`] for why this crate uses
//! that instead of autodiff w.r.t. the time input).
//!
//! Ground truth: this is a *linear* ODE system in disguise
//! (`du/dt = -alpha Delta u`), so it has a closed form via the spectral
//! decomposition of `Delta` that `spectral_hypergraph::spectral::dense_eigen`
//! already gives us for free:
//!
//! ```text
//! u(t) = sum_k exp(-alpha * t * lambda_k) * (v_k . u0) * v_k
//! ```
//!
//! (`Delta = sum_k lambda_k v_k v_k^T`, and the ODE decouples in that
//! eigenbasis.) The demo checks the trained SHPINN against this closed-form
//! solution at several held-out time points it was *not* directly fitted to
//! the value of (it only saw the PDE residual there, not the true `u`),
//! which is a meaningfully stronger check than comparing training loss.
//!
//! Run with: `cargo run --release --example heat_demo`

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;
use spectral_hypergraph::spectral::dense_eigen;

use shpinn::embedding::laplacian_eigenmap;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::operator::dense_laplacian_tensor;
use shpinn::physics::{central_time_derivative, heat_residual};
use shpinn::pinn::ShpinnConfig;

/// Half-width of the central-difference stencil used for `du/dt`; see
/// [`shpinn::physics::central_time_derivative`].
const TIME_EPS: f64 = 1e-3;

type Be = Autodiff<NdArray<f32>>;

const K_EMBED: usize = 4;
const ALPHA: f64 = 1.0;
const EPOCHS: usize = 4000;
const LR: f64 = 0.005;
const DATA_WEIGHT: f64 = 10.0;
const PHYSICS_WEIGHT: f64 = 1.0;
/// Collocation times used *during training* (residual enforced at all of
/// them; the initial condition is enforced only at `t = 0.0`).
const TRAIN_TIMES: [f64; 5] = [0.0, 0.1, 0.2, 0.35, 0.5];
/// Times used only for the *held-out* ground-truth check after training.
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

/// Exact `u(t)` via the eigendecomposition of `Delta`:
/// `u(t) = sum_k exp(-alpha * t * lambda_k) * (v_k . u0) * v_k`.
fn spectral_ground_truth(
    eig: &spectral_hypergraph::spectral::EigenDecomposition,
    u0: &nalgebra::DVector<f64>,
    t: f64,
    alpha: f64,
) -> nalgebra::DVector<f64> {
    let n = u0.len();
    let mut u = nalgebra::DVector::<f64>::zeros(n);
    for k in 0..n {
        let vk = eig.eigenvectors.column(k);
        let coeff = vk.dot(u0);
        let decay = (-alpha * t * eig.eigenvalues[k]).exp();
        u += vk.into_owned() * (coeff * decay);
    }
    u
}

/// Builds the `[n, K_EMBED + 1]` input for time `t`: the fixed spectral
/// coordinates in the first `K_EMBED` columns, `t` repeated down the last
/// column. The returned time column shares no data with `coords` so
/// `require_grad` on the full tensor only makes sense once we split
/// gradients back out via [`time_derivative`], which reads column
/// `K_EMBED`.
fn with_time_column<B: Backend>(coords: &Tensor<B, 2>, n: usize, t: f64, device: &B::Device) -> Tensor<B, 2> {
    let t_col = Tensor::<B, 1>::from_floats(vec![t as f32; n].as_slice(), device).reshape([n, 1]);
    Tensor::cat(vec![coords.clone(), t_col], 1)
}

fn main() {
    let device = <Be as Backend>::Device::default();
    let hg = small_hypergraph();
    let n = hg.num_vertices();

    let lap_dense = dense_normalized_laplacian(&hg).unwrap();
    let eig = dense_eigen(&lap_dense);

    // Initial condition: a "hot spot" at vertex 0, zero elsewhere.
    let mut u0 = nalgebra::DVector::<f64>::zeros(n);
    u0[0] = 1.0;

    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let laplacian_tensor = dense_laplacian_tensor::<Be>(&hg, &device).unwrap();

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
            let residual = heat_residual::<Be>(&laplacian_tensor, u.clone(), u_t, ALPHA);

            let (mask, target) = if t == 0.0 {
                (&all_mask, &u0_target)
            } else {
                (&zero_mask, &zero_target)
            };
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

    // --- Verification at held-out times -----------------------------------
    println!();
    println!("Trained SHPINN vs. closed-form spectral solution, at held-out times:");
    let mut worst_rmse = 0.0f64;
    for &t in CHECK_TIMES.iter() {
        let input = with_time_column::<Be>(&coords, n, t, &device);
        let u_pred = net.forward(input).into_data().convert::<f32>();
        let u_true = spectral_ground_truth(&eig, &u0, t, ALPHA);

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
        println!("\nPASS: SHPINN matches the closed-form diffusion solution at every held-out time (RMSE < {RMSE_TOLERANCE}).");
    } else {
        println!("\nFAIL: worst held-out RMSE {worst_rmse:.5} >= tolerance {RMSE_TOLERANCE}.");
        std::process::exit(1);
    }
}
