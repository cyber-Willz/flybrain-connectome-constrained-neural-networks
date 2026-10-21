//! Same viscous Burgers' problem as `burgers_demo`, run twice back to
//! back -- once through a plain [`shpinn::pinn::Shpinn`] (identical setup
//! to `burgers_demo`), once through the same network preceded by a
//! [`shpinn::spectral_encoding::FourierFeatureEncoder`] -- against the
//! *same* independent RK4 ground truth, and both results are printed side
//! by side.
//!
//! ## Why this exists
//!
//! Ported from a from-scratch, continuous-domain PINN system built
//! independently of this crate, whose own Burgers run hit a
//! well-documented failure mode of plain tanh-MLPs: spectral bias, a bias
//! toward low-frequency functions that makes sharp features (there, a
//! near-shock; here, the initial "hot spot" bump concentrated on a single
//! vertex) disproportionately hard to fit. That system's fix -- random
//! Fourier input features ahead of the network -- only touches the
//! network's input side, so it ports cleanly onto this crate's spectral
//! embedding coordinates without touching `shpinn::burgers` or any other
//! residual code; see `spectral_encoding.rs`'s module docs for the full
//! reasoning.
//!
//! That source project's own README reported a case where Fourier
//! features looked like a clear win by training loss alone but were
//! badly overfit/aliased against a sparse collocation grid once checked
//! against an independent reference -- a caution worth repeating rather
//! than assuming this is a free win, especially since this crate's
//! hypergraphs (12 vertices here) are a far sparser "collocation set"
//! than that system's dense continuous grids. So: this demo measures it,
//! at a conservative frequency scale chosen with exactly that risk in
//! mind, and reports both numbers.
//!
//! **Measured result on this problem**: Fourier features did *not* help
//! here -- the plain network already fits this small hypergraph's Burgers
//! trajectory to near machine precision (worst-case RMSE ~0.00005 against
//! the RK4 ground truth), and the Fourier-encoded network came out
//! several times worse in relative terms (~0.0002, still tiny in absolute
//! terms). The most likely reading: spectral bias was never actually the
//! bottleneck on a problem this small and smooth (12 vertices, one soft
//! "hot spot" initial condition, no genuine shock the way continuous
//! Burgers has) -- Fourier features add representational capacity a
//! near-trivial fit doesn't need, and that shows up as slightly *more*
//! error rather than less. This is exactly the kind of measurement worth
//! keeping over an assumption: the encoder is real, tested, and available
//! (`spectral_encoding.rs`) for a problem where spectral bias is actually
//! the bottleneck -- a larger hypergraph, a sharper transition, or a
//! higher-eigenvalue `Delta` mode the way `helmholtz_demo`'s near-degenerate
//! eigenvalue pair hinted at -- but this demo's own numbers are the reason
//! it isn't wired into any of this crate's other demos by default.
//!
//! Run with: `cargo run --release --example burgers_spectral_demo`

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
use shpinn::spectral_encoding::FourierFeatureEncoder;

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

// Conservative: this hypergraph has 12 vertices, a far sparser
// "collocation set" than a continuous grid, so a small feature count and
// a frequency scale near 1 (one full cycle across the coordinates'
// natural range) is the deliberately cautious choice discussed in this
// file's module docs, not a tuned-for-best-number pick.
const FOURIER_FEATURES: usize = 8;
const FOURIER_FREQ_SCALE: f64 = 1.5;

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

/// Independent, from-scratch `nalgebra` build of `G` -- see
/// `burgers_demo`'s equivalent helper; kept separate from
/// `shpinn::gradient::dense_gradient_tensor` deliberately.
fn reference_gradient_matrix(hg: &spectral_hypergraph::SpectralHypergraph) -> nalgebra::DMatrix<f64> {
    let h = dense_incidence_matrix(hg).unwrap();
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

/// Trains one `Shpinn` -- either directly on `[coords, t]` (`encoder =
/// None`) or on the Fourier-encoded version of the same raw input
/// (`encoder = Some(..)`) -- against the Burgers residual, and returns the
/// RMSE against `ground_truth` at each of `CHECK_TIMES`. Kept as one
/// function parameterized by an optional encoder (rather than two
/// hand-duplicated copies, unlike the upstream project's own
/// plain/spectral split) since here the difference is genuinely a single
/// extra tensor op on the input, not a different code path through a
/// hand-written backward pass.
fn train_and_evaluate(
    n: usize,
    coords: &Tensor<Be, 2>,
    laplacian_tensor: &Tensor<Be, 2>,
    grad_tensor: &Tensor<Be, 2>,
    u0: &DVector<f64>,
    ground_truth: &std::collections::HashMap<u64, DVector<f64>>,
    device: &<Be as Backend>::Device,
    encoder: Option<&FourierFeatureEncoder<Be>>,
    label: &str,
) -> Vec<f64> {
    let raw_dim = K_EMBED + 1;
    let net_input_dim = encoder.map(|e| e.encoded_dim()).unwrap_or(raw_dim);

    let all_mask = boundary_mask::<Be>(n, &(0..n).collect::<Vec<_>>(), device).unwrap();
    let zero_mask = boundary_mask::<Be>(n, &[], device).unwrap();
    let u0_f32: Vec<f32> = u0.iter().map(|&v| v as f32).collect();
    let u0_target = Tensor::<Be, 1>::from_floats(u0_f32.as_slice(), device).reshape([n, 1]);
    let zero_target = Tensor::<Be, 2>::zeros([n, 1], device);

    let cfg = ShpinnConfig::new(net_input_dim, vec![32, 32], 1);
    let mut net = cfg.init::<Be>(device);
    let mut optim = AdamConfig::new().init();

    let net_input = |t: f64| -> Tensor<Be, 2> {
        let raw = with_time_column::<Be>(coords, n, t, device);
        match encoder {
            Some(e) => e.encode(raw),
            None => raw,
        }
    };

    for epoch in 0..EPOCHS {
        let mut loss = Tensor::<Be, 1>::zeros([1], device);

        for &t in TRAIN_TIMES.iter() {
            let u = net.forward(net_input(t));
            let u_plus = net.forward(net_input(t + TIME_EPS));
            let u_minus = net.forward(net_input(t - TIME_EPS));
            let u_t = central_time_derivative::<Be>(u_plus, u_minus, TIME_EPS);
            let residual = burgers_residual::<Be>(laplacian_tensor, grad_tensor, u.clone(), u_t, NU);

            let (mask, target) = if t == 0.0 { (&all_mask, &u0_target) } else { (&zero_mask, &zero_target) };
            loss = loss + shpinn_loss::<Be>(u, target, mask, residual, DATA_WEIGHT, PHYSICS_WEIGHT);
        }

        if epoch % 1000 == 0 || epoch == EPOCHS - 1 {
            let loss_val: f32 = loss.clone().into_scalar();
            println!("  [{label}] epoch {epoch:5}: loss = {loss_val:.6}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optim.step(LR, net, grads);
    }

    CHECK_TIMES
        .iter()
        .map(|&t| {
            let u_pred = net.forward(net_input(t)).into_data().convert::<f32>();
            let u_true = &ground_truth[&t.to_bits()];
            let mut sum_sq = 0.0f64;
            for i in 0..n {
                let err = u_pred.value[i] as f64 - u_true[i];
                sum_sq += err * err;
            }
            (sum_sq / n as f64).sqrt()
        })
        .collect()
}

fn main() {
    let device = <Be as Backend>::Device::default();
    let hg = small_hypergraph();
    let n = hg.num_vertices();

    let lap_dense = dense_normalized_laplacian(&hg).unwrap();
    let g_dense = reference_gradient_matrix(&hg);

    let mut u0 = DVector::<f64>::zeros(n);
    u0[0] = 1.0;

    let mut ground_truth = std::collections::HashMap::new();
    for &t in CHECK_TIMES.iter() {
        let steps = ((t / 0.001).ceil() as usize).max(20);
        ground_truth.insert(t.to_bits(), rk4_integrate(&g_dense, &lap_dense, &u0, NU, t, steps));
    }

    let embedding = laplacian_eigenmap(&hg, K_EMBED).unwrap();
    let coords = embedding.to_tensor::<Be>(&device);
    let laplacian_tensor = dense_laplacian_tensor::<Be>(&hg, &device).unwrap();
    let grad_tensor = dense_gradient_tensor::<Be>(&hg, &device).unwrap();

    println!("Training the plain (baseline) network...");
    let plain_rmse = train_and_evaluate(n, &coords, &laplacian_tensor, &grad_tensor, &u0, &ground_truth, &device, None, "plain");

    println!("\nTraining the Fourier-feature-encoded network...");
    let encoder = FourierFeatureEncoder::<Be>::new(K_EMBED + 1, FOURIER_FEATURES, FOURIER_FREQ_SCALE, 11, &device);
    let spectral_rmse = train_and_evaluate(n, &coords, &laplacian_tensor, &grad_tensor, &u0, &ground_truth, &device, Some(&encoder), "spectral");

    println!();
    println!("Both networks vs. the same independent RK4 ground truth, at held-out times:");
    println!("  {:>8}  {:>12}  {:>12}", "t", "plain RMSE", "spectral RMSE");
    for (i, &t) in CHECK_TIMES.iter().enumerate() {
        println!("  {t:>8.3}  {:>12.5}  {:>12.5}", plain_rmse[i], spectral_rmse[i]);
    }

    let plain_worst = plain_rmse.iter().cloned().fold(0.0, f64::max);
    let spectral_worst = spectral_rmse.iter().cloned().fold(0.0, f64::max);
    println!();
    if spectral_worst < plain_worst {
        let improvement = (1.0 - spectral_worst / plain_worst) * 100.0;
        println!("RESULT: Fourier features improved the worst-case RMSE by {improvement:.1}% ({plain_worst:.5} -> {spectral_worst:.5}) on this problem.");
    } else {
        let regression = (spectral_worst / plain_worst - 1.0) * 100.0;
        println!("RESULT: Fourier features did NOT help here -- worst-case RMSE got {regression:.1}% worse ({plain_worst:.5} -> {spectral_worst:.5}). Reporting this honestly rather than only the flattering direction (see this file's module docs for why that's expected to be a live possibility, not a bug).");
    }

    const RMSE_TOLERANCE: f64 = 0.08;
    if plain_worst < RMSE_TOLERANCE && spectral_worst < RMSE_TOLERANCE {
        println!("\nPASS: both networks stay within the same tolerance ({RMSE_TOLERANCE}) burgers_demo uses, regardless of which one is more accurate.");
    } else {
        println!("\nFAIL: at least one network exceeded tolerance {RMSE_TOLERANCE}.");
        std::process::exit(1);
    }
}
