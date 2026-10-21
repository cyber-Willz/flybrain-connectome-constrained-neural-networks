//! Connectome-constrained recurrent network ("FlyBrainNet") and the
//! machinery to fit it.
//!
//! **What is fixed and what is learned.** The wiring pattern (which neuron
//! synapses onto which) and each neuron's Dale's-law sign come from the
//! connectome and never change. The synaptic *magnitudes* are
//! `W_ij = mask_ij * base_ij * exp(theta_ij)` where `base` is the signed
//! synapse-count matrix **rescaled by the Perron radius of `|W|`**
//! (`gain / rho_abs`; the same "divide by the spectral radius" device NBSC
//! uses for `A / rho_B`), and `theta` is a learnable per-edge log-scale
//! initialised at 0. The sign of every synapse is therefore invariant under
//! training, and unconnected pairs stay exactly zero.
//!
//! **Depth = time.** Unrolling `T` Euler steps of
//! `r <- (1-a) r + a tanh(W r + b + drive)` is a `T`-layer weight-tied
//! deep stack over the connectome graph. The Dirichlet energy of the
//! activity matrix `r_t` (rows = neurons, w.r.t. the symmetrized wiring
//! graph) is tracked per step — the per-layer energy diagnostic of the
//! energy-stabilized-deep-stacks report — and can be added to the loss via
//! `nbsc::burn_layer::dirichlet_energy_differentiable`.
//!
//! Baselines for "what does the wiring buy you": the same network on a
//! degree-matched random wiring, and a fully dense unconstrained network.

use burn::backend::{Autodiff, NdArray};
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use nbsc::burn_layer::{degrees_tensor, dense_adjacency_tensor, dirichlet_energy, dirichlet_energy_differentiable};
use nbsc::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::analysis::dense_spectral_radius;
use crate::connectome::{CellClass, Connectome};

pub type Be = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
pub struct FlyBrainNet<B: Backend> {
    theta: Param<Tensor<B, 2>>,
    delta: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
    base: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    n: usize,
    alpha: f64,
    delta_gate: f64,
}

#[derive(Clone, Debug)]
pub struct NetConfig {
    /// Target Perron radius of `|W|` at initialisation.
    pub gain: f64,
    /// Euler step `dt / tau`.
    pub alpha: f64,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self { gain: 1.5, alpha: 0.2 }
    }
}

/// Signed count matrix rescaled so the Perron radius of `|W|` equals `gain`.
pub fn rescaled_signed_weights(c: &Connectome, gain: f64) -> Vec<f32> {
    let unsigned: Vec<f64> = c.count_matrix().iter().map(|&x| x as f64).collect();
    let rho = dense_spectral_radius(c.n, &unsigned, 80, 0).max(1e-9);
    let k = (gain / rho) as f32;
    c.signed_count_matrix().iter().map(|&x| x * k).collect()
}

fn t2<B: Backend>(v: &[f32], n: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 1>::from_floats(v, device).reshape([n, n])
}

impl<B: Backend> FlyBrainNet<B> {
    /// Connectome-constrained network: fixed wiring + signs, learnable magnitudes.
    pub fn from_connectome(c: &Connectome, cfg: &NetConfig, device: &B::Device) -> Self {
        let n = c.n;
        Self {
            theta: Param::from_tensor(Tensor::zeros([n, n], device)),
            delta: Param::from_tensor(Tensor::zeros([n, n], device)),
            bias: Param::from_tensor(Tensor::zeros([n], device)),
            base: t2::<B>(&rescaled_signed_weights(c, cfg.gain), n, device),
            mask: t2::<B>(&c.mask_matrix(), n, device),
            n,
            alpha: cfg.alpha,
            delta_gate: 0.0,
        }
    }

    /// Unconstrained baseline: every off-diagonal pair may connect, signs and
    /// magnitudes are free, random init at the same overall gain.
    pub fn dense_unconstrained(n: usize, cfg: &NetConfig, seed: u64, device: &B::Device) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = Normal::new(0.0, cfg.gain / (n as f64).sqrt() * 0.5).unwrap();
        let mut d = vec![0.0f32; n * n];
        let mut m = vec![1.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 0.0;
            for j in 0..n {
                d[i * n + j] = std.sample(&mut rng) as f32;
            }
        }
        Self {
            theta: Param::from_tensor(Tensor::zeros([n, n], device)),
            delta: Param::from_tensor(t2::<B>(&d, n, device)),
            bias: Param::from_tensor(Tensor::zeros([n], device)),
            base: Tensor::zeros([n, n], device),
            mask: t2::<B>(&m, n, device),
            n,
            alpha: cfg.alpha,
            delta_gate: 1.0,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Effective `[post, pre]` weight matrix.
    pub fn weights(&self) -> Tensor<B, 2> {
        let mut w = self.base.clone() * self.theta.val().exp();
        if self.delta_gate > 0.0 {
            w = w + self.delta.val() * self.delta_gate;
        }
        w * self.mask.clone()
    }

    pub fn bias_col(&self) -> Tensor<B, 2> {
        self.bias.val().unsqueeze_dim::<2>(1)
    }

    /// Unroll `steps` Euler steps from `r = 0` under a constant `drive`
    /// (`[n, batch]`); returns the state after every step.
    pub fn unroll(&self, drive: &Tensor<B, 2>, steps: usize) -> Vec<Tensor<B, 2>> {
        let w = self.weights();
        let b = self.bias_col();
        let [n, batch] = drive.dims();
        let device = drive.device();
        let mut r = Tensor::<B, 2>::zeros([n, batch], &device);
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let pre = w.clone().matmul(r.clone()) + b.clone() + drive.clone();
            r = r * (1.0 - self.alpha) + tanh(pre) * self.alpha;
            out.push(r.clone());
        }
        out
    }

    pub fn theta_values(&self) -> Vec<f32> {
        self.theta.val().into_data().convert::<f32>().value
    }
}

// ----------------------------------------------------------------------
// Ground-truth simulator (plain f64 CPU code, independent of Burn)
// ----------------------------------------------------------------------

/// Euler-unrolled network on the CPU, identical update rule to
/// [`FlyBrainNet::unroll`]. `w` is `[post, pre]` row-major, `drive` is
/// `[n, batch]` row-major; returns `steps` arrays of `[n, batch]`.
pub fn simulate_cpu(n: usize, w: &[f64], bias: &[f64], drive: &[f64], batch: usize, steps: usize, alpha: f64) -> Vec<Vec<f64>> {
    let mut r = vec![0.0f64; n * batch];
    let mut out = Vec::with_capacity(steps);
    let mut pre = vec![0.0f64; n * batch];
    for _ in 0..steps {
        for i in 0..n {
            let row = &w[i * n..(i + 1) * n];
            for b in 0..batch {
                let mut acc = bias[i] + drive[i * batch + b];
                for (j, &wij) in row.iter().enumerate() {
                    if wij != 0.0 {
                        acc += wij * r[j * batch + b];
                    }
                }
                pre[i * batch + b] = acc;
            }
        }
        for k in 0..n * batch {
            r[k] = (1.0 - alpha) * r[k] + alpha * pre[k].tanh();
        }
        out.push(r.clone());
    }
    out
}

/// A "ground-truth fly": the connectome's own wiring and signs with
/// log-normally perturbed magnitudes `theta* ~ N(0, sigma^2)` and small
/// random biases. This is the hidden system a connectome-constrained model
/// is asked to recover.
#[derive(Clone, Debug)]
pub struct Teacher {
    pub theta: Vec<f32>,
    pub bias: Vec<f32>,
    pub w: Vec<f64>,
}

pub fn make_teacher(c: &Connectome, cfg: &NetConfig, sigma: f64, seed: u64) -> Teacher {
    let n = c.n;
    let mut rng = StdRng::seed_from_u64(seed);
    let nd = Normal::new(0.0, sigma).unwrap();
    let base = rescaled_signed_weights(c, cfg.gain);
    let mask = c.mask_matrix();
    let mut theta = vec![0.0f32; n * n];
    let mut w = vec![0.0f64; n * n];
    for k in 0..n * n {
        if mask[k] > 0.0 {
            theta[k] = nd.sample(&mut rng) as f32;
            w[k] = base[k] as f64 * (theta[k] as f64).exp();
        }
    }
    let bn = Normal::new(0.0, 0.1).unwrap();
    let bias: Vec<f32> = (0..n).map(|_| bn.sample(&mut rng) as f32).collect();
    Teacher { theta, bias, w }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub batch: usize,
    pub steps: usize,
    /// `[n, batch]` row-major constant drive (non-zero only on sensory neurons).
    pub drive: Vec<f32>,
    /// `steps` arrays of `[n, batch]`.
    pub target: Vec<Vec<f32>>,
}

/// Random sensory step-inputs, simulated through the teacher.
pub fn make_dataset(c: &Connectome, teacher: &Teacher, cfg: &NetConfig, batch: usize, steps: usize, amp: f64, seed: u64) -> Dataset {
    let n = c.n;
    let mut rng = StdRng::seed_from_u64(seed);
    let nd = Normal::new(0.0, amp).unwrap();
    let mut drive = vec![0.0f64; n * batch];
    for &i in &c.indices_of(CellClass::Sensory) {
        for b in 0..batch {
            drive[i * batch + b] = nd.sample(&mut rng);
        }
    }
    let bias: Vec<f64> = teacher.bias.iter().map(|&x| x as f64).collect();
    let states = simulate_cpu(n, &teacher.w, &bias, &drive, batch, steps, cfg.alpha);
    Dataset {
        batch,
        steps,
        drive: drive.iter().map(|&x| x as f32).collect(),
        target: states.into_iter().map(|s| s.into_iter().map(|x| x as f32).collect()).collect(),
    }
}

// ----------------------------------------------------------------------
// Fitting
// ----------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FitConfig {
    pub epochs: usize,
    pub lr: f64,
    /// Weight on the Dirichlet-energy regulariser (0 = off).
    pub energy_lambda: f64,
    /// `n`-vector of 0/1: which neurons' activity the loss may see.
    pub obs_mask: Option<Vec<f32>>,
    pub log_every: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self { epochs: 300, lr: 0.03, energy_lambda: 0.0, obs_mask: None, log_every: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    /// `sum (pred - target)^2 / sum target^2` over every neuron, step and trial.
    pub rel_mse: f64,
    /// Same restricted to neurons the loss never saw (NaN if all observed).
    pub rel_mse_unobserved: f64,
    /// Mean Dirichlet energy (CPU diagnostic) of the student's activity at each step.
    pub energy_profile: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub train_curve: Vec<f32>,
    pub train: EvalResult,
    pub test: EvalResult,
    /// Mean Dirichlet energy of the teacher's own test activity per step
    /// (the "target" energy profile).
    pub teacher_energy_profile: Vec<f32>,
    /// Pearson correlation between learned and true `theta` on wired pairs
    /// (only meaningful when the model uses the same wiring as the teacher).
    pub theta_corr: Option<f64>,
}

fn to_tensor2(v: &[f32], n: usize, b: usize, device: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    Tensor::<Be, 1>::from_floats(v, device).reshape([n, b])
}

pub fn evaluate(
    net: &FlyBrainNet<Be>,
    ds: &Dataset,
    graph: &Graph,
    obs: Option<&[f32]>,
    device: &<Be as Backend>::Device,
) -> (EvalResult, Vec<Vec<f32>>) {
    let n = net.n();
    let drive = to_tensor2(&ds.drive, n, ds.batch, device);
    let states = net.unroll(&drive, ds.steps);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut num_u = 0.0f64;
    let mut den_u = 0.0f64;
    let mut energy = Vec::new();
    let mut preds = Vec::new();
    for (t, s) in states.iter().enumerate() {
        let pred: Vec<f32> = s.clone().into_data().convert::<f32>().value;
        for i in 0..n {
            let unobs = obs.map(|o| o[i] < 0.5).unwrap_or(false);
            for b in 0..ds.batch {
                let (p, y) = (pred[i * ds.batch + b] as f64, ds.target[t][i * ds.batch + b] as f64);
                num += (p - y) * (p - y);
                den += y * y;
                if unobs {
                    num_u += (p - y) * (p - y);
                    den_u += y * y;
                }
            }
        }
        energy.push(dirichlet_energy::<Be>(graph, s));
        preds.push(pred);
    }
    (
        EvalResult { rel_mse: num / den.max(1e-12), rel_mse_unobserved: if den_u > 0.0 { num_u / den_u } else { f64::NAN }, energy_profile: energy },
        preds,
    )
}

pub fn teacher_energy_profile(ds: &Dataset, graph: &Graph, device: &<Be as Backend>::Device) -> Vec<f32> {
    let n = graph.n;
    ds.target.iter().map(|s| dirichlet_energy::<Be>(graph, &to_tensor2(s, n, ds.batch, device))).collect()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa.sqrt() * sbb.sqrt()).max(1e-12)
}

/// Train `net` to reproduce `train`'s activity and evaluate on `test`.
/// `true_theta` (if the wiring matches the teacher's) enables weight-recovery scoring.
pub fn fit(
    mut net: FlyBrainNet<Be>,
    train: &Dataset,
    test: &Dataset,
    graph: &Graph,
    mask01: &[f32],
    true_theta: Option<&[f32]>,
    cfg: &FitConfig,
) -> (FlyBrainNet<Be>, FitResult) {
    let device = <Be as Backend>::Device::default();
    let n = net.n();
    let adj = dense_adjacency_tensor::<Be>(graph, &device);
    let deg = degrees_tensor::<Be>(graph, &device);
    let obs_vec: Vec<f32> = cfg.obs_mask.clone().unwrap_or_else(|| vec![1.0; n]);
    let obs = Tensor::<Be, 1>::from_floats(obs_vec.as_slice(), &device).reshape([n, 1]);
    let n_obs: f32 = obs_vec.iter().sum();
    let norm = (train.steps * train.batch) as f32 * n_obs;

    let drive = to_tensor2(&train.drive, n, train.batch, &device);
    let targets: Vec<Tensor<Be, 2>> = train.target.iter().map(|t| to_tensor2(t, n, train.batch, &device)).collect();

    let mut optim = AdamConfig::new().init();
    let mut curve = Vec::with_capacity(cfg.epochs);
    for epoch in 0..cfg.epochs {
        let states = net.unroll(&drive, train.steps);
        let mut sq = Tensor::<Be, 1>::zeros([1], &device);
        let mut reg = Tensor::<Be, 1>::zeros([1], &device);
        for (s, y) in states.iter().zip(&targets) {
            let d = (s.clone() - y.clone()) * obs.clone();
            sq = sq + (d.clone() * d).sum();
            if cfg.energy_lambda > 0.0 {
                reg = reg + dirichlet_energy_differentiable::<Be>(&adj, &deg, graph.m(), s);
            }
        }
        let mut loss = sq / norm;
        let data_loss: f32 = loss.clone().into_scalar();
        if cfg.energy_lambda > 0.0 {
            loss = loss + reg * (cfg.energy_lambda / train.steps as f64);
        }
        curve.push(data_loss);
        if cfg.log_every > 0 && (epoch % cfg.log_every == 0 || epoch + 1 == cfg.epochs) {
            println!("    epoch {epoch:4}: data mse {data_loss:.6}");
        }
        let grads = GradientsParams::from_grads(loss.backward(), &net);
        net = optim.step(cfg.lr, net, grads);
    }

    let obs_ref = cfg.obs_mask.as_deref();
    let (train_eval, _) = evaluate(&net, train, graph, obs_ref, &device);
    let (test_eval, _) = evaluate(&net, test, graph, obs_ref, &device);
    let theta_corr = true_theta.map(|tt| {
        let learned = net.theta_values();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for k in 0..n * n {
            if mask01[k] > 0.0 {
                a.push(learned[k] as f64);
                b.push(tt[k] as f64);
            }
        }
        pearson(&a, &b)
    });
    let teacher_energy_profile = teacher_energy_profile(test, graph, &device);
    (net, FitResult { train_curve: curve, train: train_eval, test: test_eval, teacher_energy_profile, theta_corr })
}

/// Random 0/1 observation mask with `frac` of neurons observed.
pub fn random_obs_mask(n: usize, frac: f64, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| if rng.gen::<f64>() < frac { 1.0 } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::SyntheticConfig;

    fn tiny() -> Connectome {
        Connectome::synthetic(&SyntheticConfig { n: 96, n_modules: 6, seed: 1, ..Default::default() })
    }

    #[test]
    fn burn_unroll_matches_cpu_simulator() {
        let c = tiny();
        let cfg = NetConfig::default();
        let device = <Be as Backend>::Device::default();
        let net = FlyBrainNet::<Be>::from_connectome(&c, &cfg, &device);
        let teacher = Teacher { theta: vec![0.0; c.n * c.n], bias: vec![0.0; c.n], w: rescaled_signed_weights(&c, cfg.gain).iter().map(|&x| x as f64).collect() };
        let ds = make_dataset(&c, &teacher, &cfg, 4, 10, 1.0, 3);
        let drive = to_tensor2(&ds.drive, c.n, 4, &device);
        let st = net.unroll(&drive, 10);
        let last: Vec<f32> = st[9].clone().into_data().convert::<f32>().value;
        let max_err = last.iter().zip(&ds.target[9]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "burn vs cpu max err {max_err}");
    }

    #[test]
    fn weights_respect_mask_and_dale_signs() {
        let c = tiny();
        let device = <Be as Backend>::Device::default();
        let net = FlyBrainNet::<Be>::from_connectome(&c, &NetConfig::default(), &device);
        let w: Vec<f32> = net.weights().into_data().convert::<f32>().value;
        let mask = c.mask_matrix();
        for post in 0..c.n {
            for pre in 0..c.n {
                let x = w[post * c.n + pre];
                if mask[post * c.n + pre] == 0.0 {
                    assert_eq!(x, 0.0);
                } else {
                    assert_eq!(x.signum(), c.sign[pre], "synapse {pre}->{post} violates Dale's law");
                }
            }
        }
    }

    #[test]
    fn perron_radius_of_init_weights_equals_gain() {
        let c = tiny();
        let w = rescaled_signed_weights(&c, 1.5);
        let abs: Vec<f64> = w.iter().map(|x| x.abs() as f64).collect();
        let r = dense_spectral_radius(c.n, &abs, 80, 0);
        assert!((r - 1.5).abs() < 1e-3, "got {r}");
    }

    #[test]
    fn training_reduces_loss_and_keeps_signs() {
        let c = tiny();
        let cfg = NetConfig::default();
        let teacher = make_teacher(&c, &cfg, 0.4, 7);
        let tr = make_dataset(&c, &teacher, &cfg, 8, 12, 1.0, 1);
        let te = make_dataset(&c, &teacher, &cfg, 8, 12, 1.0, 2);
        let device = <Be as Backend>::Device::default();
        let net = FlyBrainNet::<Be>::from_connectome(&c, &cfg, &device);
        let g = c.symmetrized_graph();
        let fc = FitConfig { epochs: 60, lr: 0.05, ..Default::default() };
        let (net, res) = fit(net, &tr, &te, &g, &c.mask_matrix(), Some(&teacher.theta), &fc);
        assert!(res.train_curve.last().unwrap() < &(res.train_curve[0] * 0.5), "loss {:?}", (res.train_curve[0], res.train_curve.last()));
        assert!(res.theta_corr.unwrap() > 0.2);
        // signs preserved after training
        let w: Vec<f32> = net.weights().into_data().convert::<f32>().value;
        for post in 0..c.n {
            for pre in 0..c.n {
                let x = w[post * c.n + pre];
                if x != 0.0 {
                    assert_eq!(x.signum(), c.sign[pre]);
                }
            }
        }
    }
}
