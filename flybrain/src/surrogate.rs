//! SHPINN continuous-time surrogate of a connectome's rate dynamics.
//!
//! The setting this models: you have the **connectome** (wiring + signs +
//! approximate synapse-count weights) and a **partial recording** — activity
//! of a random subset of neurons at sampled times under a known stimulus —
//! and you want the activity of the *unrecorded* neurons.
//!
//! * Coordinates: Laplacian eigenmap of the axonal-arbor hypergraph (the
//!   true hypergraph Laplacian, via `shpinn::embedding`), so neurons with
//!   similar wiring context get similar coordinates.
//! * Network: `shpinn::pinn::Shpinn`, a `tanh`-MLP over `[coords, t]`.
//! * Physics: the connectome's own rate equation
//!   `du/dt + u - tanh(W u + b + I(t)) = 0`, enforced at random collocation
//!   times; `du/dt` by central finite difference
//!   (`shpinn::physics::central_time_derivative`), `W u` by one matmul.
//!   Because the equation of every *recorded* neuron contains its
//!   unrecorded presynaptic partners, the residual couples the unobserved
//!   activity to the observed one.
//!
//! Three variants are compared: data only, physics with the *nominal*
//! count-derived `W` (wrong magnitudes — what you actually have), and
//! physics with the true `W` (oracle upper bound).

use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use shpinn::embedding::laplacian_eigenmap;
use shpinn::physics::central_time_derivative;
use shpinn::pinn::{Shpinn, ShpinnConfig};

use crate::connectome::{CellClass, Connectome};
use crate::dynamics::{rescaled_signed_weights, Be, NetConfig, Teacher};
use crate::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicsMode {
    DataOnly,
    NominalW,
    OracleW,
}

impl PhysicsMode {
    pub fn name(&self) -> &'static str {
        match self {
            PhysicsMode::DataOnly => "data only",
            PhysicsMode::NominalW => "physics, nominal W",
            PhysicsMode::OracleW => "physics, true W (oracle)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurrogateConfig {
    pub k_embed: usize,
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub data_times_per_epoch: usize,
    pub colloc_times_per_epoch: usize,
    pub data_weight: f64,
    pub physics_weight: f64,
    pub eps: f64,
    pub t_max: f64,
    /// Fraction of neurons recorded.
    pub obs_frac: f64,
    /// Spacing of the recording grid, in integration steps of `0.005`
    /// (20 = a sample every 0.1 time units, 100 = every 0.5).
    pub data_stride: usize,
    /// Std-dev of Gaussian measurement noise added to the recorded samples
    /// (the held-out truth stays clean). `0.0` = noiseless recording.
    pub obs_noise: f64,
    pub seed: u64,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            k_embed: 24,
            hidden: vec![96, 96, 96],
            epochs: 2500,
            lr: 2e-3,
            data_times_per_epoch: 5,
            colloc_times_per_epoch: 5,
            data_weight: 10.0,
            physics_weight: 1.0,
            eps: 5e-3,
            t_max: 3.0,
            obs_frac: 0.3,
            data_stride: 20,
            obs_noise: 0.0,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurrogateResult {
    pub mode: PhysicsMode,
    /// `||pred - truth|| / ||truth||` on held-out times, unrecorded neurons.
    pub rel_err_unobserved: f64,
    pub rel_err_observed: f64,
    /// Value of the total training loss at the last epoch.
    pub final_loss: f32,
}

/// Stimulus time profile `g(t)`; drive on sensory neuron `j` is `amp_j * g(t)`.
pub fn drive_profile(t: f64) -> f64 {
    0.6 + 0.4 * (2.1 * t).sin()
}

/// RK4 integration of `du/dt = -u + tanh(W u + b + amp * g(t))` from `u(0)=0`.
/// Returns states at `t = k * dt`, `k = 0..=steps`.
pub fn simulate_continuous(n: usize, w: &[f64], bias: &[f64], amp: &[f64], dt: f64, steps: usize) -> Vec<Vec<f64>> {
    let f = |u: &[f64], t: f64, out: &mut [f64]| {
        let g = drive_profile(t);
        for i in 0..n {
            let row = &w[i * n..(i + 1) * n];
            let mut acc = bias[i] + amp[i] * g;
            for (j, &wij) in row.iter().enumerate() {
                if wij != 0.0 {
                    acc += wij * u[j];
                }
            }
            out[i] = -u[i] + acc.tanh();
        }
    };
    let mut u = vec![0.0f64; n];
    let mut states = vec![u.clone()];
    let (mut k1, mut k2, mut k3, mut k4, mut tmp) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for s in 0..steps {
        let t = s as f64 * dt;
        f(&u, t, &mut k1);
        for i in 0..n {
            tmp[i] = u[i] + 0.5 * dt * k1[i];
        }
        f(&tmp, t + 0.5 * dt, &mut k2);
        for i in 0..n {
            tmp[i] = u[i] + 0.5 * dt * k2[i];
        }
        f(&tmp, t + 0.5 * dt, &mut k3);
        for i in 0..n {
            tmp[i] = u[i] + dt * k3[i];
        }
        f(&tmp, t + dt, &mut k4);
        for i in 0..n {
            u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        states.push(u.clone());
    }
    states
}

pub struct SurrogateProblem {
    pub(crate) cfg: SurrogateConfig,
    pub(crate) n: usize,
    pub(crate) device: <Be as Backend>::Device,
    coords: Tensor<Be, 2>,
    /// Ground truth at `t = k * DT`.
    pub(crate) truth: Vec<Vec<f64>>,
    /// What was actually recorded: `obs_data[k][i]` is neuron `i` at grid
    /// time `k * stride * DT` (truth + noise for recorded neurons; `u(0)=0`
    /// is exact; unrecorded entries are unused filler).
    pub(crate) obs_data: Vec<Vec<f64>>,
    pub(crate) obs: Vec<bool>,
    pub(crate) obs_col: Tensor<Be, 2>,
    pub(crate) amp_col: Tensor<Be, 2>,
    pub(crate) w_nominal: Tensor<Be, 2>,
    pub(crate) w_true: Tensor<Be, 2>,
    pub(crate) b_true: Tensor<Be, 2>,
    zero_b: Tensor<Be, 2>,
    pub(crate) n_edges: usize,
}

pub(crate) const DT: f64 = 0.005;

/// Cubic Hermite (Catmull–Rom slopes) interpolation of a uniformly sampled
/// series `y[k]` at times `k * s`, evaluated at `t` (clamped to the sampled range).
pub fn hermite_eval(y: &[f64], s: f64, t: f64) -> f64 {
    let kmax = y.len() - 1;
    if kmax == 0 {
        return y[0];
    }
    let t = t.clamp(0.0, kmax as f64 * s);
    let k = ((t / s).floor() as usize).min(kmax - 1);
    let x = (t - k as f64 * s) / s;
    let slope = |j: usize| -> f64 {
        if j == 0 {
            (y[1] - y[0]) / s
        } else if j == kmax {
            (y[kmax] - y[kmax - 1]) / s
        } else {
            (y[j + 1] - y[j - 1]) / (2.0 * s)
        }
    };
    let (x2, x3) = (x * x, x * x * x);
    (2.0 * x3 - 3.0 * x2 + 1.0) * y[k] + (x3 - 2.0 * x2 + x) * s * slope(k) + (-2.0 * x3 + 3.0 * x2) * y[k + 1] + (x3 - x2) * s * slope(k + 1)
}

impl SurrogateProblem {
    pub fn new(c: &Connectome, teacher: &Teacher, net_cfg: &NetConfig, cfg: SurrogateConfig, amp_scale: f64) -> Result<Self> {
        let n = c.n;
        let device = <Be as Backend>::Device::default();
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5eed);

        // Stimulus amplitudes on sensory neurons.
        let mut amp = vec![0.0f64; n];
        for &i in &c.indices_of(CellClass::Sensory) {
            amp[i] = amp_scale * (rng.gen::<f64>() * 2.0 - 1.0) * 1.7;
        }
        let bias: Vec<f64> = teacher.bias.iter().map(|&x| x as f64).collect();
        let steps = (cfg.t_max / DT).round() as usize;
        let truth = simulate_continuous(n, &teacher.w, &bias, &amp, DT, steps);

        // Spectral coordinates from the arbor hypergraph.
        let hg = c.arbor_hypergraph()?;
        let emb = laplacian_eigenmap(&hg, cfg.k_embed)?;
        let mut cdata = Vec::with_capacity(n * cfg.k_embed);
        let scale = (n as f64).sqrt();
        for i in 0..n {
            for j in 0..cfg.k_embed {
                cdata.push((emb.coords[(i, j)] * scale) as f32);
            }
        }
        let coords = Tensor::<Be, 1>::from_floats(cdata.as_slice(), &device).reshape([n, cfg.k_embed]);

        let obs: Vec<bool> = (0..n).map(|_| rng.gen::<f64>() < cfg.obs_frac).collect();
        let stride_ = cfg.data_stride.max(2);
        let n_grid_ = (truth.len() - 1) / stride_;
        let mut nrng = StdRng::seed_from_u64(cfg.seed ^ 0xbeef_u64);
        let noise = Normal::new(0.0, cfg.obs_noise.max(0.0)).expect("noise sd");
        let obs_data: Vec<Vec<f64>> = (0..=n_grid_)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        let y = truth[k * stride_][i];
                        if k > 0 && obs[i] && cfg.obs_noise > 0.0 {
                            y + noise.sample(&mut nrng)
                        } else {
                            y
                        }
                    })
                    .collect()
            })
            .collect();
        let n_edges = w_edge_count(c);
        let obs_f: Vec<f32> = obs.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let col = |v: &[f32]| Tensor::<Be, 1>::from_floats(v, &device).reshape([n, 1]);
        let w_nom: Vec<f32> = rescaled_signed_weights(c, net_cfg.gain);
        let w_true: Vec<f32> = teacher.w.iter().map(|&x| x as f32).collect();
        let t2 = |v: &[f32]| Tensor::<Be, 1>::from_floats(v, &device).reshape([n, n]);
        Ok(Self {
            n,
            coords,
            obs_col: col(&obs_f),
            amp_col: col(&amp.iter().map(|&x| x as f32).collect::<Vec<_>>()),
            w_nominal: t2(&w_nom),
            w_true: t2(&w_true),
            b_true: col(&teacher.bias),
            zero_b: col(&vec![0.0; n]),
            truth,
            obs_data,
            obs,
            device,
            cfg,
            n_edges,
        })
    }

    pub fn n_observed(&self) -> usize {
        self.obs.iter().filter(|&&b| b).count()
    }

    fn inputs(&self, t: f64) -> Tensor<Be, 2> {
        let tcol = Tensor::<Be, 1>::from_floats(vec![t as f32; self.n].as_slice(), &self.device).reshape([self.n, 1]);
        Tensor::cat(vec![self.coords.clone(), tcol], 1)
    }

    /// Recorded values at grid index `k` (noisy if `obs_noise > 0`).
    fn data_col(&self, k: usize) -> Tensor<Be, 2> {
        let v: Vec<f32> = self.obs_data[k].iter().map(|&x| x as f32).collect();
        Tensor::<Be, 1>::from_floats(v.as_slice(), &self.device).reshape([self.n, 1])
    }

    /// Non-neural baseline for the recorded neurons: linear interpolation
    /// between the two neighbouring recording times, scored at the same
    /// held-out midpoint times as [`Self::run`].
    pub fn linear_interp_baseline_rel_err(&self) -> f64 {
        self.interp_baseline_rel_err(false)
    }

    /// Cubic-Hermite counterpart of [`Self::linear_interp_baseline_rel_err`]
    /// (the stronger non-neural baseline).
    pub fn cubic_interp_baseline_rel_err(&self) -> f64 {
        self.interp_baseline_rel_err(true)
    }

    fn interp_baseline_rel_err(&self, cubic: bool) -> f64 {
        let s = self.cfg.data_stride.max(2) as f64 * DT;
        let series = |i: usize| -> Vec<f64> { self.obs_data.iter().map(|r| r[i]).collect() };
        let stride = self.cfg.data_stride.max(2);
        let n_grid = (self.truth.len() - 1) / stride;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for k in 0..n_grid {
            let mid = k * stride + stride / 2;
            if mid >= self.truth.len() || (k + 1) * stride >= self.truth.len() {
                break;
            }
            for i in 0..self.n {
                if self.obs[i] {
                    let guess = if cubic { hermite_eval(&series(i), s, mid as f64 * DT) } else { 0.5 * (self.obs_data[k][i] + self.obs_data[k + 1][i]) };
                    let y = self.truth[mid][i];
                    num += (guess - y) * (guess - y);
                    den += y * y;
                }
            }
        }
        (num / den.max(1e-12)).sqrt()
    }

    pub fn run(&self, mode: PhysicsMode) -> SurrogateResult {
        let cfg = &self.cfg;
        let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_mul(7919).wrapping_add(11));
        let mut net: Shpinn<Be> = ShpinnConfig::new(cfg.k_embed + 1, cfg.hidden.clone(), 1).init(&self.device);
        let mut optim = AdamConfig::new().init();
        let stride = cfg.data_stride.max(2);
        let n_data_grid = (self.truth.len() - 1) / stride; // grid indices 0, S, 2S, ..., n_data_grid*S <= steps
        let n_obs = self.n_observed().max(1) as f32;
        let (w, b) = match mode {
            PhysicsMode::DataOnly => (None, None),
            PhysicsMode::NominalW => (Some(&self.w_nominal), Some(&self.zero_b)),
            PhysicsMode::OracleW => (Some(&self.w_true), Some(&self.b_true)),
        };
        let targets: Vec<Tensor<Be, 2>> = (0..=n_data_grid).map(|k| self.data_col(k)).collect();

        let mut last = 0.0f32;
        for _epoch in 0..cfg.epochs {
            // --- data term: initial condition u(0)=0 everywhere + recorded neurons at grid times
            let u0 = net.forward(self.inputs(0.0));
            let mut loss = (u0.clone() * u0).mean() * cfg.data_weight;
            for _ in 0..cfg.data_times_per_epoch {
                let k = rng.gen_range(1..=n_data_grid);
                let t = (k * stride) as f64 * DT;
                let u = net.forward(self.inputs(t));
                let d = (u - targets[k].clone()) * self.obs_col.clone();
                loss = loss + (d.clone() * d).sum() * (cfg.data_weight / n_obs as f64);
            }
            // --- physics term
            if let (Some(w), Some(b)) = (w, b) {
                for _ in 0..cfg.colloc_times_per_epoch {
                    let t = rng.gen::<f64>() * cfg.t_max;
                    let u = net.forward(self.inputs(t));
                    let up = net.forward(self.inputs(t + cfg.eps));
                    let um = net.forward(self.inputs(t - cfg.eps));
                    let ut = central_time_derivative::<Be>(up, um, cfg.eps);
                    let drive = self.amp_col.clone() * drive_profile(t);
                    let pre = w.clone().matmul(u.clone()) + b.clone() + drive;
                    let res = ut + u - tanh(pre);
                    loss = loss + (res.clone() * res).mean() * (cfg.physics_weight / cfg.colloc_times_per_epoch as f64);
                }
            }
            last = loss.clone().into_scalar();
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = optim.step(cfg.lr, net, grads);
        }

        // --- evaluation on held-out times (odd multiples of half the grid spacing)
        let (mut num_u, mut den_u, mut num_o, mut den_o) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..n_data_grid {
            let step = k * stride + stride / 2;
            if step >= self.truth.len() {
                break;
            }
            let t = step as f64 * DT;
            let pred: Vec<f32> = net.forward(self.inputs(t)).into_data().convert::<f32>().value;
            for i in 0..self.n {
                let e = pred[i] as f64 - self.truth[step][i];
                let y = self.truth[step][i];
                if self.obs[i] {
                    num_o += e * e;
                    den_o += y * y;
                } else {
                    num_u += e * e;
                    den_u += y * y;
                }
            }
        }
        SurrogateResult {
            mode,
            rel_err_unobserved: if den_u > 0.0 { (num_u / den_u).sqrt() } else { f64::NAN },
            rel_err_observed: (num_o / den_o.max(1e-12)).sqrt(),
            final_loss: last,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::SyntheticConfig;
    use crate::dynamics::make_teacher;

    #[test]
    fn rk4_matches_analytic_for_decoupled_decay() {
        // W = 0, no bias, no drive: du/dt = -u + tanh(0) = -u, u(0)=0 => u stays 0.
        let n = 3;
        let s = simulate_continuous(n, &vec![0.0; 9], &vec![0.0; 3], &vec![0.0; 3], 0.01, 100);
        assert!(s.last().unwrap().iter().all(|&x| x.abs() < 1e-12));
        // constant bias 0.5: du/dt = -u + tanh(0.5), u(t) = tanh(.5)(1-e^{-t})
        let s = simulate_continuous(1, &[0.0], &[0.5], &[0.0], 0.01, 200);
        let exact = 0.5f64.tanh() * (1.0 - (-2.0f64).exp());
        assert!((s.last().unwrap()[0] - exact).abs() < 1e-8);
    }

    #[test]
    fn surrogate_runs_end_to_end_on_tiny_problem() {
        let c = Connectome::synthetic(&SyntheticConfig { n: 48, n_modules: 6, seed: 4, ..Default::default() });
        let net_cfg = NetConfig::default();
        let teacher = make_teacher(&c, &net_cfg, 0.3, 1);
        let cfg = SurrogateConfig { k_embed: 6, hidden: vec![16, 16], epochs: 30, ..Default::default() };
        let p = SurrogateProblem::new(&c, &teacher, &net_cfg, cfg, 1.0).unwrap();
        for m in [PhysicsMode::DataOnly, PhysicsMode::NominalW, PhysicsMode::OracleW] {
            let r = p.run(m);
            assert!(r.rel_err_unobserved.is_finite() && r.final_loss.is_finite());
        }
    }
}

fn w_edge_count(c: &Connectome) -> usize {
    c.mask_matrix().iter().filter(|&&m| m > 0.0).count()
}
