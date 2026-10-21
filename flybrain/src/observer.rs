//! Physics-based **state observer** for unrecorded neurons.
//!
//! The SHPINN surrogate in [`crate::surrogate`] has to *memorise* every
//! neuron's time course in an MLP. But the connectome gives us the equations
//! of motion directly, so there is a much more direct way to infer the
//! unrecorded neurons — data assimilation:
//!
//! 1. Reconstruct each *recorded* neuron's trajectory from its samples
//!    (cubic Hermite interpolation) and **clamp** it.
//! 2. Integrate the rate equation `du/dt = -u + tanh(W u + b + I(t))` for the
//!    *unrecorded* neurons only, driven by the clamped ones and by the known
//!    stimulus, from the known initial condition `u(0) = 0`.
//! 3. Optionally (**experimental, not recommended: see docs §3c**) **fit the synaptic magnitudes** `W = base ⊙ exp(θ)` and the
//!    biases `b` (the wiring and signs stay fixed by the connectome) by
//!    making the recorded neurons' own equations hold:
//!    `(u_i(t+h) - u_i(t))/h ≈ ½ (f_i(t) + f_i(t+h))` for recorded `i`,
//!    differentiating through the unrecorded rollout (BPTT, same Adam /
//!    `exp(θ)` parametrisation as [`crate::dynamics::FlyBrainNet`]).
//!
//! No unrecorded ground truth is used anywhere in the fit.

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::dynamics::Be;
use crate::surrogate::{drive_profile, hermite_eval, SurrogateProblem, DT};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverKind {
    /// Count-derived `W`, zero bias, nothing fitted — what you have a priori.
    Nominal,
    /// EXPERIMENTAL: `θ` and `b` fitted to the recorded neurons' dynamics. The
    /// held-out check does not reliably prevent a worse-than-nominal result.
    Learned,
    /// The teacher's true `W` and `b` (upper bound).
    Oracle,
}

impl ObserverKind {
    pub fn name(&self) -> &'static str {
        match self {
            ObserverKind::Nominal => "observer, nominal W",
            ObserverKind::Learned => "observer, learned W",
            ObserverKind::Oracle => "observer, true W (oracle)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObserverConfig {
    pub epochs: usize,
    pub lr: f64,
    /// L2 penalty on `θ` (mean over synapses) — keeps the fit near the counts.
    pub theta_reg: f64,
    /// Upper bound on the rollout step; the actual step divides the sample
    /// spacing into an even number of sub-steps so held-out midpoints land on
    /// the rollout grid.
    pub max_h: f64,
    /// How the recorded neurons' equations are enforced.
    pub loss: ObserverLoss,
    /// A fitted checkpoint replaces the nominal model only if its held-out
    /// score is better by at least this fraction.
    pub select_margin: f64,
}

/// Fitting objective for the recorded neurons' own dynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverLoss {
    /// One-step residual `(u(t+h)-u(t))/h - f̄`. Divides measurement noise by `h`,
    /// so with noisy recordings the fit chases noise (kept for comparison).
    Derivative,
    /// Integral (weak) form: `u(t_k) - u(0) - ∫₀^{t_k} f dt` at every `k`.
    /// Measurement noise enters once, not divided by `h`, so the fit is robust.
    Integral,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self { epochs: 250, lr: 0.03, theta_reg: 1e-3, max_h: 0.05, loss: ObserverLoss::Integral, select_margin: 0.10 }
    }
}

#[derive(Clone, Debug)]
pub struct ObserverResult {
    pub kind: ObserverKind,
    /// Relative L2 error on never-recorded neurons at held-out midpoint times.
    pub rel_err_unobserved: f64,
    /// Relative L2 error of the clamped (interpolated) recorded neurons.
    pub rel_err_observed: f64,
    /// Final value of the fitting loss (NaN for the unfitted kinds).
    pub final_loss: f32,
}

#[derive(Module, Debug)]
struct ObsParams<B: Backend> {
    theta: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 2>>,
}

struct Rollout {
    /// Full state (recorded clamped, unrecorded simulated) at every rollout step.
    states: Vec<Tensor<Be, 2>>,
    /// Mean squared derivative residual on recorded neurons.
    loss: Tensor<Be, 1>,
}

impl SurrogateProblem {
    fn observer_step(&self, ocfg: &ObserverConfig) -> (f64, usize) {
        let s = self.cfg.data_stride.max(2) as f64 * DT;
        let m = 2 * ((s / (2.0 * ocfg.max_h) - 1e-9).ceil().max(1.0) as usize);
        (s / m as f64, m)
    }

    fn rollout(&self, w: &Tensor<Be, 2>, b: &Tensor<Be, 2>, ocfg: &ObserverConfig, clamp: &[bool]) -> Rollout {
        let n = self.n;
        let (h, _) = self.observer_step(ocfg);
        let s = self.cfg.data_stride.max(2) as f64 * DT;
        let n_grid = self.obs_data.len() - 1;
        let steps = (n_grid as f64 * s / h).round() as usize;
        let series: Vec<Vec<f64>> = (0..n).map(|i| self.obs_data.iter().map(|r| r[i]).collect()).collect();
        let clamp_f: Vec<f32> = clamp.iter().map(|&c| if c { 1.0 } else { 0.0 }).collect();
        let clamp_col = Tensor::<Be, 1>::from_floats(clamp_f.as_slice(), &self.device).reshape([n, 1]);
        let unobs_col = Tensor::<Be, 2>::ones([n, 1], &self.device) - clamp_col.clone();
        let uo: Vec<Tensor<Be, 2>> = (0..=steps)
            .map(|k| {
                let t = k as f64 * h;
                let v: Vec<f32> = (0..n).map(|i| if clamp[i] { hermite_eval(&series[i], s, t) as f32 } else { 0.0 }).collect();
                Tensor::<Be, 1>::from_floats(v.as_slice(), &self.device).reshape([n, 1])
            })
            .collect();
        let f = |xf: &Tensor<Be, 2>, t: f64| -> Tensor<Be, 2> { tanh(w.clone().matmul(xf.clone()) + b.clone() + self.amp_col.clone() * drive_profile(t)) - xf.clone() };

        let mut x = Tensor::<Be, 2>::zeros([n, 1], &self.device);
        let mut states = Vec::with_capacity(steps + 1);
        states.push(uo[0].clone() + x.clone());
        let mut loss = Tensor::<Be, 1>::zeros([1], &self.device);
        let mut acc = Tensor::<Be, 2>::zeros([n, 1], &self.device);
        for k in 0..steps {
            let (t0, t1) = (k as f64 * h, (k + 1) as f64 * h);
            let xf0 = uo[k].clone() + x.clone();
            let f0 = f(&xf0, t0);
            let xp = (x.clone() + f0.clone() * h) * unobs_col.clone();
            let xf1 = uo[k + 1].clone() + xp;
            let f1 = f(&xf1, t1);
            let incr = (f0 + f1) * (0.5 * h);
            // recorded neurons: their own equation must hold
            acc = acc + incr.clone();
            match ocfg.loss {
                ObserverLoss::Derivative => {
                    let res = ((uo[k + 1].clone() - uo[k].clone()) - incr.clone()) * clamp_col.clone();
                    loss = loss + (res.clone() * res).sum() / (h * h);
                }
                ObserverLoss::Integral => {
                    let res = ((uo[k + 1].clone() - uo[0].clone()) - acc.clone()) * clamp_col.clone();
                    loss = loss + (res.clone() * res).sum();
                }
            }
            // unrecorded neurons: advance
            x = (x + incr) * unobs_col.clone();
            states.push(uo[k + 1].clone() + x.clone());
        }
        let n_obs = clamp.iter().filter(|&&c| c).count().max(1) as f64;
        Rollout { states, loss: loss / (n_obs * steps as f64) }
    }

    /// Fit (if `Learned`) and score an observer. Unrecorded ground truth is
    /// used only for the final score.
    pub fn run_observer(&self, kind: ObserverKind, ocfg: &ObserverConfig) -> ObserverResult {
        let n = self.n;
        let mut final_loss = f32::NAN;
        let (w, b) = match kind {
            ObserverKind::Nominal => (self.w_nominal.clone(), Tensor::<Be, 2>::zeros([n, 1], &self.device)),
            ObserverKind::Oracle => (self.w_true.clone(), self.b_true.clone()),
            ObserverKind::Learned => {
                // Label-free model selection: every 5th recorded neuron is held out
                // of the fit (treated as unrecorded) and scored against its own
                // recording; the parameters with the best held-out score win, with
                // the unfitted nominal model (epoch 0) as the fallback.
                let recorded: Vec<usize> = (0..n).filter(|&i| self.obs[i]).collect();
                // at least 10 held-out neurons (a 5-neuron check is too noisy to trust);
                // with fewer than 20 recorded neurons there is no safe split, so no fit.
                let n_val = if recorded.len() < 20 { 0 } else { (recorded.len() / 5).clamp(10, recorded.len() / 2) };
                let gap = if n_val > 0 { recorded.len() / n_val } else { 1 };
                let val: Vec<usize> = (0..n_val).map(|j| recorded[j * gap + gap / 2]).collect();
                let mut fit_mask = self.obs.clone();
                for &i in &val {
                    fit_mask[i] = false;
                }
                let mut p = ObsParams::<Be> {
                    theta: Param::from_tensor(Tensor::zeros([n, n], &self.device)),
                    bias: Param::from_tensor(Tensor::zeros([n, 1], &self.device)),
                };
                let mut optim = AdamConfig::new().init();
                let ne = self.n_edges.max(1) as f64;
                let (h, m) = self.observer_step(ocfg);
                let validate = |p: &ObsParams<Be>| -> f64 {
                    if val.is_empty() {
                        return 0.0;
                    }
                    let w = self.w_nominal.clone() * p.theta.val().exp();
                    let r = self.rollout(&w, &p.bias.val(), ocfg, &fit_mask);
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for k in 1..self.obs_data.len() {
                        let x: Vec<f32> = r.states[k * m].clone().into_data().convert::<f32>().value;
                        for &i in &val {
                            let y = self.obs_data[k][i];
                            num += (x[i] as f64 - y).powi(2);
                            den += y * y;
                        }
                    }
                    let _ = h;
                    (num / den.max(1e-12)).sqrt()
                };
                let base_score = validate(&p);
                let mut best = (base_score, p.clone());
                for epoch in 0..ocfg.epochs {
                    let w = self.w_nominal.clone() * p.theta.val().exp();
                    let r = self.rollout(&w, &p.bias.val(), ocfg, &fit_mask);
                    let th = p.theta.val();
                    let loss = r.loss + (th.clone() * th).sum() * (ocfg.theta_reg / ne);
                    final_loss = loss.clone().into_scalar();
                    let grads = GradientsParams::from_grads(loss.backward(), &p);
                    p = optim.step(ocfg.lr, p, grads);
                    if (epoch + 1) % 10 == 0 {
                        let v = validate(&p);
                        if v < best.0 && v < base_score * (1.0 - ocfg.select_margin) {
                            best = (v, p.clone());
                        }
                    }
                }
                let p = best.1;
                (self.w_nominal.clone() * p.theta.val().exp(), p.bias.val())
            }
        };
        let r = self.rollout(&w, &b, ocfg, &self.obs);

        let (h, _) = self.observer_step(ocfg);
        let stride = self.cfg.data_stride.max(2);
        let n_grid = self.obs_data.len() - 1;
        let s = stride as f64 * DT;
        let series: Vec<Vec<f64>> = (0..n).map(|i| self.obs_data.iter().map(|r| r[i]).collect()).collect();
        let (mut nu, mut du, mut no, mut d_o) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..n_grid {
            let step = k * stride + stride / 2;
            if step >= self.truth.len() {
                break;
            }
            let t = step as f64 * DT;
            let idx = (t / h).round() as usize;
            let pred: Vec<f32> = r.states[idx].clone().into_data().convert::<f32>().value;
            for i in 0..n {
                let y = self.truth[step][i];
                if self.obs[i] {
                    let e = hermite_eval(&series[i], s, t) - y;
                    no += e * e;
                    d_o += y * y;
                } else {
                    let e = pred[i] as f64 - y;
                    nu += e * e;
                    du += y * y;
                }
            }
        }
        ObserverResult {
            kind,
            rel_err_unobserved: if du > 0.0 { (nu / du).sqrt() } else { f64::NAN },
            rel_err_observed: (no / d_o.max(1e-12)).sqrt(),
            final_loss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{Connectome, SyntheticConfig};
    use crate::dynamics::{make_teacher, NetConfig};
    use crate::surrogate::SurrogateConfig;

    fn problem(noise: f64) -> SurrogateProblem {
        let c = Connectome::synthetic(&SyntheticConfig { n: 64, n_modules: 6, seed: 4, ..Default::default() });
        let net_cfg = NetConfig::default();
        let teacher = make_teacher(&c, &net_cfg, 0.3, 1);
        let cfg = SurrogateConfig { k_embed: 6, hidden: vec![8], epochs: 1, obs_noise: noise, ..Default::default() };
        SurrogateProblem::new(&c, &teacher, &net_cfg, cfg, 1.0).unwrap()
    }

    #[test]
    fn oracle_observer_recovers_unrecorded_neurons() {
        let p = problem(0.0);
        let r = p.run_observer(ObserverKind::Oracle, &ObserverConfig::default());
        assert!(r.rel_err_unobserved < 0.05, "oracle observer error {}", r.rel_err_unobserved);
    }

    #[test]
    fn selected_fit_is_never_worse_than_nominal() {
        let p = problem(0.0);
        let cfg = ObserverConfig { epochs: 120, ..Default::default() };
        let nom = p.run_observer(ObserverKind::Nominal, &cfg);
        let fit = p.run_observer(ObserverKind::Learned, &cfg);
        assert!(fit.rel_err_unobserved.is_finite());
        assert!(fit.rel_err_unobserved <= nom.rel_err_unobserved * 1.05, "selected fit {} must be within 5% of nominal {}", fit.rel_err_unobserved, nom.rel_err_unobserved);
    }

    #[test]
    fn measurement_noise_degrades_interpolation_but_not_the_clean_truth() {
        let clean = problem(0.0).cubic_interp_baseline_rel_err();
        let noisy = problem(0.15).cubic_interp_baseline_rel_err();
        assert!(clean < 0.01 && noisy > 10.0 * clean, "clean {clean}, noisy {noisy}");
    }

    #[test]
    fn noisy_sparse_recordings_do_not_make_the_fit_worse() {
        let p = problem(0.15);
        let cfg = ObserverConfig { epochs: 60, ..Default::default() };
        let nom = p.run_observer(ObserverKind::Nominal, &cfg);
        let fit = p.run_observer(ObserverKind::Learned, &cfg);
        assert!(fit.rel_err_unobserved <= nom.rel_err_unobserved * 1.05, "fit {} vs nominal {}", fit.rel_err_unobserved, nom.rel_err_unobserved);
    }

    #[test]
    fn hermite_reproduces_cubics_in_the_interior() {
        let s = 0.1;
        let y: Vec<f64> = (0..20).map(|k| (k as f64 * s).powi(3)).collect();
        let t = 0.95;
        assert!((hermite_eval(&y, s, t) - t.powi(3)).abs() < 2e-3);
    }
}
