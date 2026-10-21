# flybrain — experiment results

Everything here was produced by `cargo run --release -p flybrain --example fly_pipeline`
(full run, 3 seeds, ~55 min on 2 cores; raw log in `results/fly_pipeline_full.log`) and
`connectome_report` (`results/connectome_report.txt`).

**Scope caveat that applies to every number below.** The connectome is the
*synthetic fly-like* generator (512 neurons, 9,380 connected pairs, 40k synapses,
8 neuropil-like modules), not FlyWire/hemibrain data, which this sandbox cannot
download. The "hidden fly" the networks try to recover is the same wiring and the
same Dale's-law signs with log-normally perturbed magnitudes (σ = 0.5), so these
experiments measure the **best case in which the connectome's topology is exactly
right**. Real connectomes omit gap junctions, neuromodulation and true synaptic
weights; expect smaller gains there. Hyper-parameters (lr, epochs, λ) were set once
from smoke runs and not tuned per condition; two teachers' worth of variation is
what the ± columns show (3 seeds, population sd).

## 0. Structure and spectra vs. null models

| | connected pairs | reciprocity | clustering | max out / in | ρ(W) signed | ρ(\|W\|) | ρ_B | ‖A‖/ρ_B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| connectome | 9380 | 0.416 | 0.128 | 83 / 110 | 42.4 | 106.6 | 39.3 | 1.034 |
| degree-preserving null | 9380 | 0.059 | 0.137 | 83 / 110 | 35.3 | 102.6 | 48.4 | 1.027 |
| density-matched random | 9380 | 0.034 | 0.079 | 33 / 35 | 29.8 | 77.9 | 37.8 | 1.028 |

* Neuropil recovery from the hypergraph-Laplacian eigenmap (k-means purity; chance ≈ 0.12–0.2):
  connectome **0.545**, degree-preserving null 0.141, density-matched random 0.195. The
  spectral coordinates the SHPINN uses carry real module information that rewiring destroys.
* ‖A‖/ρ_B ≈ 1.03 > 1 on every graph here: `A/ρ_B` is (mildly) expansive on this wiring, the same
  regime the energy-stabilized-deep-stacks report documents for citation graphs. It does not
  distinguish the connectome from its nulls.
* The signed (Dale's-law) radius ρ(W) is well below the Perron radius ρ(|W|) (42 vs 107): excitation/inhibition
  cancellation matters for stability, and symmetrized-graph spectra cannot see it.

## 1. Teacher–student, all neurons recorded

16 training stimuli, 20 Euler steps, 300 Adam epochs; error = Σ(pred−target)² / Σ target² on 16 held-out stimuli.

| student | learned params | train | held-out stimuli | θ recovery (r) |
|---|---:|---:|---:|---:|
| connectome-constrained | 9,380 | 0.0008 ± 0.0000 | **0.0016 ± 0.0000** | 0.369 ± 0.011 |
| connectome-constrained + Dirichlet-energy regulariser (λ = 0.003) | 9,380 | 0.0022 ± 0.0002 | 0.0030 ± 0.0002 | 0.360 ± 0.012 |
| random wiring, same density | 9,380 | 0.0564 ± 0.0009 | 0.0974 ± 0.0044 | — |
| dense unconstrained | 261,632 | 0.0003 ± 0.0000 | 0.3986 ± 0.0100 | — |

* Correct wiring generalises ~60× better than equally sparse random wiring and ~250× better than the
  unconstrained dense network, which fits the training stimuli best (0.0003) and generalises worst (it is
  data-starved at 16 stimuli; more data would narrow that gap — the comparison is "at this data budget").
* **Function is recovered far better than individual weights**: θ recovery r ≈ 0.37. Many weight
  combinations produce nearly the same activity (hence the 0.0016 error with a weak per-synapse
  correlation). Do not read the learned θ as measured synaptic strengths.
* **The energy regulariser did not help here**: at λ = 0.003 held-out error roughly doubled (0.0016 → 0.0030).
  This matches the earlier report's finding that its value is context-dependent; I did not sweep λ.

### Dirichlet energy per unrolled step (the NBSC deep-stack diagnostic)

| source | step 1 | step 5 | step 10 | step 15 | step 20 |
|---|---:|---:|---:|---:|---:|
| teacher (target) | 1.071 | 1.064 | 1.108 | 1.152 | 1.182 |
| connectome-constrained | 1.074 | 1.067 | 1.107 | 1.146 | 1.172 |
| + energy regulariser | 1.029 | 1.021 | 1.056 | 1.093 | 1.117 |
| random wiring | 1.125 | 1.106 | 1.118 | 1.137 | 1.153 |
| dense unconstrained | 1.008 | 1.044 | 1.179 | 1.292 | 1.363 |

Energy rises mildly with depth for the teacher (expansive regime, no over-smoothing collapse). The
constrained net tracks the teacher's profile to within ~1%; the dense net's energy drifts up to 1.36 —
the profile flags the unconstrained model as dynamically wrong even before looking at accuracy. (The
regulariser pulls energy *below* the teacher's, which is why it costs accuracy.)

## 2. Teacher–student, 30% of neurons recorded

| student | held-out, all neurons | held-out, never-recorded neurons | θ recovery (r) |
|---|---:|---:|---:|
| connectome-constrained | 0.0969 ± 0.0142 | **0.1367 ± 0.0110** | 0.247 ± 0.008 |
| random wiring, same density | 1.1159 ± 0.0293 | 1.5434 ± 0.0574 | — |
| dense unconstrained | 0.4656 ± 0.0047 | 0.4850 ± 0.0281 | — |

With only 30% of neurons recorded, the wired network still predicts the unseen 70% to ~14% relative MSE.
Random wiring is worse than predicting zero (> 1). This is the same best-case caveat as §1: the unseen
neurons are predictable because the wiring is known to be correct.

## 3. SHPINN surrogate (n = 256, own connectome, 1,500 epochs, 3 problems)

Relative **L2** error (not MSE) at held-out times. Physics residual = the connectome rate equation;
"nominal W" uses the count-derived weights (wrong magnitudes, zero bias), "true W" is an oracle.

### 3a. Imputation: 30% recorded, a sample every 0.1

| variant | unrecorded neurons | recorded neurons |
|---|---:|---:|
| data only | 1.349 ± 0.217 | 0.092 ± 0.010 |
| physics, nominal W | **0.834 ± 0.060** | 0.093 ± 0.012 |
| physics, true W (oracle) | 0.815 ± 0.030 | 0.124 ± 0.061 |
| linear interpolation (no network) | — | 0.0012 ± 0.0001 |
| predict zero | 1.000 | 1.000 |

The physics residual is what makes imputation possible at all: data-only is worse than predicting zero
(1.35), physics brings it to 0.83 (≈ 30% of the variance of unseen neurons explained). That is **modest, not
good imputation**; the oracle W is barely better (0.815), so the limit is the under-determined inverse
problem and the surrogate's capacity, not the W mismatch. For scale, §2's connectome-constrained *recurrent network* reaches unrecorded relative MSE 0.137 (L2 ≈ 0.37) — a different task setup (discrete-time, step stimuli), so not a like-for-like comparison, but it suggests fitting the wired network beats imputing with the surrogate.

### 3b. Temporal super-resolution: all recorded, a sample every 0.5

| variant | recorded neurons |
|---|---:|
| data only | 0.207 ± 0.033 |
| physics, nominal W | 0.261 ± 0.104 |
| physics, true W (oracle) | 0.216 ± 0.035 |
| linear interpolation (no network) | **0.028 ± 0.001** |

**Negative result.** On these smooth rate dynamics, plain linear interpolation between samples is ~7–9×
more accurate than any surrogate variant, and the physics term does not help (nominal W is worse and
high-variance). The surrogate should not be used for interpolation of smooth rate trajectories.

## 3c. Physics observer (added in v2; n = 256, raw logs in `results/`)

The SHPINN in §3 must memorise every neuron's trajectory in an MLP. `flybrain::observer` instead uses the
connectome's rate equation directly: clamp the recorded neurons to their (cubic-interpolated) samples,
integrate `du/dt = -u + tanh(Wu + b + I(t))` for the unrecorded ones from `u(0) = 0`.

3a, 30% recorded, a sample every 0.1 (relative L2 error at held-out times; SHPINN rows from
`results/exp3_run.log`, 2 problems; observer rows from a 4-problem run):

| variant | unrecorded neurons | recorded neurons |
|---|---:|---:|
| SHPINN, data only | 1.403 ± 0.244 | 0.126 |
| SHPINN, physics, nominal W | 0.895 ± 0.067 | 0.153 |
| SHPINN, physics, true W (oracle) | 0.894 ± 0.053 | 0.122 |
| **observer, nominal W** | **0.448 ± 0.046** | 0.0001 |
| observer, learned W (experimental) | 0.479 ± 0.097 | 0.0001 |
| observer, true W (oracle) | 0.0003 | 0.0001 |
| cubic interpolation (no network) | — | 0.0001 |
| linear interpolation (no network) | — | 0.0012 |

* With nothing fitted, the observer halves the SHPINN's unrecorded-neuron error (0.89 to about 0.45) in seconds.
  With the true `W` it is essentially exact; the SHPINN with true `W` stays at 0.89, so the SHPINN's limit was
  capacity/optimisation, not information.
* Cubic interpolation matches recorded neurons (0.0001) far better than linear (0.0012) or any SHPINN (0.12–0.15).
* **Learned W is experimental and should not be used.** It fits `θ` and `b` from the recorded neurons' own
  equations (integral-form loss, backprop through the unrecorded rollout) and picks a checkpoint on held-out
  recorded neurons, with a 10% acceptance margin and nominal as the fallback. That check does not reliably
  protect it: over 4 problems it was accepted once and made things worse (0.52 to 0.64), and an earlier
  version without the margin was worse than nominal in 4 of 6 noisy sweep rows
  (`results/exp3c_v1_derivative_loss_partial.log`). Cause: weights into unrecorded neurons are constrained only
  through the unrecorded states, which can absorb model mismatch instead of reflecting truth, so one partially
  recorded trial does not identify them. Multiple stimuli (as in §2) would.

3b (all recorded, a sample every 0.5): cubic interpolation 0.0114 vs linear 0.0285 vs best SHPINN 0.19.
Interpolation, not the surrogate, is the right tool for smooth rate trajectories.

Sweep (`--only=4`, n = 96, 6 problems; 30% recorded; noise = Gaussian sd on the recorded samples, truth clean):

| sample every | noise sd | observer nominal | observer learned | observer oracle | cubic interp (recorded) | linear interp (recorded) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.00 | 0.422 | 0.422 | 0.000 | 0.000 | 0.001 |
| 0.10 | 0.05 | 0.421 | 0.421 | 0.026 | 0.206 | 0.182 |
| 0.10 | 0.15 | 0.422 | 0.422 | 0.077 | 0.618 | 0.545 |
| 0.50 | 0.00 | 0.422 | 0.422 | 0.001 | 0.012 | 0.027 |
| 0.50 | 0.05 | 0.416 | 0.416 | 0.054 | 0.199 | 0.178 |
| 0.50 | 0.15 | 0.421 | 0.421 | 0.152 | 0.595 | 0.531 |

The observer's unrecorded-neuron error with nominal `W` is insensitive to noise (the rate equation low-passes
it), whereas interpolating the recorded neurons degrades to the noise floor. At n = 96 the learned fit is
never accepted (too few recorded neurons for a trustworthy held-out check), so it equals nominal. With the true
`W` the observer would also denoise; a smoother needs a better weight estimate than this work found.

Caveat for all of §3: synthetic connectome, one stimulus per problem, few problems, so the sd values are rough.

## Where this does and does not work

* Works: connectome-constrained recurrent networks (§1–2) — fixing wiring and signs is a strong, sign-safe
  prior, and the NBSC energy profile is a cheap model-misspecification check.
* Works partially: SHPINN physics residual for imputing unrecorded neurons (§3a); the physics observer (§3c) is
  clearly better and much cheaper. Fitting synaptic weights from a single partially recorded trial is not reliable (§3c).
* Does not work / not shown: energy regularisation as an accuracy aid; SHPINN for temporal interpolation;
  recovery of individual synaptic weights; anything on real Drosophila data (not run).
