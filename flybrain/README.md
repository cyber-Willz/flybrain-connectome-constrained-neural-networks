# flybrain — connectome-constrained neural networks on fruit-fly wiring diagrams

A new crate in the `integrated_shpinn` workspace (the `pde_shpinn` archive, which
already contains the `nbsc` energy-stabilized-deep-stacks code as a subset). It
reuses `nbsc`, `shpinn`, `spectral_hypergraph` and `krylov_ds` unmodified and
adds what a *connectome* needs that those crates do not have:

| Need | Why the existing crates don't cover it | What `flybrain` adds |
|---|---|---|
| Directed, weighted, **signed** wiring | `nbsc::graph::Graph` is undirected, simple, unweighted; the `shpinn` Laplacian is symmetric | `connectome::Connectome`: directed pairs, synapse counts, Dale's-law sign per neuron, cell class, neuropil module |
| Stability of a *recurrent* net | NBSC's `rho_B` / `‖A‖/rho_B` describe the symmetrized graph only | `analysis`: Arnoldi (`krylov_ds`) spectral radius of the signed `W` and Perron radius of `|W|`, next to NBSC's `rho_B` and expansivity |
| A network whose wiring *is* the data | `nbsc` layers learn dense feature projections over a fixed graph | `dynamics::FlyBrainNet`: wiring mask and signs fixed by the connectome, magnitudes learnable (`exp(θ)` per synapse, sign-preserving), initial scale set by `gain / rho(|W|)` |
| Depth diagnostics for an unrolled recurrent stack | energy tools exist per layer | per-step Dirichlet energy of the activity matrix (nbsc's diagnostic) + optional differentiable energy regulariser in the loss |
| Higher-order structure | `shpinn` takes a hypergraph, but nothing builds one from a connectome | `Connectome::arbor_hypergraph`: one hyperedge per neuron = {neuron} ∪ its targets; its *true* hypergraph Laplacian gives the spectral coordinates |
| Honest null models | – | degree-preserving rewiring and density-matched random wiring, so "what does the real wiring buy you" is a measured number |
| Continuous-time surrogate | `shpinn` PINNs are PDE-on-hypergraph | `surrogate`: SHPINN whose physics residual is the connectome's own rate equation |
| Imputing unrecorded neurons | the SHPINN has to memorise every trajectory (error ~0.9) | `observer`: clamp recorded neurons, integrate the rate equation for the rest (error ~0.48 with nominal W, seconds) |

## Data: synthetic by default, real via CSV

**everything reported below uses the synthetic fly-like generator** (`Connectome::synthetic`): neuropil modules with chain-like feed-forward bias, sensory → interneuron → output classes,heavy-tailed degree propensities, log-normal synapse counts, ~30% inhibitory interneurons, ~40% reciprocal pairs. It is *shaped like* a fly connectome; it is not one, and none of the numbers below are claims about real *Drosophila*.


## Layout

```
flybrain/src/connectome.rs   data model, synthetic generator, CSV loader, arbor hypergraph, null models
flybrain/src/analysis.rs     wiring stats, signed/Perron spectral radius, NBSC rho_B, module recovery (k-means on hypergraph eigenmap)
flybrain/src/dynamics.rs     FlyBrainNet (Burn module), CPU ground-truth simulator, teacher/student fitting, energy diagnostics
flybrain/src/surrogate.rs    SHPINN continuous-time surrogate (RK4 ground truth, 3 physics modes, noisy recordings, cubic/linear interpolation baselines)
flybrain/src/observer.rs     physics state observer: clamp recorded neurons, integrate the rate equation for the rest, optional experimental weight fit (not recommended)
flybrain/examples/connectome_report.rs   structure + spectra vs. null models
flybrain/examples/fly_pipeline.rs        the three experiments reported in docs/flybrain_results.md
flybrain/tests/wiring_matters.rs         end-to-end check of the central claim at small scale
```

```bash
cargo test --release -p flybrain                                   # 21 tests
cargo run --release -p flybrain --example connectome_report        # ~seconds
cargo run --release -p flybrain --example fly_pipeline -- --quick  # ~2 min smoke run
cargo run --release -p flybrain --example fly_pipeline             # full run, ~1 h on 2 cores
cargo run --release -p flybrain --example fly_pipeline -- --only=3 --no-pinn --seeds=2   # observer vs interpolation, ~5 min
cargo run --release -p flybrain --example fly_pipeline -- --only=4 --seeds=2             # noise x sampling sweep, ~15 min
```

## Model

For neurons `i, j`, `W_ij = mask_ij · base_ij · exp(θ_ij)`, with
`base = sign(pre) · count · gain / ρ(|counts|)`. Dynamics (Euler, `a = dt/τ`):
`r ← (1−a) r + a · tanh(W r + b + drive)`, from `r = 0`, drive on sensory neurons
only. Unrolling `T` steps is a `T`-layer weight-tied deep stack over the
connectome; `nbsc::burn_layer::dirichlet_energy` is evaluated on `r_t` after every
step (rows = neurons, graph = symmetrized wiring).

## What was verified (and how)

* The Burn unrolled network is checked against an independent f64 CPU simulator
  (`burn_unroll_matches_cpu_simulator`, max error < 1e-4).
* Wiring mask and Dale's law hold for every synapse before **and after** training.
* `ρ(|W|)` at initialisation equals `gain` (Arnoldi, `perron_radius_of_init_weights_equals_gain`).
* Null models preserve what they claim to (degrees; edge count and count multiset).
* Synthetic generation is deterministic per seed (a `HashSet`-order bug that broke
  this was found and fixed during development).
* The ground truth for the surrogate is RK4 against an analytic solution test.

See `docs/flybrain_results.md` for the experiments, their numbers and — just as
important — where the method does **not** work.
