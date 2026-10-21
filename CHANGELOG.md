# Changelog

## v2
* New `flybrain::observer`: physics state observer for unrecorded neurons (nominal / learned / oracle W). The learned-W fit (integral-form loss, held-out checkpoint selection with 10% margin, >=10 held-out neurons) is EXPERIMENTAL: it is not reliably better than nominal W and can be worse.
* `surrogate`: noisy recordings (`obs_noise`), `hermite_eval`, cubic interpolation baseline; SHPINN data term now trains on the recorded (noisy) data.
* `fly_pipeline`: observer rows in 3a, cubic baseline in 3a/3b, new sweep (`--only=4`), `--no-pinn`; fixed nested `if want(..)` that hid the Dirichlet-energy table when `--only=2`.
* Removed redundant `[profile.release]` from `krylov_ds` and `spectral_hypergraph` (ignored in a workspace; identical to the root profile).
* Docs: §3c in `docs/flybrain_results.md`; raw logs in `results/`.
