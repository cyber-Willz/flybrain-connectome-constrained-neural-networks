# shpinn — Spectral Hypergraph Physics-Informed Neural Networks

A **Physics-Informed Neural Network** (Raissi, Perdikaris & Karniadakis,
2019) whose domain is a `spectral_hypergraph::SpectralHypergraph` instead of
Euclidean space. Built on top of your two uploaded projects
(`spectral_hypergraph` and the `nbsc_energy_stabilized_deep_stacks` bundle)
without modifying either — `shpinn` is a fourth workspace member that only
*consumes* their public APIs.

## Install

Drop the `shpinn/` directory into the workspace root (alongside `krylov_ds`,
`nbsc`, `spectral_hypergraph`) and add it to the workspace's `Cargo.toml`:

```toml
[workspace]
members = ["krylov_ds", "nbsc", "spectral_hypergraph", "shpinn"]
```

That's the only change needed to the existing workspace. `shpinn` itself
depends on the other three by path plus `burn` (pinned exactly the way
`nbsc`'s own README pins it for this workspace's rustc 1.75 floor — see
`Cargo.toml`).

```bash
# core crate + its own unit tests (embedding, operator, physics, loss, pinn,
# gradient, burgers, navier_stokes, spectral_encoding)
cargo test -p shpinn

# seven end-to-end demos, each verified against an independent ground truth
cargo run --release -p shpinn --example poisson_demo
cargo run --release -p shpinn --example heat_demo
cargo run --release -p shpinn --example helmholtz_demo
cargo run --release -p shpinn --example burgers_demo
cargo run --release -p shpinn --example burgers_spectral_demo
cargo run --release -p shpinn --example navier_stokes_demo

# the directional (non-backtracking) variant, gated behind its own feature
# since it pulls in nbsc's `burn` + `hypergraph` features
cargo run --release -p shpinn --features nbsc-ops --example advection_demo
```

All seven demos print PASS/FAIL against a ground truth computed independently
of the trained network (a direct linear solve, a closed-form spectral
solution, RK4 integration, or residual self-consistency — see below), and
exit non-zero on FAIL, so they double as regression tests, not just
qualitative demos.

## What "spectral hypergraph PINN" means here

A classical PINN's network takes Euclidean spatial coordinates `(x, y, t,
...)` as input, and gets every spatial derivative in its PDE residual via
autodiff through the network. Two things don't carry over to a hypergraph
domain unmodified:

1. **There is no ambient space to hand the network coordinates in.**
   `embedding.rs` builds an analogous continuous coordinate system out of
   the hypergraph's own spectral geometry: a **Laplacian eigenmap** — the
   `k` eigenvectors of the hypergraph's true normalized Laplacian `Delta`
   (Zhou/Huang/Schölkopf 2006) immediately following the trivial zero
   mode. Two vertices that are spectrally close (connected through many
   short, low-cardinality hyperedges) end up close together in this
   embedding — exactly the smoothness property a PINN needs its input
   coordinates to have. Both a dense (`laplacian_eigenmap`, `O(n^3)`) and a
   matrix-free (`laplacian_eigenmap_sparse`, via `krylov_ds`-style Lanczos)
   path are provided, cross-checked against each other in the test suite.

   This is built on the hypergraph's **own** Laplacian, not a
   clique-expansion of it (the approach `nbsc::hypergraph_bridge` uses to
   reuse the plain-graph NBSC/GCN pipeline) — a SHPINN wants its
   coordinates and its physics operator (next point) to come from the same
   object, so the network's smoothness prior and the residual it's judged
   against are geometrically consistent.

2. **There is no spatial derivative to take.** The hypergraph domain is
   discrete, so "the spatial differential operator" is not a derivative at
   all — it's `Delta` itself, applied once via ordinary matrix
   multiplication (`operator.rs`). Because `Delta` already *is* the
   correct discretization (Zhou/Huang/Schölkopf's construction reduces to
   the standard graph-Laplacian discretization of `-Δ_continuous` when the
   hypergraph happens to be an ordinary graph), a SHPINN never autodiffs
   w.r.t. space. `physics.rs` uses this to build the standard
   elliptic/parabolic/hyperbolic PINN trio directly on the hypergraph:

   - `poisson_residual`: `Delta u - f` (steady-state).
   - `heat_residual`: `du/dt + alpha * Delta u` (diffusion; `Delta` is PSD,
     so `+alpha*Delta` is the dissipative sign).
   - `wave_residual`: `d^2u/dt^2 + c^2 * Delta u`.

   Time, where a governing equation has it, is still continuous — but
   getting `du/dt` via a *second* autodiff pass through an
   already-captured gradient isn't supported by Burn 0.13's single-order
   autodiff. Rather than fight the framework, `central_time_derivative`
   uses a central finite difference on the network's own output at `t ±
   eps` — a standard **semi-discrete ("method of lines") scheme**: exact
   in space, finite-difference in time. The whole training loop still only
   ever calls `.backward()` once per step; nothing here needs double
   backprop.

3. **The network itself is architecturally boring, on purpose.**
   `pinn::Shpinn` is a plain `tanh`-MLP applied row-wise, with *no*
   cross-vertex mixing inside the network. All hypergraph structure enters
   through the coordinates it's fed and the operator its output is
   penalized against — never through the network's own wiring. This
   matters for correctness, not just style: it's what makes
   `central_time_derivative`'s finite-difference trick and (in an earlier,
   discarded draft) the "sum trick" for batched derivatives valid in the
   first place, and it keeps the residual an honest check rather than
   something the architecture could quietly satisfy on its own.

4. **Loss** (`loss.rs`) combines a masked boundary/data term with the
   physics-residual term — the same split a mesh-based PINN makes between
   its boundary-condition loss and its interior PDE-residual loss.
   `boundary_mask` builds the `{0,1}` mask; the Poisson demo below is the
   one place this distinction is *load-bearing*, not cosmetic (see the
   "what I got wrong first" note).

## First-order / nonlinear physics via the gradient/divergence operator pair

`Delta`'s middle term factors exactly as `G^T G` for a `[hyperedges x
vertices]` operator `G` (`gradient.rs`; the identity is checked directly
against `operator::dense_laplacian_tensor` in that module's own tests, not
just asserted). `G` is a vertex→hyperedge aggregation and `G^T` its
hyperedge→vertex adjoint — first-order structure `Delta` alone can't give
you, used for:

- `physics::helmholtz_residual` — `Delta u - k^2 u - f`, the
  frequency-domain reduction of the wave equation (indefinite, not PSD like
  Poisson — pick `k^2` away from `Delta`'s spectrum).
- `burgers.rs` — viscous Burgers' equation, `du/dt + u du/dx - nu
  d^2u/dx^2 = 0`, via the conservative-flux discretization `G^T(G(u^2/2))`
  for the nonlinear advective term.
- `navier_stokes.rs` — incompressible Navier-Stokes reduced to
  **network-flow form**: velocity lives on hyperedges, pressure on
  vertices, coupled through `G`/`G^T` — the same reduction real
  pipe/resistor/vascular network flow models use, since a hypergraph has no
  ambient space for a literal velocity vector to live in.

**A caveat worth knowing before building on `G`/`G^T`:** they're built from
an *unsigned* incidence structure (the same one `Delta` itself is built
from), not a literal signed vector-calculus gradient/divergence. Two
consequences, both discovered empirically while building `navier_stokes_demo`
and worth knowing in advance:

1. A network with more hyperedges than vertices has a genuine
   multi-dimensional "circulation" null space in edge-indexed fields —
   flows that satisfy continuity everywhere *and* experience zero viscous
   dissipation, physically the same ambiguity a persistent current in a
   superconducting loop has. An external force pointed along one of those
   directions has no steady solution (see
   `navier_stokes::momentum_residual`'s demo for the failed first attempt
   this ran into, and the fix: drive flow through boundary *pressure*
   conditions instead of an internal body force).
2. `G` doesn't carry the classical "total inflow = total outflow"
   conservation property a signed incidence matrix would — so don't assume
   net flux through a cut is a boundary-condition-independent invariant the
   way it would be in a real resistor network.

Neither is a bug; both are real properties of building these operators on a
hypergraph with no inherent orientation, and are worth designing around
rather than assuming away.

## Optional: directional physics via the non-backtracking operator

`Delta` is symmetric, so by construction it's blind to oriented-cycle /
directional structure — exactly the gap `nbsc`'s whole premise (the
Hashimoto / non-backtracking matrix `B`) exists to fill. `advective.rs`
(feature `nbsc-ops`, since it pulls in `nbsc` with its own `burn` +
`hypergraph` features) swaps the spatial operator from `Delta` to `nbsc`'s
rescaled non-backtracking recursion `T_1 = A / rho_B` (literally
`NbscLayer`'s first filter tap, pulled out standalone) for governing
equations where directionality is physically real rather than isotropic —
transport/advection along a preferred orientation, not diffusion. This is a
genuinely different operator for a genuinely different equation, not a
different discretization of the same one: `T_1` isn't symmetric in general,
so it isn't a drop-in replacement inside the Poisson/heat/wave residuals
(whose derivation assumes symmetric PSD).

## Optional: random Fourier input features against spectral bias

`spectral_encoding.rs`'s `FourierFeatureEncoder` re-expresses a `Shpinn`'s
input coordinates as `[raw | sin(2*pi*B.x) | cos(2*pi*B.x)]` for a fixed
(non-trainable) random frequency matrix `B`, before the network sees them —
ported by hand from a separate, from-scratch continuous-domain PINN system
whose own Burgers run hit **spectral bias** (plain tanh-MLPs' documented
bias toward low-frequency functions) hard enough that this fixed it. It
only touches the network's input side, so it drops in ahead of an
unmodified `Shpinn` (sized for the encoded dimension) without touching any
residual code.

`examples/burgers_spectral_demo.rs` measures it head-to-head against a
plain `Shpinn` on this crate's own Burgers problem, against the same
independent RK4 ground truth `burgers_demo` uses. **The honest result: it
didn't help here** — the plain network already fits this small (12-vertex)
hypergraph's trajectory to near machine precision, so there was no
spectral-bias bottleneck for the encoder to fix, and it came out several
times worse in relative terms (both still tiny in absolute terms; see that
file's module doc for the numbers). Kept in the crate, tested
(`spectral_encoding.rs`'s own unit tests check shape, the raw-coordinate
skip connection, the sin/cos values against a hand computation, and that
gradients flow through it), and not wired into any other demo by default —
the source project's own README already reported a case where this
technique looked like a clear win by training loss alone but was actually
overfit/aliased once checked independently, and this crate's own measurement
turned out to be a second instance of "don't assume, measure" rather than a
second instance of "it works."

## Verification, not just "loss went down"

Each `examples/*.rs` demo checks the trained network against a ground truth
computed by an entirely independent method:

| Demo | Equation | Ground truth method | Result |
|---|---|---|---|
| `poisson_demo` | `Delta u = 0`, Dirichlet BC at 3 of 24 vertices | Direct linear solve (boundary rows → identity, `nalgebra` LU) | RMSE 0.045 over all 24 vertices (PASS, tol 0.05) |
| `heat_demo` | `du/dt = -alpha * Delta u`, hot-spot initial condition | Closed-form spectral solution `u(t) = Σ_k exp(-alpha*t*lambda_k)(v_k·u0)v_k` via `dense_eigen`, at 3 held-out times never directly fitted | RMSE ≤ 0.0035 (PASS, tol 0.08) |
| `advection_demo` (`nbsc-ops`) | `du/dt = -T_1(u)` (non-backtracking transport) | RK4 integration in plain `nalgebra`, independent of the Burn training path | RMSE ≤ 0.0003 (PASS, tol 0.08) |
| `helmholtz_demo` | `Delta u - k^2 u = 0`, Dirichlet BC at 3 of 24 vertices, `k^2` in the widest spectral gap | Direct linear solve, same technique as `poisson_demo` | RMSE ≤ 0.007 (PASS, tol 0.05) |
| `burgers_demo` | `du/dt + u du/dx - nu d^2u/dx^2 = 0`, hot-spot initial condition | RK4 integration of the exact discrete ODE the residual enforces, in plain `nalgebra` | RMSE ≤ 0.0019 at every held-out time (PASS, tol 0.08) |
| `burgers_spectral_demo` | Same as `burgers_demo`, plain vs. Fourier-encoded `Shpinn` head-to-head | Same RK4 ground truth as `burgers_demo` | plain worst RMSE 0.00005, spectral worst RMSE 0.00020 — spectral encoding measured *worse* here (PASS, both within tol 0.08; see the README section above) |
| `navier_stokes_demo` | Network-flow Navier-Stokes, two vertices held at fixed pressure | Residual self-consistency (momentum + continuity RMS) and boundary fit, plus pressure-field comparison to a direct linear (Stokes-regime) solve — see the README section above for why *not* a direct `q` comparison | momentum/continuity RMS ≤ 0.0005, boundary RMSE ≤ 0.0003, pressure-field RMSE 0.045 (PASS) |

**What I got wrong on the first pass, and why it matters:** the first
`poisson_demo` run enforced `Delta u = 0` at *every* vertex, including the
three boundary vertices — but a Dirichlet boundary vertex's true Laplacian
is generally *nonzero* (it's a prescribed value, not a free unknown solving
the PDE there), so the physics loss was fighting the boundary condition at
exactly the vertices being pinned. That produced RMSE 0.256 — training loss
went down smoothly the whole time, so nothing about the loss curve would
have flagged it; only checking against the independent linear solve caught
it. Masking the residual to interior vertices only (`interior_mask = 1 -
boundary_mask`) fixed it to RMSE 0.045. Left in the demo/README rather than
silently fixed, since it's the kind of mistake this whole verification
approach exists to catch, and future work should mask residuals by boundary
condition type this same way for Neumann/Robin conditions too.

## Layout

```
shpinn/
  src/
    embedding.rs         Laplacian-eigenmap spectral coordinates (dense + matrix-free)
    operator.rs           Delta as a dense, autodiff-tracked Burn tensor
    gradient.rs            G / G^T vertex<->hyperedge operator pair (Delta = I - G^T G)
    pinn.rs                the Shpinn network (plain row-wise tanh-MLP)
    physics.rs             poisson/heat/wave/helmholtz residuals + central_time_derivative
    burgers.rs             viscous Burgers' equation via the G/G^T conservative flux
    navier_stokes.rs       incompressible network-flow Navier-Stokes via G/G^T
    spectral_encoding.rs   random Fourier input features against spectral bias (measured: no help on Burgers here — see README)
    loss.rs                boundary_mask + shpinn_loss
    advective.rs           (feature `nbsc-ops`) non-backtracking spatial operator + advective_residual
    error.rs               ShpinnError
  examples/
    poisson_demo.rs           steady-state, verified against a direct linear solve
    heat_demo.rs               diffusion, verified against a closed-form spectral solution
    helmholtz_demo.rs           indefinite steady-state, verified against a direct linear solve
    burgers_demo.rs             nonlinear advection-diffusion, verified against RK4
    burgers_spectral_demo.rs    burgers_demo, plain vs. Fourier-encoded, head-to-head
    navier_stokes_demo.rs       network-flow NS, verified against residual self-consistency + a direct solve
    advection_demo.rs          (feature `nbsc-ops`) transport, verified against RK4
  tests/ (inline `#[cfg(test)]` modules per source file — 34 tests total,
          32 without `nbsc-ops`, all passing)
```

## Toolchain

Same pins as this workspace's `nbsc` crate: rustc 1.75, Burn 0.13.2 with its
usual transitive-dependency pins (`half`, `uuid`, `bincode`, `rmp-serde`,
`rmp`, `indexmap`, `rayon`/`rayon-core`). Built and tested end-to-end in a
fresh sandbox with exactly `rustc 1.75.0` / `cargo 1.75.0` installed via
`apt`, workspace `cargo test -p shpinn` (15/15 passing) and `--features
nbsc-ops` (17/17 passing), plus release builds of all three examples, all
producing the PASS results in the table above.
