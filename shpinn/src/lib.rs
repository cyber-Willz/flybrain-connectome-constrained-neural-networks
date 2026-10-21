//! # shpinn — Spectral Hypergraph Physics-Informed Neural Networks
//!
//! A **Physics-Informed Neural Network** (Raissi, Perdikaris & Karniadakis
//! 2019) whose domain is a [`spectral_hypergraph::hypergraph::SpectralHypergraph`]
//! rather than Euclidean space:
//!
//! * **Coordinates.** A classical PINN's network takes `(x, y, t, ...)` as
//!   input. A hypergraph has no ambient coordinates, so [`embedding`]
//!   builds an analogous continuous coordinate system out of the
//!   hypergraph's own spectral geometry — a Laplacian eigenmap of `hg`'s
//!   **true** normalized hypergraph Laplacian (not a clique-expansion
//!   approximation of it).
//! * **Network.** [`pinn::Shpinn`] is an ordinary `tanh`-MLP applied
//!   row-wise to those coordinates (plus any extra continuous inputs, e.g.
//!   time). No graph structure is baked into the network itself.
//! * **Physics.** [`operator`] materializes the hypergraph's normalized
//!   Laplacian `Delta` as a dense Burn tensor, so it sits inside Burn's
//!   autodiff graph. [`physics`] uses it as the discrete spatial operator
//!   in the residuals for the standard elliptic/parabolic/hyperbolic PINN
//!   trio (Poisson / heat / wave), replacing the spatial-derivative
//!   autodiff a Euclidean PINN would use with one matrix multiply — `Delta`
//!   already *is* the correct discretization, so there is nothing to
//!   differentiate. Any remaining continuous input (time) uses a central
//!   finite difference on the network's own output
//!   ([`physics::central_time_derivative`]) rather than autodiff — see that
//!   function's docs for why.
//! * **Loss.** [`loss::shpinn_loss`] combines a masked data/boundary term
//!   with the physics residual term, the same split a mesh-based PINN makes
//!   between boundary-condition loss and interior PDE loss.
//! * **Directional physics (optional, feature `nbsc-ops`).** [`advective`]
//!   swaps the spatial operator from the symmetric `Delta` to
//!   [`nbsc`]'s non-backtracking (Hashimoto) recursion, for governing
//!   equations where oriented-cycle / directional structure is physically
//!   real and a symmetric Laplacian would wash it out.
//! * **First-order / nonlinear physics.** [`gradient`] factors `Delta`'s
//!   middle term as `G^T G` and exposes `G` (vertex -> hyperedge) and `G^T`
//!   (hyperedge -> vertex) as their own operators, for equations that need
//!   first-order structure rather than only `Delta`:
//!   - [`physics::helmholtz_residual`] -- `Delta u - k^2 u - f`, the
//!     frequency-domain reduction of the wave equation.
//!   - [`burgers`] -- viscous Burgers' equation, `du/dt + u du/dx - nu
//!     d^2u/dx^2 = 0`, via the conservative-form nonlinear flux
//!     `d/dx(u^2/2)` built from `G`/`G^T`.
//!   - [`navier_stokes`] -- incompressible Navier-Stokes in network-flow
//!     form: velocity on hyperedges, pressure on vertices, coupled through
//!     `G`/`G^T`, with a quadratic inertial self-term in place of `(u .
//!     grad)u`.
//! * **Optional input encoding.** [`spectral_encoding::FourierFeatureEncoder`]
//!   re-expresses a `Shpinn`'s input coordinates through fixed random
//!   sinusoids ahead of the network, targeting the spectral bias of plain
//!   tanh-MLPs -- ported from a separate PINN system where it fixed a real
//!   problem. Measured (not just wired in on faith) against this crate's
//!   own Burgers demo in `examples/burgers_spectral_demo.rs`; see that
//!   file and the README for the honest result.
//!
//! See `examples/poisson_demo.rs` and `examples/heat_demo.rs` for complete,
//! verified end-to-end walkthroughs (each checks the trained SHPINN against
//! an independent ground-truth solve, not just that training loss went
//! down), and `examples/advection_demo.rs` (requires `--features
//! nbsc-ops`) for the directional variant.

#[cfg(feature = "nbsc-ops")]
pub mod advective;
pub mod burgers;
pub mod embedding;
pub mod error;
pub mod gradient;
pub mod loss;
pub mod navier_stokes;
pub mod operator;
pub mod physics;
pub mod pinn;
pub mod spectral_encoding;

pub use error::{Result, ShpinnError};
