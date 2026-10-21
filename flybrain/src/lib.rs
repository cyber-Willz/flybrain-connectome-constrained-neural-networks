//! # flybrain — connectome-constrained neural networks on fruit-fly wiring diagrams
//!
//! Built on top of the `nbsc` (non-backtracking spectral convolution,
//! energy-stabilized deep stacks) and `shpinn` (spectral hypergraph PINN)
//! crates of this workspace.
//!
//! * [`connectome`] — directed / weighted / Dale-signed connectome model,
//!   a fly-like synthetic generator, a FlyWire-style CSV loader, the
//!   axonal-arbor hypergraph, and null models.
//! * [`analysis`] — wiring statistics, signed-weight spectral radius,
//!   NBSC `rho_B` / expansivity, hypergraph-eigenmap module recovery.
//! * [`dynamics`] — the connectome-constrained recurrent network
//!   ([`dynamics::FlyBrainNet`]): fixed wiring and signs, learnable
//!   magnitudes, spectral-radius rescaling, Dirichlet-energy monitoring and
//!   regularisation, teacher/student fitting.
//! * [`surrogate`] — a SHPINN continuous-time surrogate whose physics
//!   residual is the connectome's own rate equation, used to infer the
//!   activity of unrecorded neurons.
//! * [`observer`] — a physics state observer: clamp the recorded neurons,
//!   integrate the connectome's rate equation for the rest, optionally fit
//!   synaptic magnitudes from the recorded neurons' own dynamics.

pub mod analysis;
pub mod connectome;
pub mod dynamics;
pub mod error;
pub mod observer;
pub mod surrogate;

pub use error::{FlyError, Result};
