//! Error type for this crate. Wraps [`spectral_hypergraph::HypergraphError`]
//! rather than duplicating it, plus the handful of failure modes specific to
//! building embeddings and residuals (requesting more embedding dimensions
//! than the hypergraph has non-trivial eigenpairs, mismatched tensor/vertex
//! counts, etc.).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShpinnError {
    #[error("hypergraph error: {0}")]
    Hypergraph(#[from] spectral_hypergraph::HypergraphError),

    #[error(
        "requested {requested} spectral embedding dimensions but the hypergraph only has \
         {available} non-trivial (nonzero-eigenvalue) eigenpairs"
    )]
    TooManyEmbeddingDims { requested: usize, available: usize },

    #[error("boundary vertex index {index} is out of range for a hypergraph with {n} vertices")]
    BoundaryOutOfRange { index: usize, n: usize },

    #[error("expected {expected} rows, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, ShpinnError>;
