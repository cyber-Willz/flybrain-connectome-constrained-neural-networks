use thiserror::Error;

#[derive(Debug, Error)]
pub enum FlyError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("hypergraph error: {0}")]
    Hypergraph(#[from] spectral_hypergraph::error::HypergraphError),
    #[error("shpinn error: {0}")]
    Shpinn(#[from] shpinn::error::ShpinnError),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, FlyError>;
