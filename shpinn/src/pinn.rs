//! The learnable network itself: a plain MLP over spectral-embedding
//! coordinates (optionally concatenated with extra continuous inputs, e.g.
//! a time coordinate for [`crate::physics::heat_residual`]).
//!
//! Structurally this is the least novel part of a SHPINN — it is an
//! ordinary feedforward network, `tanh`-activated as classical PINNs use
//! (smooth, so its input-gradient exists and is itself smooth, which matters
//! once [`crate::physics::heat_residual`]/`wave_residual` differentiate
//! through it w.r.t. a time input). All the hypergraph-specific structure
//! lives in what feeds it ([`crate::embedding`]) and in how its output is
//! penalized ([`crate::physics`], [`crate::loss`]): the network is applied
//! identically, row-by-row, to every vertex's coordinate row, with **no**
//! cross-vertex mixing inside the network — the graph coupling enters only
//! through the physics residual's `Delta @ u` term. That separation is
//! deliberate: it keeps the "does the fixed spectral operator's action on
//! the network's own output satisfy the PDE" residual honest, the same way
//! a classical PINN never lets its network architecture itself smuggle in
//! the answer to `laplace(u) = f`.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Config, Debug)]
pub struct ShpinnConfig {
    /// Number of spectral embedding dimensions (plus any extra continuous
    /// inputs, e.g. `+1` for a time coordinate) fed to the network.
    pub input_dim: usize,
    /// Hidden layer widths, e.g. `vec![64, 64, 64]`.
    pub hidden: Vec<usize>,
    /// Output field dimension (`1` for a scalar field like temperature or
    /// pressure; `> 1` for a vector field).
    pub output_dim: usize,
}

#[derive(Module, Debug)]
pub struct Shpinn<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl ShpinnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Shpinn<B> {
        let mut dims = vec![self.input_dim];
        dims.extend(self.hidden.iter().copied());
        dims.push(self.output_dim);

        let layers = dims
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).with_bias(true).init(device))
            .collect();

        Shpinn { layers }
    }
}

impl<B: Backend> Shpinn<B> {
    /// `x` is `[n, input_dim]` (one row per hypergraph vertex, or per
    /// (vertex, time) collocation pair). Returns `[n, output_dim]`. `tanh`
    /// on every hidden layer, linear (no activation) on the output layer, so
    /// the field itself is unconstrained in sign/range.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let last = self.layers.len() - 1;
        let mut h = x;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h);
            if i != last {
                h = tanh(h);
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn forward_shape_matches_config() {
        let device = Default::default();
        let cfg = ShpinnConfig::new(4, vec![16, 16], 1);
        let net = cfg.init::<B>(&device);
        let x = Tensor::<B, 2>::zeros([10, 4], &device) + 1.0;
        let out = net.forward(x);
        assert_eq!(out.dims(), [10, 1]);
    }

    #[test]
    fn no_hidden_layers_is_a_plain_linear_map() {
        let device = Default::default();
        let cfg = ShpinnConfig::new(3, vec![], 2);
        let net = cfg.init::<B>(&device);
        let x = Tensor::<B, 2>::zeros([5, 3], &device) + 1.0;
        let out = net.forward(x);
        assert_eq!(out.dims(), [5, 2]);
    }

    #[test]
    fn vector_field_output_dim_respected() {
        let device = Default::default();
        let cfg = ShpinnConfig::new(4, vec![8], 3);
        let net = cfg.init::<B>(&device);
        let x = Tensor::<B, 2>::zeros([7, 4], &device) + 1.0;
        let out = net.forward(x);
        assert_eq!(out.dims(), [7, 3]);
    }
}
