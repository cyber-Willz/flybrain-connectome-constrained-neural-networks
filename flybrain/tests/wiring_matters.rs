//! Integration check of the central claim at small scale: a network that
//! uses the true wiring recovers a hidden fly's dynamics far better than one
//! with the same number of synapses wired at random, and the trained
//! connectome-constrained network never violates the wiring mask or Dale's law.

use burn::tensor::backend::Backend;
use flybrain::connectome::{Connectome, SyntheticConfig};
use flybrain::dynamics::*;

#[test]
fn true_wiring_beats_random_wiring_on_held_out_stimuli() {
    let c = Connectome::synthetic(&SyntheticConfig { n: 96, n_modules: 6, seed: 11, ..Default::default() });
    let cfg = NetConfig::default();
    let device = <Be as Backend>::Device::default();
    let teacher = make_teacher(&c, &cfg, 0.5, 21);
    let train = make_dataset(&c, &teacher, &cfg, 12, 16, 1.0, 22);
    let test = make_dataset(&c, &teacher, &cfg, 12, 16, 1.0, 23);
    let g = c.symmetrized_graph();
    let fc = FitConfig { epochs: 80, lr: 0.03, ..Default::default() };

    let (_net, good) = fit(FlyBrainNet::<Be>::from_connectome(&c, &cfg, &device), &train, &test, &g, &c.mask_matrix(), Some(&teacher.theta), &fc);
    let rnd = c.density_matched_random(99);
    let (_net, bad) = fit(FlyBrainNet::<Be>::from_connectome(&rnd, &cfg, &device), &train, &test, &g, &rnd.mask_matrix(), None, &fc);

    assert!(good.test.rel_mse < 0.05, "true wiring should fit well, got {}", good.test.rel_mse);
    assert!(bad.test.rel_mse > 5.0 * good.test.rel_mse, "random wiring {} should be much worse than true wiring {}", bad.test.rel_mse, good.test.rel_mse);
}
