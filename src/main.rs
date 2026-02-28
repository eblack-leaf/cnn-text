mod data;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray};
use burn::optim::AdamConfig;
use training::{TrainingConfig, train};

// ── Backend ───────────────────────────────────────────────────────────────────
type Backend         = NdArray;
type AutodiffBackend = Autodiff<Backend>;
// GPU: swap to Wgpu + add `wgpu` feature in Cargo.toml

fn main() {
    // Dataset: headerless CSV, format `label,text`.
    // Labels are arbitrary strings — 2 labels → binary, N labels → N-class.
    let data_path    = "data/dataset.csv";
    let artifact_dir = "artifacts";

    // Hyperparameters only — vocab_size and num_classes are resolved from data.
    let config = TrainingConfig::new(AdamConfig::new())
        .with_num_epochs(10)
        .with_batch_size(32)
        .with_max_seq_len(128)
        .with_vocab_size(8192);   // BPE target; actual size depends on corpus

    train::<AutodiffBackend>(data_path, &config, artifact_dir, Default::default());
}
