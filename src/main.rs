mod data;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray};
use burn::optim::AdamConfig;
use model::TextCnnConfig;
use training::{TrainingConfig, train};

// ── Backend selection ─────────────────────────────────────────────────────────
//
// CPU (default, zero setup):
type Backend         = NdArray;
type AutodiffBackend = Autodiff<Backend>;
//
// GPU — swap the two lines above for these, and add `wgpu` to Cargo.toml features:
// use burn::backend::Wgpu;
// type Backend         = Wgpu;
// type AutodiffBackend = Autodiff<Backend>;

fn main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Dataset format: headerless CSV, one sample per line.
    //   label,text
    //
    // Labels are arbitrary strings — mapped to contiguous integers in
    // alphabetical order. Two labels → binary classifier. N labels → N-class.
    // ─────────────────────────────────────────────────────────────────────────
    let data_path = "data/dataset.csv";

    // Pre-pass: resolve vocab_size and num_classes before building the model.
    let (_train_ds, _val_ds, tokenizer, class_names) =
        data::TextDataset::from_csv(data_path, 128, 0.15, 1);

    let vocab_size  = tokenizer.vocab_size();
    let num_classes = class_names.len();
    println!("vocab_size={vocab_size}  num_classes={num_classes}  classes={class_names:?}");

    // ── Model config ──────────────────────────────────────────────────────────
    // Defaults: embed_dim=64, conv1_filters=128 (k=3), conv2_filters=64 (k=5).
    // Override via builder methods:
    //   TextCnnConfig::new(vocab_size, num_classes).with_embed_dim(128)
    let model_config = TextCnnConfig::new(vocab_size, num_classes);

    // ── Training config ───────────────────────────────────────────────────────
    let config = TrainingConfig::new(model_config, AdamConfig::new())
        .with_num_epochs(10)
        .with_batch_size(32)
        .with_max_seq_len(128);

    train::<AutodiffBackend>(data_path, &config, "artifacts", Default::default());
}
