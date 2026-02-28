# cnn-text

Text classification with a small CNN trained from scratch in [Burn](https://burn.dev/) (Rust).

## Architecture

```
tokens [B, L]
  → Embedding(vocab, 64)
  → Conv1d(kernel=3, filters=128) + ReLU
  → Conv1d(kernel=5, filters=64)  + ReLU
  → GlobalMaxPool
  → Linear(64, num_classes)
```

Tokenization is BPE (via HuggingFace `tokenizers`), trained on your corpus — no pretrained weights.

## Data format

A headerless CSV with `label,text` per line:

```
spam,Congratulations you have won a free prize
ham,Let me know when you are free to meet
spam,Click here to claim your reward now
```

Place it at `data/dataset.csv` (gitignored).

## Train

```bash
cargo run --release                    # saves to artifacts/default/
cargo run --release -- train v1        # saves to artifacts/v1/
```

Hyperparameters are set in `main.rs` via `TrainingConfig`:

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `max_seq_len` | 128 | Tokens per sample (pad/truncate) |
| `vocab_size` | 8192 | BPE vocabulary size target |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `learning_rate` | 1e-3 | Adam learning rate |
| `embed_dim` | 64 | Embedding dimension |
| `conv1_filters` | 128 | Filters in first conv layer |
| `conv2_filters` | 64 | Filters in second conv layer |

Each run saves to its own directory:

```
artifacts/v1/
  tokenizer.json   # BPE vocab + padding config
  config.json      # arch hyperparams + class names
  weights.*        # model parameters
```

## Predict

```bash
cargo run --release -- predict v1 "free money click here now"
# spam  (96.2%)
```

## GPU

Swap the `burn` line in `Cargo.toml`:

```toml
burn = { version = "0.20.1", features = ["train", "wgpu"] }
```

And update the backend aliases in `main.rs`:

```rust
use burn::backend::{Autodiff, Wgpu};
type AutodiffBackend = Autodiff<Wgpu>;
```
