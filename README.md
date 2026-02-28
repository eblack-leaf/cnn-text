# cnn-text

Text classification with a small CNN trained in [Burn](https://burn.dev/) (Rust).
Training backend is selectable at build time via Cargo features (default: GPU/WebGPU).
Inference always runs on CPU.

## Architecture

```
tokens [B, L]
  → Embedding(vocab, E)
  → Conv1d(kernel=3, filters=128) + ReLU
  → Conv1d(kernel=5, filters=64)  + ReLU
  → GlobalMaxPool
  → Linear(64, num_classes)
```

Two embedding modes:

| Mode | Tokenizer | Embedding init | Typical val accuracy (AG News) |
|---|---|---|---|
| BPE | Trained on corpus | Random | ~90% |
| GloVe | Word-level (GloVe vocab) | Pretrained vectors | ~93% |

## Data format

Headerless CSV, `label,text` per line:

```
World,Fed raises rates amid inflation concerns
Sports,Liverpool beat Arsenal in extra time
```

Place it at `data/dataset.csv`. To download AG News:

```bash
cargo run -- fetch-agnews
```

To download GloVe (default: 100d; options: 50, 100, 200, 300):

```bash
cargo run -- fetch-glove           # → data/glove.6B.100d.txt
cargo run -- fetch-glove 200       # → data/glove.6B.200d.txt
```

Downloads the full 822 MB zip from Stanford, extracts the requested file, then deletes the zip.

## Train

```bash
# BPE, random embeddings
cargo run --release -- train <model>

# GloVe, fine-tune embeddings
cargo run --release -- train <model> --glove data/glove.6B.100d.txt

# GloVe, frozen embeddings (faster, good when data is small)
cargo run --release -- train <model> --glove data/glove.6B.100d.txt --freeze
```

`<model>` names the output directory under `artifacts/`. Defaults to `default`.

Hyperparameters are set in `main.rs` via `TrainingConfig`:

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Batch size |
| `max_seq_len` | 128 | Tokens per sample (pad/truncate) |
| `vocab_size` | 8192 | BPE vocabulary size (ignored with GloVe) |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `learning_rate` | 1e-3 | Adam learning rate |
| `embed_dim` | 64 | Embedding dimension (overridden by GloVe file) |
| `conv1_filters` | 128 | Filters in first conv layer |
| `conv2_filters` | 64 | Filters in second conv layer |
| `freeze_embeddings` | false | Don't update embeddings during training |

Each run saves to its own directory:

```
artifacts/<model>/
  tokenizer.json        # vocab + padding config (BPE or word-level)
  config.json           # arch hyperparams + class names
  model.mpk             # latest weights (overwritten each epoch)
  checkpoint/
    model-1.mpk         # per-epoch checkpoints
    model-2.mpk
    ...
```

## Predict

```bash
cargo run --release -- predict <model> "text to classify"
# Sports  (94.3%)
```

Prediction always runs on the latest saved weights (`model.mpk`). To roll back to an earlier epoch:

```bash
cp artifacts/<model>/checkpoint/model-3.mpk artifacts/<model>/model.mpk
```

## Backend

GPU (WebGPU) is the default. Switch to CPU with:

```bash
cargo run --no-default-features -- train
```

Or permanently in `Cargo.toml`:

```toml
[features]
default = []   # drop "gpu"
```

Note: Inference always uses `NdArray` regardless of the feature flag.
