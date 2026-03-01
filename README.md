# cnn-text

Text classification trained in [Burn](https://burn.dev/) (Rust).
Training backend is selectable at build time via Cargo features (default: GPU/WebGPU).
Inference always runs on CPU.

## Architectures

### FastText (default)

```
tokens [B, L]
  → Embedding(vocab, E)
  → Mean over L
  → Linear(E, num_classes)
```

### Kim CNN (`--arch kimcnn`)

```
tokens [B, L]
  → Embedding(vocab, E)
  → Conv1d(k=3) → ReLU → GlobalMaxPool → [B, F] ┐
  → Conv1d(k=4) → ReLU → GlobalMaxPool → [B, F] ├─ cat → [B, 3F]
  → Conv1d(k=5) → ReLU → GlobalMaxPool → [B, F] ┘
  → Dropout(0.5)
  → Linear(3F, num_classes)
```

Inference auto-detects the architecture from the saved `config.json`.

---

## Results (AG News, 4 classes)

| Arch | Embeddings | Val acc |
|---|---|---|
| FastText | BPE scratch | ~91% |
| FastText | GloVe fine-tuned | ~92% |
| FastText | GloVe frozen | ~89% |
| Kim CNN | BPE scratch | TBD |
| Kim CNN | GloVe fine-tuned | TBD |

---

## Setup

### Dataset

```bash
cargo run -- fetch-agnews
```

Writes `data/dataset.csv` — headerless `label,text` per line.

### GloVe vectors (optional)

```bash
cargo run -- fetch-glove
```

```bash
cargo run -- fetch-glove 300
```

Downloads the Stanford 822 MB zip, extracts the requested dimension file, deletes the zip.
Options: 50, 100 (default), 200, 300.

---

## Train

### FastText, BPE

```bash
cargo run --release -- train <model>
```

### FastText, GloVe

```bash
cargo run --release -- train <model> --glove data/glove.6B.100d.txt
```

### Kim CNN, BPE

```bash
cargo run --release -- train <model> --arch kimcnn
```

### Kim CNN, GloVe

```bash
cargo run --release -- train <model> --arch kimcnn --glove data/glove.6B.100d.txt
```

### Frozen embeddings (any arch)

```bash
cargo run --release -- train <model> --arch kimcnn --glove data/glove.6B.100d.txt --freeze
```

`<model>` names the output directory under `artifacts/`. Defaults to `default`.

---

## Predict

```bash
cargo run --release -- predict <model> "Fed raises rates as inflation fears grow"
```

```
Business  (97.2%)
```

Works with both FastText and Kim CNN models — architecture is detected automatically.

---

## Hyperparameters

Set in `main.rs` via `TrainingConfig`:

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 10 | Max training epochs |
| `batch_size` | 32 | Batch size |
| `max_seq_len` | 128 | Tokens per sample (pad/truncate) |
| `vocab_size` | 8192 | BPE vocabulary size — ignored with GloVe |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `learning_rate` | 1e-3 | Adam learning rate |
| `embed_dim` | 100 | Embedding dimension — overridden by GloVe file |
| `freeze_embeddings` | false | Freeze embedding weights during training |
| `patience` | 3 | Early stopping patience (0 = disabled) |
| `num_filters` | 128 | Kim CNN: filters per kernel size (total = 3×) |
| `dropout` | 0.5 | Kim CNN: dropout after concatenation |

---

## Artifacts

```
artifacts/<model>/
  config.json       # architecture hyperparams + class names
  model.mpk         # best weights (only updated when val loss improves)
  metrics.csv       # epoch,train_loss,train_acc,val_loss,val_acc
  tokenizer.json    # BPE or word-level vocab
  checkpoint/
    model-1.mpk
    model-2.mpk
    ...
```

To roll back to a specific epoch:

```bash
cp artifacts/<model>/checkpoint/model-3.mpk artifacts/<model>/model.mpk
```

---

## Backend

GPU (WebGPU) is the default. Switch to CPU-only:

```bash
cargo run --no-default-features -- train
```

Inference always uses `NdArray` (CPU) regardless of the training backend.
