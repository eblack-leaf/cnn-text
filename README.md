# On-Device Text Classification: Size vs Accuracy

Experiment to find the **smallest, fastest text classifier** that can be trained directly on a consumer device — so training data never has to leave the user. Built in Rust using [Burn](https://burn.dev/).

The key question: **how much accuracy do you give up as you shrink the model?** The sweep runner trains every combination of architecture × hyperparameter grid and produces a results table sorted by validation accuracy, with parameter counts for each run.

---

## Architectures

Four architectures ordered roughly by parameter count and compute cost:

### FastText (smallest / fastest)

```
tokens [B, L]  →  Embedding(vocab, E)  →  Masked mean  →  Linear(E, C)
```

Optional bigram features: set `bigram_buckets > 0` to add a second hash embedding for adjacent-token pairs.

### Kim CNN

```
tokens [B, L]  →  Embedding(vocab, E)  →  permute [B, E, L]
  → Conv1d(k=3) → ReLU → GlobalMaxPool → [B, F]  ┐
  → Conv1d(k=4) → ReLU → GlobalMaxPool → [B, F]  ├─ cat [B, 3F]
  → Conv1d(k=5) → ReLU → GlobalMaxPool → [B, F]  ┘
  → Dropout  →  Linear(3F, C)
```

Fully parallel — usually the best accuracy-per-parameter tradeoff.

### Bidirectional GRU

```
tokens [B, L]  →  Embedding(vocab, E)
  → GRU forward  [B, L, H]  ┐
  → GRU backward [B, L, H]  ┘ cat [B, L, 2H]  →  GlobalMaxPool  →  Dropout  →  Linear(2H, C)
```

Sequential — ~20× slower per epoch than CNN/FastText on GPU. Include only after benchmarking the faster models.

### Tiny Transformer

```
tokens [B, L]  →  TokenEmbed(vocab, E) + PosEmbed(L, E)  [B, L, E]
  →  TransformerEncoder (n_layers × MHA + FFN + LN)  [B, L, E]
  →  Mean pool  →  Dropout  →  Linear(E, C)
```

Fully parallel. PAD-masked attention. `embed_dim` must be divisible by `num_heads`.

---

## Quickstart

```bash
# 1. Build the project
cargo build --release

# 2. Fetch AG News (127k samples, 4 classes)
cargo run --release -- fetch-agnews

# 3. Run the sweep — trains every arch × embed_dim combination
cargo run --release -- sweep experiment.toml

# Results saved to artifacts/sweep_results.csv
# Sorted summary printed to terminal at the end
```

---

## Sweep Configuration (`experiment.toml`)

The sweep config is a TOML file with a `[grid]` section. Every combination of the arrays in `[grid]` becomes one training run.

```toml
dataset = "data/dataset.csv"

# Fixed training params
num_epochs  = 15
batch_size  = 128
max_seq_len = 64
patience    = 5

[grid]
arch      = ["fasttext", "kimcnn", "transformer"]
embed_dim = [32, 64, 128]
# dropout = [0.3, 0.5]          # optional extra sweep axis
# learning_rate = [1e-3, 5e-4]  # overrides per-arch defaults
```

Defaults per field if omitted:

| Field | Default | Notes |
|---|---|---|
| `num_epochs` | 15 | Max epochs (early stopping may cut short) |
| `batch_size` | 128 | |
| `max_seq_len` | 64 | AG News avg ~35 tokens — 64 is sufficient |
| `vocab_size` | 8192 | BPE vocab size; ignored with GloVe |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `patience` | 5 | Early stopping patience (0 = disabled) |
| `freeze` | false | Freeze embedding weights |
| `bigram_buckets` | 0 | FastText bigrams; set e.g. 100000 to enable |
| `num_filters` | 128 | Kim CNN: filters per kernel |
| `hidden_dim` | 128 | BiGRU: hidden size per direction |
| `num_heads` | 4 | Transformer: attention heads |
| `num_layers` | 2 | Transformer: encoder layers |
| `d_ff` | 256 | Transformer: FFN hidden dim |
| `dropout` | 0.5 | Dropout (single value or array to sweep) |
| `learning_rate` | arch-dependent | 1e-4 + 500 warmup for transformer; 1e-3 otherwise |

### Sweep output

- `artifacts/sweep_results.csv` — one row per run with: name, arch, embed_dim, dropout, lr, val_acc, best_epoch, non_embed_params, embed_params, total_params
- Terminal table sorted by val_acc descending

---

## Single Training Run

```bash
# Basic (FastText, BPE, all defaults)
cargo run --release -- train mymodel

# Kim CNN with 64-dim BPE embeddings
cargo run --release -- train mymodel --arch kimcnn --embed-dim 64

# Transformer with GloVe 100d (fetch first: cargo run -- fetch-glove 100)
cargo run --release -- train mymodel --arch transformer --glove data/glove.6B.100d.txt

# Load base params from a TOML file, override individual flags
cargo run --release -- train mymodel --config experiment.toml --arch fasttext --epochs 10
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--config <path>` | — | Load base hyperparams from a TOML file |
| `--arch <name>` | `fasttext` | `fasttext` \| `kimcnn` \| `bigru` \| `transformer` |
| `--dataset <kind>` | `agnews` | `agnews` \| `sms` \| `imdb` |
| `--data <path>` | (kind default) | Override dataset file path |
| `--glove <path>` | — | GloVe file; omit to use BPE from scratch |
| `--epochs N` | 15 | Max training epochs |
| `--batch-size N` | 128 | Batch size |
| `--lr F` | arch-dependent | AdamW learning rate |
| `--warmup-steps N` | arch-dependent | LR warmup steps |
| `--max-seq-len N` | 64 | Tokens per sample (pad/truncate) |
| `--vocab-size N` | 8192 | BPE vocabulary size |
| `--val-ratio F` | 0.15 | Validation split fraction |
| `--patience N` | 5 | Early stopping patience (0 = disabled) |
| `--embed-dim N` | 128 | Embedding dimension (overridden by GloVe file) |
| `--dropout F` | 0.5 | Dropout probability |
| `--num-filters N` | 128 | Kim CNN: filters per kernel |
| `--hidden-dim N` | 128 | BiGRU: hidden size per direction |
| `--num-heads N` | 4 | Transformer: attention heads |
| `--num-layers N` | 2 | Transformer: encoder layers |
| `--d-ff N` | 256 | Transformer: FFN hidden dim |
| `--bigram-buckets N` | 0 | FastText bigrams (0 = disabled) |
| `--freeze` | false | Freeze embedding weights |

---

## Dataset

### AG News (default)

4-class news topic classification (World / Sports / Business / SciTech). ~127k samples.

```bash
cargo run -- fetch-agnews   # writes data/dataset.csv
```

### GloVe embeddings (optional)

```bash
cargo run -- fetch-glove          # 100d (default)
cargo run -- fetch-glove 300      # 300d
```

Downloads the Stanford GloVe 822 MB zip, extracts the requested file, deletes the zip.

### Alternative datasets

| `--dataset` | Fetch command | Default path | Format |
|---|---|---|---|
| `agnews` (default) | `cargo run -- fetch-agnews` | `data/dataset.csv` | Headerless `label,text` CSV |
| `sms` | `cargo run -- fetch-sms` | `data/sms+spam+collection/SMSSpamCollection` | Tab-separated `label<TAB>text` |
| `imdb` | `cargo run -- fetch-imdb` | `data/archive/IMDB Dataset.csv` | CSV with header `review,sentiment` |

SMS source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (~5 500 samples, ham/spam).
IMDB source: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (~84 MB, 50 000 reviews, positive/negative).

---

## Inference

```bash
cargo run --release -- predict <model> "Fed raises rates as inflation fears grow"
# Business  (97.2%)
```

Architecture is detected automatically from `artifacts/<model>/config.json`.

---

## Results

*(Fill in after running the sweep — sorted by val_acc)*

| Run | arch | embed_dim | val% | non-embed params | total params |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### Model size reference

For AG News (4 classes), BPE vocab ~8192:

| arch | embed_dim | embed params | non-embed params |
|---|---|---|---|
| fasttext | 32 | ~262K | < 1K |
| fasttext | 128 | ~1.05M | < 1K |
| kimcnn | 32 | ~262K | ~150K |
| kimcnn | 128 | ~1.05M | ~540K |
| transformer (2L) | 64 | ~524K | ~270K |
| transformer (2L) | 128 | ~1.05M | ~1.05M |

The embedding table dominates. `non-embed params` is the actual model logic.

---

## Artifacts

```
artifacts/<model>/
  config.json       # architecture hyperparams + class names
  model.mpk         # best weights (updated only when val loss improves)
  metrics.csv       # epoch, train_loss, train_acc, val_loss, val_acc, epoch_secs
  tokenizer.json    # BPE or word-level vocabulary
  checkpoint/
    model-1.mpk
    model-2.mpk
    ...

artifacts/sweep_results.csv   # produced by `sweep` command
```

Roll back to a specific checkpoint:

```bash
cp artifacts/<model>/checkpoint/model-5.mpk artifacts/<model>/model.mpk
```

---

## Backend

GPU (WebGPU) is the default. Switch to CPU-only (no feature flag required at runtime):

```bash
cargo run --no-default-features --release -- sweep experiment.toml
```

Inference always uses `NdArray` (CPU) regardless of the training backend.
