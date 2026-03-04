# On-Device Text Classification: Size vs Accuracy

Experiment to find the **smallest, fastest text classifier** that can be trained directly on a consumer device — so training data never has to leave the user. Built in Rust using [Burn](https://burn.dev/).

The key question: **how much accuracy do you give up as you shrink the model?** The sweep runner trains every combination of architecture × hyperparameter grid and produces a results table sorted by validation accuracy, with parameter counts for each run.

---

## Architectures

Five architectures ordered roughly by parameter count and compute cost:

### FastText (smallest / fastest)

```
tokens [B, L]  →  Embedding(vocab, E)  →  Masked mean  →  Linear(E, C)
```

Optional bigram features: set `bigram_buckets > 0` to add a hash embedding for adjacent-token pairs — works with both BPE and word-level tokenization.

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

### CnnText (custom)

Custom architecture slot. Registered as `arch = "cnn-text"`. Implement in `src/model.rs`.

---

## Quickstart

```bash
# 1. Build the project
cargo build --release

# 2. Fetch AG News (127k samples, 4 classes)
cargo run --release -- fetch-agnews

# 3. Run the sweep
cargo run --release -- sweep experiment.toml

# Results saved to artifacts/agnews/sweep_results.csv
# Sorted summary printed to terminal at the end
```

---

## Sweep Configuration

Each experiment file uses per-architecture `[[runs]]` blocks. Only the axes declared in a block are swept for that arch — no inflated cross-products from irrelevant axes.

```toml
dataset      = "data/dataset.csv"
dataset_kind = "agnews"

num_epochs  = 15
batch_size  = 128
max_seq_len = 64
vocab_size  = 16384
patience    = 5

# Global arch defaults (used when a [[runs]] block omits the field)
num_filters  = 128
hidden_dim   = 128
num_heads    = 4
num_layers   = 2
d_ff         = 256
attn_dropout = 0.1
dropout      = 0.5

[[runs]]
arch           = "fasttext"
embed          = [32, 64, "data/glove.6B.100d.txt"]
bigram_buckets = [0, 100000]
learning_rate  = [1e-3, 5e-4]

[[runs]]
arch          = "kimcnn"
embed         = [64, 128, "data/glove.6B.100d.txt"]
dropout       = [0.3, 0.5]
learning_rate = [1e-3, 5e-4]

[[runs]]
arch          = "transformer"
embed         = [64, 128]
num_layers    = [1, 2]
num_heads     = [2, 4]
learning_rate = [1e-4, 5e-5]
```

### The `embed` axis

Each entry is either a **string** (pretrained vectors file path) or an **integer** (BPE dimension). Mixed lists are allowed:

```toml
embed = ["data/glove.6B.100d.txt", 32, "data/fasttext/wiki-news-300d-1M.vec", 128]
```

- String → load word vectors from that file, infer dim from filename
- Integer → train BPE embeddings from scratch at that dimension

The run name uses `g100d` / `g300d` for GloVe/fastText entries and `e32` / `e128` for BPE, only when the axis has more than one value.

### Global defaults

| Field | Default | Notes |
|---|---|---|
| `num_epochs` | 15 | Max epochs (early stopping may cut short) |
| `batch_size` | 128 | |
| `max_seq_len` | 64 | AG News avg ~35 tokens — 64 is sufficient |
| `vocab_size` | 8192 | BPE vocab size; ignored when using pretrained vectors |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `patience` | 5 | Early stopping patience (0 = disabled) |
| `freeze` | false | Freeze embedding weights |
| `bigram_buckets` | 0 | FastText bigrams; override per arch block |
| `num_filters` | 128 | Kim CNN: filters per kernel |
| `hidden_dim` | 128 | BiGRU: hidden size per direction |
| `num_heads` | 4 | Transformer: attention heads |
| `num_layers` | 2 | Transformer: encoder layers |
| `d_ff` | 256 | Transformer: FFN hidden dim |
| `attn_dropout` | 0.1 | Transformer attention dropout |
| `dropout` | 0.5 | KimCNN / BiGRU classifier dropout |
| `learning_rate` | arch-dependent | 1e-4 + 500 warmup for transformer; 1e-3 otherwise |

### Sweep output

Results are written to `artifacts/<dataset_kind>/sweep_results.csv` — one row per run — so different datasets never overwrite each other.

CSV columns: `dataset, name, arch, embed_source, embed_dim, bigram_buckets, dropout, attn_dropout, learning_rate, num_filters, hidden_dim, num_heads, num_layers, d_ff, val_acc, best_epoch, non_embed_params, embed_params, total_params`

---

## Single Training Run

```bash
# Basic (FastText, BPE, all defaults)
cargo run --release -- train mymodel

# Kim CNN with 64-dim BPE embeddings
cargo run --release -- train mymodel --arch kimcnn --embed-dim 64

# FastText with bigrams
cargo run --release -- train mymodel --arch fasttext --bigram-buckets 100000

# Transformer with GloVe 100d
cargo run --release -- train mymodel --arch transformer --glove data/glove.6B.100d.txt

# Load base params from a TOML file, override individual flags
cargo run --release -- train mymodel --config experiment.toml --arch fasttext --epochs 10
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--config <path>` | — | Load base hyperparams from a TOML file |
| `--arch <name>` | `fasttext` | `fasttext` \| `kimcnn` \| `bigru` \| `transformer` \| `cnn-text` |
| `--dataset <kind>` | `agnews` | `agnews` \| `sms` \| `imdb` |
| `--data <path>` | (kind default) | Override dataset file path |
| `--glove <path>` | — | Pretrained word vectors file (GloVe or fastText `.vec` format); omit to use BPE |
| `--epochs N` | 15 | Max training epochs |
| `--batch-size N` | 128 | Batch size |
| `--lr F` | arch-dependent | AdamW learning rate |
| `--warmup-steps N` | arch-dependent | LR warmup steps |
| `--max-seq-len N` | 64 | Tokens per sample (pad/truncate) |
| `--vocab-size N` | 8192 | BPE vocabulary size |
| `--val-ratio F` | 0.15 | Validation split fraction |
| `--patience N` | 5 | Early stopping patience (0 = disabled) |
| `--embed-dim N` | 128 | Embedding dimension (overridden by vector file) |
| `--dropout F` | 0.5 | Dropout probability |
| `--num-filters N` | 128 | Kim CNN: filters per kernel |
| `--hidden-dim N` | 128 | BiGRU: hidden size per direction |
| `--num-heads N` | 4 | Transformer: attention heads |
| `--num-layers N` | 2 | Transformer: encoder layers |
| `--d-ff N` | 256 | Transformer: FFN hidden dim |
| `--bigram-buckets N` | 0 | FastText bigrams (0 = disabled) |
| `--freeze` | false | Freeze embedding weights |

---

## Datasets and Embeddings

### AG News (default)

4-class news topic classification (World / Sports / Business / SciTech). ~127k samples.

```bash
cargo run -- fetch-agnews   # writes data/dataset.csv
```

### Alternative datasets

| `--dataset` | Fetch command | Default path | Samples | Classes |
|---|---|---|---|---|
| `agnews` | `cargo run -- fetch-agnews` | `data/dataset.csv` | ~127k | 4 |
| `sms` | `cargo run -- fetch-sms` | `data/sms+spam+collection/SMSSpamCollection` | ~5.5k | 2 (ham/spam) |
| `imdb` | `cargo run -- fetch-imdb` | `data/archive/IMDB Dataset.csv` | 50k | 2 (pos/neg) |

### Pretrained word vectors

Pretrained vectors are passed via the `embed` axis in a `[[runs]]` block or `--glove` flag for single runs. Both GloVe and fastText `.vec` format are supported.

```bash
# Stanford GloVe (~822 MB zip, extracts one file, deletes zip)
cargo run -- fetch-glove          # 100d → data/glove.6B.100d.txt
cargo run -- fetch-glove 300      # 300d → data/glove.6B.300d.txt

# Facebook fastText wiki-news (~600 MB zip)
cargo run -- fetch-fasttext       # 300d → data/fasttext/wiki-news-300d-1M.vec
```

---

## Inference

```bash
cargo run --release -- predict <model> "Fed raises rates as inflation fears grow"
# Business  (97.2%)
```

Architecture is detected automatically from `artifacts/<dataset>/<model>/config.json`.

---

## Results

Best result per architecture per dataset. `non-embed params` is the model logic excluding the embedding table — the number that matters for on-device cost since the embedding is just a lookup.

### AG News (4-class, ~108k train / 19k val)

| arch | embed | val_acc | non-embed params | total params |
|---|---|---|---|---|
| kimcnn | GloVe 300d | **93.11%** | 463k | 18.4M |
| kimcnn | GloVe 100d | 92.96% | 156k | 6.1M |
| transformer | GloVe 300d | 92.67% | 1.05M | 19.0M |
| transformer | GloVe 100d | 92.26% | 192k | 6.2M |
| fasttext | GloVe 300d | 92.00% | 1.2k | 17.9M |
| fasttext | GloVe 100d | 91.96% | 0.4k | 6.0M |
| kimcnn | BPE 128d | 91.01% | 199k | 1.25M |
| fasttext | BPE 128d | 90.77% | 0.5k | 1.05M |
| transformer | BPE 128d | 88.25% | 141k | 1.19M |

> Runs used `bigram_buckets = 0`. cnn-text and bigru not yet included.
> Ceiling ~93% matches published results (Kim CNN 2014: 91.5%, fastText paper: 92.5%). Label ambiguity between Business/SciTech and World/Business limits further gains without BERT-scale pretraining (BERT: ~94.9%).

### IMDB Sentiment (binary, ~42.5k train / 7.5k val, max_seq_len=300)

| arch | embed | val_acc | non-embed params | total params |
|---|---|---|---|---|
| kimcnn | GloVe 100d | **90.23%** | 155k | 7.4M |
| kimcnn | GloVe 300d | 90.21% | 462k | 22.3M |
| fasttext | GloVe 100d | 89.68% | 0.2k | 7.3M |
| fasttext | GloVe 300d | 89.57% | 0.6k | 21.8M |
| transformer | GloVe 300d | 88.52% | 1.12M | 22.9M |
| kimcnn | BPE 128d | 88.59% | 198k | 1.25M |
| fasttext | BPE 128d | 87.85% | 0.3k | 1.05M |
| transformer | GloVe 100d | 87.56% | 215k | 7.5M |
| transformer | BPE 128d | 84.61% | 171k | 1.22M |

> GloVe 100d matches 300d at a third of the embedding cost. Bigrams not yet tested on IMDB.

### SMS Spam (binary, ~4.7k train / 830 val, max_seq_len=48)

| arch | embed | val_acc | non-embed params | total params |
|---|---|---|---|---|
| kimcnn | GloVe 100d | **99.40%** | 77k | 641k |
| kimcnn | GloVe 300d | 99.28% | 231k | 1.92M |
| transformer | GloVe 300d | 99.16% | 894k | 2.59M |
| transformer | GloVe 100d | 99.04% | 138k | 702k |
| fasttext | GloVe 100d | 98.33% | 0.2k | 564k |
| fasttext | BPE 128d | 98.33% | 0.3k | 525k |
| kimcnn | BPE 32d | 98.92% | 25k | 156k |
| transformer | BPE 64d | 96.77% | 37k | 299k |

> **FastText is the on-device efficiency winner**: 98.33% with ~200 non-embed params. A 525k-total FastText model is a genuinely strong production option for SMS spam filtering. This sweep predates bigram support — FastText + bigrams expected to close the gap to KimCNN.

---

### Conclusions

1. **KimCNN wins accuracy across all datasets.** Parallel multi-scale convs with max pool is a strong, fast baseline that beats the tiny transformer in every configuration tested.

2. **GloVe beats BPE consistently.** BPE transformers on SMS top out at 96.77%; GloVe transformers reach 99.04%. Pretrained word vectors carry substantial signal that small models can't learn from scratch on limited data.

3. **GloVe 100d ≥ GloVe 300d in most cases.** 300d costs 3× the embedding memory with marginal or no accuracy gain. The exception is AG News KimCNN (93.11% vs 92.96%) — a 0.15% gain for 3× more embedding params.

4. **FastText is remarkably efficient.** Nearly competitive accuracy with essentially zero non-embed parameters — all capacity is in the embedding table. For latency-critical or memory-constrained on-device deployment it is the first choice.

5. **The tiny transformer needs GloVe to compete.** Training attention from scratch on small datasets (SMS, IMDB) with BPE produces weak results. With GloVe initialisation it reaches near-KimCNN accuracy but at higher parameter cost.

6. **AG News ceiling is ~93%.** Label ambiguity between Business/SciTech and World/Business articles cannot be resolved by better pooling or embeddings — the labels themselves overlap. Confident learning / label cleaning is the only path to meaningful improvement beyond this point.

### Model size reference

For AG News (4 classes, BPE vocab 16384, num_filters 128):

| arch | embed | embed params | non-embed params |
|---|---|---|---|
| fasttext | BPE 32d | 524k | < 1k |
| fasttext | BPE 128d | 2.10M | < 1k |
| fasttext | GloVe 100d | 5.98M | < 1k |
| kimcnn | BPE 64d | 1.05M | 100k |
| kimcnn | BPE 128d | 2.10M | 199k |
| kimcnn | GloVe 100d | 5.98M | 156k |
| transformer 2L | BPE 128d | 2.10M | 274k |
| transformer 2L | GloVe 100d | 5.98M | 192k |

The embedding table dominates total size. `non-embed params` is the actual model logic — the part that runs at inference time after the embedding lookup.

---

## Artifacts

```
artifacts/
  <dataset_kind>/           # agnews/ | imdb/ | sms/
    sweep_results.csv       # produced by `sweep` command
    <run_name>/
      config.json           # architecture hyperparams + class names
      model.mpk             # best weights (updated only when val loss improves)
      metrics.csv           # epoch, train_loss, train_acc, val_loss, val_acc, epoch_secs
      tokenizer.json        # BPE or word-level vocabulary
      checkpoint/
        model-1.mpk
        model-2.mpk
        ...
```

Roll back to a specific checkpoint:

```bash
cp artifacts/agnews/<model>/checkpoint/model-5.mpk artifacts/agnews/<model>/model.mpk
```

---

## Backend

GPU (WebGPU) is the default. Switch to CPU-only:

```bash
cargo run --no-default-features --release -- sweep experiment.toml
```

Inference always uses `NdArray` (CPU) regardless of the training backend.
