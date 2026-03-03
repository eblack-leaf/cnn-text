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

### Bidirectional GRU (`--arch bigru`)

```
tokens [B, L]
  → Embedding(vocab, E)
  → GRU forward  → [B, L, H] ┐
  → GRU backward → [B, L, H] ┘ cat → [B, L, 2H]
  → Global max pool  [B, 2H]
  → Dropout(0.5)
  → Linear(2H, num_classes)
```

> **Note:** GRUs iterate sequentially over positions and are ~20× slower than CNN/FastText on GPU.

### Tiny Transformer (`--arch transformer`)

```
tokens [B, L]
  → TokenEmbedding(vocab, E) + PosEmbedding(max_len, E)  [B, L, E]
  → TransformerEncoder (n_layers × MultiheadAttn + FFN)  [B, L, E]
  → Mean pool over L                                      [B, E]
  → Dropout(0.1)
  → Linear(E, num_classes)
```

Fully parallel — no sequential loop. Attention is PAD-masked.
`embed_dim` must be divisible by `num_heads` (default 100 / 4 = 25 ✓).

Inference auto-detects the architecture from the saved `config.json`.

---

## Results (AG News, 4 classes)

| Arch | Embeddings | Val acc | s/epoch (GPU) |
|---|---|---|---|
| Kim CNN | GloVe 100d fine-tuned | 92.8% | — |
| BiGRU | GloVe 300d fine-tuned | 92.7% | — |
| Tiny Transformer | GloVe 300d fine-tuned (warmup + grad-clip) | 92.6% | ~24s |
| FastText + bigrams | GloVe 100d fine-tuned | 92.3% | — |
| FastText | GloVe 100d fine-tuned | 92.3% | — |
| FastText + bigrams | BPE scratch | 92.1% | — |
| Kim CNN | BPE scratch | ~91% | — |
| FastText | BPE scratch | ~91% | — |
| FastText | GloVe 100d frozen | ~89% | — |

s/epoch measured on GPU (WebGPU backend); only logged for runs that print it to console — not recorded in `metrics.csv`.

### Model size

GloVe-vocab runs (59 828 words). Embedding params = token table (+ bigram table for FastText+bigrams).

| Arch | Embed dim | Model params | Embedding params | Total |
|---|---|---|---|---|
| Tiny Transformer | 300d | 1.07 M | 17.95 M | ~19.0 M |
| BiGRU | 300d | 0.33 M | 17.95 M | ~18.3 M |
| FastText + bigrams | 100d | < 0.01 M | 15.98 M | ~16.0 M |
| Tiny Transformer | 100d | 0.20 M | 5.98 M | ~6.2 M |
| Kim CNN | 100d | 0.16 M | 5.98 M | ~6.1 M |
| FastText | 100d | < 0.01 M | 5.98 M | ~6.0 M |

The embedding table is almost everything — the actual model is tiny. Kim CNN's encoder (3 Conv1d kernels) is only 154 k params; the Transformer's 2-layer encoder adds 1 M at 300d but still can't beat the CNN.

**Takeaways:**
- Kim CNN is the best bang-for-buck: matches or beats everything else at 6 M params and fast parallel inference
- BiGRU and the 300d Transformer are 3× larger for tiny accuracy gains and iterate sequentially (BiGRU ~20× slower per epoch than CNN/FastText)
- Transformer needs warmup + grad-clip to train stably with GloVe; even then it only ties CNN with 3× the parameters
- ~93% appears to be the practical ceiling for small models without pre-trained weights; SOTA is ~95.5% with large pre-trained transformers

---

## Datasets

### AG News (default)

```bash
cargo run -- fetch-agnews
```

Writes `data/dataset.csv` — headerless `label,text` per line. 4-class news topic classification (~120k samples).

### Amazon Review Polarity

Binary sentiment from Amazon product reviews. 3.6M train / 400k test, pre-split.
Place the Kaggle archive at `data/amazon_review_polarity_csv/` (contains `train.csv` and `test.csv`).

### SMS Spam Collection

Ham/spam binary classification. ~5 500 samples, tab-separated.
Place `SMSSpamCollection` at `data/sms+spam+collection/SMSSpamCollection`.

### IMDB

Binary movie-review sentiment. 50k samples with CSV header. HTML `<br />` tags are stripped.
Place `IMDB Dataset.csv` at `data/archive/IMDB Dataset.csv`.

### GloVe vectors (optional, any dataset)

```bash
cargo run -- fetch-glove          # 100d (default)
cargo run -- fetch-glove 300      # 300d
```

Downloads the Stanford 822 MB zip, extracts the requested dimension file, deletes the zip.
Options: 50, 100 (default), 200, 300.

---

## Train

### Dataset flag

Use `--dataset` to select the source. The path defaults to the standard location for each dataset; override it with `--data`.

| `--dataset` | Default path |
|---|---|
| `custom` (default) | `data/dataset.csv` |
| `amazon` | `data/amazon_review_polarity_csv/` |
| `sms` | `data/sms+spam+collection/SMSSpamCollection` |
| `imdb` | `data/archive/IMDB Dataset.csv` |

```bash
# custom dataset at a non-default path
cargo run --release -- train mymodel --dataset custom --data /path/to/my.csv
```

Amazon uses its own train/test split as the validation set; all other datasets split by `val_ratio` (default 15%).

### FastText, BPE

```bash
cargo run --release -- train <model>
cargo run --release -- train <model> --dataset imdb
cargo run --release -- train <model> --dataset sms
cargo run --release -- train <model> --dataset amazon
```

*(bigrams are on by default — set `bigram_buckets = 0` in `main.rs` to disable)*

### FastText, GloVe

```bash
cargo run --release -- train <model> --glove data/glove.6B.100d.txt
cargo run --release -- train <model> --dataset imdb --glove data/glove.6B.100d.txt
```

### FastText + bigrams, GloVe

```bash
cargo run --release -- train <model> --glove data/glove.6B.100d.txt
```

### Kim CNN, GloVe

```bash
cargo run --release -- train <model> --arch kimcnn --glove data/glove.6B.100d.txt
```

### Tiny Transformer, GloVe

```bash
cargo run --release -- train <model> --arch transformer --glove data/glove.6B.100d.txt
```

### Tiny Transformer, BPE

```bash
cargo run --release -- train <model> --arch transformer
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

Architecture is detected automatically from the saved `config.json`.

---

## Hyperparameters

Set in `main.rs` via `TrainingConfig`:

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 20 | Max training epochs |
| `batch_size` | 32 | Batch size |
| `max_seq_len` | 128 | Tokens per sample (pad/truncate) |
| `vocab_size` | 8192 | BPE vocabulary size — ignored with GloVe |
| `val_ratio` | 0.15 | Fraction held out for validation |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `embed_dim` | 100 | Embedding dimension — overridden by GloVe file |
| `freeze_embeddings` | false | Freeze embedding weights during training |
| `patience` | 3 | Early stopping patience (0 = disabled) |
| `bigram_buckets` | 100_000 | FastText: bigram hash table size (0 = disabled) |
| `num_filters` | 128 | Kim CNN: filters per kernel size (total = 3×) |
| `hidden_dim` | 128 | BiGRU: hidden size per direction |
| `dropout` | 0.5 | Kim CNN / BiGRU: dropout before classifier |
| `num_heads` | 4 | Transformer: attention heads (`embed_dim` must be divisible) |
| `num_layers` | 2 | Transformer: encoder layers |
| `d_ff` | 256 | Transformer: FFN hidden dim |
| `attn_dropout` | 0.1 | Transformer: dropout inside layers and before classifier |

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
