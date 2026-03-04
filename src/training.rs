use std::collections::HashSet;
use std::time::Instant;

use burn::{
    backend::NdArray,
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Module,
    tensor::backend::{AutodiffBackend, Backend},
    tensor::ElementConversion,
};
use burn::data::dataset::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
use crate::{
    data::{TextBatch, TextBatcher, TextDataset, Tokenizer},
    datasets::DatasetKind,
    model::Classify,
};
use crate::model::bigru::BiGruConfig;
use crate::model::cnn_text::CnnTextConfig;
use crate::model::fast_text::FastTextConfig;
use crate::model::kimcnn::KimCnnConfig;
use crate::model::transformer::TinyTransformerConfig;
// ── Training configuration ────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 128)]
    pub max_seq_len: usize,
    /// Fraction of data held out for validation.
    #[config(default = 0.15)]
    pub val_ratio: f32,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// BPE vocabulary size target (ignored when using GloVe).
    #[config(default = 8192)]
    pub vocab_size: usize,
    // ── Architecture ──────────────────────────────────────────────────────────
    #[config(default = 100)]
    pub embed_dim: usize,
    /// Freeze embedding weights during training (only meaningful with GloVe).
    #[config(default = false)]
    pub freeze_embeddings: bool,
    /// Stop if val loss hasn't improved for this many epochs. 0 = disabled.
    #[config(default = 3)]
    pub patience: usize,
    /// FastText: bigram hash buckets. 0 = no bigrams.
    #[config(default = 100_000)]
    pub bigram_buckets: usize,
    // ── Kim CNN specific ──────────────────────────────────────────────────────
    /// Filters per kernel size (total output dim = num_filters × 3).
    #[config(default = 128)]
    pub num_filters: usize,
    // ── BiGRU specific ────────────────────────────────────────────────────────
    /// Hidden size per GRU direction (classifier input = hidden_dim × 2).
    #[config(default = 128)]
    pub hidden_dim: usize,
    // ── Shared ────────────────────────────────────────────────────────────────
    /// Dropout probability (Kim CNN: after concat; BiGRU: after max pool).
    #[config(default = 0.5)]
    pub dropout: f64,
    // ── Tiny Transformer specific ─────────────────────────────────────────────
    /// Attention heads (embed_dim must be divisible by num_heads).
    #[config(default = 4)]
    pub num_heads: usize,
    /// Number of transformer encoder layers.
    #[config(default = 2)]
    pub num_layers: usize,
    /// FFN hidden dim inside each transformer layer.
    #[config(default = 256)]
    pub d_ff: usize,
    /// Dropout used inside transformer layers and before the classifier head.
    #[config(default = 0.1)]
    pub attn_dropout: f64,
    /// Linear LR warmup steps (0 = disabled). LR ramps from 0 → learning_rate over this many steps.
    #[config(default = 0)]
    pub warmup_steps: usize,
}

// ── Tokenizer / embedding cache ───────────────────────────────────────────────
//
// All runs in a sweep that share the same (dataset_path, vocab_size) reuse the
// same BPE tokenizer — training it once saves 30-60s per run.
// GloVe/fastText runs that share (glove_path, dataset_path) reuse the filtered
// word vectors — avoids re-reading a 300MB+ file for every run.

const CACHE_DIR: &str = "data/.cache";

/// Stable cache path for a BPE tokenizer.
/// Key: (dataset_path, vocab_size) — same text corpus + same vocab size = same tokenizer.
fn bpe_cache_path(dataset_path: &str, vocab_size: usize) -> String {
    let key: String = dataset_path
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    format!("{CACHE_DIR}/bpe_{key}_{vocab_size}.json")
}

/// Stable cache path for filtered GloVe/fastText vectors.
/// Key: (glove file stem, dataset_path).
fn glove_cache_path(glove_path: &str, dataset_path: &str) -> String {
    let stem = std::path::Path::new(glove_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("glove");
    let dkey: String = dataset_path
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    format!("{CACHE_DIR}/glove_{stem}_{dkey}.txt")
}

// ── GloVe helpers ─────────────────────────────────────────────────────────────

fn collect_words_from_pairs(pairs: &[(String, String)]) -> HashSet<String> {
    let mut words = HashSet::new();
    for (_, text) in pairs {
        for word in text.split_whitespace() {
            words.insert(word.to_lowercase());
        }
    }
    words
}

pub fn load_glove(
    path:         &str,
    corpus_words: &HashSet<String>,
) -> (Vec<String>, Vec<Vec<f32>>) {
    eprintln!("Loading GloVe from {path} …");
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read GloVe file {path}: {e}"));

    let mut words: Vec<String> = Vec::new();
    let mut vecs:  Vec<Vec<f32>> = Vec::new();

    for line in content.lines() {
        let (word, rest) = line.split_once(' ').unwrap();
        // Skip fastText-style header line ("word_count dim"), e.g. "999994 300"
        if word.parse::<usize>().is_ok() { continue; }
        if !corpus_words.contains(word) { continue; }
        let vec: Vec<f32> = rest
            .split_whitespace()
            .map(|v| v.parse().expect("non-float in GloVe file"))
            .collect();
        words.push(word.to_string());
        vecs.push(vec);
    }

    eprintln!(
        "Kept {}/{} GloVe words seen in corpus",
        words.len(), corpus_words.len(),
    );
    (words, vecs)
}

/// Load GloVe/fastText vectors, caching the corpus-filtered result on first use.
///
/// The full vector file (300MB–1GB) is read once per (glove_path, dataset_path)
/// combination. Subsequent runs load only the filtered subset, which is much
/// smaller and reads in a fraction of the time.
fn load_glove_cached(
    path:         &str,
    corpus_words: &HashSet<String>,
    dataset_path: &str,
) -> (Vec<String>, Vec<Vec<f32>>) {
    let cache = glove_cache_path(path, dataset_path);

    if std::path::Path::new(&cache).exists() {
        eprintln!("Loading cached vectors from {cache} …");
        let content = std::fs::read_to_string(&cache)
            .unwrap_or_else(|e| panic!("Cannot read vector cache {cache}: {e}"));
        let mut words = Vec::new();
        let mut vecs  = Vec::new();
        for line in content.lines() {
            if line.is_empty() { continue; }
            let (word, rest) = line.split_once(' ').unwrap();
            let vec: Vec<f32> = rest
                .split_whitespace()
                .map(|v| v.parse().expect("bad float in cache"))
                .collect();
            words.push(word.to_string());
            vecs.push(vec);
        }
        eprintln!("  {} cached vectors loaded", words.len());
        return (words, vecs);
    }

    // Cache miss — load the full file and filter.
    let (words, vecs) = load_glove(path, corpus_words);

    // Write filtered vectors in the same text format.
    std::fs::create_dir_all(CACHE_DIR).ok();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&cache)
            .unwrap_or_else(|e| panic!("Cannot create cache {cache}: {e}")),
    );
    use std::io::Write;
    for (word, vec) in words.iter().zip(vecs.iter()) {
        let floats = vec.iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(out, "{word} {floats}").unwrap();
    }
    eprintln!("Saved vector cache → {cache}");

    (words, vecs)
}

// ── LR warmup ─────────────────────────────────────────────────────────────────

fn fmt_duration(d: std::time::Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 { format!("{}h{:02}m", s / 3600, (s % 3600) / 60) }
    else if s >= 60 { format!("{}m{:02}s", s / 60, s % 60) }
    else { format!("{s}s") }
}

fn warmup_lr(base_lr: f64, step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps > 0 && step <= warmup_steps {
        base_lr * step as f64 / warmup_steps as f64
    } else {
        base_lr
    }
}

// ── Shared epoch loop ─────────────────────────────────────────────────────────
//
// Generic over model type M. The only arch-specific part — `save_best` — is
// injected as a closure so neither model type leaks into this function.

fn run_epochs<B, M, O>(
    mut model:         M,
    mut optimizer:     O,
    config:            &TrainingConfig,
    train_loader:      &dyn DataLoader<B, TextBatch<B>>,
    val_loader:        &dyn DataLoader<B::InnerBackend, TextBatch<B::InnerBackend>>,
    train_eval_loader: &dyn DataLoader<B::InnerBackend, TextBatch<B::InnerBackend>>,
    train_batches:     usize,
    val_batches:       usize,
    metrics_path:      &str,
    model_dir:         &str,
    save_best:         impl Fn(&M::InnerModule, &str),
) where
    B: AutodiffBackend,
    B::InnerBackend: Backend,
    M: AutodiffModule<B> + Classify<B>,
    M::InnerModule: Classify<B::InnerBackend>,
    O: Optimizer<M, B>,
{
    let train_style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:40.green/dim}] {pos:>4}/{len} {msg}",
    ).unwrap().progress_chars("=>-");

    let val_style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:40.cyan/dim}] {pos:>4}/{len} {msg}",
    ).unwrap().progress_chars("=>-");

    let mut best_val_loss = f64::MAX;
    let mut no_improve    = 0usize;
    let mut global_step   = 0usize;
    let train_start       = Instant::now();

    for epoch in 1..=config.num_epochs {

        // ── Train ─────────────────────────────────────────────────────────────

        let epoch_start = Instant::now();
        let pb = ProgressBar::new(train_batches as u64);
        pb.set_style(train_style.clone());
        pb.set_prefix(format!("Epoch {:>2}/{} train", epoch, config.num_epochs));
        let (mut sum_loss, mut n_correct, mut n_total) = (0.0f64, 0usize, 0usize);

        for batch in train_loader.iter() {
            global_step += 1;
            let batch_len = batch.tokens.dims()[0];
            let out = model.forward_classification(batch.tokens, batch.labels);

            sum_loss  += out.loss.clone().into_scalar().elem::<f64>();
            let preds  = out.output.clone().argmax(1).squeeze::<1>();
            n_correct += preds.equal(out.targets.clone()).int().sum()
                .into_scalar().elem::<i64>() as usize;
            n_total += batch_len;

            let effective_lr = warmup_lr(config.learning_rate, global_step, config.warmup_steps);
            let grads = GradientsParams::from_grads(out.loss.backward(), &model);
            model = optimizer.step(effective_lr, model, grads);

            let sps = (pb.position() + 1) as f64
                / epoch_start.elapsed().as_secs_f64().max(1e-9);
            pb.set_message(format!(
                "loss={:.3}  acc={:.1}%  {:.1}it/s",
                sum_loss / (pb.position() + 1) as f64,
                100.0 * n_correct as f64 / n_total as f64,
                sps,
            ));
            pb.inc(1);
        }

        let train_loss = sum_loss / train_batches as f64;
        let train_acc  = 100.0 * n_correct as f64 / n_total as f64;
        pb.finish_with_message(format!("loss={train_loss:.3}  acc={train_acc:.1}%"));

        // ── Validate ──────────────────────────────────────────────────────────

        let vpb = ProgressBar::new(val_batches as u64);
        vpb.set_style(val_style.clone());
        vpb.set_prefix(format!("Epoch {:>2}/{}   val", epoch, config.num_epochs));
        let valid_model = model.valid();
        let (mut vsum_loss, mut vn_correct, mut vn_total) = (0.0f64, 0usize, 0usize);

        for batch in val_loader.iter() {
            let batch_len = batch.tokens.dims()[0];
            let out = valid_model.forward_classification(batch.tokens, batch.labels);

            vsum_loss  += out.loss.clone().into_scalar().elem::<f64>();
            let preds   = out.output.clone().argmax(1).squeeze::<1>();
            vn_correct += preds.equal(out.targets.clone()).int().sum()
                .into_scalar().elem::<i64>() as usize;
            vn_total += batch_len;

            vpb.set_message(format!(
                "loss={:.3}  acc={:.1}%",
                vsum_loss / (vpb.position() as f64),
                100.0 * vn_correct as f64 / vn_total as f64,
            ));
            vpb.inc(1);
        }

        let val_loss = vsum_loss / val_batches as f64;
        let val_acc  = 100.0 * vn_correct as f64 / vn_total as f64;
        vpb.finish_with_message(format!("loss={val_loss:.3}  acc={val_acc:.1}%"));

        // ── Timing ────────────────────────────────────────────────────────────

        let epoch_secs = epoch_start.elapsed();
        let avg_epoch  = train_start.elapsed() / epoch as u32;
        let eta        = avg_epoch * (config.num_epochs - epoch) as u32;
        println!(
            "  [{:.0}s/epoch | ~{} remaining]",
            epoch_secs.as_secs_f64(),
            fmt_duration(eta),
        );

        // ── Post-epoch train eval (same weights as val, val_batches worth) ────

        let (mut te_loss, mut te_correct, mut te_total, mut te_n) =
            (0.0f64, 0usize, 0usize, 0usize);
        for batch in train_eval_loader.iter() {
            if te_n >= val_batches { break; }
            let batch_len = batch.tokens.dims()[0];
            let out = valid_model.forward_classification(batch.tokens, batch.labels);
            te_loss    += out.loss.clone().into_scalar().elem::<f64>();
            let preds   = out.output.argmax(1).squeeze::<1>();
            te_correct += preds.equal(out.targets).int().sum()
                .into_scalar().elem::<i64>() as usize;
            te_total   += batch_len;
            te_n       += 1;
        }
        let train_eval_loss = te_loss    / te_n as f64;
        let train_eval_acc  = 100.0 * te_correct as f64 / te_total as f64;

        // ── Checkpoint ────────────────────────────────────────────────────────

        let ckpt_dir = format!("{model_dir}/checkpoint");
        std::fs::create_dir_all(&ckpt_dir).unwrap();
        valid_model.clone()
            .save_file(
                format!("{ckpt_dir}/model-{epoch}"),
                &burn::record::CompactRecorder::new(),
            )
            .unwrap();

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(metrics_path).unwrap();
        writeln!(f, "{epoch},{train_eval_loss:.4},{train_eval_acc:.2},{val_loss:.4},{val_acc:.2},{:.1}", epoch_secs.as_secs_f64()).unwrap();

        // ── Early stopping / best model ───────────────────────────────────────

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            no_improve    = 0;
            save_best(&valid_model, model_dir);
            println!(
                "  best   train loss={train_eval_loss:.3} acc={train_eval_acc:.1}%  |  val loss={val_loss:.3} acc={val_acc:.1}%  ✓",
            );
        } else {
            no_improve += 1;
            let patience = config.patience;
            println!(
                "         train loss={train_eval_loss:.3} acc={train_eval_acc:.1}%  |  val loss={val_loss:.3} acc={val_acc:.1}%  (no improve {no_improve}/{patience})",
            );
            if patience > 0 && no_improve >= patience {
                println!("Early stopping — best was epoch {}", epoch - no_improve);
                break;
            }
        }
    }
}

// ── Training entry point ──────────────────────────────────────────────────────

pub fn train<B: AutodiffBackend>(
    dataset_path: &str,
    dataset_kind: &DatasetKind,
    glove_path:   Option<&str>,
    config:       &TrainingConfig,
    arch:         &str,
    model_dir:    &str,
    device:       B::Device,
) where
    B::InnerBackend: Backend,
{
    B::seed(&device, config.seed);
    std::fs::create_dir_all(model_dir).expect("Could not create model directory");

    let tokenizer_path = format!("{model_dir}/tokenizer.json");

    // ── Data ──────────────────────────────────────────────────────────────────

    let (train_pairs, test_pairs) = dataset_kind.load(dataset_path);
    let has_test_split = test_pairs.is_some();

    let (train_ds, val_ds, embedding_matrix, class_names, vocab_size, embed_dim) =
        if let Some(glove) = glove_path {
            // Collect vocabulary from all available pairs.
            let all_pairs: Vec<_> = train_pairs.iter()
                .chain(test_pairs.iter().flatten())
                .cloned()
                .collect();
            let corpus_words = collect_words_from_pairs(&all_pairs);
            let (glove_words, glove_vecs) = load_glove_cached(glove, &corpus_words, dataset_path);
            let embed_dim = glove_vecs[0].len();
            let zeros = vec![0.0f32; embed_dim];
            let mut matrix = vec![zeros.clone(), zeros];
            matrix.extend(glove_vecs);

            let tokenizer = if std::path::Path::new(&tokenizer_path).exists() {
                println!("Loading tokenizer from {tokenizer_path}");
                Tokenizer::load(&tokenizer_path)
            } else {
                println!("Building word-level tokenizer from GloVe vocab…");
                if let Some(p) = std::path::Path::new(&tokenizer_path).parent() {
                    std::fs::create_dir_all(p).ok();
                }
                let tok = Tokenizer::from_word_vocab(&glove_words, config.max_seq_len);
                tok.save(&tokenizer_path);
                tok
            };

            let (train_ds, val_ds, class_names) = if has_test_split {
                TextDataset::from_split_pairs_tokenized(
                    train_pairs, test_pairs.unwrap(), &tokenizer)
            } else {
                TextDataset::from_pairs_tokenized(train_pairs, &tokenizer, config.val_ratio)
            };
            let vocab_size = matrix.len();
            println!(
                "Loaded: {} train / {} val — {} classes: {:?}",
                train_ds.len(), val_ds.len(), class_names.len(), class_names,
            );
            (train_ds, val_ds, Some(matrix), class_names, vocab_size, embed_dim)
        } else {
            // Use a shared cache path so the BPE tokenizer is trained only once
            // per (dataset, vocab_size) across all sweep runs.
            std::fs::create_dir_all(CACHE_DIR).ok();
            let cache_tok = bpe_cache_path(dataset_path, config.vocab_size);

            let (train_ds, val_ds, tokenizer, class_names) = if has_test_split {
                TextDataset::from_split_pairs(
                    train_pairs, test_pairs.unwrap(),
                    config.max_seq_len, config.vocab_size, &cache_tok,
                )
            } else {
                TextDataset::from_pairs(
                    train_pairs,
                    config.max_seq_len, config.val_ratio, config.vocab_size, &cache_tok,
                )
            };

            // Copy tokenizer to model dir so inference (predict) can load it.
            if !std::path::Path::new(&tokenizer_path).exists() {
                std::fs::copy(&cache_tok, &tokenizer_path).ok();
            }

            let vocab_size = tokenizer.vocab_size();
            let embed_dim  = config.embed_dim;
            println!(
                "Loaded: {} train / {} val — {} classes: {:?}",
                train_ds.len(), val_ds.len(), class_names.len(), class_names,
            );
            (train_ds, val_ds, None::<Vec<Vec<f32>>>, class_names, vocab_size, embed_dim)
        };

    let train_len = train_ds.len();
    let val_len   = val_ds.len();

    // ── Loaders ───────────────────────────────────────────────────────────────

    let train_ds_eval = train_ds.clone();

    let train_loader = DataLoaderBuilder::<B, _, _>::new(TextBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(train_ds);

    let val_loader = DataLoaderBuilder::<B::InnerBackend, _, _>::new(TextBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(val_ds);

    let train_eval_loader = DataLoaderBuilder::<B::InnerBackend, _, _>::new(TextBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(train_ds_eval);

    let train_batches = (train_len + config.batch_size - 1) / config.batch_size;
    let val_batches   = (val_len   + config.batch_size - 1) / config.batch_size;

    // ── Metrics ───────────────────────────────────────────────────────────────

    let metrics_path = format!("{model_dir}/metrics.csv");
    std::fs::write(&metrics_path, "epoch,train_loss,train_acc,val_loss,val_acc,epoch_secs\n").unwrap();

    // ── Model init (arch-specific) → shared epoch loop ────────────────────────

    /// Load a model init'd on NdArray CPU then transfer to the training backend.
    macro_rules! load_with_pretrained {
        ($cfg:expr, $cpu_init:expr) => {{
            let init_dir = format!("{model_dir}/.pretrained_init");
            $cpu_init.save_file(
                format!("{init_dir}/model"),
                &burn::record::CompactRecorder::new(),
            ).unwrap();
            let m = $cfg
                .init::<B>(&device)
                .load_file(
                    format!("{init_dir}/model"),
                    &burn::record::CompactRecorder::new(),
                    &device,
                )
                .unwrap();
            std::fs::remove_dir_all(&init_dir).ok();
            m
        }};
    }

    match arch {
        "kimcnn" => {
            let mc = KimCnnConfig::new(vocab_size, class_names)
                .with_embed_dim(embed_dim)
                .with_num_filters(config.num_filters)
                .with_dropout(config.dropout)
                .with_freeze_embeddings(config.freeze_embeddings);

            let model = if let Some(ref matrix) = embedding_matrix {
                std::fs::create_dir_all(format!("{model_dir}/.pretrained_init")).unwrap();
                let cpu = mc.init_with_embeddings::<NdArray>(&Default::default(), matrix);
                load_with_pretrained!(mc, cpu)
            } else {
                mc.init::<B>(&device)
            };

            let mc2 = mc.clone();
            run_epochs::<B, _, _>(
                model, config.optimizer.init(), config,
                &*train_loader, &*val_loader, &*train_eval_loader,
                train_batches, val_batches, &metrics_path, model_dir,
                move |m, dir| m.save_pretrained(&mc2, dir),
            );
        }

        "bigru" => {
            let bc = BiGruConfig::new(vocab_size, class_names)
                .with_embed_dim(embed_dim)
                .with_hidden_dim(config.hidden_dim)
                .with_dropout(config.dropout)
                .with_freeze_embeddings(config.freeze_embeddings);

            let model = if let Some(ref matrix) = embedding_matrix {
                std::fs::create_dir_all(format!("{model_dir}/.pretrained_init")).unwrap();
                let cpu = bc.init_with_embeddings::<NdArray>(&Default::default(), matrix);
                load_with_pretrained!(bc, cpu)
            } else {
                bc.init::<B>(&device)
            };

            let bc2 = bc.clone();
            run_epochs::<B, _, _>(
                model, config.optimizer.init(), config,
                &*train_loader, &*val_loader, &*train_eval_loader,
                train_batches, val_batches, &metrics_path, model_dir,
                move |m, dir| m.save_pretrained(&bc2, dir),
            );
        }

        "transformer" => {
            let tc = TinyTransformerConfig::new(vocab_size, class_names, config.max_seq_len)
                .with_embed_dim(embed_dim)
                .with_num_heads(config.num_heads)
                .with_num_layers(config.num_layers)
                .with_d_ff(config.d_ff)
                .with_dropout(config.attn_dropout)
                .with_freeze_embeddings(config.freeze_embeddings);

            let model = if let Some(ref matrix) = embedding_matrix {
                std::fs::create_dir_all(format!("{model_dir}/.pretrained_init")).unwrap();
                let cpu = tc.init_with_embeddings::<NdArray>(&Default::default(), matrix);
                load_with_pretrained!(tc, cpu)
            } else {
                tc.init::<B>(&device)
            };

            let tc2 = tc.clone();
            run_epochs::<B, _, _>(
                model, config.optimizer.init(), config,
                &*train_loader, &*val_loader, &*train_eval_loader,
                train_batches, val_batches, &metrics_path, model_dir,
                move |m, dir| m.save_pretrained(&tc2, dir),
            );
        }

        "cnn-text" => {
            let cc = CnnTextConfig::new(vocab_size, class_names)
                .with_embed_dim(embed_dim)
                .with_num_filters(config.num_filters)
                .with_dropout(config.dropout)
                .with_freeze_embeddings(config.freeze_embeddings);

            let model = if let Some(ref matrix) = embedding_matrix {
                std::fs::create_dir_all(format!("{model_dir}/.pretrained_init")).unwrap();
                let cpu = cc.init_with_embeddings::<NdArray>(&Default::default(), matrix);
                load_with_pretrained!(cc, cpu)
            } else {
                cc.init::<B>(&device)
            };

            let cc2 = cc.clone();
            run_epochs::<B, _, _>(
                model, config.optimizer.init(), config,
                &*train_loader, &*val_loader, &*train_eval_loader,
                train_batches, val_batches, &metrics_path, model_dir,
                move |m, dir| m.save_pretrained(&cc2, dir),
            );
        }

        _ => {
            // fasttext (default)
            let effective_bigram_buckets = config.bigram_buckets;
            let mc = FastTextConfig::new(vocab_size, class_names)
                .with_embed_dim(embed_dim)
                .with_bigram_buckets(effective_bigram_buckets)
                .with_freeze_embeddings(config.freeze_embeddings);

            let model = if let Some(ref matrix) = embedding_matrix {
                std::fs::create_dir_all(format!("{model_dir}/.pretrained_init")).unwrap();
                let cpu = mc.init_with_embeddings::<NdArray>(&Default::default(), matrix);
                load_with_pretrained!(mc, cpu)
            } else {
                mc.init::<B>(&device)
            };

            let mc2 = mc.clone();
            run_epochs::<B, _, _>(
                model, config.optimizer.init(), config,
                &*train_loader, &*val_loader, &*train_eval_loader,
                train_batches, val_batches, &metrics_path, model_dir,
                move |m, dir| m.save_pretrained(&mc2, dir),
            );
        }
    }

    println!("\nDone. Model saved → {model_dir}/");
}
