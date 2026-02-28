use std::collections::HashSet;

use burn::{
    backend::NdArray,
    config::Config,
    prelude::Module,
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::{AutodiffBackend, Backend},
    module::AutodiffModule,
    tensor::ElementConversion,
};
use burn::data::dataset::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
use crate::{
    data::{TextBatcher, TextDataset, Tokenizer},
    model::TextCnnConfig,
};

// ── Training configuration ────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
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
    // ── Model architecture ────────────────────────────────────────────────────
    #[config(default = 64)]
    pub embed_dim: usize,
    #[config(default = 128)]
    pub conv1_filters: usize,
    #[config(default = 64)]
    pub conv2_filters: usize,
    /// Freeze embedding weights during training (only meaningful with GloVe).
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

// ── GloVe helpers ─────────────────────────────────────────────────────────────

fn collect_corpus_words(data_path: &str) -> HashSet<String> {
    let content = std::fs::read_to_string(data_path)
        .unwrap_or_else(|e| panic!("Cannot read {data_path}: {e}"));
    let mut words = HashSet::new();
    for line in content.lines() {
        if let Some((_, text)) = line.split_once(',') {
            for word in text.split_whitespace() {
                words.insert(word.to_lowercase());
            }
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

// ── Training entry point ──────────────────────────────────────────────────────

pub fn train<B: AutodiffBackend>(
    data_path:  &str,
    glove_path: Option<&str>,
    config:     &TrainingConfig,
    model_dir:  &str,
    device:     B::Device,
) where
    B::InnerBackend: Backend,
{
    B::seed(&device, config.seed);
    std::fs::create_dir_all(model_dir).expect("Could not create model directory");

    let tokenizer_path = format!("{model_dir}/tokenizer.json");

    // ── Data + model config ───────────────────────────────────────────────────

    let (train_ds, val_ds, tokenizer, model_config, embedding_matrix) =
        if let Some(glove) = glove_path {
            let corpus_words = collect_corpus_words(data_path);
            let (glove_words, glove_vecs) = load_glove(glove, &corpus_words);
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

            let (train_ds, val_ds, class_names) =
                TextDataset::from_csv_tokenized(data_path, &tokenizer, config.val_ratio);

            println!(
                "Loaded: {} train / {} val — {} classes: {:?}",
                train_ds.len(), val_ds.len(), class_names.len(), class_names,
            );

            let mc = TextCnnConfig::new(matrix.len(), class_names)
                .with_embed_dim(embed_dim)
                .with_conv1_filters(config.conv1_filters)
                .with_conv2_filters(config.conv2_filters)
                .with_freeze_embeddings(config.freeze_embeddings);

            (train_ds, val_ds, tokenizer, mc, Some(matrix))
        } else {
            let (train_ds, val_ds, tokenizer, class_names) = TextDataset::from_csv(
                data_path,
                config.max_seq_len,
                config.val_ratio,
                config.vocab_size,
                &tokenizer_path,
            );

            println!(
                "Loaded: {} train / {} val — {} classes: {:?}",
                train_ds.len(), val_ds.len(), class_names.len(), class_names,
            );

            let mc = TextCnnConfig::new(tokenizer.vocab_size(), class_names)
                .with_embed_dim(config.embed_dim)
                .with_conv1_filters(config.conv1_filters)
                .with_conv2_filters(config.conv2_filters);

            (train_ds, val_ds, tokenizer, mc, None)
        };

    let _ = tokenizer;

    let train_len = train_ds.len();
    let val_len   = val_ds.len();

    // ── Model ─────────────────────────────────────────────────────────────────

    let mut model = if let Some(ref matrix) = embedding_matrix {
        let init_dir = format!("{model_dir}/.pretrained_init");
        let cpu_model = model_config.init_with_embeddings::<NdArray>(&Default::default(), matrix);
        cpu_model.save_pretrained(&model_config, &init_dir);
        let m = model_config
            .init::<B>(&device)
            .load_file(format!("{init_dir}/model"), &burn::record::CompactRecorder::new(), &device)
            .unwrap();
        std::fs::remove_dir_all(&init_dir).ok();
        m
    } else {
        model_config.init::<B>(&device)
    };

    // ── Loaders ───────────────────────────────────────────────────────────────

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

    let train_batches = (train_len + config.batch_size - 1) / config.batch_size;
    let val_batches   = (val_len   + config.batch_size - 1) / config.batch_size;

    // ── Optimizer ─────────────────────────────────────────────────────────────

    let mut optimizer = config.optimizer.init();

    // ── Progress styles ───────────────────────────────────────────────────────

    let train_style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:40.green/dim}] {pos:>4}/{len} {msg}",
    )
    .unwrap()
    .progress_chars("=>-");

    let val_style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:40.cyan/dim}] {pos:>4}/{len} {msg}",
    )
    .unwrap()
    .progress_chars("=>-");

    // ── Metrics log ───────────────────────────────────────────────────────────

    let metrics_path = format!("{model_dir}/metrics.csv");
    std::fs::write(&metrics_path, "epoch,train_loss,train_acc,val_loss,val_acc\n").unwrap();

    // ── Epoch loop ────────────────────────────────────────────────────────────

    for epoch in 1..=config.num_epochs {

        // ── Train ─────────────────────────────────────────────────────────────

        let pb = ProgressBar::new(train_batches as u64);
        pb.set_style(train_style.clone());
        pb.set_prefix(format!("Epoch {:>2}/{} train", epoch, config.num_epochs));

        let (mut sum_loss, mut n_correct, mut n_total) = (0.0f64, 0usize, 0usize);

        for batch in train_loader.iter() {
            let batch_len = batch.tokens.dims()[0];
            let out = model.forward_classification(batch.tokens, batch.labels);

            sum_loss  += out.loss.clone().into_scalar().elem::<f64>();
            let preds  = out.output.clone().argmax(1).squeeze::<1>();
            n_correct += preds
                .equal(out.targets.clone())
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>() as usize;
            n_total += batch_len;

            let grads = GradientsParams::from_grads(out.loss.backward(), &model);
            model = optimizer.step(config.learning_rate, model, grads);

            pb.set_message(format!(
                "loss={:.3}  acc={:.1}%",
                sum_loss / (pb.position() as f64),
                100.0 * n_correct as f64 / n_total as f64,
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
            vn_correct += preds
                .equal(out.targets.clone())
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>() as usize;
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

        // ── Checkpoint ────────────────────────────────────────────────────────

        // Latest weights (used by inference)
        valid_model.save_pretrained(&model_config, model_dir);

        // Per-epoch checkpoint
        let ckpt_dir = format!("{model_dir}/checkpoint");
        std::fs::create_dir_all(&ckpt_dir).unwrap();
        valid_model.clone()
            .save_file(format!("{ckpt_dir}/model-{epoch}"), &burn::record::CompactRecorder::new())
            .unwrap();

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&metrics_path).unwrap();
        writeln!(f, "{epoch},{train_loss:.4},{train_acc:.2},{val_loss:.4},{val_acc:.2}").unwrap();

        println!(
            "  saved  train loss={train_loss:.3} acc={train_acc:.1}%  |  val loss={val_loss:.3} acc={val_acc:.1}%",
        );
    }

    println!("\nDone. Model saved → {model_dir}/");
}
