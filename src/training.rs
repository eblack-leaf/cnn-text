use std::collections::HashSet;

use burn::{
    backend::NdArray,
    config::Config,
    prelude::Module,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use burn::data::dataset::Dataset;
use crate::{
    data::{TextBatch, TextBatcher, TextDataset, Tokenizer},
    model::{TextCnn, TextCnnConfig},
};

// ── TrainStep / InferenceStep ─────────────────────────────────────────────────

impl<B: AutodiffBackend> TrainStep for TextCnn<B> {
    type Input  = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let out = self.forward_classification(item.tokens, item.labels);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> InferenceStep for TextCnn<B> {
    type Input  = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.tokens, item.labels)
    }
}

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

/// Collect every whitespace-split, lowercased word that appears in the dataset.
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

/// Parse a GloVe text file, keeping only words present in `corpus_words`.
/// Returns `(words, vectors)` — does NOT include PAD/UNK rows.
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
            // ── GloVe path ────────────────────────────────────────────────────
            let corpus_words = collect_corpus_words(data_path);
            let (glove_words, glove_vecs) = load_glove(glove, &corpus_words);
            let embed_dim = glove_vecs[0].len();

            // Embedding matrix: row 0 = PAD (zeros), row 1 = UNK (zeros), then GloVe
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
            // ── BPE path (original) ───────────────────────────────────────────
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

    let _ = tokenizer; // saved to disk; not needed further

    // ── Model ─────────────────────────────────────────────────────────────────

    let model = if let Some(ref matrix) = embedding_matrix {
        // Initialise with pretrained weights on CPU (NdArray) to avoid Burn's
        // Wgpu fusion backend garbage-collecting the custom Param tensor handle.
        // Save to a temp dir then reload normally on the training backend.
        let init_dir = format!("{model_dir}/.pretrained_init");
        let cpu_model = model_config.init_with_embeddings::<NdArray>(&Default::default(), matrix);
        cpu_model.save_pretrained(&model_config, &init_dir);
        let model = model_config
            .init::<B>(&device)
            .load_file(format!("{init_dir}/model"), &burn::record::CompactRecorder::new(), &device)
            .unwrap();
        std::fs::remove_dir_all(&init_dir).ok();
        model
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

    // ── Learner ───────────────────────────────────────────────────────────────

    let learner = Learner::new(model, config.optimizer.init(), config.learning_rate);

    let result: burn::train::LearningResult<TextCnn<B::InnerBackend>> =
        SupervisedTraining::new(model_dir, train_loader, val_loader)
        .metric_train_numeric(AccuracyMetric::<NdArray>::new())
        .metric_valid_numeric(AccuracyMetric::<NdArray>::new())
        .metric_train_numeric(LossMetric::<NdArray>::new())
        .metric_valid_numeric(LossMetric::<NdArray>::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    result.model.save_pretrained(&model_config, model_dir);
    println!("Saved → {model_dir}/");
}
