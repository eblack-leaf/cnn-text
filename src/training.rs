use burn::{
    backend::NdArray,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::Module,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::{
    data::{TextBatch, TextBatcher, TextDataset},
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
//
// Note: vocab_size and num_classes are NOT here — train() derives them from
// the data so they can never be wrong.

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
    /// BPE vocabulary size target (actual size may be smaller on tiny corpora).
    #[config(default = 8192)]
    pub vocab_size: usize,
    // ── Model architecture ────────────────────────────────────────────────────
    #[config(default = 64)]
    pub embed_dim: usize,
    #[config(default = 128)]
    pub conv1_filters: usize,
    #[config(default = 64)]
    pub conv2_filters: usize,
}

// ── Training entry point ──────────────────────────────────────────────────────

pub fn train<B: AutodiffBackend>(
    data_path:    &str,
    config:       &TrainingConfig,
    artifact_dir: &str,
    device:       B::Device,
) where
    B::InnerBackend: Backend,
{
    B::seed(&device, config.seed);
    std::fs::create_dir_all(artifact_dir).expect("Could not create artifact directory");

    // ── Data ──────────────────────────────────────────────────────────────────
    let tokenizer_path = "config/tokenizer.json";
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

    // ── Model — built from actual vocab/class counts ───────────────────────
    let model_config = TextCnnConfig::new(tokenizer.vocab_size(), class_names.len())
        .with_embed_dim(config.embed_dim)
        .with_conv1_filters(config.conv1_filters)
        .with_conv2_filters(config.conv2_filters);

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

    let learner = Learner::new(
        model_config.init::<B>(&device),
        config.optimizer.init(),
        1e-3_f64,
    );

    let result = SupervisedTraining::new(artifact_dir, train_loader, val_loader)
        .metric_train_numeric(AccuracyMetric::<NdArray>::new())
        .metric_valid_numeric(AccuracyMetric::<NdArray>::new())
        .metric_train_numeric(LossMetric::<NdArray>::new())
        .metric_valid_numeric(LossMetric::<NdArray>::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    result
        .model
        .save_file(format!("{artifact_dir}/model_final"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    println!("Saved → {artifact_dir}/model_final");
}
