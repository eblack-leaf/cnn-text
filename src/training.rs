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
//
// Burn 0.20: these traits use associated types instead of generic parameters.

impl<B: AutodiffBackend> TrainStep for TextCnn<B> {
    type Input  = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let out = self.forward_classification(item.tokens, item.labels);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

// Implemented for any Backend so it covers the inner (non-autodiff) module
// that the Learner uses for validation.
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
    pub model:     TextCnnConfig,
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
    /// Minimum word frequency to include in the vocabulary.
    #[config(default = 1)]
    pub min_freq: usize,
}

// ── Training entry point ──────────────────────────────────────────────────────

/// Load data, build the model, and run the training loop.
///
/// `artifact_dir` receives checkpoints and the final saved model.
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

    // ── Data ─────────────────────────────────────────────────────────────────
    let (train_ds, val_ds, _tokenizer, class_names) = TextDataset::from_csv(
        data_path,
        config.max_seq_len,
        config.val_ratio,
        config.min_freq,
    );

    println!(
        "Loaded: {} train / {} val — {} classes: {:?}",
        train_ds.len(),
        val_ds.len(),
        class_names.len(),
        class_names,
    );

    // Burn 0.20: the device is given to the DataLoaderBuilder, not the Batcher.
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

    // ── Model & learner ───────────────────────────────────────────────────────
    let model = config.model.init::<B>(&device);

    // `f64` implements `LrScheduler` as a constant LR.
    let learner = Learner::new(model, config.optimizer.init(), 1e-3_f64);

    // ── Supervised training ───────────────────────────────────────────────────
    // Metrics are always evaluated on NdArray (Burn's sync backend) regardless
    // of the training backend, hence `AccuracyMetric::<NdArray>`.
    let result = SupervisedTraining::new(artifact_dir, train_loader, val_loader)
        .metric_train_numeric(AccuracyMetric::<NdArray>::new())
        .metric_valid_numeric(AccuracyMetric::<NdArray>::new())
        .metric_train_numeric(LossMetric::<NdArray>::new())
        .metric_valid_numeric(LossMetric::<NdArray>::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    // ── Save ──────────────────────────────────────────────────────────────────
    result
        .model
        .save_file(format!("{artifact_dir}/model_final"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    println!("Saved → {artifact_dir}/model_final");
}
