use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
        loss::CrossEntropyLossConfig,
    },
    tensor::{Int, Tensor, activation::relu, backend::Backend},
    train::ClassificationOutput,
};

/// Configuration for the TextCNN.
///
/// The only field you change between tasks is `num_classes`:
///   - 2  → binary (spam / ham)
///   - N  → multi-class intent detection, topic classification, etc.
#[derive(Config, Debug)]
pub struct TextCnnConfig {
    /// Total tokens in the vocabulary (set after building the tokenizer).
    pub vocab_size: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Token embedding dimension.
    #[config(default = 64)]
    pub embed_dim: usize,
    /// Filters for Conv block 1 — kernel size 3.
    #[config(default = 128)]
    pub conv1_filters: usize,
    /// Filters for Conv block 2 — kernel size 5.
    #[config(default = 64)]
    pub conv2_filters: usize,
}

/// Text CNN:
///
///   tokens [B, L]
///     → Embedding        [B, L, E]   E = embed_dim
///     → swap_dims        [B, E, L]   channels-first for Conv1d
///     → Conv1d(3, 128) + ReLU
///     → Conv1d(5,  64) + ReLU
///     → GlobalMaxPool    [B, 64]
///     → Linear           [B, num_classes]
#[derive(Module, Debug)]
pub struct TextCnn<B: Backend> {
    embedding:  Embedding<B>,
    conv1:      Conv1d<B>,
    conv2:      Conv1d<B>,
    classifier: Linear<B>,
}

impl TextCnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextCnn<B> {
        TextCnn {
            embedding: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            conv1: Conv1dConfig::new(self.embed_dim, self.conv1_filters, 3)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            conv2: Conv1dConfig::new(self.conv1_filters, self.conv2_filters, 5)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            classifier: LinearConfig::new(self.conv2_filters, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> TextCnn<B> {
    /// Forward pass. `tokens` shape: `[batch, seq_len]` (integer token ids).
    /// Returns logits of shape `[batch, num_classes]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        // [B, L] → [B, L, E]
        let x = self.embedding.forward(tokens);
        // [B, L, E] → [B, E, L]  (Conv1d expects channels-first)
        let x = x.swap_dims(1, 2);

        // Conv block 1: kernel=3, 128 filters
        let x = relu(self.conv1.forward(x));

        // Conv block 2: kernel=5, 64 filters
        let x = relu(self.conv2.forward(x));

        // Global max pool over the sequence length: [B, 64, L] → [B, 64, 1] → [B, 64]
        let x = x.max_dim(2).squeeze::<2>();

        self.classifier.forward(x)
    }

    /// Forward with cross-entropy loss — used by TrainStep / InferenceStep.
    pub fn forward_classification(
        &self,
        tokens:  Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(tokens);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        ClassificationOutput::new(loss, logits, targets)
    }
}

// `#[derive(Module)]` already generates Display; no manual impl needed.
