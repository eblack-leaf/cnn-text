use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
        loss::CrossEntropyLossConfig,
    },
    record::CompactRecorder,
    tensor::{Int, Tensor, activation::relu, backend::Backend},
    train::ClassificationOutput,
};

#[derive(Config, Debug)]
pub struct TextCnnConfig {
    pub vocab_size:    usize,
    /// Output class names in label-index order — saved with the model so
    /// inference needs no external class list.
    pub class_names:   Vec<String>,
    #[config(default = 64)]
    pub embed_dim:     usize,
    #[config(default = 128)]
    pub conv1_filters: usize,
    #[config(default = 64)]
    pub conv2_filters: usize,
}

impl TextCnnConfig {
    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }
}

/// Text CNN:
///   tokens [B, L]
///     → Embedding        [B, L, E]
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
            classifier: LinearConfig::new(self.conv2_filters, self.num_classes()).init(device),
        }
    }
}

impl<B: Backend> TextCnn<B> {
    /// Save weights and config to `dir/`.
    pub fn save_pretrained(&self, config: &TextCnnConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/weights"), &CompactRecorder::new())
            .unwrap();
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(config).unwrap(),
        )
        .unwrap();
    }

    /// Load model and config from a directory written by `save_pretrained`.
    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, TextCnnConfig) {
        let config: TextCnnConfig = serde_json::from_str(
            &std::fs::read_to_string(format!("{dir}/config.json"))
                .unwrap_or_else(|_| panic!("No config.json in {dir}")),
        )
        .unwrap();
        let model = config
            .init::<B>(device)
            .load_file(format!("{dir}/weights"), &CompactRecorder::new(), device)
            .unwrap();
        (model, config)
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(tokens);
        let x = x.swap_dims(1, 2);
        let x = relu(self.conv1.forward(x));
        let x = relu(self.conv2.forward(x));
        let x = x.max_dim(2).squeeze::<2>();
        self.classifier.forward(x)
    }

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
