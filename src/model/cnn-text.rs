use burn::prelude::{Backend, Int, TensorData};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::activation;
use burn::Tensor;
use burn::module::{Module, Param, ParamId};
use burn::record::CompactRecorder;
use burn::config::Config;
use burn::train::ClassificationOutput;
use burn::nn::loss::CrossEntropyLossConfig;
use crate::model::Classify;

// ── CnnText ────────────────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → TokenEmbed + PosEmbed  [B, L, E]
//     → permute                [B, E, L]
//     → Conv1d(k=3) → ReLU    [B, F, L-2]  ┐
//     → Conv1d(k=4) → ReLU    [B, F, L-3]  ├─ per-branch attention pool → [B, F] each
//     → Conv1d(k=5) → ReLU    [B, F, L-4]  ┘
//     → cat [B, 3F]  →  Dropout  →  Linear [B, C]
//
// Attention pool (shared scorer):
//   x [B, L', F]  →  Linear(F, 1)  →  softmax over L'  →  weighted sum  →  [B, F]

#[derive(Config, Debug)]
pub struct CnnTextConfig {
    pub max_seq_len: usize,
    pub vocab_size:  usize,
    pub class_names: Vec<String>,
    #[config(default = 100)]
    pub embed_dim:   usize,
    #[config(default = 128)]
    pub num_filters: usize,
    #[config(default = 0.5)]
    pub dropout:     f64,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl CnnTextConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnText<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        CnnText {
            embedding,
            positional:  EmbeddingConfig::new(self.max_seq_len, self.embed_dim).init(device),
            conv3:       Conv1dConfig::new(self.embed_dim, self.num_filters, 3).init(device),
            conv4:       Conv1dConfig::new(self.embed_dim, self.num_filters, 4).init(device),
            conv5:       Conv1dConfig::new(self.embed_dim, self.num_filters, 5).init(device),
            attn:        LinearConfig::new(self.num_filters, 1).init(device),
            dropout:     DropoutConfig::new(self.dropout).init(),
            classifier:  LinearConfig::new(self.num_filters * 3, self.num_classes()).init(device),
        }
    }

    pub fn init_with_embeddings<B: Backend>(
        &self,
        device:  &B::Device,
        vectors: &[Vec<f32>],
    ) -> CnnText<B> {
        let vocab_size = vectors.len();
        let embed_dim  = vectors[0].len();
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let weight = Tensor::<B, 2>::from_data(
            TensorData::new(flat, [vocab_size, embed_dim]),
            device,
        );
        let embedding = EmbeddingConfig::new(vocab_size, embed_dim)
            .init(device)
            .load_record(EmbeddingRecord { weight: Param::initialized(ParamId::new(), weight) });
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        CnnText {
            embedding,
            positional:  EmbeddingConfig::new(self.max_seq_len, embed_dim).init(device),
            conv3:       Conv1dConfig::new(embed_dim, self.num_filters, 3).init(device),
            conv4:       Conv1dConfig::new(embed_dim, self.num_filters, 4).init(device),
            conv5:       Conv1dConfig::new(embed_dim, self.num_filters, 5).init(device),
            attn:        LinearConfig::new(self.num_filters, 1).init(device),
            dropout:     DropoutConfig::new(self.dropout).init(),
            classifier:  LinearConfig::new(self.num_filters * 3, self.num_classes()).init(device),
        }
    }

}

#[derive(Module, Debug)]
pub struct CnnText<B: Backend> {
    embedding:  Embedding<B>,
    positional: Embedding<B>,
    conv3:      Conv1d<B>,
    conv4:      Conv1d<B>,
    conv5:      Conv1d<B>,
    attn:       Linear<B>,   // shared scorer: Linear(F, 1)
    dropout:    Dropout,
    classifier: Linear<B>,
}

impl<B: Backend> CnnText<B> {
    /// Attention pool a single conv branch.
    ///
    /// x: [B, F, L']  →  swap  →  [B, L', F]
    ///   →  attn scorer  →  [B, L', 1]  →  softmax over L'
    ///   →  weighted sum  →  [B, F]
    fn attn_pool(attn: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = x.swap_dims(1, 2);                               // [B, L', F]
        let w = activation::softmax(attn.forward(x.clone()), 1); // [B, L', 1]
        (x * w).sum_dim(1).squeeze::<2>()                        // [B, F]
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [_batch_size, seq_len] = tokens.dims();
        let device = tokens.device();

        // 1. Token + positional embeddings → [B, E, L]
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>();                                               // [1, L]
        let x = (self.embedding.forward(tokens) + self.positional.forward(pos_ids))
            .swap_dims(1, 2);                                                // [B, E, L]

        // 2. Multi-scale convs with ReLU
        let c3 = activation::relu(self.conv3.forward(x.clone())); // [B, F, L-2]
        let c4 = activation::relu(self.conv4.forward(x.clone())); // [B, F, L-3]
        let c5 = activation::relu(self.conv5.forward(x));         // [B, F, L-4]

        // 3. Per-branch attention pooling → [B, F] each
        let p3 = Self::attn_pool(&self.attn, c3);
        let p4 = Self::attn_pool(&self.attn, c4);
        let p5 = Self::attn_pool(&self.attn, c5);

        // 4. Concatenate → [B, 3F], dropout, classify
        let x = self.dropout.forward(Tensor::cat(vec![p3, p4, p5], 1));
        self.classifier.forward(x)
    }

    pub fn save_pretrained(&self, config: &CnnTextConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        let mut json = serde_json::to_value(config).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("arch".to_string(), serde_json::json!("cnn-text"));
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }

    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, CnnTextConfig) {
        let config: CnnTextConfig = serde_json::from_str(
            &std::fs::read_to_string(format!("{dir}/config.json"))
                .unwrap_or_else(|_| panic!("No config.json in {dir}")),
        )
        .unwrap();
        let model = config
            .init::<B>(device)
            .load_file(format!("{dir}/model"), &CompactRecorder::new(), device)
            .unwrap();
        (model, config)
    }
}

impl<B: Backend> Classify<B> for CnnText<B> {
    fn forward_classification(
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
