use burn::module::{Module, Param, ParamId};
use burn::prelude::{Backend, Int, TensorData};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::Tensor;
use burn::tensor::activation;
use burn::record::CompactRecorder;
use burn::train::ClassificationOutput;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::config::Config;
use crate::model::Classify;
// ── Kim CNN ────────────────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → Embedding [B, L, E]
//     → permute   [B, E, L]
//     → Conv1d(k=3) → ReLU → MaxPool → [B, F]  ┐
//     → Conv1d(k=4) → ReLU → MaxPool → [B, F]  ├─ cat → [B, 3F]
//     → Conv1d(k=5) → ReLU → MaxPool → [B, F]  ┘
//     → Dropout
//     → Linear [B, num_classes]


#[derive(Config, Debug)]
pub struct KimCnnConfig {
    pub vocab_size:        usize,
    pub class_names:       Vec<String>,
    /// Embedding dimension (overridden by GloVe file dimension).
    #[config(default = 100)]
    pub embed_dim:         usize,
    /// Filters per kernel size. Total feature dim = num_filters × 3.
    #[config(default = 128)]
    pub num_filters:       usize,
    /// Dropout applied after concatenation.
    #[config(default = 0.5)]
    pub dropout:           f64,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl KimCnnConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }

    fn make_conv<B: Backend>(&self, embed_dim: usize, k: usize, device: &B::Device) -> Conv1d<B> {
        Conv1dConfig::new(embed_dim, self.num_filters, k).init(device)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> KimCnn<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        KimCnn {
            embedding,
            conv3: self.make_conv::<B>(self.embed_dim, 3, device),
            conv4: self.make_conv::<B>(self.embed_dim, 4, device),
            conv5: self.make_conv::<B>(self.embed_dim, 5, device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.num_filters * 3, self.num_classes()).init(device),
        }
    }

    pub fn init_with_embeddings<B: Backend>(
        &self,
        device:  &B::Device,
        vectors: &[Vec<f32>],
    ) -> KimCnn<B> {
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
        KimCnn {
            embedding,
            conv3: self.make_conv::<B>(embed_dim, 3, device),
            conv4: self.make_conv::<B>(embed_dim, 4, device),
            conv5: self.make_conv::<B>(embed_dim, 5, device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.num_filters * 3, self.num_classes()).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct KimCnn<B: Backend> {
    embedding:  Embedding<B>,
    conv3:      Conv1d<B>,
    conv4:      Conv1d<B>,
    conv5:      Conv1d<B>,
    dropout:    Dropout,
    classifier: Linear<B>,
}

impl<B: Backend> KimCnn<B> {
    fn conv_pool(conv: &Conv1d<B>, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // x:    [B, E, L]
        // out:  [B, F, L-k+1]  after conv
        // pool: [B, F]         after global max
        let c = activation::relu(conv.forward(x)); // [B, F, L-k+1]
        c.max_dim(2).flatten::<2>(1, 2)            // [B, F]  — flatten dim 2 away; squeeze would also remove dim 0 when B=1
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(tokens); // [B, L, E]
        let x = x.swap_dims(1, 2);             // [B, E, L]

        let c3 = Self::conv_pool(&self.conv3, x.clone()); // [B, F]
        let c4 = Self::conv_pool(&self.conv4, x.clone()); // [B, F]
        let c5 = Self::conv_pool(&self.conv5, x);         // [B, F]

        let x = Tensor::cat(vec![c3, c4, c5], 1); // [B, 3F]
        let x = self.dropout.forward(x);
        self.classifier.forward(x)                // [B, num_classes]
    }

    pub fn save_pretrained(&self, config: &KimCnnConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        // Inject "arch" so inference can auto-detect the model type.
        let mut json = serde_json::to_value(config).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("arch".to_string(), serde_json::json!("kimcnn"));
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }

    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, KimCnnConfig) {
        let config: KimCnnConfig = serde_json::from_str(
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

impl<B: Backend> Classify<B> for KimCnn<B> {
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