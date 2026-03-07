use burn::config::Config;
use burn::prelude::{Backend, Int, TensorData};
use burn::Tensor;
use burn::train::ClassificationOutput;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig};
use burn::module::{Module, Param, ParamId};
use burn::record::CompactRecorder;
use crate::model::Classify;
// ── Bidirectional GRU ──────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → Embedding            [B, L, E]
//     → GRU forward →        [B, L, H]  ┐
//     → GRU backward (flip)→ [B, L, H]  ┘ cat → [B, L, 2H]
//     → Global max pool      [B, 2H]
//     → Dropout
//     → Linear               [B, num_classes]


#[derive(Config, Debug)]
pub struct BiGruConfig {
    pub vocab_size:        usize,
    pub class_names:       Vec<String>,
    #[config(default = 100)]
    pub embed_dim:         usize,
    /// Hidden size per direction. Output fed to classifier is 2 × hidden_dim.
    #[config(default = 128)]
    pub hidden_dim:        usize,
    #[config(default = 0.5)]
    pub dropout:           f64,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl BiGruConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }

    fn make_gru<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> Gru<B> {
        GruConfig::new(embed_dim, self.hidden_dim, true).init(device)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BiGru<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        BiGru {
            embedding,
            gru_fwd:    self.make_gru::<B>(self.embed_dim, device),
            gru_bwd:    self.make_gru::<B>(self.embed_dim, device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.hidden_dim * 2, self.num_classes()).init(device),
        }
    }

    pub fn init_with_embeddings<B: Backend>(
        &self,
        device:  &B::Device,
        vectors: &[Vec<f32>],
    ) -> BiGru<B> {
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
        BiGru {
            embedding,
            gru_fwd:    self.make_gru::<B>(embed_dim, device),
            gru_bwd:    self.make_gru::<B>(embed_dim, device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.hidden_dim * 2, self.num_classes()).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct BiGru<B: Backend> {
    embedding:  Embedding<B>,
    gru_fwd:    Gru<B>,
    gru_bwd:    Gru<B>,
    dropout:    Dropout,
    classifier: Linear<B>,
}

impl<B: Backend> BiGru<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(tokens);               // [B, L, E]

        let h_fwd = self.gru_fwd.forward(x.clone(), None);   // [B, L, H]

        let h_bwd = self.gru_bwd.forward(x.flip([1]), None); // [B, L, H] on reversed input
        let h_bwd = h_bwd.flip([1]);                          // re-align with forward positions

        let h = Tensor::cat(vec![h_fwd, h_bwd], 2);          // [B, L, 2H]
        let h = h.max_dim(1).flatten::<2>(1, 2);              // [B, 2H]
        let h = self.dropout.forward(h);
        self.classifier.forward(h)                            // [B, num_classes]
    }

    pub fn save_pretrained(&self, config: &BiGruConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        let mut json = serde_json::to_value(config).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("arch".to_string(), serde_json::json!("bigru"));
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }

    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, BiGruConfig) {
        let config: BiGruConfig = serde_json::from_str(
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

impl<B: Backend> Classify<B> for BiGru<B> {
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