use burn::prelude::{Backend, Int, TensorData};
use burn::Tensor;
use burn::train::ClassificationOutput;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig};
use burn::module::{Module, Param, ParamId};
use burn::record::CompactRecorder;
use burn::config::Config;
use crate::model::Classify;

/// Hash prime for bigram IDs. Small enough to avoid i32 overflow for vocab ≤ 60k.
const BIGRAM_PRIME: i32 = 9973;
// ── FastText ───────────────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → Embedding [B, L, E]
//     → mean over L [B, E]
//     → Linear [B, num_classes]


#[derive(Config, Debug)]
pub struct FastTextConfig {
    pub vocab_size:        usize,
    pub class_names:       Vec<String>,
    #[config(default = 100)]
    pub embed_dim:         usize,
    /// Number of bigram hash buckets. 0 = no bigrams (backwards-compatible default).
    #[config(default = 0)]
    pub bigram_buckets:    usize,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl FastTextConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }

    fn make_bigrams<B: Backend>(&self, embed_dim: usize, device: &B::Device) -> Option<Embedding<B>> {
        if self.bigram_buckets > 0 {
            Some(EmbeddingConfig::new(self.bigram_buckets, embed_dim).init(device))
        } else {
            None
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FastText<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        FastText {
            embedding,
            bigrams: self.make_bigrams::<B>(self.embed_dim, device),
            classifier: LinearConfig::new(self.embed_dim, self.num_classes()).init(device),
        }
    }

    pub fn init_with_embeddings<B: Backend>(
        &self,
        device:  &B::Device,
        vectors: &[Vec<f32>],
    ) -> FastText<B> {
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
        FastText {
            embedding,
            bigrams: self.make_bigrams::<B>(embed_dim, device),
            classifier: LinearConfig::new(embed_dim, self.num_classes()).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct FastText<B: Backend> {
    embedding:  Embedding<B>,
    /// Bigram hash table. None for models trained without bigrams.
    bigrams:    Option<Embedding<B>>,
    classifier: Linear<B>,
}

impl<B: Backend> FastText<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch_size, seq_len] = tokens.dims();

        // mask_2d: [B, L] float — 1.0 for real tokens, 0.0 for PAD (id=0)
        let mask_2d = tokens.clone().not_equal_elem(0).float();
        // mask_3d: [B, L, 1] — for broadcasting against embeddings [B, L, E]
        let mask_3d = mask_2d.clone().unsqueeze_dim::<3>(2);

        let tok_emb = self.embedding.forward(tokens.clone()); // [B, L, E]

        let x = if let Some(bigrams) = &self.bigrams {
            // bigram_id = (tok_i * PRIME + tok_{i+1}).abs() % n_buckets
            let n = bigrams.weight.val().dims()[0] as i32;
            let left  = tokens.clone().slice([0..batch_size, 0..seq_len - 1]);
            let right = tokens.slice([0..batch_size, 1..seq_len]);
            let ids   = left.mul_scalar(BIGRAM_PRIME).add(right).abs().remainder_scalar(n);
            let big_emb = bigrams.forward(ids);                          // [B, L-1, E]

            let left_mask   = mask_2d.clone().slice([0..batch_size, 0..seq_len - 1]);
            let right_mask  = mask_2d.clone().slice([0..batch_size, 1..seq_len]);
            let big_mask_2d = left_mask * right_mask;                          // [B, L-1]
            let big_mask_3d = big_mask_2d.clone().unsqueeze_dim::<3>(2);      // [B, L-1, 1]

            let emb = Tensor::cat(
                vec![tok_emb * mask_3d, big_emb * big_mask_3d], 1,       // [B, 2L-1, E]
            );
            // counts: sum [B, L] + sum [B, L-1] → [B, 1] + [B, 1] = [B, 1]
            let counts = (mask_2d.clone().sum_dim(1) + big_mask_2d.sum_dim(1)).clamp_min(1.0);
            emb.sum_dim(1).flatten::<2>(1, 2) / counts                    // [B, E]
        } else {
            // counts: sum [B, L] → [B, 1]
            let counts = mask_2d.sum_dim(1).clamp_min(1.0);              // [B, 1]
            (tok_emb * mask_3d).sum_dim(1).flatten::<2>(1, 2) / counts   // [B, E]
        };

        self.classifier.forward(x)
    }

    pub fn save_pretrained(&self, config: &FastTextConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(config).unwrap(),
        )
        .unwrap();
    }

    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, FastTextConfig) {
        let config: FastTextConfig = serde_json::from_str(
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

impl<B: Backend> Classify<B> for FastText<B> {
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