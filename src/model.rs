use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        gru::{Gru, GruConfig},
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Dropout, DropoutConfig,
        Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
    },
    record::CompactRecorder,
    tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend},
    train::ClassificationOutput,
};

// ── Shared trait ───────────────────────────────────────────────────────────────

pub trait Classify<B: Backend> {
    fn forward_classification(
        &self,
        tokens:  Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B>;
}

// ── FastText ───────────────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → Embedding [B, L, E]
//     → mean over L [B, E]
//     → Linear [B, num_classes]

/// Hash prime for bigram IDs. Small enough to avoid i32 overflow for vocab ≤ 60k.
const BIGRAM_PRIME: i32 = 9973;

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
        let tok_emb = self.embedding.forward(tokens.clone()); // [B, L, E]

        let x = if let Some(bigrams) = &self.bigrams {
            // bigram_id = (tok_i * PRIME + tok_{i+1}).abs() % n_buckets
            let n = bigrams.weight.val().dims()[0] as i32;
            let left  = tokens.clone().slice([0..batch_size, 0..seq_len - 1]);
            let right = tokens.slice([0..batch_size, 1..seq_len]);
            let ids   = left.mul_scalar(BIGRAM_PRIME).add(right).abs().remainder_scalar(n);
            let big_emb = bigrams.forward(ids);                // [B, L-1, E]
            Tensor::cat(vec![tok_emb, big_emb], 1)            // [B, 2L-1, E]
                .mean_dim(1).squeeze::<2>()                    // [B, E]
        } else {
            tok_emb.mean_dim(1).squeeze::<2>()                 // [B, E]
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
        c.max_dim(2).squeeze::<2>()                // [B, F]
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
        let h = h.max_dim(1).squeeze::<2>();                  // [B, 2H]
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

// ── Tiny Transformer ───────────────────────────────────────────────────────────
//
//   tokens [B, L]
//     → TokenEmbedding [B, L, E]  +  PosEmbedding [1, L, E]
//     → TransformerEncoder (n_layers × MultiheadAttn + FFN + LN)  [B, L, E]
//     → Mean pool over L (PAD-masked attention keeps PAD reps near zero)  [B, E]
//     → Dropout
//     → Linear  [B, num_classes]
//
// Notes:
//   • embed_dim must be divisible by num_heads (default 100 / 4 = 25 ✓).
//   • 50-d GloVe requires num_heads = 2.

#[derive(Config, Debug)]
pub struct TinyTransformerConfig {
    pub vocab_size:  usize,
    pub class_names: Vec<String>,
    /// Must match max_seq_len used during tokenization.
    pub max_seq_len: usize,
    /// Token embedding / transformer d_model. Must be divisible by num_heads.
    #[config(default = 100)]
    pub embed_dim:   usize,
    #[config(default = 4)]
    pub num_heads:   usize,
    #[config(default = 2)]
    pub num_layers:  usize,
    /// FFN hidden dim inside each transformer layer.
    #[config(default = 256)]
    pub d_ff:        usize,
    /// Dropout used inside transformer layers and before the classifier.
    #[config(default = 0.1)]
    pub dropout:     f64,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl TinyTransformerConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }

    fn build<B: Backend>(
        &self,
        tok_embedding: Embedding<B>,
        embed_dim: usize,
        device: &B::Device,
    ) -> TinyTransformer<B> {
        TinyTransformer {
            tok_embedding,
            pos_embedding: EmbeddingConfig::new(self.max_seq_len, embed_dim).init(device),
            transformer: TransformerEncoderConfig::new(
                embed_dim, self.d_ff, self.num_heads, self.num_layers,
            )
            .with_dropout(self.dropout)
            .init(device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(embed_dim, self.num_classes()).init(device),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyTransformer<B> {
        let emb = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let emb = if self.freeze_embeddings { emb.no_grad() } else { emb };
        self.build(emb, self.embed_dim, device)
    }

    pub fn init_with_embeddings<B: Backend>(
        &self,
        device:  &B::Device,
        vectors: &[Vec<f32>],
    ) -> TinyTransformer<B> {
        let vocab_size = vectors.len();
        let embed_dim  = vectors[0].len();
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let weight = Tensor::<B, 2>::from_data(
            TensorData::new(flat, [vocab_size, embed_dim]),
            device,
        );
        let emb = EmbeddingConfig::new(vocab_size, embed_dim)
            .init(device)
            .load_record(EmbeddingRecord { weight: Param::initialized(ParamId::new(), weight) });
        let emb = if self.freeze_embeddings { emb.no_grad() } else { emb };
        self.build(emb, embed_dim, device)
    }
}

#[derive(Module, Debug)]
pub struct TinyTransformer<B: Backend> {
    tok_embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    transformer:   TransformerEncoder<B>,
    dropout:       Dropout,
    classifier:    Linear<B>,
}

impl<B: Backend> TinyTransformer<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [_batch_size, seq_len] = tokens.dims();
        let device = tokens.device();

        // Token embeddings + learned positional embeddings.
        let tok_emb = self.tok_embedding.forward(tokens.clone());  // [B, L, E]

        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>();                                     // [1, L]
        let pos_emb = self.pos_embedding.forward(pos_ids);        // [1, L, E]

        let x = tok_emb + pos_emb;                                // [B, L, E] broadcast

        // Transformer with padding-token attention mask.
        let pad_mask: Tensor<B, 2, Bool> = tokens.equal_elem(0i32); // [B, L]
        let x = self.transformer.forward(
            TransformerEncoderInput::new(x).mask_pad(pad_mask),
        );                                                         // [B, L, E]

        // Mean pool over all positions (PAD reps are ~0 due to masking).
        let x = x.mean_dim(1).squeeze::<2>();                     // [B, E]

        let x = self.dropout.forward(x);
        self.classifier.forward(x)                                // [B, num_classes]
    }

    pub fn save_pretrained(&self, config: &TinyTransformerConfig, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        self.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        let mut json = serde_json::to_value(config).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("arch".to_string(), serde_json::json!("transformer"));
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }

    pub fn from_pretrained(dir: &str, device: &B::Device) -> (Self, TinyTransformerConfig) {
        let config: TinyTransformerConfig = serde_json::from_str(
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

impl<B: Backend> Classify<B> for TinyTransformer<B> {
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
