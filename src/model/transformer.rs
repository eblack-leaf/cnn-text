use burn::prelude::{Backend, Bool, Int, TensorData};
use burn::Tensor;
use burn::train::ClassificationOutput;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, EmbeddingRecord, Linear, LinearConfig};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput};
use burn::record::CompactRecorder;
use burn::config::Config;
use crate::model::Classify;


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
        let x = x.mean_dim(1).flatten::<2>(1, 2);                 // [B, E]

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