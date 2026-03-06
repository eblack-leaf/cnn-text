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

// ── CnnText with Capsule Pooling ───────────────────────────────────────────────
//
//   tokens [B, L]
//     → Embedding          [B, L, E]
//     → permute            [B, E, L]
//     → Conv1d(k=3) → ReLU [B, F, L-2]  ┐
//     → Conv1d(k=4) → ReLU [B, F, L-3]  ├─ capsule pool → [B, n_caps * CAPS_DIM] each
//     → Conv1d(k=5) → ReLU [B, F, L-4]  ┘
//     → cat [B, 3 * n_caps * CAPS_DIM]  →  Dropout  →  Linear [B, C]
//
// Capsule pool (per branch):
//   x [B, F, L']
//     → permute [B, L', F]
//     → Linear(F, n_caps * CAPS_DIM) → reshape [B, L', n_caps, CAPS_DIM]  (u)
//     → dynamic routing (ROUTING_ITERS):
//         b := 0  [B, L', n_caps]
//         for t in 0..ROUTING_ITERS:
//             c  = softmax(b, dim=L')           [B, L', n_caps]
//             s  = Σ_L' c * u                   [B, n_caps, CAPS_DIM]
//             v  = squash(s)                    [B, n_caps, CAPS_DIM]
//             b += u · v  (detach b before add)
//     → flatten v → [B, n_caps * CAPS_DIM]
//
// squash(v) = v * ||v|| / (1 + ||v||²)  maps capsule norm to (0, 1)

/// Length of each output capsule vector.
const CAPS_DIM: usize = 16;
/// Dynamic-routing iterations.
const ROUTING_ITERS: usize = 3;

#[derive(Config, Debug)]
pub struct CnnTextConfig {
    pub vocab_size:  usize,
    pub class_names: Vec<String>,
    #[config(default = 100)]
    pub embed_dim:   usize,
    #[config(default = 128)]
    pub num_filters: usize,
    /// Number of output capsules per conv branch.
    #[config(default = 8)]
    pub num_caps:    usize,
    #[config(default = 0.5)]
    pub dropout:     f64,
    #[config(default = false)]
    pub freeze_embeddings: bool,
}

impl CnnTextConfig {
    pub fn num_classes(&self) -> usize { self.class_names.len() }
    fn caps_out(&self) -> usize { self.num_caps * CAPS_DIM }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnText<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device);
        let embedding = if self.freeze_embeddings { embedding.no_grad() } else { embedding };
        CnnText {
            embedding,
            conv3:      Conv1dConfig::new(self.embed_dim, self.num_filters, 3).init(device),
            conv4:      Conv1dConfig::new(self.embed_dim, self.num_filters, 4).init(device),
            conv5:      Conv1dConfig::new(self.embed_dim, self.num_filters, 5).init(device),
            caps_proj3: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            caps_proj4: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            caps_proj5: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.caps_out() * 3, self.num_classes()).init(device),
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
            conv3:      Conv1dConfig::new(embed_dim, self.num_filters, 3).init(device),
            conv4:      Conv1dConfig::new(embed_dim, self.num_filters, 4).init(device),
            conv5:      Conv1dConfig::new(embed_dim, self.num_filters, 5).init(device),
            caps_proj3: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            caps_proj4: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            caps_proj5: LinearConfig::new(self.num_filters, self.caps_out()).init(device),
            dropout:    DropoutConfig::new(self.dropout).init(),
            classifier: LinearConfig::new(self.caps_out() * 3, self.num_classes()).init(device),
        }
    }

    pub fn save_pretrained<B: Backend>(&self, model: &CnnText<B>, dir: &str) {
        std::fs::create_dir_all(dir).unwrap();
        model.clone()
            .save_file(format!("{dir}/model"), &CompactRecorder::new())
            .unwrap();
        let mut json = serde_json::to_value(self).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("arch".to_string(), serde_json::json!("cnn-text"));
        std::fs::write(
            format!("{dir}/config.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();
    }
}

#[derive(Module, Debug)]
pub struct CnnText<B: Backend> {
    embedding:  Embedding<B>,
    conv3:      Conv1d<B>,
    conv4:      Conv1d<B>,
    conv5:      Conv1d<B>,
    caps_proj3: Linear<B>,   // F → n_caps * CAPS_DIM  (prediction vectors)
    caps_proj4: Linear<B>,
    caps_proj5: Linear<B>,
    dropout:    Dropout,
    classifier: Linear<B>,
}

impl<B: Backend> CnnText<B> {
    /// Capsule squash non-linearity.
    ///
    /// s: [B, n_caps, CAPS_DIM]  →  [B, n_caps, CAPS_DIM]
    /// squash(v) = v * ||v|| / (1 + ||v||²)  — norm is mapped to (0, 1)
    fn squash(s: Tensor<B, 3>) -> Tensor<B, 3> {
        let norm_sq = s.clone().powf_scalar(2.0).sum_dim(2); // [B, n_caps, 1]
        let norm    = norm_sq.clone().sqrt();                 // [B, n_caps, 1]
        s * norm / (norm_sq + 1.0)
    }

    /// Capsule pool a single conv branch via dynamic routing.
    ///
    /// x: [B, F, L']
    ///   → proj → u [B, L', n_caps, CAPS_DIM]
    ///   → ROUTING_ITERS rounds: softmax(b) * u → squash → update b with u·v
    ///   → flatten output capsules v → [B, n_caps * CAPS_DIM]
    fn caps_pool(proj: &Linear<B>, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, _f, l] = x.dims();
        let x_t    = x.swap_dims(1, 2);                          // [B, L', F]
        let u_flat = proj.forward(x_t);                          // [B, L', n_caps * CAPS_DIM]
        let caps_out = u_flat.dims()[2];
        let n_caps   = caps_out / CAPS_DIM;
        let u = u_flat.reshape([b, l, n_caps, CAPS_DIM]);        // [B, L', n_caps, CAPS_DIM]

        let device  = u.device();
        let mut b_ij = Tensor::<B, 3>::zeros([b, l, n_caps], &device); // routing logits
        let mut v    = Tensor::<B, 3>::zeros([b, n_caps, CAPS_DIM], &device);

        for iter in 0..ROUTING_ITERS {
            let c = activation::softmax(b_ij.clone(), 2);        // [B, L', n_caps]
            let s = (c.unsqueeze_dim(3) * u.clone())             // [B, L', n_caps, CAPS_DIM]
                .sum_dim(1)
                .reshape([b, n_caps, CAPS_DIM]);                 // [B, n_caps, CAPS_DIM]
            v = Self::squash(s);

            if iter + 1 < ROUTING_ITERS {
                let agreement = (u.clone() * v.clone().unsqueeze_dim(1)) // [B, L', n_caps, CAPS_DIM]
                    .sum_dim(3)
                    .reshape([b, l, n_caps]);                    // [B, L', n_caps]
                b_ij = b_ij.detach() + agreement;
            }
        }

        v.reshape([b, n_caps * CAPS_DIM])                        // [B, n_caps * CAPS_DIM]
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(tokens).swap_dims(1, 2);   // [B, E, L]

        let c3 = activation::relu(self.conv3.forward(x.clone())); // [B, F, L-2]
        let c4 = activation::relu(self.conv4.forward(x.clone())); // [B, F, L-3]
        let c5 = activation::relu(self.conv5.forward(x));         // [B, F, L-4]

        let p3 = Self::caps_pool(&self.caps_proj3, c3);           // [B, n_caps * CAPS_DIM]
        let p4 = Self::caps_pool(&self.caps_proj4, c4);
        let p5 = Self::caps_pool(&self.caps_proj5, c5);

        let x = self.dropout.forward(Tensor::cat(vec![p3, p4, p5], 1));
        self.classifier.forward(x)
    }

    pub fn save_pretrained(&self, config: &CnnTextConfig, dir: &str) {
        config.save_pretrained(self, dir);
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
