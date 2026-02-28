use std::collections::HashMap;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};

// ── Tokenizer ────────────────────────────────────────────────────────────────

pub const PAD_TOKEN: &str = "<PAD>";  // id = 0
pub const UNK_TOKEN: &str = "<UNK>";  // id = 1

/// Simple whitespace tokenizer backed by a frequency-pruned vocabulary.
pub struct Tokenizer {
    word2idx: HashMap<String, usize>,
}

impl Tokenizer {
    /// Build vocabulary from `texts`.
    /// Words appearing fewer than `min_freq` times are mapped to `<UNK>`.
    pub fn build(texts: &[String], min_freq: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for w in text.split_whitespace() {
                *freq.entry(w.to_lowercase()).or_default() += 1;
            }
        }

        let mut word2idx = HashMap::new();
        word2idx.insert(PAD_TOKEN.to_string(), 0usize);
        word2idx.insert(UNK_TOKEN.to_string(), 1usize);

        let mut idx = 2usize;
        for (word, count) in freq {
            if count >= min_freq {
                word2idx.insert(word, idx);
                idx += 1;
            }
        }
        Self { word2idx }
    }

    pub fn vocab_size(&self) -> usize {
        self.word2idx.len()
    }

    /// Encode `text` into a padded/truncated sequence of `i32` token ids.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<i32> {
        let mut ids: Vec<i32> = text
            .split_whitespace()
            .map(|w| *self.word2idx.get(&w.to_lowercase()).unwrap_or(&1) as i32)
            .take(max_len)
            .collect();
        ids.resize(max_len, 0); // right-pad with PAD
        ids
    }
}

// ── Dataset ──────────────────────────────────────────────────────────────────

/// A single labeled example ready for batching.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextSample {
    pub tokens: Vec<i32>, // length == max_seq_len
    pub label:  i32,
}

/// In-memory labeled text dataset.
pub struct TextDataset {
    samples:             Vec<TextSample>,
    #[allow(dead_code)]  // read by callers via the returned class_names vec
    pub num_classes:     usize,
}

impl TextDataset {
    /// Load from a **headerless** CSV with format: `label,text`
    ///
    /// Labels are arbitrary strings — mapped to contiguous integers alphabetically
    /// so the mapping is deterministic regardless of row order.
    ///
    /// Returns `(train_dataset, val_dataset, tokenizer, class_names)`.
    pub fn from_csv(
        path:      &str,
        max_len:   usize,
        val_ratio: f32,
        min_freq:  usize,
    ) -> (Self, Self, Tokenizer, Vec<String>) {
        let content = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));

        let raw: Vec<(String, String)> = content
            .lines()
            .filter_map(|line| {
                let mut parts = line.splitn(2, ',');
                let label = parts.next()?.trim().to_string();
                let text  = parts.next()?.trim().to_string();
                if label.is_empty() || text.is_empty() { return None; }
                Some((label, text))
            })
            .collect();

        // Build a stable label → int mapping (alphabetical).
        let mut class_names: Vec<String> = raw.iter().map(|(l, _)| l.clone()).collect();
        class_names.sort();
        class_names.dedup();
        let num_classes = class_names.len();
        let class2idx: HashMap<&str, i32> = class_names
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as i32))
            .collect();

        let texts: Vec<String> = raw.iter().map(|(_, t)| t.clone()).collect();
        let tokenizer = Tokenizer::build(&texts, min_freq);

        let mut samples: Vec<TextSample> = raw
            .iter()
            .map(|(label, text)| TextSample {
                tokens: tokenizer.encode(text, max_len),
                label:  class2idx[label.as_str()],
            })
            .collect();

        // Deterministic shuffle before split (simple xorshift, no extra deps).
        let mut rng: u64 = 42;
        for i in (1..samples.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            samples.swap(i, j);
        }

        let n_val = ((samples.len() as f32) * val_ratio).round() as usize;
        let val_samples   = samples[..n_val].to_vec();
        let train_samples = samples[n_val..].to_vec();

        (
            TextDataset { samples: train_samples, num_classes },
            TextDataset { samples: val_samples,   num_classes },
            tokenizer,
            class_names,
        )
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

impl Dataset<TextSample> for TextDataset {
    fn get(&self, index: usize) -> Option<TextSample> {
        self.samples.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.samples.len()
    }
}

// ── Batcher ──────────────────────────────────────────────────────────────────

/// Collated batch passed to the model.
#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    /// Shape: `[batch_size, seq_len]`
    pub tokens: Tensor<B, 2, Int>,
    /// Shape: `[batch_size]`
    pub labels: Tensor<B, 1, Int>,
}

/// Batcher for `TextSample → TextBatch`.
///
/// In Burn 0.20, `Batcher` receives the device from the DataLoader —
/// the batcher itself is stateless.
#[derive(Clone, Default)]
pub struct TextBatcher;

impl<B: Backend> Batcher<B, TextSample, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<TextSample>, device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();
        let seq_len    = items[0].tokens.len();

        let token_flat: Vec<i32> = items.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let label_flat: Vec<i32> = items.iter().map(|s| s.label).collect();

        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(token_flat, [batch_size, seq_len]),
            device,
        );
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(label_flat, [batch_size]),
            device,
        );

        TextBatch { tokens, labels }
    }
}
