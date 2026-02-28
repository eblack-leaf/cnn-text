use std::collections::HashMap;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use serde::{Deserialize, Serialize};
use tokenizers::{
    AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as HfTokenizer,
    TruncationDirection, TruncationParams, TruncationStrategy,
    models::{
        TrainerWrapper,
        bpe::{BPE, BpeTrainer},
    },
    pre_tokenizers::whitespace::Whitespace,
};

// ── Tokenizer ────────────────────────────────────────────────────────────────

pub const PAD_TOKEN: &str = "[PAD]"; // id = 0
pub const UNK_TOKEN: &str = "[UNK]"; // id = 1

/// BPE tokenizer trained on your corpus.
///
/// Unlike a word-level tokenizer, BPE splits unknown/rare words into
/// known subword pieces instead of mapping them to `[UNK]`. For example
/// "spamming" → ["spam", "ming"] if "spamming" is rare but "spam" is common.
///
/// The embeddings in the CNN are still randomly initialised and trained
/// from scratch — the tokenizer only affects *how* text is split into IDs.
pub struct Tokenizer {
    inner:   HfTokenizer,
    max_len: usize,
}

impl Tokenizer {
    /// Train a BPE tokenizer on `texts` and configure it for fixed-length
    /// padded/truncated encoding.
    pub fn train(texts: &[String], vocab_size: usize, max_len: usize) -> Self {
        let special_tokens = vec![
            AddedToken::from(PAD_TOKEN, true), // id 0
            AddedToken::from(UNK_TOKEN, true), // id 1
        ];

        let trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(1)
            .special_tokens(special_tokens.clone())
            .build();

        let mut tokenizer = HfTokenizer::new(BPE::default());
        // Split on whitespace before applying BPE merges.
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        // `Tokenizer` stores a `ModelWrapper`, so the trainer must be wrapped too.
        let mut trainer_wrapper: TrainerWrapper = trainer.into();
        tokenizer
            .train(&mut trainer_wrapper, texts.iter().map(String::as_str))
            .unwrap();

        // Add special tokens so they're always recognised even if not in training data.
        tokenizer.add_special_tokens(&special_tokens);

        // Configure automatic padding (right-pad with [PAD]) and truncation.
        tokenizer.with_padding(Some(PaddingParams {
            strategy:          PaddingStrategy::Fixed(max_len),
            direction:         PaddingDirection::Right,
            pad_id:            0,
            pad_type_id:       0,
            pad_token:         PAD_TOKEN.to_string(),
            pad_to_multiple_of: None,
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                direction:  TruncationDirection::Right,
                stride:     0,
                strategy:   TruncationStrategy::LongestFirst,
            }))
            .unwrap();

        Self { inner: tokenizer, max_len }
    }

    /// Save tokenizer to a JSON file (so you don't retrain on every run).
    pub fn save(&self, path: &str) {
        self.inner.save(path, false).unwrap();
    }

    /// Load a previously saved tokenizer (use this at inference time).
    #[allow(dead_code)]
    pub fn load(path: &str, max_len: usize) -> Self {
        let mut inner = HfTokenizer::from_file(path).unwrap();
        inner.with_padding(Some(PaddingParams {
            strategy:          PaddingStrategy::Fixed(max_len),
            direction:         PaddingDirection::Right,
            pad_id:            0,
            pad_type_id:       0,
            pad_token:         PAD_TOKEN.to_string(),
            pad_to_multiple_of: None,
        }));
        inner
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                direction:  TruncationDirection::Right,
                stride:     0,
                strategy:   TruncationStrategy::LongestFirst,
            }))
            .unwrap();
        Self { inner, max_len }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode `text` to a padded/truncated `Vec<i32>` of length `max_len`.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let enc = self.inner.encode(text, false).unwrap();
        // with_padding + with_truncation guarantee exactly max_len ids.
        assert_eq!(enc.get_ids().len(), self.max_len, "tokenizer padding misconfigured");
        enc.get_ids().iter().map(|&id| id as i32).collect()
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
    samples:         Vec<TextSample>,
    #[allow(dead_code)]
    pub num_classes: usize,
}

impl TextDataset {
    /// Load from a **headerless** CSV with format: `label,text`
    ///
    /// Trains a BPE tokenizer on the corpus texts and saves it to
    /// `{artifact_dir}/tokenizer.json` so you can reload it at inference time.
    ///
    /// Returns `(train_dataset, val_dataset, tokenizer, class_names)`.
    pub fn from_csv(
        path:           &str,
        max_len:        usize,
        val_ratio:      f32,
        vocab_size:     usize,
        tokenizer_path: &str,
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

        // Stable label → int mapping (alphabetical).
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

        // Load existing tokenizer if available; train and save otherwise.
        // tokenizer_path lives next to the data (committed), not in artifacts.
        let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
            println!("Loading tokenizer from {tokenizer_path}");
            Tokenizer::load(tokenizer_path, max_len)
        } else {
            println!("Training BPE tokenizer…");
            let tok = Tokenizer::train(&texts, vocab_size, max_len);
            if let Some(parent) = std::path::Path::new(tokenizer_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            tok.save(tokenizer_path);
            println!("BPE tokenizer trained and saved: vocab_size={}", tok.vocab_size());
            tok
        };

        let mut samples: Vec<TextSample> = raw
            .iter()
            .map(|(label, text)| TextSample {
                tokens: tokenizer.encode(text),
                label:  class2idx[label.as_str()],
            })
            .collect();

        // Deterministic shuffle (xorshift, no extra deps).
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

/// Stateless batcher — device is provided by the DataLoader in Burn 0.20.
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
