use std::collections::HashMap;
use ahash::AHashMap;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use tokenizers::{
    AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as HfTokenizer,
    TruncationDirection, TruncationParams, TruncationStrategy,
    models::{TrainerWrapper, bpe::{BPE, BpeTrainer}, wordlevel::WordLevel},
    normalizers::utils::Lowercase,
    pre_tokenizers::whitespace::Whitespace,
};

// ── Tokenizer ────────────────────────────────────────────────────────────────

const PAD_TOKEN: &str = "[PAD]"; // id 0
const UNK_TOKEN: &str = "[UNK]"; // id 1

/// BPE tokenizer trained on your corpus.
///
/// Unlike a word-level tokenizer, BPE splits rare/unknown words into known
/// subword pieces instead of `[UNK]`. The embeddings are still trained from
/// scratch — BPE only affects how text is split into ids.
pub struct Tokenizer {
    inner:   HfTokenizer,
    max_len: usize,
}

impl Tokenizer {
    /// Train a BPE tokenizer on `texts` and configure it for fixed-length
    /// padded/truncated encoding.
    pub fn train(texts: &[String], vocab_size: usize, max_len: usize) -> Self {
        let special_tokens = vec![
            AddedToken::from(PAD_TOKEN, true),
            AddedToken::from(UNK_TOKEN, true),
        ];

        let trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(1)
            .special_tokens(special_tokens.clone())
            .build();

        let mut tokenizer = HfTokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        let mut trainer_wrapper: TrainerWrapper = trainer.into();
        tokenizer
            .train(&mut trainer_wrapper, texts.iter().map(String::as_str))
            .unwrap();

        tokenizer.add_special_tokens(&special_tokens);

        tokenizer.with_padding(Some(PaddingParams {
            strategy:           PaddingStrategy::Fixed(max_len),
            direction:          PaddingDirection::Right,
            pad_id:             0,
            pad_type_id:        0,
            pad_token:          PAD_TOKEN.to_string(),
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

    /// Build a word-level tokenizer from an existing vocabulary (e.g. GloVe words).
    ///
    /// `words` should NOT include PAD/UNK — those are inserted at indices 0/1.
    /// A Lowercase normalizer is applied so input text is auto-lowercased at
    /// encode time, matching GloVe's lowercase vocabulary.
    pub fn from_word_vocab(words: &[String], max_len: usize) -> Self {
        let mut vocab: AHashMap<String, u32> = AHashMap::new();
        vocab.insert(PAD_TOKEN.to_string(), 0);
        vocab.insert(UNK_TOKEN.to_string(), 1);
        for (i, word) in words.iter().enumerate() {
            vocab.insert(word.clone(), (i + 2) as u32);
        }

        let mut tokenizer = HfTokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token(UNK_TOKEN.to_string())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer.with_normalizer(Some(Lowercase));
        tokenizer.with_padding(Some(PaddingParams {
            strategy:           PaddingStrategy::Fixed(max_len),
            direction:          PaddingDirection::Right,
            pad_id:             0,
            pad_type_id:        0,
            pad_token:          PAD_TOKEN.to_string(),
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

    pub fn save(&self, path: &str) {
        self.inner.save(path, false).unwrap();
    }

    /// Load a saved tokenizer. Padding/truncation are restored from the JSON.
    pub fn load(path: &str) -> Self {
        let inner = HfTokenizer::from_file(path)
            .unwrap_or_else(|_| panic!("Cannot load tokenizer from {path}"));
        let max_len = inner
            .get_padding()
            .map(|p| match p.strategy { PaddingStrategy::Fixed(n) => n, _ => 128 })
            .unwrap_or(128);
        Self { inner, max_len }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode `text` to a padded/truncated `Vec<i32>` of length `max_len`.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let enc = self.inner.encode(text, false).unwrap();
        assert_eq!(enc.get_ids().len(), self.max_len, "tokenizer padding misconfigured");
        enc.get_ids().iter().map(|&id| id as i32).collect()
    }
}

// ── Dataset ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TextSample {
    pub tokens: Vec<i32>,
    pub label:  i32,
}

#[derive(Clone)]
pub struct TextDataset {
    samples: Vec<TextSample>,
}

impl TextDataset {
    /// Load from a **headerless** CSV with format: `label,text`
    ///
    /// Trains (or loads) a BPE tokenizer saved to `tokenizer_path`.
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

        // Stable label → int mapping (alphabetical order).
        let mut class_names: Vec<String> = raw.iter().map(|(l, _)| l.clone()).collect();
        class_names.sort();
        class_names.dedup();
        let class2idx: HashMap<&str, i32> = class_names
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as i32))
            .collect();

        let texts: Vec<String> = raw.iter().map(|(_, t)| t.clone()).collect();

        let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
            println!("Loading tokenizer from {tokenizer_path}");
            Tokenizer::load(tokenizer_path)
        } else {
            println!("Training BPE tokenizer…");
            if let Some(parent) = std::path::Path::new(tokenizer_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let tok = Tokenizer::train(&texts, vocab_size, max_len);
            tok.save(tokenizer_path);
            println!("Saved tokenizer: vocab_size={}", tok.vocab_size());
            tok
        };

        let mut samples: Vec<TextSample> = raw
            .iter()
            .map(|(label, text)| TextSample {
                tokens: tokenizer.encode(text),
                label:  class2idx[label.as_str()],
            })
            .collect();

        // Deterministic shuffle before split (xorshift, no extra deps).
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
            TextDataset { samples: train_samples },
            TextDataset { samples: val_samples },
            tokenizer,
            class_names,
        )
    }

    /// Load from CSV using a pre-built tokenizer (e.g. from GloVe).
    /// Returns `(train_dataset, val_dataset, class_names)`.
    pub fn from_csv_tokenized(
        path:      &str,
        tokenizer: &Tokenizer,
        val_ratio: f32,
    ) -> (Self, Self, Vec<String>) {
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

        let mut class_names: Vec<String> = raw.iter().map(|(l, _)| l.clone()).collect();
        class_names.sort();
        class_names.dedup();
        let class2idx: HashMap<&str, i32> = class_names
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i as i32))
            .collect();

        let mut samples: Vec<TextSample> = raw
            .iter()
            .map(|(label, text)| TextSample {
                tokens: tokenizer.encode(text),
                label:  class2idx[label.as_str()],
            })
            .collect();

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
            TextDataset { samples: train_samples },
            TextDataset { samples: val_samples },
            class_names,
        )
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
