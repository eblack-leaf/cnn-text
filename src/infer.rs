use burn::{
    backend::NdArray,
    tensor::{activation::softmax, Int, Tensor, TensorData},
};
use burn::data::dataset::Dataset;
use std::time::Instant;

use crate::data::{TextDataset, Tokenizer};
use crate::datasets::DatasetKind;
use crate::model::bigru::BiGru;
use crate::model::cnn_text::CnnText;
use crate::model::fast_text::FastText;
use crate::model::kimcnn::KimCnn;
use crate::model::transformer::TinyTransformer;

type B = NdArray;

fn read_arch(model_dir: &str) -> String {
    let raw = std::fs::read_to_string(format!("{model_dir}/config.json"))
        .unwrap_or_else(|_| panic!("No config.json in {model_dir}"));
    let val: serde_json::Value = serde_json::from_str(&raw).unwrap();
    val["arch"].as_str().unwrap_or("fasttext").to_string()
}

/// Predict the class of `text`.
/// Returns `(class_name, confidence_0..1, forward_latency_µs)`.
///
/// Model loading is excluded from the latency measurement — only the forward
/// pass (tokenization → logits) is timed.
pub fn predict(text: &str, model_dir: &str) -> (String, f32, u64) {
    let device = Default::default();
    let arch      = read_arch(model_dir);
    let tokenizer = Tokenizer::load(&format!("{model_dir}/tokenizer.json"));

    let tokens  = tokenizer.encode(text);
    let seq_len = tokens.len();

    // Macro: load model, time only forward, return (logits, class_names, µs).
    macro_rules! run {
        ($model:expr, $names:expr) => {{
            let tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(tokens, [1, seq_len]),
                &device,
            );
            let t0     = Instant::now();
            let logits = $model.forward(tensor);
            (logits, $names, t0.elapsed().as_micros() as u64)
        }};
    }

    let (logits, class_names, latency_us) = match arch.as_str() {
        "cnn-text" => {
            let (m, c) = CnnText::<B>::from_pretrained(model_dir, &device);
            run!(m, c.class_names)
        }
        "kimcnn" => {
            let (m, c) = KimCnn::<B>::from_pretrained(model_dir, &device);
            run!(m, c.class_names)
        }
        "bigru" => {
            let (m, c) = BiGru::<B>::from_pretrained(model_dir, &device);
            run!(m, c.class_names)
        }
        "transformer" => {
            let (m, c) = TinyTransformer::<B>::from_pretrained(model_dir, &device);
            run!(m, c.class_names)
        }
        _ => {
            let (m, c) = FastText::<B>::from_pretrained(model_dir, &device);
            run!(m, c.class_names)
        }
    };

    let probs = softmax(logits, 1).flatten::<1>(0, 1);
    let data  = probs.into_data();
    let probs = data.as_slice::<f32>().unwrap();

    let (idx, &conf) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    (class_names[idx].clone(), conf, latency_us)
}

/// Evaluate a saved model on the val split of `dataset_path`.
///
/// Reproduces the exact same val split used during training (xorshift64, seed=42).
/// Prints per-class precision/recall/F1, macro F1, accuracy, and per-sample
/// forward latency (median + p95, batch=1, model load excluded).
pub fn eval(model_dir: &str, dataset_path: &str, dataset_kind: &DatasetKind, val_ratio: f32) {
    let device    = Default::default();
    let arch      = read_arch(model_dir);
    let tokenizer = Tokenizer::load(&format!("{model_dir}/tokenizer.json"));

    // Reproduce the exact val split from training (same seed=42 shuffle).
    let (pairs, _) = dataset_kind.load(dataset_path);
    let (_, val_ds, class_names) =
        TextDataset::from_pairs_tokenized(pairs, &tokenizer, val_ratio);

    let n_classes = class_names.len();
    let n         = val_ds.len();
    println!("Evaluating on {n} val samples  ({n_classes} classes: {class_names:?})");

    // Run the full val set through the model, recording (true, pred) + latency per sample.
    macro_rules! run_all {
        ($model:expr) => {{
            let model     = $model;
            let mut preds     = Vec::with_capacity(n);
            let mut latencies = Vec::with_capacity(n);
            for i in 0..n {
                let sample  = val_ds.get(i).unwrap();
                let seq_len = sample.tokens.len();
                let tensor  = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(sample.tokens, [1, seq_len]),
                    &device,
                );
                let t0     = Instant::now();
                let logits = model.forward(tensor);
                latencies.push(t0.elapsed().as_micros() as u64);

                let data  = softmax(logits, 1).into_data();
                let probs = data.as_slice::<f32>().unwrap();
                let pred  = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                preds.push((sample.label as usize, pred));
            }
            (preds, latencies)
        }};
    }

    let (preds, mut latencies) = match arch.as_str() {
        "cnn-text"    => { let (m, _) = CnnText::<B>::from_pretrained(model_dir, &device);          run_all!(m) }
        "kimcnn"      => { let (m, _) = KimCnn::<B>::from_pretrained(model_dir, &device);           run_all!(m) }
        "bigru"       => { let (m, _) = BiGru::<B>::from_pretrained(model_dir, &device);            run_all!(m) }
        "transformer" => { let (m, _) = TinyTransformer::<B>::from_pretrained(model_dir, &device);  run_all!(m) }
        _             => { let (m, _) = FastText::<B>::from_pretrained(model_dir, &device);         run_all!(m) }
    };

    // ── Confusion matrix → per-class P/R/F1 ───────────────────────────────────
    // cm[true_label][pred_label]
    let mut cm = vec![vec![0usize; n_classes]; n_classes];
    for &(truth, pred) in &preds {
        cm[truth][pred] += 1;
    }

    println!();
    let mut f1s = Vec::with_capacity(n_classes);
    for c in 0..n_classes {
        let tp              = cm[c][c];
        let fp: usize       = (0..n_classes).filter(|&r| r != c).map(|r| cm[r][c]).sum();
        let fn_: usize      = (0..n_classes).filter(|&r| r != c).map(|r| cm[c][r]).sum();
        let prec            = if tp + fp  > 0 { tp as f64 / (tp + fp)  as f64 } else { 0.0 };
        let recall          = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1              = if prec + recall > 0.0 { 2.0 * prec * recall / (prec + recall) } else { 0.0 };
        let support         = tp + fn_;
        f1s.push(f1);
        println!("  {:>12}  prec={:.3}  rec={:.3}  F1={:.4}  (n={support})",
                 class_names[c], prec, recall, f1);
    }

    let correct  = preds.iter().filter(|&&(t, p)| t == p).count();
    let accuracy = 100.0 * correct as f64 / n as f64;
    let macro_f1 = f1s.iter().sum::<f64>() / f1s.len() as f64;

    // ── Latency percentiles ────────────────────────────────────────────────────
    latencies.sort_unstable();
    let median_us = latencies[latencies.len() / 2];
    let p95_us    = latencies[(latencies.len() * 95) / 100];

    println!();
    println!("  accuracy   {accuracy:.2}%");
    println!("  macro F1   {macro_f1:.4}");
    println!("  latency    median={median_us}µs  p95={p95_us}µs  (forward only, batch=1)");
}
