#![recursion_limit = "256"]

mod data;
mod datasets;
mod fetch;
mod infer;
mod model;
mod sweep;
mod training;

use burn::backend::Autodiff;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamWConfig;
use serde::Deserialize;
use training::{TrainingConfig, train};

#[cfg(feature = "gpu")]
type AutodiffBackend = Autodiff<burn::backend::Wgpu>;

#[cfg(not(feature = "gpu"))]
type AutodiffBackend = Autodiff<burn::backend::NdArray>;

// ── CLI helpers ───────────────────────────────────────────────────────────────

fn flag_str<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].as_str())
}

fn flag_usize(args: &[String], name: &str) -> Option<usize> {
    flag_str(args, name).and_then(|s| s.parse().ok())
}

fn flag_f64(args: &[String], name: &str) -> Option<f64> {
    flag_str(args, name).and_then(|s| s.parse().ok())
}

fn flag_f32(args: &[String], name: &str) -> Option<f32> {
    flag_str(args, name).and_then(|s| s.parse().ok())
}

fn flag_bool(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

// ── Single-run file config (--config path) ────────────────────────────────────
//
// A flat TOML file that sets defaults for a single training run.
// CLI flags take precedence over file values.

#[derive(Deserialize, Default)]
struct RunFileConfig {
    dataset:        Option<String>,  // path to data file
    dataset_kind:   Option<String>,  // "agnews" | "sms" | "imdb"
    glove:          Option<String>,
    arch:           Option<String>,
    num_epochs:     Option<usize>,
    batch_size:     Option<usize>,
    max_seq_len:    Option<usize>,
    vocab_size:     Option<usize>,
    val_ratio:      Option<f32>,
    learning_rate:  Option<f64>,
    warmup_steps:   Option<usize>,
    patience:       Option<usize>,
    freeze:         Option<bool>,
    bigram_buckets: Option<usize>,
    embed_dim:      Option<usize>,
    dropout:        Option<f64>,
    num_filters:    Option<usize>,
    hidden_dim:     Option<usize>,
    num_heads:      Option<usize>,
    num_layers:     Option<usize>,
    d_ff:           Option<usize>,
}

// ── Build TrainingConfig from file config + CLI args ──────────────────────────

fn build_config(args: &[String], file: &RunFileConfig, arch: &str) -> TrainingConfig {
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

    // Arch-specific defaults (can be overridden by file or CLI)
    let (default_lr, default_warmup) = if arch == "transformer" {
        (1e-4, 500usize)
    } else {
        (1e-3, 0usize)
    };

    // Precedence: CLI > file > arch defaults > struct defaults
    macro_rules! resolve_usize {
        ($cli_flag:literal, $file_field:expr, $fallback:expr) => {
            flag_usize(args, $cli_flag)
                .or($file_field)
                .unwrap_or($fallback)
        };
    }
    macro_rules! resolve_f64 {
        ($cli_flag:literal, $file_field:expr, $fallback:expr) => {
            flag_f64(args, $cli_flag)
                .or($file_field)
                .unwrap_or($fallback)
        };
    }
    macro_rules! resolve_f32 {
        ($cli_flag:literal, $file_field:expr, $fallback:expr) => {
            flag_f32(args, $cli_flag)
                .or($file_field)
                .unwrap_or($fallback)
        };
    }

    let freeze = flag_bool(args, "--freeze") || file.freeze.unwrap_or(false);

    TrainingConfig::new(optimizer)
        .with_num_epochs(   resolve_usize!("--epochs",         file.num_epochs,     15))
        .with_batch_size(   resolve_usize!("--batch-size",     file.batch_size,    128))
        .with_max_seq_len(  resolve_usize!("--max-seq-len",    file.max_seq_len,    64))
        .with_vocab_size(   resolve_usize!("--vocab-size",     file.vocab_size,   8192))
        .with_patience(     resolve_usize!("--patience",       file.patience,        5))
        .with_bigram_buckets(resolve_usize!("--bigram-buckets",file.bigram_buckets,  0))
        .with_embed_dim(    resolve_usize!("--embed-dim",      file.embed_dim,     128))
        .with_num_filters(  resolve_usize!("--num-filters",    file.num_filters,   128))
        .with_hidden_dim(   resolve_usize!("--hidden-dim",     file.hidden_dim,    128))
        .with_num_heads(    resolve_usize!("--num-heads",      file.num_heads,       4))
        .with_num_layers(   resolve_usize!("--num-layers",     file.num_layers,      2))
        .with_d_ff(         resolve_usize!("--d-ff",           file.d_ff,          256))
        .with_warmup_steps( resolve_usize!("--warmup-steps",   file.warmup_steps,  default_warmup))
        .with_learning_rate(resolve_f64!(  "--lr",             file.learning_rate, default_lr))
        .with_dropout(      resolve_f64!(  "--dropout",        file.dropout,       0.5))
        .with_attn_dropout( resolve_f64!(  "--dropout",        file.dropout,       0.1))
        .with_val_ratio(    resolve_f32!(  "--val-ratio",      file.val_ratio,     0.15))
        .with_freeze_embeddings(freeze)
}

// ── Usage ─────────────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!(
r#"Usage:
  fetch-agnews                      Download AG News → data/dataset.csv
  fetch-sms                         Download SMS Spam Collection (UCI)
  fetch-imdb                        Download Stanford IMDB → data/archive/IMDB Dataset.csv
  fetch-glove [50|100|200|300]      Download Stanford GloVe embeddings

  sweep <experiment.toml>           Run full hyperparameter sweep

  train [<model>] [options]         Train a single model
    --config <path>                 Load base hyperparams from TOML file
    --arch <name>                   fasttext|kimcnn|bigru|transformer  [fasttext]
    --dataset <kind>                agnews|sms|imdb  [agnews]
    --data <path>                   Override dataset path
    --glove <path>                  GloVe file (omit to use BPE from scratch)
    --epochs N                      Training epochs  [15]
    --batch-size N                  Batch size  [128]
    --lr F                          Learning rate  [1e-3; 1e-4 for transformer]
    --warmup-steps N                LR warmup steps  [0; 500 for transformer]
    --max-seq-len N                 Sequence length (pad/truncate)  [64]
    --vocab-size N                  BPE vocabulary size  [8192]
    --val-ratio F                   Validation split fraction  [0.15]
    --patience N                    Early stopping patience (0=off)  [5]
    --embed-dim N                   Embedding dimension  [128]
    --dropout F                     Dropout probability  [0.5]
    --num-filters N                 Kim CNN: filters per kernel  [128]
    --hidden-dim N                  BiGRU: hidden size per direction  [128]
    --num-heads N                   Transformer: attention heads  [4]
    --num-layers N                  Transformer: encoder layers  [2]
    --d-ff N                        Transformer: FFN hidden dim  [256]
    --bigram-buckets N              FastText: bigram hash buckets (0=off)  [0]
    --freeze                        Freeze embedding weights

  predict <model> "<text>"          Run inference on a saved model
"#
    );
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {

        // ── fetch-agnews ──────────────────────────────────────────────────────
        Some("fetch-agnews") => {
            fetch::agnews();
        }

        // ── fetch-sms ─────────────────────────────────────────────────────────
        Some("fetch-sms") => {
            fetch::sms();
        }

        // ── fetch-imdb ────────────────────────────────────────────────────────
        Some("fetch-imdb") => {
            fetch::imdb();
        }

        // ── fetch-glove [dim] ─────────────────────────────────────────────────
        Some("fetch-glove") => {
            let dim: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            fetch::glove(dim);
        }

        // ── predict <model> "text" ─────────────────────────────────────────────
        Some("predict") => {
            let model = args.get(2).expect("Usage: predict <model> \"<text>\"");
            let text  = args.get(3).expect("Usage: predict <model> \"<text>\"");
            let (class, confidence) = infer::predict(text, &format!("artifacts/{model}"));
            println!("{class}  ({:.1}%)", confidence * 100.0);
        }

        // ── sweep <experiment.toml> ───────────────────────────────────────────
        Some("sweep") => {
            let config_path = args.get(2).unwrap_or_else(|| {
                eprintln!("Usage: sweep <experiment.toml>");
                std::process::exit(1);
            });
            sweep::run_sweep::<AutodiffBackend>(config_path, Default::default());
        }

        // ── train [<model>] [options] ─────────────────────────────────────────
        Some("train") | None => {
            let model = args.get(2)
                .filter(|a| !a.starts_with("--"))
                .map(String::as_str)
                .unwrap_or("default");

            // Optional file config (--config path)
            let file_cfg: RunFileConfig = flag_str(&args, "--config")
                .map(|path| {
                    let content = std::fs::read_to_string(path)
                        .unwrap_or_else(|e| panic!("Cannot read config {path}: {e}"));
                    toml::from_str(&content)
                        .unwrap_or_else(|e| panic!("Invalid config file: {e}"))
                })
                .unwrap_or_default();

            let arch = flag_str(&args, "--arch")
                .or(file_cfg.arch.as_deref())
                .unwrap_or("fasttext");

            let glove = flag_str(&args, "--glove")
                .or(file_cfg.glove.as_deref());

            let dataset_name = flag_str(&args, "--dataset")
                .or(file_cfg.dataset_kind.as_deref())
                .unwrap_or("agnews");
            let dataset_kind = datasets::DatasetKind::from_str(dataset_name)
                .unwrap_or_else(|| {
                    eprintln!("Unknown dataset '{dataset_name}'. Valid: agnews, sms, imdb");
                    std::process::exit(1);
                });

            let dataset_path = flag_str(&args, "--data")
                .or(file_cfg.dataset.as_deref())
                .unwrap_or_else(|| dataset_kind.default_path());

            let config = build_config(&args, &file_cfg, arch);

            train::<AutodiffBackend>(
                dataset_path,
                &dataset_kind,
                glove,
                &config,
                arch,
                &format!("artifacts/{model}"),
                Default::default(),
            );
        }

        Some("--help") | Some("-h") | Some("help") => print_usage(),

        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            print_usage();
            std::process::exit(1);
        }
    }
}
