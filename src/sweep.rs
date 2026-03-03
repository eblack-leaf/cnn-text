// ── Sweep / experiment runner ─────────────────────────────────────────────────
//
// Each [[runs]] block in the TOML defines one architecture with its own sweep
// axes. Only the axes listed in that block are cross-producted; everything else
// uses the global default. This means FastText, KimCNN, and Transformer each
// declare exactly the hyperparameters that matter for them.
//
// The `embed` axis accepts a mixed list of GloVe paths (strings) and BPE dims
// (integers) in the same array, e.g.:
//   embed = ["data/glove.6B.100d.txt", 32, "data/glove.6B.300d.txt", 64]

use serde::Deserialize;
use burn::{
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    tensor::backend::{AutodiffBackend, Backend},
};
use crate::{datasets::DatasetKind, training::{TrainingConfig, train}};

// ── EmbedEntry ────────────────────────────────────────────────────────────────

/// One entry in the `embed` sweep axis.
/// - TOML string  → GloVe file path, e.g. `"data/glove.6B.100d.txt"`
/// - TOML integer → BPE embed dimension, e.g. `64`
#[derive(Clone, Debug)]
pub enum EmbedEntry {
    Glove(String),
    Bpe(usize),
}

impl<'de> Deserialize<'de> for EmbedEntry {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct EmbedVisitor;
        impl<'de> serde::de::Visitor<'de> for EmbedVisitor {
            type Value = EmbedEntry;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a GloVe file path (string) or BPE embed dim (integer)")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<EmbedEntry, E> {
                Ok(EmbedEntry::Glove(v.to_string()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<EmbedEntry, E> {
                Ok(EmbedEntry::Glove(v))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<EmbedEntry, E> {
                Ok(EmbedEntry::Bpe(v as usize))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<EmbedEntry, E> {
                if v < 0 { return Err(E::custom("embed dim cannot be negative")); }
                Ok(EmbedEntry::Bpe(v as usize))
            }
        }
        d.deserialize_any(EmbedVisitor)
    }
}

/// Parse the embed dimension from a GloVe filename, e.g. "glove.6B.100d.txt" → 100.
fn glove_dim_from_path(path: &str) -> usize {
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    name.split(|c: char| !c.is_alphanumeric())
        .find_map(|part| part.strip_suffix('d').and_then(|n| n.parse::<usize>().ok()))
        .unwrap_or(100)
}

impl EmbedEntry {
    /// Resolved embedding dimension (parsed from filename for GloVe).
    pub fn embed_dim(&self) -> usize {
        match self {
            EmbedEntry::Bpe(d) => *d,
            EmbedEntry::Glove(p) => glove_dim_from_path(p),
        }
    }

    /// Short tag used in run names, e.g. "e64" or "g100d".
    pub fn name_tag(&self) -> String {
        match self {
            EmbedEntry::Bpe(d) => format!("e{d}"),
            EmbedEntry::Glove(p) => format!("g{}d", glove_dim_from_path(p)),
        }
    }

    pub fn glove_path(&self) -> Option<&str> {
        match self {
            EmbedEntry::Glove(p) => Some(p.as_str()),
            EmbedEntry::Bpe(_) => None,
        }
    }
}

// ── TOML config structs ───────────────────────────────────────────────────────

fn default_dataset()      -> String { "data/dataset.csv".to_string() }
fn default_dataset_kind() -> String { "agnews".to_string() }
fn default_epochs()       -> usize  { 15 }
fn default_batch()        -> usize  { 128 }
fn default_seq_len()      -> usize  { 64 }
fn default_vocab()        -> usize  { 8192 }
fn default_val_ratio()    -> f32    { 0.15 }
fn default_patience()     -> usize  { 5 }
fn default_filters()      -> usize  { 128 }
fn default_hidden()       -> usize  { 128 }
fn default_heads()        -> usize  { 4 }
fn default_layers()       -> usize  { 2 }
fn default_dff()          -> usize  { 256 }
fn default_attn_dropout() -> f64    { 0.1 }
fn default_dropout()      -> f64    { 0.5 }

/// Global fixed parameters shared by every run in the experiment.
/// Each `[[runs]]` block can override these with per-arch sweep axes.
#[derive(Deserialize)]
pub struct ExperimentConfig {
    #[serde(default = "default_dataset")]
    pub dataset:        String,
    #[serde(default = "default_dataset_kind")]
    pub dataset_kind:   String,

    // ── Training ──────────────────────────────────────────────────────────────
    #[serde(default = "default_epochs")]    pub num_epochs:     usize,
    #[serde(default = "default_batch")]     pub batch_size:     usize,
    #[serde(default = "default_seq_len")]   pub max_seq_len:    usize,
    #[serde(default = "default_vocab")]     pub vocab_size:     usize,
    #[serde(default = "default_val_ratio")] pub val_ratio:      f32,
    #[serde(default = "default_patience")]  pub patience:       usize,
    #[serde(default)]                       pub freeze:         bool,
    #[serde(default)]                       pub bigram_buckets: usize,

    // ── Global arch defaults (used when a [[runs]] block doesn't list them) ──
    #[serde(default = "default_filters")]      pub num_filters:  usize,
    #[serde(default = "default_hidden")]       pub hidden_dim:   usize,
    #[serde(default = "default_heads")]        pub num_heads:    usize,
    #[serde(default = "default_layers")]       pub num_layers:   usize,
    #[serde(default = "default_dff")]          pub d_ff:         usize,
    #[serde(default = "default_attn_dropout")] pub attn_dropout: f64,
    #[serde(default = "default_dropout")]      pub dropout:      f64,

    /// One block per architecture. Each defines its own sweep axes.
    pub runs: Vec<ArchGrid>,
}

/// Per-architecture sweep configuration (`[[runs]]` in TOML).
///
/// Every field is an optional list. A single-element list means "fixed at this
/// value". An empty list (or omitted field) means "use the global default".
/// Multiple elements means "sweep this axis".
///
/// The `embed` axis accepts both strings (GloVe paths) and integers (BPE dims)
/// in the same list:
///   `embed = ["data/glove.6B.100d.txt", 32, "data/glove.6B.300d.txt", 64]`
#[derive(Deserialize)]
pub struct ArchGrid {
    pub arch: String,

    // ── Shared ────────────────────────────────────────────────────────────────
    #[serde(default)] pub embed:         Vec<EmbedEntry>,
    #[serde(default)] pub learning_rate: Vec<f64>,

    // ── KimCNN / BiGRU ────────────────────────────────────────────────────────
    #[serde(default)] pub dropout:       Vec<f64>,
    #[serde(default)] pub num_filters:   Vec<usize>,
    #[serde(default)] pub hidden_dim:    Vec<usize>,

    // ── Transformer ───────────────────────────────────────────────────────────
    #[serde(default)] pub num_heads:       Vec<usize>,
    #[serde(default)] pub num_layers:      Vec<usize>,
    #[serde(default)] pub d_ff:            Vec<usize>,
    #[serde(default)] pub attn_dropout:    Vec<f64>,
    #[serde(default)] pub warmup_steps:    Vec<usize>,

    // ── FastText ──────────────────────────────────────────────────────────────
    #[serde(default)] pub bigram_buckets:  Vec<usize>,
}

// ── Run specification ─────────────────────────────────────────────────────────

pub struct RunSpec {
    pub name:           String,
    pub arch:           String,
    pub embed_dim:      usize,
    pub glove:          Option<String>,
    pub dropout:        f64,
    pub attn_dropout:   f64,
    pub learning_rate:  f64,
    pub warmup_steps:   usize,
    pub num_filters:    usize,
    pub hidden_dim:     usize,
    pub num_heads:      usize,
    pub num_layers:     usize,
    pub d_ff:           usize,
    // Inherited from ExperimentConfig
    pub dataset:        String,
    pub num_epochs:     usize,
    pub batch_size:     usize,
    pub max_seq_len:    usize,
    pub vocab_size:     usize,
    pub val_ratio:      f32,
    pub patience:       usize,
    pub freeze:         bool,
    pub bigram_buckets: usize,
}

// ── Cross-product helpers ─────────────────────────────────────────────────────

/// If `vals` is non-empty use it, otherwise fall back to a single `default`.
fn resolve<T: Clone>(vals: &[T], default: T) -> Vec<T> {
    if vals.is_empty() { vec![default] } else { vals.to_vec() }
}

/// Only include a name tag when the axis has more than one value (i.e. is swept).
fn tag_if_swept<'a, T>(vals: &'a [T], label: &'a str, fmt: impl Fn(&T) -> String + 'a) -> impl Fn(&T) -> String + 'a {
    let swept = vals.len() > 1;
    move |v| if swept { format!("-{label}{}", fmt(v)) } else { String::new() }
}

/// Generate all runs for one `[[runs]]` block.
fn arch_runs(ag: &ArchGrid, cfg: &ExperimentConfig) -> Vec<RunSpec> {
    let arch = &ag.arch;

    let default_lr     = if arch == "transformer" { 1e-4 } else { 1e-3 };
    let default_warmup = if arch == "transformer" { 500  } else { 0    };

    // Embed axis: default to BPE 128 if nothing specified
    let embeds: Vec<EmbedEntry> = if ag.embed.is_empty() {
        vec![EmbedEntry::Bpe(128)]
    } else {
        ag.embed.clone()
    };
    let embed_swept = ag.embed.len() > 1;

    let lrs           = resolve(&ag.learning_rate,  default_lr);
    let dropouts      = resolve(&ag.dropout,        cfg.dropout);
    let attn_drops    = resolve(&ag.attn_dropout,   cfg.attn_dropout);
    let num_filters_v = resolve(&ag.num_filters,    cfg.num_filters);
    let hidden_dims   = resolve(&ag.hidden_dim,     cfg.hidden_dim);
    let num_heads_v   = resolve(&ag.num_heads,      cfg.num_heads);
    let num_layers_v  = resolve(&ag.num_layers,     cfg.num_layers);
    let d_ffs         = resolve(&ag.d_ff,           cfg.d_ff);
    let warmups       = resolve(&ag.warmup_steps,   default_warmup);
    let bigrams       = resolve(&ag.bigram_buckets, cfg.bigram_buckets);

    // Tag closures — only emit a tag when that axis has >1 value
    let lr_tag = tag_if_swept(&ag.learning_rate,  "lr", |v| format!("{v:.0e}"));
    let do_tag = tag_if_swept(&ag.dropout,        "do", |v| format!("{:.0}", v * 100.0));
    let ad_tag = tag_if_swept(&ag.attn_dropout,   "ad", |v| format!("{:.0}", v * 100.0));
    let f_tag  = tag_if_swept(&ag.num_filters,    "f",  |v| format!("{v}"));
    let h_tag  = tag_if_swept(&ag.hidden_dim,     "h",  |v| format!("{v}"));
    let nh_tag = tag_if_swept(&ag.num_heads,      "nh", |v| format!("{v}"));
    let nl_tag = tag_if_swept(&ag.num_layers,     "nl", |v| format!("{v}"));
    let df_tag = tag_if_swept(&ag.d_ff,           "ff", |v| format!("{v}"));
    let ws_tag = tag_if_swept(&ag.warmup_steps,   "w",  |v| format!("{v}"));
    let bg_tag = tag_if_swept(&ag.bigram_buckets, "bg", |v| {
        if *v == 0 { "0".to_string() } else { format!("{}k", v / 1000) }
    });

    let mut runs = Vec::new();

    for embed       in &embeds {
    for &lr         in &lrs {
    for &dropout    in &dropouts {
    for &attn_drop  in &attn_drops {
    for &nf         in &num_filters_v {
    for &hd         in &hidden_dims {
    for &nh         in &num_heads_v {
    for &nl         in &num_layers_v {
    for &df         in &d_ffs {
    for &warmup     in &warmups {
    for &bigram     in &bigrams {
        let embed_tag = if embed_swept {
            format!("-{}", embed.name_tag())
        } else {
            String::new()
        };

        let name = format!("{arch}{}{}{}{}{}{}{}{}{}{}{}",
            embed_tag,    lr_tag(&lr),
            do_tag(&dropout), ad_tag(&attn_drop),
            f_tag(&nf),   h_tag(&hd),
            nh_tag(&nh),  nl_tag(&nl),
            df_tag(&df),  ws_tag(&warmup),
            bg_tag(&bigram),
        );

        runs.push(RunSpec {
            name,
            arch:           arch.clone(),
            embed_dim:      embed.embed_dim(),
            glove:          embed.glove_path().map(|s| s.to_string()),
            dropout,
            attn_dropout:   attn_drop,
            learning_rate:  lr,
            warmup_steps:   warmup,
            num_filters:    nf,
            hidden_dim:     hd,
            num_heads:      nh,
            num_layers:     nl,
            d_ff:           df,
            dataset:        cfg.dataset.clone(),
            num_epochs:     cfg.num_epochs,
            batch_size:     cfg.batch_size,
            max_seq_len:    cfg.max_seq_len,
            vocab_size:     cfg.vocab_size,
            val_ratio:      cfg.val_ratio,
            patience:       cfg.patience,
            freeze:         cfg.freeze,
            bigram_buckets: bigram,
        });
    }}}}}}}}}}}

    runs
}

/// Generate all runs across every `[[runs]]` block in the experiment config.
pub fn generate_runs(cfg: &ExperimentConfig) -> Vec<RunSpec> {
    cfg.runs.iter().flat_map(|ag| arch_runs(ag, cfg)).collect()
}

// ── Parameter counting ────────────────────────────────────────────────────────

pub fn count_params(
    arch:           &str,
    vocab_size:     usize,
    embed_dim:      usize,
    num_classes:    usize,
    num_filters:    usize,
    hidden_dim:     usize,
    num_layers:     usize,
    d_ff:           usize,
    max_seq_len:    usize,
    bigram_buckets: usize,
) -> (usize, usize) {
    let embed_params = vocab_size * embed_dim;

    let non_embed = match arch {
        "fasttext" => {
            embed_dim * num_classes
                + if bigram_buckets > 0 { bigram_buckets * embed_dim } else { 0 }
        }
        "kimcnn" => {
            let conv3 = num_filters * embed_dim * 3 + num_filters;
            let conv4 = num_filters * embed_dim * 4 + num_filters;
            let conv5 = num_filters * embed_dim * 5 + num_filters;
            let cls   = num_filters * 3 * num_classes + num_classes;
            conv3 + conv4 + conv5 + cls
        }
        "bigru" => {
            let gru_one = 3 * (embed_dim * hidden_dim + hidden_dim * hidden_dim + 2 * hidden_dim);
            let cls = hidden_dim * 2 * num_classes + num_classes;
            gru_one * 2 + cls
        }
        "transformer" => {
            let pos_emb = max_seq_len * embed_dim;
            let mha     = 4 * (embed_dim * embed_dim + embed_dim);
            let ffn     = embed_dim * d_ff + d_ff + d_ff * embed_dim + embed_dim;
            let ln      = 4 * embed_dim;
            let cls     = embed_dim * num_classes + num_classes;
            pos_emb + (mha + ffn + ln) * num_layers + cls
        }
        "cnn-text" => {
            let pos_emb = max_seq_len * embed_dim;
            let conv3   = num_filters * embed_dim * 3 + num_filters;
            let conv4   = num_filters * embed_dim * 4 + num_filters;
            let conv5   = num_filters * embed_dim * 5 + num_filters;
            let attn    = 3 * (num_filters + 1);       // three Linear(F, 1) scorers
            let cls     = num_filters * 3 * num_classes + num_classes;
            pos_emb + conv3 + conv4 + conv5 + attn + cls
        }

        _ => 0,
    };

    (non_embed, embed_params)
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1_000.0) }
    else { format!("{n}") }
}

// ── Results ───────────────────────────────────────────────────────────────────

struct SweepResult {
    name:             String,
    arch:             String,
    embed_source:     String,   // "bpe" or glove stem e.g. "glove100d"
    embed_dim:        usize,
    bigram_buckets:   usize,
    dropout:          f64,
    attn_dropout:     f64,
    learning_rate:    f64,
    num_filters:      usize,
    hidden_dim:       usize,
    num_heads:        usize,
    num_layers:       usize,
    d_ff:             usize,
    best_val_acc:     f64,
    best_epoch:       usize,
    non_embed_params: usize,
    embed_params:     usize,
    total_params:     usize,
}

fn read_best_from_metrics(path: &str) -> (f64, usize) {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut best_acc = 0.0f64;
    let mut best_epoch = 0usize;
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 5 { continue; }
        let epoch: usize = cols[0].parse().unwrap_or(0);
        let acc:   f64   = cols[4].parse().unwrap_or(0.0);
        if acc > best_acc { best_acc = acc; best_epoch = epoch; }
    }
    (best_acc, best_epoch)
}

fn read_model_meta(model_dir: &str) -> (usize, usize) {
    let json = std::fs::read_to_string(format!("{model_dir}/config.json")).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap_or_default();
    let vocab  = v["vocab_size"].as_u64().unwrap_or(8192) as usize;
    let nclass = v["class_names"].as_array().map(|a| a.len()).unwrap_or(4);
    (vocab, nclass)
}

// ── Sweep entry point ─────────────────────────────────────────────────────────

pub fn run_sweep<B: AutodiffBackend>(config_path: &str, device: B::Device)
where
    B::InnerBackend: Backend,
{
    let content = std::fs::read_to_string(config_path)
        .unwrap_or_else(|e| panic!("Cannot read config {config_path}: {e}"));
    let cfg: ExperimentConfig = toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Invalid experiment config: {e}"));

    let runs = generate_runs(&cfg);
    println!("Sweep: {} total runs across {} arch block(s)", runs.len(), cfg.runs.len());

    let dataset_kind = DatasetKind::from_str(&cfg.dataset_kind).unwrap_or_else(|| {
        eprintln!(
            "Unknown dataset_kind '{}'. Valid: agnews, sms, imdb. Defaulting to agnews.",
            cfg.dataset_kind,
        );
        DatasetKind::AgNews
    });

    let mut results = Vec::<SweepResult>::new();

    for (i, spec) in runs.iter().enumerate() {
        println!(
            "\n{}\n  Run {}/{}: {}\n{}",
            "─".repeat(72), i + 1, runs.len(), spec.name, "─".repeat(72),
        );

        let model_dir = format!("artifacts/{}/{}", cfg.dataset_kind, spec.name);

        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.01)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));

        let training_cfg = TrainingConfig::new(optimizer)
            .with_num_epochs(spec.num_epochs)
            .with_batch_size(spec.batch_size)
            .with_max_seq_len(spec.max_seq_len)
            .with_vocab_size(spec.vocab_size)
            .with_val_ratio(spec.val_ratio)
            .with_learning_rate(spec.learning_rate)
            .with_warmup_steps(spec.warmup_steps)
            .with_patience(spec.patience)
            .with_freeze_embeddings(spec.freeze)
            .with_bigram_buckets(spec.bigram_buckets)
            .with_embed_dim(spec.embed_dim)
            .with_dropout(spec.dropout)
            .with_attn_dropout(spec.attn_dropout)
            .with_num_filters(spec.num_filters)
            .with_hidden_dim(spec.hidden_dim)
            .with_num_heads(spec.num_heads)
            .with_num_layers(spec.num_layers)
            .with_d_ff(spec.d_ff);

        train::<B>(
            &spec.dataset,
            &dataset_kind,
            spec.glove.as_deref(),
            &training_cfg,
            &spec.arch,
            &model_dir,
            device.clone(),
        );

        let (best_val_acc, best_epoch) =
            read_best_from_metrics(&format!("{model_dir}/metrics.csv"));
        let (actual_vocab, num_classes) = read_model_meta(&model_dir);

        let (non_embed, embed) = count_params(
            &spec.arch, actual_vocab, spec.embed_dim, num_classes,
            spec.num_filters, spec.hidden_dim, spec.num_layers,
            spec.d_ff, spec.max_seq_len, spec.bigram_buckets,
        );

        let embed_source = match &spec.glove {
            Some(p) => format!("glove{}d", glove_dim_from_path(p)),
            None    => "bpe".to_string(),
        };

        results.push(SweepResult {
            name: spec.name.clone(),
            arch: spec.arch.clone(),
            embed_source,
            embed_dim:      spec.embed_dim,
            bigram_buckets: spec.bigram_buckets,
            dropout:        spec.dropout,
            attn_dropout:   spec.attn_dropout,
            learning_rate:  spec.learning_rate,
            num_filters:    spec.num_filters,
            hidden_dim:     spec.hidden_dim,
            num_heads:      spec.num_heads,
            num_layers:     spec.num_layers,
            d_ff:           spec.d_ff,
            best_val_acc, best_epoch,
            non_embed_params: non_embed,
            embed_params:     embed,
            total_params:     non_embed + embed,
        });
    }

    // ── Write CSV ─────────────────────────────────────────────────────────────

    std::fs::create_dir_all(format!("artifacts/{}", cfg.dataset_kind)).ok();
    let results_path = format!("artifacts/{}/sweep_results.csv", cfg.dataset_kind);
    let mut csv = concat!(
        "dataset,name,arch,embed_source,embed_dim,bigram_buckets,",
        "dropout,attn_dropout,learning_rate,",
        "num_filters,hidden_dim,num_heads,num_layers,d_ff,",
        "val_acc,best_epoch,non_embed_params,embed_params,total_params\n",
    ).to_string();
    for r in &results {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{},{},{},{}\n",
            cfg.dataset_kind,
            r.name, r.arch, r.embed_source, r.embed_dim, r.bigram_buckets,
            r.dropout, r.attn_dropout, r.learning_rate,
            r.num_filters, r.hidden_dim, r.num_heads, r.num_layers, r.d_ff,
            r.best_val_acc, r.best_epoch,
            r.non_embed_params, r.embed_params, r.total_params,
        ));
    }
    std::fs::write(&results_path, &csv).unwrap();
    println!("\nResults saved → {results_path}");

    // ── Summary table ─────────────────────────────────────────────────────────

    results.sort_by(|a, b| b.best_val_acc.partial_cmp(&a.best_val_acc).unwrap());

    let sep = "─".repeat(78);
    println!("\n{sep}");
    println!("  Sweep complete — {} runs", results.len());
    println!("{sep}");
    println!("{:<36} {:>6}  {:>5}  {:>9}  {:>9}", "run", "val%", "epoch", "non-embed", "total");
    println!("{:-<36} {:-<6}  {:-<5}  {:-<9}  {:-<9}", "", "", "", "", "");
    for r in &results {
        println!("{:<36} {:>5.2}%  {:>5}  {:>9}  {:>9}",
            r.name, r.best_val_acc, r.best_epoch,
            fmt_params(r.non_embed_params),
            fmt_params(r.total_params),
        );
    }
    println!("{sep}");
}
