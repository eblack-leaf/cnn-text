mod data;
mod fetch;
mod infer;
mod model;
mod training;

use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use training::{TrainingConfig, train};

#[cfg(feature = "gpu")]
type AutodiffBackend = Autodiff<burn::backend::Wgpu>;

#[cfg(not(feature = "gpu"))]
type AutodiffBackend = Autodiff<burn::backend::NdArray>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        // cargo run -- fetch-agnews
        Some("fetch-agnews") => {
            fetch::agnews();
        }

        // cargo run -- fetch-glove [50|100|200|300]
        Some("fetch-glove") => {
            let dim: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            fetch::glove(dim);
        }

        // cargo run -- predict <model> "text"
        Some("predict") => {
            let model = args.get(2).expect("Usage: predict <model> \"<text>\"");
            let text  = args.get(3).expect("Usage: predict <model> \"<text>\"");
            let (class, confidence) = infer::predict(text, &format!("artifacts/{model}"));
            println!("{class}  ({:.1}%)", confidence * 100.0);
        }

        // cargo run -- train [<model>] [--glove <path>] [--freeze]
        Some("train") | None => {
            let model = args.get(2)
                .filter(|a| !a.starts_with("--"))
                .map(String::as_str)
                .unwrap_or("default");

            let glove = args.windows(2)
                .find(|w| w[0] == "--glove")
                .map(|w| w[1].as_str());

            let freeze = args.iter().any(|a| a == "--freeze");

            let config = TrainingConfig::new(AdamConfig::new())
                .with_num_epochs(10)
                .with_batch_size(32)
                .with_max_seq_len(128)
                .with_vocab_size(8192)
                .with_freeze_embeddings(freeze);

            train::<AutodiffBackend>(
                "data/dataset.csv",
                glove,
                &config,
                &format!("artifacts/{model}"),
                Default::default(),
            );
        }

        Some(cmd) => eprintln!(
            "Unknown command: {cmd}\nUsage: fetch-agnews | fetch-glove [dim] | train [model] | predict <model> \"<text>\""
        ),
    }
}
