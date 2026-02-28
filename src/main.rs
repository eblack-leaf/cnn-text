mod data;
mod infer;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray};
use burn::optim::AdamConfig;
use training::{TrainingConfig, train};

type AutodiffBackend = Autodiff<NdArray>;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        // cargo run -- predict <run> "text"
        Some("predict") => {
            let run  = args.get(2).expect("Usage: predict <run> \"<text>\"");
            let text = args.get(3).expect("Usage: predict <run> \"<text>\"");
            let (class, confidence) = infer::predict(text, &format!("artifacts/{run}"));
            println!("{class}  ({:.1}%)", confidence * 100.0);
        }

        // cargo run -- train <run>   (or just cargo run for "default")
        Some("train") | None => {
            let run = args.get(2).map(String::as_str).unwrap_or("default");

            let config = TrainingConfig::new(AdamConfig::new())
                .with_num_epochs(10)
                .with_batch_size(32)
                .with_max_seq_len(128)
                .with_vocab_size(8192);

            train::<AutodiffBackend>(
                "data/dataset.csv",
                &config,
                &format!("artifacts/{run}"),
                Default::default(),
            );
        }

        Some(cmd) => eprintln!("Unknown command: {cmd}. Use 'train' or 'predict'."),
    }
}
