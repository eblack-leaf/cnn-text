use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, TensorData, activation::softmax},
};

use crate::{data::Tokenizer, model::{BiGru, FastText, KimCnn, TinyTransformer}};

type B = NdArray;

/// Predict the class of `text`. Returns `(class_name, confidence_0..1)`.
///
/// Detects the model architecture from `config.json` automatically:
/// KimCnn saves `"arch": "kimcnn"`; anything else defaults to FastText.
pub fn predict(text: &str, model_dir: &str) -> (String, f32) {
    let device = Default::default();

    // Detect arch from the saved config.
    let config_raw = std::fs::read_to_string(format!("{model_dir}/config.json"))
        .unwrap_or_else(|_| panic!("No config.json in {model_dir}"));
    let config_val: serde_json::Value = serde_json::from_str(&config_raw).unwrap();
    let arch = config_val["arch"].as_str().unwrap_or("fasttext");

    let tokenizer = Tokenizer::load(&format!("{model_dir}/tokenizer.json"));

    let tokens  = tokenizer.encode(text);
    let seq_len = tokens.len();
    let tensor  = Tensor::<B, 2, Int>::from_data(
        TensorData::new(tokens, [1, seq_len]),
        &device,
    );

    let (logits, class_names) = match arch {
        "kimcnn" => {
            let (model, cfg) = KimCnn::<B>::from_pretrained(model_dir, &device);
            (model.forward(tensor), cfg.class_names)
        }
        "bigru" => {
            let (model, cfg) = BiGru::<B>::from_pretrained(model_dir, &device);
            (model.forward(tensor), cfg.class_names)
        }
        "transformer" => {
            let (model, cfg) = TinyTransformer::<B>::from_pretrained(model_dir, &device);
            (model.forward(tensor), cfg.class_names)
        }
        _ => {
            let (model, cfg) = FastText::<B>::from_pretrained(model_dir, &device);
            (model.forward(tensor), cfg.class_names)
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

    (class_names[idx].clone(), conf)
}
