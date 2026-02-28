use burn::{
    backend::NdArray,
    tensor::{Int, Tensor, TensorData, activation::softmax},
};

use crate::{data::Tokenizer, model::TextCnn};

type B = NdArray;

/// Predict the class of `text`.
/// Returns `(class_name, confidence_0..1)`.
pub fn predict(text: &str, model_dir: &str) -> (String, f32) {
    let device = Default::default();

    let (model, config) = TextCnn::<B>::from_pretrained(model_dir, &device);
    let tokenizer = Tokenizer::load(&format!("{model_dir}/tokenizer.json"));

    let tokens = tokenizer.encode(text);
    let seq_len = tokens.len();
    let token_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(tokens, [1, seq_len]),
        &device,
    );

    let probs = softmax(model.forward(token_tensor), 1) // [1, num_classes]
        .flatten::<1>(0, 1);                             // [num_classes]

    let prob_data = probs.into_data();
    let prob_vec  = prob_data.as_slice::<f32>().unwrap();

    let idx = prob_vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    (config.class_names[idx].clone(), prob_vec[idx])
}
