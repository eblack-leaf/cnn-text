use burn::{
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

#[path = "cnn-text.rs"]
pub(crate) mod cnn_text;
pub(crate) mod transformer;
pub(crate) mod bigru;
pub(crate) mod kimcnn;
#[path = "fast-text.rs"]
pub(crate) mod fast_text;
// ── Shared trait ───────────────────────────────────────────────────────────────

pub trait Classify<B: Backend> {
    fn forward_classification(
        &self,
        tokens:  Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B>;
}

