//! MiniLM sentence embeddings via ONNX Runtime — reproduces
//! sentence-transformers/all-MiniLM-L6-v2: BERT encoder -> mean pooling over
//! the attention mask -> L2 normalize. Model files are fetched from the
//! Hugging Face Hub on first use and cached locally.

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub const MODEL_REPO: &str = "Xenova/all-MiniLM-L6-v2";
pub const MAX_SEQ_LENGTH: usize = 256;

pub struct Embedder {
    tokenizer: Tokenizer,
    session: Session,
    needs_token_type_ids: bool,
}

impl Embedder {
    /// `quantized: false` (fp32) matches the reference implementation.
    pub fn new(quantized: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(MODEL_REPO.to_string());
        let onnx_file = if quantized {
            "onnx/model_quantized.onnx"
        } else {
            "onnx/model.onnx"
        };
        let model_path = repo.get(onnx_file)?;
        let tok_path = repo.get("tokenizer.json")?;

        let mut tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQ_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| e.to_string())?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let session = Session::builder()?.commit_from_file(&model_path)?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        Ok(Self {
            tokenizer,
            session,
            needs_token_type_ids,
        })
    }

    /// Returns `(n, 384)` L2-normalized embeddings, one `Vec<f32>` per input.
    pub fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| e.to_string())?;
        let n = encodings.len();
        let len = encodings[0].get_ids().len();

        let to_i64 = |get: &dyn Fn(&tokenizers::Encoding) -> Vec<i64>| -> Vec<i64> {
            encodings.iter().flat_map(|e| get(e)).collect()
        };
        let ids = to_i64(&|e| e.get_ids().iter().map(|&v| v as i64).collect());
        let mask = to_i64(&|e| e.get_attention_mask().iter().map(|&v| v as i64).collect());

        let shape = [n, len];
        let mut inputs: Vec<(&str, ort::session::SessionInputValue)> = vec![
            ("input_ids", Tensor::from_array((shape, ids))?.into()),
            (
                "attention_mask",
                Tensor::from_array((shape, mask.clone()))?.into(),
            ),
        ];
        if self.needs_token_type_ids {
            let type_ids = to_i64(&|e| e.get_type_ids().iter().map(|&v| v as i64).collect());
            inputs.push((
                "token_type_ids",
                Tensor::from_array((shape, type_ids))?.into(),
            ));
        }

        let outputs = self.session.run(inputs)?;
        let (out_shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let dim = out_shape[2] as usize;

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut emb = vec![0f32; dim];
            let mut count = 0f32;
            for t in 0..len {
                if mask[i * len + t] == 1 {
                    let off = (i * len + t) * dim;
                    for d in 0..dim {
                        emb[d] += data[off + d];
                    }
                    count += 1.0;
                }
            }
            let count = count.max(1e-9);
            for v in emb.iter_mut() {
                *v /= count;
            }
            let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in emb.iter_mut() {
                *v /= norm;
            }
            result.push(emb);
        }
        Ok(result)
    }
}
