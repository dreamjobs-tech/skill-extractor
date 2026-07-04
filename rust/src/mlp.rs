//! MLP classifier forward pass — consumes mlp.json exported from the trained
//! sklearn MLPClassifier (base64 float32, row-major, shape [in, out]).

use base64::Engine;
use serde::Deserialize;

#[derive(Deserialize)]
struct LayerSpec {
    shape: [usize; 2],
    weights_b64: String,
    bias_b64: String,
}

#[derive(Deserialize)]
struct MlpSpec {
    format: String,
    layers: Vec<LayerSpec>,
}

struct Layer {
    rows: usize,
    cols: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

pub struct Mlp {
    layers: Vec<Layer>,
}

fn decode_f32(b64: &str) -> Result<Vec<f32>, base64::DecodeError> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl Mlp {
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let spec: MlpSpec = serde_json::from_str(json)?;
        if spec.format != "mlp-weights-v1" {
            return Err(format!("unsupported weights format: {}", spec.format).into());
        }
        let layers = spec
            .layers
            .iter()
            .map(|l| {
                Ok(Layer {
                    rows: l.shape[0],
                    cols: l.shape[1],
                    w: decode_f32(&l.weights_b64)?,
                    b: decode_f32(&l.bias_b64)?,
                })
            })
            .collect::<Result<Vec<_>, base64::DecodeError>>()?;
        Ok(Self { layers })
    }

    /// `xs`: L2-normalized 384-d embeddings -> P(skill) per row.
    pub fn predict_proba(&self, xs: &[Vec<f32>]) -> Vec<f32> {
        let last = self.layers.len() - 1;
        xs.iter()
            .map(|x| {
                let mut h = x.clone();
                for (i, l) in self.layers.iter().enumerate() {
                    let mut next = l.b.clone();
                    for r in 0..l.rows {
                        let hv = h[r];
                        if hv == 0.0 {
                            continue;
                        }
                        let off = r * l.cols;
                        for j in 0..l.cols {
                            next[j] += hv * l.w[off + j];
                        }
                    }
                    if i < last {
                        for v in next.iter_mut() {
                            if *v < 0.0 {
                                *v = 0.0; // relu
                            }
                        }
                    }
                    h = next;
                }
                1.0 / (1.0 + (-h[0]).exp()) // logistic
            })
            .collect()
    }
}
