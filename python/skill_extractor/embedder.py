"""MiniLM sentence embeddings via ONNX Runtime — no torch.

Reproduces sentence-transformers/all-MiniLM-L6-v2: BERT encoder -> mean
pooling over the attention mask -> L2 normalize. Model files are fetched
from the Hugging Face Hub on first use and cached locally.
"""
import numpy as np

MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


class Embedder:
    def __init__(self, quantized: bool = False):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        onnx_file = "onnx/model_quantized.onnx" if quantized else "onnx/model.onnx"
        model_path = hf_hub_download(MODEL_REPO, onnx_file)
        tok_path = hf_hub_download(MODEL_REPO, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (n, 384) float32 L2-normalized embeddings."""
        encs = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encs], dtype=np.int64
            )
        (hidden,) = self.session.run(["last_hidden_state"], feeds)

        m = mask[:, :, None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        emb = summed / counts
        norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return (emb / norms).astype(np.float32)
