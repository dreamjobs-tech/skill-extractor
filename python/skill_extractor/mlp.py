"""MLP classifier forward pass — consumes weights exported from the trained
sklearn MLPClassifier (mlp.json: base64 float32, row-major, shape [in, out])."""
import base64
import json

import numpy as np


class MLP:
    def __init__(self, path: str):
        with open(path) as f:
            spec = json.load(f)
        if spec.get("format") != "mlp-weights-v1":
            raise ValueError(f"unsupported weights format: {spec.get('format')}")
        self.layers = [
            (
                np.frombuffer(base64.b64decode(l["weights_b64"]), dtype=np.float32)
                .reshape(l["shape"]),
                np.frombuffer(base64.b64decode(l["bias_b64"]), dtype=np.float32),
            )
            for l in spec["layers"]
        ]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (n, 384) L2-normalized embeddings -> (n,) P(skill)."""
        h = x.astype(np.float32)
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < last:
                np.maximum(h, 0.0, out=h)  # relu
        return (1.0 / (1.0 + np.exp(-h[:, 0])))  # logistic
