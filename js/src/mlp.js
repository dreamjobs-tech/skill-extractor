/**
 * MLP classifier forward pass — consumes mlp.json exported from the trained
 * sklearn MLPClassifier (base64 float32, row-major, shape [in, out]).
 */

function decodeF32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

export class MLP {
  constructor(spec) {
    if (spec.format !== 'mlp-weights-v1') {
      throw new Error(`unsupported weights format: ${spec.format}`);
    }
    this.layers = spec.layers.map((l) => ({
      rows: l.shape[0],
      cols: l.shape[1],
      w: decodeF32(l.weights_b64),
      b: decodeF32(l.bias_b64),
    }));
  }

  /** @param {Float32Array[]} xs L2-normalized 384-d embeddings @returns {number[]} P(skill) */
  predictProba(xs) {
    return xs.map((x) => {
      let h = x;
      for (let i = 0; i < this.layers.length; i++) {
        const { rows, cols, w, b } = this.layers[i];
        const next = new Float32Array(cols);
        for (let j = 0; j < cols; j++) next[j] = b[j];
        for (let r = 0; r < rows; r++) {
          const hv = h[r];
          if (hv === 0) continue;
          const off = r * cols;
          for (let j = 0; j < cols; j++) next[j] += hv * w[off + j];
        }
        if (i < this.layers.length - 1) {
          for (let j = 0; j < cols; j++) if (next[j] < 0) next[j] = 0; // relu
        }
        h = next;
      }
      return 1 / (1 + Math.exp(-h[0])); // logistic
    });
  }
}
