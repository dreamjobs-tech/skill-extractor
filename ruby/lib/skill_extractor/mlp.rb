# MLP classifier forward pass — consumes mlp.json exported from the trained
# sklearn MLPClassifier (base64 float32, row-major, shape [in, out]).
require "base64"
require "json"

module SkillExtractor
  class MLP
    def initialize(path)
      spec = JSON.parse(File.read(path))
      unless spec["format"] == "mlp-weights-v1"
        raise ArgumentError, "unsupported weights format: #{spec["format"]}"
      end
      @layers = spec["layers"].map do |l|
        {
          rows: l["shape"][0],
          cols: l["shape"][1],
          w: Base64.decode64(l["weights_b64"]).unpack("e*"),
          b: Base64.decode64(l["bias_b64"]).unpack("e*")
        }
      end
    end

    # xs: array of 384-dim L2-normalized embeddings -> array of P(skill)
    def predict_proba(xs)
      last = @layers.length - 1
      xs.map do |x|
        h = x
        @layers.each_with_index do |l, i|
          cols = l[:cols]
          w = l[:w]
          nxt = l[:b].dup
          h.each_with_index do |hv, r|
            next if hv.zero?
            off = r * cols
            cols.times { |j| nxt[j] += hv * w[off + j] }
          end
          nxt.map! { |v| v.negative? ? 0.0 : v } if i < last # relu
          h = nxt
        end
        1.0 / (1.0 + Math.exp(-h[0])) # logistic
      end
    end
  end
end
