# skill-extractor: extract skills from job postings and resumes.
# 32K-skill gazetteer + MiniLM embeddings + MLP context classifier,
# trained on 491K labeled samples (73% F1 held-out).
require "json"

require_relative "skill_extractor/matcher"
require_relative "skill_extractor/mlp"
require_relative "skill_extractor/version"

module SkillExtractor
  MODEL = "Xenova/all-MiniLM-L6-v2"
  CONTEXT_BEFORE = 20
  CONTEXT_AFTER = 21
  DEFAULT_THRESHOLD = 0.5
  DATA_DIR = File.expand_path("../data", __dir__)

  class Extractor
    # fp32 (default) matches the reference implementation bit-for-bit
    def initialize(quantized: false)
      @quantized = quantized
      @matcher = KeywordMatcher.new(JSON.parse(File.read(File.join(DATA_DIR, "skills.json"))))
      @mlp = MLP.new(File.join(DATA_DIR, "mlp.json"))
      @pipe = nil
    end

    attr_reader :matcher, :mlp

    # Gazetteer matches with their +/-20-word context windows.
    def candidates(text)
      matches = @matcher.extract(text)
      return [] if matches.empty?

      words = text.split
      matches.map do |skill, start, _end|
        word_idx = text[0...start].split.length
        lo = [0, word_idx - CONTEXT_BEFORE].max
        hi = [words.length, word_idx + CONTEXT_AFTER].min
        { skill: skill, context: words[lo...hi].join(" ") }
      end
    end

    # P(skill) for each candidate input string.
    def classify(inputs)
      @pipe ||= begin
        require "informers"
        Informers.pipeline("embedding", MODEL, quantized: @quantized)
      end
      @mlp.predict_proba(@pipe.(inputs))
    end

    # Extract confirmed skills from a job posting or resume text.
    def extract(text, threshold: DEFAULT_THRESHOLD)
      cands = candidates(text)
      return [] if cands.empty?

      probs = classify(cands.map { |c| "#{c[:skill]} : #{c[:context]}" })
      found = {}
      probs.each_with_index { |p, i| found[cands[i][:skill]] = true if p >= threshold }
      found.keys.sort
    end
  end

  @default = nil

  # Module-level convenience wrapper around a shared Extractor.
  def self.extract_skills(text, threshold: DEFAULT_THRESHOLD)
    @default ||= Extractor.new
    @default.extract(text, threshold: threshold)
  end
end
