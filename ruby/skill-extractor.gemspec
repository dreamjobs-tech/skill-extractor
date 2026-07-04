require_relative "lib/skill_extractor/version"

Gem::Specification.new do |spec|
  spec.name = "skill-extractor"
  spec.version = SkillExtractor::VERSION
  spec.summary = "Extract skills from job postings and resumes — 32K-skill gazetteer + MiniLM embeddings + MLP context classifier (73% F1)."
  spec.description = "Gazetteer candidates filtered by a MiniLM + MLP context classifier trained on 491K labeled samples, so prose like 'can-do attitude' doesn't become a skill. Runs on ONNX Runtime via the informers gem."
  spec.homepage = "https://github.com/dreamjobs-tech/skill-extractor"
  spec.license = "MIT"
  spec.authors = ["Qarera"]
  spec.email = ["yashthenuan21@gmail.com"]

  spec.files = Dir["lib/**/*.rb", "data/*.json", "README.md", "LICENSE"]
  spec.require_paths = ["lib"]
  spec.required_ruby_version = ">= 3.0"

  spec.metadata = {
    "source_code_uri" => "https://github.com/dreamjobs-tech/skill-extractor/tree/main/ruby",
    "homepage_uri" => "https://www.qarera.com"
  }

  spec.add_dependency "informers", ">= 1.0"
end
