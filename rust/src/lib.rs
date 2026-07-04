//! skill-extractor: extract skills from job postings and resumes.
//!
//! A 32K-skill gazetteer proposes candidates; a MiniLM + MLP context
//! classifier accepts or rejects each one. Trained on 491K labeled samples
//! (73% F1 on held-out job postings).
//!
//! ```no_run
//! use skill_extractor::SkillExtractor;
//!
//! let mut ex = SkillExtractor::new(false).unwrap();
//! let skills = ex.extract("5+ years Python, Docker and Kubernetes required.", 0.5).unwrap();
//! assert!(skills.contains(&"python".to_string()));
//! ```

mod embedder;
mod matcher;
mod mlp;

pub use embedder::Embedder;
pub use matcher::KeywordMatcher;
pub use mlp::Mlp;

use std::collections::BTreeSet;

pub const CONTEXT_BEFORE: usize = 20;
pub const CONTEXT_AFTER: usize = 21;
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Gazetteer + MLP weights ship inside the crate; the MiniLM ONNX model is
/// downloaded from the Hugging Face Hub on first use.
pub static SKILLS_JSON: &str = include_str!("../data/skills.json");
pub static MLP_JSON: &str = include_str!("../data/mlp.json");

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub skill: String,
    pub context: String,
}

pub struct SkillExtractor {
    matcher: KeywordMatcher,
    mlp: Mlp,
    embedder: Embedder,
}

impl SkillExtractor {
    /// `quantized: false` (fp32) matches the reference implementation.
    pub fn new(quantized: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let skills: Vec<String> = serde_json::from_str(SKILLS_JSON)?;
        Ok(Self {
            matcher: KeywordMatcher::new(&skills),
            mlp: Mlp::from_json(MLP_JSON)?,
            embedder: Embedder::new(quantized)?,
        })
    }

    /// Gazetteer matches with their ±20-word context windows.
    pub fn candidates(&self, text: &str) -> Vec<Candidate> {
        let matches = self.matcher.extract(text);
        if matches.is_empty() {
            return Vec::new();
        }
        let words: Vec<&str> = text.split_whitespace().collect();
        let chars: Vec<char> = text.chars().collect();
        matches
            .into_iter()
            .map(|(skill, start, _end)| {
                let prefix: String = chars[..start.min(chars.len())].iter().collect();
                let word_idx = prefix.split_whitespace().count();
                let lo = word_idx.saturating_sub(CONTEXT_BEFORE);
                let hi = (word_idx + CONTEXT_AFTER).min(words.len());
                Candidate {
                    skill,
                    context: words[lo..hi].join(" "),
                }
            })
            .collect()
    }

    /// P(skill) for each candidate input string.
    pub fn classify(&mut self, inputs: &[String]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let embeddings = self.embedder.encode(inputs)?;
        Ok(self.mlp.predict_proba(&embeddings))
    }

    /// Extract confirmed skills from a job posting or resume text.
    pub fn extract(
        &mut self,
        text: &str,
        threshold: f32,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let cands = self.candidates(text);
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<String> = cands
            .iter()
            .map(|c| format!("{} : {}", c.skill, c.context))
            .collect();
        let probs = self.classify(&inputs)?;
        let found: BTreeSet<String> = probs
            .iter()
            .zip(&cands)
            .filter(|(p, _)| **p >= threshold)
            .map(|(_, c)| c.skill.clone())
            .collect();
        Ok(found.into_iter().collect())
    }
}
