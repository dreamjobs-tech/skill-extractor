//! Parity tests against the shared fixtures — must match the Python reference.
//! Fixtures live at ../fixtures relative to the crate; skipped when absent
//! (i.e. when the crate is built outside the monorepo).

use serde::Deserialize;
use skill_extractor::{Candidate, KeywordMatcher, SkillExtractor, SKILLS_JSON};

#[derive(Deserialize)]
struct MatcherCase {
    text: String,
    spans: Vec<(String, usize, usize)>,
}

#[derive(Deserialize)]
struct FixtureCandidate {
    skill: String,
    context: String,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    candidates: Vec<FixtureCandidate>,
    embed_inputs: Vec<String>,
    probs: Vec<f32>,
    skills: Vec<String>,
}

#[derive(Deserialize)]
struct Fixtures {
    prob_tolerance: f32,
    cases: Vec<Case>,
}

fn fixture(name: &str) -> Option<String> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name);
    std::fs::read_to_string(p).ok()
}

#[test]
fn matcher_parity() {
    let Some(raw) = fixture("matcher_fixtures.json") else {
        eprintln!("fixtures not present — skipping");
        return;
    };
    let cases: Vec<MatcherCase> = serde_json::from_str(&raw).unwrap();
    let skills: Vec<String> = serde_json::from_str(SKILLS_JSON).unwrap();
    let m = KeywordMatcher::new(&skills);
    let mut spans = 0;
    for c in &cases {
        let got = m.extract(&c.text);
        assert_eq!(got, c.spans, "mismatch on {:?}", &c.text[..c.text.len().min(80)]);
        spans += got.len();
    }
    println!("matcher parity ok: {} cases, {} spans", cases.len(), spans);
}

#[test]
fn extraction_parity() {
    let Some(raw) = fixture("extraction_fixtures.json") else {
        eprintln!("fixtures not present — skipping");
        return;
    };
    let fixtures: Fixtures = serde_json::from_str(&raw).unwrap();
    let mut ex = SkillExtractor::new(false).unwrap();

    for c in &fixtures.cases {
        let cands = ex.candidates(&c.text);
        let want: Vec<Candidate> = c
            .candidates
            .iter()
            .map(|x| Candidate {
                skill: x.skill.clone(),
                context: x.context.clone(),
            })
            .collect();
        assert_eq!(cands, want, "candidates mismatch");

        let inputs: Vec<String> = cands
            .iter()
            .map(|x| format!("{} : {}", x.skill, x.context))
            .collect();
        assert_eq!(inputs, c.embed_inputs, "embed inputs mismatch");

        if !inputs.is_empty() {
            let probs = ex.classify(&inputs).unwrap();
            for (i, (got, want)) in probs.iter().zip(&c.probs).enumerate() {
                assert!(
                    (got - want).abs() < fixtures.prob_tolerance,
                    "prob[{i}] {got} vs {want}"
                );
            }
        }

        let skills = ex.extract(&c.text, 0.5).unwrap();
        assert_eq!(skills, c.skills, "final skills mismatch");
    }
    println!("all {} cases passed", fixtures.cases.len());
}
