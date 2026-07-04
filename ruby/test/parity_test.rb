# Parity tests against the shared fixtures — must match the Python reference.
require "json"
require_relative "../lib/skill_extractor"

root = File.expand_path("../..", __dir__)

# 1) Matcher span parity (300 fuzz cases from the Python reference)
skills = JSON.parse(File.read(File.join(SkillExtractor::DATA_DIR, "skills.json")))
m = SkillExtractor::KeywordMatcher.new(skills)
matcher_cases = JSON.parse(File.read(File.join(root, "fixtures", "matcher_fixtures.json")))
spans = 0
matcher_cases.each do |c|
  got = m.extract(c["text"]).map { |k, s, e| [k, s, e] }
  unless got == c["spans"]
    warn "MATCHER MISMATCH on #{c["text"][0, 80].inspect}\n got: #{got[0, 5]}\nwant: #{c["spans"][0, 5]}"
    exit 1
  end
  spans += got.length
end
puts "matcher parity ok: #{matcher_cases.length} cases, #{spans} spans"

# 2) Full pipeline parity
fixtures = JSON.parse(File.read(File.join(root, "fixtures", "extraction_fixtures.json")))
ex = SkillExtractor::Extractor.new
tol = fixtures["prob_tolerance"]
failures = 0
fixtures["cases"].each do |c|
  label = c["text"][0, 40].inspect
  cands = ex.candidates(c["text"])
  want_cands = c["candidates"].map { |x| { skill: x["skill"], context: x["context"] } }
  if cands != want_cands
    warn "FAIL #{label}: candidates mismatch"
    failures += 1
    next
  end

  inputs = cands.map { |x| "#{x[:skill]} : #{x[:context]}" }
  unless inputs == c["embed_inputs"]
    warn "FAIL #{label}: embed inputs mismatch"
    failures += 1
    next
  end

  unless inputs.empty?
    probs = ex.classify(inputs)
    probs.each_with_index do |p, i|
      if (p - c["probs"][i]).abs >= tol
        warn "FAIL #{label}: prob[#{i}] #{p} vs #{c["probs"][i]}"
        failures += 1
      end
    end
  end

  got = ex.extract(c["text"])
  if got == c["skills"]
    puts "ok   #{label}"
  else
    warn "FAIL #{label}: skills #{got} vs #{c["skills"]}"
    failures += 1
  end
end

abort "#{failures} failure(s)" if failures.positive?
puts "\nall #{fixtures["cases"].length} cases passed"
