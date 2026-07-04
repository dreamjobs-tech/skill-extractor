# Keyword matcher — a faithful port of FlashText's extract_keywords.
# Word boundaries: any char outside [A-Za-z0-9_]. Case-insensitive,
# longest-match-wins, non-overlapping. Identical spans across all
# skill-extractor language implementations.
module SkillExtractor
  class KeywordMatcher
    KEYWORD = :__keyword__
    WORD_CHARS = ("a".."z").to_a + ("A".."Z").to_a + ("0".."9").to_a + ["_"]
    WORD_SET = WORD_CHARS.to_h { |c| [c, true] }.freeze

    def initialize(keywords = [])
      @trie = {}
      keywords.each { |kw| add(kw) }
    end

    def add(keyword)
      node = @trie
      kw = keyword.downcase
      kw.each_char do |ch|
        node = (node[ch] ||= {})
      end
      node[KEYWORD] = kw
    end

    # Returns [[keyword, start, end], ...] spans, matching FlashText exactly.
    def extract(sentence)
      out = []
      return out if sentence.nil? || sentence.empty?

      sentence = sentence.downcase
      chars = sentence.chars
      current = @trie
      seq_start = 0
      seq_end = 0
      reset = false
      idx = 0
      n = chars.length
      while idx < n
        ch = chars[idx]
        if !WORD_SET[ch]
          if current.key?(KEYWORD) || current.key?(ch)
            longest = nil
            longer_found = false
            if current.key?(KEYWORD)
              longest = current[KEYWORD]
              seq_end = idx
            end
            if current.key?(ch)
              cont = current[ch]
              idy = idx + 1
              broke = false
              while idy < n
                inner = chars[idy]
                if !WORD_SET[inner] && cont.key?(KEYWORD)
                  longest = cont[KEYWORD]
                  seq_end = idy
                  longer_found = true
                end
                if cont.key?(inner)
                  cont = cont[inner]
                else
                  broke = true
                  break
                end
                idy += 1
              end
              if !broke && cont.key?(KEYWORD)
                longest = cont[KEYWORD]
                seq_end = idy
                longer_found = true
              end
              idx = seq_end if longer_found
            end
            current = @trie
            out << [longest, seq_start, idx] if longest
            reset = true
          else
            current = @trie
            reset = true
          end
        elsif current.key?(ch)
          current = current[ch]
        else
          current = @trie
          reset = true
          idy = idx + 1
          while idy < n
            break unless WORD_SET[chars[idy]]
            idy += 1
          end
          idx = idy
        end
        if idx + 1 >= n && current.key?(KEYWORD)
          out << [current[KEYWORD], seq_start, n]
        end
        idx += 1
        if reset
          reset = false
          seq_start = idx
        end
      end
      out
    end
  end
end
