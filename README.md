# Voynich Manuscript Analysis Toolkit

A computational analysis toolkit for the Voynich Manuscript (Beinecke MS 408, Yale University), implementing a complete pipeline from statistical analysis through candidate decipherment.

## Key Results

Analysis of 191,545 characters and 37,025 words from the Takahashi EVA transcription yields:

**Statistical Analysis (Phases 1-4)**

| Metric | Value | Significance |
|--------|-------|-------------|
| Index of Coincidence | 0.0771 | Natural language range (0.065-0.077), not random (0.038) |
| Conditional entropy H(1)/H(0) | 0.613 | Lower than any known language (~0.75-0.85) |
| Cipher scorecard | Homophonic: 1.0, Polyalphabetic: 0.13 | Polyalphabetic and glossolalia excluded |
| Homophone groups | 19 EVA chars -> 10 functional groups | ~47% of characters are redundant |
| Morpheme root length | 3.14 chars | Compatible with Semitic triconsonantal morphology |

**Decipherment Attempt (Phases 5-7)**

Two independent decoding paths converge on the same character mapping:

| Approach | Method | Result |
|----------|--------|--------|
| Italian path | Hill-climbing + plant-name constraints (Champollion method) | 19/19 chars mapped |
| Hebrew path | Homophone groups + frequency scoring vs Hebrew lexicon | 19/19 chars mapped |
| **Cross-path agreement** | **16/19 characters identical** | **84.2% convergence** |

The remaining 3 characters (f, i, q) were resolved through positional profile analysis and contextual distribution:
- **f** = allograph of p (lamed/l) -- confirmed by cosine similarity 0.998 and bigram context overlap 0.67
- **q** = prefix (stripped) -- 98.8% word-initial position, likely vav (ו, "and")
- **ii** = he (e), standalone **i** = resh (r) -- confirmed by `aiin` -> "dei" (of the, 457 occurrences)

**Validation**

| Test | Result |
|------|--------|
| Full decode coverage | 99.7% (36,922 / 37,025 words) |
| Italian lexicon matches | 111 unique words, 2,400 occurrences (6.5%) |
| Hebrew lexicon matches | 3,991 occurrences (10.8%) |
| Anchor word: shemesh (שמש, sun) | 2 occurrences, BOTH on zodiac pages only |
| Section distribution | Astronomical terms in zodiac, liquid terms in balneological (3/4 falsification tests passed) |

**Top decoded Italian words**: dei (481x), drito/dritto (236x), sto (198x), seta/setta (171x), moto (160x), speta/spetta (100x), gir (92x), deo (33x)

## Proposed Mapping

The hypothesis is that the Voynich Manuscript encodes a Judeo-Italian vernacular written in Hebrew script, read right-to-left.

| EVA | Hebrew | Name | Italian | Status |
|-----|--------|------|---------|--------|
| a | y | yod | i | convergent |
| c | A | aleph | a | convergent |
| d | r | resh | r | convergent |
| e | p | pe | p | convergent |
| f | l | lamed | l | allograph of p |
| g | X | chet | k | convergent |
| h | E | ayin | e | convergent |
| ii | h | he | e | composite glyph |
| i | r | resh | r | allograph of d |
| k | t | tav | t | convergent |
| l | m | mem | m | convergent |
| m | g | gimel | g | convergent |
| n | d | dalet | d | convergent |
| o | w | vav | o | convergent |
| p | l | lamed | l | convergent |
| q | -- | -- | prefix | stripped (vav?) |
| r | h | he | e | convergent |
| s | n | nun | n | convergent |
| t | J | tet | t | convergent |
| y | S | shin | s | convergent |

## Installation

```bash
git clone https://github.com/antenore/voynich-toolkit.git
cd voynich-toolkit
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+. Dependencies are installed automatically (click, numpy, scipy, matplotlib, rapidfuzz, etc.).

## Commands

### Phase 1-4: Statistical Analysis

```bash
# Visual pipeline (requires manuscript PDF)
voynich extract --pdf voynich.pdf     # PDF -> 214 page images
voynich segment-text                  # Isolate text regions
voynich segment-glyphs               # Segment individual glyphs
voynich stats                         # Frequency analysis, Zipf, entropy

# EVA transcription analysis (no PDF needed)
voynich eva                           # Compare with EVA transcription
voynich word-structure                # Word tokenizer + positional profiles
voynich language-fingerprint          # Conditional entropy, IoC, Heaps law
voynich char-embeddings               # PPMI, SVD embeddings, clustering
voynich cipher-hypothesis             # Cipher scorecard + morphological decomposition

# Lexicon preparation
voynich prepare-lexicon               # Download Hebrew lexicon (STEPBible)
voynich prepare-italian-lexicon       # Prepare medieval Italian lexicon

# Initial decipherment attempts
voynich decipher --restarts 20        # Hebrew-only hill-climbing
voynich decipher-italian --restarts 30  # Italian (Judeo-Italian) hill-climbing
voynich champollion                   # Plant-name constrained decoder
```

### Phase 5: Fuzzy Matching & Independent Analysis

```bash
voynich fuzzy-decode                  # Complete 4 unmapped chars, fuzzy-match vs Italian
voynich plant-search                  # Search for plant names in decoded text
voynich copyist-errors                # Analyze potential copyist errors in EVA space
voynich hebrew-decode                 # Alternative Hebrew path via homophone groups
```

### Phase 6: Full Decode & Validation

```bash
voynich full-decode                   # Decode corpus with convergent 16-char mapping
voynich anchor-words                  # Multilingual anchor word search (IT/HE/LA)
voynich zodiac-test                   # Zodiac section vs sign/month/planet names
```

### Phase 7: Character Resolution

```bash
voynich prefix-resolve                # Resolve f/i/q via prefix test + brute force
python analysis_full_19char.py        # Full 19-char decode with all resolutions
```

### Global Options

```bash
voynich --force <command>             # Re-run even if output exists
voynich --output-dir results/ <cmd>   # Alternative output directory
voynich --help                        # All available commands
```

## Project Structure

```
voynich-toolkit/
├── pyproject.toml
├── README.md
├── PAPER.md                          # Academic paper draft (Phases 1-4)
├── LICENSE                           # MIT
├── eva_data/
│   └── LSI_ivtff_0d.txt             # EVA IVTFF transcription (Takahashi)
├── analysis_*.py                     # Phase 7 standalone analysis scripts
├── output/
│   ├── stats/                        # JSON reports + PNG visualizations
│   ├── lexicon/                      # Hebrew & Italian lexicons
│   ├── eva/                          # EVA comparison data
│   ├── pages/                        # Extracted page images (not in repo)
│   ├── text_regions/                 # Segmented text regions (not in repo)
│   └── glyphs/                       # Individual glyphs (not in repo)
└── src/voynich_toolkit/
    ├── cli.py                        # Unified CLI (click)
    ├── config.py                     # ToolkitConfig dataclass
    ├── utils.py                      # Shared utilities
    ├── extract.py                    # PDF -> images
    ├── segment_text.py               # Text region segmentation
    ├── segment_glyphs.py             # Glyph segmentation
    ├── statistics.py                 # Glyph statistics
    ├── eva_compare.py                # EVA comparison
    ├── section_analysis.py           # Cross-section analysis
    ├── image_text_correlation.py     # Image-text correlation
    ├── word_structure.py             # Word tokenizer + positional profiles
    ├── language_fingerprint.py       # Entropy, IoC, Heaps, KL divergence
    ├── char_embeddings.py            # PPMI, SVD embeddings
    ├── cipher_hypothesis.py          # Cipher scorecard + morphology
    ├── prepare_lexicon.py            # Hebrew lexicon (STEPBible)
    ├── prepare_italian_lexicon.py    # Italian lexicon (Dante, TLIO)
    ├── decipher.py                   # Hebrew-only hill-climbing
    ├── italian_decipher.py           # Italian hill-climbing decoder
    ├── champollion.py                # Plant-name constrained decoder
    ├── fuzzy_utils.py                # Fuzzy matching utilities
    ├── fuzzy_decode.py               # Complete mapping + fuzzy match
    ├── plant_search.py               # Botanical name search
    ├── copyist_errors.py             # Copyist error analysis
    ├── hebrew_decode.py              # Alternative Hebrew path
    ├── full_decode.py                # Full corpus decode (16-char)
    ├── anchor_words.py               # Anchor word validation
    ├── zodiac_test.py                # Zodiac section validation
    └── prefix_resolve.py             # f/i/q resolution
```

## Source Data

- **EVA IVTFF**: `LSI_ivtff_0d.txt` (Takahashi, 1998) -- 191,545 characters, 37,025 words, 225 folios
- **Hebrew lexicon**: STEPBible TBESH (CC BY 4.0) + curated medieval glossaries -- 6,449 consonantal forms
- **Italian lexicon**: Dante concordance + TLIO + medieval botanical/medical terms -- 60,738 forms

## Caveats

This is exploratory research, not a confirmed decipherment. Key limitations:

1. **Low Italian lexicon match rate** (6.5%) -- many decoded words do not match known Italian forms
2. **Decoded text is not yet readable** as coherent Italian sentences
3. **No formal null hypothesis test** yet -- the statistical significance of the convergence needs quantification
4. **Limited medieval lexicon coverage** -- the Dante-based lexicon may miss technical vocabulary (botanical, medical, astrological)
5. **The ii/i split** (Phase 7) is the least proven part of the mapping

The shemesh (שמש, sun) finding on zodiac pages is the strongest individual result, but one word does not make a decipherment.

## References

- [Beinecke Digital Library - MS 408](https://collections.library.yale.edu/catalog/2002046)
- [EVA Transcription Archive](http://www.voynich.nu/transcr.html)
- Currier, P. (1976). *Papers on the Voynich Manuscript*
- D'Imperio, M.E. (1978). *The Voynich Manuscript: An Elegant Enigma*. NSA.
- Montemurro, M.A. & Zanette, D.H. (2013). Keywords and co-occurrence patterns in the Voynich Manuscript. *PLoS ONE*, 8(6).
- Rugg, G. (2004). An elegant hoax? *Cryptologia*, 28(1), 31-46.
- Stolfi, J. (1998). *An Interlinear Archive of Voynich Manuscript Transcriptions in EVA*
- Takahashi, T. (1998). *Complete EVA transcription of the Voynich Manuscript*

## License

MIT License. See [LICENSE](LICENSE).
