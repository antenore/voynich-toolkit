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

**Validation (updated 2026-02-17 — Phase 11 Critical Re-Assessment)**

| Test | Result |
|------|--------|
| Full decode coverage | 99.7% (36,922 / 37,025 words) |
| Hebrew lexicon matches | 40.3% (full 491K) / 24.3% (glossed 28K) / 17.2% (STEPBible 6.5K) |
| Random mapping matches | 29.9% (full) / 12.1% (glossed) / 9.0% (STEPBible) |
| Random string ("monkey") | 26.8% (full) / 11.1% (glossed) / ~5% (STEPBible) |
| z-score vs random | 3.7 (full) / **4.4 (glossed)** / 3.6 (STEPBible) |
| Plant name search | permutation p=0.017 (*) |
| Anchor words (d=0) | 28 / 445, permutation p=0.004 (**) |
| Zodiac vocabulary | permutation p=0.071 (ns) |
| Italian-layer analysis | 4.5% match (z=3.82) — ruled out |
| Mapping optimality | 16/17 optimal under 1-to-1 constraint |
| Semantic coherence | z=4.35-14.54 (p≤0.002) — words cluster, but glosses don't form sentences |

**Critical assessment**: The mapping produces a statistically significant signal (z=3.6-4.4 across all lexicon sizes, ruling out pure chance). However, the decoded text does **not** read as coherent Hebrew — the best passages with 100% semantic coverage produce incoherent glosses. The z=42.8 headline figure is inflated by the 491K-form lexicon (random strings match at 26.8%). With the more honest glossed-only lexicon (28K forms with dictionary definitions), the z-score is 4.4 — still significant, but the match rate is only 24.3%.

**Reality test**: Random mappings barely beat random strings (ratio ~1.1x), while the real mapping beats random strings by 2.2-3.4x depending on lexicon size. This rules out pure chance but the modest absolute differential (~8 percentage points) suggests either a partially correct mapping or structural artifact.

**Phase 12-16 findings** (2026-02-17/18):

| Finding | Result |
|---------|--------|
| **Reading direction** (P15) | RTL confirmed: +8.5pp, z=22.97. LTR has weak residual signal (z=2.44, palindromes) |
| **Mapping stability** (P15) | 14/17 optimal with honest 45K lexicon (identical to full) |
| **Naibbe cipher** (P13) | REJECTED: Monte Carlo produces 20.7% vs real 40.3% (z=12.1). 8/9 diagnostics favor mono |
| **Judeo-Italian** (P14) | Plausible partial: 7.4% match (z=4.59), explains 10% of Hebrew matches |
| **Currier A/B** (P12) | Both significant (A: z=4.02, B: z=3.85). A stronger: +7.0pp (p<0.0001) |
| **Scribe analysis** (P15) | **Hand 1 drives the signal**: 48.5% vs 35-42% others. Hand 4 (also Lang A) = only 35.1% |
| **Null model** (P16B) | Signal ABOVE null model: match z=98.2, bigrams z=40.9, Zipfian gloss distribution |
| **Section entropy** (P16C) | Sections NOT uniform (chi2=163.7, p≈0). Range 12-25% honest. Z/A lowest (12%) |
| **Layout-aware** (P17) | Labels 13.2% vs paragraphs 24.7% (z=-8.10). Z has zero paragraphs — explains low rate |
| **Meta-analysis** (P18) | h2 entropy: 2.12→2.44 bits (+0.32) — mapping decompresses toward Hebrew (3.72). 15 papers evaluated |

**Note on multiple scribes**: Lisa Fagin Davis (2020) identified 5 distinct scribal hands in the manuscript. The Naibbe cipher (Greshko, 2025, Cryptologia) demonstrates that a verbose homophonic substitution cipher can produce Voynich-like ciphertext from Latin/Italian using historically plausible 15th-century materials. If the manuscript uses homophonic substitution or different scribes employed different conventions, a single monoalphabetic mapping is structurally inadequate.

**Note on Hand 1 specificity**: Phase 15 scribe analysis reveals the Hebrew signal is dominated by Hand 1 (86 pages, all Lang A herbal). Hand 4 — also Lang A — matches at only 35.1%, lower than Hand 2 (Lang B, 41.5%). The Currier A>B difference was actually "Hand 1 > everyone else". The mapping may be specific to one scribe's conventions.

**Top decoded Hebrew words** (with dictionary gloss):

| Consonantal | Hebrew | Freq | Meaning |
|-------------|--------|------|---------|
| bhyr | בָּהִיר | 846 | bright, brilliant (of light) |
| Spk | שָׁפַךְ | 345 | to pour |
| bryt | בְּרִית | 325 | covenant |
| myt | מות | 211 | to die |
| swk | סוּךְ | 207 | to anoint |
| mwt | מות | 141 | to die |
| Srk | שׂרך | 145 | to twist |
| mwr | מור | 112 | to change |
| gyr | גִּיר | 92 | chalk, plaster |
| mwpt | מוֹפֵת | 70 | wonder, sign |

346 unique decoded words with known Hebrew meanings (8,227 occurrences, 22.8% of text). 1,098 total words matching the expanded lexicon (44.7% of text). See `output/stats/glossed_words.tsv` for the full table.

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
```

### Phase 8: Validation & Enrichment

```bash
voynich enrich-lexicon                # Jastrow + Sefaria corpus + Klein + PN filter
voynich anchor-words                  # Domain anchor words + permutation test
voynich zodiac-test                   # Zodiac section validation + permutation test
voynich plant-search                  # Botanical fuzzy matching + permutation test
voynich cross-language                # Hebrew vs Aramaic vs Random baseline
voynich section-specificity           # Section specificity index + heatmap
voynich validation-summary            # Aggregate scorecard (JSON/TXT/LaTeX)
```

### Phase 9: Mapping Refinement

```bash
voynich allograph-analysis            # l/e and k/t allograph investigation
voynich digraph-analysis              # EVA digraphs → Hebrew letters (ch→kaf)
voynich dual-role                     # Positional splits (d→bet, h→samekh)
voynich allograph-lr-deep             # l/r allography analysis
```

### Phase 10: Analysis & Validation

```bash
voynich phrase-completion             # 4-tier resolution of unknown words
voynich italian-layer                 # Italian-layer validation
voynich mapping-audit                 # Per-letter optimality test
voynich qof-investigation             # t→qof differential test
voynich semantic-coherence            # Semantic coherence analysis + permutation test
voynich phrase-reconstruction         # Anchor confidence + reconstruction + gematria
```

### Phase 12-16: Deep Analysis

```bash
voynich currier-split                 # Currier A/B split + permutation test
voynich naibbe-test                   # Naibbe verbose homophonic cipher hypothesis
voynich judeo-italian                 # Judeo-Italian hypothesis test
voynich direction-test                # RTL vs LTR reading direction test
voynich mapping-audit-honest          # Per-letter audit with honest 45K lexicon
voynich scribe-analysis               # Per-scribe match rate analysis
voynich hand1-dive                    # Hand 1 deep dive (vocab/structure/audit)
voynich null-model-test               # Null model test vs synthetic random words
voynich section-entropy               # Section match rates, uniformity, EVA profiles
voynich layout-analysis               # Layout-aware (label vs paragraph vs circular)
voynich meta-analysis                 # Meta-analysis: h2, MATTR, Zipf, literature table
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
├── LICENSE                           # MIT
├── eva_data/
│   └── LSI_ivtff_0d.txt             # EVA IVTFF transcription (Takahashi)
├── paper/
│   └── paper.tex                     # LaTeX source for the academic paper
├── scripts/
│   └── build_sqlite_db.py           # Rebuild voynich.db from output files
├── output/                           # Generated artifacts (not in repo)
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
    ├── full_decode.py                # Full corpus decode (19 Hebrew letters)
    ├── anchor_words.py               # Anchor word validation + permutation
    ├── zodiac_test.py                # Zodiac section validation + permutation
    ├── prefix_resolve.py             # f/i/q resolution
    ├── enrich_lexicon.py             # Jastrow dict + PN filter
    ├── cross_language_baseline.py    # Hebrew vs Aramaic vs Random
    ├── section_specificity.py        # Domain term concentration
    ├── validation_summary.py         # Aggregate scorecard
    ├── permutation_stats.py          # Random mapping framework
    ├── allograph_analysis.py         # Phase 9 l/e, k/t allographs
    ├── allograph_kt_deep.py          # Phase 9 k/t deep analysis
    ├── allograph_lr_deep.py          # Phase 9 l/r allography
    ├── digraph_analysis.py           # Phase 9 ch→kaf digraph
    ├── dual_role_analysis.py         # Phase 9 positional splits
    ├── deep_yl_analysis.py           # Phase 9 shin/mem investigation
    ├── mater_lectionis.py            # Phase 9 mater tolerance
    ├── pe_tet_investigation.py       # Phase 9 pe/tet medial analysis
    ├── phrase_completion.py          # Phase 10 multi-tier word resolution
    ├── italian_layer_analysis.py     # Phase 10 Italian-layer validation
    ├── mapping_audit.py              # Phase 10 per-letter optimality test
    ├── qof_investigation.py          # Phase 10 qof swap differential test
    ├── semantic_coherence.py         # Phase 10 semantic coherence + permutation
    ├── phrase_reconstruction.py      # Phase 11 anchor confidence + reconstruction + gematria
    ├── currier_split.py              # Phase 12 Currier A/B split analysis
    ├── naibbe_test.py                # Phase 13 Naibbe verbose cipher hypothesis
    ├── judeo_italian_test.py         # Phase 14 Judeo-Italian hypothesis
    ├── direction_test.py             # Phase 15 RTL vs LTR direction test
    ├── scribe_analysis.py            # Phase 15 per-scribe match rate analysis
    ├── hand1_deep_dive.py            # Phase 16 Hand 1 deep dive
    ├── null_model_test.py            # Phase 16B null model vs synthetic
    ├── section_entropy.py            # Phase 16C section entropy analysis
    ├── layout_analysis.py            # Phase 17 layout-aware analysis
    └── meta_analysis.py              # Phase 18 meta-analysis (h2, MATTR, literature)
```

## Source Data

- **EVA IVTFF**: `LSI_ivtff_0d.txt` (Takahashi, 1998) -- 191,545 characters, 37,025 words, 225 folios
- **Hebrew lexicon**: STEPBible TBESH (CC BY 4.0) + Jastrow Dictionary + Sefaria corpus (250M tokens, freq≥5) + Klein Etymological Dictionary + curated medieval glossaries -- 491,137 consonantal forms (proper nouns filtered)
- **Italian lexicon**: Dante concordance + TLIO + medieval botanical/medical terms -- 60,738 forms

## Open Directions

30 investigations completed across Phases 1-18. Remaining options:

1. **Hand 4 allography**: H4 uses EVA 'e' at 2x the rate of H1 — possible unmapped allograph (limited power: 817 words).
2. **Label-specific lexicon**: Build specialized Hebrew lexicon (plant names, star names, body parts) and test against labels.
3. **Paragraph syntax analysis**: Detect syntactic patterns in paragraph text (verb-initial, noun phrases, prepositions).
4. **Paper finalization**: 13-page draft with meta-analysis section ready for submission.

## Caveats

This is exploratory research, not a confirmed decipherment. Key limitations:

1. **Text does not read as Hebrew**: Despite statistically significant signal (z=3.6-4.4), the decoded text produces incoherent glosses. Best passages with 100% lexicon coverage read as disconnected word lists, not sentences.
2. **Lexicon inflation**: The 491K-form lexicon inflates both real (45.7%) and random (29.9%) match rates. Random Hebrew strings match at 26.8%. The "honest" assessment uses glossed-only (28K) lexicon: 24.3% match, z=4.4.
3. **Italian ruled out**: 4.5% match rate (z=3.82), vowel ratio 0.302 vs expected 0.468 — too consonantal.
4. **Multiple scribes**: Davis (2020) identified 5 scribal hands. A single monoalphabetic mapping may not apply to all scribes equally.
5. **Competing hypothesis**: The Naibbe cipher (Greshko 2025, Cryptologia) demonstrates that verbose homophonic substitution can produce Voynich-like ciphertext from Latin/Italian using 15th-century materials. If the real cipher is homophonic, our monoalphabetic approach captures only residual signal.
6. **3 Hebrew letters unmapped**: zayin, tsade, and qof — all strategies exhausted.
7. **The ii/i split** (Phase 7) is the least proven part of the mapping.
8. **SmS (שמש) claim downgraded**: ~5 occurrences, not zodiac-exclusive, alternative Hebrew sun words absent.

The strongest remaining evidence is the z-score stability across lexicon sizes (3.6-4.4) and the monkey-test differential (real mapping = 2.2-3.4x random strings, while random mappings ≈ 1.1x). This rules out pure chance but leaves open whether the signal reflects a partially correct mapping, a more complex cipher partially captured, or a structural artifact of the EVA transcription.

## References

- [Beinecke Digital Library - MS 408](https://collections.library.yale.edu/catalog/2002046)
- [EVA Transcription Archive](http://www.voynich.nu/transcr.html)
- Currier, P. (1976). *Papers on the Voynich Manuscript*
- D'Imperio, M.E. (1978). *The Voynich Manuscript: An Elegant Enigma*. NSA.
- Davis, L.F. (2020). How Many Glyphs and How Many Scribes? Digital Paleography and the Voynich Manuscript. *Manuscript Studies*, 5(1). [MUSE](https://muse.jhu.edu/article/754633)
- Greshko, M.A. (2025). The Naibbe cipher: a substitution cipher that encrypts Latin and Italian as Voynich Manuscript-like ciphertext. *Cryptologia*. [DOI](https://doi.org/10.1080/01611194.2025.2566408)
- Montemurro, M.A. & Zanette, D.H. (2013). Keywords and co-occurrence patterns in the Voynich Manuscript. *PLoS ONE*, 8(6).
- Rugg, G. (2004). An elegant hoax? *Cryptologia*, 28(1), 31-46.
- Stolfi, J. (1998). *An Interlinear Archive of Voynich Manuscript Transcriptions in EVA*
- Takahashi, T. (1998). *Complete EVA transcription of the Voynich Manuscript*

## License

MIT License. See [LICENSE](LICENSE).
