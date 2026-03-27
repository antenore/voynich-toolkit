# Voynich Manuscript Analysis Toolkit

> **This hypothesis is wrong.** After 26 phases of analysis and months of work, the Hebrew cipher hypothesis does not hold up. The decoded text has no Hebrew grammatical structure (0/7 expected prefixes in range, the definite article is completely absent, shin at 41% initial frequency), the signal only works on short words where coincidental matches are easy, and no one can read a single page. The "translations" in this repo are AI-generated paraphrases, not real readings. I'm leaving the code and data up because the infrastructure (58MB database, permutation framework, crib attack pipeline) might be useful to someone testing a different hypothesis. See the [honest baseline](#honest-baseline) below for what actually works and what doesn't.

A computational toolkit for analyzing the Voynich Manuscript (Beinecke MS 408, Yale University). Originally built to test the hypothesis that the text is a Hebrew consonantal cipher. That hypothesis failed, but the toolkit and data remain available for other approaches.

**[Read the paper (PDF)](paper/paper.pdf)** - *A Statistical Evaluation of the Hebrew Cipher Hypothesis for the Voynich Manuscript* (pre-Phase 26, being revised)

## The short version

We built a character mapping (19 EVA characters to 19 Hebrew consonants) through two independent paths that converge at 84% agreement. Statistical validation is solid (z=3.6-4.4 across all lexicon sizes, p<0.005). But for 25 phases the decoded text read as incoherent word lists, not sentences. We tried everything: lexicon expansion, scribal correction, DictaLM validation, Italian substrate analysis, Currier A/B split, cross-validation. Nothing produced readable text.

Then we flipped the approach. Instead of decoding and hoping for meaning, we asked: what would a 15th-century physician actually write on a page showing a plant illustration? We generated plausible medical Hebrew texts, encoded them through our cipher, and compared against the real manuscript. 70% of unknown words matched at edit distance 1 or less, across 918 words on 6 pages.

On folio 27r, the crib match rate reached 97% — but the honest breakdown is: 68% pre-existing anchor words (already in Hebrew dictionaries), 17% new exact matches from the crib attack, and 16% near-matches (d=1, off by one letter). An AI interprets the decoded consonants as a pharmaceutical extraction recipe, but **this interpretation is unverified and potentially circular** — the AI was told the page shows a plant, so it generated pharmaceutical vocabulary.

Vocabulary reinterpretations worth investigating: `mwk` ("be poor" in biblical Hebrew, possibly "cotton" in rabbinic), `SEn` ("to lean", possibly "to apply"), `Sk` (unclear, possibly "mixture"). These are **hypotheses**, not confirmed meanings.

## How we got here

### Phases 1-4: The text is real language

Statistical fingerprinting of 191,545 characters and 37,025 words from the Takahashi EVA transcription:

| Metric | Value | What it means |
|--------|-------|---------------|
| Index of Coincidence | 0.0771 | Natural language (0.065-0.077), not random (0.038) |
| Morpheme root length | 3.14 chars | Compatible with Semitic triconsonantal roots |
| Cipher scorecard | Homophonic: 1.0, Polyalphabetic: 0.13 | Not polyalphabetic, not glossolalia |
| Homophone groups | 19 EVA chars in 10 groups | ~47% of characters are allographic variants |

### Phases 5-7: Two paths, one mapping

We ran two completely independent decipherment attempts:

| Path | Method | Result |
|------|--------|--------|
| Italian | Hill-climbing + plant-name constraints | 19/19 chars mapped |
| Hebrew | Homophone groups + frequency scoring vs lexicon | 19/19 chars mapped |
| **Agreement** | **16/19 identical mappings** | **84.2% convergence** |

The remaining 3 were resolved: f=allograph of p (lamed), q=prefix (stripped), ii=he / standalone i=resh.

### Phases 8-25: Signal is real, text is unreadable

25 phases of validation, refinement, and reality checks. The mapping produces a statistically significant signal - the decoded text matches Hebrew at rates 1.7-3.4x above random baselines (z=3.6-4.4, stable across lexicon sizes). But the best passages produced word salad, not sentences.

Everything we tried:

| Phase | What we tested | Outcome |
|-------|---------------|---------|
| 11 | Reality check: is the signal real? | Yes (z=4.4 even with honest 28K lexicon) |
| 12 | Currier A/B split | Both significant, Hand 1 drives the signal |
| 13 | Naibbe verbose cipher | Rejected (z=12.1 vs Monte Carlo) |
| 14 | Judeo-Italian substrate | Partial: 7.4% match, z=4.59 |
| 15 | Reading direction | RTL confirmed (z=22.97) |
| 16 | Null model, section entropy | Signal above null model. Sections not uniform |
| 17-18 | Layout-aware, meta-analysis | Labels 13.2% vs paragraphs 24.7% |
| 20-21 | DictaLM validation | z=56.71 on 211 confirmed Hebrew forms |
| 22 | Cross-validation, convergence | 13/17 optimal on held-out data |
| 23 | Scribal correction, lexicon expansion | Null result |
| 25 | Italian/Veneto cipher-native test | z=3.96 pre-1500, combined coverage 43.5% |

At this point we had a z-score of 56.71 on DictaLM-validated forms but still couldn't read a single sentence. We were close to calling it: the mapping captures something real, but it's not enough for decipherment.

### Phase 26: The breakthrough

The idea was simple, almost obvious in hindsight. We know what section each page belongs to (herbal, balneological, pharmaceutical). We know what the illustrations show. We have anchor words with confirmed meanings. So instead of decoding and searching dictionaries, we asked: what would the author have written?

We built a reverse cipher encoder (Hebrew consonantal to EVA) and used Claude Sonnet to generate plausible medical texts that a 15th-century Italian Jewish physician might write. Then we encoded those texts and compared against the real manuscript, word by word.

Results across 6 pages (918 target words):

| Folio | Section | Words | Match rate (d<=1) |
|-------|---------|-------|-------------------|
| f15v | herbal | 41 | 75.6% |
| f75r | balneological | 349 | 73.6% |
| f45r | herbal | 67 | 71.6% |
| f27r | herbal | 62 | 69.4% |
| f77r | balneological | 250 | 62.4% |
| f99r | pharmaceutical | 149 | 55.7% |
| **Total** | | **918** | **67.3%** |

The near-match analysis (295 cases at edit distance 1) revealed three systematic scribal equivalences, all well-documented in medieval Hebrew paleography:

- **t / J (tav / tet)**: phonetic merger in Italian Hebrew, both = /t/
- **X / E (chet / ayin)**: guttural confusion, both lost in Italian pronunciation
- **w / r (vav / resh)**: visually near-identical in cursive Hebrew script

With these equivalences, f27r reached 97% crib match rate (see honest breakdown above).

## AI interpretation of f27r (UNVERIFIED)

> **WARNING: The following is Claude Sonnet's AI paraphrase of Hebrew consonants, NOT a verified translation.** No Hebraist has confirmed this reading. The vocabulary assignments (mwk=cotton, etc.) are hypotheses. The AI was told this is a herbal page, which may bias the interpretation toward pharmaceutical content.

> Cotton rests as a sign, cotton rests and appears clear-bright - pour chalk.
>
> Pour, anoint, cook the mixture with strong drink, lower the heat, let the residue settle.
>
> Remove what falls away, what remains rests - the solution clear and limpid.
>
> The mixture reduces, anoint with cotton while it rests, producing liquid essence.
>
> Grind the mixture, soften the essence, pour until clear.
>
> Pour when opened, rest the mixture. The mixture is completed.
>
> Clear, when the edges flow, pour the poured liquid until sweet.

The AI interprets f75r (balneological page) as a medicinal bathing protocol. Same caveat: this is an AI paraphrase conditioned on section type, not a verified reading.

## Proposed vocabulary (5 pages) — HYPOTHESES, NOT CONFIRMED

These words appear consistently across multiple pages. Cross-page consistency could reflect: (a) real manuscript content, or (b) AI self-consistency (same model generates similar vocabulary for similar sections).

| Hebrew | Meaning | Old dictionary gloss | Pages |
|--------|---------|---------------------|-------|
| mwk | cotton/soft filtering material | "be poor" | f27r x6, f45r, f15v, f99r |
| Spk | to pour | to pour | f27r x3, f45r, f75r x15+, f99r |
| SEn | to rest/apply (medicine) | "to lean" | f27r x5, f45r, f75r x5+ |
| swk | to anoint | to anoint | f27r x2, f15v x5, f45r, f99r |
| bhyr | clear/limpid (liquid) | bright, brilliant | all 5 pages |
| Sk | mixture/preparation | (grammar code) | f27r x6, f75r x6+, f15v |
| gyr | chalk/ite (clarifier) | chalk, plaster | f27r, f99r |
| Skr | alcohol/strong drink | "drunken" | f27r, f45r |
| mr | bitter (quality indicator) | bitter | f75r x5+, f45r, f99r |
| Spt | to pour/set | to set | f27r, f75r, f99r |
| mytq | sweet (completion test) | sweet | f27r |
| wdryn | roses | (not found) | f75r |

The AI-generated interpretations show consistent structure across herbal pages. This may reflect genuine manuscript content, or AI self-consistency (same prompts, same model).

## The mapping

19 EVA characters mapped to 19 of 22 Hebrew consonants. Read right-to-left.

| EVA | Hebrew | Name | Italian | Notes |
|-----|--------|------|---------|-------|
| a | y | yod | i | convergent |
| c | A | aleph | a | convergent |
| d | r | resh | r | convergent |
| e | p | pe | p | convergent |
| f | l | lamed | l | allograph of p |
| g | X | chet | k | convergent |
| h | E | ayin | e | convergent |
| ii | h | he | e | composite glyph |
| i | r | resh | r | allograph of d |
| ch | k | kaf | k | digraph |
| k | t | tav | t | convergent |
| l | m | mem | m | convergent |
| m | g | gimel | g | convergent |
| n | d/b | dalet/bet | d/b | bet at word-initial |
| o | w | vav | o | convergent |
| p | l | lamed | l | convergent |
| q | - | - | prefix | stripped |
| r | h/s | he/samekh | e/s | samekh at word-initial |
| s | n | nun | n | convergent |
| t | J | tet | t | convergent |
| y | S | shin | s | convergent |

Scribal equivalences (confirmed by near-match analysis): t/J (tav/tet), X/E (chet/ayin), w/r (vav/resh). Three Hebrew letters unmapped: zayin, tsade, qof.

## Installation

```bash
git clone https://github.com/antenore/voynich-toolkit.git
cd voynich-toolkit
pip install -e .
```

Requires Python 3.10+. For the crib attack pipeline, you also need an Anthropic API key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

### Statistical analysis (Phases 1-4)

```bash
voynich eva                           # EVA transcription analysis
voynich word-structure                # Word tokenizer + positional profiles
voynich language-fingerprint          # Entropy, IoC, Heaps law
voynich cipher-hypothesis             # Cipher scorecard + morphology
```

### Decipherment (Phases 5-9)

```bash
voynich prepare-lexicon               # Hebrew lexicon (STEPBible + Jastrow + Sefaria + Klein)
voynich decipher --restarts 20        # Hebrew hill-climbing
voynich champollion                   # Plant-name constrained decoder (Italian path)
voynich fuzzy-decode                  # Complete mapping + fuzzy match
voynich full-decode                   # Decode full corpus with 19-char mapping
```

### Validation (Phases 10-25)

```bash
voynich anchor-words                  # Domain anchor words + permutation
voynich semantic-coherence            # Semantic coherence + permutation
voynich currier-split                 # Currier A/B split analysis
voynich naibbe-test                   # Naibbe verbose cipher hypothesis
voynich cross-validation              # Hand-based + random cross-validation
voynich dictalm-validate              # DictaLM validation via API (~55min)
voynich veneto-italian-test           # Cipher-native Italian/Veneto test
```

All validation commands accept `--force` to re-run.

### Crib attack / known-plaintext (Phase 26)

```bash
# Word-level: generate 10 candidates per unknown word, score against real EVA
python -m src.voynich_toolkit.crib_attack wordlevel f27r 10

# Page-level: generate 5 full-page variants
python -m src.voynich_toolkit.crib_attack page f27r 5

# Use a different model (Haiku for cost, Sonnet default)
python -m src.voynich_toolkit.crib_attack wordlevel f27r 10 claude-haiku-4-5-20251001
```

## Project structure

```
voynich-toolkit/
  src/voynich_toolkit/
    cli.py                  # Unified CLI entry point
    full_decode.py          # Canonical 19-char decoder (EVA to Hebrew)
    crib_encoder.py         # Reverse cipher (Hebrew to EVA) + allograph/scribal variants
    crib_attack.py          # Known-plaintext attack pipeline (word-level + page-level)
    permutation_stats.py    # Random mapping permutation framework
    ...                     # 40+ analysis modules (Phases 1-25)
  eva_data/
    LSI_ivtff_0d.txt        # EVA transcription (IVTFF format, Takahashi 1998)
  paper/paper.tex           # Academic paper (being revised for Phase 26)
  scripts/                  # Utility and interpretation scripts
  output/stats/             # Generated reports (JSON, TXT, LaTeX)
  voynich.db                # SQLite database (58.7 MB, 512K rows)
```

## Source data

- **EVA IVTFF**: Takahashi (1998) - 191,545 characters, 37,025 words, 225 folios
- **Hebrew lexicon**: STEPBible + Jastrow + Sefaria corpus + Klein - 494,469 consonantal forms
- **Italian lexicon**: Dante concordance + TLIO + medieval terms - 60,738 forms
- **North Italian dialects**: kaikki.org (Venetian, Ligurian, Emilian, etc.) - 20,442 forms

## What's still open

**This is not a confirmed decipherment.** Honest assessment:

1. **3 Hebrew letters still unmapped** (zayin, tsade, qof). They may be hiding behind EVA characters we've assigned to other letters.
2. **The crib attack relies on generated text.** Sonnet generates what a physician *might* write. The 70% match rate is strong evidence, but we're measuring against our own expectations, not ground truth.
3. **Multiple scribes.** Davis (2020) identified 5 hands. Our mapping works best on Hand 1 (86 herbal pages). Other hands may use different conventions.
4. **The pe anomaly.** Near-match analysis shows three different Hebrew letters (shin, resh, chet) all substituting toward pe (EVA `e`) at rates of 9-14 per pattern. This may indicate a mapping issue with EVA `e` or an unmapped letter.
5. **We haven't read the whole manuscript.** Five pages with coherent readings doesn't mean the approach works everywhere. The pharmaceutical section (f99r) scored lower at 56%.
6. **Competing hypotheses exist.** The Naibbe cipher (Greshko 2025, Cryptologia) shows verbose homophonic substitution can produce Voynich-like text from Latin/Italian. If the real cipher is homophonic, our monoalphabetic mapping captures only residual signal.

7. **DictaBERT syntax is random.** Decoded passages score 2.74/6 on syntactic quality (real Hebrew = 6.0, random consonants = 2.5). Word order carries no information.
8. **Botanical specificity is negative.** Herbal pages have *fewer* botanical dictionary matches than other sections (z=-6.0).
9. **Zodiac validation fails.** Only 1/12 zodiac signs matched (p=0.071, not significant).
10. **Hebrew prefix distribution is wrong.** 0/7 grammatical prefixes fall in the expected frequency range. Shin appears at 41.3% of word-initial positions (expected 1-5%). The definite article he is completely absent (expected 5-12%). Lamed (to/for) is at 0.1% (expected 4-10%). The decoded text does not have the grammatical structure of any form of Hebrew.
11. **2-letter words are pure noise.** Match rate 56.6% real vs 56.4% random (z=0.01). The 1,535 "matched" 2-letter tokens contribute nothing.
12. **Signal collapses at 5+ letters.** Only 71 matches out of 16,804 tokens (0.4%, p=0.055). If the mapping were correct, longer words should still match.

## Honest baseline

**What IS established:**
- Cross-validation z=5.72 on held-out data (13/17 letters optimal in both splits)
- **Strong signal at 3-4 letter words**: z=3.62 (3 letters, 2.3x ratio), z=8.14 (4 letters, 7.8x ratio)
- Permutation tests pass: anchor z=4.2, plant z=3.2, Currier A z=4.02, B z=3.85
- Crib attack: 1,758 exact matches vs 0 in 700K random attempts
- Domain specificity (partial): balneological z=8.53, astronomical z=3.51

**What is NOT established:**
- Readable text (no Hebraist can read any page)
- Correct interpretation (all English text is AI paraphrase)
- Hebrew grammatical structure (0/7 prefixes in range, shin 41.3%)
- Signal at word length 5+ (collapses to 0.4%)
- Zodiac validation (1/12 signs)
- Botanical specificity (z=-6.0, negative)
- Syntax (DictaBERT 2.74/6, random word order)
- That the manuscript is a "pharmaceutical handbook" (circular AI interpretation)

## Where this stands

The mapping captures a **real lexical signal at 3-4 letter words** (z=8.14) that cannot be explained by chance. But the decoded text has **no Hebrew grammatical structure** (wrong prefix distribution, random word order, no article). The signal may reflect a partial phonotactic overlap between the Voynich text's structure and Hebrew triconsonantal roots, rather than actual Hebrew content.

Further progress requires either:
1. **A Hebraist** examining the decoded consonants without AI assistance
2. **A different cipher model** (e.g., homophonic, polyalphabetic) that might recover the missing structure
3. **Known-text identification** — finding a passage that matches a known medieval Hebrew source

Without external expertise, the computational approach has reached its limit. The toolkit, database (512K rows), and all 26 phases of analysis are open source for anyone to investigate.

## Related work

- **[epilectrik/voynich](https://github.com/epilectrik/voynich)** (Joe DiPrima) - Independent computational analysis. Different hypothesis: the text encodes closed-loop control programs (possibly for distillation). Uses progressive constraint architecture, token morphology decomposition. Finding: illustration swap invariance (p=1.0). Complementary to our approach.
- Greshko, M.A. (2025). The Naibbe cipher. *Cryptologia*. [DOI](https://doi.org/10.1080/01611194.2025.2566408)
- Davis, L.F. (2020). How Many Glyphs and How Many Scribes? *Manuscript Studies*, 5(1). [MUSE](https://muse.jhu.edu/article/754633)
- Montemurro & Zanette (2013). Keywords and co-occurrence patterns. *PLoS ONE*, 8(6).
- D'Imperio, M.E. (1978). *The Voynich Manuscript: An Elegant Enigma*. NSA.

## License

MIT License. See [LICENSE](LICENSE).
