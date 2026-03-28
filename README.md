# Voynich Toolkit

[![DOI](https://img.shields.io/badge/DOI-10.1080%2F01611194.2025.2566408-blue)](https://doi.org/10.1080/01611194.2025.2566408)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The [Voynich Manuscript](https://en.wikipedia.org/wiki/Voynich_manuscript) (Beinecke MS 408, Yale) is a 600-year-old book that nobody can read. 240 pages of text in an unknown script, with illustrations of plants, stars, and naked people in pools. Carbon-dated to 1404-1438. Every attempt to decipher it has failed.

This toolkit doesn't try to decipher it either. It tests hypotheses — mathematically. Someone claims the text is Hebrew? Let's measure. Someone says one of the scribes was just drawing pretty patterns? Let's check the entropy. Someone says the lines work like inventory lists? Let's build a null model and compare.

The goal is simple: **take every testable claim about the Voynich and either confirm it, deny it, or show it's inconclusive.** No speculation, no "I think it might be...", just numbers.

## What we've tested so far

| # | Hypothesis | Result | What it means | Details |
|---|-----------|--------|---------------|---------|
| A | The EVA-to-Hebrew mapping captures a real signal | **Signal real at 3-4 letters, collapses at 5+** | There's a pattern (z=5.72) but it doesn't behave like Hebrew. 0/7 grammatical prefixes in expected range. No page is readable. | [hand_zscore](output/stats/hand_zscore_summary.txt) |
| A-sem | The decoded text matches a Semitic language | **No.** Closest to random. | Hebrew is *further* from the decoded text than a uniform random distribution. Shin at 41% kills it. | [semitic_kl_test](output/stats/semitic_kl_test_summary.txt) |
| B | There's an Italian layer underneath | **Weak signal** (z=3.96) | Real but depends entirely on Hypothesis A. If that mapping is wrong, this falls too. | — |
| C | Different scribes use different writing systems | **No.** All 8 hands show the same signal. | Tested all 5 Davis scribes + 3 unattributed hands. 8/8 significant. The system is consistent. | [hand_zscore](output/stats/hand_zscore_summary.txt) |
| D | One or more scribes just drew decorative patterns | **No.** Every hand has real structure. | No hand has entropy near zero. All 8 have distinct bigram signatures (each writes differently, but all write *something*). | [hand_structure](output/stats/hand_structure_summary.txt) |
| Reg | Some sections are registers/inventories, not prose | **Consistent but not proven** | Sections B, H, S, P score high on three independent register metrics. But this structure could also come from formulaic prose, litanies, or nomenclature. | [register_test](output/stats/register_test_summary.txt) |
| Reg-hand | Different scribes specialise as "labellers" vs "talliers" | **No specialisation found** | All hands write both unique first-words and repeated continuations. No universal tally token. Hand 2 is the most repetitive. | [hand_register](output/stats/hand_register_summary.txt) |
| Currier-7a | No word repetition crosses a line boundary | **Confirmed** (z=-3.60) | 11 repeats out of 4,499 boundaries — way fewer than the 30 expected by chance. Lines are self-contained units. | [currier_line_test](output/stats/currier_line_test_summary.txt) |
| Currier-7b | Some characters are line-end markers | **Confirmed** — 'm' at z=+55.5 | The character 'm' appears at line-end 66.5% of the time (1,079 occurrences). That's not random. | [currier_line_test](output/stats/currier_line_test_summary.txt) |
| Currier-7c | Split gallows appear only at paragraph start | **Not confirmed** | 87% of split gallows are on continuation lines, not paragraph starts. Currier's observation doesn't hold. | [currier_line_test](output/stats/currier_line_test_summary.txt) |
| ?-hand | Hand ? (unattributed pages) is a single scribe | **Probably not** | Hand ? covers all sections, all Currier languages, and has entropy +12 standard deviations above expected. Split by section, different parts behave in opposite ways. | [hand_unknown](output/stats/hand_unknown_summary.txt) |
| ?C=H3 | Some unattributed pages were actually written by Hand 3 | **Clustering says yes** | Section C pages of Hand ? cluster with Hand 3 at k=3 and k=4. Their entropy is nearly identical (8.62 vs 8.64). | [hand_clustering](output/stats/hand_clustering_summary.txt) |
| Stolfi | Stolfi's paragraph markers reflect real boundaries | **Yes.** Validated. | Character frequencies, line lengths, vocabulary, and simple gallows all differ significantly between @P, +P, and =P lines. | [stolfi_paragraph_test](output/stats/stolfi_paragraph_test_summary.txt) |
| Stolfi-gal | Simple gallows are paragraph-start markers | **Yes** (z=+10.73) | t, k, p, f appear 18% more often on first lines of paragraphs. They work like capital letters. Split gallows (cth, ckh, cph, cfh) do NOT. | [stolfi_paragraph_test](output/stats/stolfi_paragraph_test_summary.txt) |
| Para-coh | Lines within a paragraph share structural properties | **Yes.** +178% vocabulary, +28% bigrams | Lines in the same paragraph use similar characters and word patterns — but never repeat the same word across boundaries. Compatible with structured documents, but other explanations possible. | [paragraph_coherence](output/stats/paragraph_coherence_test_summary.txt) |
| M&Z-MI | Words carry section-specific information (Montemurro 2013) | **Confirmed** (z=+40.24) | Total MI = 0.159 bits vs null 0.032. Words like "shedy" are section-specific. The text has thematic structure. | [montemurro_test](output/stats/montemurro_test_summary.txt) |
| M&Z-links | Sections with similar illustrations share vocabulary | **Not confirmed** | Herbal-Pharma (both have plants) are NOT specially linked (z=+0.26). Astro-Zodiac are LESS similar than random (z=-6.02). | [montemurro_test](output/stats/montemurro_test_summary.txt) |
| Rugg | A Cardan grille (Rugg 2004) can reproduce all confirmed properties | **No.** 10/16 reproduced, 6 missing. | The grille produces Zipf, slot grammar, section MI, and Currier A/B — but fails on line-level structure ('m' end-markers, gallows paragraph markers, line self-containment) and vocabulary size (181 types vs 8,493). | [rugg_test](output/stats/rugg_test_summary.txt) |

## What the data show

Structural findings (hypothesis-independent, all with permutation tests):

- The manuscript was written by at least 5 different people ([Davis 2020](https://muse.jhu.edu/article/754633))
- All of them used the same writing system (Hypothesis C = closed)
- None of them was just drawing — every scribe produces structured, non-trivial text
- Words rarely cross line boundaries (z=-3.60, 11 out of 4,499) — lines behave as self-contained units
- The character 'm' appears at line-end 71% of the time (z=+55.5) — consistent with a line-end marker
- Simple gallows (t, k, p, f) appear more often at paragraph starts (z=+10.73)
- Split gallows (cth, ckh, cph, cfh) are NOT paragraph markers — their function is unknown
- Paragraphs group lines with similar character and vocabulary profiles (+178% Jaccard within paragraphs)
- But lines within a paragraph almost never share exact words across boundaries
- Words are section-specific: the text has thematic structure (MI z=+40.24, Montemurro confirmed)
- Sections with similar illustrations do NOT share more vocabulary (Montemurro's specific prediction wrong)
- The unattributed pages (Hand ?) are probably a mix of multiple scribes
- Some sections have statistical properties consistent with structured documents (registers, catalogues, formularies) — but this is a compatible hypothesis, not a conclusion

## Where we disagree with published studies

Some of our numbers contradict things that have been repeated in the literature for decades. Not opinions: permutation tests with z-scores.

**Currier (1976)** said split gallows only show up at the start of paragraphs. We counted: 87% of them (1,869 out of 2,149) sit on continuation lines, not paragraph starts. z=+0.36, not even close to significant. What actually marks paragraph starts is simple gallows (t, k, p, f) at z=+10.73. Currier lumped them together, and the field has been repeating it for 50 years. His line-boundary observations are solid, though: 'm' as line-end marker (z=+55.5) and self-contained lines (z=-3.60) both check out.

**Montemurro & Zanette (2013)** predicted that sections with similar illustrations would share vocabulary. Turns out it's the opposite for the most obvious pair: Astronomical and Zodiac sections are LESS similar than random (z=-6.02). Herbal and Pharmaceutical, both full of plant drawings, show no special link either (z=+0.26). Their core insight is right: words ARE section-specific (MI z=+40.24). But the illustrations don't predict which sections talk about similar things. The sections are isolated, not linked.

**Davis (2020)** identified 5 scribes, and we confirm they all use the same system. But Hand ? (the unattributed pages) is probably not one person: z=+12.25, with opposite statistical signs depending on which section you look at. Also, Hand 1 drives 48.5% of the signal while Hand 4 (same Language A) only hits 21.7% match rate. If they were using the same cipher key, those numbers should be close. They're not.

**Stolfi (1998)** paragraph markers are real: chi²=1583.7, no question. But his system doesn't capture whatever split gallows are doing. They follow a different logic entirely.

Nobody knows what split gallows do. Not paragraph markers, not section markers, not language markers. 2,149 of them in the manuscript, concentrated in the herbal section. They tend to avoid line-initial position (z=-13.39) and cph/cth/cfh tend toward word-start — but ckh behaves differently. That's one of the big open questions.

## What we don't know

- What language the text is in (if any)
- Whether the text has linguistic content at all (could be notation, mnemonic, cipher, or something else)
- Why the EVA-to-Hebrew mapping produces a signal at 3-4 letters but fails everywhere else
- ~~Whether a rule-based generator (Rugg 2004) can reproduce all the confirmed structural properties~~ **Tested: 10/16 reproduced, 6 missing** — the grille fails on line-level structure and vocabulary size
- What the actual function of split gallows is

## Installation

```bash
git clone https://github.com/antenore/voynich-toolkit.git
cd voynich-toolkit
pip install -e .
```

Python 3.10+. Database (voynich.db, 58.7 MB) is generated by the analysis commands.

## Commands

```bash
# Per-hand analysis (Phase 27)
voynich --force hand-characterization    # Phase 0: entropy, Zipf, TTR per scribe
voynich --force hand-zscore              # Phase 1a: does every scribe show the same signal?
voynich --force semitic-kl-test          # Phase 1b: does the mapping look like any Semitic language?
voynich --force hand-structure           # Phase 2: entropy + Zipf vs null model per hand
voynich --force hand-unknown             # Phase 2e: is Hand ? one scribe or many?
voynich --force hand-positional          # Phase 4: character positions + trigrams per hand
voynich --force hand-clustering          # Phase 5: do hands cluster by scribe, section, or language?
voynich --force register-test            # Phase 6: do some sections look like inventories?
voynich --force hand-register            # Phase 6b: do different hands play different roles?
voynich --force currier-line-test        # Phase 7: Currier's 1976 line-boundary observations
voynich --force stolfi-paragraph-test   # Phase 7b: validate Stolfi's paragraph markers
voynich --force paragraph-coherence-test # Phase 7c: intra-paragraph coherence
voynich --force montemurro-test          # Phase 8: Montemurro & Zanette (2013) verification
voynich --force rugg-test               # Phase 27.9: Rugg grille falsification test

# Earlier phases (validation, mapping, cross-analysis)
voynich --force full-decode              # Decode entire corpus
voynich --force cross-validation         # Hand-based + random 50/50 split
voynich --force currier-split            # Currier A/B language split
voynich --force veneto-italian-test      # Italian/Veneto cipher-native test
```

All commands accept `--force` to re-run.

## Project structure

```
voynich-toolkit/
  src/voynich_toolkit/     # All analysis modules (~50 Python files)
  eva_data/
    LSI_ivtff_0d.txt       # EVA transcription (Takahashi 1998, IVTFF format)
  output/stats/            # Results: JSON + TXT summaries
  voynich.db               # SQLite database (58.7 MB, 500K+ rows)
  archive/                 # Old README, paper drafts, status files
```

## Source data

- **EVA transcription**: Takahashi (1998) — 191,545 characters, 37,025 words, 225 folios
- **Hebrew lexicon**: STEPBible + Jastrow + Sefaria + Klein — 494,469 consonantal forms
- **Scribe attribution**: [Davis (2020)](https://muse.jhu.edu/article/754633) — 5 identified hands

## References

- Davis, L.F. (2020). "How Many Glyphs and How Many Scribes?" *Manuscript Studies*, 5(1). [MUSE](https://muse.jhu.edu/article/754633)
- Currier, P.H. (1976). "Some Important New Statistical Findings." [Papers](https://www.voynich.nu/extra/curr_main.html)
- Greshko, M.A. (2025). "The Naibbe Cipher." *Cryptologia*. [DOI](https://doi.org/10.1080/01611194.2025.2566408)
- Montemurro & Zanette (2013). "Keywords and Co-Occurrence Patterns." *PLoS ONE*, 8(6). [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066344)
- Bowern, C. (2021). "The Linguistics of the Voynich Manuscript." *Annual Review of Linguistics*. [PDF](https://alumniacademy.yale.edu/sites/default/files/2021-07/The%20Linguistics%20of%20the%20Voynich%20Manuscript.pdf)

## Related work

- **[epilectrik/voynich](https://github.com/epilectrik/voynich)** (Joe DiPrima) — Independent computational analysis with a different hypothesis (closed-loop control programs).

## License

MIT. See [LICENSE](LICENSE).
