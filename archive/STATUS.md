# Voynich Toolkit — Status

## Current Metrics

| Metric | Value |
|--------|-------|
| Match rate (honest 45K lexicon) | ~20% (80% unmatched) |
| Cross-validation z-score | **5.72** (13/17 letters optimal in both splits) |
| z-score DictaLM valid-only (211 forms) | 56.71 (cherry-picked 19.4%; full set z=3.70) |
| DictaBERT syntax | 2.74/6 (random=2.5, real Hebrew=6.0) — **word order random** |
| z-score Italian cipher-native (pre-1500) | 3.96 |
| Botanical specificity | z=-6.0 (**anti-concentration**) |
| Zodiac validation | 1/12 (not significant, p=0.071) |
| Validation scorecard | 7/14 FDR-significant |
| Paper | 21 pp., needs corrections before submission |
| Hebrew prefix test | **0/7 in range** — shin 41.3% (exp 1-5%), he(article) 0% |
| Length-stratified signal | z=8.14 at 4 letters, **z=0.01 at 2 letters** (pure noise) |
| 5+ letter match rate | 0.4% (71/16,804) — signal collapses at longer words |
| Text readability | **Unreadable** — no Hebraist can read any page |
| Active work | Phase 26: Crib attack — signal confirmed, text unreadable |

## Phase History

### Phase 26 — Crib Attack / Known-Plaintext (2026-03-25)
**Status: Active**

New approach: generate plausible medical Hebrew texts, encode through cipher, compare against real EVA.

- New modules: `crib_encoder.py` (Hebrew consonantal → EVA), `crib_attack.py` (full pipeline)
- f27r (herbal, 93 words): 29 strong + 31 weak anchors, 62 target words
- Word-level match: **49/62 = 79%** (24 exact d=0 + 25 near d=1)
- Page-level match: up to 65% with full-page generation
- Key discovery: **t↔J (tav↔tet) interchangeability** — 5+ near-matches differ only by this swap
  - Consistent with phonetic merger in medieval Italian Hebrew (/t/ for both)
- Next: expand to more pages, feed confirmed words back as anchors

### Phase 25 — Italian/Veneto Cipher-Native Test (2026-02-xx)
**Status: Completed**

- z-score Italian cipher-native (pre-1500): **3.96**
- Combined Heb+Ita token coverage: 43.5%
- Tables: `veneto_italian_test` (23 rows), `veneto_italian_matches` (30 rows)

### Phase 24 — Visual Analysis Page-by-Page (2026-02-23)
**Status: Ongoing**

Complementary visual-semantic analysis: screenshot + EVA label decode + iconographic interpretation.

### Phase 22–23 — Scribal Correction, Lexicon Expansion, Angelic Test
**Status: Exhausted (null results)**

All three avenues produced no significant improvement. See `memory/phase22_23_exhausted.md`.

### Phase 20–21 — DictaLM Validation
**Status: Completed**

- DictaLM z-score (valid-only, 211 forms): **56.71**
- DictaBERT: 2.74/6
- Judeo-Arabic and Ladino hypotheses rejected
- See `memory/phase20_21_results.md`

### Phase 15–16 — Full Results
See `memory/phase15_16_details.md`.

### Phase 11 — Reality Check
See `memory/phase11_reality_check.md`.

### Phase 9–10 — Diagnosis
See `memory/phase9_diagnosis.md`.

## Known Issues / Exhausted Avenues
- Scribal error correction: null result
- Lexicon expansion beyond 45K: diminishing returns
- Angelic/mystical Hebrew test: not significant
- Naibbe verbose cipher hypothesis: rejected
- Judeo-Italian and Ladino: rejected

## Diagnosis
Traditional decode → gloss → interpret pipeline hit ceiling at ~20% match rate. The crib attack (Phase 26) produces 1,758 exact matches with zero-controls at zero, confirming the cipher relationship is non-random. However, the text remains unreadable (DictaBERT 2.74/6), botanical specificity is negative (z=-6.0), zodiac validation fails (1/12), and all English "translations" are AI paraphrases conditioned on section type. Status: **real signal, unknown meaning**.
