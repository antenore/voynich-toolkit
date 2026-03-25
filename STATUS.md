# Voynich Toolkit — Status

## Current Metrics

| Metric | Value |
|--------|-------|
| Match rate (honest 45K lexicon) | ~20% |
| z-score DictaLM valid-only (211 forms) | **56.71** |
| z-score Italian cipher-native (pre-1500) | **3.96** |
| Combined Heb+Ita token coverage | 43.5% |
| Validation scorecard | 7/14 |
| Paper | 19 pp., arXiv-ready, endorser needed |
| Active work | Phase 26: Crib attack (known-plaintext), 79% word-level on f27r |
| Active work | Phase 25: Italian/Veneto test completed |
| Active work | Visual analysis page-by-page (Phase 24) |

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
Traditional decode → gloss → interpret pipeline hit ceiling at ~20% match rate. The crib attack (Phase 26) flips the direction and shows promise at 79% word-level matching on first tested folio.
