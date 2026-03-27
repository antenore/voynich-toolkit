# Translation Audit and Handoff for Repository Internationalization

## Purpose

This file is a translation-oriented audit of the repository. It is meant as a support document for an LLM or human translator who will convert the project to English for public release.

The goal is to make the repository understandable and usable for non-Italian readers without changing code behavior, command semantics, data formats, or research content.

## Translation Objective

Translate repository content that is currently written in Italian into English, with the following high-level rule:

- Public-facing documentation should be in English.
- Code comments, docstrings, and user-facing CLI/help/report strings should be in English.
- Italian lexicon data must remain in Italian.
- Code logic, identifiers, command names, data schemas, and linguistic data must not be altered unless a change is purely textual and behavior-preserving.

## Critical Constraints

These constraints should be treated as mandatory.

### 1. Do not change code logic

No translation step should modify:

- control flow;
- function names;
- class names;
- variable names;
- module names;
- import paths;
- CLI command names;
- option names;
- JSON keys that are consumed programmatically;
- CSV column names if they are parsed by code;
- any constant or symbol used as data rather than prose.

If a string is both human-readable and machine-consumed, assume it must remain unchanged unless verified otherwise.

### 2. Keep Italian lexical content in Italian

Italian lexical resources are part of the research/data layer and must remain in Italian. This includes:

- Italian word lists;
- glosses intended to represent Italian lexicon content;
- phonemic Italian forms;
- historical Italian examples used as lexical evidence;
- lexical entries embedded in JSON or TXT resources.

### 3. Preserve mixed-language research tokens

Do not translate or normalize:

- EVA tokens;
- Hebrew strings or transliterations;
- Aramaic strings or transliterations;
- Voynich folio IDs;
- manuscript section abbreviations;
- lexical match targets;
- example decoded tokens used as evidence.

Examples such as `mwk`, `bhyr`, `f27r`, `EVA`, `Currier A/B`, or transliterated lexical forms are data, not prose.

### 4. Be careful with examples inside docstrings

Some docstrings and comments may contain Italian lexical items as examples. Those examples should remain in Italian if they are illustrating Italian lexical data, not ordinary prose.

Good rule:

- translate explanatory prose around the example;
- preserve the lexical example itself.

## Main Finding from the Audit

Italian is present in this repository primarily in:

- documentation files;
- comments in Python files;
- Python docstrings;
- user-facing CLI/help strings;
- generated report files committed to the repository;
- Italian lexical datasets and Italian lexical examples.

Italian does not appear to be used systematically in identifiers. This is important because it means translation can focus on prose and user-facing text while leaving code structure intact.

## Recommended Translation Strategy

### Tier 1: Translate completely

Translate these files fully into English because they are documentation or human-readable reports intended for readers.

- `PUBBLICAZIONI.md`
- `EPILECTRIK_NOTES.md`
- `output/readings_merged.md` if this file is intended to remain in the public repo

### Tier 2: Translate prose inside source files

Translate only:

- comments;
- docstrings;
- CLI help text;
- console output intended for users;
- explanatory labels in reports.

Do not translate:

- identifiers;
- data constants;
- symbolic labels that are part of the method;
- lexical examples that are intentionally Italian.

### Tier 3: Review generated outputs

Many files in `output/stats` are generated artifacts. If they are part of the published repository and expected to be read directly by users, their human-readable prose should be in English. If they are not central to publication, the cleaner option may be to regenerate them after source translation or exclude them from the public-facing subset.

### Tier 4: Preserve lexical datasets

Files that exist to store Italian lexical resources should remain Italian. They may still need a short English README or note elsewhere, but their internal lexical content should not be translated.

## File Inventory

The sections below classify files by content type and recommended action.

## A. Documentation Files with Italian Prose

These files contain substantial Italian prose and should be translated fully.

### `PUBBLICAZIONI.md`

Type:

- documentation;
- publication planning notes.

Observed content:

- publication options;
- conference recommendations;
- outreach/publication workflow;
- status checklist.

Recommended action:

- translate fully into English;
- preserve proper names, paper title, platform names, and URLs;
- keep checklist semantics unchanged.

Risk level:

- low.

### `EPILECTRIK_NOTES.md`

Type:

- documentation;
- collaboration/research notes.

Observed content:

- comparison of hypotheses;
- architecture summaries;
- methodological notes;
- proposed cross-tests;
- collaboration notes.

Recommended action:

- translate fully into English;
- preserve quoted claims as faithfully as possible;
- preserve research labels such as `Currier B`, `Tier 0`, `AZC`, `HT`, `PREFIX+MIDDLE+SUFFIX`;
- preserve technical metrics and symbolic notation.

Risk level:

- medium, because it contains dense research terminology and some hybrid notation.

### `output/readings_merged.md`

Type:

- generated or semi-curated human-readable report;
- narrative output.

Observed content:

- substantial Italian prose;
- interpretation summaries;
- vocabulary tables;
- recurring structure summaries.

Recommended action:

- translate if the file will remain visible in the public repository;
- preserve all lexical tokens and manuscript evidence forms;
- preserve Hebrew/transliteration examples exactly;
- preserve confidence labels, but render them in English consistently.

Risk level:

- medium to high, because prose and evidence tokens are mixed line-by-line.

## B. Source Files Containing Italian Comments, Docstrings, or User-Facing Strings

These files contain Italian prose in source code. The code logic should not be changed. Translation should be limited to prose.

### Core configuration and pipeline files

- `src/voynich_toolkit/config.py`
- `src/voynich_toolkit/extract.py`
- `src/voynich_toolkit/segment_text.py`
- `src/voynich_toolkit/segment_glyphs.py`
- `src/voynich_toolkit/statistics.py`
- `src/voynich_toolkit/image_text_correlation.py`
- `src/voynich_toolkit/utils.py`

Typical Italian content:

- module docstrings;
- step descriptions;
- explanatory comments;
- user-facing messages.

Recommended action:

- translate all comments/docstrings/help strings to English;
- do not change config field names or filesystem paths.

### Analysis and reporting files

- `src/voynich_toolkit/word_structure.py`
- `src/voynich_toolkit/section_analysis.py`
- `src/voynich_toolkit/language_fingerprint.py`
- `src/voynich_toolkit/char_embeddings.py`
- `src/voynich_toolkit/eva_compare.py`
- `src/voynich_toolkit/cipher_hypothesis.py`
- `src/voynich_toolkit/hand1_deep_dive.py`
- `src/voynich_toolkit/scribe_analysis.py`
- `src/voynich_toolkit/null_model_test.py`

Typical Italian content:

- analysis descriptions;
- diagnostic text;
- summary headers;
- interpretation notes.

Recommended action:

- translate human-readable text only;
- preserve formulae, metrics, and token examples;
- preserve report structure unless there is a strong reason to standardize labels.

### Decipherment and lexicon preparation files

- `src/voynich_toolkit/decipher.py`
- `src/voynich_toolkit/italian_decipher.py`
- `src/voynich_toolkit/champollion.py`
- `src/voynich_toolkit/prepare_lexicon.py`
- `src/voynich_toolkit/prepare_italian_lexicon.py`
- `src/voynich_toolkit/cross_language_baseline.py`
- `src/voynich_toolkit/anchor_words.py`

Typical Italian content:

- method descriptions;
- comments about hypotheses;
- user-facing reports;
- explanation of Italian/Judeo-Italian conventions.

Recommended action:

- translate explanatory prose to English;
- preserve Italian lexical examples;
- preserve mappings, transliterations, and phonemic transformation rules;
- do not rename any constant or mapping table.

Important note:

`prepare_italian_lexicon.py` and related files are especially sensitive because they mix English code structure, Italian lexicon content, Judeo-Italian phonemic conventions, and explanatory prose. These files should be translated carefully and selectively.

### CLI interface

- `src/voynich_toolkit/cli.py`

Typical Italian content:

- help text;
- command descriptions.

Recommended action:

- translate help text and docstrings to English;
- do not rename CLI commands or options unless the project explicitly wants a breaking interface change.

For publication, preserving command names is usually preferable.

## C. Generated Output Files with Italian Human-Readable Text

These files contain Italian prose in generated reports. They are not core logic files.

### High-priority human-readable outputs

- `output/eva/comparison_report.csv`
- `output/stats/hand1_deep_dive_summary.txt`
- `output/stats/hand_unknown_summary.txt`
- `output/stats/hand_characterization_summary.txt`
- `output/stats/hand_positional_summary.txt`
- `output/stats/hand_structure_summary.txt`
- `output/stats/hand_zscore_summary.txt`
- `output/stats/semitic_kl_test_summary.txt`
- `output/stats/null_model_test_summary.txt`
- `output/stats/champollion_matches.txt`
- `output/stats/decipher_matches.txt`
- `output/stats/italian_decipher_matches.txt`
- `output/stats/italian_decoded_text.txt`

Recommended action:

- if these outputs are meant to be shipped as examples, translate their human-readable prose;
- if they are generated from code, it may be better to translate the source strings and regenerate them;
- preserve tabular structure and all evidence tokens;
- preserve lexical entries and match targets.

Risk level:

- medium, because some files mix prose with evidence tables.

### JSON outputs containing Italian explanatory values

- `output/stats/hand_unknown.json`
- `output/stats/semitic_kl_test.json`
- `output/stats/analysis_report.json`

Recommended action:

- do not rename JSON keys unless they are confirmed to be purely cosmetic and not consumed anywhere;
- translate only string values that are clearly human-facing explanations;
- prefer regeneration from translated source if possible.

Risk level:

- high if edited manually, because JSON is more likely to be reused programmatically.

## D. Italian Lexicon and Data Files That Should Remain Italian

These files contain Italian lexical material and should remain in Italian.

- `output/lexicon/italian_lexicon.json`
- `output/lexicon/pre1500_italian.json`
- `output/lexicon/divina_commedia.json`
- `output/lexicon/dante_wordlist.json`
- `output/lexicon/kaikki_archaic_italian.json`
- `output/lexicon/north_italian_lexicons.json`
- `output/lexicon/tlio_lemmario.json`

Type:

- lexical datasets;
- historical Italian word resources;
- reference material for analysis.

Recommended action:

- do not translate the lexical entries;
- do not translate Italian gloss fields if they are part of the lexicon resource;
- if needed, document them externally in English rather than modifying the data.

Risk level:

- very high if translated, because translation would corrupt the research dataset.

## E. Files Identified as Not Requiring Translation

These were checked and should generally be left alone.

### English documentation or prose

- `README.md`
- `STATUS.md`

These are already in English.

### Files that produced false positives

- `paper/paper.tex`
- `eva_data/LSI_ivtff_0d.txt`

Reason:

- matched search terms incidentally;
- do not contain meaningful Italian prose requiring translation.

### Files dominated by non-Italian generated content

- `output/readings/*.txt` individual reading files;
- `output/readings/_summary.json`;
- `output/stats/crib_wordlevel_*.json` files.

Reason:

- these are mostly in English or data-oriented;
- any Italian-like hits were false positives, token collisions, or metadata.

## Mixed-Content Translation Rules

These rules are intended for an LLM performing the translation.

### Translate

- explanatory prose;
- headings;
- comments;
- docstrings;
- warning text;
- CLI help;
- human-readable report summaries;
- diagnostic explanations;
- recommendation text.

### Preserve exactly

- code identifiers;
- file names;
- command names;
- option flags;
- YAML/JSON keys unless verified safe;
- CSV headers unless verified safe;
- regexes;
- mappings and dictionaries used by code;
- EVA/Hebrew/Aramaic/Italian lexical tokens used as evidence;
- folio identifiers;
- numeric values;
- formulas;
- examples of lexical entries when they are data rather than prose.

### Preserve with very high care

- table structures;
- code fences;
- bullet semantics;
- confidence labels and rankings;
- inline examples that mix prose and lexical evidence.

## Specific Translation Considerations

### 1. Italian words in lexicon examples inside source files

A docstring might say something like:

- "Loads medieval Italian lexicon"
- followed by example forms or glosses.

In that case:

- translate the descriptive sentence;
- keep the example forms unchanged.

### 2. Output files may be better regenerated than hand-edited

For many files under `output/stats`, the safest publication workflow is:

1. translate source-level user-facing strings;
2. rerun generation commands;
3. commit fresh English outputs where appropriate.

This is safer than manually translating generated files because it reduces inconsistency between code and reports.

### 3. Public repo vs internal research archive

If the repo is going public for general use, you may want to separate:

- publication-facing files;
- archival/generated research outputs;
- raw lexicon assets.

Not every committed artifact needs to be translated if it is not part of the intended public-facing surface.

### 4. Preserve Judeo-Italian and historical-language terminology

Terms like:

- Judeo-Italian;
- medieval Italian;
- lexical gloss;
- phonemic normalization;
- matres lectionis;
- Currier A/B;
- folio;
- herbal/balneological/pharmaceutical

should be translated carefully and consistently, but the underlying data examples should remain unchanged.

## Suggested Priority Order

For a practical release workflow, this order is recommended.

### Priority 1

- `PUBBLICAZIONI.md`
- `EPILECTRIK_NOTES.md`
- source files with Italian CLI/help/docstring text

Reason:

- these affect comprehension immediately;
- they are part of the repository surface;
- translating them improves accessibility without touching data.

### Priority 2

- `output/readings_merged.md`
- high-visibility summary reports in `output/stats`

Reason:

- useful for readers;
- visible in the repo;
- secondary to source documentation.

### Priority 3

- selective cleanup of generated JSON/TXT report prose

Reason:

- useful, but lower value than source and top-level documentation;
- potentially better handled through regeneration.

### Priority 4

- leave Italian lexicon resources unchanged

Reason:

- these are research assets, not UI text.

## Suggested LLM Translation Prompt Constraints

If another LLM is used to perform the translation, it should be instructed with rules like these:

1. Translate all Italian prose into English.
2. Do not change code logic or identifiers.
3. Do not rename commands, flags, file paths, JSON keys, or function names.
4. Keep Italian lexicon entries in Italian.
5. Preserve EVA, Hebrew, Aramaic, folio IDs, and evidence tokens exactly.
6. In mixed-content docstrings, translate the explanation but preserve lexical examples.
7. Maintain formatting, tables, indentation, and code fences.
8. When unsure whether a token is prose or data, preserve it and translate only surrounding prose.

## Practical Release Recommendation

For publication, the repository should ideally end up with:

- English top-level documentation;
- English comments/docstrings/user-facing strings in source files;
- English public-facing reports that are intentionally kept in the repo;
- unchanged Italian lexicon resources;
- unchanged research evidence tokens and manuscript data.

This approach maximizes accessibility while minimizing the risk of damaging the analytical pipeline or the linguistic datasets.

## Final Note

This audit is conservative by design. It is better to leave a few research tokens untranslated than to accidentally translate data, identifiers, or lexical evidence. The repository appears structurally suitable for internationalization because Italian is concentrated mostly in prose, not in executable naming or core program structure.
