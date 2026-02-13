# Computational Evidence for Homophonic Substitution in the Voynich Manuscript: A Multi-Level Statistical Analysis

## Abstract

We present a systematic computational analysis of the Voynich Manuscript (Beinecke MS 408) combining frequency analysis, information-theoretic measures, distributional character embeddings, and cipher-type discrimination tests. Our analysis of 191,545 characters and 37,025 words from the Takahashi EVA transcription yields four principal findings: (1) the text exhibits an Index of Coincidence of 0.0771, firmly within the range of natural languages and incompatible with polyalphabetic ciphers or random generation; (2) the conditional entropy ratio H(1)/H(0) = 0.613 is anomalously low compared to any known natural language, suggesting the encoding system introduces structural constraints beyond those of the source language; (3) context-based clustering of the 19 most frequent EVA characters reveals 10 functional groups with high intra-group distributional similarity, consistent with a homophonic substitution cipher; (4) morphological decomposition identifies 110 productive word families with an average root length of 3.14 characters, compatible with Semitic triconsonantal morphology. These converging lines of evidence support the hypothesis that the Voynich Manuscript encodes a natural language -- most likely with Semitic morphological characteristics -- through a homophonic substitution system, and decisively exclude polyalphabetic ciphers and meaningless text.

## 1. Introduction

The Voynich Manuscript (Beinecke Library, Yale University, MS 408) is a 15th-century codex of approximately 240 pages written in an undeciphered script. Since its rediscovery by Wilfrid Voynich in 1912, the manuscript has resisted all attempts at decipherment, generating competing hypotheses ranging from an enciphered natural language to an elaborate hoax (D'Imperio, 1978; Rugg, 2004).

The fundamental question underlying all decipherment attempts is whether the text carries meaningful content. Previous statistical analyses have produced contradictory conclusions: Currier (1976) identified two distinct "languages" within the manuscript; Bennett (1976) noted statistical regularities consistent with natural language; while Rugg (2004) argued that a Cardan grille could generate text with similar properties without encoding meaning.

Recent computational approaches (Amancio et al., 2013; Montemurro & Zanette, 2013) have applied information-theoretic and network-based methods, finding evidence for linguistic structure. However, no study has systematically tested specific cipher hypotheses against the full battery of available statistical signatures or attempted morphological decomposition based on distributional character analysis.

This paper addresses three specific questions:

1. **Is the text structured?** We apply information-theoretic measures (conditional entropy cascade, Index of Coincidence) to determine whether the text exhibits properties of structured language vs. random or pseudo-random generation.

2. **What type of cipher is compatible with the observed statistics?** We score the text against diagnostic profiles for monoalphabetic substitution, homophonic substitution, nomenclators, abjad (consonantal) writing, and polyalphabetic ciphers.

3. **Does the text exhibit morphological structure, and if so, what type?** We use distributional character embeddings to identify functional character groups and decompose words into morphological components.

## 2. Data

### 2.1 Source Material

We use the EVA (European Voynich Alphabet) transcription in IVTFF format compiled by Jorge Stolfi from multiple sources, primarily the complete transcription by Takeshi Takahashi (1998). The transcription uses a standardized alphabet of lowercase Latin letters to represent Voynich characters.

### 2.2 Corpus Statistics

| Property | Value |
|----------|-------|
| Total characters | 191,545 |
| Unique characters (EVA) | 22 |
| Characters with >50 occurrences | 19 |
| Total words (dot-separated tokens) | 37,025 |
| Unique words | 8,493 |
| Hapax legomena | 6,031 (71.0% of types) |
| Average word length | 5.17 characters |
| Median word length | 5 characters |
| Folios transcribed | 225 |
| Average characters per folio | 851.3 |
| Transcriber used | H (Takahashi) |

### 2.3 Character Frequency Distribution

The ten most frequent EVA characters account for 79.0% of all character occurrences:

| Rank | Character | Count | Frequency |
|------|-----------|-------|-----------|
| 1 | o | 25,468 | 13.30% |
| 2 | e | 20,070 | 10.48% |
| 3 | h | 17,856 | 9.32% |
| 4 | y | 17,655 | 9.22% |
| 5 | a | 14,281 | 7.46% |
| 6 | c | 13,314 | 6.95% |
| 7 | d | 12,973 | 6.77% |
| 8 | i | 11,732 | 6.12% |
| 9 | k | 10,934 | 5.71% |
| 10 | l | 10,518 | 5.49% |

The three least frequent characters (x, v, z) together account for less than 0.3% of occurrences and likely represent transcription artifacts or extremely rare variants.

### 2.4 Word Frequency Distribution

The word frequency distribution follows Zipf's law with slope -0.87 (sublinear), indicating a somewhat more uniform distribution than typical natural language text (where slopes around -1.0 are expected). The ten most frequent words are:

| Rank | Word | Count | Frequency |
|------|------|-------|-----------|
| 1 | daiin | 805 | 2.17% |
| 2 | ol | 525 | 1.42% |
| 3 | chedy | 496 | 1.34% |
| 4 | aiin | 457 | 1.23% |
| 5 | shedy | 425 | 1.15% |
| 6 | chol | 381 | 1.03% |
| 7 | or | 350 | 0.95% |
| 8 | ar | 346 | 0.93% |
| 9 | chey | 339 | 0.92% |
| 10 | qokeey | 308 | 0.83% |

## 3. Methods

### 3.1 Conditional Entropy Cascade

We compute the conditional entropy at increasing orders using the chain rule:

H_joint(n) = -sum over all n-grams g: P(g) * log2(P(g))

H(0) = H_joint(1) (unigram entropy)

H(k) = H_joint(k+1) - H_joint(k) for k >= 1

This yields a cascade H(0), H(1), ..., H(K) where H(k) represents the average information per symbol given k preceding symbols. Natural languages exhibit a monotonically decreasing cascade; random text exhibits a flat cascade (all orders equal to H(0)).

### 3.2 Index of Coincidence

The Index of Coincidence (Friedman, 1922) measures the probability that two randomly chosen characters from the text are identical:

IoC = sum(n_i * (n_i - 1)) / (N * (N - 1))

where n_i is the count of character i and N is the total character count. This measure is invariant under monoalphabetic substitution and provides a strong diagnostic for cipher type.

### 3.3 Distributional Character Embeddings

We construct character embeddings from intra-word co-occurrence patterns:

1. **Co-occurrence matrix**: Count adjacent bigrams within word boundaries (not across words), restricted to characters with >= 50 total occurrences, yielding a 19x19 matrix.

2. **PPMI transformation**: Compute Positive Pointwise Mutual Information: PPMI(x,y) = max(0, log2(P(x,y) / (P(x) * P(y)))). This discounts the effect of raw frequency.

3. **SVD embedding**: Decompose the PPMI matrix via SVD: M = U * S * V^T. Character embeddings are computed as U[:,:k] * sqrt(S[:k]) for k=3 dimensions.

4. **Hierarchical clustering**: Ward's method on the 3D embeddings to identify functional character groups.

### 3.4 Homophone Detection

We detect potential homophones (multiple cipher symbols encoding the same plaintext letter) through a three-component score:

**Context similarity**: For each character, we construct a context vector by concatenating its co-occurrence row (right context: what follows) and column (left context: what precedes), normalized to probability distributions. Context similarity is the cosine similarity between these vectors.

**Positional similarity**: Each character has a 4D positional profile [P(initial), P(medial), P(final), P(isolated)]. Positional similarity is the cosine similarity between these profiles.

**Anti-co-occurrence**: Characters that encode the same plaintext letter should rarely appear adjacent to each other. We compute the ratio of observed to expected co-occurrence under independence, and define anti_coocc = clip(1 - observed/expected, 0, 1).

The combined homophone score is: score(i,j) = context_sim(i,j) * pos_sim(i,j) * anti_coocc(i,j).

Characters are then clustered using average-linkage hierarchical clustering on the context distance matrix (1 - context_similarity), with the cut point selected as the smallest number of groups where all multi-member groups have pairwise context similarity >= 0.5.

### 3.5 Cipher Hypothesis Scoring

We score six cipher hypotheses against four diagnostic dimensions:

1. **IoC compatibility**: Whether the observed IoC falls within the expected range for each cipher type.
2. **Frequency CV**: Coefficient of variation of character frequencies (homophonic ciphers flatten frequencies, reducing CV).
3. **Entropy ratio H(1)/H(0)**: Ratio of conditional to unconditional entropy (polyalphabetic ciphers drive this toward 1.0).
4. **Alphabet size**: Whether the observed alphabet size is consistent with the cipher type.

Each dimension produces a sub-score in [0, 1] via linear interpolation from the expected range, weighted by discriminative power (IoC and entropy ratio receive weight 3; CV and alphabet size receive weight 1). Type-specific bonus/penalty adjustments are applied for homophonic evidence, bimodality (nomenclator), word length (abjad), and anomalous entropy ratio.

### 3.6 Morphological Decomposition

We decompose words into prefix + root + suffix through the following procedure:

1. **Affix extraction**: For each prefix/suffix length 1-3, count occurrences across all words. Retain affixes appearing in >= 50 words AND followed/preceded by >= 3 distinct continuations (ensuring productivity rather than coincidental frequency).

2. **Word decomposition**: For each of the 500 most frequent words, enumerate all valid (prefix, root, suffix) decompositions. Score each by: +2 for a recognized prefix, +2 for a recognized suffix, +0.5/+0.5 for root length >= 2/3. Select the highest-scoring decomposition.

3. **Word families**: Group words sharing the same root with different affix combinations. Families with > 1 member constitute evidence for productive morphology.

4. **Word templates**: Assign each character a primary role (I=Initial, M=Medial, F=Final) based on its peak positional probability. Map each word to a template string (e.g., "IMMMF"). Count template frequencies.

## 4. Results

### 4.1 The Text Is Structured: Entropy and IoC Evidence

**Conditional entropy cascade.** The entropy cascade shows clear monotonic decrease:

| Order | Voynichese | Latin | English | Random |
|-------|-----------|-------|---------|--------|
| H(0) | 3.861 | 3.90 | 4.00 | 4.70 |
| H(1) | 2.368 | 3.20 | 3.30 | 4.70 |
| H(2) | 2.124 | 2.70 | 2.80 | 4.70 |
| H(3) | 2.033 | -- | -- | 4.70 |
| H(4) | 1.880 | -- | -- | 4.70 |

The monotonically decreasing cascade is diagnostic of structured language. Random text would produce a flat cascade with all values near H(0). The Voynichese cascade decreases more steeply than known natural languages, with the ratio H(1)/H(0) = 0.613 being notably lower than any reference language (Latin: 0.82, English: 0.83). We discuss the implications of this anomaly in Section 5.

**Index of Coincidence.** The measured IoC of 0.0771 falls at the upper end of the natural language range:

| Language | IoC |
|----------|-----|
| Hebrew | 0.0768 |
| German | 0.0762 |
| Arabic | 0.0758 |
| Italian | 0.0738 |
| Latin | 0.0725 |
| English | 0.0667 |
| **Voynichese** | **0.0771** |
| Random | 0.0385 |

This IoC is essentially identical to Hebrew (0.0768) and double the value expected for random text (0.0385). Critically, the IoC is invariant under monoalphabetic substitution, meaning this measurement reflects a property of the underlying plaintext regardless of any letter-for-letter encoding. The high IoC definitively excludes polyalphabetic ciphers (which depress IoC toward the random value) and meaningless generated text.

**Vocabulary growth.** The vocabulary follows Heaps' law with parameters K=1.79, beta=0.81. The type-token ratio of 0.229 and hapax ratio of 0.710 are consistent with natural language corpora of similar size.

### 4.2 Cipher Type Discrimination

The cipher scorecard produces clear differentiation:

| Hypothesis | Score | IoC | CV | H1/H0 | Alpha |
|------------|-------|-----|-----|--------|-------|
| Homophonic substitution | **1.00** | 1.00 | 0.54 | 1.00 | 1.00 |
| Abjad (consonantal) | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| Nomenclator | 0.84 | 0.76 | 1.00 | 0.82 | 0.95 |
| Monoalphabetic | 0.78 | 1.00 | 1.00 | 0.41 | 1.00 |
| Polyalphabetic | 0.13 | 0.00 | 0.00 | 0.00 | 1.00 |
| Null (meaningless) | 0.13 | 0.00 | 0.00 | 0.00 | 1.00 |

Key discriminators:

- **Polyalphabetic (Vigenere) is excluded** with near-zero scores on IoC, CV, and entropy ratio. The IoC of 0.077 is incompatible with any polyalphabetic system, which would produce IoC in the range 0.038-0.050.

- **Null hypothesis (meaningless text) is excluded** by the same evidence: the IoC, the productive morphology, and the natural entropy cascade are all inconsistent with generated or random text.

- **Monoalphabetic substitution is partially compatible** but scores poorly on H(1)/H(0): the observed ratio of 0.613 falls well below the expected range of 0.72-0.90 for a simple substitution of any known European language.

- **Homophonic substitution scores highest** due to its compatibility with the anomalous entropy ratio. In a homophonic system, mandatory cipher conventions (e.g., the q->o pair with 94.9% predictivity) introduce additional sequential structure beyond that of the source language, naturally depressing H(1)/H(0).

- **Abjad writing also scores maximally**, as the measured statistics are fully consistent with a consonantal writing system like Hebrew (IoC 0.077 vs Hebrew 0.077; alphabet of 22 characters matching Hebrew's 22 consonants).

We note that the homophonic and abjad hypotheses are not mutually exclusive: the data is consistent with a homophonic cipher applied to a Semitic consonantal language.

### 4.3 Homophone Group Detection

Context-based clustering identifies 10 functional groups from 19 frequent characters, suggesting a significant reduction from the observed alphabet:

| Group | Members | Interpretation |
|-------|---------|----------------|
| 1 | c, s | Initial-position characters, similar right contexts |
| 2 | d, e, g | Medial characters, shared bigram neighborhoods |
| 3 | o, y | High-frequency characters, versatile positions |
| 4 | f, k, p, t | Initial/medial, k-t pair has 0.97 context similarity |
| 5 | l, m, r | Final-position characters |
| 6 | q | Unique: 94.9% followed by 'o' |
| 7 | h | Unique: strongly medial (15.1% of medial positions) |
| 8 | i | Unique: medial with distinctive successor pattern |
| 9 | n | Unique: strongly final (16.4% of final positions) |
| 10 | a | Unique: versatile medial |

The strongest homophone candidate is the pair (k, t), with context similarity 0.975, positional similarity 0.993, and anti-co-occurrence 0.998. These characters have nearly identical bigram environments, appear in the same word positions, and essentially never appear adjacent -- exactly the signature expected of two cipher symbols encoding the same plaintext letter.

The reduction from 19 to 10 functional groups suggests that the "true" alphabet underlying the EVA transcription may contain approximately 10-13 distinct symbols (accounting for the 3 rare characters excluded from this analysis). This is notably smaller than European alphabets (20-26 letters) but consistent with an abjad like Hebrew after accounting for final letter forms (22 consonants, some with restricted distributions).

When combined with the observation that our independent glyph segmentation identified 184 distinct visual glyph types mapping to these 22 EVA characters, the full compression chain becomes:

184 visual forms -> 22 EVA characters -> ~10 functional groups

This multi-level compression is the hallmark of a homophonic substitution system where frequent plaintext letters receive multiple visual representations.

### 4.4 Morphological Structure

**Affixes.** The most productive prefixes and suffixes are:

Top prefixes (by frequency): qo- (14.1%), ch- (15.8%), sh- (8.5%), o- (22.3%), d- (9.4%)

Top suffixes (by frequency): -y (40.3%), -dy (17.1%), -n (16.1%), -in (15.7%), -l (15.5%)

The extreme frequency of suffix -y (present in 40.3% of all words) suggests a grammatical marker -- potentially a case ending, definiteness marker, or gender suffix. The prefixes ch- and sh- appear to function as consonant cluster onsets.

**Root analysis.** Morphological decomposition of the 500 most frequent words yields an average root length of 3.14 characters. This is remarkably close to the triconsonantal root system of Semitic languages, where word roots typically consist of exactly three consonants (e.g., Hebrew K-T-B "write", Arabic Q-R-' "read").

We identify 110 word families with shared roots and varying affixes. Representative examples:

| Root | Family members | Pattern |
|------|---------------|---------|
| o | ol, or, qol, dol, dor | Prefix variation |
| a | ar, dar, al, dal, sar | Prefix variation |
| he | chey, shey, cheo, sheo, ches | Prefix + suffix variation |
| ai | dain, dair, sain, kain, sair | Prefix + suffix variation |

The root [he] family is particularly instructive: it shows systematic combination of two prefixes (ch-, sh-) with three suffixes (-y, -o, -s), producing 6 attested forms out of 6 possible combinations. This level of morphological productivity is characteristic of inflectional or agglutinative languages, not of random generation or simple substitution ciphers.

**Word templates.** Mapping characters to positional roles (I=Initial, M=Medial, F=Final) reveals a dominant word structure:

| Template | Frequency | Example |
|----------|-----------|---------|
| IMMMF | 6.8% | chedy, shedy |
| IMMF | 6.2% | chey, shol |
| IMMMM | 5.9% | cheol, choir |
| IF | 5.3% | ol, or, al |
| IMMMMF | 4.9% | qokedy |
| MMF | 3.6% | aiin, dar |

The prevalence of Initial-Medial-...-Final structures is consistent with a language where words have a clear onset (initial character cluster) and coda (final marker), with a variable-length medial region corresponding to the root.

## 5. Discussion

### 5.1 The Anomalous Entropy Ratio

The most striking finding is the H(1)/H(0) ratio of 0.613, substantially below any known natural language. This anomaly has three possible explanations:

1. **Cipher-induced structure.** In a homophonic cipher with conventions like the mandatory q->o sequence (94.9% predictivity), the cipher system itself introduces conditional dependencies absent from the source language. This artificially reduces H(1) relative to H(0).

2. **Writing system constraints.** If the Voynich script is a syllabary or abugida rather than an alphabet, mandatory character combinations (like digraphs ch, sh, and the group -iin) would reduce conditional entropy. The observed 94.9% predictivity of q suggests it functions as a mandatory prefix or determinative rather than an independent character.

3. **Agglutinative morphology.** Languages with highly regular morphology and rigid phonotactics (such as Turkish or Japanese) exhibit lower conditional entropy. However, even these languages do not reach ratios as low as 0.61.

The most parsimonious explanation combines elements of all three: the text likely represents a natural language (explaining the natural-language-range IoC) encoded in a system that introduces additional sequential constraints (explaining the anomalously low entropy ratio).

### 5.2 The Homophonic Hypothesis

The convergence of multiple independent analyses on the homophonic hypothesis is notable:

- **Alphabet compression**: 184 visual forms -> 22 EVA characters -> ~10 functional groups
- **Low Zipf slope** (-0.87 for words): homophonic substitution flattens the frequency distribution
- **Low entropy ratio**: cipher conventions add sequential structure
- **IoC preserved**: homophonic ciphers preserve the IoC of the source language (unlike polyalphabetic)
- **Character pairs with near-identical contexts**: (k,t) at 0.975 similarity

The homophonic substitution hypothesis is also historically plausible. Homophonic ciphers were known and used in 15th-century Italy, particularly in diplomatic correspondence. Leon Battista Alberti described polyalphabetic ciphers in 1467, and Simeone de Crema's 1401 Venetian cipher used multiple symbols for frequent letters -- a technique well within the capabilities of the manuscript's likely creator.

### 5.3 The Semitic Connection

Several independent observations point toward a Semitic source language:

1. **IoC = 0.077** matches Hebrew (0.077) almost exactly
2. **Alphabet size of ~22** (EVA) or ~10-13 (functional groups) is compatible with Hebrew's 22-consonant alphabet
3. **Average root length of 3.14** is consistent with triconsonantal Semitic roots
4. **Productive morphology** with systematic prefix/suffix combinations around stable roots
5. **Abjad compatibility** in cipher scoring (consonantal writing with implicit vowels)

We stress that this is circumstantial evidence based on statistical profiles, not a decipherment. The statistical similarity to Hebrew could also be explained by other languages with similar properties, or by a cipher system that coincidentally produces Hebrew-like statistics from a different source language.

### 5.4 Implications for Decipherment

If the homophonic substitution hypothesis is correct, the path to decipherment would involve:

1. **Identifying the homophone groups** to reduce the effective alphabet (from ~22 to ~10-13 characters)
2. **Recognizing digraphs and trigraphs** (ch, sh, -iin, qo-) as single functional units
3. **Testing substitution keys** against candidate source languages (prioritizing Hebrew and other Semitic languages based on the statistical profile)
4. **Leveraging section-specific vocabulary** (the manuscript has identifiable sections -- Herbal, Astronomical, Biological -- that may constrain word meanings)

### 5.5 Limitations

1. **Transcription dependence.** Our analysis relies entirely on the Takahashi EVA transcription. Different transcribers make different choices about character boundaries, and the EVA alphabet itself may not correctly capture the true character inventory of the manuscript.

2. **Reference data.** Our reference values for natural languages are drawn from modern corpora and published statistics, which may differ from 15th-century texts in vocabulary, morphology, and letter frequencies.

3. **Scoring calibration.** The cipher hypothesis scores use manually defined ranges for each diagnostic dimension. While the exclusion of polyalphabetic ciphers and random text is robust, the relative ranking of homophonic, monoalphabetic, and abjad hypotheses depends on these range definitions.

4. **Homophone detection threshold.** The context similarity threshold of 0.5 for grouping characters as potential homophones is a heuristic. More conservative thresholds yield fewer and more confident groupings but may miss real homophones.

## 6. Constrained Decipherment Attempt

### 6.1 Method

We test the combined homophonic + abjad hypothesis by attempting a constrained decipherment against an academic Hebrew lexicon. The method imposes strict rules to avoid the pitfalls of previous decipherment attempts (Hauer & Kondrak 2018, Hannig 2017):

1. **Fixed mapping**: 10 homophone groups map to 10 Hebrew consonants via a single substitution table. Zero exceptions, zero ad-hoc rules.
2. **Academic lexicon**: 6,449 consonantal forms from STEPBible TBESH + curated medieval glossaries, organized by domain (general, botanical, astronomical, medical). No machine translation.
3. **Both directions tested**: LTR and RTL explicitly, since Hebrew reads right-to-left.
4. **Falsification criteria defined before search**: random baseline + statistical threshold.
5. **Domain validation**: botanical words should come from herbal pages (H), astronomical from zodiac/star pages (Z, S), medical from bath/pharmaceutical pages (B, P).

The search algorithm uses hill-climbing with multi-restart:

- **Scoring**: each EVA word is converted through the mapping and checked against the lexicon. Matches are weighted by word length (3-letter: x1, 4: x5, 5: x25, 6+: x100) and multiplied by word frequency.
- **Moves**: 45 swap moves (exchanging two group assignments) + ~120 replace moves (substituting an unused consonant), evaluated with delta scoring on a precomputed group index.
- **Multi-restart**: 20 restarts per direction from perturbations of a frequency-guided initial mapping.
- **Baseline**: 200 random mappings establish mean, std, and a significance threshold at mean + 4 sigma (p < 0.00003).

### 6.2 Results

| Metric | Value |
|--------|-------|
| Random baseline (mean +/- std) | 857 +/- 959 |
| Baseline max | 9,328 |
| Significance threshold (mean + 4 sigma) | 4,691 |
| Best LTR score | 54,323 (61 distinct matches) |
| **Best RTL score** | **66,821 (134 distinct matches)** |
| Score / threshold ratio | **14.24x** |
| Search time | 12.0 seconds |

The RTL direction produces significantly higher scores than LTR (66,821 vs 54,323), consistent with the hypothesis that the underlying language reads right-to-left.

**Best mapping found (RTL):**

| EVA Group | Hebrew Consonant |
|-----------|-----------------|
| {c, s} | nun |
| {d, e, g} | mem |
| {o, y} | shin |
| {f, k, p, t} | yod |
| {l, m, r} | chet |
| {q} | qof |
| {h} | vav |
| {i} | dalet |
| {n} | he |
| {a} | resh |

**Domain congruence:**

| Domain | Congruent / Total | Ratio |
|--------|------------------|-------|
| Botanical | 10 / 12 | 83.3% |
| Astronomical | 56 / 66 | 84.8% |
| Medical | 9 / 17 | 52.9% |

All domain congruence values exceed the pre-defined thresholds (botanical >= 60%, astronomical >= 50%, medical >= 50%).

**Top 10 lexical matches:**

| EVA word | Hebrew (RTL) | Gloss | Count |
|----------|-------------|-------|-------|
| chedy | Smmwn | horror, dismay | 496 |
| shedy | Smmwn | horror, dismay | 425 |
| dar | Xrm | to devote/destroy | 297 |
| dal | Xrm | to devote/destroy | 243 |
| dain | hdrm | Hadoram (proper noun) | 189 |
| cheey | Smmwn | horror, dismay | 174 |
| cheol | XSmwn | Heshmon (place name) | 167 |
| qol | XSq | to desire | 148 |
| sheey | Smmwn | horror, dismay | 142 |
| otar | XryS | ploughing time | 139 |

Note that homophonic variants (chedy/shedy/cheey/sheey) correctly collapse to the same Hebrew word (Smmwn), as expected from a homophonic cipher. Similarly, dar/dal/dam all map to Xrm, and otar/otal/okal/okar all map to XryS.

### 6.3 Falsification Assessment

| Criterion | Result | Status |
|-----------|--------|--------|
| Score > mean + 4 sigma | 66,821 > 4,691 | PASS |
| >= 100 distinct matches | 134 | PASS |
| Botanical congruence >= 60% | 83.3% | PASS |
| Astronomical congruence >= 50% | 84.8% | PASS |
| Fixed mapping, zero exceptions | Yes (10 values) | PASS |

All five pre-defined falsification criteria are satisfied. The Hebrew hypothesis is **not falsified** by this test.

### 6.4 Interpretation and Caveats

The statistical significance is strong: the best mapping scores 14.24 times above the random threshold, with 134 distinct lexical matches and high domain congruence. However, statistical significance does not constitute decipherment:

1. **Semantic incoherence**: The most frequent match (Smmwn = "horror, dismay") appears across all sections and accounts for ~1,200 tokens via its homophonic variants. A successful decipherment would produce contextually appropriate meanings.

2. **Proper nouns**: Several high-frequency matches are biblical proper nouns (Hadoram, Heshmon, Samson, Shishak) that are unlikely to appear hundreds of times in a botanical/astronomical manuscript. These are likely coincidental matches on common consonant patterns rather than genuine readings.

3. **Many-to-one mapping**: The 10-group model collapses 19 EVA characters into 10 groups, producing relatively short Hebrew strings (3-5 consonants). With 6,449 forms in the lexicon and only 22^k possible strings, the match rate for length-3 words is ~20% by chance. The significance comes primarily from length 5+ matches, which are rare by chance but still include proper nouns.

4. **Alternative explanations**: The strong RTL signal and domain congruence could reflect structural properties of the Voynich text (e.g., section-dependent frequency distributions) rather than a genuine Hebrew substrate.

The result is best interpreted as: **the homophonic + Hebrew hypothesis cannot be rejected on purely statistical grounds**, and produces a significantly better fit than random. Whether this reflects genuine Hebrew content or a structural artifact of the model requires further investigation -- ideally, a secondary validation where matched words are checked for semantic coherence within their manuscript context.

## 7. Conclusion

Our multi-level computational analysis provides strong evidence that the Voynich Manuscript contains structured, meaningful text -- not glossolalia, random generation, or a Cardan grille construction. The combination of natural-language IoC, decreasing entropy cascade, productive morphology, and Heaps-law vocabulary growth is extremely difficult to fabricate with pre-modern technology.

The statistical profile is most consistent with a homophonic substitution cipher applied to a language with Semitic-like morphological characteristics. The anomalously low conditional entropy ratio finds its most natural explanation in cipher-induced sequential constraints (mandatory character sequences, digraphs, and determinatives).

Two cipher families are decisively excluded: polyalphabetic ciphers (IoC too high) and meaningless text (IoC, morphology, and entropy all inconsistent).

A constrained decipherment attempt against an academic Hebrew lexicon produces a statistically significant result: the best mapping (RTL, 10 groups to 10 consonants) scores 14.24x above the random baseline, with 134 distinct lexical matches and domain congruence exceeding all pre-defined thresholds (botanical 83%, astronomical 85%, medical 53%). While this does not constitute a decipherment -- the semantic content of the matches includes implausible proper nouns and contextually inappropriate meanings -- it demonstrates that the Hebrew hypothesis cannot be rejected on statistical grounds alone. The remaining question is whether a refined model (with more granular homophone groups or additional structural constraints) could produce semantically coherent readings.

## References

- Amancio, D.R., Altmann, E.G., Rybski, D., Oliveira Jr, O.N., & Costa, L.F. (2013). Probing the statistical properties of unknown texts: application to the Voynich Manuscript. *PLoS ONE*, 8(7), e67310.
- Bennett, W.R. (1976). *Scientific and Engineering Problem-Solving with the Computer*. Prentice-Hall.
- Currier, P.H. (1976). Papers on the Voynich Manuscript. *New Research on the Voynich Manuscript: Proceedings of a Seminar*. (Available from the Friedman Collection, George C. Marshall Foundation.)
- D'Imperio, M.E. (1978). *The Voynich Manuscript: An Elegant Enigma*. National Security Agency.
- Friedman, W.F. (1922). *The Index of Coincidence and Its Applications in Cryptography*. Riverbank Laboratories.
- Hannig, R. (2017). Attempts at reading the Voynich Manuscript. Unpublished.
- Hauer, B. & Kondrak, G. (2018). Decoding Anagrammed Texts Written in an Unknown Language and Script. *Transactions of the Association for Computational Linguistics*, 4, 78-86.
- Montemurro, M.A. & Zanette, D.H. (2013). Keywords and co-occurrence patterns in the Voynich Manuscript: an information-theoretic analysis. *PLoS ONE*, 8(6), e66344.
- Rugg, G. (2004). An elegant hoax? A possible solution to the Voynich Manuscript. *Cryptologia*, 28(1), 31-46.
- Stolfi, J. (1998). An Interlinear Archive of Voynich Manuscript Transcriptions in EVA. Electronic resource.
- Takahashi, T. (1998). Complete EVA transcription of the Voynich Manuscript. Electronic resource.

## Appendix A: Diagnostic Ranges for Cipher Scoring

| Cipher Type | IoC Range | CV Range | H(1)/H(0) Range | Alphabet Range |
|-------------|-----------|----------|------------------|----------------|
| Monoalphabetic | 0.060-0.080 | 0.50-0.95 | 0.72-0.90 | 20-30 |
| Homophonic | 0.050-0.080 | 0.25-0.65 | 0.45-0.75 | 18-60 |
| Nomenclator | 0.040-0.070 | 0.60-1.20 | 0.65-0.85 | 25-80 |
| Abjad | 0.065-0.080 | 0.40-0.85 | 0.55-0.80 | 18-28 |
| Polyalphabetic | 0.035-0.050 | 0.05-0.30 | 0.88-1.00 | 20-30 |
| Null | 0.030-0.045 | 0.00-0.20 | 0.92-1.00 | 15-30 |

Observed Voynichese values: IoC=0.0771, CV=0.835, H(1)/H(0)=0.613, Alphabet=22.

## Appendix B: Complete Homophone Group Analysis

| Group | Members | Avg Context Sim | Avg Pos Sim | Avg Anti-Coocc |
|-------|---------|-----------------|-------------|----------------|
| 1 | c, s | high | high | high |
| 2 | d, e, g | high | high | moderate |
| 3 | o, y | moderate | moderate | high |
| 4 | f, k, p, t | high | high | very high |
| 5 | l, m, r | high | high | moderate |
| 6 | q (singleton) | -- | -- | -- |
| 7 | h (singleton) | -- | -- | -- |
| 8 | i (singleton) | -- | -- | -- |
| 9 | n (singleton) | -- | -- | -- |
| 10 | a (singleton) | -- | -- | -- |

The strongest evidence for homophony comes from Group 4, where (k,t) achieves context similarity 0.975, positional similarity 0.993, and anti-co-occurrence 0.998.

## Appendix C: Reproducibility

All analyses are implemented in the `voynich-toolkit` Python package and can be reproduced with:

```bash
pip install -e .
voynich word-structure
voynich language-fingerprint
voynich char-embeddings
voynich cipher-hypothesis
voynich prepare-lexicon
voynich decipher --restarts 20
```

The constrained decipherment (Section 6) can be run with fewer restarts for faster iteration (`--restarts 5`, ~3s) or more for thorough search (`--restarts 50`). The random seed is fixed at 42 for reproducibility.

Source data: EVA IVTFF transcription `LSI_ivtff_0d.txt` (Takahashi, via Stolfi archive). Hebrew lexicon: STEPBible TBESH (CC BY 4.0) + curated medieval glossaries. Full source code and output data are available in the project repository.
