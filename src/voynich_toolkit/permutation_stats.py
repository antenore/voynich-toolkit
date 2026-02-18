"""
Permutation statistics framework for the Voynich Toolkit.

Phase 8B: provides p-values, effect sizes, and FDR correction for
all validation modules (zodiac, anchor, plant search).

Pattern: generate N random bijective EVA→Hebrew mappings (19-of-22),
compute a score with each, compare against the real mapping's score.
"""
import random
from collections import Counter
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats

from .prepare_lexicon import CONSONANT_NAMES

# All 22 Hebrew consonants (ASCII encoding)
HEBREW_LETTERS = list("AbgdhwzXJyklmnsEpCqrSt")

# 19 EVA chars used in the mapping
EVA_CHARS = list("acdefghiklmnopqrsty")


# =====================================================================
# Random mapping generation
# =====================================================================

def generate_random_mapping(n_eva=19, seed=None, eva_chars=None):
    """Generate a random bijective EVA→Hebrew mapping.

    Picks n_eva distinct Hebrew letters from the 22-letter alphabet,
    then shuffles the assignment.

    Args:
        n_eva: number of EVA chars (ignored if eva_chars is provided)
        seed: random seed
        eva_chars: explicit list/sequence of EVA chars to use as keys.
                   If None, uses EVA_CHARS[:n_eva] (legacy behavior).

    Returns: dict {eva_char: hebrew_char}
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    keys = list(eva_chars) if eva_chars is not None else EVA_CHARS[:n_eva]
    chosen = rng.sample(HEBREW_LETTERS, len(keys))
    rng.shuffle(chosen)
    return dict(zip(keys, chosen))


# =====================================================================
# Permutation test: mapping quality
# =====================================================================

def permutation_test_mapping(score_fn, real_mapping, n_perms=1000,
                             seed=42):
    """Test whether the real mapping scores significantly above random.

    The score_fn receives a full mapping dict (including preprocessed-char
    placeholders \\x01, \\x02, \\x03 if present in real_mapping).
    Random mappings are generated with the SAME keys as real_mapping,
    ensuring fair comparison.

    Args:
        score_fn: callable(mapping_dict) → float score
        real_mapping: the actual EVA→Hebrew mapping dict (may include
                      placeholder keys from build_full_mapping())
        n_perms: number of random permutations to test
        seed: random seed for reproducibility

    Returns: dict with p_value, z_score, cohens_d, real_score,
             random_mean, random_std, n_perms
    """
    rng = random.Random(seed)

    real_score = score_fn(real_mapping)

    real_keys = sorted(real_mapping.keys())

    random_scores = []
    for i in range(n_perms):
        rand_map = generate_random_mapping(
            eva_chars=real_keys, seed=rng.randint(0, 2**31))
        s = score_fn(rand_map)
        random_scores.append(s)

    arr = np.array(random_scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())

    # p-value: fraction of random scores >= real score
    n_ge = int(np.sum(arr >= real_score))
    p_value = (n_ge + 1) / (n_perms + 1)  # +1 for conservative estimate

    # z-score
    z_score = (real_score - mean) / std if std > 0 else float('inf')

    # Cohen's d effect size
    cohens_d = (real_score - mean) / std if std > 0 else float('inf')

    return {
        "real_score": round(float(real_score), 4),
        "random_mean": round(mean, 4),
        "random_std": round(std, 4),
        "random_max": round(float(arr.max()), 4),
        "p_value": round(p_value, 6),
        "z_score": round(z_score, 2),
        "cohens_d": round(cohens_d, 2),
        "n_perms": n_perms,
        "n_random_ge_real": n_ge,
        "significant_001": p_value < 0.001,
        "significant_01": p_value < 0.01,
        "significant_05": p_value < 0.05,
    }


# =====================================================================
# Permutation test: section specificity
# =====================================================================

def permutation_test_sections(score_fn, section_labels, n_perms=1000,
                              seed=42):
    """Test whether domain matches concentrate in expected sections.

    Shuffles section labels across pages and recomputes the score.

    Args:
        score_fn: callable(section_label_dict) → float score
            section_label_dict maps folio → section
        section_labels: actual dict {folio: section}
        n_perms: number of permutations
        seed: random seed

    Returns: dict with p_value, z_score, cohens_d, etc.
    """
    rng = random.Random(seed)

    real_score = score_fn(section_labels)

    folios = list(section_labels.keys())
    sections = list(section_labels.values())

    random_scores = []
    for _ in range(n_perms):
        shuffled = sections.copy()
        rng.shuffle(shuffled)
        shuffled_labels = dict(zip(folios, shuffled))
        s = score_fn(shuffled_labels)
        random_scores.append(s)

    arr = np.array(random_scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())

    n_ge = int(np.sum(arr >= real_score))
    p_value = (n_ge + 1) / (n_perms + 1)

    z_score = (real_score - mean) / std if std > 0 else float('inf')
    cohens_d = z_score  # Same as z for permutation tests

    return {
        "real_score": round(float(real_score), 4),
        "random_mean": round(mean, 4),
        "random_std": round(std, 4),
        "p_value": round(p_value, 6),
        "z_score": round(z_score, 2),
        "cohens_d": round(cohens_d, 2),
        "n_perms": n_perms,
        "significant_001": p_value < 0.001,
        "significant_01": p_value < 0.01,
    }


# =====================================================================
# FDR correction (Benjamini-Hochberg)
# =====================================================================

def fdr_correction(p_values: dict, alpha: float = 0.05) -> dict:
    """Apply Benjamini-Hochberg FDR correction to multiple p-values.

    Args:
        p_values: dict {test_name: p_value}
        alpha: significance level (default 0.05)

    Returns: dict {test_name: {p_original, p_adjusted, significant}}
    """
    if not p_values:
        return {}

    # Sort by p-value
    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_tests)

    results = {}
    for rank, (name, p) in enumerate(sorted_tests, 1):
        # BH adjusted p-value
        p_adj = min(1.0, p * m / rank)
        results[name] = {
            "p_original": round(p, 6),
            "p_adjusted": round(p_adj, 6),
            "rank": rank,
            "significant": p_adj < alpha,
        }

    # Enforce monotonicity: p_adj[i] >= p_adj[i-1]
    sorted_by_rank = sorted(results.items(),
                            key=lambda x: x[1]["rank"], reverse=True)
    running_min = 1.0
    for name, data in sorted_by_rank:
        data["p_adjusted"] = round(min(running_min, data["p_adjusted"]), 6)
        running_min = data["p_adjusted"]
        data["significant"] = data["p_adjusted"] < alpha

    return results


# =====================================================================
# Utility: score functions for existing modules
# =====================================================================

def build_full_mapping(mapping):
    """Build augmented mapping with preprocessed-char placeholders.

    Takes a standard 17-char EVA mapping (like FULL_MAPPING) and adds
    entries for the preprocessed placeholders:
      \\x03 (ch digraph), \\x01 (ii), \\x02 (standalone i)

    If mapping already contains these keys, they are kept as-is.
    Strips 'q' key if present (q is prefix, not a letter mapping).

    Returns: dict with 20 keys (17 standard + 3 placeholders).
    """
    aug = {k: v for k, v in mapping.items() if k != "q"}

    # Add placeholders from full_decode constants if not already present
    if "\x03" not in aug:
        from .full_decode import CH_HEBREW
        aug["\x03"] = CH_HEBREW  # ch → kaf
    if "\x01" not in aug:
        from .full_decode import II_HEBREW
        aug["\x01"] = II_HEBREW  # ii → he
    if "\x02" not in aug:
        from .full_decode import I_HEBREW
        aug["\x02"] = I_HEBREW   # i → resh

    return aug


def decode_eva_with_mapping(eva_word, mapping, mode="hebrew",
                            direction="rtl"):
    """Decode an EVA word using preprocess_eva and a given mapping.

    Handles ch digraph, ii/i split, q-prefix stripping, RTL reversal.
    Returns decoded string (Hebrew or Italian) or None if unmapped char.

    The mapping should contain preprocessed-char placeholders (\\x01,
    \\x02, \\x03). Use build_full_mapping() to add them if needed.

    Args:
        eva_word: raw EVA word
        mapping: dict with standard EVA chars + placeholder keys
        mode: "hebrew" returns Hebrew consonants, "italian" returns
              Italian phonemes via HEBREW_TO_ITALIAN.
        direction: "rtl" or "ltr" for reversal after preprocessing
    """
    from .full_decode import preprocess_eva
    from .full_decode import INITIAL_D_HEBREW, INITIAL_H_HEBREW
    from .prepare_italian_lexicon import HEBREW_TO_ITALIAN

    _, processed = preprocess_eva(eva_word)
    chars = (list(reversed(processed)) if direction == "rtl"
             else list(processed))

    parts = []
    heb_initial = None
    for i, ch in enumerate(chars):
        h = mapping.get(ch)
        if h is None:
            return None
        if i == 0:
            heb_initial = h
        if mode == "italian":
            parts.append(HEBREW_TO_ITALIAN.get(h, "?"))
        else:
            parts.append(h)

    # Positional splits — decision always on Hebrew value, not Italian
    if heb_initial == "d":
        parts[0] = (HEBREW_TO_ITALIAN.get(INITIAL_D_HEBREW, "?")
                    if mode == "italian" else INITIAL_D_HEBREW)
    elif heb_initial == "h":
        parts[0] = (HEBREW_TO_ITALIAN.get(INITIAL_H_HEBREW, "?")
                    if mode == "italian" else INITIAL_H_HEBREW)

    return "".join(parts)


def make_lexicon_match_scorer(decoded_words, lexicon_set, min_len=3):
    """Create a score function that counts lexicon matches.

    Args:
        decoded_words: list of EVA words (raw strings)
        lexicon_set: set of consonantal forms to match against
        min_len: minimum word length

    Returns: callable(mapping_dict) → int (number of matches)
    """
    def score_fn(mapping):
        n_matches = 0
        for eva_word in decoded_words:
            if len(eva_word) < min_len:
                continue
            hebrew = decode_eva_with_mapping(eva_word, mapping, mode="hebrew")
            if hebrew and hebrew in lexicon_set:
                n_matches += 1
        return n_matches

    return score_fn


def make_italian_match_scorer(decoded_words, italian_set, min_len=3):
    """Create a score function for Italian lexicon matches.

    Returns: callable(mapping_dict) → int
    """
    def score_fn(mapping):
        n_matches = 0
        for eva_word in decoded_words:
            if len(eva_word) < min_len:
                continue
            italian = decode_eva_with_mapping(eva_word, mapping, mode="italian")
            if italian and italian in italian_set:
                n_matches += 1
        return n_matches

    return score_fn
