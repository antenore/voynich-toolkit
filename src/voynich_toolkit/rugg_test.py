"""
Rugg Grille Test — Can a Cardan grille reproduce confirmed Voynich properties?

Phase 27.9: Takes Rugg's (2004) mechanism as-is (table + Cardan grille) and checks
which of 11 confirmed structural properties it reproduces without modification.

If the grille reproduces ALL properties → the model is sufficient.
If it MISSES some → those properties require something beyond the grille.

Rugg's mechanism:
  1. A syllable table with columns: PREFIX | CORE1 | CORE2 | SUFFIX
  2. A Cardan grille (card with holes) selecting which columns to read
  3. Words = concatenation of syllables read through holes at each table row
  4. Different grilles → different "languages" (Currier A/B)

The table is built from actual EVA syllable frequencies (faithful to Rugg's
approach of deriving the table from observed patterns).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2_contingency

from .config import ToolkitConfig
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# ── Constants ────────────────────────────────────────────────────

SEED = 42
N_PERM = 500          # permutations for null models
TABLE_ROWS = 40       # Rugg table height (number of syllable rows)
N_GRILLES = 3         # number of distinct grilles (≈ Currier languages)

# EVA characters that are "simple gallows"
SIMPLE_GALLOWS = {"t", "k", "p", "f"}

# Paragraph-start markers in IVTFF
PARA_START_RE = re.compile(r"^<")  # we detect @P from line_words index 0


# =====================================================================
# Step 1: Extract syllable table from real EVA corpus
# =====================================================================

def extract_syllable_stats(pages: list[dict]) -> dict:
    """Analyse real EVA words to extract prefix/core/suffix distributions.

    Returns dict with 'prefixes', 'cores', 'suffixes' as weighted lists,
    plus word-length distribution and line-length distribution.
    """
    prefix_counts: Counter = Counter()
    core_counts: Counter = Counter()
    suffix_counts: Counter = Counter()
    word_lengths: list[int] = []
    line_lengths: list[int] = []  # words per line
    words_per_page: list[int] = []

    for page in pages:
        words_per_page.append(len(page["words"]))
        for line in page.get("line_words", []):
            line_lengths.append(len(line))
            for word in line:
                word_lengths.append(len(word))
                _decompose_word(word, prefix_counts, core_counts, suffix_counts)

    return {
        "prefixes": _normalise(prefix_counts),
        "cores": _normalise(core_counts),
        "suffixes": _normalise(suffix_counts),
        "word_lengths": word_lengths,
        "line_lengths": line_lengths,
        "words_per_page": words_per_page,
    }


def _decompose_word(word: str, pfx: Counter, core: Counter, sfx: Counter):
    """Decompose an EVA word into prefix + core + suffix.

    Prefix: q, qo, or empty
    Suffix: last 1-3 chars if they match common endings (y, dy, aiin, iin, m, n, s)
    Core: everything in between
    """
    w = word

    # Extract prefix
    if w.startswith("qo"):
        pfx["qo"] += 1
        w = w[2:]
    elif w.startswith("q") and len(w) > 1:
        pfx["q"] += 1
        w = w[1:]
    else:
        pfx[""] += 1

    if not w:
        core[""] += 1
        sfx[""] += 1
        return

    # Extract suffix (greedy, longest match first)
    suffix_found = ""
    for s in ("aiin", "iin", "ain", "dy", "in", "y", "m", "n", "s", "g"):
        if w.endswith(s) and len(w) > len(s):
            suffix_found = s
            break

    if suffix_found:
        sfx[suffix_found] += 1
        w = w[: -len(suffix_found)]
    else:
        sfx[""] += 1

    # Core = what remains
    if w:
        core[w] += 1
    else:
        core[""] += 1


def _normalise(counts: Counter) -> list[tuple[str, float]]:
    """Convert Counter to sorted (value, probability) list."""
    total = sum(counts.values())
    if total == 0:
        return [("", 1.0)]
    return sorted(
        [(k, v / total) for k, v in counts.items()],
        key=lambda x: -x[1],
    )


# =====================================================================
# Step 2: Build Rugg's syllable table
# =====================================================================

def build_rugg_table(
    stats: dict,
    n_rows: int = TABLE_ROWS,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Build a Rugg-style syllable table with PREFIX|CORE|SUFFIX columns.

    Each row is a dict: {"prefix": str, "core": str, "suffix": str}
    Cells are sampled from actual EVA frequency distributions.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    def _sample_column(dist: list[tuple[str, float]], n: int) -> list[str]:
        values = [v for v, _ in dist]
        probs = np.array([p for _, p in dist], dtype=np.float64)
        probs /= probs.sum()  # ensure normalisation
        return list(rng.choice(values, size=n, p=probs))

    prefixes = _sample_column(stats["prefixes"], n_rows)
    cores = _sample_column(stats["cores"], n_rows)
    suffixes = _sample_column(stats["suffixes"], n_rows)

    return [
        {"prefix": prefixes[i], "core": cores[i], "suffix": suffixes[i]}
        for i in range(n_rows)
    ]


# =====================================================================
# Step 3: Define Cardan grilles
# =====================================================================

# Each grille is a tuple of column names to include.
# Different grilles produce different "languages" (Currier A/B/C).
GRILLE_CONFIGS = [
    ("prefix", "core", "suffix"),      # Full word (most common)
    ("core", "suffix"),                 # No prefix
    ("prefix", "core"),                 # No suffix
]


def generate_word(table: list[dict], grille: tuple, row_idx: int) -> str:
    """Generate one word by reading table row through grille holes."""
    row = table[row_idx % len(table)]
    return "".join(row[col] for col in grille)


# =====================================================================
# Step 4: Generate synthetic corpus matching real structure
# =====================================================================

def generate_rugg_corpus(
    real_pages: list[dict],
    stats: dict,
    rng: np.random.Generator,
    n_tables: int = N_GRILLES,
) -> list[dict]:
    """Generate a synthetic corpus with same structure as the real one.

    For each real page: same number of lines, same words-per-line.
    Uses multiple tables (one per Currier language) and cycles through
    grille configurations.

    Returns list of synthetic page dicts with same schema as real pages.
    """
    # Build one table per "language"
    tables = [build_rugg_table(stats, TABLE_ROWS, rng) for _ in range(n_tables)]

    synthetic_pages = []
    for page in real_pages:
        # Pick table based on Currier language
        lang = page.get("language", "?")
        if lang == "A":
            table_idx = 0
        elif lang == "B":
            table_idx = 1
        else:
            table_idx = 2

        table = tables[table_idx]
        grille_pool = GRILLE_CONFIGS

        syn_line_words = []
        syn_words = []

        for line in page.get("line_words", []):
            n_words = len(line)
            syn_line = []
            for _ in range(n_words):
                grille = grille_pool[rng.integers(0, len(grille_pool))]
                row_idx = rng.integers(0, len(table))
                word = generate_word(table, grille, row_idx)
                if not word:
                    word = "o"  # fallback (single char)
                syn_line.append(word)
            syn_line_words.append(syn_line)
            syn_words.extend(syn_line)

        synthetic_pages.append({
            "folio": page["folio"],
            "section": page["section"],
            "language": page["language"],
            "hand": page.get("hand", "?"),
            "words": syn_words,
            "line_words": syn_line_words,
        })

    return synthetic_pages


# =====================================================================
# Step 5: Property measurement functions
# =====================================================================

def measure_all_properties(pages: list[dict]) -> dict:
    """Measure all 11 confirmed properties on a corpus (real or synthetic).

    Returns dict of property_name → measurement.
    """
    results = {}

    # Flatten
    all_words = []
    all_line_words = []
    section_words: dict[str, list[str]] = defaultdict(list)
    hand_words: dict[str, list[str]] = defaultdict(list)
    para_first_lines: list[list[str]] = []
    para_continuation_lines: list[list[str]] = []

    for page in pages:
        for w in page["words"]:
            all_words.append(w)
            section_words[page["section"]].append(w)
            hand_words[page.get("hand", "?")].append(w)

        for i, line in enumerate(page.get("line_words", [])):
            all_line_words.append(line)
            if i == 0:
                para_first_lines.append(line)
            else:
                para_continuation_lines.append(line)

    # --- Property 1: Line self-containment ---
    results["line_self_containment"] = _measure_line_containment(all_line_words)

    # --- Property 2: 'm' line-end concentration ---
    results["m_end_marker"] = _measure_end_marker(all_line_words, "m")

    # --- Property 3: Simple gallows at paragraph start ---
    results["gallows_para_start"] = _measure_gallows_para_start(
        para_first_lines, para_continuation_lines
    )

    # --- Property 4: Paragraph internal coherence ---
    results["para_coherence"] = _measure_para_coherence(pages)

    # --- Property 5: Word-section mutual information ---
    results["word_section_mi"] = _measure_word_section_mi(section_words)

    # --- Property 6: Currier A/B difference ---
    results["currier_diff"] = _measure_currier_diff(pages)

    # --- Property 7: Zipf's law ---
    results["zipf"] = _measure_zipf(all_words)

    # --- Property 8: Entropy ---
    results["entropy"] = _measure_entropy(all_words)

    # --- Property 9: Hand bigram signatures ---
    results["hand_bigrams"] = _measure_hand_bigrams(hand_words)

    # --- Property 10: Slot grammar (positional structure) ---
    results["slot_grammar"] = _measure_slot_grammar(all_words)

    # --- Property 11: Vocabulary richness (TTR proxy for match rate signal) ---
    results["vocabulary"] = _measure_vocabulary(all_words)

    # --- Property 12: Per-hand entropy variance (Phase 0) ---
    results["hand_entropy_variance"] = _measure_hand_entropy_variance(hand_words)

    # --- Property 13: Hand 1-2 vocabulary below null (Phase 2c) ---
    results["hand12_jaccard"] = _measure_hand_pair_jaccard(hand_words, "1", "2")

    # --- Property 14: Astro-Zodiac anti-correlation (Phase 8b) ---
    results["astro_zodiac_anti"] = _measure_section_pair_jaccard(
        section_words, "S", "Z"
    )

    # --- Property 15: Hand ? entropy anomaly (Phase 2a) ---
    results["hand_unknown_anomaly"] = _measure_hand_unknown_anomaly(hand_words)

    # --- Property 16: Split gallows NOT paragraph markers (Phase 7b) ---
    results["split_gallows_uniform"] = _measure_split_gallows(
        para_first_lines, para_continuation_lines
    )

    return results


# --- Individual property measurements ---

def _measure_line_containment(line_words: list[list[str]]) -> dict:
    """Check if any word appears split across adjacent lines (bigram overlap)."""
    n_boundaries = 0
    n_cross = 0
    for i in range(len(line_words) - 1):
        if not line_words[i] or not line_words[i + 1]:
            continue
        n_boundaries += 1
        last_word = line_words[i][-1]
        first_word = line_words[i + 1][0]
        # A cross-boundary word would be: last chars of line + first chars of next
        # We proxy this by checking if last word of line i == first word of line i+1
        # (exact word repetition across boundary)
        if last_word == first_word:
            n_cross += 1

    rate = n_cross / n_boundaries if n_boundaries > 0 else 0
    return {
        "n_boundaries": n_boundaries,
        "n_cross_repeats": n_cross,
        "cross_rate": round(rate, 5),
        "note": "Real Voynich: 0.24% (z=-3.60)",
    }


def _measure_end_marker(line_words: list[list[str]], marker_char: str) -> dict:
    """Fraction of occurrences of marker_char that are line-final."""
    total_occ = 0
    line_final_occ = 0

    for line in line_words:
        for i, word in enumerate(line):
            if marker_char in word:
                total_occ += word.count(marker_char)
                if i == len(line) - 1:
                    line_final_occ += word.count(marker_char)

    rate = line_final_occ / total_occ if total_occ > 0 else 0
    return {
        "marker": marker_char,
        "total_occurrences": total_occ,
        "line_final_occurrences": line_final_occ,
        "line_final_rate": round(rate, 4),
        "note": "Real Voynich: 66.5% line-final (z=+55.5)",
    }


def _measure_gallows_para_start(
    first_lines: list[list[str]],
    continuation_lines: list[list[str]],
) -> dict:
    """Concentration of simple gallows on paragraph-first lines."""
    def _gallows_rate(lines: list[list[str]]) -> float:
        total_words = 0
        gallows_words = 0
        for line in lines:
            for word in line:
                total_words += 1
                if any(c in SIMPLE_GALLOWS for c in word):
                    gallows_words += 1
        return gallows_words / total_words if total_words > 0 else 0

    first_rate = _gallows_rate(first_lines)
    cont_rate = _gallows_rate(continuation_lines)

    return {
        "para_first_gallows_rate": round(first_rate, 4),
        "continuation_gallows_rate": round(cont_rate, 4),
        "difference": round(first_rate - cont_rate, 4),
        "note": "Real Voynich: 0.554 vs 0.492, z=+10.73",
    }


def _measure_para_coherence(pages: list[dict]) -> dict:
    """Intra-paragraph vs inter-paragraph vocabulary Jaccard similarity."""
    # Group lines by page into paragraphs (simple: each page = 1 paragraph)
    paragraphs = []
    for page in pages:
        if page.get("line_words"):
            para_words = set()
            for line in page["line_words"]:
                para_words.update(line)
            if para_words:
                paragraphs.append(para_words)

    if len(paragraphs) < 4:
        return {"intra_jaccard": 0, "inter_jaccard": 0, "ratio": 0}

    # Intra: consecutive paragraphs on same "page sequence"
    intra_jaccards = []
    for i in range(len(paragraphs) - 1):
        a, b = paragraphs[i], paragraphs[i + 1]
        inter = len(a & b)
        union = len(a | b)
        if union > 0:
            intra_jaccards.append(inter / union)

    # Inter: random pairs (sample 200)
    rng = np.random.default_rng(SEED + 100)
    inter_jaccards = []
    for _ in range(min(200, len(paragraphs) * 2)):
        i, j = rng.integers(0, len(paragraphs), size=2)
        if i == j:
            continue
        a, b = paragraphs[i], paragraphs[j]
        inter = len(a & b)
        union = len(a | b)
        if union > 0:
            inter_jaccards.append(inter / union)

    intra_mean = float(np.mean(intra_jaccards)) if intra_jaccards else 0
    inter_mean = float(np.mean(inter_jaccards)) if inter_jaccards else 0

    return {
        "intra_jaccard": round(intra_mean, 4),
        "inter_jaccard": round(inter_mean, 4),
        "ratio": round(intra_mean / inter_mean, 2) if inter_mean > 0 else 0,
        "note": "Real Voynich: intra=0.025, inter=0.009, ratio=2.78x",
    }


def _measure_word_section_mi(section_words: dict[str, list[str]]) -> dict:
    """Mutual information between words and sections."""
    total = sum(len(ws) for ws in section_words.values())
    if total == 0:
        return {"mi_bits": 0, "n_sections": 0}

    word_counts: Counter = Counter()
    section_counts: Counter = Counter()
    joint: Counter = Counter()

    for sec, words in section_words.items():
        section_counts[sec] += len(words)
        for w in words:
            word_counts[w] += 1
            joint[(w, sec)] += 1

    mi = 0.0
    for (w, s), n_ws in joint.items():
        p_ws = n_ws / total
        p_w = word_counts[w] / total
        p_s = section_counts[s] / total
        if p_ws > 0 and p_w > 0 and p_s > 0:
            mi += p_ws * math.log2(p_ws / (p_w * p_s))

    return {
        "mi_bits": round(mi, 4),
        "n_sections": len(section_words),
        "total_tokens": total,
        "note": "Real Voynich: MI=0.159 bits (z=+40.24)",
    }


def _measure_currier_diff(pages: list[dict]) -> dict:
    """Vocabulary overlap between Currier A and B pages."""
    vocab_a: set = set()
    vocab_b: set = set()

    for page in pages:
        lang = page.get("language", "?")
        if lang == "A":
            vocab_a.update(page["words"])
        elif lang == "B":
            vocab_b.update(page["words"])

    if not vocab_a or not vocab_b:
        return {"jaccard": 0, "n_a": 0, "n_b": 0}

    inter = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)
    jaccard = inter / union if union > 0 else 0

    return {
        "jaccard": round(jaccard, 4),
        "n_a_types": len(vocab_a),
        "n_b_types": len(vocab_b),
        "shared_types": inter,
        "note": "Real Voynich: Currier A/B differ (z=4.02/3.85)",
    }


def _measure_zipf(words: list[str]) -> dict:
    """Zipf law fit: log-log slope of rank vs frequency."""
    freq = Counter(words)
    if not freq:
        return {"slope": 0, "n_types": 0}

    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1, dtype=np.float64)
    freqs = np.array(sorted_freq, dtype=np.float64)

    # Log-log linear regression
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    slope, _ = np.polyfit(log_r, log_f, 1)

    return {
        "slope": round(float(slope), 3),
        "n_types": len(freq),
        "n_tokens": len(words),
        "note": "Zipf slope ~ -1 for natural language",
    }


def _measure_entropy(words: list[str]) -> dict:
    """Shannon entropy of the word distribution (bits)."""
    freq = Counter(words)
    total = sum(freq.values())
    if total == 0:
        return {"entropy_bits": 0}

    h = 0.0
    for c in freq.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)

    return {
        "entropy_bits": round(h, 3),
        "n_types": len(freq),
        "n_tokens": total,
        "note": "Real Voynich: 7.9-9.5 bits per hand",
    }


def _measure_hand_bigrams(hand_words: dict[str, list[str]]) -> dict:
    """Check if different hands have distinct bigram signatures (chi-square)."""
    hands = sorted(hand_words.keys())
    if len(hands) < 2:
        return {"chi2": 0, "p": 1, "n_hands": len(hands)}

    # Build bigram frequency per hand
    hand_bigrams: dict[str, Counter] = {}
    all_bigrams: set = set()
    for hand in hands:
        bg = Counter()
        for word in hand_words[hand]:
            for i in range(len(word) - 1):
                pair = word[i : i + 2]
                bg[pair] += 1
                all_bigrams.add(pair)
        hand_bigrams[hand] = bg

    # Build contingency table (top 50 bigrams)
    top_bigrams = sorted(all_bigrams, key=lambda b: sum(
        hand_bigrams[h].get(b, 0) for h in hands
    ), reverse=True)[:50]

    if len(top_bigrams) < 2:
        return {"chi2": 0, "p": 1, "n_hands": len(hands)}

    table = []
    for hand in hands:
        row = [hand_bigrams[hand].get(b, 0) for b in top_bigrams]
        if sum(row) > 0:
            table.append(row)

    if len(table) < 2:
        return {"chi2": 0, "p": 1, "n_hands": len(hands)}

    try:
        chi2, p, _, _ = chi2_contingency(table)
    except ValueError:
        chi2, p = 0.0, 1.0

    return {
        "chi2": round(float(chi2), 1),
        "p": round(float(p), 6),
        "n_hands": len(hands),
        "n_bigram_types": len(top_bigrams),
        "significant": p < 0.001,
        "note": "Real Voynich: all hands p=0 (distinct signatures)",
    }


def _measure_slot_grammar(words: list[str]) -> dict:
    """Measure positional character preferences (slot grammar strength).

    Computes chi-square of character × position contingency table.
    Strong slot grammar → high chi2.
    """
    max_pos = 5
    position_char: dict[int, Counter] = defaultdict(Counter)

    for word in words:
        for i, ch in enumerate(word[:max_pos]):
            position_char[i][ch] += 1

    # Build contingency table
    all_chars = set()
    for pos_counts in position_char.values():
        all_chars.update(pos_counts.keys())

    all_chars = sorted(all_chars)
    positions = sorted(position_char.keys())

    if len(all_chars) < 2 or len(positions) < 2:
        return {"chi2": 0, "p": 1}

    table = []
    for pos in positions:
        row = [position_char[pos].get(ch, 0) for ch in all_chars]
        table.append(row)

    try:
        chi2, p, dof, _ = chi2_contingency(table)
    except ValueError:
        chi2, p, dof = 0.0, 1.0, 0

    # Cramér's V for effect size
    n = sum(sum(row) for row in table)
    k = min(len(positions), len(all_chars))
    cramers_v = math.sqrt(chi2 / (n * (k - 1))) if n > 0 and k > 1 else 0

    return {
        "chi2": round(float(chi2), 1),
        "p": round(float(p), 6),
        "cramers_v": round(cramers_v, 4),
        "dof": int(dof),
        "significant": p < 0.001,
        "note": "Real Voynich: rigid 5-rule slot grammar",
    }


def _measure_vocabulary(words: list[str]) -> dict:
    """Type-token ratio and hapax legomena."""
    if not words:
        return {"ttr": 0, "hapax_ratio": 0}

    freq = Counter(words)
    n_types = len(freq)
    n_tokens = len(words)
    n_hapax = sum(1 for c in freq.values() if c == 1)

    return {
        "ttr": round(n_types / n_tokens, 4),
        "hapax_ratio": round(n_hapax / n_types, 4) if n_types > 0 else 0,
        "n_types": n_types,
        "n_tokens": n_tokens,
    }


def _measure_hand_entropy_variance(hand_words: dict[str, list[str]]) -> dict:
    """Per-hand entropy variance (Phase 0: range 7.9-9.5 bits)."""
    entropies = {}
    for hand, words in hand_words.items():
        if len(words) < 100:
            continue
        freq = Counter(words)
        total = sum(freq.values())
        h = 0.0
        for c in freq.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        entropies[hand] = round(h, 3)

    vals = list(entropies.values())
    return {
        "per_hand": entropies,
        "min": round(min(vals), 3) if vals else 0,
        "max": round(max(vals), 3) if vals else 0,
        "range": round(max(vals) - min(vals), 3) if vals else 0,
        "n_hands": len(entropies),
        "note": "Real Voynich: range 7.9-9.5 bits (1.6 bit spread)",
    }


def _measure_hand_pair_jaccard(
    hand_words: dict[str, list[str]], hand_a: str, hand_b: str,
) -> dict:
    """Vocabulary Jaccard between two specific hands (Phase 2c)."""
    vocab_a = set(hand_words.get(hand_a, []))
    vocab_b = set(hand_words.get(hand_b, []))

    if not vocab_a or not vocab_b:
        return {"jaccard": 0, "note": "hands not found"}

    inter = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)
    jaccard = inter / union if union > 0 else 0

    return {
        "jaccard": round(jaccard, 4),
        "n_a": len(vocab_a),
        "n_b": len(vocab_b),
        "shared": inter,
        "note": "Real Voynich: H1-H2 z=-8.09 (BELOW null, less overlap than random)",
    }


def _measure_section_pair_jaccard(
    section_words: dict[str, list[str]], sec_a: str, sec_b: str,
) -> dict:
    """Vocabulary Jaccard between two sections (Phase 8b: Astro-Zodiac)."""
    vocab_a = set(section_words.get(sec_a, []))
    vocab_b = set(section_words.get(sec_b, []))

    if not vocab_a or not vocab_b:
        return {"jaccard": 0, "note": "sections not found"}

    inter = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)
    jaccard = inter / union if union > 0 else 0

    return {
        "jaccard": round(jaccard, 4),
        "n_a": len(vocab_a),
        "n_b": len(vocab_b),
        "shared": inter,
        "note": "Real Voynich: Astro-Zodiac z=-6.02 (LESS similar than random)",
    }


def _measure_hand_unknown_anomaly(hand_words: dict[str, list[str]]) -> dict:
    """Hand ? entropy vs other hands (Phase 2a: z=+12.25 anomaly)."""
    if "?" not in hand_words or len(hand_words["?"]) < 100:
        return {"anomaly": False, "note": "hand ? not found"}

    # Entropy for hand ?
    freq_q = Counter(hand_words["?"])
    total_q = sum(freq_q.values())
    h_q = -sum((c / total_q) * math.log2(c / total_q) for c in freq_q.values() if c > 0)

    # Mean entropy for other hands
    other_h = []
    for hand, words in hand_words.items():
        if hand == "?" or len(words) < 100:
            continue
        freq = Counter(words)
        total = sum(freq.values())
        h = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
        other_h.append(h)

    if not other_h:
        return {"anomaly": False}

    mean_other = float(np.mean(other_h))
    std_other = float(np.std(other_h, ddof=1)) if len(other_h) > 1 else 1.0

    z = (h_q - mean_other) / std_other if std_other > 0 else 0

    return {
        "h_unknown": round(h_q, 3),
        "h_others_mean": round(mean_other, 3),
        "h_others_std": round(std_other, 3),
        "z_anomaly": round(z, 2),
        "anomaly": abs(z) > 2,
        "note": "Real Voynich: hand ? z=+12.25 (enormous anomaly)",
    }


def _measure_split_gallows(
    first_lines: list[list[str]],
    continuation_lines: list[list[str]],
) -> dict:
    """Split gallows should NOT concentrate at paragraph start (Phase 7b: z=+1.34 ns).

    Split gallows = cth, ckh, cph, cfh digraphs.
    """
    split_patterns = ("cth", "ckh", "cph", "cfh")

    def _split_rate(lines: list[list[str]]) -> float:
        total_words = 0
        split_words = 0
        for line in lines:
            for word in line:
                total_words += 1
                if any(pat in word for pat in split_patterns):
                    split_words += 1
        return split_words / total_words if total_words > 0 else 0

    first_rate = _split_rate(first_lines)
    cont_rate = _split_rate(continuation_lines)

    return {
        "para_first_rate": round(first_rate, 4),
        "continuation_rate": round(cont_rate, 4),
        "difference": round(first_rate - cont_rate, 4),
        "uniform": abs(first_rate - cont_rate) < 0.03,
        "note": "Real Voynich: split gallows uniform (z=+1.34, ns)",
    }


# =====================================================================
# Step 6: Compare real vs Rugg properties
# =====================================================================

def compare_properties(real: dict, rugg: dict) -> list[dict]:
    """Compare each property between real and Rugg-generated corpus.

    Returns list of comparison dicts with verdict per property.
    """
    comparisons = []

    # 1. Line self-containment
    comparisons.append(_compare(
        "line_self_containment",
        "Lines are self-contained units",
        real["line_self_containment"]["cross_rate"],
        rugg["line_self_containment"]["cross_rate"],
        direction="low_is_good",
        threshold=0.01,
    ))

    # 2. 'm' line-end concentration
    comparisons.append(_compare(
        "m_end_marker",
        "'m' concentrates at line-final position (word-final char with line-end preference)",
        real["m_end_marker"]["line_final_rate"],
        rugg["m_end_marker"]["line_final_rate"],
        direction="high_is_good",
        threshold=0.30,
    ))

    # 3. Gallows paragraph start
    comparisons.append(_compare(
        "gallows_para_start",
        "Simple gallows concentrate at paragraph start",
        real["gallows_para_start"]["difference"],
        rugg["gallows_para_start"]["difference"],
        direction="high_is_good",
        threshold=0.02,
    ))

    # 4. Paragraph coherence
    comparisons.append(_compare(
        "para_coherence",
        "Paragraphs have internal vocabulary coherence",
        real["para_coherence"]["ratio"],
        rugg["para_coherence"]["ratio"],
        direction="high_is_good",
        threshold=1.5,
    ))

    # 5. Word-section MI
    comparisons.append(_compare(
        "word_section_mi",
        "Words carry section-specific information",
        real["word_section_mi"]["mi_bits"],
        rugg["word_section_mi"]["mi_bits"],
        direction="high_is_good",
        threshold=0.05,
    ))

    # 6. Currier A/B
    comparisons.append(_compare(
        "currier_diff",
        "Currier A and B have different vocabularies",
        1 - real["currier_diff"]["jaccard"],  # dissimilarity
        1 - rugg["currier_diff"]["jaccard"],
        direction="high_is_good",
        threshold=0.3,
    ))

    # 7. Zipf
    comparisons.append(_compare(
        "zipf",
        "Word frequencies follow Zipf's law (slope ~ -1)",
        abs(real["zipf"]["slope"] + 1),  # distance from -1
        abs(rugg["zipf"]["slope"] + 1),
        direction="low_is_good",
        threshold=0.3,
    ))

    # 8. Entropy
    comparisons.append(_compare(
        "entropy",
        "Shannon entropy in natural range (7-10 bits)",
        real["entropy"]["entropy_bits"],
        rugg["entropy"]["entropy_bits"],
        direction="match",
        threshold=1.0,  # within 1 bit
    ))

    # 9. Hand bigram signatures
    comparisons.append(_compare(
        "hand_bigrams",
        "Different hands have distinct bigram signatures",
        1 if real["hand_bigrams"].get("significant", False) else 0,
        1 if rugg["hand_bigrams"].get("significant", False) else 0,
        direction="match_bool",
        threshold=0,
    ))

    # 10. Slot grammar
    comparisons.append(_compare(
        "slot_grammar",
        "Rigid positional character structure (slot grammar)",
        real["slot_grammar"]["cramers_v"],
        rugg["slot_grammar"]["cramers_v"],
        direction="high_is_good",
        threshold=0.1,
    ))

    # 11. Vocabulary richness
    comparisons.append(_compare(
        "vocabulary",
        "Vocabulary size and distribution",
        real["vocabulary"]["ttr"],
        rugg["vocabulary"]["ttr"],
        direction="match",
        threshold=0.05,
    ))

    # 12. Per-hand entropy range (Phase 0)
    comparisons.append(_compare(
        "hand_entropy_range",
        "Per-hand entropy spread ~1.6 bits (Phase 0)",
        real["hand_entropy_variance"]["range"],
        rugg["hand_entropy_variance"]["range"],
        direction="match",
        threshold=1.0,
    ))

    # 13. Hand 1-2 low Jaccard (Phase 2c)
    comparisons.append(_compare(
        "hand12_low_jaccard",
        "Hands 1-2 share LESS vocabulary than expected (Phase 2c: z=-8.09)",
        real["hand12_jaccard"]["jaccard"],
        rugg["hand12_jaccard"]["jaccard"],
        direction="low_is_good",
        threshold=0.15,
    ))

    # 14. Astro-Zodiac anti-correlation (Phase 8b)
    comparisons.append(_compare(
        "astro_zodiac_anti",
        "Astro-Zodiac share LESS vocabulary than random (Phase 8b: z=-6.02)",
        real["astro_zodiac_anti"]["jaccard"],
        rugg["astro_zodiac_anti"]["jaccard"],
        direction="low_is_good",
        threshold=0.15,
    ))

    # 15. Hand ? entropy anomaly (Phase 2a)
    comparisons.append(_compare(
        "hand_unknown_anomaly",
        "Hand ? has anomalous entropy vs other hands (Phase 2a: z=+12.25)",
        1 if real["hand_unknown_anomaly"].get("anomaly", False) else 0,
        1 if rugg["hand_unknown_anomaly"].get("anomaly", False) else 0,
        direction="match_bool",
        threshold=0,
    ))

    # 16. Split gallows uniform (Phase 7b)
    comparisons.append(_compare(
        "split_gallows_uniform",
        "Split gallows are NOT paragraph markers (Phase 7b: z=+1.34 ns)",
        1 if real["split_gallows_uniform"].get("uniform", False) else 0,
        1 if rugg["split_gallows_uniform"].get("uniform", False) else 0,
        direction="match_bool",
        threshold=0,
    ))

    return comparisons


def _compare(
    name: str,
    description: str,
    real_val: float,
    rugg_val: float,
    direction: str,
    threshold: float,
) -> dict:
    """Compare a single property.

    direction: 'high_is_good', 'low_is_good', 'match', 'match_bool'
    """
    if direction == "match_bool":
        reproduced = real_val == rugg_val
    elif direction == "match":
        reproduced = abs(real_val - rugg_val) <= threshold
    elif direction == "high_is_good":
        reproduced = rugg_val >= threshold
    elif direction == "low_is_good":
        reproduced = rugg_val <= threshold
    else:
        reproduced = False

    return {
        "property": name,
        "description": description,
        "real_value": round(real_val, 4) if isinstance(real_val, float) else real_val,
        "rugg_value": round(rugg_val, 4) if isinstance(rugg_val, float) else rugg_val,
        "reproduced": reproduced,
        "direction": direction,
        "threshold": threshold,
    }


# =====================================================================
# Step 7: Summary formatting
# =====================================================================

def format_summary(
    real_props: dict,
    rugg_props: dict,
    comparisons: list[dict],
) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("RUGG GRILLE TEST — Phase 27.9")
    lines.append("Can Rugg's (2004) Cardan grille reproduce confirmed properties?")
    lines.append("=" * 72)

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)

    lines.append(f"\nResult: {n_repro}/{n_total} properties reproduced by grille")
    lines.append("")

    # Table
    lines.append(f"{'Property':<30s} {'Real':>10s} {'Rugg':>10s} {'Reproduced':>12s}")
    lines.append("-" * 72)

    for c in comparisons:
        rv = c["real_value"]
        gv = c["rugg_value"]

        rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
        gv_str = f"{gv:.4f}" if isinstance(gv, float) else str(gv)
        verdict = "YES" if c["reproduced"] else "** NO **"

        lines.append(f"  {c['property']:<28s} {rv_str:>10s} {gv_str:>10s} {verdict:>12s}")

    lines.append("-" * 72)

    # Verdict
    lines.append("")
    if n_repro == n_total:
        lines.append("VERDICT: RUGG MODEL SUFFICIENT")
        lines.append("  The grille mechanism reproduces ALL confirmed properties.")
        lines.append("  No additional mechanism is needed beyond the table + grille.")
    else:
        missed = [c for c in comparisons if not c["reproduced"]]
        lines.append(f"VERDICT: RUGG MODEL INSUFFICIENT ({n_total - n_repro} properties missing)")
        lines.append("  The grille CANNOT reproduce these properties:")
        for m in missed:
            lines.append(f"    - {m['description']}")
            lines.append(f"      (real={m['real_value']}, rugg={m['rugg_value']})")
        lines.append("")
        lines.append("  These properties require something BEYOND a simple table + grille.")

    lines.append("")

    # Detail section
    lines.append("=" * 72)
    lines.append("DETAILED MEASUREMENTS")
    lines.append("=" * 72)

    for label, props in [("REAL VOYNICH", real_props), ("RUGG GRILLE", rugg_props)]:
        lines.append(f"\n--- {label} ---")
        for key, val in props.items():
            lines.append(f"  {key}:")
            if isinstance(val, dict):
                for k, v in val.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"    {val}")

    lines.append("")
    return "\n".join(lines) + "\n"


# =====================================================================
# Step 8: Save to SQLite
# =====================================================================

def save_to_db(config: ToolkitConfig, comparisons: list[dict], real: dict, rugg: dict):
    """Save comparison results to SQLite database."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}, skipping DB save")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS rugg_test")
    cur.execute("""
        CREATE TABLE rugg_test (
            property TEXT PRIMARY KEY,
            description TEXT,
            real_value REAL,
            rugg_value REAL,
            reproduced INTEGER,
            direction TEXT,
            threshold REAL
        )
    """)

    for c in comparisons:
        rv = float(c["real_value"]) if isinstance(c["real_value"], (int, float)) else 0
        gv = float(c["rugg_value"]) if isinstance(c["rugg_value"], (int, float)) else 0
        cur.execute(
            "INSERT INTO rugg_test VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c["property"], c["description"], rv, gv,
             1 if c["reproduced"] else 0, c["direction"], c["threshold"]),
        )

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Rugg Grille Test — can a Cardan grille reproduce confirmed Voynich properties?"""
    report_path = config.stats_dir / "rugg_test.json"
    summary_path = config.stats_dir / "rugg_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Rugg test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 27.9 — Rugg Grille Test")

    # 1. Parse real EVA corpus
    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    # 2. Extract syllable statistics
    print_step("Extracting syllable statistics for Rugg table...")
    stats = extract_syllable_stats(real_pages)
    click.echo(f"    Prefixes: {len(stats['prefixes'])} types, "
               f"Cores: {len(stats['cores'])} types, "
               f"Suffixes: {len(stats['suffixes'])} types")
    click.echo(f"    Top prefixes: {', '.join(f'{v}={p:.2f}' for v, p in stats['prefixes'][:5])}")
    click.echo(f"    Top suffixes: {', '.join(f'{v}={p:.2f}' for v, p in stats['suffixes'][:5])}")

    # 3. Generate Rugg corpus
    print_step(f"Generating Rugg corpus ({N_GRILLES} tables, "
               f"{TABLE_ROWS} rows each)...")
    rng = np.random.default_rng(SEED)
    rugg_pages = generate_rugg_corpus(real_pages, stats, rng, N_GRILLES)
    total_rugg = sum(len(p["words"]) for p in rugg_pages)
    click.echo(f"    Generated {total_rugg:,} words across {len(rugg_pages)} pages")

    # Show sample
    sample_page = rugg_pages[0]
    if sample_page["line_words"]:
        sample_line = sample_page["line_words"][0][:5]
        click.echo(f"    Sample (page 1, line 1): {' '.join(sample_line)}")

    # 4. Measure properties on real corpus
    print_step("Measuring 11 properties on REAL corpus...")
    real_props = measure_all_properties(real_pages)
    for key, val in real_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    # 5. Measure properties on Rugg corpus
    print_step("Measuring 11 properties on RUGG corpus...")
    rugg_props = measure_all_properties(rugg_pages)
    for key, val in rugg_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    # 6. Compare
    print_step("Comparing real vs Rugg...")
    comparisons = compare_properties(real_props, rugg_props)

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)
    click.echo(f"\n    RESULT: {n_repro}/{n_total} properties reproduced")

    for c in comparisons:
        icon = "OK" if c["reproduced"] else "MISS"
        click.echo(f"    [{icon:>4s}] {c['property']:<28s} "
                   f"real={c['real_value']:<10} rugg={c['rugg_value']:<10}")

    # 7. Save results
    print_step("Saving results...")

    report = {
        "real_properties": _serialise(real_props),
        "rugg_properties": _serialise(rugg_props),
        "comparisons": comparisons,
        "summary": {
            "reproduced": n_repro,
            "total": n_total,
            "verdict": "SUFFICIENT" if n_repro == n_total else "INSUFFICIENT",
            "missed": [c["property"] for c in comparisons if not c["reproduced"]],
        },
        "parameters": {
            "table_rows": TABLE_ROWS,
            "n_grilles": N_GRILLES,
            "seed": SEED,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(real_props, rugg_props, comparisons)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    # Save to DB
    save_to_db(config, comparisons, real_props, rugg_props)
    click.echo(f"    DB:   rugg_test table")

    # Print summary
    click.echo(f"\n{summary}")


def _compact_dict(d: dict) -> str:
    """Format dict as compact one-liner."""
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _serialise(props: dict) -> dict:
    """Make properties JSON-serialisable."""
    out = {}
    for k, v in props.items():
        if isinstance(v, dict):
            out[k] = {
                kk: (round(vv, 6) if isinstance(vv, float) else vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out
