"""
Phase 7d — Split gallows role analysis.

Split gallows (cth, ckh, cph, cfh) are NOT paragraph markers (Phase 7c, z=+0.36),
NOT section markers, NOT language markers, NOT hand-specific. Their function is
unknown. This module runs five discriminating tests to narrow down the surviving
hypotheses:

  7d-1 — Position within line
         Do words containing split gallows cluster at specific positions in the
         line (start, middle, end)? Null: shuffle word order within each line.

  7d-2 — Position within word
         Where does the split gallows trigram fall inside the word?
         Fixed position (prefix/suffix) vs variable (part of root).
         Null: shuffle characters within each word containing a split gallows.

  7d-3 — Complementary distribution with simple gallows
         If cth is an allograph of t, lines with cth should have fewer t.
         For each pair (cXh, X), compare rate of X on cXh-lines vs other lines.
         Null: shuffle which lines contain cXh words (within folio).

  7d-4 — Folio distribution
         Are split gallows concentrated on certain folios?
         Coefficient of variation + Gini vs null (random redistribution).
         Enriched with section/hand metadata.

  7d-5 — Lexical context (register role)
         Are SG-words more like labels (unique, line-initial) or tally tokens
         (repeated, continuation)? Compare line-initial rate, TTR, frequency.
         Null: randomly relabel same count of words as "SG-words".

Hypothesis elimination matrix:

  Within-line field markers: mid-line clustering, variable word-pos, register sections
  Semantic determinatives:   position 0-1, fixed prefix, uniform across sections
  Allographs (cth ~ t):     complementary distribution (z << 0), mirrors simple gallows
  Register-level indicators: concentrated in register sections, tally-like lexical profile

All tests on raw EVA — no decoding, no lexicon.

Output:
  split_gallows_test.json
  split_gallows_test_summary.txt
  DB table: split_gallows_test
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .currier_line_test import parse_ivtff_lines, SPLIT_GALLOWS
from .full_decode import SECTION_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SEED = 42
N_NULL = 500

SIMPLE_GALLOWS = {"t", "k", "p", "f"}

# Mapping: split gallows -> corresponding simple gallows
SG_TO_SIMPLE = {"cth": "t", "ckh": "k", "cph": "p", "cfh": "f"}


# =====================================================================
# Helpers
# =====================================================================

def find_sg_in_word(word: str) -> list[tuple[str, int, int]]:
    """Find all split gallows occurrences in a word.

    Returns list of (sg_type, char_start, char_end).
    Searches longest first to avoid substring collisions.
    """
    hits: list[tuple[str, int, int]] = []
    for sg in sorted(SPLIT_GALLOWS, key=len, reverse=True):
        start = 0
        while True:
            idx = word.find(sg, start)
            if idx == -1:
                break
            hits.append((sg, idx, idx + len(sg)))
            start = idx + len(sg)
    return hits


def word_contains_sg(word: str) -> bool:
    """Check if word contains any split gallows sequence."""
    return any(sg in word for sg in SPLIT_GALLOWS)


def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# 7d-1 — Position within line
# =====================================================================

def test_position_in_line(lines: list[dict]) -> dict:
    """For each word containing a split gallows, record its normalized
    position in the line (0.0 = first word, 1.0 = last word).

    Returns: mean position, fraction at position 0, count, histogram.
    """
    positions: list[float] = []
    n_at_start = 0
    n_total = 0
    by_gallows: dict[str, list[float]] = {sg: [] for sg in sorted(SPLIT_GALLOWS)}

    for line in lines:
        words = line["words"]
        if len(words) < 2:
            continue
        for i, w in enumerate(words):
            hits = find_sg_in_word(w)
            if hits:
                pos = i / (len(words) - 1) if len(words) > 1 else 0.0
                positions.append(pos)
                n_total += 1
                if i == 0:
                    n_at_start += 1
                for sg, _, _ in hits:
                    by_gallows[sg].append(pos)

    mean_pos = float(np.mean(positions)) if positions else 0.0
    frac_initial = n_at_start / n_total if n_total > 0 else 0.0

    # Histogram: split into thirds
    n_start = sum(1 for p in positions if p < 0.33)
    n_mid = sum(1 for p in positions if 0.33 <= p < 0.67)
    n_end = sum(1 for p in positions if p >= 0.67)

    return {
        "mean_position": round(mean_pos, 4),
        "frac_line_initial": round(frac_initial, 4),
        "n_sg_words": n_total,
        "n_at_start": n_at_start,
        "position_thirds": {"start": n_start, "mid": n_mid, "end": n_end},
        "by_gallows": {
            sg: {
                "n": len(ps),
                "mean_pos": round(float(np.mean(ps)), 4) if ps else None,
            }
            for sg, ps in by_gallows.items()
        },
    }


def null_position_in_line(lines: list[dict],
                          n_perms: int = N_NULL,
                          seed: int = SEED) -> dict:
    """Null: shuffle word order within each line, recompute mean position
    and line-initial fraction of SG-words."""
    rng = random.Random(seed)
    null_means: list[float] = []
    null_initials: list[float] = []

    for _ in range(n_perms):
        positions: list[float] = []
        n_at_start = 0
        n_total = 0
        for line in lines:
            words = list(line["words"])
            if len(words) < 2:
                continue
            rng.shuffle(words)
            for i, w in enumerate(words):
                if word_contains_sg(w):
                    pos = i / (len(words) - 1)
                    positions.append(pos)
                    n_total += 1
                    if i == 0:
                        n_at_start += 1
        if positions:
            null_means.append(float(np.mean(positions)))
            null_initials.append(n_at_start / n_total if n_total > 0 else 0.0)

    return {
        "mean_position": {
            "null_mean": float(np.mean(null_means)),
            "null_std": float(np.std(null_means, ddof=1)),
        },
        "frac_line_initial": {
            "null_mean": float(np.mean(null_initials)),
            "null_std": float(np.std(null_initials, ddof=1)),
        },
    }


# =====================================================================
# 7d-2 — Position within word
# =====================================================================

def test_position_in_word(lines: list[dict]) -> dict:
    """For each split gallows in a word, record its character-level
    fractional position (0.0 = start, 1.0 = end).

    Returns: mean position, fraction at word-start, fraction at word-end.
    """
    positions: list[float] = []
    n_prefix = 0  # starts at char 0
    n_suffix = 0  # ends at last char
    n_total = 0
    by_gallows: dict[str, list[float]] = {sg: [] for sg in sorted(SPLIT_GALLOWS)}

    for line in lines:
        for w in line["words"]:
            hits = find_sg_in_word(w)
            for sg, start, end in hits:
                wlen = len(w)
                if wlen <= len(sg):
                    frac = 0.5  # sg IS the whole word
                else:
                    frac = start / (wlen - len(sg))
                positions.append(frac)
                n_total += 1
                if start == 0:
                    n_prefix += 1
                if end == wlen:
                    n_suffix += 1
                by_gallows[sg].append(frac)

    mean_pos = float(np.mean(positions)) if positions else 0.0
    frac_prefix = n_prefix / n_total if n_total > 0 else 0.0
    frac_suffix = n_suffix / n_total if n_total > 0 else 0.0
    frac_middle = 1.0 - frac_prefix - frac_suffix

    return {
        "mean_char_position": round(mean_pos, 4),
        "frac_prefix": round(frac_prefix, 4),
        "frac_middle": round(frac_middle, 4),
        "frac_suffix": round(frac_suffix, 4),
        "n_total": n_total,
        "n_prefix": n_prefix,
        "n_suffix": n_suffix,
        "by_gallows": {
            sg: {
                "n": len(ps),
                "mean_pos": round(float(np.mean(ps)), 4) if ps else None,
                "frac_prefix": round(
                    sum(1 for p in ps if p == 0.0) / len(ps), 4
                ) if ps else None,
            }
            for sg, ps in by_gallows.items()
        },
    }


def null_position_in_word(lines: list[dict],
                          n_perms: int = N_NULL,
                          seed: int = SEED) -> dict:
    """Null: for each word containing a split gallows, shuffle its characters
    and find where the trigram lands (if it reassembles by chance).
    More robust: randomly reinsert the split gallows at a valid position."""
    rng = random.Random(seed)

    # Collect all (word, sg_type) pairs
    sg_words: list[tuple[str, str]] = []
    for line in lines:
        for w in line["words"]:
            hits = find_sg_in_word(w)
            for sg, _, _ in hits:
                sg_words.append((w, sg))

    if not sg_words:
        return {
            "mean_char_position": {"null_mean": 0.0, "null_std": 0.0},
        }

    null_means: list[float] = []

    for _ in range(n_perms):
        positions: list[float] = []
        for word, sg in sg_words:
            # Remove the sg from the word, reinsert at random position
            remainder = word.replace(sg, "", 1)
            wlen = len(remainder) + len(sg)
            max_insert = len(remainder)
            insert_pos = rng.randint(0, max_insert)
            if wlen <= len(sg):
                frac = 0.5
            else:
                frac = insert_pos / (wlen - len(sg))
            positions.append(frac)
        null_means.append(float(np.mean(positions)))

    return {
        "mean_char_position": {
            "null_mean": float(np.mean(null_means)),
            "null_std": float(np.std(null_means, ddof=1)),
        },
    }


# =====================================================================
# 7d-3 — Complementary distribution with simple gallows
# =====================================================================

def _count_simple_in_words(words: list[str], target: str) -> int:
    """Count occurrences of a simple gallows character in a list of words,
    excluding occurrences that are part of split gallows."""
    count = 0
    for w in words:
        # Remove all split gallows first, then count the simple character
        clean = w
        for sg in SPLIT_GALLOWS:
            clean = clean.replace(sg, "")
        count += clean.count(target)
    return count


def test_complementary_dist(lines: list[dict]) -> dict:
    """For each pair (cXh, X): on lines where cXh appears, compute rate
    of standalone X per word. Compare to lines without cXh.

    If allographs: negative difference (complementary distribution).
    """
    pairs: dict[str, dict] = {}

    for sg, simple in SG_TO_SIMPLE.items():
        lines_with: list[list[str]] = []
        lines_without: list[list[str]] = []

        for line in lines:
            words = line["words"]
            if not words:
                continue
            has_sg = any(sg in w for w in words)
            if has_sg:
                lines_with.append(words)
            else:
                lines_without.append(words)

        n_with = len(lines_with)
        n_without = len(lines_without)

        # Rate of standalone simple gallows per word
        if n_with > 0:
            total_words_with = sum(len(ws) for ws in lines_with)
            count_simple_with = sum(
                _count_simple_in_words(ws, simple) for ws in lines_with
            )
            rate_with = count_simple_with / total_words_with if total_words_with > 0 else 0.0
        else:
            rate_with = 0.0
            total_words_with = 0

        if n_without > 0:
            total_words_without = sum(len(ws) for ws in lines_without)
            count_simple_without = sum(
                _count_simple_in_words(ws, simple) for ws in lines_without
            )
            rate_without = (
                count_simple_without / total_words_without
                if total_words_without > 0
                else 0.0
            )
        else:
            rate_without = 0.0
            total_words_without = 0

        diff = rate_with - rate_without

        pairs[f"{sg}_vs_{simple}"] = {
            "observed_rate_diff": round(diff, 6),
            "rate_with_sg": round(rate_with, 6),
            "rate_without_sg": round(rate_without, 6),
            "n_lines_with": n_with,
            "n_lines_without": n_without,
        }

    return pairs


def null_complementary_dist(lines: list[dict],
                            n_perms: int = N_NULL,
                            seed: int = SEED) -> dict:
    """Null: for each pair (cXh, X), shuffle which lines are "with-cXh"
    lines (within folio), recompute rate difference."""
    rng = random.Random(seed)

    # Group lines by folio for within-folio shuffling
    by_folio: dict[str, list[dict]] = {}
    for line in lines:
        folio = line["folio"]
        if folio not in by_folio:
            by_folio[folio] = []
        by_folio[folio].append(line)

    # Pre-compute which lines have each sg type
    sg_line_flags: dict[str, list[bool]] = {}
    for sg in SG_TO_SIMPLE:
        flags = []
        for line in lines:
            flags.append(any(sg in w for w in line["words"]))
        sg_line_flags[sg] = flags

    null_diffs: dict[str, list[float]] = {
        f"{sg}_vs_{simple}": [] for sg, simple in SG_TO_SIMPLE.items()
    }

    for _ in range(n_perms):
        for sg, simple in SG_TO_SIMPLE.items():
            key = f"{sg}_vs_{simple}"
            # Count how many lines per folio have this sg
            n_with_per_folio: dict[str, int] = {}
            for line in lines:
                folio = line["folio"]
                if folio not in n_with_per_folio:
                    n_with_per_folio[folio] = 0
                if any(sg in w for w in line["words"]):
                    n_with_per_folio[folio] += 1

            # Shuffle: within each folio, randomly assign which lines are "with"
            shuffled_with: list[list[str]] = []
            shuffled_without: list[list[str]] = []
            for folio, folio_lines in by_folio.items():
                n_with = n_with_per_folio.get(folio, 0)
                indices = list(range(len(folio_lines)))
                rng.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j < n_with:
                        shuffled_with.append(folio_lines[idx]["words"])
                    else:
                        shuffled_without.append(folio_lines[idx]["words"])

            # Compute rate difference on shuffled assignment
            if shuffled_with:
                tw = sum(len(ws) for ws in shuffled_with)
                cw = sum(_count_simple_in_words(ws, simple) for ws in shuffled_with)
                rw = cw / tw if tw > 0 else 0.0
            else:
                rw = 0.0

            if shuffled_without:
                tow = sum(len(ws) for ws in shuffled_without)
                cow = sum(_count_simple_in_words(ws, simple) for ws in shuffled_without)
                row = cow / tow if tow > 0 else 0.0
            else:
                row = 0.0

            null_diffs[key].append(rw - row)

    result = {}
    for key, diffs in null_diffs.items():
        result[key] = {
            "null_mean": float(np.mean(diffs)),
            "null_std": float(np.std(diffs, ddof=1)),
        }
    return result


# =====================================================================
# 7d-4 — Folio distribution
# =====================================================================

def test_folio_distribution(lines: list[dict],
                            folio_meta: dict[str, dict]) -> dict:
    """Count split gallows per folio. Compute CV and Gini.
    Enrich with section/hand metadata."""
    folio_sg: Counter = Counter()
    folio_words: Counter = Counter()

    for line in lines:
        folio = line["folio"]
        words = line["words"]
        folio_words[folio] += len(words)
        for w in words:
            if word_contains_sg(w):
                folio_sg[folio] += 1

    # Rate per folio (only folios with words)
    rates: list[float] = []
    folio_details: list[dict] = []
    for folio in sorted(folio_words.keys()):
        nw = folio_words[folio]
        nsg = folio_sg.get(folio, 0)
        rate = nsg / nw if nw > 0 else 0.0
        rates.append(rate)
        meta = folio_meta.get(folio, {})
        folio_details.append({
            "folio": folio,
            "n_words": nw,
            "n_sg": nsg,
            "sg_rate": round(rate, 4),
            "section": meta.get("section", "?"),
            "hand": meta.get("hand", "?"),
        })

    rates_arr = np.array(rates)
    mean_rate = float(np.mean(rates_arr))
    std_rate = float(np.std(rates_arr, ddof=1))
    cv = std_rate / mean_rate if mean_rate > 1e-10 else 0.0

    # Gini coefficient
    sorted_rates = np.sort(rates_arr)
    n = len(sorted_rates)
    if n > 0 and sorted_rates.sum() > 0:
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_rates) - (n + 1) * np.sum(sorted_rates))
                      / (n * np.sum(sorted_rates)))
    else:
        gini = 0.0

    # Top quartile folios grouped by section
    folio_details.sort(key=lambda d: d["sg_rate"], reverse=True)
    top_q = folio_details[:max(1, len(folio_details) // 4)]
    top_sections = Counter(d["section"] for d in top_q)
    top_hands = Counter(d["hand"] for d in top_q)

    return {
        "cv": round(cv, 4),
        "gini": round(gini, 4),
        "mean_rate": round(mean_rate, 6),
        "std_rate": round(std_rate, 6),
        "n_folios": len(rates),
        "top_quartile_sections": dict(top_sections.most_common()),
        "top_quartile_hands": dict(top_hands.most_common()),
        "top_10_folios": folio_details[:10],
    }


def null_folio_distribution(lines: list[dict],
                            n_perms: int = N_NULL,
                            seed: int = SEED) -> dict:
    """Null: randomly redistribute which words are 'SG-words' across all
    folios, preserving total count and folio sizes."""
    rng = random.Random(seed)

    # Build flat list of (folio, is_sg) tuples
    entries: list[tuple[str, bool]] = []
    for line in lines:
        folio = line["folio"]
        for w in line["words"]:
            entries.append((folio, word_contains_sg(w)))

    n_sg = sum(1 for _, is_sg in entries if is_sg)
    folios_list = [folio for folio, _ in entries]

    # Folio word counts (fixed)
    folio_word_counts: Counter = Counter(folios_list)
    folio_order = sorted(folio_word_counts.keys())

    null_cvs: list[float] = []

    for _ in range(n_perms):
        # Randomly assign n_sg words as "SG" across all positions
        indices = list(range(len(entries)))
        rng.shuffle(indices)
        sg_indices = set(indices[:n_sg])

        folio_sg: Counter = Counter()
        for idx in sg_indices:
            folio_sg[folios_list[idx]] += 1

        rates = []
        for folio in folio_order:
            nw = folio_word_counts[folio]
            nsg = folio_sg.get(folio, 0)
            rates.append(nsg / nw if nw > 0 else 0.0)

        arr = np.array(rates)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1))
        cv = s / m if m > 1e-10 else 0.0
        null_cvs.append(cv)

    return {
        "cv": {
            "null_mean": float(np.mean(null_cvs)),
            "null_std": float(np.std(null_cvs, ddof=1)),
        },
    }


# =====================================================================
# 7d-5 — Lexical context (register role)
# =====================================================================

def test_lexical_context(lines: list[dict]) -> dict:
    """Compare SG-words vs non-SG words on line-initial rate, TTR, frequency.

    SG-words = words containing at least one split gallows.
    """
    sg_tokens: list[str] = []
    sg_at_initial = 0
    non_sg_tokens: list[str] = []
    non_sg_at_initial = 0

    for line in lines:
        words = line["words"]
        for i, w in enumerate(words):
            if word_contains_sg(w):
                sg_tokens.append(w)
                if i == 0:
                    sg_at_initial += 1
            else:
                non_sg_tokens.append(w)
                if i == 0:
                    non_sg_at_initial += 1

    n_sg = len(sg_tokens)
    n_non = len(non_sg_tokens)

    sg_types = len(set(sg_tokens))
    non_types = len(set(non_sg_tokens))

    sg_ttr = sg_types / n_sg if n_sg > 0 else 0.0
    non_ttr = non_types / n_non if n_non > 0 else 0.0

    sg_initial_rate = sg_at_initial / n_sg if n_sg > 0 else 0.0
    non_initial_rate = non_sg_at_initial / n_non if n_non > 0 else 0.0

    # Mean frequency (how often each word type appears)
    sg_freq = Counter(sg_tokens)
    non_freq = Counter(non_sg_tokens)
    sg_mean_freq = float(np.mean(list(sg_freq.values()))) if sg_freq else 0.0
    non_mean_freq = float(np.mean(list(non_freq.values()))) if non_freq else 0.0

    initial_rate_diff = sg_initial_rate - non_initial_rate

    return {
        "initial_rate_diff": round(initial_rate_diff, 6),
        "sg_words": {
            "n_tokens": n_sg,
            "n_types": sg_types,
            "ttr": round(sg_ttr, 4),
            "initial_rate": round(sg_initial_rate, 4),
            "mean_freq": round(sg_mean_freq, 2),
            "n_at_initial": sg_at_initial,
        },
        "non_sg_words": {
            "n_tokens": n_non,
            "n_types": non_types,
            "ttr": round(non_ttr, 4),
            "initial_rate": round(non_initial_rate, 4),
            "mean_freq": round(non_mean_freq, 2),
            "n_at_initial": non_sg_at_initial,
        },
    }


def null_lexical_context(lines: list[dict],
                         n_perms: int = N_NULL,
                         seed: int = SEED) -> dict:
    """Null: randomly relabel the same number of word tokens as 'SG-words',
    recompute line-initial rate difference."""
    rng = random.Random(seed)

    # Build flat list of (word, is_initial) tuples
    entries: list[tuple[str, bool]] = []
    n_sg_real = 0
    for line in lines:
        words = line["words"]
        for i, w in enumerate(words):
            entries.append((w, i == 0))
            if word_contains_sg(w):
                n_sg_real += 1

    null_diffs: list[float] = []
    indices = list(range(len(entries)))

    for _ in range(n_perms):
        rng.shuffle(indices)
        sg_set = set(indices[:n_sg_real])

        sg_initial = 0
        sg_total = 0
        non_initial = 0
        non_total = 0

        for idx in range(len(entries)):
            _, is_init = entries[idx]
            if idx in sg_set:
                sg_total += 1
                if is_init:
                    sg_initial += 1
            else:
                non_total += 1
                if is_init:
                    non_initial += 1

        sg_rate = sg_initial / sg_total if sg_total > 0 else 0.0
        non_rate = non_initial / non_total if non_total > 0 else 0.0
        null_diffs.append(sg_rate - non_rate)

    return {
        "initial_rate_diff": {
            "null_mean": float(np.mean(null_diffs)),
            "null_std": float(np.std(null_diffs, ddof=1)),
        },
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(results: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS split_gallows_test")
    cur.execute("""
        CREATE TABLE split_gallows_test (
            test           TEXT,
            key            TEXT,
            observed       REAL,
            null_mean      REAL,
            null_std       REAL,
            z_score        REAL,
            detail_json    TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    for test_name, test_data in results.items():
        rows = test_data.get("db_rows", [])
        for row in rows:
            cur.execute(
                "INSERT INTO split_gallows_test VALUES (?,?,?,?,?,?,?)",
                (
                    test_name,
                    row.get("key", test_name),
                    row.get("observed"),
                    row.get("null_mean"),
                    row.get("null_std"),
                    row.get("z"),
                    json.dumps(row.get("detail", {}), ensure_ascii=False),
                ),
            )
    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(results: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 7d — Split Gallows Role Analysis")
    lines.append("  What do split gallows (cth, ckh, cph, cfh) actually do?")
    lines.append("  Five tests to discriminate between surviving hypotheses.")
    lines.append("  Bonferroni threshold: |z| > 3.3 for significance (p < 0.001/5)")
    lines.append("=" * 80)

    # 7d-1
    d1 = results.get("7d_position_in_line", {})
    lines.append("\n── 7d-1: Position within line ──")
    lines.append(f"  SG-words found:         {d1.get('n_sg_words', 0)}")
    lines.append(f"  Mean position (0=start): {d1.get('mean_position', 'n/a')}")
    z_pos = d1.get("z_mean_position")
    lines.append(f"  z (mean position):       {_fz(z_pos)}")
    lines.append(f"  Fraction at line-start:  {d1.get('frac_line_initial', 'n/a')}")
    z_init = d1.get("z_line_initial")
    lines.append(f"  z (line-initial):        {_fz(z_init)}")
    thirds = d1.get("position_thirds", {})
    lines.append(f"  Distribution:            start={thirds.get('start',0)}  "
                 f"mid={thirds.get('mid',0)}  end={thirds.get('end',0)}")

    # 7d-2
    d2 = results.get("7d_position_in_word", {})
    lines.append("\n── 7d-2: Position within word ──")
    lines.append(f"  Mean char position:      {d2.get('mean_char_position', 'n/a')}")
    z_wp = d2.get("z_mean_char_position")
    lines.append(f"  z (char position):       {_fz(z_wp)}")
    lines.append(f"  Prefix (word-start):     {d2.get('frac_prefix', 'n/a')}  "
                 f"({d2.get('n_prefix', 0)} / {d2.get('n_total', 0)})")
    lines.append(f"  Middle:                  {d2.get('frac_middle', 'n/a')}")
    lines.append(f"  Suffix (word-end):       {d2.get('frac_suffix', 'n/a')}")

    # 7d-3
    d3 = results.get("7d_complementary_dist", {})
    lines.append("\n── 7d-3: Complementary distribution (allograph test) ──")
    lines.append(f"  {'Pair':>12}  {'rate_with':>10}  {'rate_without':>12}  {'diff':>8}  {'z':>8}")
    lines.append("  " + "-" * 55)
    pairs = d3.get("pairs", {})
    for key in sorted(pairs.keys()):
        p = pairs[key]
        lines.append(
            f"  {key:>12}  {p.get('rate_with_sg', 0):>10.4f}  "
            f"{p.get('rate_without_sg', 0):>12.4f}  "
            f"{p.get('observed_rate_diff', 0):>+8.4f}  "
            f"{_fz(p.get('z')):>8}"
        )
    overall_z = d3.get("overall_z")
    if overall_z is not None:
        lines.append(f"  Overall z (mean of 4):   {_fz(overall_z)}")

    # 7d-4
    d4 = results.get("7d_folio_distribution", {})
    lines.append("\n── 7d-4: Folio distribution ──")
    lines.append(f"  Folios analyzed:         {d4.get('n_folios', 0)}")
    lines.append(f"  CV (coeff of variation): {d4.get('cv', 'n/a')}")
    z_cv = d4.get("z_cv")
    lines.append(f"  z (CV):                  {_fz(z_cv)}")
    lines.append(f"  Gini coefficient:        {d4.get('gini', 'n/a')}")
    tqs = d4.get("top_quartile_sections", {})
    if tqs:
        sec_str = ", ".join(f"{SECTION_NAMES.get(s,s)}={n}" for s, n in
                            sorted(tqs.items(), key=lambda x: -x[1]))
        lines.append(f"  Top quartile by section: {sec_str}")

    # 7d-5
    d5 = results.get("7d_lexical_context", {})
    lines.append("\n── 7d-5: Lexical context (register role) ──")
    sg = d5.get("sg_words", {})
    nsg = d5.get("non_sg_words", {})
    lines.append(f"  {'':>20}  {'SG-words':>10}  {'non-SG':>10}")
    lines.append("  " + "-" * 44)
    lines.append(f"  {'Tokens':>20}  {sg.get('n_tokens',0):>10,}  {nsg.get('n_tokens',0):>10,}")
    lines.append(f"  {'Types':>20}  {sg.get('n_types',0):>10,}  {nsg.get('n_types',0):>10,}")
    lines.append(f"  {'TTR':>20}  {sg.get('ttr',0):>10.4f}  {nsg.get('ttr',0):>10.4f}")
    lines.append(f"  {'Line-initial rate':>20}  {sg.get('initial_rate',0):>10.4f}  "
                 f"{nsg.get('initial_rate',0):>10.4f}")
    lines.append(f"  {'Mean type freq':>20}  {sg.get('mean_freq',0):>10.1f}  "
                 f"{nsg.get('mean_freq',0):>10.1f}")
    z_lex = d5.get("z_initial_rate_diff")
    lines.append(f"  Initial rate diff:       {d5.get('initial_rate_diff', 'n/a')}")
    lines.append(f"  z (initial rate diff):   {_fz(z_lex)}")

    # Hypothesis verdict
    lines.append("\n" + "=" * 80)
    lines.append("  HYPOTHESIS ASSESSMENT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Hypothesis                  Supporting evidence         Verdict")
    lines.append("  " + "-" * 70)

    # Collect z-scores for verdict
    verdicts = _assess_hypotheses(results)
    for hyp, (evidence, verdict) in verdicts.items():
        lines.append(f"  {hyp:<30}{evidence:<28}{verdict}")

    lines.append("")
    lines.append("  Note: multiple testing (5 tests). Bonferroni: |z| > 3.3 for significance.")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


def _fz(z) -> str:
    """Format z-score for display."""
    if z is None:
        return "n/a"
    return f"{z:+.2f}"


def _assess_hypotheses(results: dict) -> dict:
    """Assess each hypothesis based on test results."""
    d3 = results.get("7d_complementary_dist", {})
    d1 = results.get("7d_position_in_line", {})
    d5 = results.get("7d_lexical_context", {})

    # Check complementary distribution (allograph test)
    comp_zs = []
    for key, p in d3.get("pairs", {}).items():
        z = p.get("z")
        if z is not None:
            comp_zs.append(z)
    mean_comp_z = float(np.mean(comp_zs)) if comp_zs else 0.0
    all_comp_neg = all(z < -3.3 for z in comp_zs) if comp_zs else False

    z_pos = d1.get("z_mean_position")
    z_init = d1.get("z_line_initial")
    z_lex = d5.get("z_initial_rate_diff")

    verdicts = {}

    # Allographs
    if all_comp_neg:
        verdicts["Allographs (cXh ~ X)"] = ("7d-3: complementary dist", "SUPPORTED")
    elif comp_zs and all(abs(z) < 2 for z in comp_zs):
        verdicts["Allographs (cXh ~ X)"] = ("7d-3: no complement.", "ELIMINATED")
    else:
        verdicts["Allographs (cXh ~ X)"] = (f"7d-3: z_mean={mean_comp_z:+.1f}", "INCONCLUSIVE")

    # Determinatives
    if z_init is not None and z_init > 3.3:
        verdicts["Semantic determinatives"] = ("7d-1: line-initial bias", "SUPPORTED")
    elif z_init is not None and abs(z_init) < 2:
        verdicts["Semantic determinatives"] = ("7d-1: no initial bias", "WEAKENED")
    else:
        verdicts["Semantic determinatives"] = (f"7d-1: z={_fz(z_init)}", "INCONCLUSIVE")

    # Field markers
    if z_pos is not None and abs(z_pos) > 3.3:
        verdicts["Within-line field markers"] = ("7d-1: position bias", "SUPPORTED")
    else:
        verdicts["Within-line field markers"] = (f"7d-1: z={_fz(z_pos)}", "INCONCLUSIVE")

    # Register indicators
    if z_lex is not None and z_lex < -3.3:
        verdicts["Register-level indicators"] = ("7d-5: tally-like", "SUPPORTED")
    elif z_lex is not None and z_lex > 3.3:
        verdicts["Register-level indicators"] = ("7d-5: label-like (not tally)", "WEAKENED")
    else:
        verdicts["Register-level indicators"] = (f"7d-5: z={_fz(z_lex)}", "INCONCLUSIVE")

    return verdicts


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 7d: split gallows role analysis — 5 discriminating tests."""
    report_path = config.stats_dir / "split_gallows_test.json"
    summary_path = config.stats_dir / "split_gallows_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  split_gallows_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 7d — Split Gallows Role Analysis")

    # 1. Parse IVTFF (line-level with paragraph structure)
    print_step("Parsing IVTFF (line-level)...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    ivtff_lines = parse_ivtff_lines(eva_file)
    # Filter out labels for most tests (keep for folio distribution)
    content_lines = [l for l in ivtff_lines if l["para_type"] != "label"]
    click.echo(f"    {len(ivtff_lines)} lines total, {len(content_lines)} content lines")

    # Count total SG words
    n_sg_total = sum(
        1 for line in content_lines for w in line["words"] if word_contains_sg(w)
    )
    click.echo(f"    {n_sg_total} words containing split gallows")

    # 2. Parse EVA corpus for folio metadata (section, hand)
    print_step("Loading folio metadata...")
    eva_data = parse_eva_words(eva_file)
    folio_meta: dict[str, dict] = {}
    for page in eva_data["pages"]:
        folio_meta[page["folio"]] = {
            "section": page.get("section", "?"),
            "hand": page.get("hand", "?"),
        }
    click.echo(f"    {len(folio_meta)} folios with metadata")

    results: dict = {}

    # ── 7d-1: Position within line ──
    print_step("7d-1: Position within line (500 perms)...")
    obs1 = test_position_in_line(content_lines)
    null1 = null_position_in_line(content_lines, seed=SEED)
    z_pos = z_score(obs1["mean_position"],
                    null1["mean_position"]["null_mean"],
                    null1["mean_position"]["null_std"])
    z_init = z_score(obs1["frac_line_initial"],
                     null1["frac_line_initial"]["null_mean"],
                     null1["frac_line_initial"]["null_std"])
    results["7d_position_in_line"] = {
        **obs1,
        "z_mean_position": round(z_pos, 3) if z_pos is not None else None,
        "z_line_initial": round(z_init, 3) if z_init is not None else None,
        "null_mean_position": null1["mean_position"],
        "null_frac_initial": null1["frac_line_initial"],
        "db_rows": [
            {"key": "mean_position", "observed": obs1["mean_position"],
             "null_mean": null1["mean_position"]["null_mean"],
             "null_std": null1["mean_position"]["null_std"],
             "z": round(z_pos, 3) if z_pos is not None else None,
             "detail": obs1},
            {"key": "frac_line_initial", "observed": obs1["frac_line_initial"],
             "null_mean": null1["frac_line_initial"]["null_mean"],
             "null_std": null1["frac_line_initial"]["null_std"],
             "z": round(z_init, 3) if z_init is not None else None},
        ],
    }
    click.echo(f"    mean_pos={obs1['mean_position']:.4f} z={_fz(z_pos)}  "
               f"frac_initial={obs1['frac_line_initial']:.4f} z={_fz(z_init)}")

    # ── 7d-2: Position within word ──
    print_step("7d-2: Position within word (500 perms)...")
    obs2 = test_position_in_word(content_lines)
    null2 = null_position_in_word(content_lines, seed=SEED + 100)
    z_wp = z_score(obs2["mean_char_position"],
                   null2["mean_char_position"]["null_mean"],
                   null2["mean_char_position"]["null_std"])
    results["7d_position_in_word"] = {
        **obs2,
        "z_mean_char_position": round(z_wp, 3) if z_wp is not None else None,
        "null_mean_char_position": null2["mean_char_position"],
        "db_rows": [
            {"key": "mean_char_position", "observed": obs2["mean_char_position"],
             "null_mean": null2["mean_char_position"]["null_mean"],
             "null_std": null2["mean_char_position"]["null_std"],
             "z": round(z_wp, 3) if z_wp is not None else None,
             "detail": obs2},
        ],
    }
    click.echo(f"    mean_char_pos={obs2['mean_char_position']:.4f} z={_fz(z_wp)}  "
               f"prefix={obs2['frac_prefix']:.3f} suffix={obs2['frac_suffix']:.3f}")

    # ── 7d-3: Complementary distribution ──
    print_step("7d-3: Complementary distribution (500 perms)...")
    obs3 = test_complementary_dist(content_lines)
    null3 = null_complementary_dist(content_lines, seed=SEED + 200)

    pairs_result: dict = {}
    comp_zs: list[float] = []
    for key, obs_pair in obs3.items():
        n3 = null3.get(key, {})
        z_c = z_score(obs_pair["observed_rate_diff"],
                      n3.get("null_mean", 0), n3.get("null_std", 1e-10))
        obs_pair["z"] = round(z_c, 3) if z_c is not None else None
        pairs_result[key] = obs_pair
        if z_c is not None:
            comp_zs.append(z_c)
        click.echo(f"    {key}: diff={obs_pair['observed_rate_diff']:+.4f} z={_fz(z_c)}")

    overall_comp_z = float(np.mean(comp_zs)) if comp_zs else None
    results["7d_complementary_dist"] = {
        "pairs": pairs_result,
        "overall_z": round(overall_comp_z, 3) if overall_comp_z is not None else None,
        "db_rows": [
            {"key": key, "observed": p["observed_rate_diff"],
             "null_mean": null3.get(key, {}).get("null_mean"),
             "null_std": null3.get(key, {}).get("null_std"),
             "z": p.get("z"),
             "detail": p}
            for key, p in pairs_result.items()
        ],
    }

    # ── 7d-4: Folio distribution ──
    print_step("7d-4: Folio distribution (500 perms)...")
    obs4 = test_folio_distribution(ivtff_lines, folio_meta)  # all lines incl labels
    null4 = null_folio_distribution(ivtff_lines, seed=SEED + 300)
    z_cv = z_score(obs4["cv"],
                   null4["cv"]["null_mean"],
                   null4["cv"]["null_std"])
    results["7d_folio_distribution"] = {
        **obs4,
        "z_cv": round(z_cv, 3) if z_cv is not None else None,
        "null_cv": null4["cv"],
        "db_rows": [
            {"key": "cv", "observed": obs4["cv"],
             "null_mean": null4["cv"]["null_mean"],
             "null_std": null4["cv"]["null_std"],
             "z": round(z_cv, 3) if z_cv is not None else None,
             "detail": {k: v for k, v in obs4.items() if k != "top_10_folios"}},
        ],
    }
    click.echo(f"    CV={obs4['cv']:.4f} z={_fz(z_cv)}  "
               f"Gini={obs4['gini']:.4f}  "
               f"folios={obs4['n_folios']}")

    # ── 7d-5: Lexical context ──
    print_step("7d-5: Lexical context (500 perms)...")
    obs5 = test_lexical_context(content_lines)
    null5 = null_lexical_context(content_lines, seed=SEED + 400)
    z_lex = z_score(obs5["initial_rate_diff"],
                    null5["initial_rate_diff"]["null_mean"],
                    null5["initial_rate_diff"]["null_std"])
    results["7d_lexical_context"] = {
        **obs5,
        "z_initial_rate_diff": round(z_lex, 3) if z_lex is not None else None,
        "null_initial_rate_diff": null5["initial_rate_diff"],
        "db_rows": [
            {"key": "initial_rate_diff", "observed": obs5["initial_rate_diff"],
             "null_mean": null5["initial_rate_diff"]["null_mean"],
             "null_std": null5["initial_rate_diff"]["null_std"],
             "z": round(z_lex, 3) if z_lex is not None else None,
             "detail": obs5},
        ],
    }
    click.echo(f"    SG initial_rate={obs5['sg_words']['initial_rate']:.4f}  "
               f"non-SG={obs5['non_sg_words']['initial_rate']:.4f}  "
               f"diff={obs5['initial_rate_diff']:+.4f} z={_fz(z_lex)}")

    # ── Save JSON ──
    print_step("Saving JSON...")
    # Remove db_rows from JSON output (internal bookkeeping)
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items() if kk != "db_rows"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # ── Save TXT ──
    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # ── Save to DB ──
    print_step("Writing DB table split_gallows_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(results, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo("    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
