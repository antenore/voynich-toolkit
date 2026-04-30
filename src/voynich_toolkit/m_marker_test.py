"""
Phase 12 — Why does 'm' concentrate at line-end?

The character 'm' appears in line-final position ~71% of the time (z=+55.5,
Phase 7a), but 29% is NOT at line-end, and the rate varies 49-94% by section.
No generative model (Markov, Rugg, Timm, Naibbe) reproduces this concentration.

This module tests three competing hypotheses:
  A. LINGUISTIC SUFFIX — 'm' encodes a language feature that naturally
     closes units (e.g., Latin accusative -am/-um/-em)
  B. SYSTEM CONVENTION — 'm' is a scribal preference for closing lines
     with -m-final words, not independent of content
  C. SEGMENTATION ARTIFACT — 'm' is just a frequent word-final character
     that happens to land at line-end by chance

Phase 15 audit result: 'm' is a WORD-FINAL CHARACTER with SECTION-DEPENDENT
line-end preference. NOT a pure system marker (varies 49-94% by section,
27% of word-final 'm' is not at line-end, pharma has 15% medial 'm').

Sub-tests:
  12a — Per-section variance of 'm' line-final rate
  12b — Per-hand variance of 'm' line-final rate
  12c — Word-final vs line-final conditional probability
  12d — Correlation with line length
  12e — All-character line-end bias ranking

All tests are on raw EVA — no decoding, no lexicon.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import pearsonr

from .config import ToolkitConfig
from .currier_line_test import parse_ivtff_lines
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_PERM = 500
EVA_CHARS = "acdefghiklmnopqrsty"


# =====================================================================
# Helper: enrich lines with section/hand metadata from parse_eva_words
# =====================================================================

def enrich_lines_with_metadata(lines: list[dict], pages: list[dict]) -> None:
    """Add section and hand metadata to each line dict (in-place).

    Matches lines to pages by folio. Each page has line_words which
    maps 1:1 to consecutive lines on that folio.
    """
    # Build folio → page lookup
    folio_pages: dict[str, dict] = {}
    for page in pages:
        folio_pages[page["folio"]] = page

    for line in lines:
        folio = line["folio"]
        page = folio_pages.get(folio)
        if page:
            line["section"] = page.get("section", "?")
            line["hand"] = page.get("hand", "?")
        else:
            line["section"] = "?"
            line["hand"] = "?"


# =====================================================================
# 12a — Per-section 'm' line-final rate
# =====================================================================

def test_m_per_section(lines: list[dict], n_perms: int = N_PERM,
                       seed: int = SEED) -> dict:
    """Compute 'm' line-final rate for each section.

    If rate is constant across sections → system marker.
    If rate varies significantly → linguistic (content-dependent).

    Also computes null model per section (shuffle words within section).
    """
    # Group lines by section
    section_lines: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        if line["para_type"] == "label":
            continue
        section_lines[line.get("section", "?")].append(line)

    results = {}
    rng = random.Random(seed)

    for section in sorted(section_lines.keys()):
        slines = section_lines[section]
        if len(slines) < 10:
            continue

        # Observed: fraction of 'm' occurrences that are line-final
        total_m = 0
        final_m = 0
        for line in slines:
            for w in line["words"]:
                total_m += w.count("m")
            last_word = line["words"][-1]
            if last_word and last_word[-1] == "m":
                final_m += 1

        observed_rate = final_m / total_m if total_m > 0 else 0.0

        # Null model: shuffle words within this section's lines
        nulls = []
        all_words = [w for line in slines for w in line["words"]]
        line_lengths = [len(line["words"]) for line in slines]

        for _ in range(n_perms):
            shuffled = list(all_words)
            rng.shuffle(shuffled)
            null_final = 0
            pos = 0
            for ll in line_lengths:
                chunk = shuffled[pos:pos + ll]
                pos += ll
                if chunk and chunk[-1] and chunk[-1][-1] == "m":
                    null_final += 1
            null_rate = null_final / total_m if total_m > 0 else 0.0
            nulls.append(null_rate)

        null_mean = float(np.mean(nulls))
        null_std = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else 0.001
        z = (observed_rate - null_mean) / null_std if null_std > 0 else 0.0

        results[section] = {
            "n_lines": len(slines),
            "total_m": total_m,
            "line_final_m": final_m,
            "observed_rate": round(observed_rate, 4),
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "z_score": round(z, 2),
        }

    # Cross-section variance
    rates = [v["observed_rate"] for v in results.values()]
    cv = float(np.std(rates) / np.mean(rates)) if np.mean(rates) > 0 else 0.0

    return {
        "per_section": results,
        "cross_section_cv": round(cv, 4),
        "cross_section_mean": round(float(np.mean(rates)), 4),
        "cross_section_std": round(float(np.std(rates)), 4),
        "n_sections": len(results),
        "interpretation": (
            "SYSTEM_MARKER" if cv < 0.10
            else "LINGUISTIC" if cv > 0.25
            else "AMBIGUOUS"
        ),
    }


# =====================================================================
# 12b — Per-hand 'm' line-final rate
# =====================================================================

def test_m_per_hand(lines: list[dict], n_perms: int = N_PERM,
                    seed: int = SEED) -> dict:
    """Compute 'm' line-final rate for each Davis hand.

    If rate is constant across hands → system rule.
    If one hand differs → different scribal convention.
    """
    hand_lines: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        if line["para_type"] == "label":
            continue
        hand_lines[line.get("hand", "?")].append(line)

    results = {}
    rng = random.Random(seed)

    for hand in sorted(hand_lines.keys()):
        hlines = hand_lines[hand]
        if len(hlines) < 20:
            continue

        total_m = 0
        final_m = 0
        for line in hlines:
            for w in line["words"]:
                total_m += w.count("m")
            last_word = line["words"][-1]
            if last_word and last_word[-1] == "m":
                final_m += 1

        observed_rate = final_m / total_m if total_m > 0 else 0.0

        # Null: shuffle within hand's lines
        all_words = [w for line in hlines for w in line["words"]]
        line_lengths = [len(line["words"]) for line in hlines]

        nulls = []
        for _ in range(n_perms):
            shuffled = list(all_words)
            rng.shuffle(shuffled)
            null_final = 0
            pos = 0
            for ll in line_lengths:
                chunk = shuffled[pos:pos + ll]
                pos += ll
                if chunk and chunk[-1] and chunk[-1][-1] == "m":
                    null_final += 1
            null_rate = null_final / total_m if total_m > 0 else 0.0
            nulls.append(null_rate)

        null_mean = float(np.mean(nulls))
        null_std = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else 0.001
        z = (observed_rate - null_mean) / null_std if null_std > 0 else 0.0

        results[hand] = {
            "n_lines": len(hlines),
            "total_m": total_m,
            "line_final_m": final_m,
            "observed_rate": round(observed_rate, 4),
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "z_score": round(z, 2),
        }

    rates = [v["observed_rate"] for v in results.values()]
    cv = float(np.std(rates) / np.mean(rates)) if np.mean(rates) > 0 else 0.0

    return {
        "per_hand": results,
        "cross_hand_cv": round(cv, 4),
        "cross_hand_mean": round(float(np.mean(rates)), 4),
        "cross_hand_std": round(float(np.std(rates)), 4),
        "n_hands": len(results),
        "interpretation": (
            "SYSTEM_RULE" if cv < 0.10
            else "SCRIBE_VARIABLE" if cv > 0.25
            else "AMBIGUOUS"
        ),
    }


# =====================================================================
# 12c — Word-final vs line-final conditional probability
# =====================================================================

def test_m_word_vs_line_final(lines: list[dict]) -> dict:
    """Compare P(word ends with 'm' | word is line-final) vs
    P(word ends with 'm' | word is NOT line-final).

    If the two are equal → 'm' is a word-level suffix (linguistic)
    If line-final >> non-line-final → 'm' is a line-level marker (system)
    """
    # Count words ending with 'm' by position
    line_final_total = 0
    line_final_m = 0
    non_final_total = 0
    non_final_m = 0

    for line in lines:
        if line["para_type"] == "label":
            continue
        words = line["words"]
        for i, w in enumerate(words):
            if not w:
                continue
            is_last = (i == len(words) - 1)
            ends_m = w[-1] == "m"

            if is_last:
                line_final_total += 1
                if ends_m:
                    line_final_m += 1
            else:
                non_final_total += 1
                if ends_m:
                    non_final_m += 1

    p_final = line_final_m / line_final_total if line_final_total > 0 else 0
    p_non_final = non_final_m / non_final_total if non_final_total > 0 else 0

    # Two-proportion z-test
    p_pooled = (line_final_m + non_final_m) / (line_final_total + non_final_total) \
        if (line_final_total + non_final_total) > 0 else 0
    if p_pooled > 0 and p_pooled < 1:
        se = math.sqrt(p_pooled * (1 - p_pooled) *
                       (1 / line_final_total + 1 / non_final_total))
        z = (p_final - p_non_final) / se if se > 0 else 0.0
    else:
        z = 0.0

    ratio = p_final / p_non_final if p_non_final > 0 else float("inf")

    return {
        "line_final_words": line_final_total,
        "line_final_ending_m": line_final_m,
        "p_m_given_line_final": round(p_final, 4),
        "non_final_words": non_final_total,
        "non_final_ending_m": non_final_m,
        "p_m_given_non_final": round(p_non_final, 4),
        "ratio": round(ratio, 2),
        "z_score": round(z, 2),
        "interpretation": (
            "SYSTEM_MARKER" if ratio > 3.0 and z > 5.0
            else "LINGUISTIC_SUFFIX" if ratio < 1.5
            else "MIXED"
        ),
    }


# =====================================================================
# 12d — Correlation with line length
# =====================================================================

def test_m_vs_line_length(lines: list[dict]) -> dict:
    """Correlate line length (word count) with whether line ends in 'm'.

    If correlated → segmentation artifact.
    If uncorrelated → intentional placement.
    """
    lengths = []
    ends_m = []

    for line in lines:
        if line["para_type"] == "label":
            continue
        words = line["words"]
        if not words:
            continue
        lengths.append(len(words))
        ends_m.append(1 if words[-1][-1] == "m" else 0)

    lengths = np.array(lengths, dtype=float)
    ends_m = np.array(ends_m, dtype=float)

    r, p = pearsonr(lengths, ends_m)

    # Also: mean line length when ending in 'm' vs not
    m_lines = lengths[ends_m == 1]
    non_m_lines = lengths[ends_m == 0]

    return {
        "n_lines": len(lengths),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p), 6),
        "mean_length_m_final": round(float(np.mean(m_lines)), 2) if len(m_lines) > 0 else 0,
        "mean_length_non_m_final": round(float(np.mean(non_m_lines)), 2) if len(non_m_lines) > 0 else 0,
        "n_m_final_lines": int(np.sum(ends_m)),
        "pct_m_final": round(float(np.mean(ends_m)) * 100, 1),
        "interpretation": (
            "SEGMENTATION_ARTIFACT" if abs(r) > 0.15 and p < 0.01
            else "NOT_ARTIFACT" if abs(r) < 0.05
            else "WEAK_CORRELATION"
        ),
    }


# =====================================================================
# 12e — All-character line-end bias ranking
# =====================================================================

def test_all_char_line_end_bias(lines: list[dict], n_perms: int = N_PERM,
                                seed: int = SEED) -> dict:
    """Compute line-final bias for ALL 19 EVA characters.

    Rank by bias strength. If only 'm' is special → unique marker.
    If a group of characters shows bias → suffix class.
    """
    # Count total and line-final for each char
    total_char: Counter = Counter()
    final_char: Counter = Counter()

    for line in lines:
        if line["para_type"] == "label":
            continue
        for w in line["words"]:
            for ch in w:
                total_char[ch] += 1
        last_word = line["words"][-1]
        if last_word:
            final_char[last_word[-1]] += 1

    # Compute observed rate
    char_bias = {}
    for ch in EVA_CHARS:
        t = total_char.get(ch, 0)
        f = final_char.get(ch, 0)
        if t < 10:
            continue
        char_bias[ch] = {
            "total": t,
            "line_final": f,
            "observed_rate": round(f / t, 4) if t > 0 else 0.0,
        }

    # Null model for top characters (only those with observed_rate > 0.05)
    rng = random.Random(seed)

    # Pre-compute shuffleable structure
    folio_lines: dict[str, list[list[str]]] = defaultdict(list)
    for line in lines:
        if line["para_type"] == "label":
            continue
        folio_lines[line["folio"]].append(list(line["words"]))

    for ch in list(char_bias.keys()):
        if char_bias[ch]["observed_rate"] < 0.02:
            char_bias[ch]["z_score"] = None
            char_bias[ch]["null_mean"] = None
            continue

        total = char_bias[ch]["total"]
        nulls = []

        for _ in range(n_perms):
            null_final = 0
            for folio, flines in folio_lines.items():
                all_w = [w for line in flines for w in line]
                rng.shuffle(all_w)
                pos = 0
                for line in flines:
                    n = len(line)
                    chunk = all_w[pos:pos + n]
                    pos += n
                    if chunk and chunk[-1] and chunk[-1][-1] == ch:
                        null_final += 1
            null_rate = null_final / total if total > 0 else 0.0
            nulls.append(null_rate)

        null_mean = float(np.mean(nulls))
        null_std = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else 0.001
        z = (char_bias[ch]["observed_rate"] - null_mean) / null_std if null_std > 0 else 0.0

        char_bias[ch]["null_mean"] = round(null_mean, 4)
        char_bias[ch]["null_std"] = round(null_std, 4)
        char_bias[ch]["z_score"] = round(z, 2)

    # Rank by observed rate
    ranked = sorted(char_bias.items(), key=lambda x: -x[1]["observed_rate"])

    # Count how many chars have z > 3 (significant line-final bias)
    significant = [ch for ch, v in char_bias.items()
                   if v.get("z_score") is not None and v["z_score"] > 3.0]

    return {
        "per_char": dict(ranked),
        "top5": [(ch, v["observed_rate"]) for ch, v in ranked[:5]],
        "n_significant": len(significant),
        "significant_chars": sorted(significant),
        "interpretation": (
            "UNIQUE_MARKER" if len(significant) <= 2
            else "SUFFIX_CLASS" if len(significant) <= 5
            else "BROAD_PATTERN"
        ),
    }


# =====================================================================
# Summary formatting
# =====================================================================

def format_summary(results: dict) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 12 — Why is 'm' a line-end marker?")
    lines.append("=" * 72)

    # 12a — Per-section
    r12a = results["12a_per_section"]
    lines.append(f"\n  12a — PER-SECTION 'm' LINE-FINAL RATE")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Section':<10} {'Lines':>6} {'Total m':>8} {'Final m':>8} "
                 f"{'Rate':>8} {'Null':>8} {'z':>8}")
    for sec, v in sorted(r12a["per_section"].items()):
        lines.append(f"  {sec:<10} {v['n_lines']:>6} {v['total_m']:>8} "
                     f"{v['line_final_m']:>8} {v['observed_rate']:>8.4f} "
                     f"{v['null_mean']:>8.4f} {v['z_score']:>+8.2f}")
    lines.append(f"\n  Cross-section CV: {r12a['cross_section_cv']:.4f} "
                 f"(mean={r12a['cross_section_mean']:.4f}, "
                 f"std={r12a['cross_section_std']:.4f})")
    lines.append(f"  → {r12a['interpretation']}")

    # 12b — Per-hand
    r12b = results["12b_per_hand"]
    lines.append(f"\n  12b — PER-HAND 'm' LINE-FINAL RATE")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Hand':<10} {'Lines':>6} {'Total m':>8} {'Final m':>8} "
                 f"{'Rate':>8} {'Null':>8} {'z':>8}")
    for hand, v in sorted(r12b["per_hand"].items()):
        lines.append(f"  {hand:<10} {v['n_lines']:>6} {v['total_m']:>8} "
                     f"{v['line_final_m']:>8} {v['observed_rate']:>8.4f} "
                     f"{v['null_mean']:>8.4f} {v['z_score']:>+8.2f}")
    lines.append(f"\n  Cross-hand CV: {r12b['cross_hand_cv']:.4f} "
                 f"(mean={r12b['cross_hand_mean']:.4f}, "
                 f"std={r12b['cross_hand_std']:.4f})")
    lines.append(f"  → {r12b['interpretation']}")

    # 12c — Word-final vs line-final
    r12c = results["12c_word_vs_line"]
    lines.append(f"\n  12c — WORD-FINAL vs LINE-FINAL CONDITIONAL PROBABILITY")
    lines.append("  " + "-" * 66)
    lines.append(f"  P(word ends 'm' | line-final word):     "
                 f"{r12c['p_m_given_line_final']:.4f} "
                 f"({r12c['line_final_ending_m']}/{r12c['line_final_words']})")
    lines.append(f"  P(word ends 'm' | NOT line-final word): "
                 f"{r12c['p_m_given_non_final']:.4f} "
                 f"({r12c['non_final_ending_m']}/{r12c['non_final_words']})")
    lines.append(f"  Ratio: {r12c['ratio']:.2f}x, z={r12c['z_score']:+.2f}")
    lines.append(f"  → {r12c['interpretation']}")

    # 12d — Line length correlation
    r12d = results["12d_line_length"]
    lines.append(f"\n  12d — CORRELATION WITH LINE LENGTH")
    lines.append("  " + "-" * 66)
    lines.append(f"  Pearson r: {r12d['pearson_r']:.4f} (p={r12d['pearson_p']:.6f})")
    lines.append(f"  Mean length (m-final lines): {r12d['mean_length_m_final']:.2f} words")
    lines.append(f"  Mean length (non-m lines):   {r12d['mean_length_non_m_final']:.2f} words")
    lines.append(f"  Lines ending in 'm': {r12d['pct_m_final']:.1f}%")
    lines.append(f"  → {r12d['interpretation']}")

    # 12e — All-character ranking
    r12e = results["12e_all_char_bias"]
    lines.append(f"\n  12e — ALL-CHARACTER LINE-END BIAS RANKING")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Char':<6} {'Total':>8} {'Final':>8} {'Rate':>8} "
                 f"{'Null':>8} {'z':>8}")
    for ch, v in r12e["per_char"].items():
        z_str = f"{v['z_score']:+8.2f}" if v.get("z_score") is not None else "    N/A"
        null_str = f"{v['null_mean']:>8.4f}" if v.get("null_mean") is not None else "     N/A"
        lines.append(f"  {ch:<6} {v['total']:>8} {v['line_final']:>8} "
                     f"{v['observed_rate']:>8.4f} {null_str} {z_str}")
    lines.append(f"\n  Significant chars (z>3): {r12e['significant_chars']}")
    lines.append(f"  → {r12e['interpretation']}")

    # Overall verdict
    lines.append("\n" + "=" * 72)
    lines.append("  OVERALL VERDICT")
    lines.append("=" * 72)

    verdicts = {
        "section": r12a["interpretation"],
        "hand": r12b["interpretation"],
        "word_vs_line": r12c["interpretation"],
        "line_length": r12d["interpretation"],
        "char_ranking": r12e["interpretation"],
    }

    # Determine overall
    system_markers = sum(1 for v in verdicts.values()
                         if v in ("SYSTEM_MARKER", "SYSTEM_RULE",
                                  "UNIQUE_MARKER", "NOT_ARTIFACT"))
    linguistic = sum(1 for v in verdicts.values()
                     if v in ("LINGUISTIC", "LINGUISTIC_SUFFIX",
                              "SCRIBE_VARIABLE", "SUFFIX_CLASS"))
    artifact = sum(1 for v in verdicts.values()
                   if v in ("SEGMENTATION_ARTIFACT",))

    if system_markers >= 3:
        overall = ("SYSTEM MARKER — 'm' is a writing system convention "
                   "applied at line-end, independent of content and scribe")
    elif linguistic >= 3:
        overall = ("LINGUISTIC SUFFIX — 'm' reflects a property of the "
                   "underlying language, varying with content")
    elif artifact >= 2:
        overall = ("SEGMENTATION ARTIFACT — 'm' at line-end is a "
                   "byproduct of how text fills lines")
    else:
        overall = ("MIXED — evidence points in multiple directions; "
                   "'m' may have both linguistic and systemic roles")

    lines.append(f"\n  Verdicts: {verdicts}")
    lines.append(f"  System={system_markers}, Linguistic={linguistic}, Artifact={artifact}")
    lines.append(f"\n  {overall}")
    lines.append("=" * 72)

    return "\n".join(lines) + "\n"


# =====================================================================
# Save to DB
# =====================================================================

def save_to_db(config: ToolkitConfig, results: dict):
    """Save results to SQLite database."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}, skipping")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS m_marker_test")
    cur.execute("""
        CREATE TABLE m_marker_test (
            test TEXT,
            key TEXT,
            observed REAL,
            null_mean REAL,
            null_std REAL,
            z_score REAL,
            detail_json TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    def _insert(test, key, observed, null_mean=None, null_std=None,
                z_score=None, detail=None):
        cur.execute(
            "INSERT INTO m_marker_test VALUES (?, ?, ?, ?, ?, ?, ?)",
            (test, key, observed, null_mean, null_std, z_score,
             json.dumps(detail) if detail else None),
        )

    # 12a: per-section
    for sec, v in results["12a_per_section"]["per_section"].items():
        _insert("12a_per_section", sec,
                v["observed_rate"], v["null_mean"], v["null_std"], v["z_score"])
    _insert("12a_per_section", "_cv",
            results["12a_per_section"]["cross_section_cv"])

    # 12b: per-hand
    for hand, v in results["12b_per_hand"]["per_hand"].items():
        _insert("12b_per_hand", hand,
                v["observed_rate"], v["null_mean"], v["null_std"], v["z_score"])
    _insert("12b_per_hand", "_cv",
            results["12b_per_hand"]["cross_hand_cv"])

    # 12c: word vs line
    r = results["12c_word_vs_line"]
    _insert("12c_word_vs_line", "p_line_final", r["p_m_given_line_final"])
    _insert("12c_word_vs_line", "p_non_final", r["p_m_given_non_final"])
    _insert("12c_word_vs_line", "ratio", r["ratio"],
            z_score=r["z_score"])

    # 12d: line length
    r = results["12d_line_length"]
    _insert("12d_line_length", "pearson_r", r["pearson_r"])
    _insert("12d_line_length", "pct_m_final", r["pct_m_final"])

    # 12e: all-char (top 5)
    for ch, v in list(results["12e_all_char_bias"]["per_char"].items())[:10]:
        _insert("12e_all_char_bias", ch,
                v["observed_rate"],
                v.get("null_mean"), v.get("null_std"), v.get("z_score"))

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 12 — Why is 'm' a line-end marker?"""
    report_path = config.stats_dir / "m_marker_test.json"
    summary_path = config.stats_dir / "m_marker_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  m-marker test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 12 — Why is 'm' a line-end marker?")

    # 1. Parse IVTFF for line structure
    print_step("Parsing IVTFF for line structure...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    lines = parse_ivtff_lines(eva_file)
    click.echo(f"    {len(lines)} lines parsed")

    # 2. Parse EVA for section/hand metadata
    print_step("Enriching with section/hand metadata...")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    enrich_lines_with_metadata(lines, pages)

    sections_found = set(line.get("section", "?") for line in lines)
    hands_found = set(line.get("hand", "?") for line in lines)
    click.echo(f"    Sections: {sorted(sections_found)}")
    click.echo(f"    Hands: {sorted(hands_found)}")

    results = {}

    # 3. Test 12a — per-section
    print_step("12a — Per-section 'm' line-final rate...")
    results["12a_per_section"] = test_m_per_section(lines)
    r12a = results["12a_per_section"]
    click.echo(f"    CV={r12a['cross_section_cv']:.4f} → {r12a['interpretation']}")

    # 4. Test 12b — per-hand
    print_step("12b — Per-hand 'm' line-final rate...")
    results["12b_per_hand"] = test_m_per_hand(lines)
    r12b = results["12b_per_hand"]
    click.echo(f"    CV={r12b['cross_hand_cv']:.4f} → {r12b['interpretation']}")

    # 5. Test 12c — word-final vs line-final
    print_step("12c — Word-final vs line-final conditional probability...")
    results["12c_word_vs_line"] = test_m_word_vs_line_final(lines)
    r12c = results["12c_word_vs_line"]
    click.echo(f"    P(m|final)={r12c['p_m_given_line_final']:.4f}, "
               f"P(m|non-final)={r12c['p_m_given_non_final']:.4f}, "
               f"ratio={r12c['ratio']:.2f}x → {r12c['interpretation']}")

    # 6. Test 12d — line length
    print_step("12d — Correlation with line length...")
    results["12d_line_length"] = test_m_vs_line_length(lines)
    r12d = results["12d_line_length"]
    click.echo(f"    r={r12d['pearson_r']:.4f}, p={r12d['pearson_p']:.6f} "
               f"→ {r12d['interpretation']}")

    # 7. Test 12e — all-character ranking
    print_step("12e — All-character line-end bias ranking...")
    results["12e_all_char_bias"] = test_all_char_line_end_bias(lines)
    r12e = results["12e_all_char_bias"]
    click.echo(f"    Top 5: {r12e['top5']}")
    click.echo(f"    Significant (z>3): {r12e['significant_chars']} "
               f"→ {r12e['interpretation']}")

    # 8. Save
    print_step("Saving results...")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, results)
    click.echo(f"    DB:   m_marker_test table")

    click.echo(f"\n{summary}")
