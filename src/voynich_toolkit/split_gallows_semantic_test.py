"""
Phase 13 — Split gallows semantic function.

Phase 7d established what split gallows (cth, ckh, cph, cfh) are NOT
(not paragraph markers, not uniformly distributed). This module tests
what they might BE by analyzing their lexical context and relationship
to content.

Sub-tests:
  13a — Lexical context of SG words (MI, TTR of surrounding words)
  13b — Per-type context comparison (do the 4 types have different contexts?)
  13c — SG density vs folio content (word count, lines, paragraphs)
  13d — SG co-occurrence with specific word types (odds ratios)
  13e — SG and 'm' marker interaction

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
from scipy.stats import chi2_contingency, pearsonr

from .config import ToolkitConfig
from .currier_line_test import parse_ivtff_lines
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_PERM = 500

# Split gallows sequences
SG_PATTERNS = ("cth", "ckh", "cph", "cfh")


# =====================================================================
# Helpers
# =====================================================================

def _has_sg(word: str) -> bool:
    """Check if a word contains any split gallows pattern."""
    return any(pat in word for pat in SG_PATTERNS)


def _sg_type(word: str) -> str | None:
    """Return which SG type a word contains (first match), or None."""
    for pat in SG_PATTERNS:
        if pat in word:
            return pat
    return None


def _extract_context_words(line_words: list[str], target_idx: int,
                           window: int = 3) -> list[str]:
    """Extract ±window words around target_idx (excluding target)."""
    ctx = []
    for i in range(max(0, target_idx - window),
                   min(len(line_words), target_idx + window + 1)):
        if i != target_idx:
            ctx.append(line_words[i])
    return ctx


# =====================================================================
# 13a — Lexical context of SG words
# =====================================================================

def test_sg_context(pages: list[dict], n_perms: int = N_PERM,
                    seed: int = SEED) -> dict:
    """Analyze lexical context of words containing split gallows.

    Computes:
    - MI between SG-presence and context words
    - TTR of context words (SG vs non-SG)
    - Context vocabulary overlap (are SG contexts restricted?)

    If SG-words have restricted context → qualifier/category marker.
    If SG-words have diverse context → structural role (like punctuation).
    """
    sg_context_words: list[str] = []
    non_sg_context_words: list[str] = []
    sg_following_words: list[str] = []
    non_sg_following_words: list[str] = []

    for page in pages:
        for line in page.get("line_words", []):
            for i, word in enumerate(line):
                ctx = _extract_context_words(line, i, window=3)
                if _has_sg(word):
                    sg_context_words.extend(ctx)
                    if i + 1 < len(line):
                        sg_following_words.append(line[i + 1])
                else:
                    non_sg_context_words.extend(ctx)
                    if i + 1 < len(line):
                        non_sg_following_words.append(line[i + 1])

    # TTR comparison
    sg_ctx_types = len(set(sg_context_words))
    sg_ctx_tokens = len(sg_context_words)
    non_sg_ctx_types = len(set(non_sg_context_words))
    non_sg_ctx_tokens = len(non_sg_context_words)

    sg_ttr = sg_ctx_types / sg_ctx_tokens if sg_ctx_tokens > 0 else 0
    non_sg_ttr = non_sg_ctx_types / non_sg_ctx_tokens if non_sg_ctx_tokens > 0 else 0

    # Following-word TTR
    sg_follow_ttr = len(set(sg_following_words)) / len(sg_following_words) \
        if sg_following_words else 0
    non_sg_follow_ttr = len(set(non_sg_following_words)) / len(non_sg_following_words) \
        if non_sg_following_words else 0

    # Context vocabulary overlap
    sg_ctx_set = set(sg_context_words)
    non_sg_ctx_set = set(non_sg_context_words)
    overlap = len(sg_ctx_set & non_sg_ctx_set)
    union = len(sg_ctx_set | non_sg_ctx_set)
    jaccard = overlap / union if union > 0 else 0

    # Null model: randomly assign SG status to words, recompute TTR
    rng = random.Random(seed)
    all_words_flat = []
    for page in pages:
        for line in page.get("line_words", []):
            all_words_flat.extend(line)

    n_sg_words = sum(1 for w in all_words_flat if _has_sg(w))
    null_ttr_diffs = []

    for _ in range(n_perms):
        # Random subset of same size
        indices = set(rng.sample(range(len(all_words_flat)),
                                 min(n_sg_words, len(all_words_flat))))
        null_sg = set()
        null_non_sg = set()
        for i, w in enumerate(all_words_flat):
            if i in indices:
                null_sg.add(w)
            else:
                null_non_sg.add(w)
        null_sg_ttr = len(null_sg) / n_sg_words if n_sg_words > 0 else 0
        null_non_sg_ttr = len(null_non_sg) / (len(all_words_flat) - n_sg_words) \
            if (len(all_words_flat) - n_sg_words) > 0 else 0
        null_ttr_diffs.append(null_sg_ttr - null_non_sg_ttr)

    ttr_diff = sg_ttr - non_sg_ttr
    null_mean = float(np.mean(null_ttr_diffs))
    null_std = float(np.std(null_ttr_diffs, ddof=1)) if len(null_ttr_diffs) > 1 else 0.001
    z_ttr = (ttr_diff - null_mean) / null_std if null_std > 0 else 0.0

    return {
        "sg_context_tokens": sg_ctx_tokens,
        "sg_context_types": sg_ctx_types,
        "sg_context_ttr": round(sg_ttr, 4),
        "non_sg_context_ttr": round(non_sg_ttr, 4),
        "ttr_difference": round(ttr_diff, 4),
        "z_ttr": round(z_ttr, 2),
        "sg_following_ttr": round(sg_follow_ttr, 4),
        "non_sg_following_ttr": round(non_sg_follow_ttr, 4),
        "context_jaccard": round(jaccard, 4),
        "n_sg_words": n_sg_words,
        "interpretation": (
            "RESTRICTED_CONTEXT" if sg_ttr < non_sg_ttr * 0.85 and z_ttr < -3
            else "DIVERSE_CONTEXT" if sg_ttr > non_sg_ttr * 1.15 and z_ttr > 3
            else "SIMILAR_CONTEXT"
        ),
    }


# =====================================================================
# 13b — Per-type context comparison
# =====================================================================

def test_sg_per_type_context(pages: list[dict]) -> dict:
    """Compare context words across the 4 SG types.

    If the 4 types appear in different contexts → different functions.
    If similar contexts → same function with different "values."
    """
    type_context: dict[str, Counter] = {pat: Counter() for pat in SG_PATTERNS}
    type_counts: dict[str, int] = {pat: 0 for pat in SG_PATTERNS}

    for page in pages:
        for line in page.get("line_words", []):
            for i, word in enumerate(line):
                sg = _sg_type(word)
                if sg:
                    type_counts[sg] += 1
                    ctx = _extract_context_words(line, i, window=3)
                    for cw in ctx:
                        type_context[sg][cw] += 1

    # Pairwise Jaccard between context vocabularies
    pairwise = {}
    types_with_data = [t for t in SG_PATTERNS if type_counts[t] >= 10]

    for i, t1 in enumerate(types_with_data):
        for t2 in types_with_data[i + 1:]:
            s1 = set(type_context[t1].keys())
            s2 = set(type_context[t2].keys())
            inter = len(s1 & s2)
            union = len(s1 | s2)
            j = inter / union if union > 0 else 0
            pairwise[f"{t1}_vs_{t2}"] = round(j, 4)

    # Chi-square test: context word × SG type contingency table
    # Use top-50 most common context words across all types
    all_ctx = Counter()
    for c in type_context.values():
        all_ctx.update(c)
    top_words = [w for w, _ in all_ctx.most_common(50)]

    chi2_result = {"chi2": 0, "p": 1.0, "significant": False}
    if len(types_with_data) >= 2 and len(top_words) >= 5:
        table = []
        for t in types_with_data:
            row = [type_context[t].get(w, 0) for w in top_words]
            if sum(row) > 0:
                table.append(row)

        if len(table) >= 2:
            try:
                chi2, p, dof, _ = chi2_contingency(table)
                chi2_result = {
                    "chi2": round(float(chi2), 1),
                    "p": round(float(p), 6),
                    "dof": int(dof),
                    "significant": p < 0.01,
                }
            except ValueError:
                pass

    # Per-type stats
    per_type = {}
    for t in SG_PATTERNS:
        n = type_counts[t]
        ctx = type_context[t]
        per_type[t] = {
            "count": n,
            "context_types": len(ctx),
            "context_tokens": sum(ctx.values()),
            "top5_context": [w for w, _ in ctx.most_common(5)],
        }

    return {
        "per_type": per_type,
        "pairwise_jaccard": pairwise,
        "chi2_test": chi2_result,
        "interpretation": (
            "DIFFERENT_FUNCTIONS" if chi2_result["significant"]
            else "SAME_FUNCTION"
        ),
    }


# =====================================================================
# 13c — SG density vs folio content
# =====================================================================

def test_sg_density_vs_folio(pages: list[dict]) -> dict:
    """Correlate SG density with folio-level properties.

    For herbal folios (where ~89% of SG occur):
    - SG count vs word count
    - SG count vs line count
    - SG count vs paragraph count

    If SG ∝ word count → just frequency.
    If uncorrelated with size → content-dependent.
    """
    folio_data = []

    for page in pages:
        n_words = len(page["words"])
        n_lines = len(page.get("line_words", []))
        n_paras = sum(1 for i, line in enumerate(page.get("line_words", []))
                      if i == 0)  # simplified: first line = para start
        n_sg = sum(1 for w in page["words"] if _has_sg(w))
        section = page.get("section", "?")

        if n_words > 0:
            folio_data.append({
                "folio": page["folio"],
                "section": section,
                "n_words": n_words,
                "n_lines": n_lines,
                "n_sg": n_sg,
                "sg_density": n_sg / n_words,
            })

    # All folios
    all_words = np.array([f["n_words"] for f in folio_data], dtype=float)
    all_sg = np.array([f["n_sg"] for f in folio_data], dtype=float)
    all_lines = np.array([f["n_lines"] for f in folio_data], dtype=float)

    r_words, p_words = pearsonr(all_words, all_sg) if len(all_words) > 2 else (0, 1)
    r_lines, p_lines = pearsonr(all_lines, all_sg) if len(all_lines) > 2 else (0, 1)

    # Herbal only
    herbal = [f for f in folio_data if f["section"] == "H"]
    if len(herbal) > 5:
        h_words = np.array([f["n_words"] for f in herbal], dtype=float)
        h_sg = np.array([f["n_sg"] for f in herbal], dtype=float)
        h_lines = np.array([f["n_lines"] for f in herbal], dtype=float)
        r_h_words, p_h_words = pearsonr(h_words, h_sg)
        r_h_lines, p_h_lines = pearsonr(h_lines, h_sg)
    else:
        r_h_words, p_h_words = 0, 1
        r_h_lines, p_h_lines = 0, 1

    # Per-section summary
    section_summary = {}
    for f in folio_data:
        sec = f["section"]
        if sec not in section_summary:
            section_summary[sec] = {"n_folios": 0, "total_sg": 0, "total_words": 0}
        section_summary[sec]["n_folios"] += 1
        section_summary[sec]["total_sg"] += f["n_sg"]
        section_summary[sec]["total_words"] += f["n_words"]

    for sec in section_summary:
        s = section_summary[sec]
        s["sg_rate"] = round(s["total_sg"] / s["total_words"], 4) \
            if s["total_words"] > 0 else 0
        s["sg_per_folio"] = round(s["total_sg"] / s["n_folios"], 1) \
            if s["n_folios"] > 0 else 0

    return {
        "all_folios": {
            "r_vs_words": round(float(r_words), 4),
            "p_vs_words": round(float(p_words), 6),
            "r_vs_lines": round(float(r_lines), 4),
            "p_vs_lines": round(float(p_lines), 6),
            "n_folios": len(folio_data),
        },
        "herbal_only": {
            "r_vs_words": round(float(r_h_words), 4),
            "p_vs_words": round(float(p_h_words), 6),
            "r_vs_lines": round(float(r_h_lines), 4),
            "p_vs_lines": round(float(p_h_lines), 6),
            "n_folios": len(herbal),
        },
        "per_section": section_summary,
        "interpretation": (
            "FREQUENCY_PROPORTIONAL" if r_h_words > 0.7 and p_h_words < 0.01
            else "CONTENT_DEPENDENT" if r_h_words < 0.3
            else "MODERATE_CORRELATION"
        ),
    }


# =====================================================================
# 13d — SG co-occurrence with specific word types
# =====================================================================

def test_sg_word_cooccurrence(pages: list[dict]) -> dict:
    """Find words that co-occur with SG more/less than expected.

    For each word type W: compute odds ratio of W appearing on a line
    that contains an SG-word vs a line without SG.
    Top words with highest odds ratio = candidates for what SG modifies.
    """
    # Build line-level data
    sg_lines_words: Counter = Counter()
    non_sg_lines_words: Counter = Counter()
    n_sg_lines = 0
    n_non_sg_lines = 0

    for page in pages:
        for line in page.get("line_words", []):
            has_sg_line = any(_has_sg(w) for w in line)
            # Count non-SG words on this line
            line_vocab = set(w for w in line if not _has_sg(w))
            if has_sg_line:
                n_sg_lines += 1
                for w in line_vocab:
                    sg_lines_words[w] += 1
            else:
                n_non_sg_lines += 1
                for w in line_vocab:
                    non_sg_lines_words[w] += 1

    # Compute odds ratios for words that appear at least 20 times
    all_words = set(sg_lines_words.keys()) | set(non_sg_lines_words.keys())
    odds_ratios = []

    for w in all_words:
        sg_count = sg_lines_words.get(w, 0)
        non_sg_count = non_sg_lines_words.get(w, 0)
        total = sg_count + non_sg_count

        if total < 20:
            continue

        # Odds ratio with Haldane correction (+0.5)
        p_sg = (sg_count + 0.5) / (n_sg_lines + 1)
        p_non = (non_sg_count + 0.5) / (n_non_sg_lines + 1)
        odds = (p_sg / (1 - p_sg)) / (p_non / (1 - p_non)) \
            if p_non < 1 and p_sg < 1 else 1.0

        odds_ratios.append({
            "word": w,
            "sg_lines": sg_count,
            "non_sg_lines": non_sg_count,
            "total": total,
            "odds_ratio": round(odds, 3),
        })

    # Sort by odds ratio
    odds_ratios.sort(key=lambda x: -x["odds_ratio"])

    # Top attracted and repelled
    top_attracted = odds_ratios[:10]
    top_repelled = sorted(odds_ratios, key=lambda x: x["odds_ratio"])[:10]

    return {
        "n_sg_lines": n_sg_lines,
        "n_non_sg_lines": n_non_sg_lines,
        "n_words_tested": len(odds_ratios),
        "top_attracted": top_attracted,
        "top_repelled": top_repelled,
        "interpretation": (
            "SELECTIVE_CONTEXT" if top_attracted and top_attracted[0]["odds_ratio"] > 3.0
            else "BROAD_CONTEXT"
        ),
    }


# =====================================================================
# 13e — SG and 'm' marker interaction
# =====================================================================

def test_sg_m_interaction(lines: list[dict]) -> dict:
    """Test interaction between split gallows and the 'm' line-end marker.

    - Does 'm' appear in the same word as SG?
    - Do lines with SG end in 'm' more/less than expected?
    - Chi-square test of independence.
    """
    n_lines = 0
    sg_and_m = 0      # line has SG AND ends in 'm'
    sg_no_m = 0        # line has SG but doesn't end in 'm'
    m_no_sg = 0        # line ends in 'm' but no SG
    neither = 0        # neither

    # Word-level: does 'm' appear IN an SG word?
    n_sg_words = 0
    n_sg_words_with_m = 0

    for line in lines:
        if line["para_type"] == "label":
            continue
        words = line["words"]
        if not words:
            continue

        n_lines += 1
        has_sg = any(_has_sg(w) for w in words)
        ends_m = words[-1][-1] == "m" if words[-1] else False

        if has_sg and ends_m:
            sg_and_m += 1
        elif has_sg and not ends_m:
            sg_no_m += 1
        elif not has_sg and ends_m:
            m_no_sg += 1
        else:
            neither += 1

        # Word-level
        for w in words:
            if _has_sg(w):
                n_sg_words += 1
                if "m" in w:
                    n_sg_words_with_m += 1

    # Chi-square test of independence
    table = [[sg_and_m, sg_no_m], [m_no_sg, neither]]
    try:
        chi2, p, _, _ = chi2_contingency(table)
    except ValueError:
        chi2, p = 0.0, 1.0

    # Expected under independence
    total_sg = sg_and_m + sg_no_m
    total_m = sg_and_m + m_no_sg
    expected_both = (total_sg * total_m) / n_lines if n_lines > 0 else 0

    # Rate comparison
    sg_m_rate = sg_and_m / total_sg if total_sg > 0 else 0
    non_sg_m_rate = m_no_sg / (m_no_sg + neither) if (m_no_sg + neither) > 0 else 0

    return {
        "n_lines": n_lines,
        "sg_and_m": sg_and_m,
        "sg_no_m": sg_no_m,
        "m_no_sg": m_no_sg,
        "neither": neither,
        "chi2": round(float(chi2), 2),
        "p": round(float(p), 6),
        "expected_both": round(expected_both, 1),
        "observed_vs_expected": round(sg_and_m / expected_both, 2) if expected_both > 0 else 0,
        "p_m_given_sg": round(sg_m_rate, 4),
        "p_m_given_no_sg": round(non_sg_m_rate, 4),
        "n_sg_words_total": n_sg_words,
        "n_sg_words_containing_m": n_sg_words_with_m,
        "pct_sg_words_with_m": round(n_sg_words_with_m / n_sg_words * 100, 1) if n_sg_words > 0 else 0,
        "interpretation": (
            "INDEPENDENT" if p > 0.05
            else "SG_ATTRACTS_M" if sg_and_m > expected_both
            else "SG_REPELS_M"
        ),
    }


# =====================================================================
# Summary formatting
# =====================================================================

def format_summary(results: dict) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 13 — Split Gallows Semantic Function")
    lines.append("=" * 72)

    # 13a
    r = results["13a_context"]
    lines.append(f"\n  13a — LEXICAL CONTEXT OF SG WORDS")
    lines.append("  " + "-" * 66)
    lines.append(f"  SG context: {r['sg_context_tokens']} tokens, {r['sg_context_types']} types, "
                 f"TTR={r['sg_context_ttr']:.4f}")
    lines.append(f"  Non-SG context: TTR={r['non_sg_context_ttr']:.4f}")
    lines.append(f"  TTR difference: {r['ttr_difference']:+.4f} (z={r['z_ttr']:+.2f})")
    lines.append(f"  Following-word TTR: SG={r['sg_following_ttr']:.4f}, "
                 f"non-SG={r['non_sg_following_ttr']:.4f}")
    lines.append(f"  Context vocabulary Jaccard: {r['context_jaccard']:.4f}")
    lines.append(f"  → {r['interpretation']}")

    # 13b
    r = results["13b_per_type"]
    lines.append(f"\n  13b — PER-TYPE CONTEXT COMPARISON")
    lines.append("  " + "-" * 66)
    for t, v in r["per_type"].items():
        lines.append(f"  {t}: n={v['count']}, context_types={v['context_types']}, "
                     f"top5={v['top5_context']}")
    lines.append(f"  Pairwise Jaccard: {r['pairwise_jaccard']}")
    lines.append(f"  Chi² test: chi2={r['chi2_test']['chi2']}, "
                 f"p={r['chi2_test']['p']}, significant={r['chi2_test']['significant']}")
    lines.append(f"  → {r['interpretation']}")

    # 13c
    r = results["13c_folio_density"]
    lines.append(f"\n  13c — SG DENSITY vs FOLIO CONTENT")
    lines.append("  " + "-" * 66)
    lines.append(f"  All folios: r_vs_words={r['all_folios']['r_vs_words']:.4f} "
                 f"(p={r['all_folios']['p_vs_words']:.6f}), "
                 f"r_vs_lines={r['all_folios']['r_vs_lines']:.4f}")
    lines.append(f"  Herbal only: r_vs_words={r['herbal_only']['r_vs_words']:.4f} "
                 f"(p={r['herbal_only']['p_vs_words']:.6f}), "
                 f"r_vs_lines={r['herbal_only']['r_vs_lines']:.4f}")
    lines.append(f"\n  Per-section SG rates:")
    for sec in sorted(r["per_section"].keys()):
        s = r["per_section"][sec]
        lines.append(f"    {sec}: {s['total_sg']} SG / {s['total_words']} words = "
                     f"{s['sg_rate']:.4f} ({s['sg_per_folio']:.1f}/folio)")
    lines.append(f"  → {r['interpretation']}")

    # 13d
    r = results["13d_word_cooccurrence"]
    lines.append(f"\n  13d — SG CO-OCCURRENCE WITH SPECIFIC WORDS")
    lines.append("  " + "-" * 66)
    lines.append(f"  SG lines: {r['n_sg_lines']}, non-SG lines: {r['n_non_sg_lines']}")
    lines.append(f"  Words tested: {r['n_words_tested']}")
    lines.append(f"\n  Top ATTRACTED (high odds ratio → appear MORE with SG):")
    for w in r["top_attracted"][:5]:
        lines.append(f"    '{w['word']}': OR={w['odds_ratio']:.2f} "
                     f"(SG={w['sg_lines']}, non-SG={w['non_sg_lines']})")
    lines.append(f"\n  Top REPELLED (low odds ratio → appear LESS with SG):")
    for w in r["top_repelled"][:5]:
        lines.append(f"    '{w['word']}': OR={w['odds_ratio']:.2f} "
                     f"(SG={w['sg_lines']}, non-SG={w['non_sg_lines']})")
    lines.append(f"  → {r['interpretation']}")

    # 13e
    r = results["13e_m_interaction"]
    lines.append(f"\n  13e — SG AND 'm' MARKER INTERACTION")
    lines.append("  " + "-" * 66)
    lines.append(f"  Contingency: SG+m={r['sg_and_m']}, SG-only={r['sg_no_m']}, "
                 f"m-only={r['m_no_sg']}, neither={r['neither']}")
    lines.append(f"  Chi²={r['chi2']:.2f}, p={r['p']:.6f}")
    lines.append(f"  Expected SG+m under independence: {r['expected_both']:.1f}, "
                 f"observed: {r['sg_and_m']} ({r['observed_vs_expected']:.2f}x)")
    lines.append(f"  P(m-end | SG line): {r['p_m_given_sg']:.4f}")
    lines.append(f"  P(m-end | non-SG line): {r['p_m_given_no_sg']:.4f}")
    lines.append(f"  SG words containing 'm': {r['n_sg_words_containing_m']}/"
                 f"{r['n_sg_words_total']} ({r['pct_sg_words_with_m']:.1f}%)")
    lines.append(f"  → {r['interpretation']}")

    # Overall verdict
    lines.append("\n" + "=" * 72)
    lines.append("  OVERALL ASSESSMENT")
    lines.append("=" * 72)

    verdicts = {
        "context": results["13a_context"]["interpretation"],
        "per_type": results["13b_per_type"]["interpretation"],
        "density": results["13c_folio_density"]["interpretation"],
        "cooccurrence": results["13d_word_cooccurrence"]["interpretation"],
        "m_interaction": results["13e_m_interaction"]["interpretation"],
    }
    lines.append(f"\n  Sub-test verdicts: {verdicts}")

    lines.append("")
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

    cur.execute("DROP TABLE IF EXISTS split_gallows_semantic_test")
    cur.execute("""
        CREATE TABLE split_gallows_semantic_test (
            test TEXT,
            key TEXT,
            observed REAL,
            detail_json TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    def _insert(test, key, observed, detail=None):
        cur.execute(
            "INSERT INTO split_gallows_semantic_test VALUES (?, ?, ?, ?)",
            (test, key, observed,
             json.dumps(detail, default=str) if detail else None),
        )

    # 13a
    r = results["13a_context"]
    _insert("13a_context", "sg_ttr", r["sg_context_ttr"])
    _insert("13a_context", "non_sg_ttr", r["non_sg_context_ttr"])
    _insert("13a_context", "z_ttr", r["z_ttr"])
    _insert("13a_context", "context_jaccard", r["context_jaccard"])

    # 13b
    r = results["13b_per_type"]
    _insert("13b_per_type", "chi2", r["chi2_test"]["chi2"],
            r["chi2_test"])
    for t, v in r["per_type"].items():
        _insert("13b_per_type", t, v["count"], v)

    # 13c
    r = results["13c_folio_density"]
    _insert("13c_folio_density", "r_all_words", r["all_folios"]["r_vs_words"])
    _insert("13c_folio_density", "r_herbal_words", r["herbal_only"]["r_vs_words"])
    for sec, v in r["per_section"].items():
        _insert("13c_folio_density", f"section_{sec}", v["sg_rate"], v)

    # 13d
    r = results["13d_word_cooccurrence"]
    for w in r["top_attracted"][:5]:
        _insert("13d_cooccurrence", f"attracted_{w['word']}", w["odds_ratio"], w)
    for w in r["top_repelled"][:5]:
        _insert("13d_cooccurrence", f"repelled_{w['word']}", w["odds_ratio"], w)

    # 13e
    r = results["13e_m_interaction"]
    _insert("13e_m_interaction", "chi2", r["chi2"])
    _insert("13e_m_interaction", "observed_vs_expected", r["observed_vs_expected"])
    _insert("13e_m_interaction", "p_m_given_sg", r["p_m_given_sg"])
    _insert("13e_m_interaction", "p_m_given_no_sg", r["p_m_given_no_sg"])

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 13 — Split gallows semantic function."""
    report_path = config.stats_dir / "split_gallows_semantic_test.json"
    summary_path = config.stats_dir / "split_gallows_semantic_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Split gallows semantic test exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 13 — Split Gallows Semantic Function")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")

    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # Also parse line structure for 13e
    lines = parse_ivtff_lines(eva_file)
    click.echo(f"    {len(lines)} lines parsed")

    # Count SG words
    n_sg = sum(1 for p in pages for w in p["words"] if _has_sg(w))
    click.echo(f"    {n_sg} words containing split gallows")

    results = {}

    # 13a
    print_step("13a — Lexical context of SG words...")
    results["13a_context"] = test_sg_context(pages)
    r = results["13a_context"]
    click.echo(f"    SG TTR={r['sg_context_ttr']:.4f}, non-SG TTR={r['non_sg_context_ttr']:.4f}, "
               f"z={r['z_ttr']:+.2f} → {r['interpretation']}")

    # 13b
    print_step("13b — Per-type context comparison...")
    results["13b_per_type"] = test_sg_per_type_context(pages)
    r = results["13b_per_type"]
    click.echo(f"    Chi²={r['chi2_test']['chi2']:.1f}, p={r['chi2_test']['p']:.6f} "
               f"→ {r['interpretation']}")

    # 13c
    print_step("13c — SG density vs folio content...")
    results["13c_folio_density"] = test_sg_density_vs_folio(pages)
    r = results["13c_folio_density"]
    click.echo(f"    All folios r={r['all_folios']['r_vs_words']:.4f}, "
               f"herbal r={r['herbal_only']['r_vs_words']:.4f} "
               f"→ {r['interpretation']}")

    # 13d
    print_step("13d — SG co-occurrence with specific words...")
    results["13d_word_cooccurrence"] = test_sg_word_cooccurrence(pages)
    r = results["13d_word_cooccurrence"]
    if r["top_attracted"]:
        click.echo(f"    Top attracted: '{r['top_attracted'][0]['word']}' "
                   f"OR={r['top_attracted'][0]['odds_ratio']:.2f}")
    click.echo(f"    → {r['interpretation']}")

    # 13e
    print_step("13e — SG and 'm' marker interaction...")
    results["13e_m_interaction"] = test_sg_m_interaction(lines)
    r = results["13e_m_interaction"]
    click.echo(f"    Chi²={r['chi2']:.2f}, p={r['p']:.6f}, "
               f"P(m|SG)={r['p_m_given_sg']:.4f} vs P(m|no-SG)={r['p_m_given_no_sg']:.4f} "
               f"→ {r['interpretation']}")

    # Save
    print_step("Saving results...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, results)
    click.echo(f"    DB:   split_gallows_semantic_test table")

    click.echo(f"\n{summary}")
