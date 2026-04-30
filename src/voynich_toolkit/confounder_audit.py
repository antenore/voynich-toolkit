"""
Phase 15 — Confounder Audit.

Retests every high/medium-risk finding with proper controls.
For each: test the confounder, measure the impact, revise the finding.

Sub-tests:
  15a — Section vocabulary MI excluding hapax legomena
  15b — Hand bigrams within-section (section confounding test)
  15c — SG type contexts within-section (CRITICAL — does chi²=334.7 survive?)
  15d — Gallows at paragraph start: base rate correction + effect size
  15e — 'm' section-rate investigation (why pharma=49%, balneo=94%?)

Principle: a high z-score is NOT automatically a strong finding.
The exceptions must be examined and explained.
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
from scipy.stats import chi2_contingency

from .config import ToolkitConfig
from .currier_line_test import parse_ivtff_lines
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_PERM = 500

SIMPLE_GALLOWS = set("tkpf")
SG_PATTERNS = ("cth", "ckh", "cph", "cfh")


def _has_sg(word: str) -> bool:
    return any(pat in word for pat in SG_PATTERNS)


def _sg_type(word: str) -> str | None:
    for pat in SG_PATTERNS:
        if pat in word:
            return pat
    return None


# =====================================================================
# 15a — Section vocabulary MI excluding hapax
# =====================================================================

def test_mi_without_hapax(pages: list[dict], n_perms: int = N_PERM,
                          seed: int = SEED) -> dict:
    """Recompute MI(word; section) at different frequency thresholds.

    Tests whether z=+40 is inflated by hapax legomena (words appearing
    only once, which are trivially section-specific).
    """
    # Build word counts
    word_total: Counter = Counter()
    section_words: dict[str, Counter] = defaultdict(Counter)
    for page in pages:
        sec = page["section"]
        for w in page["words"]:
            word_total[w] += 1
            section_words[sec][w] += 1

    thresholds = [1, 2, 5, 10, 20]
    results_by_threshold = {}
    rng = random.Random(seed)

    for min_freq in thresholds:
        # Filter words
        vocab = {w for w, c in word_total.items() if c >= min_freq}
        n_excluded = len(word_total) - len(vocab)

        # Compute MI on filtered vocabulary
        total = sum(c for sec_c in section_words.values()
                    for w, c in sec_c.items() if w in vocab)
        if total == 0:
            results_by_threshold[min_freq] = {
                "mi_bits": 0, "vocab_size": 0, "excluded": n_excluded,
            }
            continue

        wc: Counter = Counter()
        sc: Counter = Counter()
        joint: Counter = Counter()
        for sec, counts in section_words.items():
            for w, c in counts.items():
                if w not in vocab:
                    continue
                wc[w] += c
                sc[sec] += c
                joint[(w, sec)] += c

        mi = 0.0
        for (w, s), n_ws in joint.items():
            p_ws = n_ws / total
            p_w = wc[w] / total
            p_s = sc[s] / total
            if p_ws > 0 and p_w > 0 and p_s > 0:
                mi += p_ws * math.log2(p_ws / (p_w * p_s))

        # Null model: shuffle section assignments
        nulls = []
        page_words_list = []
        page_sections = []
        for page in pages:
            filtered = [w for w in page["words"] if w in vocab]
            if filtered:
                page_words_list.append(filtered)
                page_sections.append(page["section"])

        for _ in range(n_perms):
            shuffled_sections = list(page_sections)
            rng.shuffle(shuffled_sections)
            null_sc: Counter = Counter()
            null_joint: Counter = Counter()
            null_wc: Counter = Counter()
            for words, sec in zip(page_words_list, shuffled_sections):
                for w in words:
                    null_wc[w] += 1
                    null_sc[sec] += 1
                    null_joint[(w, sec)] += 1
            null_total = sum(null_wc.values())
            null_mi = 0.0
            for (w, s), n_ws in null_joint.items():
                p_ws = n_ws / null_total
                p_w = null_wc[w] / null_total
                p_s = null_sc[s] / null_total
                if p_ws > 0 and p_w > 0 and p_s > 0:
                    null_mi += p_ws * math.log2(p_ws / (p_w * p_s))
            nulls.append(null_mi)

        null_mean = float(np.mean(nulls))
        null_std = float(np.std(nulls, ddof=1)) if len(nulls) > 1 else 0.001
        z = (mi - null_mean) / null_std if null_std > 0 else 0.0

        results_by_threshold[min_freq] = {
            "mi_bits": round(mi, 4),
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "z_score": round(z, 2),
            "vocab_size": len(vocab),
            "excluded": n_excluded,
            "total_tokens": total,
        }

    # Verdict
    z_full = results_by_threshold[1]["z_score"]
    z_min5 = results_by_threshold[5]["z_score"]
    z_min10 = results_by_threshold[10]["z_score"]
    drop_pct = (1 - z_min5 / z_full) * 100 if z_full > 0 else 0

    return {
        "by_threshold": results_by_threshold,
        "z_drop_at_freq5": round(drop_pct, 1),
        "verdict": (
            "SURVIVES" if z_min10 > 10
            else "WEAKENED" if z_min10 > 3
            else "COLLAPSES"
        ),
    }


# =====================================================================
# 15b — Hand bigrams within-section
# =====================================================================

def test_hand_bigrams_within_section(pages: list[dict]) -> dict:
    """Per-section hand bigram chi² — section confounding test.

    For each section with ≥2 hands and ≥200 tokens per hand:
    compute bigram chi² between hands.
    If still significant → genuine scribe difference.
    If not → hand bigrams were section confounding.
    """
    # Group words by (section, hand)
    sec_hand_words: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for page in pages:
        sec = page["section"]
        hand = page.get("hand", "?")
        sec_hand_words[sec][hand].extend(page["words"])

    results_by_section = {}

    for sec in sorted(sec_hand_words.keys()):
        hands = sec_hand_words[sec]
        # Filter hands with enough data
        eligible = {h: w for h, w in hands.items() if len(w) >= 200}
        if len(eligible) < 2:
            continue

        # Build bigram frequency per hand
        hand_bigrams: dict[str, Counter] = {}
        all_bigrams: set = set()
        for hand, words in eligible.items():
            bg = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    pair = word[i:i + 2]
                    bg[pair] += 1
                    all_bigrams.add(pair)
            hand_bigrams[hand] = bg

        # Top 50 bigrams
        top = sorted(all_bigrams, key=lambda b: sum(
            hand_bigrams[h].get(b, 0) for h in eligible
        ), reverse=True)[:50]

        if len(top) < 5:
            continue

        # Contingency table
        table = []
        hand_labels = []
        for hand in sorted(eligible.keys()):
            row = [hand_bigrams[hand].get(b, 0) for b in top]
            if sum(row) > 0:
                table.append(row)
                hand_labels.append(hand)

        if len(table) < 2:
            continue

        try:
            chi2, p, dof, _ = chi2_contingency(table)
            significant = p < 0.01
        except ValueError:
            chi2, p, dof = 0, 1, 0
            significant = False

        results_by_section[sec] = {
            "hands": hand_labels,
            "n_hands": len(hand_labels),
            "tokens_per_hand": {h: len(eligible[h]) for h in hand_labels},
            "chi2": round(float(chi2), 1),
            "p": round(float(p), 6),
            "dof": int(dof),
            "significant": significant,
        }

    n_tested = len(results_by_section)
    n_significant = sum(1 for r in results_by_section.values() if r["significant"])

    return {
        "by_section": results_by_section,
        "n_sections_tested": n_tested,
        "n_significant": n_significant,
        "verdict": (
            "SURVIVES" if n_significant >= n_tested * 0.7
            else "WEAKENED" if n_significant >= 2
            else "COLLAPSES"
        ),
    }


# =====================================================================
# 15c — SG type contexts within-section (CRITICAL)
# =====================================================================

def test_sg_context_within_section(pages: list[dict]) -> dict:
    """Repeat SG type × context chi² ONLY within herbal section.

    If chi² still significant → types genuinely have different functions.
    If collapses → "different functions" was just section confounding.
    """
    results = {}

    for target_section in ["H", "all"]:
        if target_section == "all":
            section_pages = pages
        else:
            section_pages = [p for p in pages if p.get("section") == target_section]

        type_context: dict[str, Counter] = {pat: Counter() for pat in SG_PATTERNS}
        type_counts: dict[str, int] = {pat: 0 for pat in SG_PATTERNS}

        for page in section_pages:
            for line in page.get("line_words", []):
                for i, word in enumerate(line):
                    sg = _sg_type(word)
                    if sg:
                        type_counts[sg] += 1
                        # Context ±3
                        for j in range(max(0, i - 3), min(len(line), i + 4)):
                            if j != i:
                                type_context[sg][line[j]] += 1

        types_with_data = [t for t in SG_PATTERNS if type_counts[t] >= 5]

        if len(types_with_data) < 2:
            results[target_section] = {
                "chi2": 0, "p": 1, "significant": False,
                "n_types": len(types_with_data),
                "type_counts": type_counts,
                "note": "insufficient data",
            }
            continue

        # Chi² on context word × SG type
        all_ctx = Counter()
        for c in type_context.values():
            all_ctx.update(c)
        top_words = [w for w, _ in all_ctx.most_common(50)]

        table = []
        type_labels = []
        for t in types_with_data:
            row = [type_context[t].get(w, 0) for w in top_words]
            if sum(row) > 0:
                table.append(row)
                type_labels.append(t)

        if len(table) < 2:
            results[target_section] = {
                "chi2": 0, "p": 1, "significant": False,
                "n_types": len(type_labels),
                "type_counts": type_counts,
            }
            continue

        try:
            chi2, p, dof, _ = chi2_contingency(table)
        except ValueError:
            chi2, p, dof = 0, 1, 0

        # Pairwise Jaccard
        pairwise = {}
        for i, t1 in enumerate(types_with_data):
            for t2 in types_with_data[i + 1:]:
                s1 = set(type_context[t1].keys())
                s2 = set(type_context[t2].keys())
                inter = len(s1 & s2)
                union = len(s1 | s2)
                pairwise[f"{t1}_vs_{t2}"] = round(inter / union, 4) if union > 0 else 0

        results[target_section] = {
            "chi2": round(float(chi2), 1),
            "p": round(float(p), 6),
            "dof": int(dof),
            "significant": p < 0.01,
            "n_types": len(type_labels),
            "type_counts": {t: type_counts[t] for t in types_with_data},
            "pairwise_jaccard": pairwise,
        }

    # Compare all vs herbal-only
    chi2_all = results.get("all", {}).get("chi2", 0)
    chi2_herbal = results.get("H", {}).get("chi2", 0)
    drop_pct = (1 - chi2_herbal / chi2_all) * 100 if chi2_all > 0 else 0

    return {
        "by_scope": results,
        "chi2_all": chi2_all,
        "chi2_herbal_only": chi2_herbal,
        "chi2_drop_pct": round(drop_pct, 1),
        "herbal_significant": results.get("H", {}).get("significant", False),
        "verdict": (
            "SURVIVES" if results.get("H", {}).get("significant", False)
            else "COLLAPSES"
        ),
    }


# =====================================================================
# 15d — Gallows base rate correction
# =====================================================================

def test_gallows_base_rate(lines: list[dict]) -> dict:
    """Gallows at paragraph start: effect size with base rate correction.

    Reports: absolute difference, relative risk, Cohen's h, and the
    base rate that makes the raw z-score misleading.
    """
    para_start_total = 0
    para_start_gallows = 0
    cont_total = 0
    cont_gallows = 0

    for line in lines:
        if line["para_type"] == "label":
            continue
        words = line["words"]
        if not words:
            continue

        first_word = words[0]
        has_gallows = any(c in SIMPLE_GALLOWS for c in first_word)

        if line["para_type"] == "para_start":
            para_start_total += 1
            if has_gallows:
                para_start_gallows += 1
        elif line["para_type"] in ("para_cont", "para_end"):
            cont_total += 1
            if has_gallows:
                cont_gallows += 1

    p_start = para_start_gallows / para_start_total if para_start_total > 0 else 0
    p_cont = cont_gallows / cont_total if cont_total > 0 else 0

    # Also compute: gallows in ANY word on the line (not just first)
    para_start_any_gallows = 0
    cont_any_gallows = 0
    for line in lines:
        if line["para_type"] == "label":
            continue
        words = line["words"]
        if not words:
            continue
        has_any = any(any(c in SIMPLE_GALLOWS for c in w) for w in words)
        if line["para_type"] == "para_start":
            if has_any:
                para_start_any_gallows += 1
        elif line["para_type"] in ("para_cont", "para_end"):
            if has_any:
                cont_any_gallows += 1

    p_start_any = para_start_any_gallows / para_start_total if para_start_total > 0 else 0
    p_cont_any = cont_any_gallows / cont_total if cont_total > 0 else 0

    # Effect sizes
    abs_diff = p_start - p_cont
    relative_risk = p_start / p_cont if p_cont > 0 else float("inf")

    # Cohen's h
    h1 = 2 * math.asin(math.sqrt(p_start))
    h2 = 2 * math.asin(math.sqrt(p_cont))
    cohens_h = h1 - h2

    # Base rate: overall gallows rate in first words
    overall = (para_start_gallows + cont_gallows) / (para_start_total + cont_total) \
        if (para_start_total + cont_total) > 0 else 0

    return {
        "para_start_total": para_start_total,
        "para_start_gallows_first_word": para_start_gallows,
        "p_start_first_word": round(p_start, 4),
        "cont_total": cont_total,
        "cont_gallows_first_word": cont_gallows,
        "p_cont_first_word": round(p_cont, 4),
        "absolute_difference": round(abs_diff, 4),
        "relative_risk": round(relative_risk, 3),
        "cohens_h": round(cohens_h, 4),
        "base_rate_first_word": round(overall, 4),
        "p_start_any_word": round(p_start_any, 4),
        "p_cont_any_word": round(p_cont_any, 4),
        "interpretation": {
            "cohens_h_meaning": (
                "LARGE" if abs(cohens_h) > 0.8
                else "MEDIUM" if abs(cohens_h) > 0.5
                else "SMALL" if abs(cohens_h) > 0.2
                else "NEGLIGIBLE"
            ),
            "base_rate_problem": (
                f"Base rate is {overall:.1%} — gallows are common in first words "
                f"everywhere. The +{abs_diff:.1%} difference at para-start is real "
                f"but the absolute numbers show gallows are NOT exclusive paragraph markers."
            ),
        },
        "verdict": (
            "SURVIVES_WEAK" if cohens_h > 0.2
            else "NEGLIGIBLE_EFFECT"
        ),
    }


# =====================================================================
# 15e — 'm' section-rate investigation
# =====================================================================

def test_m_section_detail(pages: list[dict]) -> dict:
    """Investigate WHY 'm' line-final rate varies by section.

    For sections with extreme rates (pharma=49%, balneo=94%):
    - What words contain 'm'? Are they different words?
    - What fraction of 'm' is medial (inside words) vs word-final?
    - Is it the same 'm' used differently, or different words?
    """
    section_m_words: dict[str, Counter] = defaultdict(Counter)
    section_m_position: dict[str, Counter] = defaultdict(Counter)
    section_line_info: dict[str, list] = defaultdict(list)

    for page in pages:
        sec = page["section"]
        for line in page.get("line_words", []):
            if not line:
                continue
            last_word = line[-1]
            ends_m = last_word[-1] == "m" if last_word else False
            section_line_info[sec].append({"ends_m": ends_m, "n_words": len(line)})

            for wi, word in enumerate(line):
                is_last = (wi == len(line) - 1)
                for ci, ch in enumerate(word):
                    if ch != "m":
                        continue
                    is_wf = (ci == len(word) - 1)
                    if is_wf and is_last:
                        section_m_position[sec]["word_final_line_final"] += 1
                    elif is_wf:
                        section_m_position[sec]["word_final_not_line_final"] += 1
                        section_m_words[sec][word] += 1
                    else:
                        section_m_position[sec]["medial"] += 1
                        section_m_words[sec][word] += 1

    per_section = {}
    for sec in sorted(section_m_position.keys()):
        pos = section_m_position[sec]
        total = sum(pos.values())
        lf = pos.get("word_final_line_final", 0)
        wf = pos.get("word_final_not_line_final", 0)
        med = pos.get("medial", 0)

        # Top non-line-final words
        top_words = section_m_words[sec].most_common(10)

        # Line stats
        lines_info = section_line_info[sec]
        n_lines = len(lines_info)
        n_m_lines = sum(1 for l in lines_info if l["ends_m"])
        avg_len_m = float(np.mean([l["n_words"] for l in lines_info if l["ends_m"]])) \
            if n_m_lines > 0 else 0
        avg_len_no_m = float(np.mean([l["n_words"] for l in lines_info if not l["ends_m"]])) \
            if n_lines - n_m_lines > 0 else 0

        per_section[sec] = {
            "total_m": total,
            "line_final": lf,
            "word_final_not_line": wf,
            "medial": med,
            "pct_line_final": round(lf / total * 100, 1) if total > 0 else 0,
            "pct_medial": round(med / total * 100, 1) if total > 0 else 0,
            "top_non_final_words": [(w, c) for w, c in top_words],
            "n_lines": n_lines,
            "pct_lines_ending_m": round(n_m_lines / n_lines * 100, 1) if n_lines > 0 else 0,
            "avg_line_length_m": round(avg_len_m, 1),
            "avg_line_length_no_m": round(avg_len_no_m, 1),
        }

    # Compare pharma vs balneo specifically
    p = per_section.get("P", {})
    b = per_section.get("B", {})

    return {
        "per_section": per_section,
        "pharma_vs_balneo": {
            "pharma_pct_lf": p.get("pct_line_final", 0),
            "balneo_pct_lf": b.get("pct_line_final", 0),
            "pharma_pct_medial": p.get("pct_medial", 0),
            "balneo_pct_medial": b.get("pct_medial", 0),
            "pharma_top_non_final": p.get("top_non_final_words", [])[:5],
            "balneo_top_non_final": b.get("top_non_final_words", [])[:5],
        },
        "verdict": "SECTION_DEPENDENT",
    }


# =====================================================================
# Summary formatting
# =====================================================================

def format_summary(results: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 15 — CONFOUNDER AUDIT")
    lines.append("Testing what survives when we control for confounders")
    lines.append("=" * 72)

    # 15a
    r = results["15a_mi_hapax"]
    lines.append(f"\n  15a — SECTION VOCAB MI WITHOUT HAPAX")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Min freq':<10} {'Vocab':>8} {'Excluded':>10} {'MI bits':>10} "
                 f"{'z-score':>10} {'Tokens':>10}")
    for thresh, v in sorted(r["by_threshold"].items()):
        lines.append(f"  {thresh:>8}  {v['vocab_size']:>8} {v['excluded']:>10} "
                     f"{v['mi_bits']:>10.4f} {v['z_score']:>+10.2f} {v.get('total_tokens', 0):>10}")
    lines.append(f"  z-drop at freq≥5: {r['z_drop_at_freq5']:.1f}%")
    lines.append(f"  → {r['verdict']}")

    # 15b
    r = results["15b_hand_bigrams"]
    lines.append(f"\n  15b — HAND BIGRAMS WITHIN-SECTION")
    lines.append("  " + "-" * 66)
    for sec, v in sorted(r["by_section"].items()):
        sig = "***" if v["significant"] else "ns"
        lines.append(f"  Section {sec}: hands={v['hands']}, chi²={v['chi2']:.1f}, "
                     f"p={v['p']:.6f} [{sig}]")
        tokens = v.get("tokens_per_hand", {})
        lines.append(f"    tokens: {tokens}")
    lines.append(f"  {r['n_significant']}/{r['n_sections_tested']} sections significant")
    lines.append(f"  → {r['verdict']}")

    # 15c
    r = results["15c_sg_context"]
    lines.append(f"\n  15c — SG TYPE CONTEXTS WITHIN-SECTION (CRITICAL)")
    lines.append("  " + "-" * 66)
    for scope, v in r["by_scope"].items():
        lines.append(f"  Scope={scope}: chi²={v['chi2']:.1f}, p={v['p']:.6f}, "
                     f"significant={v['significant']}")
        if v.get("type_counts"):
            lines.append(f"    Type counts: {v['type_counts']}")
        if v.get("pairwise_jaccard"):
            lines.append(f"    Pairwise Jaccard: {v['pairwise_jaccard']}")
    lines.append(f"  chi² drop: all={r['chi2_all']:.1f} → herbal={r['chi2_herbal_only']:.1f} "
                 f"({r['chi2_drop_pct']:.1f}% drop)")
    lines.append(f"  → {r['verdict']}")

    # 15d
    r = results["15d_gallows_base"]
    lines.append(f"\n  15d — GALLOWS BASE RATE CORRECTION")
    lines.append("  " + "-" * 66)
    lines.append(f"  P(gallows in 1st word | para_start): {r['p_start_first_word']:.4f} "
                 f"({r['para_start_gallows_first_word']}/{r['para_start_total']})")
    lines.append(f"  P(gallows in 1st word | continuation): {r['p_cont_first_word']:.4f} "
                 f"({r['cont_gallows_first_word']}/{r['cont_total']})")
    lines.append(f"  Absolute diff: {r['absolute_difference']:+.4f}")
    lines.append(f"  Relative risk: {r['relative_risk']:.3f}")
    lines.append(f"  Cohen's h: {r['cohens_h']:.4f} ({r['interpretation']['cohens_h_meaning']})")
    lines.append(f"  Base rate (all first words): {r['base_rate_first_word']:.4f}")
    lines.append(f"  P(any gallows on line | para_start): {r['p_start_any_word']:.4f}")
    lines.append(f"  P(any gallows on line | continuation): {r['p_cont_any_word']:.4f}")
    lines.append(f"  {r['interpretation']['base_rate_problem']}")
    lines.append(f"  → {r['verdict']}")

    # 15e
    r = results["15e_m_section"]
    lines.append(f"\n  15e — 'm' SECTION-RATE INVESTIGATION")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Sec':<4} {'Tot m':>6} {'%LF':>7} {'%Med':>7} {'%lines→m':>9} "
                 f"{'AvgLen m':>9} {'AvgLen ¬m':>9}")
    for sec in sorted(r["per_section"].keys()):
        v = r["per_section"][sec]
        lines.append(f"  {sec:<4} {v['total_m']:>6} {v['pct_line_final']:>6.1f}% "
                     f"{v['pct_medial']:>6.1f}% {v['pct_lines_ending_m']:>8.1f}% "
                     f"{v['avg_line_length_m']:>9.1f} {v['avg_line_length_no_m']:>9.1f}")

    pb = r.get("pharma_vs_balneo", {})
    lines.append(f"\n  Pharma vs Balneo:")
    lines.append(f"    Pharma: {pb.get('pharma_pct_lf', 0):.1f}% LF, "
                 f"{pb.get('pharma_pct_medial', 0):.1f}% medial")
    lines.append(f"    Balneo: {pb.get('balneo_pct_lf', 0):.1f}% LF, "
                 f"{pb.get('balneo_pct_medial', 0):.1f}% medial")
    lines.append(f"    Pharma non-final words: {pb.get('pharma_top_non_final', [])}")
    lines.append(f"    Balneo non-final words: {pb.get('balneo_top_non_final', [])}")

    # Overall
    lines.append("\n" + "=" * 72)
    lines.append("  AUDIT SUMMARY")
    lines.append("=" * 72)

    verdicts = {
        "15a MI without hapax": results["15a_mi_hapax"]["verdict"],
        "15b Hand bigrams within-section": results["15b_hand_bigrams"]["verdict"],
        "15c SG contexts within-section": results["15c_sg_context"]["verdict"],
        "15d Gallows base rate": results["15d_gallows_base"]["verdict"],
        "15e 'm' section rate": results["15e_m_section"]["verdict"],
    }

    for name, verdict in verdicts.items():
        icon = {"SURVIVES": "OK", "SURVIVES_WEAK": "~OK", "WEAKENED": "!!", "COLLAPSES": "XX",
                "NEGLIGIBLE_EFFECT": "!!", "SECTION_DEPENDENT": "!!"}
        lines.append(f"  [{icon.get(verdict, '??'):>4s}] {name}: {verdict}")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


# =====================================================================
# Save to DB
# =====================================================================

def save_to_db(config: ToolkitConfig, results: dict):
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS confounder_audit")
    cur.execute("""
        CREATE TABLE confounder_audit (
            test TEXT,
            key TEXT,
            original_value REAL,
            controlled_value REAL,
            delta REAL,
            verdict TEXT,
            detail_json TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    def _ins(test, key, orig, ctrl, verdict, detail=None):
        delta = ctrl - orig if orig is not None and ctrl is not None else None
        cur.execute("INSERT INTO confounder_audit VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (test, key, orig, ctrl, delta, verdict,
                     json.dumps(detail, default=str) if detail else None))

    # 15a
    r = results["15a_mi_hapax"]
    z1 = r["by_threshold"].get(1, {}).get("z_score", 0)
    for thresh, v in r["by_threshold"].items():
        _ins("15a_mi_hapax", f"min_freq_{thresh}", z1, v["z_score"], r["verdict"])

    # 15b
    r = results["15b_hand_bigrams"]
    for sec, v in r["by_section"].items():
        _ins("15b_hand_bigrams", sec, None, v["chi2"], r["verdict"], v)

    # 15c
    r = results["15c_sg_context"]
    _ins("15c_sg_context", "all", r["chi2_all"], r["chi2_herbal_only"], r["verdict"])

    # 15d
    r = results["15d_gallows_base"]
    _ins("15d_gallows_base", "cohens_h", None, r["cohens_h"], r["verdict"])
    _ins("15d_gallows_base", "abs_diff", None, r["absolute_difference"], r["verdict"])

    # 15e
    r = results["15e_m_section"]
    for sec, v in r["per_section"].items():
        _ins("15e_m_section", sec, None, v["pct_line_final"], r["verdict"])

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 15 — Confounder Audit."""
    report_path = config.stats_dir / "confounder_audit.json"
    summary_path = config.stats_dir / "confounder_audit_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Confounder audit exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 15 — CONFOUNDER AUDIT")

    # Parse
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    lines = parse_ivtff_lines(eva_file)
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages, {len(lines)} lines")

    results = {}

    # 15a
    print_step("15a — MI without hapax (5 thresholds, 500 perms each)...")
    results["15a_mi_hapax"] = test_mi_without_hapax(pages)
    r = results["15a_mi_hapax"]
    click.echo(f"    z-drop at freq≥5: {r['z_drop_at_freq5']:.1f}% → {r['verdict']}")

    # 15b
    print_step("15b — Hand bigrams within-section...")
    results["15b_hand_bigrams"] = test_hand_bigrams_within_section(pages)
    r = results["15b_hand_bigrams"]
    click.echo(f"    {r['n_significant']}/{r['n_sections_tested']} sections significant "
               f"→ {r['verdict']}")

    # 15c
    print_step("15c — SG type contexts within-section (CRITICAL)...")
    results["15c_sg_context"] = test_sg_context_within_section(pages)
    r = results["15c_sg_context"]
    click.echo(f"    chi² all={r['chi2_all']:.1f}, herbal={r['chi2_herbal_only']:.1f} "
               f"(drop {r['chi2_drop_pct']:.1f}%) → {r['verdict']}")

    # 15d
    print_step("15d — Gallows base rate correction...")
    results["15d_gallows_base"] = test_gallows_base_rate(lines)
    r = results["15d_gallows_base"]
    click.echo(f"    Cohen's h={r['cohens_h']:.4f} ({r['interpretation']['cohens_h_meaning']}) "
               f"→ {r['verdict']}")

    # 15e
    print_step("15e — 'm' section-rate investigation...")
    results["15e_m_section"] = test_m_section_detail(pages)
    r = results["15e_m_section"]
    click.echo(f"    → {r['verdict']}")

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
    click.echo(f"    DB:   confounder_audit table")

    click.echo(f"\n{summary}")
