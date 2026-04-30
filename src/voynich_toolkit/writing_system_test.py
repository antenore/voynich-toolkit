"""
Phase 14 — Writing system minimal specification.

Synthesizes all confirmed properties into a minimal specification:
what is the smallest set of rules/mechanisms that produces text
with ALL these properties?

Sub-tests:
  14a — Property dependency matrix (which properties imply others?)
  14b — Minimal mechanism inventory (6 levels: char/word/line/para/section/scribe)
  14c — Matlach reduced alphabet recomputation (19→~10 chars)
  14d — Cross-hand slot grammar consistency

All tests are on raw EVA — no decoding, no lexicon.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2_contingency

from .config import ToolkitConfig
from .rugg_test import _measure_slot_grammar, _measure_zipf, _measure_entropy
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42

# =====================================================================
# 14a — Property dependency matrix
# =====================================================================

# The 16+ confirmed properties and which mechanism level they require
PROPERTY_CATALOG = {
    "line_self_containment": {
        "description": "Lines are self-contained units (z=-3.60)",
        "level": "LINE",
        "phase": "7a",
        "depends_on": [],
        "implied_by": ["m_end_marker"],
    },
    "m_end_marker": {
        "description": "'m' concentrates at line-final (71%, z=+55.5)",
        "level": "LINE",
        "phase": "7a,12",
        "depends_on": [],
        "implied_by": [],
    },
    "m_g_f_class": {
        "description": "m,g,f form line-final character class (Phase 12)",
        "level": "LINE",
        "phase": "12",
        "depends_on": ["m_end_marker"],
        "implied_by": [],
    },
    "gallows_para_start": {
        "description": "Simple gallows appear more at paragraph starts (z=+10.73)",
        "level": "PARAGRAPH",
        "phase": "7b",
        "depends_on": [],
        "implied_by": [],
    },
    "split_gallows_not_para": {
        "description": "Split gallows are NOT paragraph markers (z=+1.34 ns)",
        "level": "WORD",
        "phase": "7b",
        "depends_on": [],
        "implied_by": [],
    },
    "sg_different_functions": {
        "description": "4 SG types have different contexts (chi²=334.7)",
        "level": "WORD",
        "phase": "13",
        "depends_on": [],
        "implied_by": [],
    },
    "sg_independent_m": {
        "description": "SG and 'm' line-end concentration are independent (chi²=0.01)",
        "level": "LINE",
        "phase": "13",
        "depends_on": [],
        "implied_by": [],
    },
    "para_coherence": {
        "description": "Paragraphs group thematically similar lines",
        "level": "PARAGRAPH",
        "phase": "7c",
        "depends_on": [],
        "implied_by": [],
    },
    "word_section_mi": {
        "description": "Words carry section-specific information (MI z=+40)",
        "level": "SECTION",
        "phase": "8a",
        "depends_on": [],
        "implied_by": [],
    },
    "astro_zodiac_anti": {
        "description": "Astro-Zodiac share LESS vocabulary (z=-6.02)",
        "level": "SECTION",
        "phase": "8b",
        "depends_on": ["word_section_mi"],
        "implied_by": [],
    },
    "hand_system_uniform": {
        "description": "5+ scribes all use the same system (8/8 significant)",
        "level": "SCRIBE",
        "phase": "0,1a",
        "depends_on": [],
        "implied_by": [],
    },
    "hand_bigrams_distinct": {
        "description": "Different hands have distinct bigram signatures",
        "level": "SCRIBE",
        "phase": "2",
        "depends_on": [],
        "implied_by": [],
    },
    "hand_unknown_anomaly": {
        "description": "Hand ? is anomalous even within sections (|z|=6.86)",
        "level": "SCRIBE",
        "phase": "2e",
        "depends_on": [],
        "implied_by": [],
    },
    "slot_grammar": {
        "description": "Rigid positional character structure (Cramér's V)",
        "level": "WORD",
        "phase": "0",
        "depends_on": [],
        "implied_by": [],
    },
    "zipf_law": {
        "description": "Word frequencies follow Zipf's law (slope ~ -1)",
        "level": "CHARACTER",
        "phase": "9c",
        "depends_on": [],
        "implied_by": [],
    },
    "low_entropy": {
        "description": "Character entropy anomalously low (h2=2.37)",
        "level": "CHARACTER",
        "phase": "9a",
        "depends_on": [],
        "implied_by": [],
    },
    "sg_herbal_concentration": {
        "description": "Split gallows concentrated in herbal (9.2%, z=+8.51)",
        "level": "SECTION",
        "phase": "7d,13",
        "depends_on": [],
        "implied_by": [],
    },
}


def test_property_dependencies() -> dict:
    """Build property dependency matrix.

    Returns which properties imply others (based on logical analysis)
    and counts independent vs dependent properties.
    """
    n_total = len(PROPERTY_CATALOG)

    # Count by level
    by_level = defaultdict(list)
    for name, prop in PROPERTY_CATALOG.items():
        by_level[prop["level"]].append(name)

    # Count dependencies
    n_with_deps = sum(1 for p in PROPERTY_CATALOG.values() if p["depends_on"])
    n_implied = sum(1 for p in PROPERTY_CATALOG.values() if p["implied_by"])
    n_independent = n_total - n_with_deps

    # Build dependency edges
    edges = []
    for name, prop in PROPERTY_CATALOG.items():
        for dep in prop["depends_on"]:
            edges.append({"from": dep, "to": name, "type": "depends_on"})
        for imp in prop["implied_by"]:
            edges.append({"from": imp, "to": name, "type": "implied_by"})

    return {
        "n_total": n_total,
        "n_independent": n_independent,
        "n_with_dependencies": n_with_deps,
        "n_implied": n_implied,
        "by_level": {k: len(v) for k, v in sorted(by_level.items())},
        "level_detail": {k: v for k, v in sorted(by_level.items())},
        "edges": edges,
    }


# =====================================================================
# 14b — Minimal mechanism inventory
# =====================================================================

MECHANISM_LEVELS = [
    "CHARACTER",   # character frequency tables
    "WORD",        # word-level rules (slot grammar)
    "LINE",        # line-boundary awareness
    "PARAGRAPH",   # paragraph structure
    "SECTION",     # section differentiation
    "SCRIBE",      # per-scribe variation
]

# What each tested model has
MODEL_CAPABILITIES = {
    "Markov_order2": {
        "CHARACTER": True,
        "WORD": False,
        "LINE": False,
        "PARAGRAPH": False,
        "SECTION": False,
        "SCRIBE": False,
        "result": "0/4 structural (Phase 9f)",
    },
    "Rugg_grille": {
        "CHARACTER": True,
        "WORD": True,
        "LINE": False,
        "PARAGRAPH": False,
        "SECTION": True,  # different grilles per language
        "SCRIBE": False,
        "result": "10/16 (Phase 27.9)",
    },
    "Timm_self_citation": {
        "CHARACTER": True,
        "WORD": True,  # copies words
        "LINE": True,  # line-local copying
        "PARAGRAPH": False,
        "SECTION": False,
        "SCRIBE": False,
        "result": "0/4 structural (Phase 10)",
    },
    "Naibbe_cipher": {
        "CHARACTER": True,
        "WORD": True,  # word-level encryption
        "LINE": False,
        "PARAGRAPH": False,
        "SECTION": True,  # different plaintext per section
        "SCRIBE": False,
        "result": "8/16 (Phase 11)",
    },
}


def test_mechanism_inventory() -> dict:
    """Categorize each property by mechanism level and cross-reference with models.

    Returns which mechanism levels are needed and which models have them.
    """
    # Properties per level
    level_properties = defaultdict(list)
    for name, prop in PROPERTY_CATALOG.items():
        level_properties[prop["level"]].append(name)

    # For each model: which levels it covers and how many properties it could reproduce
    model_analysis = {}
    for model_name, capabilities in MODEL_CAPABILITIES.items():
        covered_levels = [l for l in MECHANISM_LEVELS if capabilities.get(l)]
        # Properties in covered levels
        covered_props = []
        for level in covered_levels:
            covered_props.extend(level_properties.get(level, []))
        uncovered_props = [p for p in PROPERTY_CATALOG
                          if PROPERTY_CATALOG[p]["level"] not in covered_levels]

        model_analysis[model_name] = {
            "covered_levels": covered_levels,
            "n_covered_levels": len(covered_levels),
            "n_covered_properties": len(covered_props),
            "n_uncovered_properties": len(uncovered_props),
            "uncovered_properties": uncovered_props,
            "result": capabilities["result"],
        }

    # Minimal model: what levels are needed?
    required_levels = set()
    for prop in PROPERTY_CATALOG.values():
        required_levels.add(prop["level"])

    return {
        "level_properties": {k: v for k, v in sorted(level_properties.items())},
        "required_levels": sorted(required_levels),
        "n_required_levels": len(required_levels),
        "model_analysis": model_analysis,
        "conclusion": (
            f"A complete model needs ALL {len(required_levels)} mechanism levels: "
            f"{', '.join(sorted(required_levels))}. "
            f"No tested model has all of them."
        ),
    }


# =====================================================================
# 14c — Matlach reduced alphabet
# =====================================================================

# Matlach et al. (2022) proposed alphabet reduction.
# Based on their "Symbol roles revisited" paper, approximate mapping:
# Key merges: i→i (keep), n→i (n is two i's), l→l (keep), r→l (similar),
# e→e (keep), a→a (keep), o→o (keep), but these are approximate.
# We define a conservative reduction based on visual/functional similarity.
MATLACH_MAP = {
    "a": "A",    # bench (a-group)
    "c": "C",    # c is part of ch/sh digraphs
    "d": "D",    # d-group
    "e": "E",    # e-group
    "f": "F",    # gallows (f≈p merge candidate)
    "g": "G",    # rare, keep separate
    "h": "H",    # h (part of ch/sh/th/kh)
    "i": "I",    # i-group
    "k": "K",    # gallows (k≈t merge candidate)
    "l": "L",    # l-group (l≈r merge candidate)
    "m": "M",    # m-group (keep — line-end preference, section-dependent)
    "n": "I",    # n → i (n = ii in many analyses)
    "o": "O",    # o-group (o≈a merge candidate)
    "p": "F",    # p → f (same gallows family, one vs two loops)
    "q": "Q",    # q-group (keep — word-initial only)
    "r": "L",    # r → l (similar shapes)
    "s": "S",    # s-group
    "t": "K",    # t → k (same gallows family)
    "y": "Y",    # y-group (keep — word-final frequent)
}
# This reduces 19 → 13 characters: A C D E F G H I K L M O Q S Y
# Conservative: only merging pairs with strong visual/functional evidence


def apply_matlach_reduction(words: list[str]) -> list[str]:
    """Apply Matlach alphabet reduction to a word list."""
    reduced = []
    for word in words:
        rw = "".join(MATLACH_MAP.get(ch, ch) for ch in word)
        reduced.append(rw)
    return reduced


def test_matlach_alphabet(pages: list[dict]) -> dict:
    """Recompute key metrics with Matlach reduced alphabet.

    Compare: entropy, Zipf, slot grammar, vocabulary size.
    """
    # Original words
    orig_words = [w for p in pages for w in p["words"]]

    # Reduced words
    reduced_words = apply_matlach_reduction(orig_words)

    # Original metrics
    orig_entropy = _measure_entropy(orig_words)
    orig_zipf = _measure_zipf(orig_words)
    orig_slot = _measure_slot_grammar(orig_words)
    orig_vocab = len(set(orig_words))

    # Reduced metrics
    red_entropy = _measure_entropy(reduced_words)
    red_zipf = _measure_zipf(reduced_words)
    red_slot = _measure_slot_grammar(reduced_words)
    red_vocab = len(set(reduced_words))

    # Character entropy
    orig_chars = [ch for w in orig_words for ch in w]
    red_chars = [ch for w in reduced_words for ch in w]

    orig_char_freq = Counter(orig_chars)
    red_char_freq = Counter(red_chars)

    orig_h1 = -sum((c / len(orig_chars)) * math.log2(c / len(orig_chars))
                    for c in orig_char_freq.values())
    red_h1 = -sum((c / len(red_chars)) * math.log2(c / len(red_chars))
                   for c in red_char_freq.values())

    # How many types merge?
    merge_count = orig_vocab - red_vocab

    return {
        "mapping": MATLACH_MAP,
        "n_original_chars": 19,
        "n_reduced_chars": len(set(MATLACH_MAP.values())),
        "merges": {"n→i": True, "p→f": True, "r→l": True, "t→k": True},
        "original": {
            "char_entropy_h1": round(orig_h1, 4),
            "word_entropy": orig_entropy["entropy_bits"],
            "zipf_slope": orig_zipf["slope"],
            "slot_cramers_v": orig_slot["cramers_v"],
            "vocabulary": orig_vocab,
        },
        "reduced": {
            "char_entropy_h1": round(red_h1, 4),
            "word_entropy": red_entropy["entropy_bits"],
            "zipf_slope": red_zipf["slope"],
            "slot_cramers_v": red_slot["cramers_v"],
            "vocabulary": red_vocab,
        },
        "delta": {
            "char_entropy": round(red_h1 - orig_h1, 4),
            "word_entropy": round(red_entropy["entropy_bits"] - orig_entropy["entropy_bits"], 3),
            "zipf_slope": round(red_zipf["slope"] - orig_zipf["slope"], 3),
            "slot_cramers_v": round(red_slot["cramers_v"] - orig_slot["cramers_v"], 4),
            "vocabulary_merged": merge_count,
            "vocabulary_pct_reduction": round(merge_count / orig_vocab * 100, 1),
        },
        "interpretation": (
            "SIGNIFICANT_CHANGE" if abs(red_h1 - orig_h1) > 0.3
            else "MINOR_CHANGE" if abs(red_h1 - orig_h1) > 0.1
            else "NEGLIGIBLE_CHANGE"
        ),
    }


# =====================================================================
# 14d — Cross-hand slot grammar consistency
# =====================================================================

def test_cross_hand_slot_grammar(pages: list[dict]) -> dict:
    """Compute slot grammar (Cramér's V) for each hand separately.

    If all hands similar → uniform writing system.
    If one hand differs → different encoding rules.
    """
    # Group words by hand
    hand_words: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        hand = page.get("hand", "?")
        hand_words[hand].extend(page["words"])

    results = {}
    for hand in sorted(hand_words.keys()):
        words = hand_words[hand]
        if len(words) < 200:
            continue
        slot = _measure_slot_grammar(words)
        results[hand] = {
            "n_words": len(words),
            "cramers_v": slot["cramers_v"],
            "chi2": slot["chi2"],
            "significant": slot.get("significant", False),
        }

    # Consistency
    vs = [v["cramers_v"] for v in results.values()]
    mean_v = float(np.mean(vs)) if vs else 0
    std_v = float(np.std(vs)) if vs else 0
    cv = std_v / mean_v if mean_v > 0 else 0

    # Outliers (> 2 std from mean)
    outliers = []
    for hand, v in results.items():
        z = (v["cramers_v"] - mean_v) / std_v if std_v > 0 else 0
        results[hand]["z_vs_mean"] = round(z, 2)
        if abs(z) > 2:
            outliers.append(hand)

    return {
        "per_hand": results,
        "mean_cramers_v": round(mean_v, 4),
        "std_cramers_v": round(std_v, 4),
        "cv": round(cv, 4),
        "n_hands": len(results),
        "outliers": outliers,
        "interpretation": (
            "UNIFORM_SYSTEM" if cv < 0.10
            else "MOSTLY_UNIFORM" if cv < 0.20
            else "VARIABLE_SYSTEM"
        ),
    }


# =====================================================================
# Summary formatting
# =====================================================================

def format_summary(results: dict) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 14 — Writing System Minimal Specification")
    lines.append("=" * 72)

    # 14a
    r = results["14a_dependencies"]
    lines.append(f"\n  14a — PROPERTY DEPENDENCY MATRIX")
    lines.append("  " + "-" * 66)
    lines.append(f"  Total properties: {r['n_total']}")
    lines.append(f"  Independent: {r['n_independent']}, "
                 f"With dependencies: {r['n_with_dependencies']}")
    lines.append(f"  By level: {r['by_level']}")
    lines.append(f"  Dependencies:")
    for edge in r["edges"]:
        lines.append(f"    {edge['from']} → {edge['to']} ({edge['type']})")

    # 14b
    r = results["14b_mechanisms"]
    lines.append(f"\n  14b — MINIMAL MECHANISM INVENTORY")
    lines.append("  " + "-" * 66)
    lines.append(f"  Required levels: {r['required_levels']} ({r['n_required_levels']} total)")
    lines.append(f"\n  Properties per level:")
    for level, props in r["level_properties"].items():
        lines.append(f"    {level}: {len(props)} — {props}")
    lines.append(f"\n  Model comparison:")
    for model, analysis in r["model_analysis"].items():
        lines.append(f"    {model}:")
        lines.append(f"      Levels: {analysis['covered_levels']}")
        lines.append(f"      Covered: {analysis['n_covered_properties']}/{r['level_properties']}")
        lines.append(f"      Missing: {analysis['uncovered_properties']}")
        lines.append(f"      Result: {analysis['result']}")
    lines.append(f"\n  {r['conclusion']}")

    # 14c
    r = results["14c_matlach"]
    lines.append(f"\n  14c — MATLACH REDUCED ALPHABET (19→{r['n_reduced_chars']} chars)")
    lines.append("  " + "-" * 66)
    lines.append(f"  Merges: n→i, p→f, r→l, t→k")
    lines.append(f"  {'Metric':<25s} {'Original':>10s} {'Reduced':>10s} {'Delta':>10s}")
    lines.append(f"  {'char entropy (H1)':<25s} "
                 f"{r['original']['char_entropy_h1']:>10.4f} "
                 f"{r['reduced']['char_entropy_h1']:>10.4f} "
                 f"{r['delta']['char_entropy']:>+10.4f}")
    lines.append(f"  {'word entropy':<25s} "
                 f"{r['original']['word_entropy']:>10.3f} "
                 f"{r['reduced']['word_entropy']:>10.3f} "
                 f"{r['delta']['word_entropy']:>+10.3f}")
    lines.append(f"  {'Zipf slope':<25s} "
                 f"{r['original']['zipf_slope']:>10.3f} "
                 f"{r['reduced']['zipf_slope']:>10.3f} "
                 f"{r['delta']['zipf_slope']:>+10.3f}")
    lines.append(f"  {'slot Cramér V':<25s} "
                 f"{r['original']['slot_cramers_v']:>10.4f} "
                 f"{r['reduced']['slot_cramers_v']:>10.4f} "
                 f"{r['delta']['slot_cramers_v']:>+10.4f}")
    lines.append(f"  {'vocabulary types':<25s} "
                 f"{r['original']['vocabulary']:>10d} "
                 f"{r['reduced']['vocabulary']:>10d} "
                 f"{-r['delta']['vocabulary_merged']:>+10d}")
    lines.append(f"\n  Vocabulary reduction: {r['delta']['vocabulary_pct_reduction']:.1f}%")
    lines.append(f"  → {r['interpretation']}")

    # 14d
    r = results["14d_cross_hand_slot"]
    lines.append(f"\n  14d — CROSS-HAND SLOT GRAMMAR CONSISTENCY")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Hand':<6s} {'Words':>8s} {'Cramér V':>10s} {'z vs mean':>10s}")
    for hand, v in sorted(r["per_hand"].items()):
        marker = " *" if hand in r.get("outliers", []) else ""
        lines.append(f"  {hand:<6s} {v['n_words']:>8d} {v['cramers_v']:>10.4f} "
                     f"{v['z_vs_mean']:>+10.2f}{marker}")
    lines.append(f"\n  Mean: {r['mean_cramers_v']:.4f}, CV: {r['cv']:.4f}")
    if r["outliers"]:
        lines.append(f"  Outliers: {r['outliers']}")
    lines.append(f"  → {r['interpretation']}")

    # Overall
    lines.append("\n" + "=" * 72)
    lines.append("  WRITING SYSTEM SPECIFICATION")
    lines.append("=" * 72)
    lines.append(f"\n  The Voynich writing system requires at minimum:")
    lines.append(f"  1. CHARACTER level: low entropy, Zipf-compliant frequency distribution")
    lines.append(f"  2. WORD level: slot grammar (positional char constraints), SG markers")
    lines.append(f"  3. LINE level: line as semantic unit, m/g/f line-end concentration (m robust, g/f fragile)")
    lines.append(f"  4. PARAGRAPH level: gallows at paragraph start, internal coherence")
    lines.append(f"  5. SECTION level: different vocabularies, SG concentration in herbal")
    lines.append(f"  6. SCRIBE level: per-hand bigram signatures, Hand ? anomaly")
    lines.append(f"\n  No tested model covers all 6 levels.")
    lines.append("=" * 72)

    return "\n".join(lines) + "\n"


# =====================================================================
# Save to DB
# =====================================================================

def save_to_db(config: ToolkitConfig, results: dict):
    """Save results to SQLite database."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS writing_system_test")
    cur.execute("""
        CREATE TABLE writing_system_test (
            test TEXT,
            key TEXT,
            observed REAL,
            detail_json TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    def _ins(test, key, val, detail=None):
        cur.execute("INSERT INTO writing_system_test VALUES (?, ?, ?, ?)",
                    (test, key, val,
                     json.dumps(detail, default=str) if detail else None))

    # 14a
    r = results["14a_dependencies"]
    _ins("14a_dependencies", "n_total", r["n_total"])
    _ins("14a_dependencies", "n_independent", r["n_independent"])
    for level, count in r["by_level"].items():
        _ins("14a_dependencies", f"level_{level}", count)

    # 14b
    for model, analysis in results["14b_mechanisms"]["model_analysis"].items():
        _ins("14b_mechanisms", model, analysis["n_covered_levels"], analysis)

    # 14c
    r = results["14c_matlach"]
    _ins("14c_matlach", "char_entropy_delta", r["delta"]["char_entropy"])
    _ins("14c_matlach", "word_entropy_delta", r["delta"]["word_entropy"])
    _ins("14c_matlach", "zipf_delta", r["delta"]["zipf_slope"])
    _ins("14c_matlach", "slot_v_delta", r["delta"]["slot_cramers_v"])
    _ins("14c_matlach", "vocab_reduction_pct", r["delta"]["vocabulary_pct_reduction"])

    # 14d
    for hand, v in results["14d_cross_hand_slot"]["per_hand"].items():
        _ins("14d_cross_hand_slot", hand, v["cramers_v"], v)

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 14 — Writing system minimal specification."""
    report_path = config.stats_dir / "writing_system_test.json"
    summary_path = config.stats_dir / "writing_system_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Writing system test exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 14 — Writing System Minimal Specification")

    # Parse
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    results = {}

    # 14a
    print_step("14a — Property dependency matrix...")
    results["14a_dependencies"] = test_property_dependencies()
    r = results["14a_dependencies"]
    click.echo(f"    {r['n_total']} properties, {r['n_independent']} independent")
    click.echo(f"    By level: {r['by_level']}")

    # 14b
    print_step("14b — Minimal mechanism inventory...")
    results["14b_mechanisms"] = test_mechanism_inventory()
    r = results["14b_mechanisms"]
    click.echo(f"    Required: {r['required_levels']}")
    for model, a in r["model_analysis"].items():
        click.echo(f"    {model}: {a['n_covered_levels']}/6 levels, "
                   f"result={a['result']}")

    # 14c
    print_step("14c — Matlach reduced alphabet recomputation...")
    results["14c_matlach"] = test_matlach_alphabet(pages)
    r = results["14c_matlach"]
    click.echo(f"    19→{r['n_reduced_chars']} chars")
    click.echo(f"    Δ char entropy: {r['delta']['char_entropy']:+.4f}")
    click.echo(f"    Δ Zipf slope: {r['delta']['zipf_slope']:+.3f}")
    click.echo(f"    Δ slot Cramér V: {r['delta']['slot_cramers_v']:+.4f}")
    click.echo(f"    Vocabulary reduction: {r['delta']['vocabulary_pct_reduction']:.1f}%")
    click.echo(f"    → {r['interpretation']}")

    # 14d
    print_step("14d — Cross-hand slot grammar consistency...")
    results["14d_cross_hand_slot"] = test_cross_hand_slot_grammar(pages)
    r = results["14d_cross_hand_slot"]
    click.echo(f"    Mean Cramér V: {r['mean_cramers_v']:.4f}, CV: {r['cv']:.4f}")
    if r["outliers"]:
        click.echo(f"    Outliers: {r['outliers']}")
    click.echo(f"    → {r['interpretation']}")

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
    click.echo(f"    DB:   writing_system_test table")

    click.echo(f"\n{summary}")
