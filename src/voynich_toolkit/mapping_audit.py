"""Mapping audit: per-letter optimality test.

For each EVA→Hebrew mapping, tests all 22 Hebrew alternatives (keeping
all other mappings fixed) and reports whether the current assignment is
optimal. Identifies weak/wrong letters and suggests improvements.

Also tests: positional splits (d@init→b, h@init→s), digraph (ch→k),
and probes unmapped Hebrew letters (zayin, tsade, qof).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy

import numpy as np

from .config import ToolkitConfig
from .full_decode import (
    CH_HEBREW,
    FULL_MAPPING,
    II_HEBREW,
    INITIAL_D_HEBREW,
    INITIAL_H_HEBREW,
    I_HEBREW,
    preprocess_eva,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step

HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"


# ── Parametric decoder ──────────────────────────────────────────


def decode_hebrew(
    eva_word: str,
    mapping: dict[str, str],
    ch_hebrew: str = CH_HEBREW,
    ii_hebrew: str = II_HEBREW,
    i_hebrew: str = I_HEBREW,
    initial_d: str = INITIAL_D_HEBREW,
    initial_h: str = INITIAL_H_HEBREW,
) -> str | None:
    """Decode EVA word to Hebrew with parameterized mapping.

    Returns Hebrew string or None if any char is unknown.
    """
    prefix, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))

    hebrew_parts = []
    for ch in chars:
        if ch == "\x01":
            hebrew_parts.append(ii_hebrew)
        elif ch == "\x02":
            hebrew_parts.append(i_hebrew)
        elif ch == "\x03":
            hebrew_parts.append(ch_hebrew)
        elif ch in mapping:
            hebrew_parts.append(mapping[ch])
        else:
            return None  # unknown char

    # Positional splits
    if hebrew_parts and hebrew_parts[0] == "d":
        hebrew_parts[0] = initial_d
    if hebrew_parts and hebrew_parts[0] == "h":
        hebrew_parts[0] = initial_h

    return "".join(hebrew_parts)


# ── Data loading ────────────────────────────────────────────────


def load_data(config: ToolkitConfig):
    """Load EVA words and full Hebrew lexicon (491K forms)."""
    with open(config.stats_dir / "full_decode.json") as f:
        decode_data = json.load(f)

    # Collect unique EVA words with frequencies
    eva_freqs: Counter = Counter()
    for page_data in decode_data["pages"].values():
        for w in page_data.get("words_eva", []):
            if w:
                eva_freqs[w] += 1

    with open(config.lexicon_dir / "lexicon_enriched.json") as f:
        lex_data = json.load(f)

    lexicon_set = set(lex_data["all_consonantal_forms"])
    form_to_gloss = lex_data["form_to_gloss"]

    return eva_freqs, lexicon_set, form_to_gloss


def load_honest_lexicon(config: ToolkitConfig) -> tuple[set, dict]:
    """Load Hebrew lexicon excluding Sefaria-Corpus bulk forms.

    Returns only forms from STEPBible, Jastrow, Klein, and Curated sources
    (~28K unique consonantal forms). Eliminates the yod-frequency artefact
    caused by the 445K Sefaria inflected-form corpus.

    Returns: (lexicon_set, form_to_gloss)
    """
    with open(config.lexicon_dir / "lexicon_enriched.json") as f:
        lex_data = json.load(f)

    honest_sources = {"STEPBible", "Jastrow", "Klein",
                      "Curato-Botanico", "Curato-IbnEzra",
                      "Curato-Medico", "Curato-Generale"}

    honest_forms: set[str] = set()
    by_domain = lex_data.get("by_domain", {})
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if entry.get("source", "") in honest_sources:
                form = entry.get("consonants", "")
                if form:
                    honest_forms.add(form)

    form_to_gloss = lex_data.get("form_to_gloss", {})
    return honest_forms, form_to_gloss


def count_matches(eva_freqs, lexicon_set, **decode_kwargs):
    """Decode all EVA words and count lexicon matches.

    Returns (matched_types, matched_tokens, total_types, total_tokens).
    """
    matched_types = 0
    matched_tokens = 0
    total_types = 0
    total_tokens = 0

    for word, freq in eva_freqs.items():
        heb = decode_hebrew(word, **decode_kwargs)
        if heb is None:
            continue
        total_types += 1
        total_tokens += freq
        if heb in lexicon_set:
            matched_types += 1
            matched_tokens += freq

    return matched_types, matched_tokens, total_types, total_tokens


# ── Per-letter audit ────────────────────────────────────────────


def audit_letter(
    eva_char: str,
    eva_freqs: Counter,
    lexicon_set: set,
    base_mapping: dict,
    allowed_hebrew: set[str] | None = None,
):
    """Test Hebrew alternatives for one EVA char.

    If allowed_hebrew is set, only test those chars (constrained mode).
    Always includes the current assignment for comparison.
    Returns dict with current assignment, best alternative, and all scores.
    """
    current_hebrew = base_mapping[eva_char]
    test_chars = set(HEBREW_CHARS) if allowed_hebrew is None else (allowed_hebrew | {current_hebrew})

    results = {}
    for heb_char in test_chars:
        test_mapping = dict(base_mapping)
        test_mapping[eva_char] = heb_char

        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=test_mapping
        )
        results[heb_char] = {"types": mt, "tokens": mk}

    # Sort by tokens (primary) then types
    ranked = sorted(
        results.items(), key=lambda x: (x[1]["tokens"], x[1]["types"]), reverse=True
    )

    best_char = ranked[0][0]
    best_tokens = ranked[0][1]["tokens"]
    current_tokens = results[current_hebrew]["tokens"]
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == current_hebrew
    )

    return {
        "eva": eva_char,
        "current": current_hebrew,
        "current_name": CONSONANT_NAMES.get(current_hebrew, "?"),
        "current_tokens": current_tokens,
        "current_types": results[current_hebrew]["types"],
        "current_rank": current_rank,
        "best": best_char,
        "best_name": CONSONANT_NAMES.get(best_char, "?"),
        "best_tokens": best_tokens,
        "best_types": ranked[0][1]["types"],
        "gap_tokens": best_tokens - current_tokens,
        "gap_pct": round(
            100 * (best_tokens - current_tokens) / current_tokens, 1
        )
        if current_tokens
        else 0,
        "top5": [
            {
                "hebrew": ch,
                "name": CONSONANT_NAMES.get(ch, "?"),
                "tokens": info["tokens"],
                "types": info["types"],
            }
            for ch, info in ranked[:5]
        ],
    }


# ── Special element audit ──────────────────────────────────────


def audit_special(eva_freqs, lexicon_set, base_mapping):
    """Audit ch digraph, ii/i allographs, and positional splits."""
    results = {}

    # ch digraph: try all 22 Hebrew
    print_step("Auditing ch digraph...")
    ch_results = {}
    for heb_char in HEBREW_CHARS:
        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=base_mapping, ch_hebrew=heb_char
        )
        ch_results[heb_char] = {"types": mt, "tokens": mk}

    ranked = sorted(
        ch_results.items(), key=lambda x: (x[1]["tokens"], x[1]["types"]), reverse=True
    )
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == CH_HEBREW
    )
    results["ch"] = {
        "element": "ch (digraph)",
        "current": CH_HEBREW,
        "current_name": CONSONANT_NAMES.get(CH_HEBREW, "?"),
        "current_tokens": ch_results[CH_HEBREW]["tokens"],
        "current_rank": current_rank,
        "best": ranked[0][0],
        "best_name": CONSONANT_NAMES.get(ranked[0][0], "?"),
        "best_tokens": ranked[0][1]["tokens"],
        "gap_tokens": ranked[0][1]["tokens"] - ch_results[CH_HEBREW]["tokens"],
        "top5": [
            {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": info["tokens"]}
            for ch, info in ranked[:5]
        ],
    }

    # ii: try all 22
    print_step("Auditing ii (he)...")
    ii_results = {}
    for heb_char in HEBREW_CHARS:
        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=base_mapping, ii_hebrew=heb_char
        )
        ii_results[heb_char] = {"types": mt, "tokens": mk}

    ranked = sorted(
        ii_results.items(), key=lambda x: (x[1]["tokens"], x[1]["types"]), reverse=True
    )
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == II_HEBREW
    )
    results["ii"] = {
        "element": "ii (allograph)",
        "current": II_HEBREW,
        "current_name": CONSONANT_NAMES.get(II_HEBREW, "?"),
        "current_tokens": ii_results[II_HEBREW]["tokens"],
        "current_rank": current_rank,
        "best": ranked[0][0],
        "best_name": CONSONANT_NAMES.get(ranked[0][0], "?"),
        "best_tokens": ranked[0][1]["tokens"],
        "gap_tokens": ranked[0][1]["tokens"] - ii_results[II_HEBREW]["tokens"],
        "top5": [
            {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": info["tokens"]}
            for ch, info in ranked[:5]
        ],
    }

    # i standalone: try all 22
    print_step("Auditing i (resh)...")
    i_results = {}
    for heb_char in HEBREW_CHARS:
        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=base_mapping, i_hebrew=heb_char
        )
        i_results[heb_char] = {"types": mt, "tokens": mk}

    ranked = sorted(
        i_results.items(), key=lambda x: (x[1]["tokens"], x[1]["types"]), reverse=True
    )
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == I_HEBREW
    )
    results["i"] = {
        "element": "i (standalone)",
        "current": I_HEBREW,
        "current_name": CONSONANT_NAMES.get(I_HEBREW, "?"),
        "current_tokens": i_results[I_HEBREW]["tokens"],
        "current_rank": current_rank,
        "best": ranked[0][0],
        "best_name": CONSONANT_NAMES.get(ranked[0][0], "?"),
        "best_tokens": ranked[0][1]["tokens"],
        "gap_tokens": ranked[0][1]["tokens"] - i_results[I_HEBREW]["tokens"],
        "top5": [
            {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": info["tokens"]}
            for ch, info in ranked[:5]
        ],
    }

    # Positional: d@initial → try all 22
    print_step("Auditing d@initial (bet)...")
    dinit_results = {}
    for heb_char in HEBREW_CHARS:
        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=base_mapping, initial_d=heb_char
        )
        dinit_results[heb_char] = {"types": mt, "tokens": mk}

    ranked = sorted(
        dinit_results.items(),
        key=lambda x: (x[1]["tokens"], x[1]["types"]),
        reverse=True,
    )
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == INITIAL_D_HEBREW
    )
    results["d@init"] = {
        "element": "d@initial (positional→bet)",
        "current": INITIAL_D_HEBREW,
        "current_name": CONSONANT_NAMES.get(INITIAL_D_HEBREW, "?"),
        "current_tokens": dinit_results[INITIAL_D_HEBREW]["tokens"],
        "current_rank": current_rank,
        "best": ranked[0][0],
        "best_name": CONSONANT_NAMES.get(ranked[0][0], "?"),
        "best_tokens": ranked[0][1]["tokens"],
        "gap_tokens": ranked[0][1]["tokens"]
        - dinit_results[INITIAL_D_HEBREW]["tokens"],
        "top5": [
            {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": info["tokens"]}
            for ch, info in ranked[:5]
        ],
    }

    # Positional: h@initial → try all 22
    print_step("Auditing h@initial (samekh)...")
    hinit_results = {}
    for heb_char in HEBREW_CHARS:
        mt, mk, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=base_mapping, initial_h=heb_char
        )
        hinit_results[heb_char] = {"types": mt, "tokens": mk}

    ranked = sorted(
        hinit_results.items(),
        key=lambda x: (x[1]["tokens"], x[1]["types"]),
        reverse=True,
    )
    current_rank = next(
        i + 1 for i, (ch, _) in enumerate(ranked) if ch == INITIAL_H_HEBREW
    )
    results["h@init"] = {
        "element": "h@initial (positional→samekh)",
        "current": INITIAL_H_HEBREW,
        "current_name": CONSONANT_NAMES.get(INITIAL_H_HEBREW, "?"),
        "current_tokens": hinit_results[INITIAL_H_HEBREW]["tokens"],
        "current_rank": current_rank,
        "best": ranked[0][0],
        "best_name": CONSONANT_NAMES.get(ranked[0][0], "?"),
        "best_tokens": ranked[0][1]["tokens"],
        "gap_tokens": ranked[0][1]["tokens"]
        - hinit_results[INITIAL_H_HEBREW]["tokens"],
        "top5": [
            {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": info["tokens"]}
            for ch, info in ranked[:5]
        ],
    }

    return results


# ── Summary ─────────────────────────────────────────────────────


def get_used_hebrew() -> set[str]:
    """Get all Hebrew letters currently assigned in the mapping."""
    used = set(FULL_MAPPING.values())
    used.add(CH_HEBREW)    # kaf
    used.add(II_HEBREW)    # he
    used.add(I_HEBREW)     # resh
    used.add(INITIAL_D_HEBREW)  # bet
    used.add(INITIAL_H_HEBREW)  # samekh
    return used


def audit_constrained(eva_freqs, lexicon_set, base_mapping):
    """Constrained audit: only test FREE Hebrew letters (not already assigned).

    This avoids the frequency bias of the unconstrained test.
    For allograph groups (f/p→lamed, d/i→resh), the shared letter is
    considered "used" and tested normally.
    """
    used = get_used_hebrew()
    free_hebrew = set(HEBREW_CHARS) - used  # zayin, tsade, qof + any others

    results = []
    for eva_char in sorted(base_mapping.keys()):
        current = base_mapping[eva_char]
        # Available: free Hebrew + current assignment
        available = free_hebrew | {current}

        # Test each available option
        scores = {}
        for heb_char in available:
            test_mapping = dict(base_mapping)
            test_mapping[eva_char] = heb_char
            _, mk, _, _ = count_matches(eva_freqs, lexicon_set, mapping=test_mapping)
            scores[heb_char] = mk

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        current_rank = next(i + 1 for i, (ch, _) in enumerate(ranked) if ch == current)
        best_char, best_tokens = ranked[0]

        results.append({
            "eva": eva_char,
            "current": current,
            "current_name": CONSONANT_NAMES.get(current, "?"),
            "current_tokens": scores[current],
            "current_rank": current_rank,
            "n_tested": len(available),
            "best": best_char,
            "best_name": CONSONANT_NAMES.get(best_char, "?"),
            "best_tokens": best_tokens,
            "gap_tokens": best_tokens - scores[current],
            "top3": [
                {"hebrew": ch, "name": CONSONANT_NAMES.get(ch, "?"), "tokens": tk}
                for ch, tk in ranked[:3]
            ],
        })

    return results


def format_summary(baseline, letter_results, special_results, constrained_results=None):
    """Format human-readable audit report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MAPPING AUDIT — Per-letter Optimality Test")
    lines.append("=" * 70)

    lines.append(f"\n  Baseline: {baseline['matched_types']:,} types, "
                 f"{baseline['matched_tokens']:,} tokens matched "
                 f"({baseline['match_rate_tokens']:.1f}%)")

    # ── CONSTRAINED AUDIT (primary result) ──
    if constrained_results:
        used = get_used_hebrew()
        free = set(HEBREW_CHARS) - used
        lines.append(f"\n{'='*70}")
        lines.append(f"  CONSTRAINED AUDIT (1-to-1: only free Hebrew letters)")
        lines.append(f"  Free Hebrew: {', '.join(sorted(free))} "
                     f"({', '.join(CONSONANT_NAMES.get(c,'?') for c in sorted(free))})")
        lines.append(f"{'='*70}")

        lines.append(
            f"\n  {'EVA':>4s}  {'Current':>10s}  {'Rank':>4s}  {'Tokens':>7s}  "
            f"{'Best free':>10s}  {'Gap':>7s}  {'Status'}"
        )
        lines.append("  " + "-" * 60)

        for r in sorted(constrained_results, key=lambda x: -x["gap_tokens"]):
            if r["current_rank"] == 1:
                status = "OPTIMAL"
            elif r["gap_tokens"] > 0:
                status = f"SWAP? +{r['gap_tokens']} → {r['best_name']}"
            else:
                status = "OK (tied)"
            lines.append(
                f"  {r['eva']:>4s}  {r['current_name']:>10s}  "
                f"{r['current_rank']:>4d}  {r['current_tokens']:>7,}  "
                f"{r['best_name']:>10s}  {r['gap_tokens']:>+7,}  {status}"
            )

        c_optimal = sum(1 for r in constrained_results if r["current_rank"] == 1)
        lines.append(
            f"\n  Constrained verdict: {c_optimal}/{len(constrained_results)} "
            f"at rank #1 (among free Hebrew only)"
        )

    # ── UNCONSTRAINED AUDIT (reference) ──
    lines.append(f"\n{'='*70}")
    lines.append(f"  UNCONSTRAINED AUDIT (all 22 Hebrew, reference only)")
    lines.append(f"  ⚠ Frequency bias: high-freq letters (yod,tav,shin) dominate")
    lines.append(f"{'='*70}")

    lines.append(f"\n── Standard Letters (17 EVA chars) ──")
    lines.append(
        f"  {'EVA':>4s}  {'Current':>10s}  {'Rank':>4s}  {'Tokens':>7s}  "
        f"{'Best':>10s}  {'Best tok':>8s}  {'Gap':>7s}  {'Status'}"
    )
    lines.append("  " + "-" * 66)

    for r in sorted(letter_results, key=lambda x: x["current_rank"]):
        status = "OK" if r["current_rank"] == 1 else f"(#{r['current_rank']})"
        lines.append(
            f"  {r['eva']:>4s}  {r['current_name']:>10s}  {r['current_rank']:>4d}  "
            f"{r['current_tokens']:>7,}  {r['best_name']:>10s}  "
            f"{r['best_tokens']:>8,}  {r['gap_tokens']:>+7,}  {status}"
        )

    # ── Special elements ──
    lines.append("\n── Special Elements ──")
    lines.append(
        f"  {'Element':>20s}  {'Current':>10s}  {'Rank':>4s}  {'Tokens':>7s}  "
        f"{'Best':>10s}  {'Gap':>7s}"
    )
    lines.append("  " + "-" * 60)

    for key in ["ch", "ii", "i", "d@init", "h@init"]:
        r = special_results[key]
        lines.append(
            f"  {r['element']:>20s}  {r['current_name']:>10s}  "
            f"{r['current_rank']:>4d}  {r['current_tokens']:>7,}  "
            f"{r['best_name']:>10s}  {r['gap_tokens']:>+7,}"
        )

    # ── Final verdict ──
    optimal_count = sum(1 for r in letter_results if r["current_rank"] == 1)
    special_optimal = sum(
        1 for k in ["ch", "ii", "i", "d@init", "h@init"]
        if special_results[k]["current_rank"] == 1
    )
    total_elements = len(letter_results) + 5
    total_optimal = optimal_count + special_optimal

    lines.append(f"\n── Verdict ──")
    lines.append(
        f"  Unconstrained: {total_optimal}/{total_elements} at rank #1"
    )
    if constrained_results:
        c_optimal = sum(1 for r in constrained_results if r["current_rank"] == 1)
        lines.append(
            f"  Constrained:   {c_optimal}/{len(constrained_results)} at rank #1"
        )
        swappable = [r for r in constrained_results if r["gap_tokens"] > 0 and r["current_rank"] > 1]
        if swappable:
            lines.append(f"\n  Potential swaps (constrained):")
            for r in sorted(swappable, key=lambda x: -x["gap_tokens"]):
                lines.append(
                    f"    {r['eva']} : {r['current_name']} → {r['best_name']} (+{r['gap_tokens']} tokens)"
                )
        else:
            lines.append("  No beneficial swaps found under 1-to-1 constraint.")

    return "\n".join(lines)


# ── Main entry point ────────────────────────────────────────────


def run(config: ToolkitConfig, force: bool = False, lexicon_mode: str = "full"):
    """Run mapping audit.

    Args:
        lexicon_mode: "full" (491K, default) or "honest" (28K, no Sefaria-Corpus).
                      The honest mode eliminates the yod-frequency artefact from
                      the Sefaria bulk-corpus and gives more reliable gap_pct values.
    """
    suffix = "_honest" if lexicon_mode == "honest" else ""
    out_json = config.stats_dir / f"mapping_audit{suffix}.json"
    out_txt = config.stats_dir / f"mapping_audit{suffix}_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    mode_label = "HONEST (no Sefaria-Corpus)" if lexicon_mode == "honest" else "FULL (491K)"
    print_header(f"Mapping Audit — Per-letter Optimality [{mode_label}]")

    # ── Load ──
    print_step("Loading data...")
    eva_freqs, full_lexicon_set, form_to_gloss = load_data(config)

    if lexicon_mode == "honest":
        lexicon_set, _ = load_honest_lexicon(config)
    else:
        lexicon_set = full_lexicon_set

    print(f"      EVA words: {len(eva_freqs):,} types, {sum(eva_freqs.values()):,} tokens")
    print(f"      Lexicon: {len(lexicon_set):,} forms [{mode_label}]")

    # ── Baseline ──
    print_step("Computing baseline...")
    bt, bk, tt, tk = count_matches(
        eva_freqs, lexicon_set, mapping=FULL_MAPPING
    )
    baseline = {
        "matched_types": bt,
        "matched_tokens": bk,
        "total_types": tt,
        "total_tokens": tk,
        "match_rate_types": round(100 * bt / tt, 1) if tt else 0,
        "match_rate_tokens": round(100 * bk / tk, 1) if tk else 0,
    }
    print(f"      Baseline: {bt:,} types, {bk:,} tokens ({baseline['match_rate_tokens']}%)")

    # ── Audit each standard letter (17 in FULL_MAPPING) ──
    letter_results = []
    eva_chars = sorted(FULL_MAPPING.keys())
    for i, eva_char in enumerate(eva_chars):
        print_step(f"Auditing EVA '{eva_char}' ({i+1}/{len(eva_chars)})...")
        result = audit_letter(eva_char, eva_freqs, lexicon_set, FULL_MAPPING)
        letter_results.append(result)
        rank_str = f"#{result['current_rank']}"
        if result["current_rank"] > 1:
            rank_str += f" (best={result['best_name']}, +{result['gap_tokens']})"
        print(f"      {eva_char} → {result['current_name']}: {rank_str}")

    # ── Audit special elements ──
    print_step("Auditing special elements...")
    special_results = audit_special(eva_freqs, lexicon_set, FULL_MAPPING)

    for key in ["ch", "ii", "i", "d@init", "h@init"]:
        r = special_results[key]
        rank_str = f"#{r['current_rank']}"
        if r["current_rank"] > 1:
            rank_str += f" (best={r['best_name']}, +{r['gap_tokens']})"
        print(f"      {r['element']}: {rank_str}")

    # ── Constrained audit (1-to-1) ──
    print_step("Constrained audit (free Hebrew letters only)...")
    used = get_used_hebrew()
    free = set(HEBREW_CHARS) - used
    print(f"      Used: {len(used)} Hebrew letters. Free: {', '.join(sorted(free))}")
    constrained_results = audit_constrained(eva_freqs, lexicon_set, FULL_MAPPING)

    c_optimal = sum(1 for r in constrained_results if r["current_rank"] == 1)
    swappable = [r for r in constrained_results if r["gap_tokens"] > 0 and r["current_rank"] > 1]
    print(f"      Constrained: {c_optimal}/{len(constrained_results)} at rank #1")
    for r in sorted(swappable, key=lambda x: -x["gap_tokens"]):
        print(f"      SWAP? {r['eva']} : {r['current_name']} → {r['best_name']} (+{r['gap_tokens']})")

    # ── Output ──
    print_step("Writing output...")
    output = {
        "lexicon_mode": lexicon_mode,
        "lexicon_forms": len(lexicon_set),
        "baseline": baseline,
        "letter_audit": letter_results,
        "special_audit": special_results,
        "constrained_audit": constrained_results,
    }

    config.ensure_dirs()
    with open(out_json, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    summary = format_summary(baseline, letter_results, special_results, constrained_results)
    with open(out_txt, "w") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")

    optimal = sum(1 for r in letter_results if r["current_rank"] == 1)
    optimal += sum(
        1 for k in ["ch", "ii", "i", "d@init", "h@init"]
        if special_results[k]["current_rank"] == 1
    )
    total = len(letter_results) + 5
    print(f"\n      Unconstrained: {optimal}/{total} at rank #1")
    print(f"      Constrained:   {c_optimal}/{len(constrained_results)} at rank #1")
