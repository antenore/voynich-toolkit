"""Cross-analysis with epilectrik/voynich token classifications.

Phase 19: Joe DiPrima's analysis (github.com/epilectrik/voynich) classifies
all EVA tokens into morphological components (PREFIX+MIDDLE+SUFFIX), roles
(PP/RI/INFRA/UNKNOWN), kernel operators (K/H/E), regimes (PRECISION/
HIGH_ENERGY/SETTLING), and flow states (INITIAL→TERMINAL).

This module decodes each of his 8,150 EVA token types through our Hebrew
cipher mapping and tests whether his classifications predict our Hebrew
match rate. If "kernel E" tokens match at different rates than "kernel K"
tokens, it reveals which structural components our mapping captures.

Data source: ../epilectrik-voynich/data/token_dictionary.json
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew
from .mapping_audit import load_honest_lexicon
from .utils import print_header, print_step


EPILECTRIK_ROOT = Path(__file__).resolve().parents[3] / "epilectrik-voynich"


# =====================================================================
# Data loading
# =====================================================================


def load_epilectrik_tokens(root: Path = EPILECTRIK_ROOT):
    """Load token dictionary from epilectrik/voynich repo.

    Returns list of dicts, each with:
        eva, total, morphology{articulator,prefix,middle,suffix},
        role, systems[], fl_state, is_label
    """
    token_path = root / "data" / "token_dictionary.json"
    if not token_path.exists():
        raise click.ClickException(
            f"epilectrik token dictionary not found: {token_path}\n"
            "  Clone the repo: git clone https://github.com/epilectrik/voynich.git epilectrik-voynich"
        )

    with open(token_path) as f:
        data = json.load(f)

    tokens = []
    for eva_word, entry in data["tokens"].items():
        morph = entry.get("morphology", {})
        dist = entry.get("distribution", {})
        role_info = entry.get("role", {})
        tokens.append({
            "eva": eva_word,
            "total": dist.get("total", 0),
            "prefix": morph.get("prefix"),
            "middle": morph.get("middle", ""),
            "suffix": morph.get("suffix"),
            "articulator": morph.get("articulator"),
            "role": role_info.get("primary", "UNKNOWN"),
            "systems": entry.get("systems", []),
            "fl_state": entry.get("fl_state"),
            "is_label": dist.get("is_label", False),
        })

    return tokens


def load_middle_dictionary(root: Path = EPILECTRIK_ROOT):
    """Load MIDDLE classifications (kernel, regime) from epilectrik repo."""
    mid_path = root / "data" / "middle_dictionary.json"
    if not mid_path.exists():
        return {}

    with open(mid_path) as f:
        data = json.load(f)

    middles = {}
    for middle_str, entry in data.get("middles", {}).items():
        middles[middle_str] = {
            "kernel": entry.get("kernel"),
            "regime": entry.get("regime"),
        }
    return middles


def load_macro_states(root: Path = EPILECTRIK_ROOT):
    """Load macro_state mapping (49 classes → 6 states) from decoder_maps."""
    maps_path = root / "data" / "decoder_maps.json"
    if not maps_path.exists():
        return {}

    with open(maps_path) as f:
        data = json.load(f)

    macro_map = {}
    ms = data.get("maps", {}).get("macro_state", {}).get("entries", {})
    for state_name, class_ids in ms.items():
        if isinstance(class_ids, list):
            for cid in class_ids:
                macro_map[str(cid)] = state_name
    return macro_map


# =====================================================================
# Decode and classify
# =====================================================================


def decode_and_classify(tokens, middle_dict, lexicon_full, lexicon_honest):
    """Decode each token through our mapping and check lexicon match.

    Returns list of dicts with classification + match results.
    """
    results = []
    for tok in tokens:
        eva = tok["eva"]
        hebrew = decode_to_hebrew(eva)

        if hebrew is None:
            match_full = False
            match_honest = False
        else:
            match_full = hebrew in lexicon_full
            match_honest = hebrew in lexicon_honest

        # Look up kernel/regime from middle dictionary
        mid = tok["middle"] or ""
        mid_info = middle_dict.get(mid, {})
        kernel = mid_info.get("kernel")
        regime = mid_info.get("regime")

        # Primary system (first listed, or multi)
        systems = tok["systems"]
        if len(systems) == 1:
            system = systems[0]
        elif len(systems) > 1:
            system = "multi"
        else:
            system = "unknown"

        results.append({
            "eva": eva,
            "hebrew": hebrew,
            "total": tok["total"],
            "match_full": match_full,
            "match_honest": match_honest,
            "role": tok["role"],
            "system": system,
            "kernel": kernel or "none",
            "regime": regime or "none",
            "fl_state": tok["fl_state"] or "none",
            "is_label": tok["is_label"],
            "prefix": tok["prefix"] or "BARE",
            "middle": mid,
            "suffix": tok["suffix"] or "BARE",
        })

    return results


# =====================================================================
# Statistical analysis
# =====================================================================


def analyze_axis(results, axis_name, lexicon_key="match_honest"):
    """Compute match rate by classification axis.

    Returns dict with per-category stats and chi-square test.
    """
    groups = defaultdict(lambda: {"matched": 0, "total": 0,
                                   "matched_tok": 0, "total_tok": 0})

    for r in results:
        cat = r[axis_name]
        freq = r["total"]
        matched = r[lexicon_key]
        cat = str(cat)  # handle booleans
        groups[cat]["total"] += 1
        groups[cat]["total_tok"] += freq
        if matched:
            groups[cat]["matched"] += 1
            groups[cat]["matched_tok"] += freq

    # Build contingency table for chi-square (types)
    categories = sorted(groups.keys())
    if len(categories) < 2:
        return {"categories": {}, "chi2": None, "p": None, "dof": None}

    contingency = []
    for cat in categories:
        g = groups[cat]
        contingency.append([g["matched"], g["total"] - g["matched"]])

    contingency = np.array(contingency)

    # Remove rows with zero total
    row_sums = contingency.sum(axis=1)
    mask = row_sums > 0
    contingency = contingency[mask]
    categories = [c for c, m in zip(categories, mask) if m]

    if len(categories) < 2 or contingency.min() < 0:
        chi2_val, p_val, dof = None, None, None
    else:
        try:
            chi2_val, p_val, dof, _ = chi2_contingency(contingency)
        except ValueError:
            chi2_val, p_val, dof = None, None, None

    cat_stats = {}
    for cat in categories:
        g = groups[cat]
        rate_types = g["matched"] / g["total"] if g["total"] > 0 else 0
        rate_tokens = g["matched_tok"] / g["total_tok"] if g["total_tok"] > 0 else 0
        cat_stats[cat] = {
            "types": g["total"],
            "matched_types": g["matched"],
            "rate_types": round(rate_types, 4),
            "tokens": g["total_tok"],
            "matched_tokens": g["matched_tok"],
            "rate_tokens": round(rate_tokens, 4),
        }

    return {
        "categories": cat_stats,
        "chi2": round(chi2_val, 2) if chi2_val is not None else None,
        "p": round(p_val, 6) if p_val is not None else None,
        "dof": dof,
    }


# =====================================================================
# Main runner
# =====================================================================


def run(config: ToolkitConfig, force: bool = False):
    """Run cross-analysis with epilectrik token classifications."""
    out_path = config.stats_dir / "cross_analysis_epilectrik.json"
    txt_path = config.stats_dir / "cross_analysis_epilectrik.txt"

    if out_path.exists() and not force:
        print(f"Output exists: {out_path} (use --force to re-run)")
        return

    print_header("Cross-Analysis: epilectrik/voynich Token Classifications")

    # Load data
    print_step("Loading epilectrik token dictionary")
    tokens = load_epilectrik_tokens()
    print(f"  Loaded {len(tokens)} token types")

    print_step("Loading epilectrik MIDDLE dictionary")
    middle_dict = load_middle_dictionary()
    print(f"  Loaded {len(middle_dict)} MIDDLE classifications")

    print_step("Loading Hebrew lexicons")
    with open(config.lexicon_dir / "lexicon_enriched.json") as f:
        lex_data = json.load(f)
    lexicon_full = set(lex_data["all_consonantal_forms"])

    lexicon_honest, _ = load_honest_lexicon(config)
    print(f"  Full: {len(lexicon_full):,} forms, Honest: {len(lexicon_honest):,} forms")

    # Decode and classify
    print_step("Decoding 8K tokens through Hebrew mapping")
    results = decode_and_classify(tokens, middle_dict, lexicon_full, lexicon_honest)

    decoded = sum(1 for r in results if r["hebrew"] is not None)
    matched_full = sum(1 for r in results if r["match_full"])
    matched_honest = sum(1 for r in results if r["match_honest"])
    print(f"  Decoded: {decoded}/{len(results)} ({100*decoded/len(results):.1f}%)")
    print(f"  Match full: {matched_full}/{decoded} ({100*matched_full/decoded:.1f}%)")
    print(f"  Match honest: {matched_honest}/{decoded} ({100*matched_honest/decoded:.1f}%)")

    # Analyze each classification axis
    axes = [
        ("role", "Role (PP/RI/INFRA/UNKNOWN)"),
        ("system", "Currier System (A/B/AZC)"),
        ("kernel", "Kernel Operator (K/H/E)"),
        ("regime", "Regime (PRECISION/HIGH_ENERGY/SETTLING)"),
        ("fl_state", "Flow State (INITIAL→TERMINAL)"),
        ("is_label", "Label vs Text"),
    ]

    axis_results = {}
    for axis_key, axis_label in axes:
        print_step(f"Analyzing: {axis_label}")
        for lex_label, lex_key in [("honest", "match_honest"), ("full", "match_full")]:
            analysis = analyze_axis(results, axis_key, lex_key)
            axis_results[f"{axis_key}_{lex_label}"] = analysis

            # Print summary
            if lex_label == "honest":
                chi2_str = f"chi2={analysis['chi2']}, p={analysis['p']}" if analysis['chi2'] else "N/A"
                print(f"  [{lex_label}] {chi2_str}")
                for cat, stats in sorted(analysis["categories"].items(),
                                         key=lambda x: -x[1]["rate_types"]):
                    print(f"    {cat:20s}: {stats['rate_types']*100:5.1f}% "
                          f"({stats['matched_types']}/{stats['types']} types, "
                          f"{stats['rate_tokens']*100:.1f}% of {stats['tokens']} tok)")

    # Length confound check: match rate by EVA word length
    print_step("Length confound check")
    len_groups = defaultdict(lambda: {"matched": 0, "total": 0})
    kernel_lengths = defaultdict(list)
    for r in results:
        if r["hebrew"] is None:
            continue
        eva_len = len(r["eva"])
        len_groups[eva_len]["total"] += 1
        if r["match_honest"]:
            len_groups[eva_len]["matched"] += 1
        kernel_lengths[r["kernel"]].append(eva_len)

    print("  EVA length → honest match rate:")
    length_stats = {}
    for length in sorted(len_groups.keys()):
        g = len_groups[length]
        rate = g["matched"] / g["total"] if g["total"] > 0 else 0
        length_stats[str(length)] = {
            "types": g["total"], "matched": g["matched"],
            "rate": round(rate, 4),
        }
        if g["total"] >= 20:
            print(f"    len={length:2d}: {rate*100:5.1f}% ({g['matched']}/{g['total']})")

    print("\n  Mean EVA length by kernel:")
    kernel_mean_len = {}
    for k in sorted(kernel_lengths.keys()):
        vals = kernel_lengths[k]
        mean_l = np.mean(vals)
        kernel_mean_len[k] = round(float(mean_l), 2)
        print(f"    {k:6s}: {mean_l:.2f} chars ({len(vals)} types)")

    # Length-controlled kernel analysis: match rate within 5-6 char tokens
    print_step("Length-controlled kernel test (5-6 char EVA tokens)")
    for target_len in [4, 5, 6]:
        subset = [r for r in results
                  if r["hebrew"] is not None and len(r["eva"]) == target_len]
        if not subset:
            continue
        kern_sub = defaultdict(lambda: {"m": 0, "t": 0})
        for r in subset:
            kern_sub[r["kernel"]]["t"] += 1
            if r["match_honest"]:
                kern_sub[r["kernel"]]["m"] += 1
        print(f"  len={target_len} ({len(subset)} types):")
        for k in sorted(kern_sub.keys()):
            g = kern_sub[k]
            rate = g["m"] / g["t"] if g["t"] > 0 else 0
            if g["t"] >= 10:
                print(f"    {k:6s}: {rate*100:5.1f}% ({g['m']}/{g['t']})")

    # Save JSON
    output = {
        "n_tokens": len(tokens),
        "n_decoded": decoded,
        "match_full_types": matched_full,
        "match_honest_types": matched_honest,
        "overall_rate_full": round(matched_full / decoded, 4) if decoded else 0,
        "overall_rate_honest": round(matched_honest / decoded, 4) if decoded else 0,
        "axes": axis_results,
        "length_stats": length_stats,
        "kernel_mean_length": kernel_mean_len,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  JSON: {out_path}")

    # Save TXT summary
    lines = ["Cross-Analysis: epilectrik/voynich Token Classifications",
             "=" * 60, ""]
    lines.append(f"Token types: {len(tokens)}")
    lines.append(f"Decoded:     {decoded} ({100*decoded/len(tokens):.1f}%)")
    lines.append(f"Match full:  {matched_full} ({100*matched_full/decoded:.1f}%)")
    lines.append(f"Match honest: {matched_honest} ({100*matched_honest/decoded:.1f}%)")
    lines.append("")

    for axis_key, axis_label in axes:
        lines.append(f"\n--- {axis_label} ---")
        for lex_label in ["honest", "full"]:
            key = f"{axis_key}_{lex_label}"
            analysis = axis_results[key]
            chi2_str = f"chi2={analysis['chi2']}, p={analysis['p']}" if analysis['chi2'] else "N/A"
            lines.append(f"  [{lex_label}] {chi2_str}")
            for cat, stats in sorted(analysis["categories"].items(),
                                     key=lambda x: -x[1]["rate_types"]):
                lines.append(f"    {cat:20s}: {stats['rate_types']*100:5.1f}% "
                             f"({stats['matched_types']}/{stats['types']} types)")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  TXT: {txt_path}")


if __name__ == "__main__":
    run(ToolkitConfig())
