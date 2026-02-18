"""Investigation: EVA t → qof swap.

The mapping audit found that swapping EVA 't' from tet(J) to qof(q)
gains +127 tokens. This module runs the differential test vs random
to confirm the swap is genuine signal, not noise.

Protocol (same as Phase 9 dual-role, allograph-lr-deep):
  1. Baseline: current mapping (t→tet)
  2. Proposed: swap t→qof
  3. N random permutations: for each, compute both t→tet and t→qof
  4. Differential = real_gain - mean(random_gain)
  5. z-score on differential
  6. Word-level analysis: which words gain/lose from the swap
"""

from __future__ import annotations

import json
import random as _random
from collections import Counter

import numpy as np

from .config import ToolkitConfig
from .full_decode import FULL_MAPPING, preprocess_eva
from .mapping_audit import (
    HEBREW_CHARS,
    II_HEBREW,
    I_HEBREW,
    CH_HEBREW,
    INITIAL_D_HEBREW,
    INITIAL_H_HEBREW,
    count_matches,
    decode_hebrew,
    load_data,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step

N_PERMUTATIONS = 1000
RNG_SEED = 42


def decode_all(eva_freqs, mapping, **kwargs):
    """Decode all EVA words, return {hebrew_word: (eva_word, freq)}."""
    result = {}
    for eva_word, freq in eva_freqs.items():
        heb = decode_hebrew(eva_word, mapping, **kwargs)
        if heb is not None:
            result[heb] = (eva_word, freq)
    return result


def word_level_analysis(eva_freqs, lexicon_set, form_to_gloss):
    """Compare t→tet vs t→qof at word level."""
    mapping_tet = dict(FULL_MAPPING)  # t→J (tet) is default
    mapping_qof = dict(FULL_MAPPING)
    mapping_qof["t"] = "q"  # swap to qof

    gained = []  # words that match with qof but not tet
    lost = []  # words that match with tet but not qof
    both = []  # words that match with both

    for eva_word, freq in eva_freqs.items():
        heb_tet = decode_hebrew(eva_word, mapping_tet)
        heb_qof = decode_hebrew(eva_word, mapping_qof)
        if heb_tet is None or heb_qof is None:
            continue

        in_tet = heb_tet in lexicon_set
        in_qof = heb_qof in lexicon_set

        if in_qof and not in_tet:
            gained.append({
                "eva": eva_word,
                "hebrew_tet": heb_tet,
                "hebrew_qof": heb_qof,
                "freq": freq,
                "gloss": form_to_gloss.get(heb_qof, ""),
            })
        elif in_tet and not in_qof:
            lost.append({
                "eva": eva_word,
                "hebrew_tet": heb_tet,
                "hebrew_qof": heb_qof,
                "freq": freq,
                "gloss": form_to_gloss.get(heb_tet, ""),
            })
        elif in_tet and in_qof:
            both.append({
                "eva": eva_word,
                "hebrew_tet": heb_tet,
                "hebrew_qof": heb_qof,
                "freq": freq,
                "gloss_tet": form_to_gloss.get(heb_tet, ""),
                "gloss_qof": form_to_gloss.get(heb_qof, ""),
            })

    gained.sort(key=lambda x: -x["freq"])
    lost.sort(key=lambda x: -x["freq"])
    both.sort(key=lambda x: -x["freq"])
    return gained, lost, both


def differential_test(eva_freqs, lexicon_set, n_perms=N_PERMUTATIONS):
    """Differential test: real gain vs random gain from t→qof swap."""
    rng = _random.Random(RNG_SEED)

    # Real mapping: baseline (t→tet) and proposed (t→qof)
    mapping_tet = dict(FULL_MAPPING)
    _, real_tokens_tet, _, _ = count_matches(
        eva_freqs, lexicon_set, mapping=mapping_tet
    )

    mapping_qof = dict(FULL_MAPPING)
    mapping_qof["t"] = "q"
    _, real_tokens_qof, _, _ = count_matches(
        eva_freqs, lexicon_set, mapping=mapping_qof
    )

    real_gain = real_tokens_qof - real_tokens_tet

    # Random permutations
    values = list(FULL_MAPPING.values())
    random_gains = []

    for i in range(n_perms):
        # Create a random mapping
        shuffled = values[:]
        rng.shuffle(shuffled)
        rand_mapping = dict(zip(sorted(FULL_MAPPING.keys()), shuffled))

        # Baseline with random mapping (t keeps whatever random assigned)
        _, rand_tokens_base, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=rand_mapping
        )

        # Swap t→qof in the random mapping
        rand_mapping_qof = dict(rand_mapping)
        rand_mapping_qof["t"] = "q"
        _, rand_tokens_qof, _, _ = count_matches(
            eva_freqs, lexicon_set, mapping=rand_mapping_qof
        )

        random_gains.append(rand_tokens_qof - rand_tokens_base)

    mean_random = float(np.mean(random_gains))
    std_random = float(np.std(random_gains)) or 1.0
    z_score = (real_gain - mean_random) / std_random
    p_value = sum(1 for g in random_gains if g >= real_gain) / n_perms

    return {
        "real_tokens_tet": real_tokens_tet,
        "real_tokens_qof": real_tokens_qof,
        "real_gain": real_gain,
        "random_gain_mean": round(mean_random, 1),
        "random_gain_std": round(std_random, 1),
        "differential": round(real_gain - mean_random, 1),
        "z_score": round(z_score, 2),
        "p_value": p_value,
        "n_perms": n_perms,
    }


def format_summary(diff_results, gained, lost, both):
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("  QOF INVESTIGATION — EVA t : tet → qof")
    lines.append("=" * 60)

    d = diff_results
    lines.append("\n── Differential Test ──")
    lines.append(f"  Baseline (t→tet):    {d['real_tokens_tet']:>7,} tokens")
    lines.append(f"  Proposed (t→qof):    {d['real_tokens_qof']:>7,} tokens")
    lines.append(f"  Real gain:           {d['real_gain']:>+7,} tokens")
    lines.append(f"  Random gain (mean):  {d['random_gain_mean']:>+7} ± {d['random_gain_std']}")
    lines.append(f"  Differential:        {d['differential']:>+7} tokens")
    lines.append(f"  z-score:             {d['z_score']:>7}")
    lines.append(f"  p-value:             {d['p_value']}")
    lines.append(f"  Permutations:        {d['n_perms']}")

    verdict = "CONFIRMED" if d["z_score"] > 2 and d["differential"] > 0 else "REJECTED"
    if d["z_score"] > 0 and d["z_score"] <= 2:
        verdict = "MARGINAL"
    lines.append(f"\n  Verdict: {verdict}")

    lines.append(f"\n── Word-level Impact ──")
    lines.append(f"  Gained (qof matches, tet doesn't): {len(gained)} types, "
                 f"{sum(w['freq'] for w in gained):,} tokens")
    lines.append(f"  Lost (tet matches, qof doesn't):   {len(lost)} types, "
                 f"{sum(w['freq'] for w in lost):,} tokens")
    lines.append(f"  Both match:                        {len(both)} types, "
                 f"{sum(w['freq'] for w in both):,} tokens")

    gained_glossed = [w for w in gained if w["gloss"] and "[attestato" not in w["gloss"]]
    lost_glossed = [w for w in lost if w["gloss"] and "[attestato" not in w["gloss"]]

    lines.append(f"\n  Gained with gloss: {len(gained_glossed)}")
    lines.append(f"  Lost with gloss:   {len(lost_glossed)}")

    if gained:
        lines.append(f"\n── Top Gained Words (t→qof) ──")
        lines.append(f"  {'EVA':>12s}  {'Tet':>8s}  {'Qof':>8s}  {'Freq':>5s}  Gloss")
        for w in gained[:20]:
            g = w["gloss"][:50] if w["gloss"] else ""
            lines.append(
                f"  {w['eva']:>12s}  {w['hebrew_tet']:>8s}  {w['hebrew_qof']:>8s}  "
                f"{w['freq']:>5d}  {g}"
            )

    if lost:
        lines.append(f"\n── Top Lost Words (tet no longer matches) ──")
        lines.append(f"  {'EVA':>12s}  {'Tet':>8s}  {'Qof':>8s}  {'Freq':>5s}  Gloss")
        for w in lost[:20]:
            g = w["gloss"][:50] if w["gloss"] else ""
            lines.append(
                f"  {w['eva']:>12s}  {w['hebrew_tet']:>8s}  {w['hebrew_qof']:>8s}  "
                f"{w['freq']:>5d}  {g}"
            )

    return "\n".join(lines)


def run(config: ToolkitConfig, force: bool = False):
    """Run qof investigation."""
    out_json = config.stats_dir / "qof_investigation.json"
    out_txt = config.stats_dir / "qof_investigation_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    print_header("Qof Investigation — EVA t : tet → qof")

    print_step("Loading data...")
    eva_freqs, lexicon_set, form_to_gloss = load_data(config)
    print(f"      EVA words: {len(eva_freqs):,}, lexicon: {len(lexicon_set):,}")

    print_step("Word-level analysis...")
    gained, lost, both = word_level_analysis(eva_freqs, lexicon_set, form_to_gloss)
    net_types = len(gained) - len(lost)
    net_tokens = sum(w["freq"] for w in gained) - sum(w["freq"] for w in lost)
    print(f"      Gained: {len(gained)} types, Lost: {len(lost)} types, Net: {net_types:+d} types, {net_tokens:+d} tokens")

    print_step(f"Differential test ({N_PERMUTATIONS} permutations)...")
    diff_results = differential_test(eva_freqs, lexicon_set)
    print(f"      Real gain: {diff_results['real_gain']:+d}, "
          f"Random gain: {diff_results['random_gain_mean']:+.0f} ± {diff_results['random_gain_std']:.0f}")
    print(f"      Differential: {diff_results['differential']:+.0f}, "
          f"z={diff_results['z_score']}, p={diff_results['p_value']}")

    verdict = "CONFIRMED" if diff_results["z_score"] > 2 and diff_results["differential"] > 0 else "REJECTED"
    if diff_results["z_score"] > 0 and diff_results["z_score"] <= 2:
        verdict = "MARGINAL"
    print(f"\n      Verdict: {verdict}")

    print_step("Writing output...")
    output = {
        "differential": diff_results,
        "gained_count": len(gained),
        "lost_count": len(lost),
        "both_count": len(both),
        "net_types": net_types,
        "net_tokens": net_tokens,
        "gained": gained[:50],
        "lost": lost[:50],
        "verdict": verdict,
    }

    config.ensure_dirs()
    with open(out_json, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    summary = format_summary(diff_results, gained, lost, both)
    with open(out_txt, "w") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")
