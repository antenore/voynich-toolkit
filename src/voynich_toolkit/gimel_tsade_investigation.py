"""Investigation: EVA m → tsade swap.

Phase 15 P2b follow-up: the honest mapping audit (no Sefaria-Corpus) found
that EVA 'm' currently maps to gimel (g), but tsade (C) gives +118 more
matched tokens. Gimel ranked 18th/22 in the unconstrained audit and dead
last (4th/4) in the constrained audit.

This module runs a formal differential test to determine whether the
m→tsade improvement is genuine signal or noise.

Protocol:
  1. Honest lexicon (45K forms, no Sefaria-Corpus bulk forms)
  2. Word-level analysis: gained/lost/both when m→gimel vs m→tsade
  3. Differential permutation test (1000 perms):
       real_gain = tokens(m→tsade) − tokens(m→gimel)
       random_gain = mean gain from shuffling any one mapping slot
       z = (real_gain − random_gain_mean) / random_gain_std
  4. Joint swap test: m→tsade + t→qof together
  5. Full-lexicon comparison (for context vs previous audits)
  6. Verdict: CONFIRMED / MARGINAL / REJECTED

Outputs:
  gimel_tsade_investigation.json
  gimel_tsade_investigation_summary.txt
"""

from __future__ import annotations

import json
import random as _random
from collections import Counter

import numpy as np

from .config import ToolkitConfig
from .full_decode import FULL_MAPPING
from .mapping_audit import (
    HEBREW_CHARS,
    CH_HEBREW,
    II_HEBREW,
    I_HEBREW,
    INITIAL_D_HEBREW,
    INITIAL_H_HEBREW,
    count_matches,
    decode_hebrew,
    load_data,
    load_honest_lexicon,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step

N_PERMUTATIONS = 1000
RNG_SEED = 42

# The swap under investigation
EVA_CHAR = "m"
CURRENT_HEBREW = "g"   # gimel
PROPOSED_HEBREW = "C"  # tsade


# =====================================================================
# Word-level analysis
# =====================================================================

def word_level_analysis(eva_freqs, lexicon_set, form_to_gloss,
                        current=CURRENT_HEBREW, proposed=PROPOSED_HEBREW):
    """Compare m→gimel vs m→tsade at word level.

    Returns (gained, lost, both) sorted by frequency descending.
    """
    mapping_current = dict(FULL_MAPPING)   # m→g (gimel)
    mapping_proposed = dict(FULL_MAPPING)
    mapping_proposed[EVA_CHAR] = proposed  # m→C (tsade)

    gained, lost, both, unchanged = [], [], [], []

    for eva_word, freq in eva_freqs.items():
        heb_cur = decode_hebrew(eva_word, mapping_current)
        heb_pro = decode_hebrew(eva_word, mapping_proposed)
        if heb_cur is None or heb_pro is None:
            continue

        in_cur = heb_cur in lexicon_set
        in_pro = heb_pro in lexicon_set

        entry = {
            "eva": eva_word,
            "hebrew_gimel": heb_cur,
            "hebrew_tsade": heb_pro,
            "freq": freq,
        }

        if in_pro and not in_cur:
            entry["gloss"] = form_to_gloss.get(heb_pro, "")
            gained.append(entry)
        elif in_cur and not in_pro:
            entry["gloss"] = form_to_gloss.get(heb_cur, "")
            lost.append(entry)
        elif in_cur and in_pro:
            entry["gloss_gimel"] = form_to_gloss.get(heb_cur, "")
            entry["gloss_tsade"] = form_to_gloss.get(heb_pro, "")
            both.append(entry)

    for lst in (gained, lost, both):
        lst.sort(key=lambda x: -x["freq"])

    return gained, lost, both


# =====================================================================
# Differential permutation test
# =====================================================================

def differential_test(eva_freqs, lexicon_set, n_perms=N_PERMUTATIONS,
                      proposed=PROPOSED_HEBREW):
    """Differential test: real gain from m→tsade vs random gain from same swap.

    For each permutation, shuffles the existing Hebrew assignments among
    EVA chars, then measures the gain from swapping slot 'm' to tsade.
    Compares the real differential against the random distribution.
    """
    rng = _random.Random(RNG_SEED)

    # Real baseline (m→gimel)
    mapping_cur = dict(FULL_MAPPING)
    _, real_tokens_cur, _, _ = count_matches(eva_freqs, lexicon_set, mapping=mapping_cur)

    # Real proposed (m→tsade)
    mapping_pro = dict(FULL_MAPPING)
    mapping_pro[EVA_CHAR] = proposed
    _, real_tokens_pro, _, _ = count_matches(eva_freqs, lexicon_set, mapping=mapping_pro)

    real_gain = real_tokens_pro - real_tokens_cur

    # Random permutations: shuffle the Hebrew assignments, then measure gain
    # from forcing slot 'm' to proposed Hebrew
    values = list(FULL_MAPPING.values())
    random_gains = []

    for _ in range(n_perms):
        shuffled = values[:]
        rng.shuffle(shuffled)
        rand_mapping = dict(zip(sorted(FULL_MAPPING.keys()), shuffled))

        _, rand_tokens_base, _, _ = count_matches(eva_freqs, lexicon_set, mapping=rand_mapping)

        rand_mapping_pro = dict(rand_mapping)
        rand_mapping_pro[EVA_CHAR] = proposed
        _, rand_tokens_pro, _, _ = count_matches(eva_freqs, lexicon_set, mapping=rand_mapping_pro)

        random_gains.append(rand_tokens_pro - rand_tokens_base)

    arr = np.array(random_gains, dtype=float)
    mean_random = float(arr.mean())
    std_random = float(arr.std()) or 1.0
    z_score = (real_gain - mean_random) / std_random
    p_value = float(np.sum(arr >= real_gain)) / n_perms

    return {
        "real_tokens_current": real_tokens_cur,
        "real_tokens_proposed": real_tokens_pro,
        "real_gain": real_gain,
        "random_gain_mean": round(mean_random, 1),
        "random_gain_std": round(std_random, 1),
        "differential": round(real_gain - mean_random, 1),
        "z_score": round(z_score, 2),
        "p_value": round(p_value, 4),
        "n_perms": n_perms,
    }


# =====================================================================
# Joint swap test: m→tsade + t→qof together
# =====================================================================

def joint_swap_test(eva_freqs, lexicon_set):
    """Test the combined swap m→tsade AND t→qof vs baseline.

    Returns a comparison of 4 mapping variants:
      baseline:   m→gimel, t→tet
      m_only:     m→tsade, t→tet
      t_only:     m→gimel, t→qof
      joint:      m→tsade, t→qof
    """
    results = {}
    variants = {
        "baseline":  {EVA_CHAR: CURRENT_HEBREW, "t": "J"},  # gimel, tet
        "m_tsade":   {EVA_CHAR: PROPOSED_HEBREW, "t": "J"}, # tsade, tet
        "t_qof":     {EVA_CHAR: CURRENT_HEBREW,  "t": "q"}, # gimel, qof
        "joint":     {EVA_CHAR: PROPOSED_HEBREW, "t": "q"}, # tsade, qof
    }

    for label, overrides in variants.items():
        mapping = dict(FULL_MAPPING)
        mapping.update(overrides)
        mt, mk, tt, tk = count_matches(eva_freqs, lexicon_set, mapping=mapping)
        results[label] = {
            "matched_types": mt,
            "matched_tokens": mk,
            "total_types": tt,
            "total_tokens": tk,
            "match_rate_tokens": round(100 * mk / tk, 2) if tk else 0,
        }

    baseline_tok = results["baseline"]["matched_tokens"]
    for label, r in results.items():
        r["gain_vs_baseline"] = r["matched_tokens"] - baseline_tok

    return results


# =====================================================================
# Gimel rarity analysis
# =====================================================================

def gimel_rarity_analysis(lexicon_set):
    """How often does gimel appear in the Hebrew lexicon?

    Reports: fraction of lexicon forms containing gimel (g),
    vs tsade (C), for context on why gimel matches poorly.
    """
    counts = Counter()
    for form in lexicon_set:
        for ch in set(form):
            counts[ch] += 1

    n_forms = len(lexicon_set)
    stats = {}
    for ch in ["g", "C", "q", "z", "y", "n", "m", "r", "h", "k"]:
        name = CONSONANT_NAMES.get(ch, ch)
        stats[ch] = {
            "name": name,
            "forms_containing": counts[ch],
            "pct_of_lexicon": round(100 * counts[ch] / n_forms, 1),
        }

    return stats, n_forms


# =====================================================================
# Formatting
# =====================================================================

def _verdict(diff):
    """Derive verdict from differential test results."""
    z = diff["z_score"]
    d = diff["differential"]
    if z > 2.0 and d > 0:
        return "CONFIRMED"
    if z > 1.0 and d > 0:
        return "MARGINAL"
    return "REJECTED"


def format_summary(diff_honest, diff_full, gained, lost, both,
                   joint, rarity_stats, n_lex_honest, n_lex_full):
    lines = []
    lines.append("=" * 64)
    lines.append("  GIMEL→TSADE INVESTIGATION — EVA m : gimel → tsade")
    lines.append("=" * 64)

    for label, diff, n_lex in [
        ("HONEST (45K, no Sefaria-Corpus)", diff_honest, n_lex_honest),
        ("FULL   (491K)",                   diff_full,   n_lex_full),
    ]:
        v = _verdict(diff)
        lines.append(f"\n── Differential Test [{label}] ──")
        lines.append(f"  Lexicon forms:       {n_lex:>10,}")
        lines.append(f"  Baseline (m→gimel):  {diff['real_tokens_current']:>10,} tokens")
        lines.append(f"  Proposed (m→tsade):  {diff['real_tokens_proposed']:>10,} tokens")
        lines.append(f"  Real gain:           {diff['real_gain']:>+10,} tokens")
        lines.append(f"  Random gain (mean):  {diff['random_gain_mean']:>+10.1f} ± {diff['random_gain_std']:.1f}")
        lines.append(f"  Differential:        {diff['differential']:>+10.1f} tokens")
        lines.append(f"  z-score:             {diff['z_score']:>10.2f}")
        lines.append(f"  p-value:             {diff['p_value']:>10.4f}")
        lines.append(f"  Verdict:             {v}")

    # Word-level
    lines.append(f"\n── Word-level Impact (honest lexicon) ──")
    n_gained_tok = sum(w["freq"] for w in gained)
    n_lost_tok = sum(w["freq"] for w in lost)
    n_both_tok = sum(w["freq"] for w in both)
    lines.append(f"  Gained (tsade matches, gimel doesn't): "
                 f"{len(gained):>4} types, {n_gained_tok:>6,} tokens")
    lines.append(f"  Lost   (gimel matches, tsade doesn't): "
                 f"{len(lost):>4} types, {n_lost_tok:>6,} tokens")
    lines.append(f"  Both match:                            "
                 f"{len(both):>4} types, {n_both_tok:>6,} tokens")
    lines.append(f"  Net:                                   "
                 f"{len(gained)-len(lost):>+4} types, {n_gained_tok-n_lost_tok:>+6,} tokens")

    if gained:
        lines.append(f"\n── Top Gained Words (tsade matches, gimel doesn't) ──")
        lines.append(f"  {'EVA':>12s}  {'Gimel':>10s}  {'Tsade':>10s}  {'Freq':>5s}  Gloss")
        for w in gained[:20]:
            g = (w.get("gloss") or "")[:45]
            lines.append(f"  {w['eva']:>12s}  {w['hebrew_gimel']:>10s}  "
                         f"{w['hebrew_tsade']:>10s}  {w['freq']:>5d}  {g}")

    if lost:
        lines.append(f"\n── Top Lost Words (gimel matches, tsade doesn't) ──")
        lines.append(f"  {'EVA':>12s}  {'Gimel':>10s}  {'Tsade':>10s}  {'Freq':>5s}  Gloss")
        for w in lost[:15]:
            g = (w.get("gloss") or "")[:45]
            lines.append(f"  {w['eva']:>12s}  {w['hebrew_gimel']:>10s}  "
                         f"{w['hebrew_tsade']:>10s}  {w['freq']:>5d}  {g}")

    # Joint swap
    lines.append(f"\n── Joint Swap Test: m→tsade + t→qof ──")
    lines.append(f"  {'Variant':12s}  {'Tokens':>8s}  {'Gain':>7s}  {'Rate%':>6s}")
    for label in ("baseline", "m_tsade", "t_qof", "joint"):
        r = joint[label]
        lines.append(f"  {label:12s}  {r['matched_tokens']:>8,}  "
                     f"{r['gain_vs_baseline']:>+7,}  {r['match_rate_tokens']:>6.2f}%")

    # Gimel rarity
    lines.append(f"\n── Gimel Rarity in Hebrew Lexicon (honest, {n_lex_honest:,} forms) ──")
    lines.append(f"  {'Letter':>6s}  {'Name':>12s}  {'Forms':>8s}  {'% lex':>6s}")
    for ch, info in sorted(rarity_stats.items(),
                           key=lambda x: -x[1]["pct_of_lexicon"]):
        lines.append(f"  {ch:>6s}  {info['name']:>12s}  "
                     f"{info['forms_containing']:>8,}  {info['pct_of_lexicon']:>6.1f}%")

    # Overall verdict
    v_honest = _verdict(diff_honest)
    v_full = _verdict(diff_full)
    lines.append(f"\n{'=' * 64}")
    lines.append(f"  VERDICT (honest): {v_honest}")
    lines.append(f"  VERDICT (full):   {v_full}")
    if v_honest == "CONFIRMED":
        lines.append("  → m→tsade swap supported by honest lexicon test.")
        lines.append("    Gimel is rare in Hebrew; tsade provides more matches.")
    elif v_honest == "MARGINAL":
        lines.append("  → m→tsade shows marginal improvement; not conclusive.")
    else:
        lines.append("  → m→tsade swap not supported; gimel assignment retained.")
    lines.append("=" * 64)

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False):
    """Run gimel→tsade investigation."""
    out_json = config.stats_dir / "gimel_tsade_investigation.json"
    out_txt = config.stats_dir / "gimel_tsade_investigation_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    print_header("Gimel→Tsade Investigation — EVA m : gimel(g) → tsade(C)")

    # Load data
    print_step("Loading EVA corpus and lexicons...")
    eva_freqs, full_lex_set, form_to_gloss = load_data(config)
    honest_lex_set, _ = load_honest_lexicon(config)
    print(f"      EVA types: {len(eva_freqs):,}, tokens: {sum(eva_freqs.values()):,}")
    print(f"      Full lexicon: {len(full_lex_set):,} forms")
    print(f"      Honest lexicon: {len(honest_lex_set):,} forms")

    # Word-level analysis (honest lexicon)
    print_step("Word-level analysis (honest lexicon)...")
    gained, lost, both = word_level_analysis(eva_freqs, honest_lex_set, form_to_gloss)
    n_g = sum(w["freq"] for w in gained)
    n_l = sum(w["freq"] for w in lost)
    print(f"      Gained: {len(gained)} types ({n_g:,} tok) | "
          f"Lost: {len(lost)} types ({n_l:,} tok) | "
          f"Net: {len(gained)-len(lost):+d} types, {n_g-n_l:+,} tok")

    # Differential test — honest lexicon (primary)
    print_step(f"Differential test — HONEST lexicon ({N_PERMUTATIONS} perms)...")
    diff_honest = differential_test(eva_freqs, honest_lex_set)
    v_h = _verdict(diff_honest)
    print(f"      Real gain: {diff_honest['real_gain']:+d}  "
          f"Random: {diff_honest['random_gain_mean']:+.1f}±{diff_honest['random_gain_std']:.1f}  "
          f"Differential: {diff_honest['differential']:+.1f}  "
          f"z={diff_honest['z_score']}  p={diff_honest['p_value']}  → {v_h}")

    # Differential test — full lexicon (for comparison)
    print_step(f"Differential test — FULL lexicon ({N_PERMUTATIONS} perms)...")
    diff_full = differential_test(eva_freqs, full_lex_set)
    v_f = _verdict(diff_full)
    print(f"      Real gain: {diff_full['real_gain']:+d}  "
          f"Random: {diff_full['random_gain_mean']:+.1f}±{diff_full['random_gain_std']:.1f}  "
          f"Differential: {diff_full['differential']:+.1f}  "
          f"z={diff_full['z_score']}  p={diff_full['p_value']}  → {v_f}")

    # Joint swap test
    print_step("Joint swap test (m→tsade + t→qof)...")
    joint_honest = joint_swap_test(eva_freqs, honest_lex_set)
    joint_full = joint_swap_test(eva_freqs, full_lex_set)
    print(f"      [honest] baseline={joint_honest['baseline']['matched_tokens']:,}  "
          f"m_tsade={joint_honest['m_tsade']['gain_vs_baseline']:+,}  "
          f"t_qof={joint_honest['t_qof']['gain_vs_baseline']:+,}  "
          f"joint={joint_honest['joint']['gain_vs_baseline']:+,}")
    print(f"      [full]   baseline={joint_full['baseline']['matched_tokens']:,}  "
          f"m_tsade={joint_full['m_tsade']['gain_vs_baseline']:+,}  "
          f"t_qof={joint_full['t_qof']['gain_vs_baseline']:+,}  "
          f"joint={joint_full['joint']['gain_vs_baseline']:+,}")

    # Gimel rarity
    print_step("Gimel rarity analysis...")
    rarity_stats, _ = gimel_rarity_analysis(honest_lex_set)
    g_pct = rarity_stats["g"]["pct_of_lexicon"]
    C_pct = rarity_stats["C"]["pct_of_lexicon"]
    print(f"      Gimel in lexicon: {g_pct}% of forms | Tsade: {C_pct}%")

    # Save
    print_step("Writing output...")
    output = {
        "eva_char": EVA_CHAR,
        "current_hebrew": CURRENT_HEBREW,
        "current_name": CONSONANT_NAMES.get(CURRENT_HEBREW, "?"),
        "proposed_hebrew": PROPOSED_HEBREW,
        "proposed_name": CONSONANT_NAMES.get(PROPOSED_HEBREW, "?"),
        "differential_honest": diff_honest,
        "differential_full": diff_full,
        "verdict_honest": v_h,
        "verdict_full": v_f,
        "word_analysis": {
            "gained_types": len(gained),
            "gained_tokens": n_g,
            "lost_types": len(lost),
            "lost_tokens": n_l,
            "net_types": len(gained) - len(lost),
            "net_tokens": n_g - n_l,
            "gained": gained[:50],
            "lost": lost[:30],
        },
        "joint_swap_honest": joint_honest,
        "joint_swap_full": joint_full,
        "gimel_rarity": rarity_stats,
    }

    config.ensure_dirs()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    summary = format_summary(
        diff_honest, diff_full, gained, lost, both,
        joint_honest, rarity_stats, len(honest_lex_set), len(full_lex_set),
    )
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")
    print(f"\n      Verdict (honest): {v_h}")
    print(f"      Verdict (full):   {v_f}")
