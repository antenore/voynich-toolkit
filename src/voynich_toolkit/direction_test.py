"""
Direction test: RTL vs LTR reading direction hypothesis test.

Phase 15 P2a: The current mapping assumes RTL (right-to-left) reading,
following Hebrew convention. This module formally tests whether RTL
significantly outperforms LTR using:

  1. Direct comparison: RTL vs LTR match rates (two-proportion z-test)
  2. Permutation test per direction: random mappings × both directions
     → z_RTL_perm vs z_LTR_perm above random
  3. Per-Currier-language breakdown (A and B)
  4. Verdict: which direction has the stronger, more significant signal

Key question: Is RTL genuinely better, or is the signal direction-agnostic
(which would suggest the structure arises from EVA positional regularities
rather than a true directional cipher)?

Outputs:
  direction_test.json    — full report
  direction_test.txt     — human-readable summary
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import norm

from .config import ToolkitConfig
from .full_decode import (
    CH_HEBREW,
    FULL_MAPPING,
    I_HEBREW,
    II_HEBREW,
    INITIAL_D_HEBREW,
    INITIAL_H_HEBREW,
    preprocess_eva,
)
from .permutation_stats import HEBREW_LETTERS
from .utils import print_header, print_step
from .word_structure import parse_eva_words


MIN_LEN = 3
N_PERMS = 500   # per direction — ~2 min total
SEED = 42


# =====================================================================
# Full mapping including preprocessed-char placeholders
# =====================================================================

def _build_full_mapping(base_mapping):
    """Augment a 17-char mapping with the 3 placeholder chars."""
    aug = dict(base_mapping)
    aug.setdefault("\x03", CH_HEBREW)   # ch → kaf
    aug.setdefault("\x01", II_HEBREW)   # ii → he
    aug.setdefault("\x02", I_HEBREW)    # i  → resh
    return aug


REAL_FULL_MAPPING = _build_full_mapping(FULL_MAPPING)

# EVA chars present in the real full mapping (excluding placeholders)
_REAL_EVA_KEYS = [k for k in REAL_FULL_MAPPING if len(k) == 1 and k.isprintable()]


# =====================================================================
# Preprocessing + scoring (optimised for permutation loop)
# =====================================================================

def _preprocess_corpus(eva_words, min_len=MIN_LEN):
    """Pre-process EVA words once; returns list of processed char lists.

    Each element is a list of chars (including placeholders \\x01–\\x03).
    Avoids repeating the preprocessing step for every permutation.
    """
    preprocessed = []
    for word in eva_words:
        if len(word) < min_len:
            continue
        _, proc = preprocess_eva(word)
        if not proc:
            continue
        preprocessed.append(list(proc))
    return preprocessed


def _score(preprocessed, mapping, lexicon_set, direction):
    """Compute lexicon match rate for a given mapping and direction.

    Args:
        preprocessed: list of char lists from _preprocess_corpus()
        mapping: dict {char: hebrew_char} including placeholders
        lexicon_set: set of Hebrew consonantal forms
        direction: 'rtl' or 'ltr'

    Returns: float match rate in [0, 1]
    """
    n_total = 0
    n_matched = 0

    for chars in preprocessed:
        seq = list(reversed(chars)) if direction == "rtl" else list(chars)
        parts = []
        ok = True
        for ch in seq:
            h = mapping.get(ch)
            if h is None:
                ok = False
                break
            parts.append(h)
        if not ok or not parts:
            continue

        n_total += 1

        # Positional splits on the first decoded Hebrew char
        if parts[0] == "d":
            parts[0] = INITIAL_D_HEBREW   # dalet → bet at Hebrew-initial
        elif parts[0] == "h":
            parts[0] = INITIAL_H_HEBREW   # he → samekh at Hebrew-initial

        if "".join(parts) in lexicon_set:
            n_matched += 1

    return n_matched / max(n_total, 1)


# =====================================================================
# Random mapping generator (bijective, same keys as real mapping)
# =====================================================================

def _random_mapping(rng):
    """Generate a random bijective mapping with the same keys as the real mapping."""
    keys = list(REAL_FULL_MAPPING.keys())
    chosen = rng.sample(HEBREW_LETTERS, len(keys))
    return dict(zip(keys, chosen))


# =====================================================================
# Core comparison function
# =====================================================================

def _analyse_corpus(label, preprocessed, lexicon_set, n_perms, rng):
    """Run RTL vs LTR analysis + permutation tests for a corpus subset.

    Returns a result dict with:
      - rtl/ltr match rates
      - direct z-test
      - permutation z-scores for each direction
      - verdict
    """
    rtl_real = _score(preprocessed, REAL_FULL_MAPPING, lexicon_set, "rtl")
    ltr_real = _score(preprocessed, REAL_FULL_MAPPING, lexicon_set, "ltr")

    n_total = len(preprocessed)

    # Approximate token counts (the scorer skips unmapped chars)
    rtl_n = sum(1 for c in preprocessed
                if all(REAL_FULL_MAPPING.get(x) for x in c))
    ltr_n = rtl_n  # same words, same mapping → same denominator

    rtl_matched = int(round(rtl_real * rtl_n))
    ltr_matched = int(round(ltr_real * ltr_n))

    # Two-proportion z-test: RTL vs LTR (one-tailed, H1: RTL > LTR)
    p_pool = (rtl_matched + ltr_matched) / max(2 * rtl_n, 1)
    se = (p_pool * (1 - p_pool) * 2 / max(rtl_n, 1)) ** 0.5
    z_direct = (rtl_real - ltr_real) / se if se > 0 else 0.0
    p_direct = float(1 - norm.cdf(z_direct))

    # Permutation tests for each direction
    rtl_rand_scores = []
    ltr_rand_scores = []

    t0 = time.time()
    for _ in range(n_perms):
        rand_map = _random_mapping(rng)
        rtl_rand_scores.append(_score(preprocessed, rand_map, lexicon_set, "rtl"))
        ltr_rand_scores.append(_score(preprocessed, rand_map, lexicon_set, "ltr"))

    elapsed = time.time() - t0

    rtl_arr = np.array(rtl_rand_scores)
    ltr_arr = np.array(ltr_rand_scores)

    def _perm_stats(real, arr):
        mean = float(arr.mean())
        std = float(arr.std())
        z = (real - mean) / std if std > 0 else 0.0
        n_ge = int(np.sum(arr >= real))
        p = (n_ge + 1) / (n_perms + 1)
        return {
            "real_rate": round(real, 4),
            "random_mean": round(mean, 4),
            "random_std": round(std, 4),
            "z_score": round(z, 2),
            "p_value": round(p, 6),
            "n_above_real": n_ge,
        }

    rtl_perm = _perm_stats(rtl_real, rtl_arr)
    ltr_perm = _perm_stats(ltr_real, ltr_arr)

    # Verdict for this corpus
    z_rtl = rtl_perm["z_score"]
    z_ltr = ltr_perm["z_score"]
    diff = rtl_real - ltr_real

    if diff > 0.01 and z_direct > 3:
        verdict = "RTL_FAVORED_STRONG"
    elif diff > 0 and z_direct > 1.5:
        verdict = "RTL_FAVORED"
    elif diff < -0.01 and z_direct < -3:
        verdict = "LTR_FAVORED_STRONG"
    elif diff < 0 and z_direct < -1.5:
        verdict = "LTR_FAVORED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "label": label,
        "n_preprocessed_words": n_total,
        "rtl": {
            "real_rate": round(rtl_real, 4),
            "perm": rtl_perm,
        },
        "ltr": {
            "real_rate": round(ltr_real, 4),
            "perm": ltr_perm,
        },
        "direct_comparison": {
            "diff_rtl_minus_ltr": round(diff, 4),
            "diff_pp": round(diff * 100, 2),
            "z_score": round(z_direct, 2),
            "p_value": round(p_direct, 6),
            "rtl_wins": bool(rtl_real > ltr_real),
        },
        "perm_z_gap": round(z_rtl - z_ltr, 2),
        "elapsed_s": round(elapsed, 1),
        "n_perms": n_perms,
        "verdict": verdict,
    }


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Direction test: RTL vs LTR reading direction."""
    report_path = config.stats_dir / "direction_test.json"
    txt_path = config.stats_dir / "direction_test.txt"

    if report_path.exists() and not force:
        click.echo("  Direction test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 15 P2a — Direction Test: RTL vs LTR")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words on {len(pages)} pages")

    # 2. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    base_path = config.lexicon_dir / "lexicon.json"
    if enriched_path.exists():
        with open(enriched_path) as f:
            hlex = json.load(f)
        lex_label = "enriched"
    elif base_path.exists():
        with open(base_path) as f:
            hlex = json.load(f)
        lex_label = "base"
    else:
        raise click.ClickException("No Hebrew lexicon. Run: voynich prepare-lexicon")

    lexicon_set = set(hlex["all_consonantal_forms"])
    click.echo(f"    {len(lexicon_set):,} forms ({lex_label})")

    # 3. Split corpus by Currier language
    print_step("Splitting corpus by Currier language...")
    from .currier_split import split_corpus_by_language
    split = split_corpus_by_language(pages)
    for lang in ("A", "B"):
        d = split[lang]
        click.echo(f"    Language {lang}: {d['n_pages']} pages, "
                   f"{len(d['words']):,} words")
    click.echo(f"    Discarded (no language tag): {split['_discarded']} pages")

    # 4. Pre-process corpora
    print_step("Pre-processing word lists...")
    all_words = [w for p in pages for w in p["words"]]
    full_pp = _preprocess_corpus(all_words)
    a_pp = _preprocess_corpus(split["A"]["words"])
    b_pp = _preprocess_corpus(split["B"]["words"])
    click.echo(f"    Full corpus: {len(full_pp):,} words (len>={MIN_LEN}, mappable)")
    click.echo(f"    Language A:  {len(a_pp):,} words")
    click.echo(f"    Language B:  {len(b_pp):,} words")

    # 5. Quick RTL vs LTR preview (no perms)
    print_step("Quick comparison (no permutations)...")
    rtl_quick = _score(full_pp, REAL_FULL_MAPPING, lexicon_set, "rtl")
    ltr_quick = _score(full_pp, REAL_FULL_MAPPING, lexicon_set, "ltr")
    click.echo(f"    RTL match rate: {rtl_quick*100:.2f}%")
    click.echo(f"    LTR match rate: {ltr_quick*100:.2f}%")
    click.echo(f"    Difference:     {(rtl_quick-ltr_quick)*100:+.2f}pp")

    # 6. Full corpus permutation test
    print_step(f"Permutation test — full corpus ({N_PERMS} perms per direction)...")
    click.echo("    This may take ~2 minutes...")
    rng = random.Random(SEED)
    full_result = _analyse_corpus("full", full_pp, lexicon_set, N_PERMS, rng)
    click.echo(f"    RTL: perm z={full_result['rtl']['perm']['z_score']:.2f}, "
               f"p={full_result['rtl']['perm']['p_value']:.4f}")
    click.echo(f"    LTR: perm z={full_result['ltr']['perm']['z_score']:.2f}, "
               f"p={full_result['ltr']['perm']['p_value']:.4f}")
    click.echo(f"    RTL-LTR direct: z={full_result['direct_comparison']['z_score']:.2f}, "
               f"p={full_result['direct_comparison']['p_value']:.4f}")
    click.echo(f"    Verdict: {full_result['verdict']}")
    click.echo(f"    Elapsed: {full_result['elapsed_s']:.0f}s")

    # 7. Per-language permutation tests (200 perms, faster)
    n_perms_lang = 200
    print_step(f"Permutation test — Language A ({n_perms_lang} perms per direction)...")
    rng_a = random.Random(SEED + 1)
    a_result = _analyse_corpus("A", a_pp, lexicon_set, n_perms_lang, rng_a)
    click.echo(f"    RTL z={a_result['rtl']['perm']['z_score']:.2f}  "
               f"LTR z={a_result['ltr']['perm']['z_score']:.2f}  "
               f"direct z={a_result['direct_comparison']['z_score']:.2f}  "
               f"→ {a_result['verdict']}")

    print_step(f"Permutation test — Language B ({n_perms_lang} perms per direction)...")
    rng_b = random.Random(SEED + 2)
    b_result = _analyse_corpus("B", b_pp, lexicon_set, n_perms_lang, rng_b)
    click.echo(f"    RTL z={b_result['rtl']['perm']['z_score']:.2f}  "
               f"LTR z={b_result['ltr']['perm']['z_score']:.2f}  "
               f"direct z={b_result['direct_comparison']['z_score']:.2f}  "
               f"→ {b_result['verdict']}")

    # 8. Overall verdict
    rtl_z_full = full_result["rtl"]["perm"]["z_score"]
    ltr_z_full = full_result["ltr"]["perm"]["z_score"]
    direct_z = full_result["direct_comparison"]["z_score"]
    direct_p = full_result["direct_comparison"]["p_value"]

    if direct_z > 3 and rtl_z_full > ltr_z_full:
        overall_verdict = "RTL_CONFIRMED"
        verdict_detail = (
            f"RTL significantly outperforms LTR (direct z={direct_z:.2f}, "
            f"p={direct_p:.4f}). RTL perm z={rtl_z_full:.2f} vs "
            f"LTR perm z={ltr_z_full:.2f}. Reading direction confirmed as RTL."
        )
    elif direct_z > 1.5 and rtl_z_full > ltr_z_full:
        overall_verdict = "RTL_FAVORED"
        verdict_detail = (
            f"RTL outperforms LTR (direct z={direct_z:.2f}) with "
            f"stronger permutation signal (RTL z={rtl_z_full:.2f} vs "
            f"LTR z={ltr_z_full:.2f}). RTL is the preferred direction."
        )
    elif abs(direct_z) <= 1.5 and abs(rtl_z_full - ltr_z_full) < 0.5:
        overall_verdict = "DIRECTION_AGNOSTIC"
        verdict_detail = (
            f"RTL and LTR perform similarly (direct z={direct_z:.2f}). "
            f"The Hebrew signal may arise from EVA positional regularities "
            f"rather than a true directional cipher."
        )
    elif direct_z < -1.5:
        overall_verdict = "LTR_FAVORED"
        verdict_detail = (
            f"Unexpected: LTR outperforms RTL (direct z={direct_z:.2f}). "
            f"This would require revisiting the RTL assumption."
        )
    else:
        overall_verdict = "INCONCLUSIVE"
        verdict_detail = (
            f"RTL and LTR not clearly separated (direct z={direct_z:.2f}). "
            f"Permutation gap: {full_result['perm_z_gap']:.2f}. Inconclusive."
        )

    # 9. Assemble and save report
    print_step("Saving report...")
    report = {
        "mapping": {k: v for k, v in FULL_MAPPING.items()},
        "lexicon_forms": len(lexicon_set),
        "min_word_len": MIN_LEN,
        "n_perms_full": N_PERMS,
        "n_perms_lang": n_perms_lang,
        "full": full_result,
        "currier_A": a_result,
        "currier_B": b_result,
        "overall_verdict": overall_verdict,
        "verdict_detail": verdict_detail,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 10. Human-readable summary
    lines = []
    lines.append("=" * 62)
    lines.append("  DIRECTION TEST — RTL vs LTR")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"  Lexicon: {len(lexicon_set):,} Hebrew forms")
    lines.append(f"  Perms: {N_PERMS} (full corpus), {n_perms_lang} (per language)")
    lines.append("")

    for res in (full_result, a_result, b_result):
        lbl = res["label"]
        dc = res["direct_comparison"]
        lines.append(f"  --- {lbl.upper()} CORPUS ---")
        lines.append(f"  Words:        {res['n_preprocessed_words']:,}")
        lines.append(f"  RTL rate:     {res['rtl']['real_rate']*100:.2f}%  "
                     f"perm z={res['rtl']['perm']['z_score']:.2f}  "
                     f"p={res['rtl']['perm']['p_value']:.4f}")
        lines.append(f"  LTR rate:     {res['ltr']['real_rate']*100:.2f}%  "
                     f"perm z={res['ltr']['perm']['z_score']:.2f}  "
                     f"p={res['ltr']['perm']['p_value']:.4f}")
        lines.append(f"  RTL-LTR diff: {dc['diff_pp']:+.2f}pp  "
                     f"z={dc['z_score']:.2f}  p={dc['p_value']:.4f}")
        lines.append(f"  Perm z gap:   {res['perm_z_gap']:+.2f}")
        lines.append(f"  Verdict:      {res['verdict']}")
        lines.append("")

    lines.append("=" * 62)
    lines.append(f"  OVERALL: {overall_verdict}")
    lines.append(f"  {verdict_detail}")
    lines.append("=" * 62)

    txt = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    click.echo(f"    Report: {report_path}")
    click.echo(f"    Summary: {txt_path}")

    # 11. Console output
    click.echo(f"\n{txt}")
