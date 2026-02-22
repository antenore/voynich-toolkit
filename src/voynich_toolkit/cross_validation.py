"""Cross-validation of the EVA-Hebrew mapping.

Addresses the reviewer concern that the mapping was derived and validated
on the same data. Two approaches:

1. **Hand-based split**: Hand 1 (train, 6,545 tokens) vs rest (test, ~26,500).
   Constrained per-letter audit on each subset independently. If X/17
   letters are optimal on both subsets, the mapping is NOT overfitted.
   Plus permutation test on the test subset with fixed mapping.

2. **Random 50/50 split**: 10 random page-level splits. For each split:
   audit + permutation test on held-out half. Reports mean +/- std.

CLI: voynich --force cross-validation
"""

from __future__ import annotations

import json
import random
from collections import Counter

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import FULL_MAPPING, decode_word
from .hand1_deep_dive import HAND1_FOLIOS, MIN_LEN, split_corpus_by_hand
from .mapping_audit import (
    HEBREW_CHARS,
    audit_letter,
    count_matches,
    decode_hebrew,
    get_used_hebrew,
    load_honest_lexicon,
)
from .permutation_stats import (
    build_full_mapping,
    make_lexicon_match_scorer,
    permutation_test_mapping,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Corpus splitting utilities
# =====================================================================


def collect_eva_freqs(words: list[str]) -> Counter:
    """Collect EVA word frequencies from a word list."""
    return Counter(w for w in words if w)


def compute_match_rate(eva_freqs: Counter, lexicon_set: set,
                       mapping: dict = None) -> dict:
    """Decode words and compute match rate."""
    if mapping is None:
        mapping = FULL_MAPPING
    mt, mk, tt, tk = count_matches(eva_freqs, lexicon_set, mapping=mapping)
    return {
        "matched_types": mt,
        "matched_tokens": mk,
        "total_types": tt,
        "total_tokens": tk,
        "match_rate": round(mk / tk, 5) if tk else 0,
    }


def run_constrained_audit(eva_freqs: Counter, lexicon_set: set,
                          base_mapping: dict) -> dict:
    """Run constrained per-letter audit on a word subset.

    For each EVA char, test only the free Hebrew letters (+ current).
    Returns dict with n_optimal, per-letter results.
    """
    used = get_used_hebrew()
    free_hebrew = set(HEBREW_CHARS) - used

    results = {}
    for eva_ch in sorted(base_mapping.keys()):
        current = base_mapping[eva_ch]
        allowed = free_hebrew | {current}

        res = audit_letter(eva_ch, eva_freqs, lexicon_set,
                           dict(base_mapping), allowed_hebrew=allowed)
        results[eva_ch] = {
            "current": res["current"],
            "current_rank": res["current_rank"],
            "best": res["best"],
            "gap_tokens": res["gap_tokens"],
            "is_optimal": res["current_rank"] == 1,
        }

    n_optimal = sum(1 for r in results.values() if r["is_optimal"])
    return {
        "n_optimal": n_optimal,
        "n_total": len(results),
        "per_letter": results,
    }


# =====================================================================
# Hand-based cross-validation
# =====================================================================


def hand_based_cv(pages: list[dict], lexicon_set: set,
                  n_perms: int = 200) -> dict:
    """Hand 1 (train) vs rest (test) cross-validation.

    1. Audit mapping on Hand 1 words only
    2. Audit mapping on non-Hand-1 words only
    3. If both subsets independently confirm X/17 letters optimal,
       mapping is stable (not overfit to any subset)
    4. Permutation test on test subset with fixed mapping
    """
    corpus = split_corpus_by_hand(pages)

    # Train = Hand 1
    train_words = corpus.get("1", {}).get("words", [])
    # Test = all other hands
    test_words = []
    for hand, data in corpus.items():
        if hand != "1":
            test_words.extend(data["words"])

    train_freqs = collect_eva_freqs(train_words)
    test_freqs = collect_eva_freqs(test_words)

    click.echo(f"    Train (Hand 1): {sum(train_freqs.values()):,} tokens, "
               f"{len(train_freqs):,} types")
    click.echo(f"    Test (others):  {sum(test_freqs.values()):,} tokens, "
               f"{len(test_freqs):,} types")

    # Match rates
    train_match = compute_match_rate(train_freqs, lexicon_set)
    test_match = compute_match_rate(test_freqs, lexicon_set)
    click.echo(f"    Train match: {train_match['match_rate']*100:.1f}%")
    click.echo(f"    Test match:  {test_match['match_rate']*100:.1f}%")

    # Constrained audit on each subset
    print_step("Auditing mapping on train subset (Hand 1)...")
    train_audit = run_constrained_audit(train_freqs, lexicon_set, FULL_MAPPING)
    click.echo(f"    Train optimal: {train_audit['n_optimal']}/{train_audit['n_total']}")

    print_step("Auditing mapping on test subset (non-Hand 1)...")
    test_audit = run_constrained_audit(test_freqs, lexicon_set, FULL_MAPPING)
    click.echo(f"    Test optimal:  {test_audit['n_optimal']}/{test_audit['n_total']}")

    # Agreement: letters optimal on BOTH subsets
    both_optimal = []
    train_only = []
    test_only = []
    neither = []
    for eva_ch in sorted(FULL_MAPPING.keys()):
        t_opt = train_audit["per_letter"][eva_ch]["is_optimal"]
        s_opt = test_audit["per_letter"][eva_ch]["is_optimal"]
        if t_opt and s_opt:
            both_optimal.append(eva_ch)
        elif t_opt:
            train_only.append(eva_ch)
        elif s_opt:
            test_only.append(eva_ch)
        else:
            neither.append(eva_ch)

    click.echo(f"    Both optimal: {len(both_optimal)}/17 "
               f"({', '.join(both_optimal)})")
    if train_only:
        click.echo(f"    Train-only:   {', '.join(train_only)}")
    if test_only:
        click.echo(f"    Test-only:    {', '.join(test_only)}")

    # Permutation test on test subset
    print_step(f"Permutation test on test subset ({n_perms} perms)...")
    full_mapping = build_full_mapping(FULL_MAPPING)
    test_words_flat = [w for w in test_words if w]
    scorer = make_lexicon_match_scorer(test_words_flat, lexicon_set, min_len=3)
    perm_result = permutation_test_mapping(
        scorer, full_mapping, n_perms=n_perms, seed=42)
    click.echo(f"    Test z-score: {perm_result['z_score']:.2f}  "
               f"p={perm_result['p_value']:.4f}")

    return {
        "train_n_tokens": sum(train_freqs.values()),
        "test_n_tokens": sum(test_freqs.values()),
        "train_match": train_match,
        "test_match": test_match,
        "train_audit": train_audit,
        "test_audit": test_audit,
        "agreement": {
            "both_optimal": both_optimal,
            "n_both": len(both_optimal),
            "train_only": train_only,
            "test_only": test_only,
            "neither": neither,
        },
        "test_permutation": perm_result,
    }


# =====================================================================
# Random 50/50 split cross-validation
# =====================================================================


def random_split_cv(pages: list[dict], lexicon_set: set,
                    n_splits: int = 10, n_perms: int = 200,
                    seed: int = 42) -> dict:
    """Random page-level 50/50 split, repeated n_splits times.

    For each split:
    - Constrained audit on train half
    - Match rate + permutation z-score on test half

    Returns mean +/- std of test z-scores and n_optimal.
    """
    rng = random.Random(seed)
    full_mapping = build_full_mapping(FULL_MAPPING)

    split_results = []
    for i in range(n_splits):
        # Shuffle pages and split 50/50
        shuffled = list(range(len(pages)))
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2

        train_words = []
        test_words = []
        for j, idx in enumerate(shuffled):
            words = pages[idx]["words"]
            if j < mid:
                train_words.extend(words)
            else:
                test_words.extend(words)

        train_freqs = collect_eva_freqs(train_words)
        test_freqs = collect_eva_freqs(test_words)

        # Audit on train
        train_audit = run_constrained_audit(
            train_freqs, lexicon_set, FULL_MAPPING)

        # Match rate on test
        test_match = compute_match_rate(test_freqs, lexicon_set)

        # Permutation test on test
        test_words_flat = [w for w in test_words if w]
        scorer = make_lexicon_match_scorer(
            test_words_flat, lexicon_set, min_len=3)
        perm = permutation_test_mapping(
            scorer, full_mapping, n_perms=n_perms, seed=seed + i)

        split_results.append({
            "split": i + 1,
            "train_tokens": sum(train_freqs.values()),
            "test_tokens": sum(test_freqs.values()),
            "train_n_optimal": train_audit["n_optimal"],
            "test_match_rate": test_match["match_rate"],
            "test_z_score": perm["z_score"],
            "test_p_value": perm["p_value"],
        })

        click.echo(f"    Split {i+1}/{n_splits}: "
                   f"train opt={train_audit['n_optimal']}/17  "
                   f"test rate={test_match['match_rate']*100:.1f}%  "
                   f"z={perm['z_score']:.2f}")

    # Summary statistics
    z_scores = [r["test_z_score"] for r in split_results]
    n_opts = [r["train_n_optimal"] for r in split_results]
    test_rates = [r["test_match_rate"] for r in split_results]

    return {
        "n_splits": n_splits,
        "n_perms": n_perms,
        "splits": split_results,
        "summary": {
            "z_mean": round(float(np.mean(z_scores)), 2),
            "z_std": round(float(np.std(z_scores)), 2),
            "z_min": round(float(np.min(z_scores)), 2),
            "z_max": round(float(np.max(z_scores)), 2),
            "n_optimal_mean": round(float(np.mean(n_opts)), 1),
            "n_optimal_min": int(np.min(n_opts)),
            "n_optimal_max": int(np.max(n_opts)),
            "test_rate_mean": round(float(np.mean(test_rates)), 5),
            "test_rate_std": round(float(np.std(test_rates)), 5),
            "all_significant": all(r["test_p_value"] < 0.05
                                   for r in split_results),
        },
    }


# =====================================================================
# Entry point
# =====================================================================


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Cross-validation of the EVA-Hebrew mapping."""
    out_json = config.stats_dir / "cross_validation.json"
    out_txt = config.stats_dir / "cross_validation.txt"

    if out_json.exists() and not force:
        click.echo(f"  Output exists: {out_json} (use --force)")
        return

    config.ensure_dirs()
    print_header("Cross-Validation of EVA-Hebrew Mapping")

    # Parse corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # Load honest lexicon
    print_step("Loading honest lexicon (45K, no Sefaria-Corpus)...")
    lexicon_set, _ = load_honest_lexicon(config)
    click.echo(f"    {len(lexicon_set):,} forms")

    # 1. Hand-based cross-validation
    print_step("=== HAND-BASED CROSS-VALIDATION ===")
    hand_cv = hand_based_cv(pages, lexicon_set, n_perms=200)

    # 2. Random 50/50 split
    print_step("=== RANDOM 50/50 SPLIT (10 repetitions) ===")
    random_cv = random_split_cv(pages, lexicon_set,
                                n_splits=10, n_perms=200)

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("  CROSS-VALIDATION SUMMARY")
    click.echo(f"{'='*60}")
    click.echo(f"\n  Hand-based (H1 train / rest test):")
    click.echo(f"    Train optimal: {hand_cv['train_audit']['n_optimal']}/17")
    click.echo(f"    Test optimal:  {hand_cv['test_audit']['n_optimal']}/17")
    click.echo(f"    Both optimal:  {hand_cv['agreement']['n_both']}/17")
    click.echo(f"    Test z-score:  {hand_cv['test_permutation']['z_score']:.2f}")
    click.echo(f"    Test p-value:  {hand_cv['test_permutation']['p_value']:.4f}")

    s = random_cv["summary"]
    click.echo(f"\n  Random 50/50 ({random_cv['n_splits']} splits):")
    click.echo(f"    z-score:  {s['z_mean']:.2f} +/- {s['z_std']:.2f} "
               f"(range {s['z_min']:.2f}--{s['z_max']:.2f})")
    click.echo(f"    Optimal:  {s['n_optimal_mean']:.1f} "
               f"({s['n_optimal_min']}--{s['n_optimal_max']})/17")
    click.echo(f"    All sig:  {s['all_significant']}")

    # Save JSON
    report = {
        "hand_based": hand_cv,
        "random_split": random_cv,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"\n  JSON: {out_json}")

    # Save TXT summary
    lines = [
        "Cross-Validation of EVA-Hebrew Mapping",
        "=" * 60,
        "",
        "1. HAND-BASED (Hand 1 train / rest test)",
        f"   Train tokens:    {hand_cv['train_n_tokens']:,}",
        f"   Test tokens:     {hand_cv['test_n_tokens']:,}",
        f"   Train match:     {hand_cv['train_match']['match_rate']*100:.1f}%",
        f"   Test match:      {hand_cv['test_match']['match_rate']*100:.1f}%",
        f"   Train optimal:   {hand_cv['train_audit']['n_optimal']}/17",
        f"   Test optimal:    {hand_cv['test_audit']['n_optimal']}/17",
        f"   Both optimal:    {hand_cv['agreement']['n_both']}/17 "
        f"({', '.join(hand_cv['agreement']['both_optimal'])})",
        f"   Test z-score:    {hand_cv['test_permutation']['z_score']:.2f}",
        f"   Test p-value:    {hand_cv['test_permutation']['p_value']:.4f}",
        "",
        f"2. RANDOM 50/50 ({random_cv['n_splits']} splits)",
        f"   z-score mean:    {s['z_mean']:.2f} +/- {s['z_std']:.2f}",
        f"   z-score range:   {s['z_min']:.2f} -- {s['z_max']:.2f}",
        f"   optimal mean:    {s['n_optimal_mean']:.1f}/17",
        f"   test rate mean:  {s['test_rate_mean']*100:.1f}%",
        f"   all significant: {s['all_significant']}",
    ]
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    click.echo(f"  TXT: {out_txt}")
