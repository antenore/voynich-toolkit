"""
Phase 2e — Sub-analysis of hand ? by section (revised 2026-03-28).

Hand ? has entropy +12.25 above the null (Phase 2) and is the only hand covering
ALL sections and all Currier languages. This is consistent with an aggregate
of multiple scribes not identified by Davis.

This module tests whether the anomaly is a mixing artifact, a section effect,
or an intrinsic property:

  H_mixing: each sub-corpus ?-per-section has normal entropy (z~0) vs global null.
            The anomaly is entirely due to combining different styles.
  H_section_effect: Hand ?'s per-section z-scores are normal when compared against
            a section-controlled null (sampling within-section words only).
            Other multi-section hands show similar section-dependent variation.
  H_intrinsic: even with section-controlled null, Hand ? remains anomalous
               AND other hands do not show similar variation.

Three sub-tests (added 2026-03-28):
  1. Section baseline: entropy of each section (all hands combined)
  2. Section-controlled null: for each (hand, section) pair, null model samples
     from within-section words only, not the global corpus
  3. Cross-hand comparison: apply the same test to ALL multi-section hands

Output:
  hand_unknown.json
  hand_unknown_summary.txt
  DB tables: hand_unknown_sections, hand_unknown_cross_hand
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2 as scipy_chi2

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .hand_characterization import eva_profile
from .hand_structure import (
    N_NULL_SAMPLES,
    SEED,
    bigram_freq,
    null_distribution,
    z_score_vs_null,
)
from .scribe_analysis import split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# Minimum tokens for reliable entropy estimation
MIN_TOKENS_SECTION = 200


# =====================================================================
# Corpus splitting
# =====================================================================

def split_unknown_by_section(corpus: dict) -> dict:
    """Split pages of hand ? by section.

    Returns: dict[section_code] -> list of words
    """
    unknown_pages = corpus.get("?", {}).get("pages", [])
    by_section: dict[str, list[str]] = {}
    for p in unknown_pages:
        sec = p.get("section", "?")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p["words"])
    return by_section


def get_all_words_by_section(pages: list[dict]) -> dict[str, list[str]]:
    """Get all words per section (all hands combined).

    Returns: dict[section_code] -> list of words
    """
    by_section: dict[str, list[str]] = {}
    for p in pages:
        sec = p.get("section", "?")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p["words"])
    return by_section


def split_hand_by_section(corpus: dict, hand: str) -> dict[str, list[str]]:
    """Split a specific hand's pages by section.

    Returns: dict[section_code] -> list of words
    """
    hand_pages = corpus.get(hand, {}).get("pages", [])
    by_section: dict[str, list[str]] = {}
    for p in hand_pages:
        sec = p.get("section", "?")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p["words"])
    return by_section


# =====================================================================
# Core analysis
# =====================================================================

def section_entropy_vs_null(words: list[str], all_words: list[str],
                             section: str, seed_offset: int = 0) -> dict:
    """Compute entropy + Zipf with null model for a sub-corpus.

    Uses GLOBAL corpus as null model (original Phase 2e method).

    Returns: dict with observed metrics + z-score vs null
    """
    n = len(words)
    if n < 10:
        return {"n_tokens": n, "skipped": True, "reason": "< 10 tokens"}

    profile = eva_profile(words)
    null = null_distribution(all_words, n, n_samples=N_NULL_SAMPLES,
                              seed=SEED + seed_offset)

    z_ent = z_score_vs_null(
        profile["shannon_entropy"], null["entropy_mean"], null["entropy_std"])
    z_zip = None
    if profile["zipf_slope"] is not None and null["zipf_mean"] is not None:
        z_zip = z_score_vs_null(
            profile["zipf_slope"], null["zipf_mean"], null["zipf_std"])

    return {
        "n_tokens":          n,
        "n_unique":          profile["n_unique"],
        "entropy_obs":       profile["shannon_entropy"],
        "entropy_null_mean": round(null["entropy_mean"], 4),
        "entropy_null_std":  round(null["entropy_std"], 6),
        "z_entropy":         round(z_ent, 3) if z_ent is not None else None,
        "zipf_obs":          profile["zipf_slope"],
        "zipf_null_mean":    round(null["zipf_mean"], 4) if null["zipf_mean"] else None,
        "z_zipf":            round(z_zip, 3) if z_zip is not None else None,
        "unstable_flag":     n < 1000,
    }


def section_controlled_entropy(words: list[str], section_words: list[str],
                                seed_offset: int = 0) -> dict:
    """Compute entropy with SECTION-CONTROLLED null model.

    Instead of sampling from the global corpus, samples from words
    within the same section only. This controls for section-level
    entropy differences.

    Returns: dict with observed entropy, section-controlled null stats, z-score
    """
    n = len(words)
    if n < MIN_TOKENS_SECTION:
        return {"n_tokens": n, "skipped": True,
                "reason": f"< {MIN_TOKENS_SECTION} tokens"}

    # Need enough section words to sample from
    if len(section_words) < n:
        return {"n_tokens": n, "skipped": True,
                "reason": "section pool smaller than hand sample"}

    profile = eva_profile(words)
    null = null_distribution(section_words, n, n_samples=N_NULL_SAMPLES,
                              seed=SEED + seed_offset + 7000)

    z_ent = z_score_vs_null(
        profile["shannon_entropy"], null["entropy_mean"], null["entropy_std"])

    return {
        "n_tokens":              n,
        "n_unique":              profile["n_unique"],
        "entropy_obs":           profile["shannon_entropy"],
        "section_null_mean":     round(null["entropy_mean"], 4),
        "section_null_std":      round(null["entropy_std"], 6),
        "z_entropy_section":     round(z_ent, 3) if z_ent is not None else None,
        "unstable_flag":         n < 1000,
    }


def section_baseline(section_words: dict[str, list[str]]) -> dict[str, dict]:
    """Compute raw entropy for each section (all hands combined).

    Returns: dict[section] -> {n_tokens, entropy, n_unique}
    """
    results = {}
    for sec, words in sorted(section_words.items()):
        if len(words) < 10:
            continue
        profile = eva_profile(words)
        results[sec] = {
            "n_tokens":  len(words),
            "n_unique":  profile["n_unique"],
            "entropy":   round(profile["shannon_entropy"], 4),
            "zipf":      profile["zipf_slope"],
        }
    return results


def cross_hand_section_analysis(corpus: dict, section_words_all: dict[str, list[str]],
                                 pages: list[dict]) -> dict:
    """Apply section-controlled entropy test to ALL hands, not just Hand ?.

    For each hand that appears in a section with >= MIN_TOKENS_SECTION tokens,
    compute entropy vs section-controlled null.

    Returns: dict[hand] -> dict[section] -> {z_entropy_section, ...}
    """
    results: dict[str, dict] = {}

    for hand in sorted(corpus.keys()):
        hand_by_sec = split_hand_by_section(corpus, hand)

        # Only include hands with at least 2 sections with enough tokens
        eligible_sections = {sec: words for sec, words in hand_by_sec.items()
                           if len(words) >= MIN_TOKENS_SECTION}
        if not eligible_sections:
            continue

        hand_results: dict[str, dict] = {}
        for sec, words in sorted(eligible_sections.items()):
            sec_all_words = section_words_all.get(sec, [])
            result = section_controlled_entropy(
                words, sec_all_words,
                seed_offset=(ord(hand[0]) if hand else 0) * 100 + hash(sec) % 100
            )
            hand_results[sec] = result

        if hand_results:
            # Summary stats for this hand
            z_scores = [d["z_entropy_section"] for d in hand_results.values()
                       if not d.get("skipped") and d.get("z_entropy_section") is not None]
            results[hand] = {
                "sections": hand_results,
                "n_sections_tested": len([d for d in hand_results.values()
                                         if not d.get("skipped")]),
                "z_scores": z_scores,
                "mean_abs_z": round(np.mean([abs(z) for z in z_scores]), 3) if z_scores else None,
                "max_abs_z": round(max(abs(z) for z in z_scores), 3) if z_scores else None,
                "n_anomalous": sum(1 for z in z_scores if abs(z) > 2),
            }

    return results


def bigram_chisquare_section(words: list[str], all_words: list[str]) -> dict:
    """Chi-square bigrams for a sub-corpus vs global corpus (top-50)."""
    global_bg = bigram_freq(all_words)
    top50 = [bg for bg, _ in global_bg.most_common(50)]
    global_total = sum(global_bg[bg] for bg in top50)
    global_freq = np.array([global_bg[bg] / global_total for bg in top50])

    hand_bg = bigram_freq(words)
    hand_total = sum(hand_bg.get(bg, 0) for bg in top50)
    if hand_total == 0:
        return {"skipped": True, "reason": "no bigrams"}

    observed = np.array([hand_bg.get(bg, 0) for bg in top50], dtype=float)
    expected = global_freq * hand_total

    mask = expected >= 5
    obs_filt = observed[mask]
    exp_filt = expected[mask]
    df = int(mask.sum()) - 1

    if df <= 0:
        return {"skipped": True, "reason": "too few cells with expected >= 5"}

    chi2_stat = float(np.sum((obs_filt - exp_filt) ** 2 / exp_filt))
    p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

    return {
        "n_bigrams_total": int(hand_total),
        "n_cells_used":    int(mask.sum()),
        "chi2":            round(chi2_stat, 2),
        "df":              df,
        "p_value":         round(p_value, 6),
        "significant_05":  bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.001),
    }


def section_profile_for_davis_hands(corpus: dict, section: str,
                                     all_words: list[str]) -> dict:
    """Compute mean entropy of the same section in Davis hands 1-5.

    Used as benchmark: if ? in section X has z similar to Davis hands in
    the same section X, then ? is not anomalous relative to the section context.
    """
    davis = {"1", "2", "3", "4", "5"}
    results = {}
    for hand in davis:
        pages = corpus.get(hand, {}).get("pages", [])
        words = [w for p in pages if p.get("section") == section
                 for w in p["words"]]
        if len(words) < 20:
            continue
        p = eva_profile(words)
        results[hand] = {
            "n_tokens": len(words),
            "entropy":  p["shannon_entropy"],
            "zipf":     p["zipf_slope"],
        }
    return results


# =====================================================================
# Verdict (revised 2026-03-28)
# =====================================================================

def compute_verdict(by_section_results: dict,
                    cross_hand: dict) -> dict:
    """Determine whether the anomaly is mixing, section effect, or intrinsic.

    Uses section-controlled z-scores (not global) and compares Hand ?
    against other hands.

    Criteria:
    - SECTION_EFFECT: Hand ?'s section-controlled |z| values are comparable
      to other multi-section hands (mean |z| within 1 SD of other hands)
    - INTRINSIC_ANOMALY: Hand ? has significantly more anomalous sections
      than other hands, even with section-controlled null
    - MIXING_ARTIFACT: all section-controlled z-scores near 0 for Hand ?
    - MIXED_EVIDENCE: inconclusive
    """
    # Get Hand ?'s section-controlled results
    q_data = cross_hand.get("?")
    if not q_data or not q_data["z_scores"]:
        return {"verdict": "INSUFFICIENT_DATA", "n_sections_tested": 0}

    q_z_scores = q_data["z_scores"]
    q_mean_abs_z = q_data["mean_abs_z"]
    q_n_anomalous = q_data["n_anomalous"]
    q_n_tested = q_data["n_sections_tested"]

    # Get other hands' section-controlled results
    other_hands = {h: d for h, d in cross_hand.items()
                   if h != "?" and d.get("z_scores")}

    # Collect mean |z| for other hands that have section-controlled data
    other_mean_abs_z = [d["mean_abs_z"] for d in other_hands.values()
                       if d.get("mean_abs_z") is not None]
    other_n_anomalous = [d["n_anomalous"] for d in other_hands.values()]

    # Decision logic
    if q_n_anomalous == 0:
        verdict = "SECTION_EFFECT"
        explanation = (
            f"All {q_n_tested} sections of Hand ? have |z| <= 2 with "
            "section-controlled null. The +12 sigma anomaly from Phase 2 is "
            "explained by combining sections with different base entropies. "
            "Hand ?'s per-section entropy is normal for each section."
        )
    elif other_mean_abs_z and q_mean_abs_z is not None:
        # Compare Hand ? vs other hands
        other_mean = float(np.mean(other_mean_abs_z))
        other_std = float(np.std(other_mean_abs_z, ddof=1)) if len(other_mean_abs_z) > 1 else 1.0

        if other_std > 0:
            q_relative_z = (q_mean_abs_z - other_mean) / other_std
        else:
            q_relative_z = 0.0

        if q_relative_z <= 2.0 and q_n_anomalous <= max(other_n_anomalous, default=0) + 1:
            verdict = "SECTION_EFFECT"
            explanation = (
                f"Hand ? mean |z|={q_mean_abs_z:.2f} is within range of other "
                f"hands (mean={other_mean:.2f}, SD={other_std:.2f}). "
                f"Hand ? has {q_n_anomalous} anomalous sections vs "
                f"max {max(other_n_anomalous, default=0)} for other hands. "
                "The section-level variation is comparable across hands."
            )
        elif q_relative_z > 2.0 or q_n_anomalous > max(other_n_anomalous, default=0) + 2:
            verdict = "INTRINSIC_ANOMALY"
            explanation = (
                f"Hand ? mean |z|={q_mean_abs_z:.2f} exceeds other hands "
                f"(mean={other_mean:.2f}, SD={other_std:.2f}, relative z={q_relative_z:.1f}). "
                f"Hand ? has {q_n_anomalous} anomalous sections vs "
                f"max {max(other_n_anomalous, default=0)} for other hands. "
                "Hand ? is genuinely anomalous even controlling for section."
            )
        else:
            verdict = "MIXED_EVIDENCE"
            explanation = (
                f"Hand ? mean |z|={q_mean_abs_z:.2f} vs others mean={other_mean:.2f}. "
                f"{q_n_anomalous} anomalous sections. "
                "Evidence is not conclusive in either direction."
            )
    else:
        # No other hands to compare against
        if q_n_anomalous >= 2:
            verdict = "INTRINSIC_ANOMALY"
            explanation = (
                f"{q_n_anomalous} sections with |z| > 2 even with section-controlled null. "
                "No other multi-section hands available for comparison."
            )
        else:
            verdict = "MIXED_EVIDENCE"
            explanation = (
                f"Only {q_n_anomalous} anomalous section(s) with section-controlled null. "
                "Insufficient data for a clear verdict."
            )

    return {
        "verdict":              verdict,
        "explanation":          explanation,
        "n_sections_tested":    q_n_tested,
        "n_anomalous":          q_n_anomalous,
        "hand_q_mean_abs_z":    q_mean_abs_z,
        "hand_q_z_scores":      {sec: round(z, 2)
                                 for sec, z_list in [("?", q_z_scores)]
                                 for z in z_list} if False else
                                dict(zip(
                                    [sec for sec in sorted(q_data["sections"].keys())
                                     if not q_data["sections"][sec].get("skipped")
                                     and q_data["sections"][sec].get("z_entropy_section") is not None],
                                    q_z_scores
                                )),
        "other_hands_mean_abs_z": {h: d["mean_abs_z"] for h, d in other_hands.items()
                                   if d.get("mean_abs_z") is not None},
        "other_hands_n_anomalous": {h: d["n_anomalous"] for h, d in other_hands.items()},
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(by_section: dict, cross_hand: dict, verdict: dict,
               db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Table 1: Hand ? per-section (original + section-controlled)
    cur.execute("DROP TABLE IF EXISTS hand_unknown_sections")
    cur.execute("""
        CREATE TABLE hand_unknown_sections (
            section              TEXT PRIMARY KEY,
            section_name         TEXT,
            n_tokens             INTEGER,
            n_unique             INTEGER,
            entropy_obs          REAL,
            entropy_null_mean    REAL,
            z_entropy_global     REAL,
            section_null_mean    REAL,
            z_entropy_section    REAL,
            zipf_obs             REAL,
            z_zipf               REAL,
            bigram_chi2          REAL,
            bigram_p             REAL,
            unstable_flag        INTEGER,
            verdict              TEXT
        )
    """)

    overall_verdict = verdict.get("verdict", "?")
    q_cross = cross_hand.get("?", {}).get("sections", {})

    for sec, d in sorted(by_section.items()):
        if d.get("skipped"):
            continue
        bg = d.get("bigrams", {})
        sc = q_cross.get(sec, {})
        cur.execute("""
            INSERT INTO hand_unknown_sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sec,
            SECTION_NAMES.get(sec, sec),
            d.get("n_tokens"),
            d.get("n_unique"),
            d.get("entropy_obs"),
            d.get("entropy_null_mean"),
            d.get("z_entropy"),
            sc.get("section_null_mean"),
            sc.get("z_entropy_section"),
            d.get("zipf_obs"),
            d.get("z_zipf"),
            bg.get("chi2") if not bg.get("skipped") else None,
            bg.get("p_value") if not bg.get("skipped") else None,
            int(d.get("unstable_flag", False)),
            overall_verdict,
        ))

    # Table 2: Cross-hand section-controlled comparison
    cur.execute("DROP TABLE IF EXISTS hand_unknown_cross_hand")
    cur.execute("""
        CREATE TABLE hand_unknown_cross_hand (
            hand                TEXT,
            section             TEXT,
            section_name        TEXT,
            n_tokens            INTEGER,
            entropy_obs         REAL,
            section_null_mean   REAL,
            z_entropy_section   REAL,
            unstable_flag       INTEGER,
            PRIMARY KEY (hand, section)
        )
    """)

    for hand, hdata in sorted(cross_hand.items()):
        for sec, sd in sorted(hdata.get("sections", {}).items()):
            if sd.get("skipped"):
                continue
            cur.execute("""
                INSERT INTO hand_unknown_cross_hand VALUES (?,?,?,?,?,?,?,?)
            """, (
                hand, sec, SECTION_NAMES.get(sec, sec),
                sd.get("n_tokens"),
                sd.get("entropy_obs"),
                sd.get("section_null_mean"),
                sd.get("z_entropy_section"),
                int(sd.get("unstable_flag", False)),
            ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(by_section: dict, davis_comparison: dict, verdict: dict,
                   baseline: dict, cross_hand: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 2e — Sub-analysis of hand ? by section (revised)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Question: is the +12 sigma entropy anomaly of hand ? due to")
    lines.append("  (a) mixing sections with different base entropies (section effect),")
    lines.append("  (b) an intrinsic property of hand ? (multiple scribes or anomalous scribe)?")

    # Section baselines
    lines.append("\n-- Section baseline (all hands combined) --")
    lines.append(f"  {'Sec':>4}  {'Name':>16}  {'N':>7}  {'H':>7}")
    lines.append("  " + "-" * 50)
    for sec, d in sorted(baseline.items()):
        sec_name = SECTION_NAMES.get(sec, sec)[:16]
        lines.append(f"  {sec:>4}  {sec_name:>16}  {d['n_tokens']:>7,}  {d['entropy']:>7.4f}")

    # Hand ? per-section: global vs section-controlled null
    lines.append("\n-- Hand ? per-section: global null vs section-controlled null --")
    lines.append(
        f"  {'Sec':>4}  {'Name':>16}  {'N':>6}  {'H_obs':>7}  "
        f"{'z_global':>8}  {'z_section':>9}  Flag"
    )
    lines.append("  " + "-" * 72)

    q_cross = cross_hand.get("?", {}).get("sections", {})
    for sec in sorted(by_section.keys()):
        d = by_section[sec]
        if d.get("skipped"):
            lines.append(f"  {sec:>4}  [skip: {d.get('reason', '?')}]")
            continue
        z_g = f"{d['z_entropy']:+.2f}" if d.get("z_entropy") is not None else "    n/a"
        sc = q_cross.get(sec, {})
        z_s = f"{sc['z_entropy_section']:+.2f}" if sc.get("z_entropy_section") is not None else "    n/a"
        if sc.get("skipped"):
            z_s = f"  skip"
        flag = "unstable" if d["unstable_flag"] else ""
        sec_name = SECTION_NAMES.get(sec, sec)[:16]
        lines.append(
            f"  {sec:>4}  {sec_name:>16}  {d['n_tokens']:>6,}  "
            f"{d['entropy_obs']:>7.4f}  {z_g:>8}  {z_s:>9}  {flag}"
        )

    # Cross-hand comparison
    lines.append("\n-- Cross-hand section-controlled entropy (all hands) --")
    lines.append(
        f"  {'Hand':>5}  {'Sec':>4}  {'N':>6}  {'H_obs':>7}  "
        f"{'H_sec_null':>10}  {'z_section':>9}"
    )
    lines.append("  " + "-" * 58)
    for hand in sorted(cross_hand.keys()):
        hdata = cross_hand[hand]
        for sec in sorted(hdata.get("sections", {}).keys()):
            sd = hdata["sections"][sec]
            if sd.get("skipped"):
                continue
            z_s = f"{sd['z_entropy_section']:+.2f}" if sd.get("z_entropy_section") is not None else "  n/a"
            lines.append(
                f"  {hand:>5}  {sec:>4}  {sd['n_tokens']:>6,}  "
                f"{sd['entropy_obs']:>7.4f}  {sd['section_null_mean']:>10.4f}  {z_s:>9}"
            )

    # Summary per hand
    lines.append("\n-- Summary per hand (section-controlled) --")
    lines.append(f"  {'Hand':>5}  {'Secs':>4}  {'mean|z|':>7}  {'max|z|':>7}  {'#anom':>5}")
    lines.append("  " + "-" * 40)
    for hand in sorted(cross_hand.keys()):
        hdata = cross_hand[hand]
        n_sec = hdata.get("n_sections_tested", 0)
        maz = f"{hdata['mean_abs_z']:.2f}" if hdata.get("mean_abs_z") is not None else "  n/a"
        maxz = f"{hdata['max_abs_z']:.2f}" if hdata.get("max_abs_z") is not None else "  n/a"
        na = hdata.get("n_anomalous", 0)
        marker = " <-- Hand ?" if hand == "?" else ""
        lines.append(f"  {hand:>5}  {n_sec:>4}  {maz:>7}  {maxz:>7}  {na:>5}{marker}")

    # Comparison with Davis hands in the same section
    lines.append("\n-- Entropy comparison same section: hand ? vs Davis hands --")
    for sec in sorted(davis_comparison.keys()):
        davis = davis_comparison[sec]
        if not davis:
            continue
        q_ent = by_section.get(sec, {}).get("entropy_obs")
        q_str = f"{q_ent:.4f}" if q_ent else "n/a"
        davis_str = "  ".join(
            f"H{h}={v['entropy']:.4f}(n={v['n_tokens']})"
            for h, v in sorted(davis.items())
        )
        lines.append(f"  Section {sec}: ?={q_str}  |  {davis_str}")

    # Verdict
    lines.append(f"\n-- Verdict --")
    lines.append(f"  {verdict['verdict']}")
    lines.append(f"  {verdict.get('explanation', '')}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 2e: sub-analysis of hand ? by section -- section effect or intrinsic?"""
    report_path = config.stats_dir / "hand_unknown.json"
    summary_path = config.stats_dir / "hand_unknown_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_unknown report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 2e — Sub-analysis of Hand ? by Section (revised)")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    all_words = [w for p in pages for w in p["words"]]
    click.echo(f"    {len(all_words):,} total words")

    # 2. Split by hand and by section
    print_step("Splitting by hand and by section...")
    corpus = split_corpus_by_hand(pages)
    unknown_by_section = split_unknown_by_section(corpus)
    click.echo(f"    Hand ?: {len(unknown_by_section)} sections -- "
               + ", ".join(f"{sec}({len(w)})" for sec, w in
                           sorted(unknown_by_section.items())))

    # 3. Section baselines (all hands combined)
    print_step("Section baselines (all hands combined)...")
    section_words_all = get_all_words_by_section(pages)
    baseline = section_baseline(section_words_all)
    for sec, d in sorted(baseline.items()):
        click.echo(f"    Section {sec}: {d['n_tokens']:,} words, H={d['entropy']:.4f}")

    # 4. Original analysis: Hand ? per section vs GLOBAL null
    print_step("Hand ? per section vs global null (500 samples each)...")
    by_section: dict[str, dict] = {}
    for i, (sec, words) in enumerate(sorted(unknown_by_section.items())):
        click.echo(f"    Section {sec} ({SECTION_NAMES.get(sec, sec)},"
                   f" {len(words)} tokens)...", nl=False)
        result = section_entropy_vs_null(words, all_words, sec, seed_offset=i * 100)
        result["bigrams"] = bigram_chisquare_section(words, all_words)
        by_section[sec] = result
        if result.get("skipped"):
            click.echo(f" skip ({result.get('reason')})")
        else:
            z_h = f"{result['z_entropy']:+.2f}" if result["z_entropy"] is not None else "n/a"
            flag = " unstable" if result["unstable_flag"] else ""
            click.echo(f" H={result['entropy_obs']:.4f} z_global={z_h}{flag}")

    # 5. Cross-hand section-controlled analysis (ALL hands)
    print_step("Section-controlled null model (all hands, 500 samples each)...")
    cross_hand = cross_hand_section_analysis(corpus, section_words_all, pages)
    for hand in sorted(cross_hand.keys()):
        hdata = cross_hand[hand]
        n_sec = hdata["n_sections_tested"]
        maz = hdata.get("mean_abs_z")
        maz_str = f"mean|z|={maz:.2f}" if maz is not None else "n/a"
        na = hdata.get("n_anomalous", 0)
        click.echo(f"    Hand {hand}: {n_sec} sections, {maz_str}, {na} anomalous")

    # 6. Comparison with Davis hands in the same section
    print_step("Comparison with Davis hands in the same section...")
    davis_comparison: dict[str, dict] = {}
    for sec in sorted(unknown_by_section.keys()):
        davis_profiles = section_profile_for_davis_hands(corpus, sec, all_words)
        if davis_profiles:
            davis_comparison[sec] = davis_profiles
            for hand, v in sorted(davis_profiles.items()):
                click.echo(f"    Section {sec} Hand {hand}: "
                           f"n={v['n_tokens']} H={v['entropy']:.4f}")

    # 7. Verdict (revised: uses section-controlled null + cross-hand comparison)
    print_step("Computing verdict (section-controlled)...")
    verdict = compute_verdict(by_section, cross_hand)
    click.echo(f"    VERDICT: {verdict['verdict']}")
    click.echo(f"    {verdict.get('explanation', '')}")

    # 8. Save JSON
    print_step("Saving JSON...")
    report = {
        "baseline":         baseline,
        "by_section":       by_section,
        "cross_hand":       cross_hand,
        "davis_comparison": davis_comparison,
        "verdict":          verdict,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 9. Save TXT
    summary = format_summary(by_section, davis_comparison, verdict, baseline, cross_hand)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 10. Save to DB
    print_step("Writing DB tables hand_unknown_sections, hand_unknown_cross_hand...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(by_section, cross_hand, verdict, db_path)
        click.echo(f"    {db_path}")
    else:
        click.echo(f"    WARN: DB not found at {db_path} -- skip DB write")

    click.echo(f"\n{summary}")
