"""
Phase 2e — Sub-analysis of hand ? by section.

Hand ? has entropy +12.25σ above the null (Phase 2) and is the only hand covering
ALL sections and all Currier languages. This is consistent with an aggregate
of multiple scribes not identified by Davis.

This module tests whether the anomaly is a mixing artifact or intrinsic property:

  H_mixing: each sub-corpus ?-per-section has normal entropy (z≈0).
            The anomaly is entirely due to combining different styles.
  H_intrinsic: even within a single section entropy remains high.
               Hand ? is structurally different from the others.

For each section present in ?:
  - Shannon entropy + z-score vs null model (N = sub-corpus size)
  - Zipf slope + z-score
  - Chi-square bigrams vs global corpus
  - Comparison with the same section in Davis hands 1–5

Post-analysis decision (documented in report):
  - If mixing confirmed → treat ?-per-section in Phases 4 and 5
  - If intrinsic anomaly → include ? as a special case

Output:
  hand_unknown.json
  hand_unknown_summary.txt
  DB table: hand_unknown_sections
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


# =====================================================================
# Core analysis
# =====================================================================

def split_unknown_by_section(corpus: dict) -> dict:
    """Split pages of hand ? by section.

    Returns: dict[section_code] → list of words
    """
    unknown_pages = corpus.get("?", {}).get("pages", [])
    by_section: dict[str, list[str]] = {}
    for p in unknown_pages:
        sec = p.get("section", "?")
        if sec not in by_section:
            by_section[sec] = []
        by_section[sec].extend(p["words"])
    return by_section


def section_entropy_vs_null(words: list[str], all_words: list[str],
                             section: str, seed_offset: int = 0) -> dict:
    """Compute entropy + Zipf with null model for a sub-corpus.

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
    """Compute mean entropy of the same section in Davis hands 1–5.

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
# Verdict
# =====================================================================

def compute_verdict(by_section_results: dict) -> dict:
    """Determine whether the anomaly is mixing or intrinsic.

    Criteria:
    - mixing_confirmed: all sections with > 200 tokens have |z_entropy| < 2
    - intrinsic_anomaly: at least one section with > 200 tokens has |z_entropy| > 3
    """
    large_sections = {
        sec: d for sec, d in by_section_results.items()
        if not d.get("skipped") and d.get("n_tokens", 0) >= 200
    }

    if not large_sections:
        return {"verdict": "INSUFFICIENT_DATA", "n_sections_tested": 0}

    z_scores = [d["z_entropy"] for d in large_sections.values()
                if d.get("z_entropy") is not None]

    if not z_scores:
        return {"verdict": "INSUFFICIENT_DATA", "n_sections_tested": len(large_sections)}

    max_abs_z = max(abs(z) for z in z_scores)
    n_anomalous = sum(1 for z in z_scores if abs(z) > 2)
    n_normal = sum(1 for z in z_scores if abs(z) <= 2)

    if n_anomalous == 0:
        verdict = "MIXING_CONFIRMED"
        explanation = (
            "All sections of ? have normal entropy vs null. "
            "The overall anomaly is an artifact of combining different styles. "
            "Recommendation: in Phases 4 and 5, use ?-per-section or exclude ?."
        )
    elif n_anomalous >= 2:
        verdict = "INTRINSIC_ANOMALY"
        explanation = (
            f"{n_anomalous} sections with |z| > 2. "
            "Hand ? has anomalous structure even within individual sections. "
            "Recommendation: include ? as a special case in comparative analyses."
        )
    else:
        verdict = "MIXED_EVIDENCE"
        explanation = (
            f"1 anomalous section out of {len(large_sections)}. "
            "Inconclusive evidence — could be a section with few pages."
        )

    return {
        "verdict":            verdict,
        "explanation":        explanation,
        "n_sections_tested":  len(large_sections),
        "n_anomalous":        n_anomalous,
        "n_normal":           n_normal,
        "max_abs_z_entropy":  round(max_abs_z, 2),
        "z_scores":           {sec: round(d["z_entropy"], 2)
                               for sec, d in large_sections.items()
                               if d.get("z_entropy") is not None},
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(by_section: dict, verdict: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS hand_unknown_sections")
    cur.execute("""
        CREATE TABLE hand_unknown_sections (
            section           TEXT PRIMARY KEY,
            section_name      TEXT,
            n_tokens          INTEGER,
            n_unique          INTEGER,
            entropy_obs       REAL,
            entropy_null_mean REAL,
            z_entropy         REAL,
            zipf_obs          REAL,
            z_zipf            REAL,
            bigram_chi2       REAL,
            bigram_p          REAL,
            unstable_flag     INTEGER,
            verdict           TEXT
        )
    """)

    overall_verdict = verdict.get("verdict", "?")
    for sec, d in sorted(by_section.items()):
        if d.get("skipped"):
            continue
        bg = d.get("bigrams", {})
        cur.execute("""
            INSERT INTO hand_unknown_sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sec,
            SECTION_NAMES.get(sec, sec),
            d.get("n_tokens"),
            d.get("n_unique"),
            d.get("entropy_obs"),
            d.get("entropy_null_mean"),
            d.get("z_entropy"),
            d.get("zipf_obs"),
            d.get("z_zipf"),
            bg.get("chi2") if not bg.get("skipped") else None,
            bg.get("p_value") if not bg.get("skipped") else None,
            int(d.get("unstable_flag", False)),
            overall_verdict,
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(by_section: dict, davis_comparison: dict, verdict: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 2e — Sub-analysis of hand ? by section")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Question: is the +12σ entropy anomaly of hand ? an artifact of")
    lines.append("  mixing different styles, or an intrinsic property of the scribe?")

    lines.append("\n── Entropy by section (hand ?) vs null model ──")
    lines.append(
        f"  {'Sec':>4}  {'Name':>16}  {'N':>6}  {'H_obs':>7}  "
        f"{'H_null':>7}  {'z_H':>6}  {'z_Zipf':>7}  chi2   Flag"
    )
    lines.append("  " + "-" * 78)

    for sec in sorted(by_section.keys()):
        d = by_section[sec]
        if d.get("skipped"):
            lines.append(f"  {sec:>4}  [skip: {d.get('reason', '?')}]")
            continue
        z_h = f"{d['z_entropy']:+.2f}" if d.get("z_entropy") is not None else "  n/a"
        z_z = f"{d['z_zipf']:+.2f}" if d.get("z_zipf") is not None else "  n/a"
        bg = d.get("bigrams", {})
        chi2_str = f"{bg['chi2']:6.0f}" if not bg.get("skipped") and bg.get("chi2") else "   n/a"
        flag = "⚠ unstab." if d["unstable_flag"] else ""
        sec_name = SECTION_NAMES.get(sec, sec)[:16]
        lines.append(
            f"  {sec:>4}  {sec_name:>16}  {d['n_tokens']:>6,}  "
            f"{d['entropy_obs']:>7.4f}  {d['entropy_null_mean']:>7.4f}  "
            f"{z_h:>6}  {z_z:>7}  {chi2_str}  {flag}"
        )

    # Comparison with Davis hands in the same section
    lines.append("\n── Entropy comparison same section: hand ? vs Davis hands ──")
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
    lines.append(f"\n── Verdict ──")
    lines.append(f"  {verdict['verdict']}")
    lines.append(f"  {verdict.get('explanation', '')}")
    if "z_scores" in verdict:
        z_str = "  ".join(f"{sec}:{z:+.2f}" for sec, z in sorted(verdict["z_scores"].items()))
        lines.append(f"  z per section: {z_str}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 2e: sub-analysis of hand ? by section — mixing or intrinsic anomaly?"""
    report_path = config.stats_dir / "hand_unknown.json"
    summary_path = config.stats_dir / "hand_unknown_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_unknown report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 2e — Sub-analysis of Hand ? by Section")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    all_words = [w for p in pages for w in p["words"]]
    click.echo(f"    {len(all_words):,} total words")

    # 2. Split by hand
    print_step("Splitting by hand and by section...")
    corpus = split_corpus_by_hand(pages)
    unknown_by_section = split_unknown_by_section(corpus)
    click.echo(f"    Hand ?: {len(unknown_by_section)} sections — "
               + ", ".join(f"{sec}({len(w)})" for sec, w in
                           sorted(unknown_by_section.items())))

    # 3. Analysis per section
    print_step("Entropy + Zipf per section of ? (500 null samples each)...")
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
            flag = " ⚠ unstable" if result["unstable_flag"] else ""
            click.echo(f" H={result['entropy_obs']:.4f} z={z_h}{flag}")

    # 4. Comparison with Davis hands in the same section
    print_step("Comparison with Davis hands in the same section...")
    davis_comparison: dict[str, dict] = {}
    for sec in sorted(unknown_by_section.keys()):
        davis_profiles = section_profile_for_davis_hands(corpus, sec, all_words)
        if davis_profiles:
            davis_comparison[sec] = davis_profiles
            for hand, v in sorted(davis_profiles.items()):
                click.echo(f"    Section {sec} Hand {hand}: "
                           f"n={v['n_tokens']} H={v['entropy']:.4f}")

    # 5. Verdict
    print_step("Computing verdict...")
    verdict = compute_verdict(by_section)
    click.echo(f"    VERDICT: {verdict['verdict']}")
    click.echo(f"    {verdict.get('explanation', '')}")

    # 6. Save JSON
    print_step("Saving JSON...")
    report = {
        "by_section":       by_section,
        "davis_comparison": davis_comparison,
        "verdict":          verdict,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 7. Save TXT
    summary = format_summary(by_section, davis_comparison, verdict)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. Save to DB
    print_step("Writing DB table hand_unknown_sections...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(by_section, verdict, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found at {db_path} — skip DB write")

    click.echo(f"\n{summary}")
