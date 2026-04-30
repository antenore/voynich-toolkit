"""
Firth vb101 Family Test — Phase 16c.

Tests Firth's claim that bigram-replacement transliterations (vb101 family)
yield substantively different views of the manuscript.

Method:
  1. Identify the top-N most frequent EVA bigrams (within words).
  2. Build several alternative transliterations by replacing each bigram with
     a single new character.
  3. Re-measure all 16 structural properties on each.
  4. Compare to baseline (v101) to see if any structural conclusion changes.

If structural properties are stable across transliterations, Firth's vb101
family is mostly a notational rewrite — it doesn't reveal new structure.

Reference:
  Firth, R.H. "A Voynich strategy". Reddit r/voynich, 2024.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import click

from .config import ToolkitConfig
from .rugg_test import (
    measure_all_properties,
    _serialise,
    _compact_dict,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words

TOP_N_BIGRAMS = 15
# Single-char replacements for bigrams (keep outside EVA alphabet)
# Use Greek letters + a few extras
REPLACEMENT_CHARS = "αβγδεζηθλμνξπρστφχψω"


def get_top_bigrams(pages: list[dict], top_n: int) -> list[tuple[str, int]]:
    """Get the top-N most frequent within-word bigrams."""
    counter: Counter = Counter()
    for page in pages:
        for word in page["words"]:
            for i in range(len(word) - 1):
                counter[word[i:i + 2]] += 1
    return counter.most_common(top_n)


def apply_bigram_replacement(
    pages: list[dict],
    bigram: str,
    replacement: str,
) -> list[dict]:
    """Apply a single bigram replacement greedily (left-to-right) to all words."""
    new_pages = []
    for page in pages:
        new_words = [_replace_bigram(w, bigram, replacement) for w in page["words"]]
        new_line_words = [
            [_replace_bigram(w, bigram, replacement) for w in line]
            for line in page.get("line_words", [])
        ]
        new_pages.append({
            "folio": page["folio"],
            "section": page["section"],
            "language": page["language"],
            "hand": page.get("hand", "?"),
            "words": new_words,
            "line_words": new_line_words,
        })
    return new_pages


def apply_cumulative_replacement(
    pages: list[dict],
    bigrams: list[str],
    replacements: list[str],
) -> list[dict]:
    """Apply a list of bigram replacements in order to all words."""
    new_pages = pages
    for bg, rep in zip(bigrams, replacements):
        new_pages = apply_bigram_replacement(new_pages, bg, rep)
    return new_pages


def _replace_bigram(word: str, bigram: str, replacement: str) -> str:
    """Replace bigram occurrences in a word, left to right, non-overlapping."""
    return word.replace(bigram, replacement)


def _key_metric_summary(props: dict) -> dict:
    """Extract scalar key metrics for compact comparison."""
    return {
        "line_self_containment": props["line_self_containment"]["cross_rate"],
        "m_line_final_rate": props["m_end_marker"]["line_final_rate"],
        "gallows_para_diff": props["gallows_para_start"]["difference"],
        "para_coherence_ratio": props["para_coherence"]["ratio"],
        "word_section_mi": props["word_section_mi"]["mi_bits"],
        "currier_jaccard": props["currier_diff"]["jaccard"],
        "zipf_slope": props["zipf"]["slope"],
        "entropy_bits": props["entropy"]["entropy_bits"],
        "slot_grammar_v": props["slot_grammar"]["cramers_v"],
        "ttr": props["vocabulary"]["ttr"],
        "n_types": props["vocabulary"]["n_types"],
    }


def format_summary(report: dict) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 16c — FIRTH vb101 FAMILY TEST")
    lines.append("Do bigram-replacement transliterations change structural conclusions?")
    lines.append("=" * 78)
    lines.append("")

    lines.append(f"Top {len(report['top_bigrams'])} bigrams (frequency-ordered):")
    for bg, c in report["top_bigrams"]:
        lines.append(f"  {bg}: {c:,}")
    lines.append("")

    # Header for comparison table
    base = report["baseline_metrics"]
    metrics = list(base.keys())

    lines.append("Per-replacement structural metrics (variant → metrics):")
    lines.append("-" * 78)
    header = f"{'variant':<20s}"
    for m in ["m_final", "MI", "gall.dif", "zipf", "entropy", "slot_v", "ttr", "n_types"]:
        header += f" {m:>9s}"
    lines.append(header)

    def fmt_row(name: str, mets: dict) -> str:
        row = f"{name:<20s}"
        row += f" {mets['m_line_final_rate']:>9.3f}"
        row += f" {mets['word_section_mi']:>9.3f}"
        row += f" {mets['gallows_para_diff']:>9.3f}"
        row += f" {mets['zipf_slope']:>9.3f}"
        row += f" {mets['entropy_bits']:>9.2f}"
        row += f" {mets['slot_grammar_v']:>9.3f}"
        row += f" {mets['ttr']:>9.3f}"
        row += f" {mets['n_types']:>9d}"
        return row

    lines.append(fmt_row("v101 (baseline)", base))
    for v in report["vb101_variants"]:
        lines.append(fmt_row(f"vb101-{v['index']:02d} ({v['bigram']}→x)", v["metrics"]))
    lines.append(fmt_row("cumulative-all", report["cumulative_metrics"]))
    lines.append("")

    # Diagnostics: how much did each metric move?
    lines.append("=" * 78)
    lines.append("METRIC STABILITY ANALYSIS")
    lines.append("=" * 78)
    lines.append("")
    lines.append("For each metric, max deviation from baseline across all vb101 variants:")
    lines.append("")

    deviations: dict[str, float] = {}
    for m in base:
        if not isinstance(base[m], (int, float)):
            continue
        max_dev = 0.0
        for v in report["vb101_variants"]:
            val = v["metrics"].get(m, base[m])
            if isinstance(val, (int, float)):
                dev = abs(val - base[m])
                if dev > max_dev:
                    max_dev = dev
        deviations[m] = max_dev

    for m in sorted(deviations, key=lambda x: -deviations[x]):
        bv = base[m]
        dev = deviations[m]
        rel = dev / abs(bv) if isinstance(bv, (int, float)) and bv else 0
        lines.append(f"  {m:<28s} baseline={bv!s:>10s}  max_dev={dev:.4f}  ({rel:.1%})")
    lines.append("")

    # Verdict
    lines.append("=" * 78)
    lines.append("VERDICT")
    lines.append("=" * 78)

    # Are structural conclusions stable?
    qualitatively_stable = (
        deviations.get("m_line_final_rate", 0) < 0.10 and
        deviations.get("word_section_mi", 0) < 0.20 and
        deviations.get("slot_grammar_v", 0) < 0.10
    )
    if qualitatively_stable:
        lines.append("STRUCTURAL CONCLUSIONS STABLE across vb101 variants.")
        lines.append("Bigram-replacement transliterations are notational rewrites.")
        lines.append("They do not reveal new structure beyond v101.")
    else:
        lines.append("SOME STRUCTURAL METRICS SHIFT under vb101 transliterations.")
        lines.append("Bigram replacement may affect specific properties — review above.")
    lines.append("")
    lines.append("Note: TTR and n_types necessarily shift because bigram replacement")
    lines.append("changes word lengths. The relevant metrics for structural inference")
    lines.append("are 'm' line-final, MI, slot grammar, and gallows para-start.")
    lines.append("")

    return "\n".join(lines) + "\n"


def save_to_db(config: ToolkitConfig, report: dict):
    """Save vb101 results to SQLite."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS firth_vb101")
    cur.execute("""
        CREATE TABLE firth_vb101 (
            variant TEXT,
            bigram TEXT,
            metric TEXT,
            value REAL,
            baseline REAL,
            deviation REAL,
            PRIMARY KEY (variant, metric)
        )
    """)
    base = report["baseline_metrics"]
    for v in report["vb101_variants"]:
        for m, val in v["metrics"].items():
            if not isinstance(val, (int, float)):
                continue
            bv = base.get(m, 0)
            dev = abs(val - bv) if isinstance(bv, (int, float)) else 0
            cur.execute(
                "INSERT INTO firth_vb101 VALUES (?, ?, ?, ?, ?, ?)",
                (f"vb101-{v['index']:02d}", v["bigram"], m,
                 float(val), float(bv) if isinstance(bv, (int, float)) else 0,
                 float(dev)),
            )
    # Cumulative
    for m, val in report["cumulative_metrics"].items():
        if not isinstance(val, (int, float)):
            continue
        bv = base.get(m, 0)
        dev = abs(val - bv) if isinstance(bv, (int, float)) else 0
        cur.execute(
            "INSERT INTO firth_vb101 VALUES (?, ?, ?, ?, ?, ?)",
            ("cumulative-all", "ALL", m,
             float(val), float(bv) if isinstance(bv, (int, float)) else 0,
             float(dev)),
        )
    conn.commit()
    conn.close()


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 16c: Firth vb101 family test."""
    report_path = config.stats_dir / "firth_vb101.json"
    summary_path = config.stats_dir / "firth_vb101_summary.txt"

    if report_path.exists() and not force:
        click.echo("  vb101 test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 16c — Firth vb101 Family Test")

    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    print_step(f"Extracting top-{TOP_N_BIGRAMS} bigrams...")
    top_bigrams = get_top_bigrams(real_pages, TOP_N_BIGRAMS)
    for bg, c in top_bigrams[:10]:
        click.echo(f"    {bg}: {c:,}")

    print_step("Measuring 16 properties on BASELINE (v101)...")
    baseline_props = measure_all_properties(real_pages)
    baseline_metrics = _key_metric_summary(baseline_props)
    for k, v in baseline_metrics.items():
        click.echo(f"    {k}: {v}")

    print_step(f"Generating {len(top_bigrams)} vb101 variants...")
    variants = []
    for i, (bg, _) in enumerate(top_bigrams):
        rep_char = REPLACEMENT_CHARS[i % len(REPLACEMENT_CHARS)]
        variant_pages = apply_bigram_replacement(real_pages, bg, rep_char)
        variant_props = measure_all_properties(variant_pages)
        variant_metrics = _key_metric_summary(variant_props)
        variants.append({
            "index": i + 1,
            "bigram": bg,
            "replacement": rep_char,
            "metrics": variant_metrics,
        })
        click.echo(
            f"    vb101-{i + 1:02d} ({bg} -> {rep_char}): "
            f"m_final={variant_metrics['m_line_final_rate']:.3f}, "
            f"MI={variant_metrics['word_section_mi']:.3f}, "
            f"slot_v={variant_metrics['slot_grammar_v']:.3f}"
        )

    print_step("Generating cumulative variant (all top bigrams replaced)...")
    bigrams = [bg for bg, _ in top_bigrams]
    replacements = [REPLACEMENT_CHARS[i % len(REPLACEMENT_CHARS)]
                    for i in range(len(bigrams))]
    cumulative_pages = apply_cumulative_replacement(
        real_pages, bigrams, replacements
    )
    cumulative_props = measure_all_properties(cumulative_pages)
    cumulative_metrics = _key_metric_summary(cumulative_props)
    click.echo(
        f"    cumulative: m_final={cumulative_metrics['m_line_final_rate']:.3f}, "
        f"MI={cumulative_metrics['word_section_mi']:.3f}, "
        f"slot_v={cumulative_metrics['slot_grammar_v']:.3f}"
    )

    print_step("Saving results...")
    report = {
        "top_bigrams": top_bigrams,
        "baseline_metrics": baseline_metrics,
        "baseline_full_properties": _serialise(baseline_props),
        "vb101_variants": variants,
        "cumulative_metrics": cumulative_metrics,
        "cumulative_full_properties": _serialise(cumulative_props),
        "parameters": {
            "top_n_bigrams": TOP_N_BIGRAMS,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(report)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, report)
    click.echo(f"    DB:   firth_vb101 table")

    click.echo(f"\n{summary}")
