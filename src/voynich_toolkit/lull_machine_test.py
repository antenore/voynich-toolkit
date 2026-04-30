"""
Lullian Combinatoric Machine Test — Phase 17.

Tests whether a Ramon Llull-style combinatoric generator can reproduce the
16 structural properties of the Voynich Manuscript.

Llull's Ars Magna (c.1305-1308) used concentric rotating disks. Each disk
held a set of letter/symbol categories. Rotation produced combinations:
e.g., (Goodness, is, Great), (Greatness, is, Good).

Our generative model:
  1. K disks (one per word slot), each with M glyph-chunk options
  2. Per-section disk sets (different "machines" for different topics)
  3. Per-scribe rotation distribution (operators with different habits)
  4. Word = concatenation of glyph-chunks at selected disk positions
  5. Page/section/line structure matched to real manuscript

Predictions:
  PASS slot grammar (disk = slot, by construction), low entropy (few
  rotations), section MI (per-section disks), Zipf (non-uniform rotation
  preference), scribe bigram differences.
  FAIL line-end markers ('m', 'g', 'f' concentration — disks are line-blind
  unless we explicitly add a terminator disk).
  UNCERTAIN gallows-at-paragraph-start (would need paragraph-aware machine).

Reference:
  Ramon Llull, *Ars Magna* (1305-1308). Documented in Eco, U.,
  *The Search for the Perfect Language* (1995), ch. 4.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click

from .config import ToolkitConfig
from .rugg_test import (
    compare_properties,
    measure_all_properties,
    _serialise,
    _compact_dict,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_RUNS = 3
N_DISKS = 5  # word slots: prefix, core1, core2, core3, suffix

# EVA glyph-chunks per disk slot. Each list = the "letters" on that disk.
# These are derived from observed EVA syllable patterns but rotation-friendly.

DISK_TEMPLATES = {
    "prefix": ["", "q", "qo", "ch", "sh", "d", "y", "ok", "ot", "ol", "o"],
    "core1":  ["o", "a", "e", "y", "ch", "sh", "ke", "te", "pe", "fe"],
    "core2":  ["e", "ee", "ai", "ar", "or", "ol", "in", "iin", "edy"],
    "core3":  ["", "d", "k", "t", "p", "f", "ch", "sh"],
    "suffix": ["", "y", "dy", "n", "m", "in", "iin", "ain", "g", "s", "l"],
}


def build_section_disks(
    rng: random.Random,
    sections: list[str],
    n_options_per_disk: int = 8,
) -> dict[str, dict[str, list[str]]]:
    """For each section, pick a subset of glyph-chunks for each disk.

    Different sections get different disk subsets (= different machines).
    This is what produces section-conditional vocabulary.
    """
    section_disks: dict[str, dict[str, list[str]]] = {}
    for section in sections:
        disks: dict[str, list[str]] = {}
        for slot, full_options in DISK_TEMPLATES.items():
            n = min(n_options_per_disk, len(full_options))
            chosen = rng.sample(full_options, n)
            disks[slot] = chosen
        section_disks[section] = disks
    return section_disks


def build_scribe_rotation_bias(
    rng: random.Random,
    scribes: list[str],
    n_options: int = 8,
) -> dict[str, list[float]]:
    """For each scribe, a rotation-position preference distribution.

    Scribes share the disks but rotate them differently. This produces
    per-scribe statistical signatures (different bigrams, etc.).
    """
    biases = {}
    for scribe in scribes:
        # Generate a non-uniform preference (Zipf-ish) over disk positions
        weights = [1.0 / (i + 1) ** rng.uniform(0.5, 1.5) for i in range(n_options)]
        biases[scribe] = weights
    return biases


def generate_word(
    disks: dict[str, list[str]],
    rotation_bias: list[float],
    rng: random.Random,
) -> str:
    """Generate one word by picking a rotation for each disk."""
    parts = []
    for slot in ["prefix", "core1", "core2", "core3", "suffix"]:
        options = disks[slot]
        n = len(options)
        # Truncate or pad bias to match n options
        weights = rotation_bias[:n] if len(rotation_bias) >= n else (
            rotation_bias + [0.5] * (n - len(rotation_bias))
        )
        chunk = rng.choices(options, weights=weights, k=1)[0]
        parts.append(chunk)
    word = "".join(parts)
    if not word:
        word = "o"
    return word


def generate_lull_corpus(
    real_pages: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Generate a Llull-style synthetic corpus matching real structure."""
    sections = sorted({p["section"] for p in real_pages})
    scribes = sorted({p.get("hand", "?") for p in real_pages})

    section_disks = build_section_disks(rng, sections, n_options_per_disk=8)
    scribe_biases = build_scribe_rotation_bias(rng, scribes, n_options=8)

    synthetic_pages = []
    for page in real_pages:
        section = page["section"]
        scribe = page.get("hand", "?")
        disks = section_disks[section]
        bias = scribe_biases[scribe]

        syn_line_words = []
        syn_words = []
        for line in page.get("line_words", []):
            n_words = len(line)
            syn_line = []
            for _ in range(n_words):
                w = generate_word(disks, bias, rng)
                syn_line.append(w)
            syn_line_words.append(syn_line)
            syn_words.extend(syn_line)

        synthetic_pages.append({
            "folio": page["folio"],
            "section": page["section"],
            "language": page["language"],
            "hand": page.get("hand", "?"),
            "words": syn_words,
            "line_words": syn_line_words,
        })

    return synthetic_pages


def run_lull_multi(
    real_pages: list[dict],
    n_runs: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Run N Llull generations, return averaged properties."""
    all_props = []
    for i in range(n_runs):
        rng = random.Random(seed + i)
        syn_pages = generate_lull_corpus(real_pages, rng)
        props = measure_all_properties(syn_pages)
        all_props.append(props)
        click.echo(
            f"    Run {i + 1}/{n_runs}: "
            f"entropy={props['entropy']['entropy_bits']:.2f}, "
            f"TTR={props['vocabulary']['ttr']:.4f}, "
            f"m_final={props['m_end_marker']['line_final_rate']:.3f}, "
            f"MI={props['word_section_mi']['mi_bits']:.3f}, "
            f"slot_v={props['slot_grammar']['cramers_v']:.3f}"
        )

    averaged = _average_properties(all_props)
    return averaged, all_props


def _average_properties(all_props: list[dict]) -> dict:
    """Average measurement dicts across runs (same as naibbe/firth tests)."""
    if not all_props:
        return {}
    if len(all_props) == 1:
        return all_props[0]
    averaged = {}
    for key in all_props[0]:
        vals = [p[key] for p in all_props]
        if isinstance(vals[0], dict):
            avg_dict = {}
            for subkey in vals[0]:
                subvals = [v.get(subkey) for v in vals]
                if all(isinstance(sv, (int, float))
                       for sv in subvals if sv is not None):
                    numeric = [sv for sv in subvals if sv is not None]
                    avg_dict[subkey] = (
                        round(sum(numeric) / len(numeric), 6) if numeric else 0
                    )
                else:
                    avg_dict[subkey] = subvals[0]
            averaged[key] = avg_dict
        else:
            averaged[key] = vals[0]
    return averaged


def save_to_db(config: ToolkitConfig, comparisons: list[dict]):
    """Save Llull results to SQLite."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS lull_machine_test")
    cur.execute("""
        CREATE TABLE lull_machine_test (
            property TEXT PRIMARY KEY,
            description TEXT,
            real_value REAL,
            lull_value REAL,
            reproduced INTEGER,
            direction TEXT,
            threshold REAL
        )
    """)
    for c in comparisons:
        rv = float(c["real_value"]) if isinstance(c["real_value"], (int, float)) else 0
        lv = float(c["rugg_value"]) if isinstance(c["rugg_value"], (int, float)) else 0
        cur.execute(
            "INSERT INTO lull_machine_test VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c["property"], c["description"], rv, lv,
             1 if c["reproduced"] else 0, c["direction"], c["threshold"]),
        )
    conn.commit()
    conn.close()


def format_summary(
    real_props: dict,
    lull_props: dict,
    comparisons: list[dict],
    n_runs: int,
) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 17 — LULLIAN COMBINATORIC MACHINE TEST")
    lines.append("Can a Ramon Llull-style rotating-disk generator reproduce")
    lines.append("the 16 confirmed structural properties of the Voynich MS?")
    lines.append("=" * 72)
    lines.append("")

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)

    lines.append(f"Result: {n_repro}/{n_total} properties reproduced")
    lines.append(f"(averaged over {n_runs} machine instantiations)")
    lines.append("")
    lines.append("Reference: Rugg grille 10/16, Naibbe cipher 8/16,")
    lines.append("           Firth sorting cipher 11/16.")
    lines.append("")

    lines.append(f"{'Property':<30s} {'Real':>10s} {'Lull':>10s} {'OK?':>10s}")
    lines.append("-" * 72)
    for c in comparisons:
        rv = c["real_value"]
        lv = c["rugg_value"]
        rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
        lv_str = f"{lv:.4f}" if isinstance(lv, float) else str(lv)
        verdict = "YES" if c["reproduced"] else "** NO **"
        lines.append(f"  {c['property']:<28s} {rv_str:>10s} {lv_str:>10s} {verdict:>10s}")
    lines.append("")

    # Verdict
    lines.append("=" * 72)
    lines.append("VERDICT")
    lines.append("=" * 72)
    missed = [c for c in comparisons if not c["reproduced"]]
    if n_repro >= 14:
        lines.append(f"LULLIAN MACHINE STRONG: {n_repro}/{n_total}.")
        lines.append("Combinatoric generation explains most VMS structure.")
    elif n_repro >= 12:
        lines.append(f"LULLIAN MACHINE COMPETITIVE: {n_repro}/{n_total}.")
        lines.append("Best result so far among tested generators.")
    elif n_repro >= 10:
        lines.append(f"LULLIAN MACHINE COMPARABLE: {n_repro}/{n_total}.")
        lines.append("Similar to Rugg/Firth, better than Naibbe.")
    else:
        lines.append(f"LULLIAN MACHINE INSUFFICIENT: {n_repro}/{n_total}.")
        lines.append("Combinatoric model alone cannot generate VMS.")
    lines.append("")
    if missed:
        lines.append("Properties NOT reproduced:")
        for m in missed:
            lines.append(f"  - {m['property']}: real={m['real_value']}, "
                         f"lull={m['rugg_value']}")
        lines.append("")

    lines.append("Note on architectural failures expected:")
    lines.append("  Disks are line-blind by construction. Line-end characters")
    lines.append("  ('m', 'g', 'f' concentration) and gallows-at-paragraph-start")
    lines.append("  cannot be reproduced without a line-position-aware extension.")
    lines.append("")

    return "\n".join(lines) + "\n"


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 17: Lullian combinatoric machine test."""
    report_path = config.stats_dir / "lull_machine_test.json"
    summary_path = config.stats_dir / "lull_machine_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Lull test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 17 — Lullian Combinatoric Machine Test")

    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    print_step("Measuring 16 properties on REAL corpus...")
    real_props = measure_all_properties(real_pages)
    for key, val in real_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    print_step(f"Generating Llull corpus ({N_RUNS} runs, "
               f"{N_DISKS} disks per word)...")
    lull_props, all_runs = run_lull_multi(real_pages, N_RUNS, SEED)

    click.echo(f"\n    Averaged Llull properties:")
    for key, val in lull_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    print_step("Comparing real vs Llull...")
    comparisons = compare_properties(real_props, lull_props)

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)
    click.echo(f"\n    RESULT: {n_repro}/{n_total} properties reproduced")

    for c in comparisons:
        icon = "OK" if c["reproduced"] else "MISS"
        click.echo(f"    [{icon:>4s}] {c['property']:<28s} "
                   f"real={c['real_value']:<10} lull={c['rugg_value']:<10}")

    print_step("Saving results...")
    report = {
        "real_properties": _serialise(real_props),
        "lull_properties": _serialise(lull_props),
        "comparisons": comparisons,
        "summary": {
            "reproduced": n_repro,
            "total": n_total,
            "verdict": "SUFFICIENT" if n_repro == n_total else "INSUFFICIENT",
            "missed": [c["property"] for c in comparisons if not c["reproduced"]],
        },
        "parameters": {
            "n_disks": N_DISKS,
            "n_runs": N_RUNS,
            "seed": SEED,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(real_props, lull_props, comparisons, N_RUNS)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, comparisons)
    click.echo(f"    DB:   lull_machine_test table")

    click.echo(f"\n{summary}")
