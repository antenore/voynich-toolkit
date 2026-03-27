"""
Phase 6 — Register/inventory structure test.

Hypothesis: some sections of the Voynich manuscript are not running prose
but structured registers (like medieval inventories or tables):

  pepper    ●●●●●●●
  cinnamon  ●●●
  saffron   ●●●●●●●●●●

This would explain the extremely low entropy found in sections ?S, ?P, ?Z
(Phase 2/2e): a tally/quantity marker repeated many times per line would
dominate the token frequency and collapse entropy.

This is a mathematical test, NOT a claim about content.

Three independent structural tests (all on raw EVA, no decoding):

  6a — Line-initial uniqueness
       In a register, each line starts with a unique label word.
       Metric: fraction of lines whose first word is a hapax (unique to that line set).
       Null: same metric on randomly shuffled lines.

  6b — Token concentration
       In a register, 1-3 tally tokens account for most of the text.
       Metric: fraction of total tokens covered by top-3 most frequent words.
       Null: same on random permutations of the word list.

  6c — Intra-line repetition
       In a register, words after the first on a line are often identical (tally marks).
       Metric: fraction of "continuation" words (position >= 2 on line) that equal
       the most frequent word on that line.
       Null: random reassignment of words to positions within pages.

All metrics compared section-by-section. Sections predicted to score highest
as registers: P (pharmaceutical), S (astronomical), Z (zodiac).
Sections predicted lowest: T (text), H (herbal prose).

Output:
  register_test.json
  register_test_summary.txt
  DB table: register_test
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .scribe_analysis import split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SEED = 42
N_NULL = 500   # permutations for null model


# =====================================================================
# Data preparation: collect lines per section
# =====================================================================

def lines_by_section(pages: list[dict]) -> dict[str, list[list[str]]]:
    """Group all manuscript lines by section code.

    Returns: dict[section] → list of lines, each line is list[str] of words.
    Excludes empty lines.
    """
    by_sec: dict[str, list[list[str]]] = {}
    for p in pages:
        sec = p.get("section", "?")
        for line in p.get("line_words", []):
            if len(line) >= 2:   # need at least 2 words for any test
                if sec not in by_sec:
                    by_sec[sec] = []
                by_sec[sec].append(line)
    return by_sec


# =====================================================================
# Phase 6a — Line-initial uniqueness
# =====================================================================

def line_initial_uniqueness(lines: list[list[str]]) -> float:
    """Fraction of lines whose first word appears ONLY as a line-initial word
    across the entire section (i.e., it is unique in the initial position).

    High value → every line starts with a different label (register-like).
    Low value → same words keep starting lines (prose-like).
    """
    initials = [line[0] for line in lines]
    freq = Counter(initials)
    n_unique_initials = sum(1 for c in freq.values() if c == 1)
    return n_unique_initials / len(initials) if initials else 0.0


def null_line_initial_uniqueness(lines: list[list[str]],
                                  n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle all words across lines (preserving line lengths),
    recompute uniqueness. Models 'what if words were placed randomly?'
    """
    rng = random.Random(seed)
    all_words = [w for line in lines for w in line]
    lengths = [len(line) for line in lines]
    nulls = []
    for _ in range(n_perms):
        rng.shuffle(all_words)
        pos = 0
        shuffled = []
        for ln in lengths:
            shuffled.append(all_words[pos:pos+ln])
            pos += ln
        nulls.append(line_initial_uniqueness(shuffled))
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std":  float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# Phase 6b — Token concentration
# =====================================================================

def token_concentration(lines: list[list[str]], top_n: int = 3) -> float:
    """Fraction of total tokens covered by the top-N most frequent words.

    High value → a few words dominate (tally-like).
    Low value → diverse vocabulary (prose-like).
    """
    words = [w for line in lines for w in line]
    if not words:
        return 0.0
    freq = Counter(words)
    top_count = sum(c for _, c in freq.most_common(top_n))
    return top_count / len(words)


def null_token_concentration(lines: list[list[str]],
                              all_words: list[str],
                              top_n: int = 3,
                              n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: sample len(section_words) tokens from the FULL corpus, recompute concentration.

    This models: 'if this section were a random slice of the full corpus,
    what concentration would we expect?' Sections with unusually few dominant
    words will score high; sections dominated by a few tokens will score low.
    """
    rng = random.Random(seed)
    n = sum(len(line) for line in lines)
    nulls = []
    for _ in range(n_perms):
        sample = rng.choices(all_words, k=n)
        freq = Counter(sample)
        top_count = sum(c for _, c in freq.most_common(top_n))
        nulls.append(top_count / n)
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std":  float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# Phase 6c — Intra-line repetition
# =====================================================================

def intra_line_repetition(lines: list[list[str]]) -> float:
    """Fraction of continuation words (position >= 2) that equal
    the most common word on their own line.

    High value → after the label, the line is mostly the same token (tally).
    Low value → varied words on each line (prose).

    Only counts lines with >= 3 words (need a label + at least 2 continuation words).
    """
    total_continuation = 0
    total_repeated = 0
    for line in lines:
        if len(line) < 3:
            continue
        continuation = line[1:]
        most_common = Counter(continuation).most_common(1)[0][0]
        repeated = sum(1 for w in continuation if w == most_common)
        total_continuation += len(continuation)
        total_repeated += repeated
    if total_continuation == 0:
        return 0.0
    return total_repeated / total_continuation


def null_intra_line_repetition(lines: list[list[str]],
                                n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: randomly reassign continuation words within each page,
    preserving line lengths. Tests whether observed repetition exceeds chance.
    """
    rng = random.Random(seed)
    nulls = []
    for _ in range(n_perms):
        shuffled = []
        # Collect all continuation words from this section and shuffle
        all_cont = [w for line in lines if len(line) >= 3
                    for w in line[1:]]
        rng.shuffle(all_cont)
        pos = 0
        for line in lines:
            if len(line) < 3:
                shuffled.append(line)
                continue
            n_cont = len(line) - 1
            new_line = [line[0]] + all_cont[pos:pos+n_cont]
            shuffled.append(new_line)
            pos += n_cont
            if pos >= len(all_cont):
                pos = 0
        nulls.append(intra_line_repetition(shuffled))
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std":  float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# Z-score helper
# =====================================================================

def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# Run all tests for one section
# =====================================================================

def analyze_section(section: str, lines: list[list[str]],
                    all_words: list[str] | None = None,
                    seed_offset: int = 0) -> dict:
    """Run all three tests for one section."""
    n_lines = len(lines)
    n_words = sum(len(l) for l in lines)

    if n_lines < 10:
        return {"section": section, "skipped": True,
                "reason": f"only {n_lines} lines (< 10)"}

    # 6a
    uniq_obs  = line_initial_uniqueness(lines)
    uniq_null = null_line_initial_uniqueness(lines, seed=SEED + seed_offset)
    z_uniq    = z_score(uniq_obs, uniq_null["null_mean"], uniq_null["null_std"])

    # 6b
    corpus = all_words if all_words is not None else [w for l in lines for w in l]
    conc_obs  = token_concentration(lines)
    conc_null = null_token_concentration(lines, corpus, seed=SEED + seed_offset + 1)
    z_conc    = z_score(conc_obs, conc_null["null_mean"], conc_null["null_std"])

    # 6c
    rep_obs  = intra_line_repetition(lines)
    rep_null = null_intra_line_repetition(lines, seed=SEED + seed_offset + 2)
    z_rep    = z_score(rep_obs, rep_null["null_mean"], rep_null["null_std"])

    # Register score: average of the three z-scores (positive = more register-like)
    zs = [z for z in [z_uniq, z_conc, z_rep] if z is not None]
    register_score = round(float(np.mean(zs)), 3) if zs else None

    return {
        "section":       section,
        "section_name":  SECTION_NAMES.get(section, section),
        "n_lines":       n_lines,
        "n_words":       n_words,
        "uniqueness": {
            "obs":       round(uniq_obs, 4),
            "null_mean": round(uniq_null["null_mean"], 4),
            "null_std":  round(uniq_null["null_std"], 6),
            "z":         round(z_uniq, 3) if z_uniq is not None else None,
        },
        "concentration": {
            "obs":       round(conc_obs, 4),
            "null_mean": round(conc_null["null_mean"], 4),
            "null_std":  round(conc_null["null_std"], 6),
            "z":         round(z_conc, 3) if z_conc is not None else None,
        },
        "repetition": {
            "obs":       round(rep_obs, 4),
            "null_mean": round(rep_null["null_mean"], 4),
            "null_std":  round(rep_null["null_std"], 6),
            "z":         round(z_rep, 3) if z_rep is not None else None,
        },
        "register_score": register_score,
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(results: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS register_test")
    cur.execute("""
        CREATE TABLE register_test (
            section          TEXT PRIMARY KEY,
            section_name     TEXT,
            n_lines          INTEGER,
            n_words          INTEGER,
            uniqueness_obs   REAL,
            uniqueness_null  REAL,
            z_uniqueness     REAL,
            concentration_obs REAL,
            concentration_null REAL,
            z_concentration  REAL,
            repetition_obs   REAL,
            repetition_null  REAL,
            z_repetition     REAL,
            register_score   REAL
        )
    """)
    for sec, d in sorted(results.items()):
        if d.get("skipped"):
            continue
        cur.execute("INSERT INTO register_test VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            sec, d["section_name"], d["n_lines"], d["n_words"],
            d["uniqueness"]["obs"],    d["uniqueness"]["null_mean"],    d["uniqueness"]["z"],
            d["concentration"]["obs"], d["concentration"]["null_mean"], d["concentration"]["z"],
            d["repetition"]["obs"],    d["repetition"]["null_mean"],    d["repetition"]["z"],
            d["register_score"],
        ))
    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(results: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 6 — Register/inventory structure test")
    lines.append("  H0: sections are running prose (no register structure)")
    lines.append("  H1: some sections show tally/inventory structure")
    lines.append("=" * 80)

    lines.append(
        f"\n  {'Sec':>4}  {'Name':>16}  {'Lines':>5}  "
        f"{'z_uniq':>7}  {'z_conc':>7}  {'z_rep':>7}  "
        f"{'Score':>7}  Verdict"
    )
    lines.append("  " + "-" * 72)

    sorted_secs = sorted(
        [d for d in results.values() if not d.get("skipped")],
        key=lambda d: d.get("register_score") or -99,
        reverse=True,
    )
    for d in sorted_secs:
        z_u = f"{d['uniqueness']['z']:+.2f}"   if d["uniqueness"]["z"]    is not None else "  n/a"
        z_c = f"{d['concentration']['z']:+.2f}" if d["concentration"]["z"] is not None else "  n/a"
        z_r = f"{d['repetition']['z']:+.2f}"   if d["repetition"]["z"]    is not None else "  n/a"
        score = d.get("register_score")
        score_str = f"{score:+.2f}" if score is not None else "  n/a"

        if score is not None and score > 2:
            verdict = "REGISTER-LIKE"
        elif score is not None and score < -1:
            verdict = "prose-like"
        else:
            verdict = "neutral"

        lines.append(
            f"  {d['section']:>4}  {d['section_name']:>16}  {d['n_lines']:>5}  "
            f"{z_u:>7}  {z_c:>7}  {z_r:>7}  {score_str:>7}  {verdict}"
        )

    lines.append("\n── What the scores mean ──")
    lines.append("  z_uniq:  how much more often each line starts with a unique word vs random")
    lines.append("  z_conc:  how much the top-3 words dominate the section vs random")
    lines.append("  z_rep:   how often continuation words repeat the same token vs random")
    lines.append("  Score:   average of the three z-scores (> +2 = strong register signal)")
    lines.append("\n── Predicted register sections: P (pharma), S (astro), Z (zodiac) ──")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 6: register/inventory structure test — 3 structural metrics per section."""
    report_path = config.stats_dir / "register_test.json"
    summary_path = config.stats_dir / "register_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  register_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 6 — Register/Inventory Structure Test")

    # 1. Parse EVA corpus (need line_words)
    print_step("Parsing EVA corpus (line-level)...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Group lines by section
    print_step("Grouping lines by section...")
    sec_lines = lines_by_section(pages)
    for sec in sorted(sec_lines.keys()):
        n = len(sec_lines[sec])
        nw = sum(len(l) for l in sec_lines[sec])
        click.echo(f"    {sec} ({SECTION_NAMES.get(sec, sec)}): "
                   f"{n} lines, {nw:,} words")

    # 3. Analyze each section
    print_step(f"Running 3 tests × {len(sec_lines)} sections "
               f"({N_NULL} null permutations each)...")
    all_words = [w for p in pages for w in p["words"]]

    results = {}
    for i, (sec, sec_ls) in enumerate(sorted(sec_lines.items())):
        click.echo(f"    Section {sec} ({SECTION_NAMES.get(sec, sec)}, "
                   f"{len(sec_ls)} lines)...", nl=False)
        result = analyze_section(sec, sec_ls, all_words=all_words, seed_offset=i * 10)
        results[sec] = result
        if result.get("skipped"):
            click.echo(f" skip")
        else:
            score = result.get("register_score")
            score_str = f"{score:+.3f}" if score is not None else "n/a"
            def _fmt(z): return f"{z:+.2f}" if z is not None else "n/a"
            click.echo(f" score={score_str}  "
                       f"z_uniq={_fmt(result['uniqueness']['z'])}  "
                       f"z_conc={_fmt(result['concentration']['z'])}  "
                       f"z_rep={_fmt(result['repetition']['z'])}")

    # 4. Save JSON
    print_step("Saving JSON...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 5. Save TXT
    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 6. Save to DB
    print_step("Writing DB table register_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(results, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
