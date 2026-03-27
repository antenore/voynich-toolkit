"""
Phase 4 — Positional patterns per hand (pure EVA, zero lexicon).

Does each hand have a "signature" in the position of EVA characters in words?

Hand ? is split by section (Phase 2e decision): each ?-section is treated
as an independent unit. Total units: ~12.

Sub-analyses:
  4a — Positional distribution of EVA chars (initial/medial/final)
       Chi-square each unit vs global corpus
  4b — Word length distribution + KS-test vs corpus
  4c — EVA trigrams: entropy and chi-square vs corpus

Output:
  hand_positional.json
  hand_positional_summary.txt
  DB table: hand_positional_chars, hand_positional_lengths, hand_positional_trigrams
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2 as scipy_chi2, ks_2samp

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .hand_characterization import eva_profile
from .hand_structure import SEED, bigram_freq
from .scribe_analysis import HAND_NAMES, split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Constants
# =====================================================================

MIN_TOKENS = 100   # minimum threshold to include a unit

# Label for ? split by section
UNKNOWN_SECTION_LABEL = "?{sec}"   # e.g. "?S", "?A"


# =====================================================================
# Construction of the ~12 analysis units
# =====================================================================

def build_analysis_units(corpus: dict, pages: list[dict]) -> dict[str, list[str]]:
    """Build the ~12 analysis units.

    - Davis hands 1–5: as from corpus
    - Hands X, Y: as from corpus
    - Hand ?: split by section → ?S, ?A, ?Z, ?P, ?C, ?T, ?H

    Returns: dict[unit_label] → list of EVA words
    """
    units: dict[str, list[str]] = {}

    # Davis hands + X + Y
    for hand, data in corpus.items():
        if hand == "?":
            continue
        words = data["words"]
        if len(words) >= MIN_TOKENS:
            units[hand] = words

    # Hand ? split by section
    unknown_pages = corpus.get("?", {}).get("pages", [])
    by_sec: dict[str, list[str]] = {}
    for p in unknown_pages:
        sec = p.get("section", "?")
        if sec not in by_sec:
            by_sec[sec] = []
        by_sec[sec].extend(p["words"])

    for sec, words in by_sec.items():
        if len(words) >= MIN_TOKENS:
            label = f"?{sec}"
            units[label] = words

    return units


def unit_label_name(label: str) -> str:
    """Human-readable name for a unit."""
    if label.startswith("?") and len(label) > 1:
        sec = label[1:]
        sec_name = SECTION_NAMES.get(sec, sec)
        return f"?-{sec_name}"
    return HAND_NAMES.get(label, label)


# =====================================================================
# Phase 4a — Positional distribution of EVA characters
# =====================================================================

def positional_char_freq(words: list[str]) -> dict[str, Counter]:
    """Compute frequency of EVA characters by position: initial, medial, final.

    Words of length 1: the character is both initial and final → count only initial.
    Words of length 2: initial + final, no medial.
    """
    pos: dict[str, Counter] = {
        "initial": Counter(),
        "medial":  Counter(),
        "final":   Counter(),
    }
    for w in words:
        if not w:
            continue
        n = len(w)
        pos["initial"][w[0]] += 1
        if n >= 2:
            pos["final"][w[-1]] += 1
        if n >= 3:
            for ch in w[1:-1]:
                pos["medial"][ch] += 1
    return pos


def chisquare_vs_global(unit_pos: dict[str, Counter],
                        global_pos: dict[str, Counter]) -> dict:
    """Chi-square for each position: unit vs global distribution.

    Uses all characters present in global with expected >= 5.
    Returns: dict[position] → {chi2, df, p_value, significant_05}
    """
    results = {}
    for position in ("initial", "medial", "final"):
        g = global_pos[position]
        u = unit_pos[position]

        g_total = sum(g.values())
        u_total = sum(u.values())
        if g_total == 0 or u_total == 0:
            results[position] = {"skipped": True, "reason": "no data"}
            continue

        # Common characters — use only those with expected >= 5
        chars = sorted(g.keys())
        g_freq = np.array([g[c] / g_total for c in chars])
        observed = np.array([u.get(c, 0) for c in chars], dtype=float)
        expected = g_freq * u_total

        mask = expected >= 5
        obs_f = observed[mask]
        exp_f = expected[mask]
        df = int(mask.sum()) - 1

        if df <= 0:
            results[position] = {"skipped": True, "reason": "too few cells"}
            continue

        chi2_stat = float(np.sum((obs_f - exp_f) ** 2 / exp_f))
        p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

        results[position] = {
            "chi2":           round(chi2_stat, 2),
            "df":             df,
            "p_value":        round(p_value, 6),
            "significant_05": bool(p_value < 0.05),
            "significant_001":bool(p_value < 0.001),
            "n_chars_used":   int(mask.sum()),
            "n_tokens":       int(u_total),
        }
    return results


def top_chars_per_position(pos: dict[str, Counter], top_n: int = 5) -> dict:
    """Top-N characters per position (for readability)."""
    return {
        position: [{"char": c, "count": n}
                   for c, n in counter.most_common(top_n)]
        for position, counter in pos.items()
    }


def run_positional_analysis(units: dict[str, list[str]]) -> dict:
    """Analysis 4a for all units.

    Returns: dict[unit] → {chi2_results, top_chars}
    """
    # Compute global distribution (entire corpus)
    all_words = [w for words in units.values() for w in words]
    global_pos = positional_char_freq(all_words)

    results = {}
    for label in sorted(units.keys()):
        words = units[label]
        unit_pos = positional_char_freq(words)
        chi2_res = chisquare_vs_global(unit_pos, global_pos)
        top = top_chars_per_position(unit_pos)
        results[label] = {
            "n_words":   len(words),
            "chi2":      chi2_res,
            "top_chars": top,
        }
    return results


# =====================================================================
# Phase 4b — Word length distribution + KS-test
# =====================================================================

def word_length_distribution(words: list[str]) -> Counter:
    return Counter(len(w) for w in words)


def ks_test_vs_global(unit_words: list[str], all_words: list[str]) -> dict:
    """KS-test: unit word length distribution vs global corpus."""
    unit_lens = [len(w) for w in unit_words]
    global_lens = [len(w) for w in all_words]
    stat, p_value = ks_2samp(unit_lens, global_lens)
    mean_len = float(np.mean(unit_lens)) if unit_lens else 0.0
    return {
        "n_words":        len(unit_words),
        "mean_length":    round(mean_len, 3),
        "ks_stat":        round(float(stat), 4),
        "ks_p_value":     round(float(p_value), 6),
        "significant_05": bool(p_value < 0.05),
        "significant_001":bool(p_value < 0.001),
        "length_dist":    dict(sorted(word_length_distribution(unit_words).items())),
    }


def run_length_analysis(units: dict[str, list[str]]) -> dict:
    """Analysis 4b for all units."""
    all_words = [w for words in units.values() for w in words]
    results = {}
    for label in sorted(units.keys()):
        results[label] = ks_test_vs_global(units[label], all_words)
    return results


# =====================================================================
# Phase 4c — EVA trigrams
# =====================================================================

def trigram_freq(words: list[str]) -> Counter:
    """Count EVA trigrams in a word list."""
    tg = Counter()
    for w in words:
        for i in range(len(w) - 2):
            tg[w[i:i+3]] += 1
    return tg


def trigram_entropy(tg: Counter) -> float:
    """Shannon entropy (bits) of the trigram distribution."""
    total = sum(tg.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in tg.values()], dtype=float)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def trigram_chisquare_vs_global(unit_tg: Counter, global_tg: Counter,
                                 top_n: int = 50) -> dict:
    """Chi-square unit trigram distribution vs corpus (top-N trigrams)."""
    top50 = [tg for tg, _ in global_tg.most_common(top_n)]
    g_total = sum(global_tg[tg] for tg in top50)
    if g_total == 0:
        return {"skipped": True, "reason": "no global trigrams"}

    g_freq = np.array([global_tg[tg] / g_total for tg in top50])
    u_total = sum(unit_tg.get(tg, 0) for tg in top50)
    if u_total == 0:
        return {"skipped": True, "reason": "no unit trigrams in top-50"}

    observed = np.array([unit_tg.get(tg, 0) for tg in top50], dtype=float)
    expected = g_freq * u_total
    mask = expected >= 5
    if mask.sum() < 2:
        return {"skipped": True, "reason": "too few cells with expected >= 5"}

    chi2_stat = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    df = int(mask.sum()) - 1
    p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

    return {
        "n_trigrams_total": int(u_total),
        "chi2":             round(chi2_stat, 2),
        "df":               df,
        "p_value":          round(p_value, 6),
        "significant_001":  bool(p_value < 0.001),
        "top5": [{"trigram": tg, "count": unit_tg.get(tg, 0)}
                 for tg in global_tg.most_common(5)],
    }


def run_trigram_analysis(units: dict[str, list[str]]) -> dict:
    """Analysis 4c for all units."""
    all_words = [w for words in units.values() for w in words]
    global_tg = trigram_freq(all_words)

    results = {}
    for label in sorted(units.keys()):
        words = units[label]
        unit_tg = trigram_freq(words)
        h_tg = trigram_entropy(unit_tg)
        chi2_res = trigram_chisquare_vs_global(unit_tg, global_tg)
        results[label] = {
            "n_words":         len(words),
            "n_trigrams":      sum(unit_tg.values()),
            "trigram_entropy": round(h_tg, 4),
            "top5_trigrams":   [{"trigram": tg, "count": c}
                                for tg, c in unit_tg.most_common(5)],
            "chi2":            chi2_res,
        }
    return results


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(positional: dict, lengths: dict, trigrams: dict,
               db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Table 4a — positional chi2 (one row per unit × position)
    cur.execute("DROP TABLE IF EXISTS hand_positional_chars")
    cur.execute("""
        CREATE TABLE hand_positional_chars (
            unit          TEXT,
            unit_name     TEXT,
            position      TEXT,
            n_tokens      INTEGER,
            chi2          REAL,
            df            INTEGER,
            p_value       REAL,
            significant_001 INTEGER,
            top_chars_json TEXT,
            PRIMARY KEY (unit, position)
        )
    """)
    for label, d in sorted(positional.items()):
        name = unit_label_name(label)
        for position, chi2_d in d["chi2"].items():
            if chi2_d.get("skipped"):
                continue
            top_json = json.dumps(d["top_chars"].get(position, []))
            cur.execute("""
                INSERT INTO hand_positional_chars VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                label, name, position,
                chi2_d.get("n_tokens"),
                chi2_d.get("chi2"),
                chi2_d.get("df"),
                chi2_d.get("p_value"),
                int(chi2_d.get("significant_001", False)),
                top_json,
            ))

    # Table 4b — word lengths
    cur.execute("DROP TABLE IF EXISTS hand_positional_lengths")
    cur.execute("""
        CREATE TABLE hand_positional_lengths (
            unit          TEXT PRIMARY KEY,
            unit_name     TEXT,
            n_words       INTEGER,
            mean_length   REAL,
            ks_stat       REAL,
            ks_p_value    REAL,
            significant_001 INTEGER,
            length_dist_json TEXT
        )
    """)
    for label, d in sorted(lengths.items()):
        cur.execute("""
            INSERT INTO hand_positional_lengths VALUES (?,?,?,?,?,?,?,?)
        """, (
            label, unit_label_name(label),
            d["n_words"], d["mean_length"],
            d["ks_stat"], d["ks_p_value"],
            int(d["significant_001"]),
            json.dumps(d["length_dist"]),
        ))

    # Table 4c — trigrams
    cur.execute("DROP TABLE IF EXISTS hand_positional_trigrams")
    cur.execute("""
        CREATE TABLE hand_positional_trigrams (
            unit             TEXT PRIMARY KEY,
            unit_name        TEXT,
            n_words          INTEGER,
            n_trigrams       INTEGER,
            trigram_entropy  REAL,
            chi2             REAL,
            chi2_p_value     REAL,
            significant_001  INTEGER,
            top5_json        TEXT
        )
    """)
    for label, d in sorted(trigrams.items()):
        chi2_d = d.get("chi2", {})
        cur.execute("""
            INSERT INTO hand_positional_trigrams VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            label, unit_label_name(label),
            d["n_words"], d["n_trigrams"],
            d["trigram_entropy"],
            chi2_d.get("chi2") if not chi2_d.get("skipped") else None,
            chi2_d.get("p_value") if not chi2_d.get("skipped") else None,
            int(chi2_d.get("significant_001", False)) if not chi2_d.get("skipped") else 0,
            json.dumps(d["top5_trigrams"]),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(units: dict, positional: dict,
                   lengths: dict, trigrams: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 4 — Positional patterns per hand (pure EVA)")
    lines.append(f"  Analysis units: {len(units)} "
                 f"(Davis 1–5 + X/Y + ?-per-section)")
    lines.append("=" * 80)

    # 4a — Positional chi2
    lines.append("\n── Phase 4a — Positional chi-square (initial/medial/final) ──")
    lines.append(
        f"  {'Unit':>6}  {'Name':>18}  "
        f"{'chi2_ini':>9}  {'p_ini':>8}  "
        f"{'chi2_med':>9}  {'p_med':>8}  "
        f"{'chi2_fin':>9}  {'p_fin':>8}"
    )
    lines.append("  " + "-" * 80)
    for label in sorted(positional.keys()):
        d = positional[label]
        name = unit_label_name(label)[:18]
        row = f"  {label:>6}  {name:>18}"
        for pos in ("initial", "medial", "final"):
            chi2_d = d["chi2"].get(pos, {})
            if chi2_d.get("skipped"):
                row += f"  {'n/a':>9}  {'':>8}"
            else:
                sig = "***" if chi2_d["significant_001"] else "   "
                row += f"  {chi2_d['chi2']:>9.1f}  {chi2_d['p_value']:>7.5f}{sig}"
        lines.append(row)

    # 4b — Word lengths
    lines.append("\n── Phase 4b — Word length distribution (KS-test vs corpus) ──")
    lines.append(
        f"  {'Unit':>6}  {'Name':>18}  {'N':>6}  "
        f"{'MeanLen':>7}  {'KS_stat':>7}  {'KS_p':>9}  Sig"
    )
    lines.append("  " + "-" * 66)
    for label in sorted(lengths.keys()):
        d = lengths[label]
        name = unit_label_name(label)[:18]
        sig = "***" if d["significant_001"] else ("*  " if d["significant_05"] else "   ")
        lines.append(
            f"  {label:>6}  {name:>18}  {d['n_words']:>6,}  "
            f"{d['mean_length']:>7.3f}  {d['ks_stat']:>7.4f}  "
            f"{d['ks_p_value']:>9.6f}  {sig}"
        )

    # 4c — Trigrams
    lines.append("\n── Phase 4c — EVA trigrams (entropy + chi-square vs corpus) ──")
    lines.append(
        f"  {'Unit':>6}  {'Name':>18}  {'H_trig':>7}  "
        f"{'chi2':>8}  {'p':>9}  Top-3 trigrams"
    )
    lines.append("  " + "-" * 72)
    for label in sorted(trigrams.keys()):
        d = trigrams[label]
        name = unit_label_name(label)[:18]
        chi2_d = d.get("chi2", {})
        if chi2_d.get("skipped"):
            chi2_str = "     n/a"
            p_str = "      n/a"
        else:
            sig = "***" if chi2_d["significant_001"] else "   "
            chi2_str = f"{chi2_d['chi2']:>8.1f}"
            p_str = f"{chi2_d['p_value']:>9.6f}{sig}"
        top3 = " ".join(t["trigram"] for t in d["top5_trigrams"][:3])
        lines.append(
            f"  {label:>6}  {name:>18}  {d['trigram_entropy']:>7.4f}  "
            f"{chi2_str}  {p_str}  {top3}"
        )

    lines.append("\n── Legend ──")
    lines.append("  *** p < 0.001 | * p < 0.05 | n/a = insufficient data")
    lines.append("  Unit ?X = hand ? limited to section X (Phase 2e decision)")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 4: EVA positional patterns per hand (~12 units, ? split by section)."""
    report_path = config.stats_dir / "hand_positional.json"
    summary_path = config.stats_dir / "hand_positional_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_positional report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 4 — Positional Patterns per Hand (pure EVA)")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Split by hand and build units
    print_step("Building ~12 analysis units (? split by section)...")
    corpus = split_corpus_by_hand(pages)
    units = build_analysis_units(corpus, pages)
    for label in sorted(units.keys()):
        name = unit_label_name(label)
        click.echo(f"    {label:>4} ({name}): {len(units[label]):,} words")
    click.echo(f"    Total units: {len(units)}")

    # 3. Phase 4a — Positional chars
    print_step("Phase 4a — Positional distribution of EVA characters...")
    positional = run_positional_analysis(units)
    for label in sorted(positional.keys()):
        d = positional[label]
        row_parts = []
        for pos in ("initial", "medial", "final"):
            chi2_d = d["chi2"].get(pos, {})
            if not chi2_d.get("skipped"):
                sig = "***" if chi2_d["significant_001"] else "ns"
                row_parts.append(f"{pos[:3]}:chi2={chi2_d['chi2']:.0f}({sig})")
        click.echo(f"    {label:>4}: " + "  ".join(row_parts))

    # 4. Phase 4b — Word lengths
    print_step("Phase 4b — Word length distribution (KS-test)...")
    lengths = run_length_analysis(units)
    for label in sorted(lengths.keys()):
        d = lengths[label]
        sig = "***" if d["significant_001"] else ("*" if d["significant_05"] else "ns")
        click.echo(f"    {label:>4}: mean={d['mean_length']:.3f}  "
                   f"KS={d['ks_stat']:.4f}  p={d['ks_p_value']:.6f}  {sig}")

    # 5. Phase 4c — Trigrams
    print_step("Phase 4c — EVA trigrams (entropy + chi-square)...")
    trigrams = run_trigram_analysis(units)
    for label in sorted(trigrams.keys()):
        d = trigrams[label]
        chi2_d = d.get("chi2", {})
        if chi2_d.get("skipped"):
            click.echo(f"    {label:>4}: H_trig={d['trigram_entropy']:.4f}  [chi2 skip]")
        else:
            sig = "***" if chi2_d["significant_001"] else "ns"
            click.echo(f"    {label:>4}: H_trig={d['trigram_entropy']:.4f}  "
                       f"chi2={chi2_d['chi2']:.1f}  p={chi2_d['p_value']:.6f}  {sig}")

    # 6. Save JSON
    print_step("Saving JSON...")
    report = {
        "n_units":    len(units),
        "unit_sizes": {k: len(v) for k, v in sorted(units.items())},
        "positional": positional,
        "lengths":    lengths,
        "trigrams":   trigrams,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 7. Save TXT
    summary = format_summary(units, positional, lengths, trigrams)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. Save to DB
    print_step("Writing DB tables hand_positional_*...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(positional, lengths, trigrams, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
