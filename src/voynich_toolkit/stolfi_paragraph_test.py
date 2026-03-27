"""
Phase 7b — Stolfi paragraph marker validation.

The IVTFF transcription (Stolfi 1998) marks paragraph boundaries with
@P (start), +P (continuation), =P (end). Phase 7c found that split gallows
are NOT concentrated at @P lines — but is that because Currier was wrong,
or because Stolfi's paragraph markers are wrong?

This module tests whether Stolfi's markers capture REAL structural boundaries
by checking if @P, +P, and =P lines have measurably different statistical
properties. If they don't, the markers may be arbitrary.

Four tests:

  7b-1 — Character frequency difference
         Do @P lines use different characters than +P lines?
         Chi-square: @P char distribution vs +P char distribution.

  7b-2 — Line length asymmetry
         @P lines should be shorter (paragraph indent = fewer words).
         =P lines should be shorter (last line often not full).
         +P lines should be the longest (middle of paragraph = full lines).
         KS-test + mean comparison.

  7b-3 — Gallows concentration by position
         Simple gallows (t, k, p, f) and split gallows (cth, ckh, cph, cfh)
         tested separately. Rate per word for @P vs +P vs =P.
         Null: random assignment of words to line positions.

  7b-4 — First-word and last-word vocabulary
         Do @P lines start with different words than +P lines?
         Do =P lines end with different words than +P lines?
         Chi-square on first-word / last-word frequency distributions.

If @P and +P show NO differences on any test, Stolfi's markers are likely
arbitrary and Phase 7c must be reinterpreted.

Output:
  stolfi_paragraph_test.json
  stolfi_paragraph_test_summary.txt
  DB table: stolfi_paragraph_test
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2 as scipy_chi2, ks_2samp, mannwhitneyu

from .config import ToolkitConfig
from .currier_line_test import parse_ivtff_lines, SPLIT_GALLOWS
from .utils import print_header, print_step


SEED = 42
N_NULL = 500

# Simple gallows characters
SIMPLE_GALLOWS = {"t", "k", "p", "f"}

# All gallows patterns (for searching in word text)
ALL_SPLIT_GALLOWS = {"cth", "ckh", "cph", "cfh"}


# =====================================================================
# Helpers
# =====================================================================

def lines_by_type(lines: list[dict]) -> dict[str, list[dict]]:
    """Group non-label lines by paragraph type."""
    groups: dict[str, list[dict]] = {
        "para_start": [],
        "para_cont": [],
        "para_end": [],
    }
    for line in lines:
        pt = line["para_type"]
        if pt in groups:
            groups[pt].append(line)
    return groups


def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# 7b-1 — Character frequency: @P vs +P
# =====================================================================

def char_freq_by_type(groups: dict[str, list[dict]]) -> dict:
    """Chi-square: are characters distributed differently on @P vs +P lines?"""
    results = {}

    for pair_name, (type_a, type_b) in [
        ("start_vs_cont", ("para_start", "para_cont")),
        ("end_vs_cont", ("para_end", "para_cont")),
    ]:
        words_a = [w for line in groups[type_a] for w in line["words"]]
        words_b = [w for line in groups[type_b] for w in line["words"]]

        chars_a = Counter(ch for w in words_a for ch in w)
        chars_b = Counter(ch for w in words_b for ch in w)

        all_chars = sorted(set(chars_a) | set(chars_b))
        total_a = sum(chars_a.values())
        total_b = sum(chars_b.values())

        if total_a < 100 or total_b < 100:
            results[pair_name] = {"skipped": True, "reason": "too few characters"}
            continue

        # Pool proportions
        pooled = {ch: (chars_a.get(ch, 0) + chars_b.get(ch, 0)) / (total_a + total_b)
                  for ch in all_chars}

        exp_a = np.array([pooled[ch] * total_a for ch in all_chars])
        exp_b = np.array([pooled[ch] * total_b for ch in all_chars])
        obs_a = np.array([chars_a.get(ch, 0) for ch in all_chars], dtype=float)
        obs_b = np.array([chars_b.get(ch, 0) for ch in all_chars], dtype=float)

        mask = (exp_a >= 5) & (exp_b >= 5)
        if mask.sum() < 2:
            results[pair_name] = {"skipped": True, "reason": "too few cells"}
            continue

        chi2_stat = float(
            np.sum((obs_a[mask] - exp_a[mask]) ** 2 / exp_a[mask]) +
            np.sum((obs_b[mask] - exp_b[mask]) ** 2 / exp_b[mask])
        )
        df = int(mask.sum()) - 1
        p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

        results[pair_name] = {
            "n_chars_a": total_a,
            "n_chars_b": total_b,
            "chi2": round(chi2_stat, 2),
            "df": df,
            "p_value": round(p_value, 6),
            "significant_001": bool(p_value < 0.001),
            "n_cells_used": int(mask.sum()),
        }

    return results


# =====================================================================
# 7b-2 — Line length asymmetry
# =====================================================================

def line_length_by_type(groups: dict[str, list[dict]]) -> dict:
    """Compare word count per line across @P, +P, =P."""
    results = {}

    lengths = {}
    for ptype, plines in groups.items():
        lens = [len(line["words"]) for line in plines]
        lengths[ptype] = lens
        results[f"{ptype}_n"] = len(lens)
        results[f"{ptype}_mean"] = round(float(np.mean(lens)), 3) if lens else 0.0
        results[f"{ptype}_median"] = float(np.median(lens)) if lens else 0.0
        results[f"{ptype}_std"] = round(float(np.std(lens, ddof=1)), 3) if len(lens) > 1 else 0.0

    # KS-test: @P vs +P
    if lengths["para_start"] and lengths["para_cont"]:
        ks_stat, ks_p = ks_2samp(lengths["para_start"], lengths["para_cont"])
        results["start_vs_cont_ks"] = round(float(ks_stat), 4)
        results["start_vs_cont_ks_p"] = round(float(ks_p), 6)

        # Mann-Whitney U (more robust for skewed distributions)
        u_stat, u_p = mannwhitneyu(lengths["para_start"], lengths["para_cont"],
                                    alternative="two-sided")
        results["start_vs_cont_mwu_p"] = round(float(u_p), 6)

    # KS-test: =P vs +P
    if lengths["para_end"] and lengths["para_cont"]:
        ks_stat, ks_p = ks_2samp(lengths["para_end"], lengths["para_cont"])
        results["end_vs_cont_ks"] = round(float(ks_stat), 4)
        results["end_vs_cont_ks_p"] = round(float(ks_p), 6)

        u_stat, u_p = mannwhitneyu(lengths["para_end"], lengths["para_cont"],
                                    alternative="two-sided")
        results["end_vs_cont_mwu_p"] = round(float(u_p), 6)

    return results


# =====================================================================
# 7b-3 — Gallows concentration by position
# =====================================================================

def count_gallows_in_words(words: list[str]) -> dict:
    """Count simple and split gallows in a word list."""
    simple = 0
    split = 0
    text = ".".join(words)
    for sg in ALL_SPLIT_GALLOWS:
        split += text.count(sg)
    # For simple gallows: count individual chars, excluding those inside split gallows
    clean = text
    for sg in ALL_SPLIT_GALLOWS:
        clean = clean.replace(sg, "___")
    for ch in SIMPLE_GALLOWS:
        simple += clean.count(ch)
    return {"simple": simple, "split": split, "total_words": len(words)}


def gallows_by_type(groups: dict[str, list[dict]]) -> dict:
    """Gallows rate (per word) by paragraph position."""
    results = {}
    for ptype, plines in groups.items():
        words = [w for line in plines for w in line["words"]]
        g = count_gallows_in_words(words)
        n = g["total_words"]
        results[ptype] = {
            "n_words": n,
            "simple_count": g["simple"],
            "split_count": g["split"],
            "simple_rate": round(g["simple"] / n, 4) if n > 0 else 0.0,
            "split_rate": round(g["split"] / n, 4) if n > 0 else 0.0,
        }
    return results


def null_gallows_rate(lines: list[dict], target_type: str,
                      gallows_type: str = "split",
                      n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle words across all line positions within each folio,
    compute gallows rate for the target type.
    """
    rng = random.Random(seed)

    # Group by folio, preserving para_type
    folio_data: dict[str, list[tuple[str, list[str]]]] = {}
    for line in lines:
        if line["para_type"] in ("label", "other"):
            continue
        f = line["folio"]
        if f not in folio_data:
            folio_data[f] = []
        folio_data[f].append((line["para_type"], list(line["words"])))

    nulls = []
    for _ in range(n_perms):
        target_count = 0
        target_words = 0
        for folio, fdata in folio_data.items():
            all_w = [w for _, ws in fdata for w in ws]
            rng.shuffle(all_w)
            pos = 0
            for ptype, orig_ws in fdata:
                n = len(orig_ws)
                chunk = all_w[pos:pos + n]
                pos += n
                if ptype == target_type:
                    g = count_gallows_in_words(chunk)
                    target_count += g[gallows_type]
                    target_words += g["total_words"]
        rate = target_count / target_words if target_words > 0 else 0.0
        nulls.append(rate)

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# 7b-4 — First-word and last-word vocabulary
# =====================================================================

def word_position_vocab(groups: dict[str, list[dict]]) -> dict:
    """Compare first-word and last-word vocabulary across line types."""
    results = {}

    for pair_name, (type_a, type_b), position in [
        ("first_word_start_vs_cont", ("para_start", "para_cont"), "first"),
        ("last_word_end_vs_cont", ("para_end", "para_cont"), "last"),
    ]:
        if position == "first":
            words_a = [line["words"][0] for line in groups[type_a] if line["words"]]
            words_b = [line["words"][0] for line in groups[type_b] if line["words"]]
        else:
            words_a = [line["words"][-1] for line in groups[type_a] if line["words"]]
            words_b = [line["words"][-1] for line in groups[type_b] if line["words"]]

        freq_a = Counter(words_a)
        freq_b = Counter(words_b)

        # Jaccard overlap of top-10 words
        top_a = {w for w, _ in freq_a.most_common(10)}
        top_b = {w for w, _ in freq_b.most_common(10)}
        union = top_a | top_b
        inter = top_a & top_b
        jaccard = len(inter) / len(union) if union else 0.0

        # Unique to @P / =P (not in +P top-50)
        common_b = {w for w, _ in freq_b.most_common(50)}
        unique_a = [w for w, c in freq_a.most_common(20) if w not in common_b]

        results[pair_name] = {
            "n_a": len(words_a),
            "n_b": len(words_b),
            "jaccard_top10": round(jaccard, 3),
            "top5_a": [{"word": w, "count": c} for w, c in freq_a.most_common(5)],
            "top5_b": [{"word": w, "count": c} for w, c in freq_b.most_common(5)],
            "unique_to_a": unique_a[:10],
        }

    return results


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(char_freq: dict, line_len: dict, gallows: dict,
               vocab: dict, gallows_null: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS stolfi_paragraph_test")
    cur.execute("""
        CREATE TABLE stolfi_paragraph_test (
            test           TEXT PRIMARY KEY,
            metric         TEXT,
            value          REAL,
            p_value        REAL,
            significant    INTEGER,
            detail_json    TEXT
        )
    """)

    # Char freq
    for pair, d in char_freq.items():
        if d.get("skipped"):
            continue
        cur.execute("INSERT INTO stolfi_paragraph_test VALUES (?,?,?,?,?,?)", (
            f"char_freq_{pair}", "chi2",
            d["chi2"], d["p_value"], int(d["significant_001"]),
            json.dumps(d),
        ))

    # Line length
    for key in ["start_vs_cont_ks_p", "end_vs_cont_ks_p"]:
        if key in line_len:
            cur.execute("INSERT INTO stolfi_paragraph_test VALUES (?,?,?,?,?,?)", (
                f"line_length_{key}", "ks_p",
                line_len.get(key.replace("_ks_p", "_ks")),
                line_len[key],
                int(line_len[key] < 0.001),
                json.dumps({k: v for k, v in line_len.items()
                            if k.startswith(key.split("_ks_p")[0])}),
            ))

    # Gallows rates
    for ptype, d in gallows.items():
        cur.execute("INSERT INTO stolfi_paragraph_test VALUES (?,?,?,?,?,?)", (
            f"gallows_{ptype}", "split_rate",
            d["split_rate"], None, 0, json.dumps(d),
        ))

    # Gallows null z-scores
    for key, d in gallows_null.items():
        cur.execute("INSERT INTO stolfi_paragraph_test VALUES (?,?,?,?,?,?)", (
            f"gallows_null_{key}", "z_score",
            d.get("z"), None, int(abs(d.get("z", 0) or 0) > 2), json.dumps(d),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(char_freq: dict, line_len: dict, gallows: dict,
                   gallows_null: dict, vocab: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 7b — Stolfi paragraph marker validation")
    lines.append("  Question: do @P / +P / =P markers reflect real structural boundaries?")
    lines.append("=" * 80)

    # 7b-1 — Character frequencies
    lines.append("\n── 7b-1 — Character frequency: @P vs +P, =P vs +P ──")
    for pair, d in char_freq.items():
        if d.get("skipped"):
            lines.append(f"  {pair}: skipped ({d.get('reason')})")
            continue
        sig = "***" if d["significant_001"] else "ns"
        lines.append(
            f"  {pair}: chi2={d['chi2']:.1f}  df={d['df']}  "
            f"p={d['p_value']:.6f}  {sig}"
        )

    # 7b-2 — Line lengths
    lines.append("\n── 7b-2 — Line length (words per line) ──")
    lines.append(
        f"  {'Type':>12}  {'N':>5}  {'Mean':>6}  {'Median':>6}  {'Std':>5}"
    )
    lines.append("  " + "-" * 42)
    for ptype in ("para_start", "para_cont", "para_end"):
        n = line_len.get(f"{ptype}_n", 0)
        mean = line_len.get(f"{ptype}_mean", 0)
        med = line_len.get(f"{ptype}_median", 0)
        std = line_len.get(f"{ptype}_std", 0)
        lines.append(f"  {ptype:>12}  {n:>5}  {mean:>6.2f}  {med:>6.1f}  {std:>5.2f}")

    for test_name in ["start_vs_cont", "end_vs_cont"]:
        ks_p = line_len.get(f"{test_name}_ks_p")
        mwu_p = line_len.get(f"{test_name}_mwu_p")
        if ks_p is not None:
            sig = "***" if ks_p < 0.001 else ("*" if ks_p < 0.05 else "ns")
            lines.append(f"  {test_name}: KS p={ks_p:.6f} {sig}  "
                         f"Mann-Whitney p={mwu_p:.6f}")

    # 7b-3 — Gallows
    lines.append("\n── 7b-3 — Gallows rate by paragraph position ──")
    lines.append(
        f"  {'Type':>12}  {'Words':>6}  {'Simple':>6}  {'Split':>6}  "
        f"{'Simp/w':>7}  {'Split/w':>7}"
    )
    lines.append("  " + "-" * 56)
    for ptype in ("para_start", "para_cont", "para_end"):
        d = gallows.get(ptype, {})
        lines.append(
            f"  {ptype:>12}  {d.get('n_words',0):>6}  "
            f"{d.get('simple_count',0):>6}  {d.get('split_count',0):>6}  "
            f"{d.get('simple_rate',0):>7.4f}  {d.get('split_rate',0):>7.4f}"
        )

    for key, d in gallows_null.items():
        z = d.get("z")
        z_str = f"z={z:+.2f}" if z is not None else "z=n/a"
        lines.append(f"  {key}: obs={d['observed']:.4f}  "
                     f"null={d['null_mean']:.4f}  {z_str}")

    # 7b-4 — Vocabulary
    lines.append("\n── 7b-4 — First-word / last-word vocabulary ──")
    for test_name, d in vocab.items():
        lines.append(f"  {test_name}:")
        lines.append(f"    Jaccard (top-10): {d['jaccard_top10']:.3f}")
        top_a = ", ".join(f"{t['word']}({t['count']})" for t in d["top5_a"][:3])
        top_b = ", ".join(f"{t['word']}({t['count']})" for t in d["top5_b"][:3])
        lines.append(f"    Top-3 A: {top_a}")
        lines.append(f"    Top-3 B: {top_b}")
        if d["unique_to_a"]:
            lines.append(f"    Unique to A (not in B top-50): "
                         f"{', '.join(d['unique_to_a'][:5])}")

    # Verdict
    lines.append("\n── Verdict ──")
    n_sig = 0
    for pair, d in char_freq.items():
        if d.get("significant_001"):
            n_sig += 1
    for test_name in ["start_vs_cont_ks_p", "end_vs_cont_ks_p"]:
        if line_len.get(test_name, 1.0) < 0.001:
            n_sig += 1

    if n_sig >= 3:
        lines.append("  STOLFI MARKERS ARE STRUCTURALLY MEANINGFUL")
        lines.append("  Multiple tests show significant differences between @P / +P / =P.")
        lines.append("  Phase 7c (split gallows) result stands — Currier was likely wrong.")
    elif n_sig == 0:
        lines.append("  STOLFI MARKERS MAY BE ARBITRARY")
        lines.append("  No test shows a significant difference between line types.")
        lines.append("  Phase 7c must be reinterpreted — the markers don't capture real paragraphs.")
    else:
        lines.append("  MIXED EVIDENCE")
        lines.append(f"  {n_sig} tests significant. Markers partially capture structure.")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 7b: validate Stolfi's paragraph markers against structural properties."""
    report_path = config.stats_dir / "stolfi_paragraph_test.json"
    summary_path = config.stats_dir / "stolfi_paragraph_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  stolfi_paragraph_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 7b — Stolfi Paragraph Marker Validation")

    # 1. Parse IVTFF
    print_step("Parsing IVTFF file...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    all_lines = parse_ivtff_lines(eva_file)

    groups = lines_by_type(all_lines)
    for ptype, plines in groups.items():
        n_words = sum(len(l["words"]) for l in plines)
        click.echo(f"    {ptype}: {len(plines)} lines, {n_words:,} words")

    # 2. Test 7b-1 — Character frequencies
    print_step("Test 7b-1 — Character frequency comparison...")
    char_freq = char_freq_by_type(groups)
    for pair, d in char_freq.items():
        if d.get("skipped"):
            click.echo(f"    {pair}: skipped")
        else:
            sig = "***" if d["significant_001"] else "ns"
            click.echo(f"    {pair}: chi2={d['chi2']:.1f}  p={d['p_value']:.6f}  {sig}")

    # 3. Test 7b-2 — Line lengths
    print_step("Test 7b-2 — Line length by paragraph position...")
    line_len = line_length_by_type(groups)
    for ptype in ("para_start", "para_cont", "para_end"):
        click.echo(f"    {ptype}: mean={line_len[f'{ptype}_mean']:.2f}  "
                   f"n={line_len[f'{ptype}_n']}")
    for test_name in ["start_vs_cont", "end_vs_cont"]:
        ks_p = line_len.get(f"{test_name}_ks_p")
        if ks_p is not None:
            sig = "***" if ks_p < 0.001 else "ns"
            click.echo(f"    {test_name}: KS p={ks_p:.6f}  {sig}")

    # 4. Test 7b-3 — Gallows
    print_step(f"Test 7b-3 — Gallows rate by position ({N_NULL} null perms)...")
    gall = gallows_by_type(groups)
    for ptype in ("para_start", "para_cont", "para_end"):
        d = gall[ptype]
        click.echo(f"    {ptype}: simple={d['simple_rate']:.4f}/w  "
                   f"split={d['split_rate']:.4f}/w")

    # Null model for split gallows on @P
    gallows_null_results = {}
    for target, gtype in [("para_start_split", "split"),
                          ("para_start_simple", "simple")]:
        ptype = "para_start"
        obs_rate = gall[ptype][f"{gtype}_rate"]
        null = null_gallows_rate(all_lines, ptype, gallows_type=gtype,
                                 n_perms=N_NULL, seed=SEED + hash(target) % 1000)
        z = z_score(obs_rate, null["null_mean"], null["null_std"])
        gallows_null_results[target] = {
            "observed": obs_rate,
            "null_mean": round(null["null_mean"], 4),
            "null_std": round(null["null_std"], 6),
            "z": round(z, 3) if z is not None else None,
        }
        z_str = f"z={z:+.2f}" if z is not None else "z=n/a"
        click.echo(f"    {target}: obs={obs_rate:.4f}  "
                   f"null={null['null_mean']:.4f}  {z_str}")

    # 5. Test 7b-4 — Vocabulary
    print_step("Test 7b-4 — First-word / last-word vocabulary...")
    vocab = word_position_vocab(groups)
    for test_name, d in vocab.items():
        click.echo(f"    {test_name}: Jaccard={d['jaccard_top10']:.3f}  "
                   f"n_a={d['n_a']}  n_b={d['n_b']}")

    # 6. Save JSON
    print_step("Saving JSON...")
    report = {
        "char_freq": char_freq,
        "line_length": line_len,
        "gallows": gall,
        "gallows_null": gallows_null_results,
        "vocabulary": vocab,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    {report_path}")

    # 7. Save TXT
    summary = format_summary(char_freq, line_len, gall, gallows_null_results, vocab)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. Save to DB
    print_step("Writing DB table stolfi_paragraph_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(char_freq, line_len, gall, vocab, gallows_null_results, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
