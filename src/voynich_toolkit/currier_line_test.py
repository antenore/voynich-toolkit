"""
Phase 7 — Currier line-boundary tests.

Verifies three structural observations originally made by Prescott Currier (1976)
that have become canonical in Voynich research but have not been formally tested
with a null model:

  7a — Line non-continuity
       Currier observed that no word repetition ever crosses a line boundary.
       If word X appears at position -1 of line N, it never appears at position 0
       of line N+1. Test: count cross-boundary repetitions vs null (shuffled lines).

  7b — Line-final character bias
       Certain characters or bigrams appear almost exclusively at the end of the
       last word of a line (~85% of all occurrences according to Currier).
       Test: for each character, compute the fraction of occurrences that are
       line-final. Compare to null (random word placement).

  7c — Split gallows at paragraph start
       "Split gallows" (cth, ckh, cph, cfh) appear only on the first line of
       paragraphs and in labels. Test: count split gallows by line position
       (paragraph-initial vs continuation) and compare to null.

All tests are on raw EVA — no decoding, no lexicon.
The IVTFF file is parsed directly (not via parse_eva_words) to preserve
paragraph markers (@P = start, +P = continuation, =P = end, @L = label).

Output:
  currier_line_test.json
  currier_line_test_summary.txt
  DB table: currier_line_test
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .utils import print_header, print_step


SEED = 42
N_NULL = 1000

# Split gallows sequences in EVA
SPLIT_GALLOWS = {"cth", "ckh", "cph", "cfh"}


# =====================================================================
# IVTFF parser — preserves line position and paragraph structure
# =====================================================================

def parse_ivtff_lines(filepath: Path,
                      transcriber: str = "H") -> list[dict]:
    """Parse IVTFF file preserving paragraph structure.

    Returns: list of line dicts, each with:
      folio, line_num, marker, para_type, words, raw_text
    where para_type is one of:
      'para_start'  (@P)
      'para_cont'   (+P)
      'para_end'    (=P)
      'label'       (@L)
      'other'
    """
    text = filepath.read_text(encoding="utf-8", errors="ignore")

    # Match: <FOLIO.LINENUM,MARKER;TRANSCRIBER>   TEXT
    line_re = re.compile(
        r"^<(f\w+)\.(\d+\w*),([^;]+);(\w)>\s+(.+)"
    )

    result = []
    for raw_line in text.split("\n"):
        raw_line = raw_line.rstrip()
        if not raw_line or raw_line.startswith("#"):
            continue
        m = line_re.match(raw_line)
        if not m:
            continue
        folio = m.group(1)
        line_num = m.group(2)
        marker = m.group(3)
        scribe = m.group(4)
        eva_text = m.group(5)

        if scribe != transcriber:
            continue

        # Determine paragraph type from marker
        if marker.startswith("@L"):
            para_type = "label"
        elif marker.startswith("@"):
            para_type = "para_start"
        elif marker.startswith("+"):
            para_type = "para_cont"
        elif marker.startswith("="):
            para_type = "para_end"
        elif marker.startswith("*"):
            para_type = "para_start"  # starred = new item
        elif marker.startswith("-"):
            para_type = "para_cont"
        else:
            para_type = "other"

        # Extract words (same logic as parse_eva_words)
        clean = re.sub(r"\{[^}]*\}", "", eva_text)
        clean = re.sub(r"<[^>]*>", "", clean)
        clean = re.sub(r"[%!?\[\]*,]", "", clean)
        words = []
        for token in clean.split("."):
            word = re.sub(r"[^a-z]", "", token)
            if word:
                words.append(word)

        if not words:
            continue

        result.append({
            "folio": folio,
            "line_num": line_num,
            "marker": marker,
            "para_type": para_type,
            "words": words,
            "raw_text": eva_text,
        })

    return result


# =====================================================================
# 7a — Line non-continuity
# =====================================================================

def count_cross_boundary_repeats(lines: list[dict]) -> dict:
    """Count how many times the last word of line N equals
    the first word of line N+1, within the same folio.

    Returns: dict with n_boundaries, n_repeats, repeat_rate, examples
    """
    n_boundaries = 0
    n_repeats = 0
    examples = []

    for i in range(len(lines) - 1):
        curr = lines[i]
        next_l = lines[i + 1]

        # Only consecutive lines on the same folio
        if curr["folio"] != next_l["folio"]:
            continue
        # Skip labels
        if curr["para_type"] == "label" or next_l["para_type"] == "label":
            continue

        n_boundaries += 1
        last_word = curr["words"][-1]
        first_word = next_l["words"][0]

        if last_word == first_word:
            n_repeats += 1
            examples.append({
                "folio": curr["folio"],
                "line": curr["line_num"],
                "word": last_word,
            })

    rate = n_repeats / n_boundaries if n_boundaries > 0 else 0.0
    return {
        "n_boundaries": n_boundaries,
        "n_repeats": n_repeats,
        "repeat_rate": round(rate, 6),
        "examples": examples[:20],
    }


def null_cross_boundary(lines: list[dict],
                        n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle all words across all lines (preserving line sizes
    and folio boundaries), count cross-boundary repeats.

    This models: 'if words were randomly distributed across lines,
    how often would the last word of one line match the first of the next?'
    """
    rng = random.Random(seed)

    # Group lines by folio, collect words
    folio_lines: dict[str, list[list[str]]] = {}
    for line in lines:
        if line["para_type"] == "label":
            continue
        f = line["folio"]
        if f not in folio_lines:
            folio_lines[f] = []
        folio_lines[f].append(list(line["words"]))

    nulls = []
    for _ in range(n_perms):
        total_repeats = 0
        total_boundaries = 0
        for folio, flines in folio_lines.items():
            # Pool all words from this folio
            all_w = [w for line in flines for w in line]
            rng.shuffle(all_w)
            # Redistribute into original line sizes
            pos = 0
            shuffled = []
            for line in flines:
                n = len(line)
                shuffled.append(all_w[pos:pos + n])
                pos += n
            # Count repeats
            for i in range(len(shuffled) - 1):
                total_boundaries += 1
                if shuffled[i][-1] == shuffled[i + 1][0]:
                    total_repeats += 1
        rate = total_repeats / total_boundaries if total_boundaries > 0 else 0.0
        nulls.append(rate)

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# 7b — Line-final character bias
# =====================================================================

def line_final_char_bias(lines: list[dict]) -> dict:
    """For each character, compute what fraction of its total occurrences
    are at the final position of the last word of a line.

    Characters with high line-final fraction are "line-end markers."
    """
    # Count total occurrences of each char in the entire corpus
    total_char: Counter = Counter()
    # Count occurrences at line-final position (last char of last word)
    final_char: Counter = Counter()

    for line in lines:
        if line["para_type"] == "label":
            continue
        for w in line["words"]:
            for ch in w:
                total_char[ch] += 1
        # Last character of last word = line-final
        last_word = line["words"][-1]
        if last_word:
            final_char[last_word[-1]] += 1

    # Compute fraction
    results = {}
    for ch in sorted(total_char.keys()):
        t = total_char[ch]
        f = final_char.get(ch, 0)
        results[ch] = {
            "total": t,
            "line_final": f,
            "fraction": round(f / t, 4) if t > 0 else 0.0,
        }

    return results


def null_line_final_bias(lines: list[dict], top_char: str,
                         n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null for the line-final fraction of one character:
    shuffle words across lines (same folio), recompute fraction.
    """
    rng = random.Random(seed)

    # Pre-compute folio line structure
    folio_lines: dict[str, list[list[str]]] = {}
    for line in lines:
        if line["para_type"] == "label":
            continue
        f = line["folio"]
        if f not in folio_lines:
            folio_lines[f] = []
        folio_lines[f].append(list(line["words"]))

    # Total occurrences of the character (constant)
    total = sum(w.count(top_char) for flines in folio_lines.values()
                for line in flines for w in line)

    nulls = []
    for _ in range(n_perms):
        final_count = 0
        for folio, flines in folio_lines.items():
            all_w = [w for line in flines for w in line]
            rng.shuffle(all_w)
            pos = 0
            for line in flines:
                n = len(line)
                chunk = all_w[pos:pos + n]
                pos += n
                if chunk and chunk[-1]:
                    if chunk[-1][-1] == top_char:
                        final_count += 1
        frac = final_count / total if total > 0 else 0.0
        nulls.append(frac)

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# 7c — Split gallows at paragraph start
# =====================================================================

def split_gallows_position(lines: list[dict]) -> dict:
    """Count split gallows occurrences by paragraph position.

    Split gallows: cth, ckh, cph, cfh
    Positions: para_start, para_cont, para_end, label
    """
    by_position: dict[str, int] = Counter()
    by_gallows: dict[str, Counter] = {g: Counter() for g in SPLIT_GALLOWS}
    total_lines_by_type = Counter()
    examples = []

    for line in lines:
        pt = line["para_type"]
        total_lines_by_type[pt] += 1

        # Search for split gallows in words
        text = ".".join(line["words"])
        for sg in SPLIT_GALLOWS:
            count = text.count(sg)
            if count > 0:
                by_position[pt] += count
                by_gallows[sg][pt] += count
                if len(examples) < 20:
                    examples.append({
                        "folio": line["folio"],
                        "line": line["line_num"],
                        "para_type": pt,
                        "gallows": sg,
                        "count": count,
                    })

    total_sg = sum(by_position.values())
    para_start_count = by_position.get("para_start", 0)
    label_count = by_position.get("label", 0)
    first_or_label = para_start_count + label_count
    first_or_label_frac = first_or_label / total_sg if total_sg > 0 else 0.0

    return {
        "total_split_gallows": total_sg,
        "by_position": dict(by_position),
        "by_gallows": {g: dict(c) for g, c in by_gallows.items()},
        "total_lines_by_type": dict(total_lines_by_type),
        "first_or_label_count": first_or_label,
        "first_or_label_fraction": round(first_or_label_frac, 4),
        "examples": examples,
    }


def null_split_gallows(lines: list[dict],
                       n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle words across ALL paragraph positions (within folio),
    count fraction of split gallows on para_start + label lines.

    If Currier is right, the observed fraction should be far higher than null.
    """
    rng = random.Random(seed)

    # Pre-compute structure
    folio_data: dict[str, list[tuple[str, list[str]]]] = {}
    for line in lines:
        f = line["folio"]
        if f not in folio_data:
            folio_data[f] = []
        folio_data[f].append((line["para_type"], list(line["words"])))

    nulls = []
    for _ in range(n_perms):
        first_or_label = 0
        total = 0
        for folio, fdata in folio_data.items():
            # Pool all words, shuffle, redistribute
            all_w = [w for _, ws in fdata for w in ws]
            rng.shuffle(all_w)
            pos = 0
            for para_type, orig_ws in fdata:
                n = len(orig_ws)
                chunk = all_w[pos:pos + n]
                pos += n
                text = ".".join(chunk)
                sg_count = sum(text.count(sg) for sg in SPLIT_GALLOWS)
                total += sg_count
                if para_type in ("para_start", "label"):
                    first_or_label += sg_count
        frac = first_or_label / total if total > 0 else 0.0
        nulls.append(frac)

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# Z-score helper
# =====================================================================

def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(results: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS currier_line_test")
    cur.execute("""
        CREATE TABLE currier_line_test (
            test           TEXT PRIMARY KEY,
            observed       REAL,
            null_mean      REAL,
            null_std       REAL,
            z_score        REAL,
            detail_json    TEXT
        )
    """)

    for test_name, d in results.items():
        cur.execute("INSERT INTO currier_line_test VALUES (?,?,?,?,?,?)", (
            test_name,
            d.get("observed"),
            d.get("null_mean"),
            d.get("null_std"),
            d.get("z"),
            json.dumps(d.get("detail", {})),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(results: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 7 — Currier line-boundary tests (1976 observations)")
    lines.append("=" * 80)

    # 7a
    r7a = results["7a_cross_boundary"]
    lines.append("\n── 7a — Line non-continuity ──")
    lines.append(f"  Currier claim: no word repetition ever crosses a line boundary.")
    lines.append(f"  Observed: {r7a['detail']['n_repeats']} repeats across "
                 f"{r7a['detail']['n_boundaries']} boundaries "
                 f"(rate = {r7a['observed']*100:.3f}%)")
    z_str = f"{r7a['z']:+.2f}" if r7a["z"] is not None else "n/a"
    lines.append(f"  Null (shuffled): mean = {r7a['null_mean']*100:.3f}%")
    lines.append(f"  z = {z_str}")
    if r7a["detail"]["n_repeats"] == 0:
        lines.append("  CONFIRMED: zero cross-boundary repeats (exactly as Currier claimed)")
    elif r7a["z"] is not None and r7a["z"] < -3:
        lines.append("  STRONG: far fewer cross-boundary repeats than expected by chance")
    else:
        lines.append("  NOT CONFIRMED: cross-boundary repeats are not abnormally rare")

    # 7b
    r7b = results["7b_line_final_bias"]
    lines.append("\n── 7b — Line-final character bias ──")
    lines.append(f"  Currier claim: some characters appear ~85% of the time at line-end.")
    lines.append(f"  Top line-final characters:")
    for ch_d in r7b["detail"]["top_biased"][:8]:
        z_c = ch_d.get("z")
        z_c_str = f"z={z_c:+.1f}" if z_c is not None else "z=n/a"
        lines.append(
            f"    '{ch_d['char']}': {ch_d['fraction']*100:5.1f}% of occurrences "
            f"are line-final (n={ch_d['total']})  {z_c_str}"
        )
    if r7b["detail"]["top_biased"]:
        best = r7b["detail"]["top_biased"][0]
        if best["fraction"] > 0.5:
            lines.append(f"  CONFIRMED: '{best['char']}' appears "
                         f"{best['fraction']*100:.1f}% at line-end")
        else:
            lines.append(f"  PARTIALLY: highest bias is {best['fraction']*100:.1f}% "
                         f"(below Currier's ~85%)")

    # 7c
    r7c = results["7c_split_gallows"]
    lines.append("\n── 7c — Split gallows at paragraph start ──")
    lines.append(f"  Currier claim: split gallows only on first line of paragraphs + labels.")
    lines.append(f"  Observed: {r7c['detail']['first_or_label_count']}/"
                 f"{r7c['detail']['total_split_gallows']} split gallows "
                 f"on para-start or label lines "
                 f"({r7c['observed']*100:.1f}%)")
    z_str = f"{r7c['z']:+.2f}" if r7c["z"] is not None else "n/a"
    lines.append(f"  Null (shuffled): mean = {r7c['null_mean']*100:.1f}%")
    lines.append(f"  z = {z_str}")

    by_pos = r7c["detail"].get("by_position", {})
    for pos in ("para_start", "para_cont", "para_end", "label", "other"):
        if pos in by_pos:
            lines.append(f"    {pos}: {by_pos[pos]} split gallows")

    if r7c["observed"] > 0.8:
        lines.append("  CONFIRMED: >80% of split gallows are at paragraph start or labels")
    elif r7c["z"] is not None and r7c["z"] > 5:
        lines.append("  STRONG: split gallows strongly biased toward paragraph start")
    else:
        lines.append("  NOT CONFIRMED: split gallows are not strongly position-biased")

    lines.append("\n── Note ──")
    lines.append("  These tests verify Currier (1976) observations with formal null models.")
    lines.append("  A confirmed observation constrains the space of possible generating mechanisms.")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 7: Currier line-boundary tests — formal verification of 1976 observations."""
    report_path = config.stats_dir / "currier_line_test.json"
    summary_path = config.stats_dir / "currier_line_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  currier_line_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 7 — Currier Line-Boundary Tests (1976)")

    # 1. Parse IVTFF directly
    print_step("Parsing IVTFF file (preserving paragraph markers)...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    lines = parse_ivtff_lines(eva_file)
    n_text = sum(1 for l in lines if l["para_type"] != "label")
    n_label = sum(1 for l in lines if l["para_type"] == "label")
    n_para_start = sum(1 for l in lines if l["para_type"] == "para_start")
    click.echo(f"    {len(lines)} lines total ({n_text} text, {n_label} labels)")
    click.echo(f"    {n_para_start} paragraph starts detected")

    results = {}

    # 2. Test 7a — Cross-boundary repeats
    print_step(f"Test 7a — Cross-boundary word repeats ({N_NULL} null permutations)...")
    cross = count_cross_boundary_repeats(lines)
    click.echo(f"    Observed: {cross['n_repeats']} repeats / "
               f"{cross['n_boundaries']} boundaries "
               f"(rate={cross['repeat_rate']*100:.3f}%)")
    if cross["examples"]:
        for ex in cross["examples"][:5]:
            click.echo(f"      {ex['folio']}.{ex['line']}: '{ex['word']}'")

    null_cross = null_cross_boundary(lines, n_perms=N_NULL, seed=SEED)
    z_cross = z_score(cross["repeat_rate"],
                      null_cross["null_mean"], null_cross["null_std"])
    click.echo(f"    Null: mean={null_cross['null_mean']*100:.3f}%  "
               f"std={null_cross['null_std']*100:.3f}%")
    z_str = f"{z_cross:+.2f}" if z_cross is not None else "n/a"
    click.echo(f"    z = {z_str}")

    results["7a_cross_boundary"] = {
        "observed": cross["repeat_rate"],
        "null_mean": round(null_cross["null_mean"], 6),
        "null_std": round(null_cross["null_std"], 6),
        "z": round(z_cross, 3) if z_cross is not None else None,
        "detail": cross,
    }

    # 3. Test 7b — Line-final character bias
    print_step("Test 7b — Line-final character bias...")
    bias = line_final_char_bias(lines)

    # Sort by fraction descending, test top-5 with null
    sorted_chars = sorted(bias.items(), key=lambda x: x[1]["fraction"],
                          reverse=True)
    top_biased = []
    for ch, d in sorted_chars[:5]:
        if d["total"] < 50:
            continue
        null_b = null_line_final_bias(lines, ch, n_perms=N_NULL,
                                       seed=SEED + ord(ch))
        z_b = z_score(d["fraction"], null_b["null_mean"], null_b["null_std"])
        entry = {
            "char": ch,
            "total": d["total"],
            "line_final": d["line_final"],
            "fraction": d["fraction"],
            "null_mean": round(null_b["null_mean"], 4),
            "z": round(z_b, 3) if z_b is not None else None,
        }
        top_biased.append(entry)
        z_str = f"z={z_b:+.1f}" if z_b is not None else "z=n/a"
        click.echo(f"    '{ch}': {d['fraction']*100:.1f}% line-final "
                   f"(n={d['total']})  {z_str}")

    results["7b_line_final_bias"] = {
        "observed": top_biased[0]["fraction"] if top_biased else 0.0,
        "null_mean": top_biased[0]["null_mean"] if top_biased else 0.0,
        "null_std": None,
        "z": top_biased[0]["z"] if top_biased else None,
        "detail": {"all_chars": bias, "top_biased": top_biased},
    }

    # 4. Test 7c — Split gallows position
    print_step(f"Test 7c — Split gallows at paragraph start ({N_NULL} perms)...")
    sg = split_gallows_position(lines)
    click.echo(f"    Total split gallows: {sg['total_split_gallows']}")
    click.echo(f"    On para_start + label: {sg['first_or_label_count']} "
               f"({sg['first_or_label_fraction']*100:.1f}%)")
    for pos, count in sorted(sg["by_position"].items()):
        click.echo(f"      {pos}: {count}")

    null_sg = null_split_gallows(lines, n_perms=N_NULL, seed=SEED + 99)
    z_sg = z_score(sg["first_or_label_fraction"],
                   null_sg["null_mean"], null_sg["null_std"])
    click.echo(f"    Null: mean={null_sg['null_mean']*100:.1f}%  "
               f"std={null_sg['null_std']*100:.1f}%")
    z_str = f"{z_sg:+.2f}" if z_sg is not None else "n/a"
    click.echo(f"    z = {z_str}")

    results["7c_split_gallows"] = {
        "observed": sg["first_or_label_fraction"],
        "null_mean": round(null_sg["null_mean"], 4),
        "null_std": round(null_sg["null_std"], 6),
        "z": round(z_sg, 3) if z_sg is not None else None,
        "detail": sg,
    }

    # 5. Save JSON
    print_step("Saving JSON...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    {report_path}")

    # 6. Save TXT
    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 7. Save to DB
    print_step("Writing DB table currier_line_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(results, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
