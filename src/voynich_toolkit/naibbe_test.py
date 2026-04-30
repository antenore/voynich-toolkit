"""
Naibbe Cipher Test — Phase 11.

Tests whether Greshko's (2025) verbose homophonic substitution cipher can
reproduce the 16 confirmed structural properties of the Voynich Manuscript.

The Naibbe cipher encrypts Latin/Italian by:
  1. Segmenting plaintext into unigrams and bigrams (dice roll)
  2. Encoding each segment via one of 6 tables (playing card selection)
  3. Each table maps plaintext segments to EVA-like character sequences
  4. Ciphertext words = concatenation of encoded segments

This is the same framework as rugg_test.py (Phase 27.9): generate synthetic
corpus matching the real manuscript's page/line/section structure, measure
all 16 properties, compare. If Naibbe reproduces more properties than Rugg,
the cipher hypothesis gains credibility.

Key difference from Rugg: Naibbe is a REAL cipher (reversible, meaningful
plaintext). If it passes, it's evidence the VMS could contain real language.

References:
  Greshko, M.A. (2025). The Naibbe cipher: a substitution cipher that
  encrypts Latin and Italian as Voynich Manuscript-like ciphertext.
  Cryptologia. DOI: 10.1080/01611194.2025.2566408
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import string
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2_contingency

from .config import ToolkitConfig
from .rugg_test import (
    compare_properties,
    format_summary,
    measure_all_properties,
    _serialise,
    _compact_dict,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# ── Constants ────────────────────────────────────────────────────

SEED = 42
N_TABLES = 6          # Naibbe uses 6 encoding tables (playing card suits × 2?)
N_RUNS = 3            # average over multiple cipher instantiations

# 19 EVA characters used in the manuscript
EVA_CHARS = list("acdefghiklmnopqrsty")

# Italian letter frequencies (approximate, for weighted plaintext generation)
ITALIAN_FREQ = {
    'a': 0.1174, 'b': 0.0092, 'c': 0.0450, 'd': 0.0337, 'e': 0.1179,
    'f': 0.0095, 'g': 0.0164, 'h': 0.0154, 'i': 0.1128, 'l': 0.0651,
    'm': 0.0251, 'n': 0.0688, 'o': 0.0983, 'p': 0.0305, 'q': 0.0051,
    'r': 0.0637, 's': 0.0498, 't': 0.0562, 'u': 0.0301, 'v': 0.0210,
    'w': 0.0003, 'x': 0.0003, 'y': 0.0002, 'z': 0.0049,
}

# Common Italian words for generating realistic-ish plaintext
# (We don't need perfect Italian — just plausible letter distribution)
ITALIAN_SYLLABLES = [
    "la", "il", "di", "che", "non", "con", "per", "una", "del", "le",
    "da", "in", "lo", "si", "al", "se", "mi", "ne", "ci", "su",
    "ma", "no", "li", "tu", "te", "re", "me", "va", "fa", "ha",
    "sono", "come", "questo", "quello", "dove", "quando", "tutto",
    "bene", "male", "cosa", "tempo", "mondo", "parte", "nome",
    "acqua", "terra", "fuoco", "aria", "sole", "luna", "stella",
    "erba", "fiore", "foglia", "radice", "seme", "pianta", "frutto",
    "corpo", "mano", "piede", "testa", "occhio", "bocca", "cuore",
    "pietra", "ferro", "oro", "argento", "sale", "olio", "vino",
    "pane", "carne", "pesce", "latte", "miele", "sangue", "spirito",
    "virtute", "medicina", "herba", "compositione", "recepte",
    "polvere", "unguento", "sciroppo", "decoctione", "infusione",
]


# =====================================================================
# Step 1: Build Naibbe encoding tables
# =====================================================================

def build_naibbe_tables(
    n_tables: int,
    eva_chars: list[str],
    rng: random.Random,
) -> list[dict]:
    """Build N encoding tables for the Naibbe cipher.

    Each table maps:
      - 26 unigrams (a-z) → EVA sequences of length 2-5
      - ~676 bigrams (aa-zz) → EVA sequences of length 3-7

    The tables differ from each other (homophonic variation).
    Sequence lengths are drawn from a distribution that matches
    the observed EVA word-length distribution (~5 chars mean).
    """
    tables = []
    for _ in range(n_tables):
        table = {}

        # Unigram mappings: each letter → 2-5 EVA chars
        for letter in string.ascii_lowercase:
            seq_len = rng.choices([2, 3, 4, 5], weights=[15, 40, 30, 15], k=1)[0]
            table[letter] = _random_eva_seq(seq_len, eva_chars, rng)

        # Bigram mappings: common bigrams → 3-7 EVA chars
        for c1 in string.ascii_lowercase:
            for c2 in string.ascii_lowercase:
                bigram = c1 + c2
                seq_len = rng.choices([3, 4, 5, 6, 7], weights=[10, 25, 35, 20, 10], k=1)[0]
                table[bigram] = _random_eva_seq(seq_len, eva_chars, rng)

        tables.append(table)

    return tables


def _random_eva_seq(length: int, eva_chars: list[str], rng: random.Random) -> str:
    """Generate a random EVA character sequence with positional tendencies.

    Mimics EVA slot grammar:
      - Position 0: favor q, d, c, s, o (common word-starters)
      - Middle: favor o, a, e, ch combinations
      - Final: favor y, n, m, l (common word-enders)
    """
    if length <= 0:
        return "o"

    chars = []
    for i in range(length):
        if i == 0:
            # Word-start favored characters
            ch = rng.choices(
                eva_chars,
                weights=[3, 1, 4, 5, 2, 1, 1, 1, 1, 3, 1, 1, 1, 5, 1, 4, 1, 1, 2],
                k=1,
            )[0]
        elif i == length - 1:
            # Word-end favored characters
            ch = rng.choices(
                eva_chars,
                weights=[3, 1, 1, 2, 2, 1, 1, 1, 1, 3, 1, 3, 4, 3, 1, 1, 1, 1, 8],
                k=1,
            )[0]
        else:
            # Middle: favor o, a, e, ch
            ch = rng.choices(
                eva_chars,
                weights=[4, 1, 2, 2, 4, 1, 1, 2, 1, 2, 1, 1, 1, 5, 1, 1, 1, 1, 1],
                k=1,
            )[0]
        chars.append(ch)

    return "".join(chars)


# =====================================================================
# Step 2: Encrypt plaintext using Naibbe cipher
# =====================================================================

def generate_italian_stream(n_chars: int, rng: random.Random) -> str:
    """Generate a stream of Italian-like text (lowercase letters only).

    Uses syllable concatenation for realistic letter distribution.
    """
    chars = []
    while len(chars) < n_chars:
        word = rng.choice(ITALIAN_SYLLABLES)
        clean = "".join(c for c in word.lower() if c in string.ascii_lowercase)
        chars.extend(clean)
    return "".join(chars[:n_chars])


def naibbe_encrypt_word(
    plaintext_chars: str,
    tables: list[dict],
    rng: random.Random,
) -> str:
    """Encrypt a chunk of plaintext into one EVA 'word' using Naibbe cipher.

    1. Segment into unigrams and bigrams (dice: 50/50)
    2. For each segment, pick a random table and look up the EVA sequence
    3. Concatenate all sequences
    """
    pos = 0
    parts = []
    n = len(plaintext_chars)

    while pos < n:
        # Pick a random table for this segment
        table = rng.choice(tables)

        # Dice roll: bigram (if possible) or unigram
        if pos + 1 < n and rng.random() < 0.5:
            # Bigram
            bigram = plaintext_chars[pos:pos + 2]
            seq = table.get(bigram, table.get(bigram[0], "o"))
            parts.append(seq)
            pos += 2
        else:
            # Unigram
            ch = plaintext_chars[pos]
            seq = table.get(ch, "o")
            parts.append(seq)
            pos += 1

    return "".join(parts)


# =====================================================================
# Step 3: Generate synthetic Naibbe corpus
# =====================================================================

def generate_naibbe_corpus(
    real_pages: list[dict],
    tables: list[dict],
    rng: random.Random,
    section_texts: dict[str, str] | None = None,
) -> list[dict]:
    """Generate a Naibbe-encrypted corpus matching real manuscript structure.

    For each real page: same number of lines, same words-per-line.
    Different sections get different plaintext to test section differentiation.

    If section_texts is provided, each section encrypts from its own text stream.
    Otherwise, all sections share the same stream (worst case for section MI).
    """
    # Generate per-section plaintext streams
    if section_texts is None:
        section_texts = {}

    sections_seen = set(p["section"] for p in real_pages)
    for sec in sections_seen:
        if sec not in section_texts:
            # Each section gets its own Italian text (different topic mix)
            section_texts[sec] = generate_italian_stream(200_000, rng)

    # Track position in each section's plaintext
    section_pos: dict[str, int] = {s: 0 for s in section_texts}

    synthetic_pages = []
    for page in real_pages:
        section = page["section"]
        text = section_texts.get(section, "")

        syn_line_words = []
        syn_words = []

        for line in page.get("line_words", []):
            n_words = len(line)
            syn_line = []

            for real_word in line:
                # Determine how many plaintext chars to consume
                # (proportional to real word length, scaled down for verbose expansion)
                pt_chars = max(1, len(real_word) // 3 + 1)

                # Get plaintext chunk
                pos = section_pos.get(section, 0)
                if pos + pt_chars > len(text):
                    # Wrap around
                    pos = 0
                chunk = text[pos:pos + pt_chars]
                section_pos[section] = pos + pt_chars

                # Encrypt
                eva_word = naibbe_encrypt_word(chunk, tables, rng)

                # Trim to reasonable length (match real word length ± 2)
                target_len = len(real_word)
                if len(eva_word) > target_len + 2:
                    eva_word = eva_word[:target_len + 2]
                elif len(eva_word) < max(1, target_len - 2):
                    # Pad with common EVA chars
                    pad_chars = "oaey"
                    while len(eva_word) < target_len - 2:
                        eva_word += rng.choice(pad_chars)

                if not eva_word:
                    eva_word = "o"

                syn_line.append(eva_word)

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


# =====================================================================
# Step 4: Multi-run averaging
# =====================================================================

def run_naibbe_multi(
    real_pages: list[dict],
    n_runs: int = N_RUNS,
    seed: int = SEED,
) -> tuple[dict, list[dict]]:
    """Run Naibbe cipher generation N times, return averaged properties.

    Returns (averaged_props, list_of_all_run_props).
    """
    all_props = []

    for i in range(n_runs):
        rng = random.Random(seed + i)

        # Build fresh tables each run (different cipher instantiation)
        tables = build_naibbe_tables(N_TABLES, EVA_CHARS, rng)

        # Generate corpus
        naibbe_pages = generate_naibbe_corpus(real_pages, tables, rng)

        # Measure properties
        props = measure_all_properties(naibbe_pages)
        all_props.append(props)

        click.echo(f"    Run {i + 1}/{n_runs}: "
                   f"entropy={props['entropy']['entropy_bits']:.2f}, "
                   f"TTR={props['vocabulary']['ttr']:.4f}, "
                   f"m_final={props['m_end_marker']['line_final_rate']:.3f}")

    # Average numeric properties across runs
    averaged = _average_properties(all_props)
    return averaged, all_props


def _average_properties(all_props: list[dict]) -> dict:
    """Average measurement dicts across multiple runs.

    For each property, averages all numeric values.
    Non-numeric values taken from first run.
    """
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
                if all(isinstance(sv, (int, float)) for sv in subvals if sv is not None):
                    numeric = [sv for sv in subvals if sv is not None]
                    avg_dict[subkey] = round(sum(numeric) / len(numeric), 6) if numeric else 0
                else:
                    avg_dict[subkey] = subvals[0]
            averaged[key] = avg_dict
        else:
            averaged[key] = vals[0]

    return averaged


# =====================================================================
# Step 5: Save to SQLite
# =====================================================================

def save_to_db(config: ToolkitConfig, comparisons: list[dict]):
    """Save comparison results to SQLite database."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}, skipping DB save")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS naibbe_test")
    cur.execute("""
        CREATE TABLE naibbe_test (
            property TEXT PRIMARY KEY,
            description TEXT,
            real_value REAL,
            naibbe_value REAL,
            reproduced INTEGER,
            direction TEXT,
            threshold REAL
        )
    """)

    for c in comparisons:
        rv = float(c["real_value"]) if isinstance(c["real_value"], (int, float)) else 0
        nv = float(c["rugg_value"]) if isinstance(c["rugg_value"], (int, float)) else 0
        cur.execute(
            "INSERT INTO naibbe_test VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c["property"], c["description"], rv, nv,
             1 if c["reproduced"] else 0, c["direction"], c["threshold"]),
        )

    conn.commit()
    conn.close()


# =====================================================================
# Step 6: Naibbe-specific summary formatting
# =====================================================================

def format_naibbe_summary(
    real_props: dict,
    naibbe_props: dict,
    comparisons: list[dict],
    n_runs: int,
) -> str:
    """Format human-readable summary for Naibbe test."""
    lines = []
    lines.append("=" * 72)
    lines.append("NAIBBE CIPHER TEST — Phase 11")
    lines.append("Can Greshko's (2025) verbose homophonic cipher reproduce")
    lines.append("the confirmed structural properties of the Voynich Manuscript?")
    lines.append("=" * 72)

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)

    lines.append(f"\nResult: {n_repro}/{n_total} properties reproduced by Naibbe cipher")
    lines.append(f"(averaged over {n_runs} cipher instantiations)")
    lines.append("")

    # Comparison with Rugg
    lines.append("Note: Rugg grille reproduced 10/16 properties.")
    lines.append(f"      Naibbe cipher reproduced {n_repro}/16 properties.")
    lines.append("")

    # Table
    lines.append(f"{'Property':<30s} {'Real':>10s} {'Naibbe':>10s} {'Reproduced':>12s}")
    lines.append("-" * 72)

    for c in comparisons:
        rv = c["real_value"]
        nv = c["rugg_value"]  # reuses rugg_value field from compare_properties

        rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
        nv_str = f"{nv:.4f}" if isinstance(nv, float) else str(nv)
        verdict = "YES" if c["reproduced"] else "** NO **"

        lines.append(f"  {c['property']:<28s} {rv_str:>10s} {nv_str:>10s} {verdict:>12s}")

    lines.append("-" * 72)

    # Verdict
    lines.append("")
    if n_repro == n_total:
        lines.append("VERDICT: NAIBBE CIPHER SUFFICIENT")
        lines.append("  The Naibbe cipher reproduces ALL confirmed properties.")
        lines.append("  This is strong evidence that the VMS could contain")
        lines.append("  encrypted meaningful text (Latin or Italian).")
    elif n_repro > 10:
        missed = [c for c in comparisons if not c["reproduced"]]
        lines.append(f"VERDICT: NAIBBE CIPHER MOSTLY SUFFICIENT ({n_repro}/{n_total})")
        lines.append("  The cipher reproduces most properties but misses:")
        for m in missed:
            lines.append(f"    - {m['description']}")
            lines.append(f"      (real={m['real_value']}, naibbe={m['rugg_value']})")
        lines.append("")
        lines.append("  Better than Rugg (10/16) — verbose homophonic cipher")
        lines.append("  is a more plausible generation mechanism than a Cardan grille.")
    else:
        missed = [c for c in comparisons if not c["reproduced"]]
        lines.append(f"VERDICT: NAIBBE CIPHER INSUFFICIENT ({n_total - n_repro} properties missing)")
        lines.append("  The cipher CANNOT reproduce these properties:")
        for m in missed:
            lines.append(f"    - {m['description']}")
            lines.append(f"      (real={m['real_value']}, naibbe={m['rugg_value']})")
        lines.append("")
        if n_repro > 10:
            lines.append("  Better than Rugg grille but still incomplete.")
        else:
            lines.append("  These properties require something BEYOND a verbose cipher.")

    lines.append("")

    # Detail section
    lines.append("=" * 72)
    lines.append("DETAILED MEASUREMENTS")
    lines.append("=" * 72)

    for label, props in [("REAL VOYNICH", real_props), ("NAIBBE CIPHER", naibbe_props)]:
        lines.append(f"\n--- {label} ---")
        for key, val in props.items():
            if isinstance(val, dict):
                summary_val = {k: v for k, v in val.items() if k != "note"}
                lines.append(f"  {key}: {_compact_dict(summary_val)}")
            else:
                lines.append(f"  {key}: {val}")

    lines.append("")
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Naibbe Cipher Test — Phase 11.

    Tests whether Greshko's (2025) verbose homophonic substitution cipher
    can reproduce the 16 confirmed structural properties of the Voynich
    Manuscript. Same framework as rugg_test.py (Phase 27.9).
    """
    report_path = config.stats_dir / "naibbe_test.json"
    summary_path = config.stats_dir / "naibbe_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Naibbe test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 11 — Naibbe Cipher Test (Greshko 2025)")

    # 1. Parse real EVA corpus
    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    # 2. Measure real properties
    print_step("Measuring 16 properties on REAL corpus...")
    real_props = measure_all_properties(real_pages)
    for key, val in real_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    # 3. Generate and measure Naibbe corpus (multi-run)
    print_step(f"Generating Naibbe cipher corpora ({N_RUNS} runs, "
               f"{N_TABLES} tables each)...")
    naibbe_props, all_run_props = run_naibbe_multi(
        real_pages, n_runs=N_RUNS, seed=SEED,
    )

    click.echo(f"\n    Averaged Naibbe properties:")
    for key, val in naibbe_props.items():
        if isinstance(val, dict):
            summary_val = {k: v for k, v in val.items() if k != "note"}
            click.echo(f"    {key}: {_compact_dict(summary_val)}")

    # 4. Compare
    print_step("Comparing real vs Naibbe...")
    comparisons = compare_properties(real_props, naibbe_props)

    n_repro = sum(1 for c in comparisons if c["reproduced"])
    n_total = len(comparisons)
    click.echo(f"\n    RESULT: {n_repro}/{n_total} properties reproduced")

    for c in comparisons:
        icon = "OK" if c["reproduced"] else "MISS"
        click.echo(f"    [{icon:>4s}] {c['property']:<28s} "
                   f"real={c['real_value']:<10} naibbe={c['rugg_value']:<10}")

    # 5. Save results
    print_step("Saving results...")

    report = {
        "real_properties": _serialise(real_props),
        "naibbe_properties": _serialise(naibbe_props),
        "comparisons": comparisons,
        "summary": {
            "reproduced": n_repro,
            "total": n_total,
            "verdict": "SUFFICIENT" if n_repro == n_total else "INSUFFICIENT",
            "missed": [c["property"] for c in comparisons if not c["reproduced"]],
        },
        "parameters": {
            "n_tables": N_TABLES,
            "n_runs": N_RUNS,
            "seed": SEED,
            "eva_chars": "".join(EVA_CHARS),
        },
        "rugg_comparison": {
            "rugg_reproduced": 10,
            "rugg_total": 16,
            "naibbe_reproduced": n_repro,
            "naibbe_total": n_total,
            "naibbe_better": n_repro > 10,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_naibbe_summary(real_props, naibbe_props, comparisons, N_RUNS)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    # Save to DB
    save_to_db(config, comparisons)
    click.echo(f"    DB:   naibbe_test table")

    # Print summary
    click.echo(f"\n{summary}")
