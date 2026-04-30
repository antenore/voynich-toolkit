"""
Firth Sorting Cipher Test — Phase 16b.

Tests Firth's "sorting cipher" hypothesis: a wealthy 15th c. producer instructs
scribes to (1) substitute letters via a mapping table and (2) sort glyphs
inside each word according to a fixed "Voynich alphabet" order.

Generation method:
  1. Take a real natural-language corpus (Italian or Latin).
  2. Apply a monoalphabetic substitution to EVA chars.
  3. Sort each word's glyphs according to a fixed permutation order.
  4. Match Voynich page/section/line structure.

Then measure all 16 structural properties via the rugg_test harness.

Predictions (a sorting cipher should pass/fail):
  PASS frequency, Zipf, entropy, line containment, slot grammar (sort enforces it).
  FAIL section vocab MI (sort is section-blind), 'm' line-final concentration
  (sort is position-agnostic at line level), paragraph coherence, Currier A/B
  if same plaintext is used.

Reference:
  Firth, R.H. "A Voynich strategy". Reddit r/voynich, 2024.
"""

from __future__ import annotations

import json
import random
import sqlite3
import string
from collections import Counter
from pathlib import Path

import click
import numpy as np

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
EVA_CHARS = list("acdefghiklmnopqrsty")  # 19 EVA chars

# Italian-like word/syllable pool (same as naibbe_test for consistency)
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
    "essere", "avere", "potere", "volere", "sapere", "vedere",
    "capo", "piede", "ramo", "luogo", "secolo", "anno", "mese",
    "giorno", "ora", "vita", "morte", "amore", "guerra", "pace",
]


def build_substitution(rng: random.Random, plain_alphabet: list[str]) -> dict[str, str]:
    """Build a monoalphabetic substitution from plaintext alphabet to EVA.

    For each plaintext char, assign one EVA char. Some plaintext chars may
    share an EVA char (homophonic collapse) since plaintext alphabet (~26) is
    larger than EVA (19).
    """
    return {ch: rng.choice(EVA_CHARS) for ch in plain_alphabet}


def build_voynich_sort_order(rng: random.Random) -> dict[str, int]:
    """Build a fixed sort order for EVA glyphs (Firth's 'Voynich alphabet').

    Inspired by Zattera's slot order: gallows in middle, q/o/d at start,
    y/m/n at end. The exact order doesn't matter for the test — what matters
    is that we apply a consistent intra-word sort.
    """
    # A reasonable slot-grammar-friendly order (start -> end)
    base_order = list("qod4cthkfp1234aeoinmlsry")
    base_order = [c for c in base_order if c in EVA_CHARS]
    # Add any missing EVA chars at the end
    for c in EVA_CHARS:
        if c not in base_order:
            base_order.append(c)
    return {ch: i for i, ch in enumerate(base_order)}


def generate_plaintext_corpus(
    real_pages: list[dict],
    rng: random.Random,
    section_specific: bool = False,
) -> dict[str, list[list[str]]]:
    """Generate plaintext sequences (lowercase Italian-ish) matching the
    structure of real pages.

    Returns dict: folio -> list of lines, each line = list of plaintext words.
    """
    out: dict[str, list[list[str]]] = {}

    # Per-section word pools (to test if section-specific plaintext helps)
    if section_specific:
        # Group syllables/words into thematic pools
        sec_pools = {
            "H": ITALIAN_SYLLABLES[:60],   # herbal: include erbe/piante
            "P": ITALIAN_SYLLABLES[40:90],  # pharma: medicine/recepte
            "B": ITALIAN_SYLLABLES[20:70],
            "S": ITALIAN_SYLLABLES[30:80],
            "Z": ITALIAN_SYLLABLES[35:85],
            "C": ITALIAN_SYLLABLES[25:75],
            "T": ITALIAN_SYLLABLES[:],
            "?": ITALIAN_SYLLABLES[:],
        }
    else:
        # All sections share the same plaintext pool
        sec_pools = {sec: ITALIAN_SYLLABLES for sec in
                     {p["section"] for p in real_pages}}

    for page in real_pages:
        folio = page["folio"]
        pool = sec_pools.get(page["section"], ITALIAN_SYLLABLES)
        page_lines = []
        for line in page.get("line_words", []):
            n_words = len(line)
            line_pt = []
            for real_word in line:
                # Pick a plaintext word with similar length
                target_len = max(2, len(real_word))
                # Try a few times to find a word close to target_len
                candidates = [w for w in pool
                              if abs(len(w) - target_len) <= 2]
                if not candidates:
                    candidates = pool
                pt_word = rng.choice(candidates)
                line_pt.append(pt_word)
            page_lines.append(line_pt)
        out[folio] = page_lines

    return out


def encrypt_word(word: str, sub: dict[str, str], sort_order: dict[str, int]) -> str:
    """Apply substitution + intra-word sort."""
    # Substitute
    eva = "".join(sub.get(c, "o") for c in word)
    if not eva:
        return "o"
    # Intra-word sort
    chars = list(eva)
    chars.sort(key=lambda c: sort_order.get(c, 99))
    return "".join(chars)


def generate_sorting_cipher_corpus(
    real_pages: list[dict],
    rng: random.Random,
    section_specific_plaintext: bool = False,
) -> list[dict]:
    """Generate a sorting-cipher synthetic corpus.

    1. Build plaintext matching real structure
    2. Build substitution (one mapping for all sections, or per Currier language)
    3. Apply substitution + intra-word sort to each word
    """
    # Build substitution table per Currier language (A/B/?)
    plain_alphabet = list(string.ascii_lowercase)
    subs = {
        "A": build_substitution(rng, plain_alphabet),
        "B": build_substitution(rng, plain_alphabet),
        "?": build_substitution(rng, plain_alphabet),
    }
    sort_order = build_voynich_sort_order(rng)

    # Generate plaintext
    plaintext = generate_plaintext_corpus(
        real_pages, rng, section_specific=section_specific_plaintext
    )

    synthetic_pages = []
    for page in real_pages:
        folio = page["folio"]
        lang = page.get("language", "?")
        sub = subs.get(lang, subs["?"])

        pt_lines = plaintext.get(folio, [])
        syn_line_words = []
        syn_words = []

        for pt_line in pt_lines:
            syn_line = []
            for pt_word in pt_line:
                eva_word = encrypt_word(pt_word, sub, sort_order)
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


def run_sorting_multi(
    real_pages: list[dict],
    n_runs: int,
    seed: int,
    section_specific_plaintext: bool = False,
) -> tuple[dict, list[dict]]:
    """Run N sorting-cipher generations, return averaged properties."""
    all_props = []
    for i in range(n_runs):
        rng = random.Random(seed + i)
        syn_pages = generate_sorting_cipher_corpus(
            real_pages, rng, section_specific_plaintext=section_specific_plaintext,
        )
        props = measure_all_properties(syn_pages)
        all_props.append(props)
        click.echo(f"    Run {i + 1}/{n_runs}: "
                   f"entropy={props['entropy']['entropy_bits']:.2f}, "
                   f"TTR={props['vocabulary']['ttr']:.4f}, "
                   f"m_final={props['m_end_marker']['line_final_rate']:.3f}, "
                   f"MI={props['word_section_mi']['mi_bits']:.3f}")

    averaged = _average_properties(all_props)
    return averaged, all_props


def _average_properties(all_props: list[dict]) -> dict:
    """Average measurement dicts across runs (same logic as naibbe_test)."""
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


def save_to_db(config: ToolkitConfig, comparisons: list[dict], variant: str):
    """Save results to SQLite. variant = 'shared' or 'sectional'."""
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        click.echo(f"  WARNING: DB not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    table = f"firth_sorting_cipher_{variant}"
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"""
        CREATE TABLE {table} (
            property TEXT PRIMARY KEY,
            description TEXT,
            real_value REAL,
            sorting_value REAL,
            reproduced INTEGER,
            direction TEXT,
            threshold REAL
        )
    """)
    for c in comparisons:
        rv = float(c["real_value"]) if isinstance(c["real_value"], (int, float)) else 0
        sv = float(c["rugg_value"]) if isinstance(c["rugg_value"], (int, float)) else 0
        cur.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c["property"], c["description"], rv, sv,
             1 if c["reproduced"] else 0, c["direction"], c["threshold"]),
        )
    conn.commit()
    conn.close()


def format_summary(
    real_props: dict,
    sorting_props_shared: dict,
    sorting_props_sectional: dict,
    comp_shared: list[dict],
    comp_sectional: list[dict],
    n_runs: int,
) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 16b — FIRTH SORTING CIPHER TEST")
    lines.append("Can a substitution + intra-word sort cipher reproduce the")
    lines.append("16 confirmed structural properties of the Voynich Manuscript?")
    lines.append("=" * 72)

    n_shared = sum(1 for c in comp_shared if c["reproduced"])
    n_sect = sum(1 for c in comp_sectional if c["reproduced"])
    n_total = len(comp_shared)

    lines.append("")
    lines.append("Two variants tested:")
    lines.append(f"  (A) SHARED plaintext   — same Italian word pool across sections")
    lines.append(f"      Reproduced: {n_shared}/{n_total}")
    lines.append(f"  (B) SECTIONAL plaintext — different word pool per section")
    lines.append(f"      Reproduced: {n_sect}/{n_total}")
    lines.append(f"  (averaged over {n_runs} runs each)")
    lines.append("")
    lines.append("Reference: Rugg grille = 10/16, Naibbe cipher = 8/16")
    lines.append("")

    for label, comp, n in [
        ("(A) SHARED PLAINTEXT", comp_shared, n_shared),
        ("(B) SECTIONAL PLAINTEXT", comp_sectional, n_sect),
    ]:
        lines.append(f"--- {label} — {n}/{n_total} reproduced ---")
        lines.append(f"{'Property':<30s} {'Real':>10s} {'Sort':>10s} {'OK?':>8s}")
        for c in comp:
            rv = c["real_value"]
            sv = c["rugg_value"]
            rv_str = f"{rv:.4f}" if isinstance(rv, float) else str(rv)
            sv_str = f"{sv:.4f}" if isinstance(sv, float) else str(sv)
            verdict = "YES" if c["reproduced"] else "** NO **"
            lines.append(f"  {c['property']:<28s} {rv_str:>10s} {sv_str:>10s} {verdict:>8s}")
        lines.append("")

    # Verdict
    lines.append("=" * 72)
    lines.append("VERDICT")
    lines.append("=" * 72)
    best = max(n_shared, n_sect)
    if best > 12:
        lines.append(f"SORTING CIPHER STRONG: {best}/{n_total} properties reproduced.")
    elif best > 10:
        lines.append(f"SORTING CIPHER COMPETITIVE: {best}/{n_total}, beats Rugg.")
    elif best > 8:
        lines.append(f"SORTING CIPHER WEAK: {best}/{n_total}, between Rugg and Naibbe.")
    else:
        lines.append(f"SORTING CIPHER INSUFFICIENT: {best}/{n_total}, worse than Naibbe.")
    lines.append("")

    # Diagnostics: which properties are characteristically missed?
    lines.append("Properties missed in BOTH variants (architectural failures):")
    missed_shared = {c["property"] for c in comp_shared if not c["reproduced"]}
    missed_sect = {c["property"] for c in comp_sectional if not c["reproduced"]}
    both_missed = missed_shared & missed_sect
    only_shared = missed_shared - missed_sect
    only_sect = missed_sect - missed_shared
    for p in sorted(both_missed):
        lines.append(f"  - {p}")
    if only_shared:
        lines.append("\nMissed only by SHARED variant (sectional plaintext rescued):")
        for p in sorted(only_shared):
            lines.append(f"  - {p}")
    if only_sect:
        lines.append("\nMissed only by SECTIONAL variant (suspicious — should help):")
        for p in sorted(only_sect):
            lines.append(f"  - {p}")
    lines.append("")

    return "\n".join(lines) + "\n"


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 16b: Firth sorting cipher test."""
    report_path = config.stats_dir / "firth_sorting_cipher.json"
    summary_path = config.stats_dir / "firth_sorting_cipher_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Sorting cipher report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 16b — Firth Sorting Cipher Test")

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

    print_step(f"Variant A: SHARED plaintext, {N_RUNS} runs...")
    sorting_props_shared, runs_shared = run_sorting_multi(
        real_pages, N_RUNS, SEED, section_specific_plaintext=False,
    )

    print_step(f"Variant B: SECTIONAL plaintext, {N_RUNS} runs...")
    sorting_props_sectional, runs_sect = run_sorting_multi(
        real_pages, N_RUNS, SEED + 1000, section_specific_plaintext=True,
    )

    print_step("Comparing...")
    comp_shared = compare_properties(real_props, sorting_props_shared)
    comp_sect = compare_properties(real_props, sorting_props_sectional)

    n_shared = sum(1 for c in comp_shared if c["reproduced"])
    n_sect = sum(1 for c in comp_sect if c["reproduced"])
    n_total = len(comp_shared)

    click.echo(f"\n    SHARED:    {n_shared}/{n_total} reproduced")
    click.echo(f"    SECTIONAL: {n_sect}/{n_total} reproduced")

    print_step("Saving results...")
    report = {
        "real_properties": _serialise(real_props),
        "sorting_shared_properties": _serialise(sorting_props_shared),
        "sorting_sectional_properties": _serialise(sorting_props_sectional),
        "comparisons_shared": comp_shared,
        "comparisons_sectional": comp_sect,
        "summary": {
            "shared_reproduced": n_shared,
            "sectional_reproduced": n_sect,
            "total": n_total,
            "best": max(n_shared, n_sect),
            "missed_both": [
                c["property"] for c in comp_shared if not c["reproduced"]
                and c["property"] in {x["property"] for x in comp_sect if not x["reproduced"]}
            ],
        },
        "parameters": {
            "n_runs": N_RUNS,
            "seed": SEED,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(
        real_props, sorting_props_shared, sorting_props_sectional,
        comp_shared, comp_sect, N_RUNS,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, comp_shared, "shared")
    save_to_db(config, comp_sect, "sectional")
    click.echo(f"    DB:   firth_sorting_cipher_{{shared,sectional}} tables")

    click.echo(f"\n{summary}")
