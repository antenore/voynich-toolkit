"""
Phase 1a — per-hand z-score with permutation test (Hypothesis C).

Null hypothesis: the Hebrew signal is uniform across Davis hands.
Test: permutation test of the EVA→Hebrew mapping on each hand separately,
including small hands (< 2000 tokens) that scribe_analysis.py excludes.

For small hands (< 2000 tokens): n_perms=1000 to lower the p-floor
from 0.005 to 0.001. The test is the same — permutes the mapping, not the data.

Answers the question: "is the signal concentrated in 1-2 hands or distributed?"

Output:
  hand_zscore.json
  hand_zscore_summary.txt
  DB table: hand_zscore
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import click

from .config import ToolkitConfig
from .currier_split import decode_and_match, run_permutation, two_proportion_ztest
from .full_decode import FULL_MAPPING
from .mapping_audit import load_honest_lexicon
from .permutation_stats import build_full_mapping
from .scribe_analysis import HAND_NAMES, MIN_WORDS_STATS, split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# Hands 1 and 2 already have 200 perms in scribe_analysis — here 1000 for all
N_PERMS_LARGE = 1000   # >= 2000 tokens
N_PERMS_SMALL = 1000   # < 2000 tokens (same test, same resolution)
SEED_BASE = 42

# Certified Davis hands to exclude from "anonymous hands" analysis
DAVIS_HANDS = {"1", "2", "3", "4", "5"}


# =====================================================================
# Core analysis
# =====================================================================

def run_all_hands_perm(corpus: dict, full_lex: set, full_map: dict) -> dict:
    """Permutation test for each hand with enough words.

    Returns: dict[hand] → permutation test result dict
    """
    results = {}
    for hand in sorted(corpus.keys()):
        words = corpus[hand]["words"]
        n_decoded = decode_and_match(words, full_lex)["n_decoded"]
        if n_decoded < MIN_WORDS_STATS:
            click.echo(f"    Hand {hand}: skip (<{MIN_WORDS_STATS} decoded)")
            continue

        n_perms = N_PERMS_LARGE if n_decoded >= 2000 else N_PERMS_SMALL
        seed = SEED_BASE + int(hand) if hand.isdigit() else SEED_BASE + ord(hand[0])

        click.echo(f"    Hand {hand} ({n_decoded:,} decoded, {n_perms} perms)...",
                   nl=False)
        perm = run_permutation(words, full_lex, full_map,
                               n_perms=n_perms, seed=seed)
        results[hand] = perm
        sig = ("***" if perm["significant_001"] else
               "**"  if perm["significant_01"]  else
               "*"   if perm["significant_05"]  else "ns")
        click.echo(f" z={perm['z_score']:.2f}  p={perm['p_value']:.4f}  {sig}")

    return results


def pairwise_vs_corpus(corpus: dict, full_lex: set) -> dict:
    """Two-proportion z-test: each hand vs pooled rate of the entire corpus.

    Null: the hand has the same match rate as the overall corpus.
    """
    # Pooled rate
    all_words = [w for c in corpus.values() for w in c["words"]]
    total = decode_and_match(all_words, full_lex)

    results = {}
    for hand in sorted(corpus.keys()):
        words = corpus[hand]["words"]
        stats = decode_and_match(words, full_lex)
        if stats["n_decoded"] < MIN_WORDS_STATS:
            continue
        zt = two_proportion_ztest(
            total["n_matched"], total["n_decoded"],
            stats["n_matched"], stats["n_decoded"],
        )
        results[hand] = {
            "corpus_rate": total["match_rate"],
            "hand_rate": stats["match_rate"],
            "n_decoded": stats["n_decoded"],
            **zt,
        }

    return results


# =====================================================================
# DB
# =====================================================================

def save_to_db(perm_results: dict, pairwise: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS hand_zscore")
    cur.execute("""
        CREATE TABLE hand_zscore (
            hand            TEXT PRIMARY KEY,
            hand_name       TEXT,
            n_decoded       INTEGER,
            match_rate      REAL,
            corpus_rate     REAL,
            z_score_perm    REAL,
            p_value_perm    REAL,
            significant_001 INTEGER,
            significant_05  INTEGER,
            z_vs_corpus     REAL,
            p_vs_corpus     REAL,
            n_perms         INTEGER,
            note            TEXT
        )
    """)

    rows = []
    all_hands = set(perm_results) | set(pairwise)
    for hand in sorted(all_hands):
        p = perm_results.get(hand, {})
        pw = pairwise.get(hand, {})
        note = "small (<2000)" if pw.get("n_decoded", 0) < 2000 else ""
        rows.append((
            hand,
            HAND_NAMES.get(hand, "?"),
            pw.get("n_decoded"),
            pw.get("hand_rate"),
            pw.get("corpus_rate"),
            p.get("z_score"),
            p.get("p_value"),
            int(p.get("significant_001", False)),
            int(p.get("significant_05", False)),
            pw.get("z_score"),
            pw.get("p_value"),
            p.get("n_perms"),
            note,
        ))

    cur.executemany("""
        INSERT INTO hand_zscore VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


# =====================================================================
# Summary formatter
# =====================================================================

def format_summary(perm_results: dict, pairwise: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  PHASE 1a — Z-score per hand (Hypothesis C: uniform signal?)")
    lines.append("=" * 78)

    lines.append(
        f"\n  Null: each hand has the same match rate as the pooled corpus.\n"
        f"  Test 1: permutation test mapping Hypothesis A (real map vs 1000 random).\n"
        f"  Test 2: two-proportion z-test vs pooled corpus rate.\n"
        f"  Lexicon: Hebrew origin (STEPBible+Jastrow+Klein) = Hypothesis A benchmark,\n"
        f"  NOT proof of linguistic origin.\n"
    )

    # Main table
    lines.append(
        f"  {'Hand':>5}  {'Name':20}  {'N dec':>6}  {'Rate':>6}  "
        f"{'z_perm':>7}  {'p_perm':>7}  Sig  "
        f"{'z_vs_corp':>9}  Note"
    )
    lines.append("  " + "-" * 80)

    all_hands = sorted(set(perm_results) | set(pairwise))
    for hand in all_hands:
        p = perm_results.get(hand, {})
        pw = pairwise.get(hand, {})
        n = pw.get("n_decoded", 0)
        rate = pw.get("hand_rate", 0)
        zp = f"{p['z_score']:7.2f}" if p else "    n/a"
        pp = f"{p['p_value']:7.4f}" if p else "    n/a"
        sig = ("***" if p.get("significant_001") else
               "**"  if p.get("significant_01")  else
               "*"   if p.get("significant_05")  else
               "ns"  if p else "  -")
        zc = f"{pw['z_score']:9.2f}" if pw else "      n/a"
        note = "small" if n < 2000 else ""
        lines.append(
            f"  {hand:>5}  {HAND_NAMES.get(hand,'?'):20}  {n:>6,}  "
            f"{rate*100:>5.1f}%  {zp}  {pp}  {sig:3}  {zc}  {note}"
        )

    # Verdict
    n_sig = sum(1 for p in perm_results.values() if p.get("significant_05"))
    n_total = len(perm_results)
    lines.append(f"\n  {'='*60}")
    lines.append(f"  Hands with significant signal (p<0.05): {n_sig}/{n_total}")
    if n_sig == n_total:
        lines.append(
            "  VERDICT: DISTRIBUTED signal — all hands show\n"
            "  the same lexical signature. Hypothesis C (different hands = different\n"
            "  systems) not supported. Mapping consistent across all scribes."
        )
    elif n_sig == 0:
        lines.append(
            "  VERDICT: NO signal — the mapping does not work for\n"
            "  any individual hand (possible aggregate corpus artifact)."
        )
    else:
        lines.append(
            f"  VERDICT: CONCENTRATED signal — only {n_sig} hands out of {n_total}\n"
            "  show the signal. The others are noise or a different system."
        )
    lines.append(f"  {'='*60}")

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 1a: permutation test z-score for each Davis hand."""
    report_path = config.stats_dir / "hand_zscore.json"
    summary_path = config.stats_dir / "hand_zscore_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_zscore report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 1a — Z-score per Hand (Hypothesis C)")

    # 1. Parse EVA
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Load lexicon
    print_step("Loading honest lexicon...")
    honest_lex, _ = load_honest_lexicon(config)
    click.echo(f"    {len(honest_lex):,} forms")

    # Note: using FULL lexicon (as scribe_analysis) for comparability
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    with open(enriched_path) as f:
        hlex = json.load(f)
    full_lex = set(hlex["all_consonantal_forms"])
    click.echo(f"    Full: {len(full_lex):,} forms")

    # 3. Split by hand
    print_step("Splitting by hand...")
    corpus = split_corpus_by_hand(pages)

    # 4. Build full mapping
    full_map = build_full_mapping(FULL_MAPPING)

    # 5. Permutation test for each hand
    print_step(f"Permutation test ({N_PERMS_LARGE} perms for large hand, "
               f"{N_PERMS_SMALL} for small hand)...")
    perm_results = run_all_hands_perm(corpus, full_lex, full_map)

    # 6. Two-proportion z-test vs pooled corpus
    print_step("Comparison vs pooled corpus rate...")
    pairwise = pairwise_vs_corpus(corpus, full_lex)
    for hand, pw in sorted(pairwise.items()):
        click.echo(
            f"    Hand {hand}: {pw['hand_rate']*100:.1f}% vs corpus "
            f"{pw['corpus_rate']*100:.1f}%  z={pw['z_score']:.2f}  "
            f"p={pw['p_value']:.4f}"
        )

    # 7. Save JSON
    print_step("Saving...")
    report = {
        "permutation_tests": perm_results,
        "pairwise_vs_corpus": pairwise,
        "n_perms_large": N_PERMS_LARGE,
        "n_perms_small": N_PERMS_SMALL,
        "null_hypothesis": "the mapping signal (Hypothesis A) is uniform across hands",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    summary = format_summary(perm_results, pairwise)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. DB
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(perm_results, pairwise, db_path)
        click.echo(f"    DB: {db_path} ✓")

    click.echo(f"\n{summary}")
