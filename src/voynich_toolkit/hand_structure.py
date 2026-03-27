"""
Phase 2 — EVA structure per hand with null model (Hypothesis D/C).

Purely structural analysis: zero lexicon, zero decoding, raw EVA only.

Sub-analyses:
  2a — Shannon entropy with null model (random sampling of N tokens from corpus)
  2b — Zipf slope with null model
  2c — Jaccard matrix across all pairs of Davis hands 1–5
  2d — EVA bigram distribution with chi-square (uniformity across hands)

Null model (2a, 2b): for each hand of size N, sample 500 times N tokens
from the global corpus and compute the statistic distribution. The hand z-score
is (observed - mean_null) / std_null.

Caveat (Heaps' law): TTR and Zipf slope are unstable for hands < 200 types.
Hands 4, 5, Y (< 1000 tokens) are flagged but included.

Output:
  hand_structure.json
  hand_structure_summary.txt
  DB table: hand_structure
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import chi2 as scipy_chi2

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .scribe_analysis import HAND_NAMES, split_corpus_by_hand
from .hand_characterization import eva_profile
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Parameters
# =====================================================================

N_NULL_SAMPLES = 500     # samples for the null model
SEED = 42
MIN_TOKENS_STATS = 200   # below this threshold: flag instability
DAVIS_HANDS = {"1", "2", "3", "4", "5"}   # certified hands for Jaccard


# =====================================================================
# Phase 2a/2b — Null model: entropy and Zipf
# =====================================================================

def null_distribution(all_words: list[str], n: int, n_samples: int = N_NULL_SAMPLES,
                      seed: int = SEED) -> dict:
    """Sample n_samples times N tokens from the global corpus.

    Returns: dict with:
      entropy_mean, entropy_std, zipf_mean, zipf_std
      (each value is the mean/std over the null model)
    """
    rng = random.Random(seed)
    entropies = []
    zipfs = []

    for _ in range(n_samples):
        sample = rng.choices(all_words, k=n)
        p = eva_profile(sample)
        entropies.append(p["shannon_entropy"])
        if p["zipf_slope"] is not None:
            zipfs.append(p["zipf_slope"])

    return {
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std":  float(np.std(entropies, ddof=1)),
        "zipf_mean":    float(np.mean(zipfs)) if zipfs else None,
        "zipf_std":     float(np.std(zipfs, ddof=1)) if zipfs else None,
    }


def z_score_vs_null(observed: float, null_mean: float, null_std: float) -> float | None:
    """z = (obs - mean) / std. Returns None if std ≈ 0."""
    if null_std is None or null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


def entropy_zipf_with_null(corpus: dict, all_words: list[str]) -> dict:
    """Compute entropy and Zipf for each hand with z-score vs null model.

    Returns: dict[hand] → {entropy, zipf_slope, null_*, z_entropy, z_zipf,
                            n_tokens, unstable_flag}
    """
    results = {}
    for hand in sorted(corpus.keys()):
        words = corpus[hand]["words"]
        n = len(words)
        if n < 10:
            continue

        click.echo(f"    Hand {hand} ({n:,} tokens, {N_NULL_SAMPLES} null samples)...",
                   nl=False)

        profile = eva_profile(words)
        null = null_distribution(all_words, n, n_samples=N_NULL_SAMPLES,
                                 seed=SEED + (int(hand) if hand.isdigit()
                                              else ord(hand[0])))

        z_ent = z_score_vs_null(
            profile["shannon_entropy"], null["entropy_mean"], null["entropy_std"])
        z_zip = None
        if profile["zipf_slope"] is not None and null["zipf_mean"] is not None:
            z_zip = z_score_vs_null(
                profile["zipf_slope"], null["zipf_mean"], null["zipf_std"])

        unstable = n < 1000

        results[hand] = {
            "n_tokens":        n,
            "entropy_obs":     profile["shannon_entropy"],
            "entropy_null_mean": round(null["entropy_mean"], 4),
            "entropy_null_std":  round(null["entropy_std"], 6),
            "z_entropy":       round(z_ent, 3) if z_ent is not None else None,
            "zipf_obs":        profile["zipf_slope"],
            "zipf_null_mean":  round(null["zipf_mean"], 4) if null["zipf_mean"] else None,
            "zipf_null_std":   round(null["zipf_std"], 6) if null["zipf_std"] else None,
            "z_zipf":          round(z_zip, 3) if z_zip is not None else None,
            "unstable_flag":   unstable,
        }

        flag = " [unstable < 1000 tokens]" if unstable else ""
        z_str = f"{z_ent:+.2f}" if z_ent is not None else "n/a"
        click.echo(f" H={profile['shannon_entropy']:.3f} z={z_str}{flag}")

    return results


# =====================================================================
# Phase 2c — Jaccard matrix
# =====================================================================

def jaccard_coefficient(set_a: set, set_b: set) -> float:
    """Jaccard = |A∩B| / |A∪B|. Returns 0 if both empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def jaccard_null(set_a: set, set_b: set, all_words: list[str],
                 n_samples: int = N_NULL_SAMPLES, seed: int = SEED) -> dict:
    """Null model for Jaccard: sample two sets of size |A| and |B| from corpus.

    The size of the random sets equals the number of distinct TYPES
    (not tokens) of the real hands, to preserve comparability.
    """
    rng = random.Random(seed)
    # Use distinct types — sample unique words from the global pool
    all_types = list(set(all_words))
    na, nb = len(set_a), len(set_b)
    # Cap to pool size
    na = min(na, len(all_types))
    nb = min(nb, len(all_types))

    nulls = []
    for _ in range(n_samples):
        sample_a = set(rng.sample(all_types, na))
        sample_b = set(rng.sample(all_types, nb))
        nulls.append(jaccard_coefficient(sample_a, sample_b))

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std":  float(np.std(nulls, ddof=1)),
    }


def jaccard_matrix(corpus: dict, all_words: list[str],
                   hands: set[str] = DAVIS_HANDS) -> dict:
    """Compute Jaccard matrix for all pairs of Davis hands 1–5.

    Returns: dict with pairs "H1-H2" → {jaccard, null_mean, null_std, z_jaccard}
    """
    # Vocabularies per hand (distinct EVA types)
    vocabs = {}
    for hand in sorted(corpus.keys()):
        if hand not in hands:
            continue
        words = corpus[hand]["words"]
        if len(words) < 10:
            continue
        vocabs[hand] = set(words)

    results = {}
    sorted_hands = sorted(vocabs.keys())
    for i, h1 in enumerate(sorted_hands):
        for h2 in sorted_hands[i+1:]:
            j_obs = jaccard_coefficient(vocabs[h1], vocabs[h2])
            null = jaccard_null(vocabs[h1], vocabs[h2], all_words,
                                n_samples=N_NULL_SAMPLES,
                                seed=SEED + i * 10)
            z = z_score_vs_null(j_obs, null["null_mean"], null["null_std"])
            results[f"{h1}-{h2}"] = {
                "hand_a":        h1,
                "hand_b":        h2,
                "types_a":       len(vocabs[h1]),
                "types_b":       len(vocabs[h2]),
                "jaccard_obs":   round(j_obs, 4),
                "null_mean":     round(null["null_mean"], 4),
                "null_std":      round(null["null_std"], 6),
                "z_jaccard":     round(z, 3) if z is not None else None,
            }
            click.echo(f"    {h1}-{h2}: J={j_obs:.4f}  null={null['null_mean']:.4f}"
                       f"  z={z:+.2f}" if z is not None else
                       f"    {h1}-{h2}: J={j_obs:.4f}  null={null['null_mean']:.4f}")

    return results


# =====================================================================
# Phase 2d — Chi-square bigrams across hands
# =====================================================================

def bigram_freq(words: list[str]) -> Counter:
    """Count EVA bigrams in a word list."""
    bg = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bg[w[i:i+2]] += 1
    return bg


def bigram_chisquare(corpus: dict, all_words: list[str]) -> dict:
    """Chi-square: does the bigram distribution of each hand match the corpus?

    H0: bigrams of hand X ~ global corpus.
    Uses only the top-50 corpus bigrams for stability.

    Returns: dict[hand] → {chi2, df, p_value, significant_05, n_bigrams}
    """
    # Global distribution
    global_bg = bigram_freq(all_words)
    top50 = [bg for bg, _ in global_bg.most_common(50)]
    global_total = sum(global_bg[bg] for bg in top50)
    global_freq = np.array([global_bg[bg] / global_total for bg in top50])

    results = {}
    for hand in sorted(corpus.keys()):
        words = corpus[hand]["words"]
        if len(words) < 50:
            continue

        hand_bg = bigram_freq(words)
        hand_total = sum(hand_bg[bg] for bg in top50)
        if hand_total == 0:
            continue

        # Observed frequencies for the top-50 bigrams
        observed = np.array([hand_bg.get(bg, 0) for bg in top50], dtype=float)
        # Expected frequencies: global distribution scaled to hand size
        expected = global_freq * hand_total

        # Remove cells with expected < 5 for chi-square validity
        mask = expected >= 5
        obs_filt = observed[mask]
        exp_filt = expected[mask]
        df = int(mask.sum()) - 1

        if df <= 0 or exp_filt.sum() == 0:
            results[hand] = {"skipped": True, "reason": "expected < 5 in too many cells"}
            continue

        chi2_stat = float(np.sum((obs_filt - exp_filt) ** 2 / exp_filt))
        p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

        results[hand] = {
            "n_bigrams_total": int(hand_total),
            "n_cells_used":    int(mask.sum()),
            "chi2":            round(chi2_stat, 2),
            "df":              df,
            "p_value":         round(p_value, 6),
            "significant_05":  bool(p_value < 0.05),
            "significant_001": bool(p_value < 0.001),
        }

        sig = ("***" if p_value < 0.001 else
               "**"  if p_value < 0.01 else
               "*"   if p_value < 0.05 else "ns")
        click.echo(f"    Hand {hand}: chi2={chi2_stat:.1f}  df={df}  "
                   f"p={p_value:.6f}  {sig}")

    return results


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(entropy_zipf: dict, jaccard: dict, bigrams: dict,
               db_path: Path) -> None:
    """Write results to DB tables hand_structure_*."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Table 2a/2b — entropy + Zipf
    cur.execute("DROP TABLE IF EXISTS hand_structure_entropy")
    cur.execute("""
        CREATE TABLE hand_structure_entropy (
            hand              TEXT PRIMARY KEY,
            n_tokens          INTEGER,
            entropy_obs       REAL,
            entropy_null_mean REAL,
            entropy_null_std  REAL,
            z_entropy         REAL,
            zipf_obs          REAL,
            zipf_null_mean    REAL,
            zipf_null_std     REAL,
            z_zipf            REAL,
            unstable_flag     INTEGER
        )
    """)
    for hand, d in sorted(entropy_zipf.items()):
        cur.execute("""
            INSERT INTO hand_structure_entropy VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            hand, d["n_tokens"],
            d["entropy_obs"], d["entropy_null_mean"], d["entropy_null_std"],
            d["z_entropy"],
            d["zipf_obs"], d["zipf_null_mean"], d["zipf_null_std"],
            d["z_zipf"],
            int(d["unstable_flag"]),
        ))

    # Table 2c — Jaccard
    cur.execute("DROP TABLE IF EXISTS hand_structure_jaccard")
    cur.execute("""
        CREATE TABLE hand_structure_jaccard (
            pair        TEXT PRIMARY KEY,
            hand_a      TEXT,
            hand_b      TEXT,
            types_a     INTEGER,
            types_b     INTEGER,
            jaccard_obs REAL,
            null_mean   REAL,
            null_std    REAL,
            z_jaccard   REAL
        )
    """)
    for pair, d in sorted(jaccard.items()):
        cur.execute("""
            INSERT INTO hand_structure_jaccard VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            pair, d["hand_a"], d["hand_b"],
            d["types_a"], d["types_b"],
            d["jaccard_obs"], d["null_mean"], d["null_std"],
            d["z_jaccard"],
        ))

    # Table 2d — bigrams chi-square
    cur.execute("DROP TABLE IF EXISTS hand_structure_bigrams")
    cur.execute("""
        CREATE TABLE hand_structure_bigrams (
            hand              TEXT PRIMARY KEY,
            n_bigrams_total   INTEGER,
            n_cells_used      INTEGER,
            chi2              REAL,
            df                INTEGER,
            p_value           REAL,
            significant_05    INTEGER,
            significant_001   INTEGER
        )
    """)
    for hand, d in sorted(bigrams.items()):
        if d.get("skipped"):
            continue
        cur.execute("""
            INSERT INTO hand_structure_bigrams VALUES (?,?,?,?,?,?,?,?)
        """, (
            hand, d["n_bigrams_total"], d["n_cells_used"],
            d["chi2"], d["df"], d["p_value"],
            int(d["significant_05"]), int(d["significant_001"]),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(entropy_zipf: dict, jaccard: dict, bigrams: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 2 — EVA structure per hand with null model")
    lines.append("=" * 80)

    # 2a/2b — Entropy + Zipf
    lines.append("\n── Phase 2a/2b — Shannon Entropy and Zipf Slope (vs null model) ──")
    lines.append(
        f"  {'Hand':>5}  {'N':>7}  {'H_obs':>7}  {'H_null':>7}  {'z_H':>6}  "
        f"{'Zipf_obs':>8}  {'Zipf_null':>9}  {'z_Zipf':>7}  Flag"
    )
    lines.append("  " + "-" * 78)
    for hand in sorted(entropy_zipf.keys()):
        d = entropy_zipf[hand]
        z_h = f"{d['z_entropy']:+.2f}" if d["z_entropy"] is not None else "  n/a"
        z_z = f"{d['z_zipf']:+.2f}" if d["z_zipf"] is not None else "  n/a"
        zipf_o = f"{d['zipf_obs']:.3f}" if d["zipf_obs"] is not None else "  n/a"
        zipf_n = f"{d['zipf_null_mean']:.3f}" if d["zipf_null_mean"] else "  n/a"
        flag = "⚠ unstable" if d["unstable_flag"] else ""
        lines.append(
            f"  {hand:>5}  {d['n_tokens']:>7,}  {d['entropy_obs']:>7.4f}  "
            f"{d['entropy_null_mean']:>7.4f}  {z_h:>6}  {zipf_o:>8}  "
            f"{zipf_n:>9}  {z_z:>7}  {flag}"
        )

    # 2c — Jaccard
    lines.append("\n── Phase 2c — Jaccard matrix (EVA vocabulary across Davis hands 1–5) ──")
    lines.append(
        f"  {'Pair':>6}  {'TypA':>5}  {'TypB':>5}  {'J_obs':>7}  "
        f"{'J_null':>7}  {'z_J':>6}"
    )
    lines.append("  " + "-" * 50)
    for pair in sorted(jaccard.keys()):
        d = jaccard[pair]
        z_j = f"{d['z_jaccard']:+.2f}" if d["z_jaccard"] is not None else "  n/a"
        lines.append(
            f"  {pair:>6}  {d['types_a']:>5,}  {d['types_b']:>5,}  "
            f"{d['jaccard_obs']:>7.4f}  {d['null_mean']:>7.4f}  {z_j:>6}"
        )

    # 2d — Chi-square bigrams
    lines.append("\n── Phase 2d — Chi-square EVA bigrams (top-50, hand vs corpus) ──")
    lines.append(
        f"  {'Hand':>5}  {'N_bg':>7}  {'chi2':>7}  {'df':>4}  "
        f"{'p':>9}  {'sig':>5}"
    )
    lines.append("  " + "-" * 50)
    for hand in sorted(bigrams.keys()):
        d = bigrams[hand]
        if d.get("skipped"):
            lines.append(f"  {hand:>5}  [skip: {d.get('reason', '?')}]")
            continue
        sig = ("***" if d["significant_001"] else
               "*"   if d["significant_05"] else "ns ")
        lines.append(
            f"  {hand:>5}  {d['n_bigrams_total']:>7,}  {d['chi2']:>7.1f}  "
            f"{d['df']:>4}  {d['p_value']:>9.6f}  {sig:>5}"
        )

    lines.append("\n── Note: null model = 500 random samples of size N from corpus ──")
    lines.append("   z > +2: entropy/Jaccard/slope significantly above random")
    lines.append("   z < -2: entropy/slope significantly below random")
    lines.append("   chi2 ***: hand bigram distribution != corpus (p < 0.001)")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 2: EVA structure per hand with null model (entropy, Zipf, Jaccard, bigrams)."""
    report_path = config.stats_dir / "hand_structure.json"
    summary_path = config.stats_dir / "hand_structure_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_structure report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 2 — EVA Structure per Hand (null model)")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Split by hand
    print_step("Splitting by hand...")
    corpus = split_corpus_by_hand(pages)
    for hand in sorted(corpus.keys()):
        c = corpus[hand]
        click.echo(f"    Hand {hand}: {c['n_pages']} pages, "
                   f"{len(c['words']):,} words")

    # Global corpus (flat list) for the null model
    all_words = [w for p in pages for w in p["words"]]
    click.echo(f"    Global corpus: {len(all_words):,} words")

    # 3. Phase 2a/2b — Entropy + Zipf with null model
    print_step("Phase 2a/2b — Shannon entropy + Zipf slope (500 null samples per hand)...")
    entropy_zipf = entropy_zipf_with_null(corpus, all_words)

    # 4. Phase 2c — Jaccard matrix (Davis hands 1–5 only)
    print_step("Phase 2c — Jaccard matrix pairs Davis 1–5...")
    jaccard = jaccard_matrix(corpus, all_words)

    # 5. Phase 2d — Chi-square bigrams
    print_step("Phase 2d — Chi-square EVA bigrams (top-50)...")
    bigrams = bigram_chisquare(corpus, all_words)

    # 6. Save JSON
    print_step("Saving JSON...")
    report = {
        "entropy_zipf": entropy_zipf,
        "jaccard":      jaccard,
        "bigrams":      bigrams,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 7. Save TXT summary
    summary = format_summary(entropy_zipf, jaccard, bigrams)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. Save to DB
    print_step("Writing DB tables hand_structure_*...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(entropy_zipf, jaccard, bigrams, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found at {db_path} — skip DB write")

    click.echo(f"\n{summary}")
