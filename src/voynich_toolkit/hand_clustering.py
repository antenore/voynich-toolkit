"""
Phase 5 — Clustering across all analysis units.

Do units cluster by scribe, by section, or by Currier language?

Hand ? is split by section (Phase 2e decision): 14 units total.
  Davis certified: 1, 2, 3, 4, 5
  Unattributed single: X, Y
  Hand ? split: ?S, ?A, ?Z, ?P, ?C, ?T, ?H

Sub-analyses:
  5a — KL-divergence matrix (word-frequency distributions between unit pairs)
       Null: KL between random samples of equal size
  5b — Chi-square bigram matrix (pairwise: do two units share bigram distribution?)
       Benjamini-Hochberg FDR correction for multiple tests
  5c — Hierarchical clustering
       Feature vector: [entropy, zipf_slope, ttr, mean_word_length,
                        trigram_entropy, match_rate (from hand_zscore)]
       Ward linkage — expected to group by Currier A/B, not by section

Key question: does ?C cluster with Hand 3?
  If yes → strong evidence of mis-attribution in Davis dataset.

Output:
  hand_clustering.json
  hand_clustering_summary.txt
  DB tables: hand_clustering_kl, hand_clustering_bigrams, hand_clustering_dendrogram
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2 as scipy_chi2

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .hand_characterization import eva_profile
from .hand_positional import build_analysis_units, unit_label_name
from .hand_structure import SEED, N_NULL_SAMPLES, bigram_freq
from .scribe_analysis import HAND_NAMES, split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Phase 5a — KL-divergence matrix
# =====================================================================

def word_freq_distribution(words: list[str]) -> dict[str, float]:
    """Normalised word-type frequency distribution."""
    freq = Counter(words)
    total = sum(freq.values())
    return {w: c / total for w, c in freq.items()}


def kl_divergence(p: dict, q: dict, epsilon: float = 1e-10) -> float:
    """KL(P || Q) — sum over vocabulary of P.

    Unknown words in Q get epsilon probability.
    """
    result = 0.0
    for w, p_val in p.items():
        q_val = q.get(w, epsilon)
        result += p_val * np.log2(p_val / q_val)
    return float(result)


def symmetric_kl(p: dict, q: dict) -> float:
    """Symmetric KL: (KL(P||Q) + KL(Q||P)) / 2."""
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2.0


def kl_null(words_a: list[str], words_b: list[str],
            all_words: list[str],
            n_samples: int = N_NULL_SAMPLES, seed: int = SEED) -> dict:
    """Null model for KL: sample |A| and |B| tokens from corpus, compute sym-KL."""
    rng = random.Random(seed)
    na, nb = len(words_a), len(words_b)
    nulls = []
    for _ in range(n_samples):
        sa = rng.choices(all_words, k=na)
        sb = rng.choices(all_words, k=nb)
        pa = word_freq_distribution(sa)
        pb = word_freq_distribution(sb)
        nulls.append(symmetric_kl(pa, pb))
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std":  float(np.std(nulls, ddof=1)),
    }


def kl_matrix(units: dict[str, list[str]], all_words: list[str]) -> dict:
    """Compute symmetric KL-divergence for all unit pairs.

    Returns: dict[pair] → {kl_obs, null_mean, null_std, z_kl}
    """
    labels = sorted(units.keys())
    results = {}
    total_pairs = len(labels) * (len(labels) - 1) // 2
    done = 0
    for i, la in enumerate(labels):
        pa = word_freq_distribution(units[la])
        for lb in labels[i+1:]:
            pb = word_freq_distribution(units[lb])
            kl_obs = symmetric_kl(pa, pb)
            null = kl_null(units[la], units[lb], all_words,
                           n_samples=100,   # lighter null for pairwise matrix
                           seed=SEED + i * 100)
            z = ((kl_obs - null["null_mean"]) / null["null_std"]
                 if null["null_std"] > 1e-10 else None)
            results[f"{la}-{lb}"] = {
                "unit_a":    la,
                "unit_b":    lb,
                "kl_obs":    round(kl_obs, 4),
                "null_mean": round(null["null_mean"], 4),
                "null_std":  round(null["null_std"], 6),
                "z_kl":      round(float(z), 3) if z is not None else None,
            }
            done += 1
    click.echo(f"    {done} pairs computed")
    return results


# =====================================================================
# Phase 5b — Pairwise bigram chi-square matrix
# =====================================================================

def bigram_chisquare_pair(words_a: list[str], words_b: list[str],
                           top_n: int = 50) -> dict:
    """Chi-square: do units A and B share the same bigram distribution?

    Uses the union of top-N bigrams from both units.
    H0: same bigram distribution.
    """
    bg_a = bigram_freq(words_a)
    bg_b = bigram_freq(words_b)

    # Union of top bigrams
    top_a = {bg for bg, _ in bg_a.most_common(top_n)}
    top_b = {bg for bg, _ in bg_b.most_common(top_n)}
    vocab = sorted(top_a | top_b)

    total_a = sum(bg_a.get(bg, 0) for bg in vocab)
    total_b = sum(bg_b.get(bg, 0) for bg in vocab)
    if total_a == 0 or total_b == 0:
        return {"skipped": True, "reason": "no bigrams"}

    obs_a = np.array([bg_a.get(bg, 0) for bg in vocab], dtype=float)
    obs_b = np.array([bg_b.get(bg, 0) for bg in vocab], dtype=float)

    # Expected under H0: pool proportions
    total = total_a + total_b
    pool = (obs_a + obs_b) / total
    exp_a = pool * total_a
    exp_b = pool * total_b

    mask = (exp_a >= 5) & (exp_b >= 5)
    if mask.sum() < 2:
        return {"skipped": True, "reason": "too few cells with expected >= 5"}

    chi2_stat = float(
        np.sum((obs_a[mask] - exp_a[mask]) ** 2 / exp_a[mask]) +
        np.sum((obs_b[mask] - exp_b[mask]) ** 2 / exp_b[mask])
    )
    df = int(mask.sum()) - 1
    p_value = float(1 - scipy_chi2.cdf(chi2_stat, df))

    return {
        "chi2":    round(chi2_stat, 2),
        "df":      df,
        "p_value": round(p_value, 6),
        "reject_h0_05":  bool(p_value < 0.05),
        "reject_h0_001": bool(p_value < 0.001),
    }


def bigram_matrix(units: dict[str, list[str]]) -> dict:
    """Pairwise bigram chi-square for all unit pairs, with BH FDR correction."""
    labels = sorted(units.keys())
    raw = {}
    for i, la in enumerate(labels):
        for lb in labels[i+1:]:
            raw[f"{la}-{lb}"] = bigram_chisquare_pair(units[la], units[lb])

    # Benjamini-Hochberg FDR correction
    pairs_with_p = [(k, d["p_value"]) for k, d in raw.items()
                    if not d.get("skipped")]
    if pairs_with_p:
        pairs_with_p.sort(key=lambda x: x[1])
        m = len(pairs_with_p)
        for rank, (k, p) in enumerate(pairs_with_p, 1):
            raw[k]["p_bh"] = round(min(p * m / rank, 1.0), 6)
            raw[k]["significant_bh_05"] = bool(raw[k]["p_bh"] < 0.05)

    return raw


# =====================================================================
# Phase 5c — Hierarchical clustering
# =====================================================================

def build_feature_matrix(units: dict[str, list[str]],
                          zscore_data: dict) -> tuple[list[str], np.ndarray]:
    """Build feature matrix for hierarchical clustering.

    Features per unit (all normalised to zero-mean/unit-variance):
      - shannon_entropy
      - zipf_slope (None → corpus mean)
      - ttr
      - mean_word_length
      - trigram_entropy
      - match_rate (from hand_zscore DB if available, else 0)

    Returns: (ordered_labels, feature_matrix)
    """
    all_words = [w for words in units.values() for w in words]
    global_profile = eva_profile(all_words)

    labels = sorted(units.keys())
    rows = []
    for label in labels:
        words = units[label]
        p = eva_profile(words)

        # trigram entropy
        from .hand_positional import trigram_freq, trigram_entropy
        tg = trigram_freq(words)
        h_tg = trigram_entropy(tg)

        # match_rate from zscore_data (keyed by original hand label)
        # For ?-sections, use the ? hand match_rate as proxy
        base_hand = label if not label.startswith("?") else "?"
        match_rate = zscore_data.get(base_hand, {}).get("match_rate", 0.0)

        rows.append([
            p["shannon_entropy"],
            p["zipf_slope"] if p["zipf_slope"] is not None else global_profile["zipf_slope"],
            p["ttr"],
            p["avg_word_length"],
            h_tg,
            match_rate,
        ])

    X = np.array(rows, dtype=float)

    # Standardise (z-score normalisation per feature)
    means = X.mean(axis=0)
    stds  = X.std(axis=0, ddof=1)
    stds[stds < 1e-10] = 1.0   # avoid division by zero
    X_norm = (X - means) / stds

    return labels, X_norm


def hierarchical_clustering(labels: list[str],
                             X: np.ndarray) -> dict:
    """Ward linkage hierarchical clustering.

    Returns: dict with linkage matrix, cluster assignments (k=2,3,4),
    and the condensed distance matrix.
    """
    from scipy.spatial.distance import pdist
    dist = pdist(X, metric="euclidean")
    Z = linkage(dist, method="ward")

    # Flat clusters at k=2,3,4
    clusters = {}
    for k in (2, 3, 4):
        assignments = fcluster(Z, k, criterion="maxclust")
        clusters[f"k{k}"] = {
            label: int(cl) for label, cl in zip(labels, assignments)
        }

    # Build linkage list for serialisation
    linkage_list = [
        {"step": i, "a": int(row[0]), "b": int(row[1]),
         "distance": round(float(row[2]), 4), "size": int(row[3])}
        for i, row in enumerate(Z)
    ]

    return {
        "labels":       labels,
        "linkage":      linkage_list,
        "clusters":     clusters,
        "dist_matrix":  squareform(dist).tolist(),
    }


# =====================================================================
# Load match_rate from DB
# =====================================================================

def load_match_rates(db_path: Path) -> dict:
    """Load per-hand match rates from hand_zscore table."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT hand, match_rate FROM hand_zscore")
        result = {row[0]: {"match_rate": row[1]} for row in cur.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(kl: dict, bigrams: dict, clustering: dict,
               db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 5a — KL matrix
    cur.execute("DROP TABLE IF EXISTS hand_clustering_kl")
    cur.execute("""
        CREATE TABLE hand_clustering_kl (
            pair       TEXT PRIMARY KEY,
            unit_a     TEXT,
            unit_b     TEXT,
            kl_obs     REAL,
            null_mean  REAL,
            null_std   REAL,
            z_kl       REAL
        )
    """)
    for pair, d in sorted(kl.items()):
        cur.execute("INSERT INTO hand_clustering_kl VALUES (?,?,?,?,?,?,?)", (
            pair, d["unit_a"], d["unit_b"],
            d["kl_obs"], d["null_mean"], d["null_std"], d["z_kl"],
        ))

    # 5b — bigram pairwise
    cur.execute("DROP TABLE IF EXISTS hand_clustering_bigrams")
    cur.execute("""
        CREATE TABLE hand_clustering_bigrams (
            pair              TEXT PRIMARY KEY,
            chi2              REAL,
            df                INTEGER,
            p_value           REAL,
            p_bh              REAL,
            significant_bh_05 INTEGER
        )
    """)
    for pair, d in sorted(bigrams.items()):
        if d.get("skipped"):
            continue
        cur.execute("INSERT INTO hand_clustering_bigrams VALUES (?,?,?,?,?,?)", (
            pair,
            d.get("chi2"), d.get("df"), d.get("p_value"),
            d.get("p_bh"), int(d.get("significant_bh_05", False)),
        ))

    # 5c — cluster assignments
    cur.execute("DROP TABLE IF EXISTS hand_clustering_dendrogram")
    cur.execute("""
        CREATE TABLE hand_clustering_dendrogram (
            unit       TEXT PRIMARY KEY,
            unit_name  TEXT,
            cluster_k2 INTEGER,
            cluster_k3 INTEGER,
            cluster_k4 INTEGER
        )
    """)
    labels = clustering["labels"]
    for label in labels:
        cur.execute("INSERT INTO hand_clustering_dendrogram VALUES (?,?,?,?,?)", (
            label, unit_label_name(label),
            clustering["clusters"]["k2"].get(label),
            clustering["clusters"]["k3"].get(label),
            clustering["clusters"]["k4"].get(label),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(kl: dict, bigrams: dict, clustering: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 5 — Clustering across all analysis units")
    lines.append("=" * 80)

    labels = clustering["labels"]

    # 5c — Cluster assignments (most informative — print first)
    lines.append("\n── Phase 5c — Hierarchical clustering (Ward linkage) ──")
    lines.append(
        f"  {'Unit':>6}  {'Name':>20}  {'k=2':>4}  {'k=3':>4}  {'k=4':>4}"
    )
    lines.append("  " + "-" * 44)
    for label in labels:
        name = unit_label_name(label)[:20]
        k2 = clustering["clusters"]["k2"].get(label, "?")
        k3 = clustering["clusters"]["k3"].get(label, "?")
        k4 = clustering["clusters"]["k4"].get(label, "?")
        lines.append(f"  {label:>6}  {name:>20}  {k2:>4}  {k3:>4}  {k4:>4}")

    # Key question: ?C with Hand 3?
    c3 = clustering["clusters"]["k3"]
    qc_cluster = c3.get("?C")
    h3_cluster  = c3.get("3")
    match = "YES ← ?C clusters with Hand 3" if qc_cluster == h3_cluster else "no"
    lines.append(f"\n  KEY QUESTION — ?C clusters with Hand 3 (k=3): {match}")
    lines.append(f"    ?C → cluster {qc_cluster}   Hand 3 → cluster {h3_cluster}")

    # 5a — KL top-10 most similar pairs
    lines.append("\n── Phase 5a — KL-divergence (10 most similar pairs) ──")
    lines.append(f"  {'Pair':>8}  {'KL_obs':>7}  {'KL_null':>7}  {'z_KL':>7}")
    lines.append("  " + "-" * 38)
    sorted_kl = sorted(kl.items(), key=lambda x: x[1]["kl_obs"])
    for pair, d in sorted_kl[:10]:
        z_str = f"{d['z_kl']:+.2f}" if d["z_kl"] is not None else "  n/a"
        lines.append(
            f"  {pair:>8}  {d['kl_obs']:>7.4f}  {d['null_mean']:>7.4f}  {z_str:>7}"
        )

    # 5b — Bigram pairs NOT rejected (most similar)
    lines.append("\n── Phase 5b — Bigram chi-square: pairs NOT rejected at BH 5% ──")
    non_rejected = [
        (pair, d) for pair, d in sorted(bigrams.items())
        if not d.get("skipped") and not d.get("significant_bh_05", True)
    ]
    if non_rejected:
        lines.append(
            f"  {'Pair':>8}  {'chi2':>8}  {'p_raw':>9}  {'p_BH':>9}"
        )
        for pair, d in sorted(non_rejected, key=lambda x: x[1]["p_value"],
                               reverse=True)[:15]:
            lines.append(
                f"  {pair:>8}  {d['chi2']:>8.1f}  "
                f"{d['p_value']:>9.6f}  {d.get('p_bh', 0):>9.6f}"
            )
    else:
        lines.append("  All pairs significantly different (p_BH < 0.05).")

    lines.append("\n── Legend ──")
    lines.append("  KL z < 0: units more similar than random")
    lines.append("  BH correction: Benjamini-Hochberg FDR at 5%")
    lines.append("  Clustering features: entropy, zipf, ttr, word_len, trigram_H, match_rate")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 5: clustering across ~14 analysis units (hand ? split by section)."""
    report_path = config.stats_dir / "hand_clustering.json"
    summary_path = config.stats_dir / "hand_clustering_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_clustering report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 5 — Clustering across all analysis units")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    all_words = [w for p in pages for w in p["words"]]
    click.echo(f"    {len(all_words):,} words, {len(pages)} pages")

    # 2. Build analysis units
    print_step("Building ~14 analysis units (hand ? split by section)...")
    corpus = split_corpus_by_hand(pages)
    units = build_analysis_units(corpus, pages)
    for label in sorted(units.keys()):
        click.echo(f"    {label:>4} ({unit_label_name(label)}): "
                   f"{len(units[label]):,} words")
    click.echo(f"    Total units: {len(units)}")

    # 3. Phase 5a — KL matrix
    print_step("Phase 5a — KL-divergence matrix (100 null samples per pair)...")
    kl = kl_matrix(units, all_words)

    # 4. Phase 5b — Bigram pairwise chi-square
    print_step("Phase 5b — Pairwise bigram chi-square + BH correction...")
    bigrams = bigram_matrix(units)
    n_pairs = len([d for d in bigrams.values() if not d.get("skipped")])
    n_rejected = sum(1 for d in bigrams.values()
                     if not d.get("skipped") and d.get("significant_bh_05"))
    click.echo(f"    {n_pairs} pairs tested, {n_rejected} rejected at BH 5%")
    n_similar = n_pairs - n_rejected
    click.echo(f"    {n_similar} pairs NOT rejected (similar bigram distribution)")

    # 5. Phase 5c — Hierarchical clustering
    print_step("Phase 5c — Hierarchical clustering (Ward linkage)...")
    db_path = config.output_dir.parent / "voynich.db"
    zscore_data = load_match_rates(db_path)
    feat_labels, X = build_feature_matrix(units, zscore_data)
    clustering = hierarchical_clustering(feat_labels, X)

    click.echo(f"    k=2 clusters: "
               + str({k: [l for l in feat_labels
                           if clustering['clusters']['k2'][l] == k]
                      for k in (1, 2)}))
    click.echo(f"    k=3 clusters: "
               + str({k: [l for l in feat_labels
                           if clustering['clusters']['k3'][l] == k]
                      for k in (1, 2, 3)}))

    # Key question
    c3 = clustering["clusters"]["k3"]
    qc, h3 = c3.get("?C"), c3.get("3")
    match_str = "YES" if qc == h3 else "NO"
    click.echo(f"    KEY: ?C cluster={qc}, Hand 3 cluster={h3} → same cluster: {match_str}")

    # 6. Save JSON
    print_step("Saving JSON...")
    report = {
        "n_units":    len(units),
        "unit_sizes": {k: len(v) for k, v in sorted(units.items())},
        "kl":         kl,
        "bigrams":    bigrams,
        "clustering": clustering,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 7. Save TXT
    summary = format_summary(kl, bigrams, clustering)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 8. Save to DB
    print_step("Writing DB tables hand_clustering_*...")
    if db_path.exists():
        save_to_db(kl, bigrams, clustering, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
