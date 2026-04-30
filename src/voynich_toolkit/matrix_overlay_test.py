"""
Matrix Overlay Test — Phase 18.

Tests the hypothesis (proposed by Antenore Gatta, 2026-04-30) that the
Voynich Manuscript was produced using a physical matrix/stencil or by
overlaying multiple pages. Three sub-tests:

  18a — Position-locked recurrence:
        If a fixed-position matrix was used, the same word should appear
        at the same line position across pages more than chance allows.

  18b — Page-pair completion:
        If pages were meant to overlay (one-time pad style), some pairs of
        pages should show statistically complementary signatures.

  18c — Hand ? × position anomaly:
        Hand ? is anomalous in all 7 sections (Phase 2e). If Hand ? marks
        matrix-error pages, its positional structure should differ from
        other hands.

None of these test the matrix DIRECTLY (would need bbox data), but each
tests a downstream consequence of the matrix/overlay hypotheses.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_PERM = 500
TOP_N_WORDS = 30  # we test position-locking on the most frequent words


# =====================================================================
# 18a — Position-locked recurrence
# =====================================================================

def measure_position_locking(pages: list[dict], top_n: int = TOP_N_WORDS) -> dict:
    """Test whether top-N words appear at fixed positions across pages.

    For each of the top-N most frequent words:
      - Count how often it appears at line position p, for p = 0, 1, 2, ..., max_pos
      - Compute Shannon entropy of its position distribution
      - Compare to permutation null where words are shuffled within lines

    Low entropy + low null-overlap ⇒ position-locked (suggests matrix).
    High entropy ⇒ word floats freely (no matrix).
    """
    rng = np.random.default_rng(SEED)

    # Collect all (word, line_position) observations
    word_positions: dict[str, list[int]] = defaultdict(list)
    line_positions_global: list[tuple[str, int]] = []

    for page in pages:
        for line in page.get("line_words", []):
            for pos, word in enumerate(line):
                word_positions[word].append(pos)
                line_positions_global.append((word, pos))

    # Get top-N words
    counts = Counter(w for line in [page.get("line_words", []) for page in pages]
                     for words in line for w in words)
    counts = Counter()
    for page in pages:
        for line in page.get("line_words", []):
            for w in line:
                counts[w] += 1
    top_words = [w for w, _ in counts.most_common(top_n)]

    def _entropy(positions: list[int]) -> float:
        if not positions:
            return 0.0
        c = Counter(positions)
        total = sum(c.values())
        h = 0.0
        for n in c.values():
            p = n / total
            if p > 0:
                h -= p * np.log2(p)
        return h

    # Real entropy per top word
    real_entropies = {w: _entropy(word_positions[w]) for w in top_words}

    # Null: shuffle word-position pairs within each line
    null_entropies_per_word: dict[str, list[float]] = {w: [] for w in top_words}
    for _ in range(N_PERM):
        # Shuffle words within each line
        shuffled_word_positions: dict[str, list[int]] = defaultdict(list)
        for page in pages:
            for line in page.get("line_words", []):
                if not line:
                    continue
                shuffled = list(line)
                rng.shuffle(shuffled)
                for pos, word in enumerate(shuffled):
                    shuffled_word_positions[word].append(pos)
        for w in top_words:
            null_entropies_per_word[w].append(_entropy(shuffled_word_positions[w]))

    # Z-score per word: how much LOWER is real entropy than null (negative z = locked)
    z_scores = {}
    for w in top_words:
        nulls = null_entropies_per_word[w]
        mean_null = float(np.mean(nulls))
        std_null = float(np.std(nulls))
        if std_null > 0:
            z = (real_entropies[w] - mean_null) / std_null
        else:
            z = 0.0
        z_scores[w] = {
            "count": counts[w],
            "real_entropy": round(real_entropies[w], 3),
            "null_entropy_mean": round(mean_null, 3),
            "null_entropy_std": round(std_null, 3),
            "z_score": round(z, 2),
            "locked": z < -2,
        }

    n_locked = sum(1 for d in z_scores.values() if d["locked"])

    # For locked words, find their preferred position
    for w in top_words:
        positions = word_positions[w]
        if not positions:
            continue
        c = Counter(positions)
        total = sum(c.values())
        # Most common position and its share
        most_common_pos, most_common_count = c.most_common(1)[0]
        z_scores[w]["preferred_position"] = most_common_pos
        z_scores[w]["preferred_position_share"] = round(most_common_count / total, 3)
        # Also: is it preferentially line-initial (pos 0), line-final (pos = last)?
        # Compute fraction in first 2 positions vs last 2 positions
        max_pos_seen = max(positions)
        first_2 = sum(1 for p in positions if p <= 1) / total
        # "line-final" is harder without knowing each line's length, approximated
        z_scores[w]["fraction_first_2_positions"] = round(first_2, 3)

    # Summarize: among locked words, where do they cluster?
    locked_words = [w for w in top_words if z_scores[w]["locked"]]
    if locked_words:
        preferred_positions = [z_scores[w]["preferred_position"] for w in locked_words]
        position_summary = Counter(preferred_positions)
    else:
        position_summary = Counter()

    return {
        "top_words_tested": top_n,
        "n_position_locked": n_locked,
        "expected_under_null_alpha_05": round(top_n * 0.025, 1),
        "locked_words_position_summary": dict(position_summary),
        "per_word": z_scores,
    }


# =====================================================================
# 18b — Page-pair completion
# =====================================================================

def measure_page_pair_completion(pages: list[dict]) -> dict:
    """Test if some pages have statistically complementary signatures.

    For each pair (p_i, p_j) of pages:
      - Compute character-distribution distance (KL divergence both ways)
      - Compute combined-distribution entropy vs singleton entropies
      - If overlay reduces effective entropy disproportionately,
        the pair is "completing"

    Output: distribution of pair-entropy reductions, top complementary pairs.
    """
    pages_with_text = [p for p in pages if p.get("words")]
    n_pages = len(pages_with_text)

    if n_pages < 10:
        return {"error": "too few pages with text"}

    # Per-page character distribution
    page_char_dists = []
    for page in pages_with_text:
        text = " ".join(page["words"])
        c = Counter(text)
        total = sum(c.values())
        if total == 0:
            page_char_dists.append({})
            continue
        page_char_dists.append({ch: n / total for ch, n in c.items()})

    def _entropy(dist: dict[str, float]) -> float:
        h = 0.0
        for p in dist.values():
            if p > 0:
                h -= p * np.log2(p)
        return h

    def _combine(d1: dict, d2: dict) -> dict:
        keys = set(d1) | set(d2)
        return {k: (d1.get(k, 0) + d2.get(k, 0)) / 2 for k in keys}

    # Singleton entropies
    page_entropies = [_entropy(d) for d in page_char_dists]
    mean_singleton_entropy = float(np.mean(page_entropies))

    # Sample pairs (full N×N is too expensive: 225×225=50K pairs OK actually)
    pair_reductions = []
    pair_info = []
    rng = np.random.default_rng(SEED)
    n_sample = min(2000, n_pages * (n_pages - 1) // 2)

    sampled_pairs = set()
    attempts = 0
    while len(pair_reductions) < n_sample and attempts < n_sample * 3:
        i = int(rng.integers(0, n_pages))
        j = int(rng.integers(0, n_pages))
        if i == j or (i, j) in sampled_pairs or (j, i) in sampled_pairs:
            attempts += 1
            continue
        sampled_pairs.add((i, j))
        attempts += 1

        d_combined = _combine(page_char_dists[i], page_char_dists[j])
        h_combined = _entropy(d_combined)
        h_avg = (page_entropies[i] + page_entropies[j]) / 2
        # If overlay is meaningful, combined entropy should DROP
        # (the merged distribution becomes more peaked when pages complete each other)
        reduction = h_avg - h_combined
        pair_reductions.append(reduction)
        pair_info.append({
            "i_folio": pages_with_text[i]["folio"],
            "j_folio": pages_with_text[j]["folio"],
            "reduction": reduction,
        })

    arr = np.array(pair_reductions)
    pair_info_sorted = sorted(pair_info, key=lambda d: -d["reduction"])

    return {
        "n_pages_with_text": n_pages,
        "mean_singleton_entropy": round(mean_singleton_entropy, 3),
        "n_pairs_sampled": len(pair_reductions),
        "mean_pair_reduction": round(float(arr.mean()), 4),
        "max_pair_reduction": round(float(arr.max()), 4),
        "pairs_with_positive_reduction_pct": round(
            float((arr > 0).mean() * 100), 1
        ),
        "pairs_with_strong_reduction_pct": round(
            float((arr > 0.1).mean() * 100), 1
        ),
        "top_5_completing_pairs": pair_info_sorted[:5],
        "interpretation": (
            "If overlay hypothesis is correct, we'd expect MANY pairs with "
            "strong positive reduction (combined entropy << average singleton). "
            "If reduction distribution is centered near 0, pages are independent."
        ),
    }


# =====================================================================
# 18c — Hand ? × position anomaly
# =====================================================================

def measure_hand_position_anomaly(pages: list[dict]) -> dict:
    """Compare position-distribution structure of Hand ? vs other hands.

    For each hand:
      - Compute the entropy of word-position distribution (line-internal)
      - Compute the variance of word-line-position distribution
    Hand ? should NOT differ if it's just a different scribe.
    Hand ? SHOULD differ if it represents matrix-error pages.
    """
    hand_word_positions: dict[str, list[int]] = defaultdict(list)
    hand_line_lengths: dict[str, list[int]] = defaultdict(list)

    for page in pages:
        hand = page.get("hand", "?")
        for line in page.get("line_words", []):
            hand_line_lengths[hand].append(len(line))
            for pos, word in enumerate(line):
                hand_word_positions[hand].append(pos)

    def _stats(positions: list[int]) -> dict:
        if not positions:
            return {"mean": 0, "std": 0, "n": 0}
        arr = np.array(positions)
        return {
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "n": len(positions),
        }

    per_hand = {}
    for hand in sorted(hand_word_positions):
        per_hand[hand] = {
            "position_stats": _stats(hand_word_positions[hand]),
            "line_length_stats": _stats(hand_line_lengths[hand]),
        }

    # Anomaly: does Hand ? have a systematically different distribution?
    if "?" in per_hand:
        unknown_pos_mean = per_hand["?"]["position_stats"]["mean"]
        unknown_line_mean = per_hand["?"]["line_length_stats"]["mean"]
        other_pos_means = [
            d["position_stats"]["mean"] for h, d in per_hand.items() if h != "?"
        ]
        other_line_means = [
            d["line_length_stats"]["mean"] for h, d in per_hand.items() if h != "?"
        ]
        if other_pos_means:
            pos_z = (unknown_pos_mean - np.mean(other_pos_means)) / (
                np.std(other_pos_means) + 1e-9
            )
            line_z = (unknown_line_mean - np.mean(other_line_means)) / (
                np.std(other_line_means) + 1e-9
            )
        else:
            pos_z, line_z = 0, 0

        anomaly = {
            "unknown_hand_position_z": round(float(pos_z), 2),
            "unknown_hand_line_length_z": round(float(line_z), 2),
            "anomalous": abs(pos_z) > 2 or abs(line_z) > 2,
        }
    else:
        anomaly = {"note": "no Hand ? data"}

    return {
        "per_hand": per_hand,
        "hand_unknown_anomaly": anomaly,
    }


# =====================================================================
# Reporting
# =====================================================================

def format_summary(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 18 — MATRIX OVERLAY TEST")
    lines.append("Tests for stencil/matrix or page-overlay hypotheses")
    lines.append("=" * 78)
    lines.append("")

    # 18a
    lines.append("--- 18a: Position-locked recurrence ---")
    a = report["position_locking"]
    lines.append(f"  Top-{a['top_words_tested']} words tested")
    lines.append(f"  Position-locked (z < -2): {a['n_position_locked']}/{a['top_words_tested']}")
    lines.append(f"  Expected by chance (alpha=0.025): "
                 f"~{a['expected_under_null_alpha_05']}")
    lines.append("")
    lines.append("  Top 10 most position-locked words (with preferred position):")
    sorted_words = sorted(a["per_word"].items(), key=lambda x: x[1]["z_score"])
    for w, d in sorted_words[:10]:
        verdict = "LOCKED" if d["locked"] else ""
        pref_pos = d.get("preferred_position", "-")
        pref_share = d.get("preferred_position_share", 0)
        first_2 = d.get("fraction_first_2_positions", 0)
        lines.append(f"    {w:<10s} count={d['count']:>4d} "
                     f"z={d['z_score']:>+6.2f} "
                     f"pref_pos={pref_pos} ({pref_share:.0%}) "
                     f"first2={first_2:.0%} {verdict}")
    lines.append("")
    pos_summary = a.get("locked_words_position_summary", {})
    if pos_summary:
        lines.append(f"  Preferred-position distribution among LOCKED words:")
        for pos in sorted(pos_summary):
            lines.append(f"    position {pos}: {pos_summary[pos]} word(s)")
        lines.append("")

    if a["n_position_locked"] > a["expected_under_null_alpha_05"] * 2:
        lines.append("  ⇒ SIGNAL: more position-locked words than chance.")
        lines.append("    Compatible with matrix hypothesis.")
    else:
        lines.append("  ⇒ NULL: position-locking at or near chance level.")
        lines.append("    No matrix evidence from this test.")
    lines.append("")

    # 18b
    lines.append("--- 18b: Page-pair completion (overlay hypothesis) ---")
    b = report["page_pair_completion"]
    if "error" in b:
        lines.append(f"  ERROR: {b['error']}")
    else:
        lines.append(f"  Pages with text: {b['n_pages_with_text']}")
        lines.append(f"  Pairs sampled: {b['n_pairs_sampled']:,}")
        lines.append(f"  Mean singleton entropy: {b['mean_singleton_entropy']:.3f}")
        lines.append(f"  Mean pair entropy reduction: {b['mean_pair_reduction']:+.4f}")
        lines.append(f"  Max pair reduction: {b['max_pair_reduction']:+.4f}")
        lines.append(f"  % pairs with positive reduction: {b['pairs_with_positive_reduction_pct']}%")
        lines.append(f"  % pairs with strong reduction (>0.1): {b['pairs_with_strong_reduction_pct']}%")
        lines.append("")
        lines.append("  Top 5 'completing' pairs:")
        for p in b["top_5_completing_pairs"]:
            lines.append(f"    {p['i_folio']} + {p['j_folio']} → "
                         f"reduction={p['reduction']:+.4f}")
        lines.append("")
        if b["pairs_with_strong_reduction_pct"] > 5:
            lines.append("  ⇒ SIGNAL: many pairs combine to lower entropy.")
            lines.append("    Compatible with overlay hypothesis.")
        else:
            lines.append("  ⇒ NULL: pages do not consistently complete each other.")
            lines.append("    Overlay hypothesis not supported.")
    lines.append("")

    # 18c
    lines.append("--- 18c: Hand ? × position anomaly ---")
    c = report["hand_position_anomaly"]
    lines.append("  Per-hand position statistics:")
    for hand, d in sorted(c["per_hand"].items()):
        ps = d["position_stats"]
        ls = d["line_length_stats"]
        lines.append(f"    Hand {hand}: pos_mean={ps['mean']:.2f} pos_std={ps['std']:.2f} "
                     f"line_len_mean={ls['mean']:.2f} (n={ps['n']})")
    lines.append("")
    a_anom = c["hand_unknown_anomaly"]
    if "note" in a_anom:
        lines.append(f"  {a_anom['note']}")
    else:
        lines.append(f"  Hand ? position-mean z-score:    {a_anom['unknown_hand_position_z']:+.2f}")
        lines.append(f"  Hand ? line-length-mean z-score: {a_anom['unknown_hand_line_length_z']:+.2f}")
        if a_anom["anomalous"]:
            lines.append(f"  ⇒ ANOMALOUS positional structure for Hand ?.")
            lines.append(f"    Compatible with matrix-error interpretation.")
        else:
            lines.append(f"  ⇒ Hand ? positional structure within normal range.")
            lines.append(f"    Hand ? is a different scribe, not matrix-error.")
    lines.append("")

    # Summary
    lines.append("=" * 78)
    lines.append("OVERALL VERDICT")
    lines.append("=" * 78)

    n_locked = a.get("n_position_locked", 0)
    expected = a.get("expected_under_null_alpha_05", 1)
    pair_strong_pct = b.get("pairs_with_strong_reduction_pct", 0)
    hand_anom = c.get("hand_unknown_anomaly", {}).get("anomalous", False)

    matrix_signal = n_locked > expected * 2
    overlay_signal = pair_strong_pct > 5
    hand_signal = hand_anom

    if matrix_signal and not overlay_signal:
        lines.append("STRONGEST SIGNAL: position-locked vocabulary (Theory 1, matrix).")
    elif overlay_signal and not matrix_signal:
        lines.append("STRONGEST SIGNAL: page-pair completion (Theory 2, overlay).")
    elif matrix_signal and overlay_signal:
        lines.append("BOTH SIGNALS PRESENT: matrix and overlay both supported.")
    else:
        lines.append("NO SIGNAL: neither matrix nor overlay hypothesis supported")
        lines.append("by these statistical proxies. Direct paleographic evidence needed.")
    lines.append("")
    if hand_signal:
        lines.append("Additional: Hand ? shows positional anomaly, possibly matrix-error.")
    lines.append("")
    lines.append("Caveat: these tests proxy spatial structure with line position only.")
    lines.append("True matrix tests require glyph-level (x,y) bbox data we do not have.")
    lines.append("")

    return "\n".join(lines) + "\n"


def save_to_db(config: ToolkitConfig, report: dict):
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS matrix_overlay_test")
    cur.execute("""
        CREATE TABLE matrix_overlay_test (
            test TEXT,
            metric TEXT,
            value REAL,
            note TEXT,
            PRIMARY KEY (test, metric)
        )
    """)
    rows = []

    a = report["position_locking"]
    rows.append(("18a", "n_position_locked", float(a["n_position_locked"]),
                 "Top words with z < -2 (locked position)"))
    rows.append(("18a", "expected_under_null", float(a["expected_under_null_alpha_05"]),
                 "Expected by chance"))

    b = report["page_pair_completion"]
    if "error" not in b:
        rows.append(("18b", "mean_pair_reduction", float(b["mean_pair_reduction"]),
                     "Mean entropy reduction across page pairs"))
        rows.append(("18b", "pairs_with_strong_reduction_pct",
                     float(b["pairs_with_strong_reduction_pct"]),
                     "% pairs with reduction > 0.1"))

    c = report["hand_position_anomaly"]
    a_anom = c.get("hand_unknown_anomaly", {})
    if "unknown_hand_position_z" in a_anom:
        rows.append(("18c", "unknown_hand_position_z",
                     float(a_anom["unknown_hand_position_z"]),
                     "Hand ? position-mean z-score"))
        rows.append(("18c", "unknown_hand_line_length_z",
                     float(a_anom["unknown_hand_line_length_z"]),
                     "Hand ? line-length-mean z-score"))

    cur.executemany("INSERT INTO matrix_overlay_test VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 18: Matrix/Overlay hypothesis test."""
    report_path = config.stats_dir / "matrix_overlay_test.json"
    summary_path = config.stats_dir / "matrix_overlay_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Matrix overlay test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 18 — Matrix/Overlay Hypothesis Test")

    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    print_step(f"18a: Position-locked recurrence (top-{TOP_N_WORDS}, "
               f"{N_PERM} permutations)...")
    pos_lock = measure_position_locking(real_pages, top_n=TOP_N_WORDS)
    click.echo(f"    Position-locked words: {pos_lock['n_position_locked']}/"
               f"{pos_lock['top_words_tested']} "
               f"(expected by chance: ~{pos_lock['expected_under_null_alpha_05']})")

    print_step("18b: Page-pair completion (overlay hypothesis)...")
    pair_comp = measure_page_pair_completion(real_pages)
    if "error" not in pair_comp:
        click.echo(f"    Mean pair entropy reduction: "
                   f"{pair_comp['mean_pair_reduction']:+.4f}")
        click.echo(f"    Pairs with strong reduction (>0.1): "
                   f"{pair_comp['pairs_with_strong_reduction_pct']}%")

    print_step("18c: Hand ? × position anomaly...")
    hand_anom = measure_hand_position_anomaly(real_pages)
    a_anom = hand_anom.get("hand_unknown_anomaly", {})
    if "unknown_hand_position_z" in a_anom:
        click.echo(f"    Hand ? position-mean z: {a_anom['unknown_hand_position_z']:+.2f}")
        click.echo(f"    Hand ? line-length-mean z: {a_anom['unknown_hand_line_length_z']:+.2f}")

    print_step("Saving results...")
    report = {
        "position_locking": pos_lock,
        "page_pair_completion": pair_comp,
        "hand_position_anomaly": hand_anom,
        "parameters": {
            "n_perm": N_PERM,
            "top_n_words": TOP_N_WORDS,
            "seed": SEED,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(report)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT:  {summary_path}")

    save_to_db(config, report)
    click.echo(f"    DB:   matrix_overlay_test table")

    click.echo(f"\n{summary}")
