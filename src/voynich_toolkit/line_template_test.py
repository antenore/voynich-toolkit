"""
Line Template Test — Phase 18d.

Extension of Phase 18a (which found that 18/30 top words are position-locked
within lines). This phase maps the FULL line-position template:

  - For each position (0, 1, 2, ..., last), characterize the word class
  - Compute Shannon entropy at each position vs null
  - Cluster words at each position to find shared morphology
  - Test section/hand dependence of the template

If the manuscript has a tabular line schema (categorical register
hypothesis), each line position should have a coherent word class, and
the template should be consistent across at least some scribes/sections.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .utils import print_header, print_step
from .word_structure import parse_eva_words

SEED = 42
N_PERM = 300
MAX_POSITION = 12  # only analyze positions 0..MAX-1
MIN_OCCURRENCES_PER_POS = 50  # require this many words at a position to test it


# =====================================================================
# Position-class characterization
# =====================================================================

def words_at_each_position(
    pages: list[dict],
    max_position: int = MAX_POSITION,
) -> dict[int, list[str]]:
    """Collect all words appearing at each line position."""
    by_position: dict[int, list[str]] = defaultdict(list)
    for page in pages:
        for line in page.get("line_words", []):
            for pos, word in enumerate(line):
                if pos < max_position:
                    by_position[pos].append(word)
    return by_position


def characterize_position(
    words_here: list[str],
    all_words: list[str],
    rng: np.random.Generator,
    n_perm: int,
) -> dict:
    """Statistical characterization of one line position.

    - Distribution entropy at this position
    - Top words and their over-representation z-score
    - Common prefix/suffix patterns
    - Length distribution
    """
    if not words_here:
        return {"n": 0}

    counts = Counter(words_here)
    total = len(words_here)
    n_types = len(counts)

    # Shannon entropy of word distribution at this position
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * np.log2(p)

    # Null: same number of draws but from the global word distribution
    global_counts = Counter(all_words)
    global_total = len(all_words)
    word_list = list(global_counts.keys())
    word_probs = np.array(
        [global_counts[w] / global_total for w in word_list],
        dtype=np.float64,
    )

    null_entropies = []
    for _ in range(n_perm):
        sampled_idx = rng.choice(len(word_list), size=total, p=word_probs)
        sampled = [word_list[i] for i in sampled_idx]
        sc = Counter(sampled)
        h = 0.0
        for c in sc.values():
            p = c / total
            if p > 0:
                h -= p * np.log2(p)
        null_entropies.append(h)

    null_mean = float(np.mean(null_entropies))
    null_std = float(np.std(null_entropies))
    z_entropy = (entropy - null_mean) / null_std if null_std > 0 else 0

    # Top 10 words at this position with their over-representation
    top_words_data = []
    for w, c in counts.most_common(10):
        global_p = global_counts[w] / global_total
        local_p = c / total
        # ratio of local to global probability (1 = same as global)
        # higher = over-represented at this position
        ratio = local_p / global_p if global_p > 0 else 0
        top_words_data.append({
            "word": w,
            "count": c,
            "local_pct": round(local_p * 100, 2),
            "global_pct": round(global_p * 100, 2),
            "over_repr_ratio": round(ratio, 2),
        })

    # Prefix/suffix patterns: top 2-char prefixes and suffixes
    prefixes = Counter(w[:2] for w in words_here if len(w) >= 2)
    suffixes = Counter(w[-2:] for w in words_here if len(w) >= 2)
    top_prefixes = [(p, c) for p, c in prefixes.most_common(5)]
    top_suffixes = [(s, c) for s, c in suffixes.most_common(5)]

    # Length distribution
    lengths = [len(w) for w in words_here]
    return {
        "n": total,
        "n_types": n_types,
        "ttr": round(n_types / total, 4),
        "entropy_bits": round(entropy, 3),
        "null_entropy_mean": round(null_mean, 3),
        "z_entropy": round(z_entropy, 2),
        "more_concentrated_than_null": z_entropy < -2,
        "top_words": top_words_data,
        "top_prefixes": top_prefixes,
        "top_suffixes": top_suffixes,
        "length_mean": round(float(np.mean(lengths)), 2),
        "length_std": round(float(np.std(lengths)), 2),
    }


# =====================================================================
# Cross-section / cross-hand template consistency
# =====================================================================

def template_by_group(
    pages: list[dict],
    group_key: str,
    max_position: int,
) -> dict[str, dict[int, Counter]]:
    """Per-group (section or hand), collect word counts at each position."""
    out: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for page in pages:
        group = page.get(group_key, "?")
        for line in page.get("line_words", []):
            for pos, word in enumerate(line):
                if pos < max_position:
                    out[group][pos][word] += 1
    return out


def compute_template_overlap(
    template_a: dict[int, Counter],
    template_b: dict[int, Counter],
    max_position: int,
) -> dict[int, float]:
    """For each position, compute Jaccard overlap between top-20 words
    of two templates."""
    overlap = {}
    for pos in range(max_position):
        a_top = {w for w, _ in template_a.get(pos, Counter()).most_common(20)}
        b_top = {w for w, _ in template_b.get(pos, Counter()).most_common(20)}
        if not a_top and not b_top:
            overlap[pos] = 0.0
            continue
        inter = len(a_top & b_top)
        union = len(a_top | b_top)
        overlap[pos] = round(inter / union, 3) if union > 0 else 0
    return overlap


def measure_template_consistency(
    pages: list[dict],
    max_position: int = MAX_POSITION,
) -> dict:
    """How consistent is the line template across sections and hands?"""
    sections_template = template_by_group(pages, "section", max_position)
    hands_template = template_by_group(pages, "hand", max_position)

    # Section pairwise overlap
    section_pairs_overlap = {}
    sections_list = sorted(sections_template.keys())
    for i, sa in enumerate(sections_list):
        for sb in sections_list[i + 1:]:
            overlap = compute_template_overlap(
                sections_template[sa], sections_template[sb], max_position
            )
            section_pairs_overlap[f"{sa}↔{sb}"] = overlap

    # Hand pairwise overlap
    hand_pairs_overlap = {}
    hands_list = sorted(hands_template.keys())
    for i, ha in enumerate(hands_list):
        for hb in hands_list[i + 1:]:
            overlap = compute_template_overlap(
                hands_template[ha], hands_template[hb], max_position
            )
            hand_pairs_overlap[f"{ha}↔{hb}"] = overlap

    # Mean overlap per position (across all section pairs and all hand pairs)
    def _mean_per_pos(pairs_overlap: dict) -> dict[int, float]:
        per_pos: dict[int, list[float]] = defaultdict(list)
        for pair_data in pairs_overlap.values():
            for pos, val in pair_data.items():
                per_pos[pos].append(val)
        return {
            pos: round(float(np.mean(vals)), 3)
            for pos, vals in per_pos.items()
        }

    return {
        "section_mean_overlap_per_pos": _mean_per_pos(section_pairs_overlap),
        "hand_mean_overlap_per_pos": _mean_per_pos(hand_pairs_overlap),
        "section_pairs_overlap": {
            k: {str(p): v for p, v in vd.items()}
            for k, vd in list(section_pairs_overlap.items())[:10]
        },
        "hand_pairs_overlap": {
            k: {str(p): v for p, v in vd.items()}
            for k, vd in list(hand_pairs_overlap.items())[:10]
        },
    }


# =====================================================================
# Reporting
# =====================================================================

def format_summary(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 18d — LINE TEMPLATE MAPPING")
    lines.append("Characterize the word class at each line position;")
    lines.append("test if the template is consistent across sections and hands")
    lines.append("=" * 78)
    lines.append("")

    lines.append("--- Per-position characterization ---")
    lines.append(f"{'Pos':>3s} {'n':>6s} {'types':>6s} {'TTR':>6s} "
                 f"{'H':>6s} {'z_H':>7s} {'top words':<40s}")
    lines.append("-" * 78)
    for pos_str, char in sorted(report["per_position"].items(), key=lambda x: int(x[0])):
        if char.get("n", 0) < MIN_OCCURRENCES_PER_POS:
            continue
        top_str = ", ".join(
            f"{tw['word']}({tw['over_repr_ratio']:.1f}x)"
            for tw in char["top_words"][:4]
        )
        flag = " *" if char.get("more_concentrated_than_null") else ""
        lines.append(
            f"{pos_str:>3s} {char['n']:>6d} {char['n_types']:>6d} "
            f"{char['ttr']:>6.3f} {char['entropy_bits']:>6.2f} "
            f"{char['z_entropy']:>+7.2f}{flag} {top_str:<40s}"
        )
    lines.append("")
    lines.append("  Legend: * = position has more concentrated word distribution")
    lines.append("          than expected from global word frequency (z_H < -2)")
    lines.append("")

    # Per position prefix/suffix patterns
    lines.append("--- Top prefixes/suffixes per position (positions 0-6) ---")
    for pos in range(7):
        char = report["per_position"].get(str(pos), {})
        if char.get("n", 0) < MIN_OCCURRENCES_PER_POS:
            continue
        pfx = ", ".join(f"{p}({c})" for p, c in char.get("top_prefixes", []))
        sfx = ", ".join(f"{s}({c})" for s, c in char.get("top_suffixes", []))
        lines.append(f"  pos {pos}: prefix={pfx} | suffix={sfx}")
    lines.append("")

    # Cross-section/hand consistency
    cons = report["consistency"]
    lines.append("--- Template consistency across groups ---")
    lines.append("Mean Jaccard overlap (top-20 words) between SECTIONS, per position:")
    for pos in sorted(cons["section_mean_overlap_per_pos"].keys()):
        val = cons["section_mean_overlap_per_pos"][pos]
        lines.append(f"  pos {pos}: {val:.3f}")
    lines.append("")
    lines.append("Mean Jaccard overlap between HANDS, per position:")
    for pos in sorted(cons["hand_mean_overlap_per_pos"].keys()):
        val = cons["hand_mean_overlap_per_pos"][pos]
        lines.append(f"  pos {pos}: {val:.3f}")
    lines.append("")

    lines.append("=" * 78)
    lines.append("INTERPRETATION")
    lines.append("=" * 78)

    n_concentrated = sum(
        1 for char in report["per_position"].values()
        if char.get("more_concentrated_than_null")
    )
    n_tested = sum(
        1 for char in report["per_position"].values()
        if char.get("n", 0) >= MIN_OCCURRENCES_PER_POS
    )

    lines.append(
        f"Positions with concentrated word class (z_H < -2): "
        f"{n_concentrated}/{n_tested}"
    )
    lines.append("")

    sec_mean = float(np.mean(list(cons["section_mean_overlap_per_pos"].values())))
    hand_mean = float(np.mean(list(cons["hand_mean_overlap_per_pos"].values())))
    lines.append(f"Mean section overlap (avg across positions): {sec_mean:.3f}")
    lines.append(f"Mean hand overlap (avg across positions):    {hand_mean:.3f}")
    lines.append("")

    if n_concentrated >= n_tested * 0.5 and hand_mean > 0.3 and sec_mean < hand_mean:
        lines.append("STRONG SIGNAL FOR LINE TEMPLATE.")
        lines.append("Most positions have coherent word classes. Hands share")
        lines.append("the template more than sections do — sections customize")
        lines.append("the template's content but the schema is shared.")
        lines.append("")
        lines.append("This pattern matches a categorical register/formulary:")
        lines.append("  - shared schema across scribes (everyone uses the same template)")
        lines.append("  - section-specific vocabulary (different topic per section)")
        lines.append("  - position 1 specifically marks a recurring functional class")
    elif n_concentrated >= n_tested * 0.3:
        lines.append("MODERATE SIGNAL: line template is partial.")
        lines.append("Some positions have concentrated classes, others don't.")
    else:
        lines.append("WEAK SIGNAL: positions are not strongly templated.")
    lines.append("")

    return "\n".join(lines) + "\n"


def save_to_db(config: ToolkitConfig, report: dict):
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS line_template_test")
    cur.execute("""
        CREATE TABLE line_template_test (
            position INTEGER,
            n INTEGER,
            n_types INTEGER,
            entropy_bits REAL,
            z_entropy REAL,
            concentrated INTEGER,
            top_words TEXT,
            top_prefix TEXT,
            top_suffix TEXT,
            section_overlap REAL,
            hand_overlap REAL,
            PRIMARY KEY (position)
        )
    """)
    sec_per_pos = report["consistency"]["section_mean_overlap_per_pos"]
    hand_per_pos = report["consistency"]["hand_mean_overlap_per_pos"]
    rows = []
    for pos_str, char in report["per_position"].items():
        if char.get("n", 0) < MIN_OCCURRENCES_PER_POS:
            continue
        pos = int(pos_str)
        top_words_str = ",".join(
            tw["word"] for tw in char.get("top_words", [])[:5]
        )
        top_pfx = char.get("top_prefixes", [["", 0]])[0][0] if char.get("top_prefixes") else ""
        top_sfx = char.get("top_suffixes", [["", 0]])[0][0] if char.get("top_suffixes") else ""
        rows.append((
            pos, char["n"], char["n_types"], char["entropy_bits"],
            char["z_entropy"],
            1 if char.get("more_concentrated_than_null") else 0,
            top_words_str, top_pfx, top_sfx,
            sec_per_pos.get(pos, 0), hand_per_pos.get(pos, 0),
        ))
    cur.executemany(
        "INSERT INTO line_template_test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 18d: Map the full line-position template."""
    report_path = config.stats_dir / "line_template_test.json"
    summary_path = config.stats_dir / "line_template_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Line template test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 18d — Line Template Mapping")

    print_step("Parsing real EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    real_pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(real_pages)} pages")

    print_step("Collecting words at each line position...")
    by_pos = words_at_each_position(real_pages, max_position=MAX_POSITION)
    for pos in sorted(by_pos):
        click.echo(f"    pos {pos}: {len(by_pos[pos]):,} occurrences")

    print_step(f"Characterizing each position ({N_PERM} permutations)...")
    rng = np.random.default_rng(SEED)
    all_words = eva_data["words"]
    per_position = {}
    for pos in sorted(by_pos):
        char = characterize_position(
            by_pos[pos], all_words, rng, n_perm=N_PERM
        )
        per_position[str(pos)] = char
        if char.get("n", 0) >= MIN_OCCURRENCES_PER_POS:
            top_str = ",".join(
                tw["word"] for tw in char.get("top_words", [])[:3]
            )
            click.echo(
                f"    pos {pos}: H={char['entropy_bits']:.2f} "
                f"z={char['z_entropy']:+.2f} top={top_str}"
            )

    print_step("Measuring template consistency across sections/hands...")
    consistency = measure_template_consistency(real_pages, MAX_POSITION)

    print_step("Saving results...")
    report = {
        "per_position": per_position,
        "consistency": consistency,
        "parameters": {
            "max_position": MAX_POSITION,
            "n_perm": N_PERM,
            "min_occurrences_per_pos": MIN_OCCURRENCES_PER_POS,
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
    click.echo(f"    DB:   line_template_test table")

    click.echo(f"\n{summary}")
