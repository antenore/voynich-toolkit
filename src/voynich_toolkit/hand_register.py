"""
Phase 6b — Per-hand register role analysis.

Motivation: Phase 6 found that sections B, H, S, P have register/inventory
structure. This phase tests whether different hands play different roles
within those register sections.

Hypothesis: certain hands predominantly write "label-type" tokens (unique
line-initial words), while others predominantly write "tally-type" tokens
(repeated continuation words).

Three tests:
  6b-1 — Per-hand line-initial uniqueness rate
         Which hands write more unique first words?
  6b-2 — Tally token consistency across hands
         Are the same 2-3 most repeated tokens shared across all hands
         in high-register sections? If yes → shared convention.
  6b-3 — Cross-analysis: register score × hand
         Do hands that clustered together in Phase 5 also share similar
         register roles?

IMPORTANT LIMITATION: $H in IVTFF is per-folio, NOT per-line or per-word.
We cannot know if different hands wrote different parts of the same line.
All inferences are at PAGE level: a hand "writes" all lines on its pages.

TERMINOLOGY: do NOT call tokens "articles" or "quantities".
  - "label-type tokens" = line-initial, unique
  - "tally-type tokens" = continuation, repeated
  What they represent in reality is unknown.

Output:
  hand_register.json
  hand_register_summary.txt
  DB table: hand_register
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .scribe_analysis import HAND_NAMES, split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SEED = 42
N_NULL = 500

# Sections with strong register signal from Phase 6
REGISTER_SECTIONS = {"B", "H", "S", "P"}


# =====================================================================
# Data preparation: collect lines per hand (within register sections)
# =====================================================================

def lines_by_hand(corpus: dict,
                  sections: set[str] | None = None) -> dict[str, list[list[str]]]:
    """Group manuscript lines by hand, optionally filtering by section.

    Returns: dict[hand] → list of lines (each line = list[str] of words).
    Only includes lines with >= 2 words.
    """
    by_hand: dict[str, list[list[str]]] = {}
    for hand, data in corpus.items():
        hand_lines: list[list[str]] = []
        for page in data["pages"]:
            sec = page.get("section", "?")
            if sections is not None and sec not in sections:
                continue
            for line in page.get("line_words", []):
                if len(line) >= 2:
                    hand_lines.append(line)
        if hand_lines:
            by_hand[hand] = hand_lines
    return by_hand


# =====================================================================
# 6b-1 — Per-hand line-initial uniqueness
# =====================================================================

def hand_initial_uniqueness(lines: list[list[str]]) -> dict:
    """Compute line-initial uniqueness for one hand's lines.

    Returns: dict with obs, n_lines, top5_initials
    """
    initials = [line[0] for line in lines]
    freq = Counter(initials)
    n_unique = sum(1 for c in freq.values() if c == 1)
    rate = n_unique / len(initials) if initials else 0.0
    return {
        "uniqueness": round(rate, 4),
        "n_lines": len(lines),
        "n_unique_initials": n_unique,
        "n_distinct_initials": len(freq),
        "top5_initials": [{"word": w, "count": c}
                          for w, c in freq.most_common(5)],
    }


def null_hand_uniqueness(lines: list[list[str]], all_lines: list[list[str]],
                          n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: sample len(lines) lines from all register-section lines,
    compute uniqueness. Tests: 'is this hand's initial-word diversity
    special compared to a random grab of the same number of lines?'
    """
    rng = random.Random(seed)
    n = len(lines)
    nulls = []
    for _ in range(n_perms):
        sample = rng.choices(all_lines, k=n)
        initials = [line[0] for line in sample]
        freq = Counter(initials)
        n_unique = sum(1 for c in freq.values() if c == 1)
        nulls.append(n_unique / len(initials) if initials else 0.0)
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# 6b-2 — Tally token consistency across hands
# =====================================================================

def hand_tally_profile(lines: list[list[str]], top_n: int = 3) -> dict:
    """Compute the top-N most frequent continuation tokens for a hand.

    Continuation = all words at position >= 2 on each line.
    """
    continuation = [w for line in lines for w in line[1:]]
    if not continuation:
        return {"top_tally": [], "n_continuation": 0, "concentration": 0.0}
    freq = Counter(continuation)
    total = len(continuation)
    top = freq.most_common(top_n)
    top_count = sum(c for _, c in top)
    return {
        "top_tally": [{"word": w, "count": c, "frac": round(c / total, 4)}
                      for w, c in top],
        "n_continuation": total,
        "concentration": round(top_count / total, 4),
    }


def tally_overlap(profiles: dict[str, dict]) -> dict:
    """Compute overlap of top-3 tally tokens across all hands.

    If all hands share the same top tokens → shared convention.
    """
    if len(profiles) < 2:
        return {"n_hands": len(profiles), "overlap": []}

    # Get top-3 words per hand
    hand_tops = {}
    for hand, prof in profiles.items():
        words = [t["word"] for t in prof.get("top_tally", [])]
        if words:
            hand_tops[hand] = set(words)

    if not hand_tops:
        return {"n_hands": 0, "overlap": []}

    # Intersection of all hands' top-3
    common = set.intersection(*hand_tops.values())
    # Pairwise Jaccard
    hands = sorted(hand_tops.keys())
    pairs = {}
    for i, h1 in enumerate(hands):
        for h2 in hands[i + 1:]:
            union = hand_tops[h1] | hand_tops[h2]
            inter = hand_tops[h1] & hand_tops[h2]
            pairs[f"{h1}-{h2}"] = round(len(inter) / len(union), 3) if union else 0.0

    return {
        "n_hands": len(hand_tops),
        "common_top_tokens": sorted(common),
        "n_common": len(common),
        "pairwise_jaccard": pairs,
        "per_hand_top": {h: sorted(s) for h, s in hand_tops.items()},
    }


# =====================================================================
# 6b-3 — Register role score per hand
# =====================================================================

def hand_register_role(lines: list[list[str]]) -> dict:
    """Classify a hand's register role.

    Computes:
      - label_ratio: fraction of line-initial hapax (unique first words)
      - tally_ratio: concentration of top-3 continuation tokens
      - role_score: label_ratio - tally_ratio
        Positive → this hand is more "labeller" (diverse first words)
        Negative → this hand is more "tallier" (few repeated cont. words)
    """
    # Label side
    initials = [line[0] for line in lines]
    freq_init = Counter(initials)
    n_hapax = sum(1 for c in freq_init.values() if c == 1)
    label_ratio = n_hapax / len(initials) if initials else 0.0

    # Tally side
    continuation = [w for line in lines for w in line[1:]]
    if continuation:
        freq_cont = Counter(continuation)
        top3_count = sum(c for _, c in freq_cont.most_common(3))
        tally_ratio = top3_count / len(continuation)
    else:
        tally_ratio = 0.0

    role_score = label_ratio - tally_ratio

    return {
        "label_ratio": round(label_ratio, 4),
        "tally_ratio": round(tally_ratio, 4),
        "role_score": round(role_score, 4),
        "n_lines": len(lines),
        "n_continuation": len(continuation),
    }


def null_role_score(lines: list[list[str]], all_lines: list[list[str]],
                    n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null for role_score: sample N lines from all register lines."""
    rng = random.Random(seed)
    n = len(lines)
    nulls = []
    for _ in range(n_perms):
        sample = rng.choices(all_lines, k=n)
        r = hand_register_role(sample)
        nulls.append(r["role_score"])
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

def save_to_db(results: dict, tally_ov: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS hand_register")
    cur.execute("""
        CREATE TABLE hand_register (
            hand              TEXT PRIMARY KEY,
            hand_name         TEXT,
            n_lines           INTEGER,
            n_continuation    INTEGER,
            uniqueness_obs    REAL,
            uniqueness_null   REAL,
            z_uniqueness      REAL,
            concentration     REAL,
            label_ratio       REAL,
            tally_ratio       REAL,
            role_score        REAL,
            role_null_mean    REAL,
            z_role            REAL,
            top_tally_json    TEXT
        )
    """)

    for hand, d in sorted(results.items()):
        if d.get("skipped"):
            continue
        cur.execute(
            "INSERT INTO hand_register VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                hand, HAND_NAMES.get(hand, hand),
                d["uniqueness"]["n_lines"],
                d["role"]["n_continuation"],
                d["uniqueness"]["uniqueness"],
                d["uniqueness"].get("null_mean"),
                d["uniqueness"].get("z"),
                d["tally"]["concentration"],
                d["role"]["label_ratio"],
                d["role"]["tally_ratio"],
                d["role"]["role_score"],
                d["role"].get("null_mean"),
                d["role"].get("z"),
                json.dumps(d["tally"]["top_tally"]),
            ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(results: dict, tally_ov: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 6b — Per-hand register role analysis")
    lines.append("  Sections analysed: B (balneo), H (herbal), S (astro), P (pharma)")
    lines.append("  Only pages attributed to each hand within register sections")
    lines.append("=" * 80)

    # Main table
    lines.append(
        f"\n  {'Hand':>5}  {'Name':>12}  {'Lines':>5}  "
        f"{'Uniq%':>6}  {'z_uniq':>7}  "
        f"{'Conc%':>6}  "
        f"{'Label':>6}  {'Tally':>6}  {'Role':>6}  {'z_role':>7}"
    )
    lines.append("  " + "-" * 82)

    sorted_hands = sorted(
        [d for d in results.values() if not d.get("skipped")],
        key=lambda d: d["role"]["role_score"],
        reverse=True,
    )
    for d in sorted_hands:
        hand = d["hand"]
        name = HAND_NAMES.get(hand, hand)[:12]
        n = d["uniqueness"]["n_lines"]
        uniq = d["uniqueness"]["uniqueness"] * 100
        z_u = d["uniqueness"].get("z")
        z_u_str = f"{z_u:+.2f}" if z_u is not None else "  n/a"
        conc = d["tally"]["concentration"] * 100
        lab = d["role"]["label_ratio"] * 100
        tal = d["role"]["tally_ratio"] * 100
        role = d["role"]["role_score"]
        z_r = d["role"].get("z")
        z_r_str = f"{z_r:+.2f}" if z_r is not None else "  n/a"
        lines.append(
            f"  {hand:>5}  {name:>12}  {n:>5}  "
            f"{uniq:>5.1f}%  {z_u_str:>7}  "
            f"{conc:>5.1f}%  "
            f"{lab:>5.1f}%  {tal:>5.1f}%  {role:>+.3f}  {z_r_str:>7}"
        )

    # Tally overlap
    lines.append("\n── Tally token overlap across hands ──")
    common = tally_ov.get("common_top_tokens", [])
    if common:
        lines.append(f"  Tokens shared by ALL hands: {', '.join(common)}")
    else:
        lines.append("  No token is in the top-3 of ALL hands.")

    per_hand = tally_ov.get("per_hand_top", {})
    for h in sorted(per_hand.keys()):
        lines.append(f"    Hand {h}: {', '.join(per_hand[h])}")

    pw = tally_ov.get("pairwise_jaccard", {})
    if pw:
        lines.append("\n  Pairwise Jaccard (top-3 tally tokens):")
        for pair in sorted(pw.keys()):
            lines.append(f"    {pair}: {pw[pair]:.3f}")

    # Legend
    lines.append("\n── What the scores mean ──")
    lines.append("  Uniq%:  line-initial uniqueness (high = diverse labels)")
    lines.append("  Conc%:  top-3 continuation token concentration (high = few tally tokens)")
    lines.append("  Label:  fraction of hapax in line-initial position")
    lines.append("  Tally:  fraction of top-3 in continuation position")
    lines.append("  Role:   label_ratio - tally_ratio (positive = more labeller)")
    lines.append("  z_role: vs null (random sample of same # lines from register sections)")
    lines.append(
        "\n  CAVEAT: $H in IVTFF is per-folio. We cannot determine if different"
    )
    lines.append("  hands wrote different parts of the same line.")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 6b: per-hand register role analysis in register sections."""
    report_path = config.stats_dir / "hand_register.json"
    summary_path = config.stats_dir / "hand_register_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_register report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 6b — Per-Hand Register Role Analysis")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus (line-level)...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # 2. Split by hand
    print_step("Splitting by hand...")
    corpus = split_corpus_by_hand(pages)

    # 3. Get lines per hand in register sections only
    print_step(f"Collecting lines per hand in register sections "
               f"({', '.join(sorted(REGISTER_SECTIONS))})...")
    hand_lines = lines_by_hand(corpus, sections=REGISTER_SECTIONS)
    for hand in sorted(hand_lines.keys()):
        n = len(hand_lines[hand])
        nw = sum(len(l) for l in hand_lines[hand])
        click.echo(f"    Hand {hand}: {n} lines, {nw:,} words")

    # All register lines (pooled, for null model)
    all_reg_lines = [line for lines in hand_lines.values() for line in lines]
    click.echo(f"    Total register lines: {len(all_reg_lines)}")

    # 4. Test 6b-1 — Per-hand uniqueness
    print_step("Test 6b-1 — Per-hand line-initial uniqueness...")
    results: dict[str, dict] = {}
    for i, (hand, hlines) in enumerate(sorted(hand_lines.items())):
        if len(hlines) < 10:
            results[hand] = {"hand": hand, "skipped": True,
                             "reason": f"only {len(hlines)} lines"}
            click.echo(f"    Hand {hand}: skip (< 10 lines)")
            continue

        uniq = hand_initial_uniqueness(hlines)
        null = null_hand_uniqueness(hlines, all_reg_lines,
                                     seed=SEED + i * 10)
        z_u = z_score(uniq["uniqueness"], null["null_mean"], null["null_std"])
        uniq["null_mean"] = round(null["null_mean"], 4)
        uniq["null_std"] = round(null["null_std"], 6)
        uniq["z"] = round(z_u, 3) if z_u is not None else None

        click.echo(f"    Hand {hand}: uniq={uniq['uniqueness']*100:.1f}%  "
                   f"z={z_u:+.2f}" if z_u is not None else
                   f"    Hand {hand}: uniq={uniq['uniqueness']*100:.1f}%  z=n/a")

        results[hand] = {"hand": hand, "uniqueness": uniq}

    # 5. Test 6b-2 — Tally token profiles + overlap
    print_step("Test 6b-2 — Tally token profiles per hand...")
    tally_profiles = {}
    for hand in sorted(hand_lines.keys()):
        if results.get(hand, {}).get("skipped"):
            continue
        hlines = hand_lines[hand]
        prof = hand_tally_profile(hlines)
        results[hand]["tally"] = prof
        tally_profiles[hand] = prof
        top3 = ", ".join(f"{t['word']}({t['frac']*100:.1f}%)"
                         for t in prof["top_tally"][:3])
        click.echo(f"    Hand {hand}: conc={prof['concentration']*100:.1f}%  "
                   f"top-3: {top3}")

    tally_ov = tally_overlap(tally_profiles)
    common = tally_ov.get("common_top_tokens", [])
    click.echo(f"    Common across ALL hands: {', '.join(common) if common else 'none'}")

    # 6. Test 6b-3 — Register role score per hand
    print_step("Test 6b-3 — Register role score (label vs tally)...")
    for i, (hand, hlines) in enumerate(sorted(hand_lines.items())):
        if results.get(hand, {}).get("skipped"):
            continue
        role = hand_register_role(hlines)
        null = null_role_score(hlines, all_reg_lines, seed=SEED + i * 10 + 5)
        z_r = z_score(role["role_score"], null["null_mean"], null["null_std"])
        role["null_mean"] = round(null["null_mean"], 4)
        role["null_std"] = round(null["null_std"], 6)
        role["z"] = round(z_r, 3) if z_r is not None else None
        results[hand]["role"] = role

        z_str = f"{z_r:+.2f}" if z_r is not None else "n/a"
        click.echo(f"    Hand {hand}: label={role['label_ratio']*100:.1f}%  "
                   f"tally={role['tally_ratio']*100:.1f}%  "
                   f"role={role['role_score']:+.3f}  z={z_str}")

    # 7. Save JSON
    print_step("Saving JSON...")
    report = {
        "register_sections": sorted(REGISTER_SECTIONS),
        "per_hand": results,
        "tally_overlap": tally_ov,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 8. Save TXT
    summary = format_summary(results, tally_ov)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 9. Save to DB
    print_step("Writing DB table hand_register...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(results, tally_ov, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
