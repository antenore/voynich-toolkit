"""
Phase 8 — Montemurro & Zanette (2013) verification.

Tests two specific claims from "Keywords and Co-Occurrence Patterns in the
Voynich Manuscript: An Information-Theoretic Analysis" (PLoS ONE 2013):

  8a — Word-section mutual information
       M&Z claim: the most informative words distribute non-uniformly across
       sections (like real languages where topic-words cluster in chapters).
       Test: compute mutual information I(word; section) for each word.
       Null: shuffle section labels across pages, recompute MI.

  8b — Section similarity via shared vocabulary
       M&Z claim: sections with similar illustrations share more vocabulary.
       Specifically: Pharmaceutical-Herbal are linked, Recipes-Astrological are linked.
       Test: compute Jaccard vocabulary overlap for all section pairs.
       Null: random section assignment, recompute Jaccard matrix.

Both tests are on raw EVA — no decoding, no lexicon.

Output:
  montemurro_test.json
  montemurro_test_summary.txt
  DB table: montemurro_test
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SEED = 42
N_NULL = 1000


# =====================================================================
# 8a — Mutual information: word × section
# =====================================================================

def compute_word_section_mi(pages: list[dict], top_n: int = 100) -> dict:
    """Compute mutual information I(word; section) for the top-N words.

    MI measures how much knowing the word tells you about which section
    you're in. High MI = word is section-specific (like "photosynthesis"
    in a biology chapter). Low MI = word appears everywhere equally.

    I(W;S) = sum_{w,s} P(w,s) * log2(P(w,s) / (P(w) * P(s)))
    """
    # Build joint distribution
    section_words: dict[str, list[str]] = {}
    for p in pages:
        sec = p.get("section", "?")
        if sec not in section_words:
            section_words[sec] = []
        section_words[sec].extend(p["words"])

    # Total corpus
    all_words = [w for ws in section_words.values() for w in ws]
    total = len(all_words)
    if total == 0:
        return {"skipped": True, "reason": "no words"}

    # Word frequencies (top-N)
    word_freq = Counter(all_words)
    top_words = [w for w, _ in word_freq.most_common(top_n)]

    # Section frequencies
    sec_totals = {sec: len(ws) for sec, ws in section_words.items()}

    # Per-word MI contribution
    word_mi = {}
    for word in top_words:
        mi = 0.0
        p_w = word_freq[word] / total
        for sec, ws in section_words.items():
            p_s = sec_totals[sec] / total
            # Count of word in section
            count_ws = sum(1 for w in ws if w == word)
            if count_ws == 0:
                continue
            p_ws = count_ws / total
            mi += p_ws * math.log2(p_ws / (p_w * p_s))
        word_mi[word] = round(mi, 6)

    # Global MI (sum over all top words)
    total_mi = sum(word_mi.values())

    # Section distribution per top word (for reporting)
    word_profiles = {}
    for word in top_words[:30]:
        profile = {}
        for sec in sorted(section_words.keys()):
            count = sum(1 for w in section_words[sec] if w == word)
            rate = count / sec_totals[sec] if sec_totals[sec] > 0 else 0.0
            profile[sec] = round(rate * 1000, 2)  # per-mille for readability
        word_profiles[word] = profile

    # Top-10 most informative words
    sorted_mi = sorted(word_mi.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_mi": round(total_mi, 4),
        "n_words_tested": len(top_words),
        "n_sections": len(section_words),
        "sections": sorted(section_words.keys()),
        "top_informative": [
            {"word": w, "mi": mi, "freq": word_freq[w]}
            for w, mi in sorted_mi[:30]
        ],
        "word_profiles": word_profiles,
    }


def null_word_section_mi(pages: list[dict], top_n: int = 100,
                          n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle section labels across pages, recompute total MI.

    If words are randomly distributed across sections, MI ≈ 0.
    """
    rng = random.Random(seed)

    # Pre-compute
    all_words = [w for p in pages for w in p["words"]]
    word_freq = Counter(all_words)
    top_words = set(w for w, _ in word_freq.most_common(top_n))
    total = len(all_words)

    sections = [p.get("section", "?") for p in pages]
    page_words = [p["words"] for p in pages]

    nulls = []
    for _ in range(n_perms):
        shuffled_secs = list(sections)
        rng.shuffle(shuffled_secs)

        # Rebuild section_words
        sec_ws: dict[str, list[str]] = {}
        for sec, ws in zip(shuffled_secs, page_words):
            if sec not in sec_ws:
                sec_ws[sec] = []
            sec_ws[sec].extend(ws)

        sec_totals = {sec: len(ws) for sec, ws in sec_ws.items()}

        mi_total = 0.0
        for word in top_words:
            p_w = word_freq[word] / total
            for sec, ws in sec_ws.items():
                p_s = sec_totals[sec] / total
                count_ws = sum(1 for w in ws if w == word)
                if count_ws == 0:
                    continue
                p_ws = count_ws / total
                mi_total += p_ws * math.log2(p_ws / (p_w * p_s))

        nulls.append(mi_total)

    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls, ddof=1)),
    }


# =====================================================================
# 8b — Section similarity (Jaccard vocabulary overlap)
# =====================================================================

def section_jaccard_matrix(pages: list[dict]) -> dict:
    """Compute Jaccard vocabulary overlap for all section pairs."""
    section_vocab: dict[str, set] = {}
    section_tokens: dict[str, int] = {}
    for p in pages:
        sec = p.get("section", "?")
        if sec not in section_vocab:
            section_vocab[sec] = set()
            section_tokens[sec] = 0
        section_vocab[sec].update(p["words"])
        section_tokens[sec] += len(p["words"])

    sections = sorted(section_vocab.keys())
    matrix = {}
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            inter = section_vocab[s1] & section_vocab[s2]
            union = section_vocab[s1] | section_vocab[s2]
            j = len(inter) / len(union) if union else 0.0
            pair = f"{s1}-{s2}"
            matrix[pair] = {
                "section_a": s1,
                "section_b": s2,
                "name_a": SECTION_NAMES.get(s1, s1),
                "name_b": SECTION_NAMES.get(s2, s2),
                "jaccard": round(j, 4),
                "intersection": len(inter),
                "union": len(union),
                "tokens_a": section_tokens[s1],
                "tokens_b": section_tokens[s2],
                "types_a": len(section_vocab[s1]),
                "types_b": len(section_vocab[s2]),
            }

    return {"sections": sections, "pairs": matrix,
            "section_sizes": {s: {"tokens": section_tokens[s],
                                   "types": len(section_vocab[s])}
                              for s in sections}}


def null_section_jaccard(pages: list[dict], pair_key: str,
                          n_perms: int = N_NULL, seed: int = SEED) -> dict:
    """Null: shuffle section labels, recompute Jaccard for a specific pair."""
    rng = random.Random(seed)
    s1, s2 = pair_key.split("-")

    sections = [p.get("section", "?") for p in pages]
    page_word_sets = [set(p["words"]) for p in pages]

    nulls = []
    for _ in range(n_perms):
        shuffled = list(sections)
        rng.shuffle(shuffled)

        vocab_a: set = set()
        vocab_b: set = set()
        for sec, ws in zip(shuffled, page_word_sets):
            if sec == s1:
                vocab_a |= ws
            elif sec == s2:
                vocab_b |= ws

        if not vocab_a or not vocab_b:
            continue
        union = vocab_a | vocab_b
        j = len(vocab_a & vocab_b) / len(union) if union else 0.0
        nulls.append(j)

    return {
        "null_mean": float(np.mean(nulls)) if nulls else 0.0,
        "null_std": float(np.std(nulls, ddof=1)) if len(nulls) > 1 else 0.0,
    }


# =====================================================================
# Helpers
# =====================================================================

def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(mi_result: dict, mi_null: dict, jaccard_result: dict,
               jaccard_nulls: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS montemurro_test")
    cur.execute("""
        CREATE TABLE montemurro_test (
            test           TEXT,
            key            TEXT,
            observed       REAL,
            null_mean      REAL,
            null_std       REAL,
            z_score        REAL,
            detail_json    TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    # MI
    z_mi = z_score(mi_result["total_mi"], mi_null["null_mean"], mi_null["null_std"])
    cur.execute("INSERT INTO montemurro_test VALUES (?,?,?,?,?,?,?)", (
        "8a_mutual_information", "total_mi",
        mi_result["total_mi"], mi_null["null_mean"], mi_null["null_std"],
        round(z_mi, 3) if z_mi is not None else None,
        json.dumps({"top10": mi_result["top_informative"][:10]}),
    ))

    # Top informative words
    for item in mi_result["top_informative"][:30]:
        cur.execute("INSERT INTO montemurro_test VALUES (?,?,?,?,?,?,?)", (
            "8a_word_mi", item["word"],
            item["mi"], None, None, None,
            json.dumps(mi_result["word_profiles"].get(item["word"], {})),
        ))

    # Jaccard pairs
    for pair_key, d in jaccard_result["pairs"].items():
        null_d = jaccard_nulls.get(pair_key, {})
        z = z_score(d["jaccard"], null_d.get("null_mean", 0),
                    null_d.get("null_std", 0))
        cur.execute("INSERT INTO montemurro_test VALUES (?,?,?,?,?,?,?)", (
            "8b_section_jaccard", pair_key,
            d["jaccard"],
            null_d.get("null_mean"),
            null_d.get("null_std"),
            round(z, 3) if z is not None else None,
            json.dumps(d),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(mi_result: dict, mi_null: dict,
                   jaccard_result: dict, jaccard_nulls: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 8 — Montemurro & Zanette (2013) verification")
    lines.append("=" * 80)

    # 8a — MI
    z_mi = z_score(mi_result["total_mi"], mi_null["null_mean"], mi_null["null_std"])
    lines.append("\n── 8a — Word-section mutual information ──")
    lines.append(f"  M&Z claim: informative words cluster in specific sections.")
    lines.append(f"  Total MI (top-{mi_result['n_words_tested']} words): "
                 f"{mi_result['total_mi']:.4f} bits")
    lines.append(f"  Null (shuffled sections): {mi_null['null_mean']:.4f} bits")
    z_str = f"{z_mi:+.2f}" if z_mi is not None else "n/a"
    lines.append(f"  z = {z_str}")

    if z_mi is not None and z_mi > 5:
        lines.append("  CONFIRMED: words carry section-specific information far above chance")
    elif z_mi is not None and z_mi > 2:
        lines.append("  PARTIALLY CONFIRMED: some section-specificity above chance")
    else:
        lines.append("  NOT CONFIRMED: words are not section-specific")

    lines.append(f"\n  Top-10 most informative words (MI contribution, per-mille rate by section):")
    sections = mi_result["sections"]
    sec_header = "  ".join(f"{s:>4}" for s in sections)
    lines.append(f"  {'Word':>12}  {'MI':>8}  {'Freq':>5}  {sec_header}")
    lines.append("  " + "-" * (30 + 6 * len(sections)))

    for item in mi_result["top_informative"][:10]:
        word = item["word"]
        profile = mi_result["word_profiles"].get(word, {})
        rates = "  ".join(f"{profile.get(s, 0):>4.0f}" for s in sections)
        lines.append(f"  {word:>12}  {item['mi']:>8.5f}  {item['freq']:>5}  {rates}")

    # 8b — Jaccard
    lines.append("\n── 8b — Section vocabulary overlap (Jaccard) ──")
    lines.append(f"  M&Z claim: Pharma-Herbal and Recipes-Astro are most linked.")

    # Sort by Jaccard descending
    sorted_pairs = sorted(jaccard_result["pairs"].items(),
                          key=lambda x: x[1]["jaccard"], reverse=True)
    lines.append(
        f"  {'Pair':>6}  {'Names':>30}  {'Jaccard':>7}  {'Null':>7}  {'z':>7}"
    )
    lines.append("  " + "-" * 62)
    for pair_key, d in sorted_pairs:
        null_d = jaccard_nulls.get(pair_key, {})
        z = z_score(d["jaccard"], null_d.get("null_mean", 0),
                    null_d.get("null_std", 0))
        z_str = f"{z:+.2f}" if z is not None else "  n/a"
        name = f"{d['name_a'][:14]}-{d['name_b'][:14]}"
        lines.append(
            f"  {pair_key:>6}  {name:>30}  {d['jaccard']:>7.4f}  "
            f"{null_d.get('null_mean', 0):>7.4f}  {z_str:>7}"
        )

    # M&Z specific predictions
    lines.append("\n  M&Z predicted links:")
    ph = jaccard_result["pairs"].get("H-P", {})
    if ph:
        lines.append(f"    Herbal-Pharma:  J={ph['jaccard']:.4f}  "
                     f"(M&Z: should be strong)")
    # Check if S-Z or S-P exists for "Recipes-Astro" (sections vary)
    for pair_key in ["S-Z", "P-S", "P-Z"]:
        pd = jaccard_result["pairs"].get(pair_key, {})
        if pd:
            lines.append(f"    {pd['name_a']}-{pd['name_b']}:  J={pd['jaccard']:.4f}")

    lines.append("\n── Note ──")
    lines.append("  MI rates are in per-mille (occurrences per 1000 words in that section).")
    lines.append("  High MI = word is concentrated in few sections (topic-specific).")
    lines.append("  Null = section labels shuffled across pages (destroys topic structure).")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 8: Montemurro & Zanette (2013) verification — MI + section Jaccard."""
    report_path = config.stats_dir / "montemurro_test.json"
    summary_path = config.stats_dir / "montemurro_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  montemurro_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 8 — Montemurro & Zanette (2013) Verification")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # Section distribution
    sec_dist = Counter(p.get("section", "?") for p in pages)
    for sec in sorted(sec_dist.keys()):
        click.echo(f"    Section {sec} ({SECTION_NAMES.get(sec, sec)}): "
                   f"{sec_dist[sec]} pages")

    # 2. Test 8a — Mutual information
    print_step("Test 8a — Word-section mutual information (top-100 words)...")
    mi_result = compute_word_section_mi(pages, top_n=100)
    if mi_result.get("skipped"):
        click.echo(f"    Skipped: {mi_result.get('reason')}")
    else:
        click.echo(f"    Total MI: {mi_result['total_mi']:.4f} bits")
        click.echo(f"    Top-5 informative words:")
        for item in mi_result["top_informative"][:5]:
            click.echo(f"      {item['word']:>12}  MI={item['mi']:.5f}  "
                       f"freq={item['freq']}")

    print_step(f"Null model for MI ({N_NULL} permutations)...")
    mi_null = null_word_section_mi(pages, top_n=100, n_perms=N_NULL, seed=SEED)
    z_mi = z_score(mi_result["total_mi"], mi_null["null_mean"], mi_null["null_std"])
    z_str = f"{z_mi:+.2f}" if z_mi is not None else "n/a"
    click.echo(f"    Null: mean={mi_null['null_mean']:.4f}  "
               f"std={mi_null['null_std']:.4f}")
    click.echo(f"    z = {z_str}")

    # 3. Test 8b — Section Jaccard
    print_step("Test 8b — Section vocabulary overlap (Jaccard)...")
    jaccard_result = section_jaccard_matrix(pages)
    sorted_pairs = sorted(jaccard_result["pairs"].items(),
                          key=lambda x: x[1]["jaccard"], reverse=True)
    for pair_key, d in sorted_pairs[:10]:
        click.echo(f"    {pair_key} ({d['name_a']}-{d['name_b']}): "
                   f"J={d['jaccard']:.4f}")

    # Null for top-5 pairs + M&Z predicted pairs
    print_step(f"Null model for Jaccard ({N_NULL} permutations, key pairs)...")
    test_pairs = set()
    # Top 5 pairs
    for pair_key, _ in sorted_pairs[:5]:
        test_pairs.add(pair_key)
    # M&Z predicted: H-P, S-Z
    for p in ["H-P", "S-Z", "P-S", "P-Z"]:
        if p in jaccard_result["pairs"]:
            test_pairs.add(p)

    jaccard_nulls = {}
    for pair_key in sorted(test_pairs):
        null = null_section_jaccard(pages, pair_key, n_perms=N_NULL,
                                     seed=SEED + hash(pair_key) % 1000)
        obs = jaccard_result["pairs"][pair_key]["jaccard"]
        z = z_score(obs, null["null_mean"], null["null_std"])
        jaccard_nulls[pair_key] = null
        z_str = f"z={z:+.2f}" if z is not None else "z=n/a"
        click.echo(f"    {pair_key}: obs={obs:.4f}  null={null['null_mean']:.4f}  "
                   f"{z_str}")

    # 4. Save JSON
    print_step("Saving JSON...")
    report = {
        "mutual_information": mi_result,
        "mi_null": mi_null,
        "jaccard": jaccard_result,
        "jaccard_nulls": jaccard_nulls,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    {report_path}")

    # 5. Save TXT
    summary = format_summary(mi_result, mi_null, jaccard_result, jaccard_nulls)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 6. Save to DB
    print_step("Writing DB table montemurro_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(mi_result, mi_null, jaccard_result, jaccard_nulls, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
