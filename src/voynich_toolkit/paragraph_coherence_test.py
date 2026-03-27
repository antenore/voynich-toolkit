"""
Phase 7c — Paragraph coherence test.

Phases 7a and 7b established a paradox:
  - Lines are self-contained: no word repetition crosses line boundaries (7a)
  - Paragraphs are real: @P / +P / =P lines have different properties (7b)

This raises the question: if lines don't share words across boundaries,
what CONNECTS lines within the same paragraph? Are paragraphs just visual
groupings, or do lines within a paragraph share deeper structural properties?

Four tests:

  7c-1 — Vocabulary overlap (Jaccard)
         Compare vocabulary overlap between pairs of lines from the SAME
         paragraph vs pairs from DIFFERENT paragraphs. If intra > inter,
         paragraphs have thematic coherence despite no word continuity.

  7c-2 — Character distribution coherence
         For each paragraph, compute variance of per-line character
         distributions. Compare within-paragraph variance vs between-paragraph
         variance. (Analogous to one-way ANOVA on character profiles.)

  7c-3 — Word-length consistency
         Are word lengths more uniform within a paragraph than between?
         Within-paragraph std vs between-paragraph std.

  7c-4 — Bigram profile similarity
         Cosine similarity of bigram distributions: intra-paragraph pairs
         vs inter-paragraph pairs.

Expected results if paragraphs = thematic sections of a register:
  - Moderate vocabulary overlap within paragraphs (shared topic → shared words)
  - Similar character/bigram profiles within paragraphs
  - But NO word repetition across line boundaries (each line = unique entry)

Expected results if paragraphs = arbitrary visual groupings:
  - No difference between intra- and inter-paragraph similarity

Output:
  paragraph_coherence_test.json
  paragraph_coherence_test_summary.txt
  DB table: paragraph_coherence_test
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
from .currier_line_test import parse_ivtff_lines
from .utils import print_header, print_step


SEED = 42
N_NULL = 1000
MIN_PARA_LINES = 3   # minimum lines to consider a paragraph


# =====================================================================
# Build paragraphs from parsed lines
# =====================================================================

def build_paragraphs(lines: list[dict]) -> list[list[dict]]:
    """Group consecutive lines into paragraphs using Stolfi markers.

    A paragraph starts at @P (para_start) and ends at the next @P or =P.
    Labels are excluded.

    Returns: list of paragraphs, each = list of line dicts.
    """
    paragraphs: list[list[dict]] = []
    current: list[dict] = []

    for line in lines:
        pt = line["para_type"]
        if pt == "label":
            continue
        if pt == "para_start":
            if current:
                paragraphs.append(current)
            current = [line]
        elif pt in ("para_cont", "para_end", "other"):
            current.append(line)
            if pt == "para_end":
                paragraphs.append(current)
                current = []

    if current:
        paragraphs.append(current)

    return paragraphs


# =====================================================================
# 7c-1 — Vocabulary overlap (Jaccard)
# =====================================================================

def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def vocabulary_overlap_test(paragraphs: list[list[dict]],
                             n_samples: int = N_NULL,
                             seed: int = SEED) -> dict:
    """Compare Jaccard vocabulary overlap: intra-paragraph vs inter-paragraph.

    Samples pairs of lines from the same paragraph (intra) and from different
    paragraphs (inter), computes Jaccard on word sets.
    """
    rng = random.Random(seed)

    # Only use paragraphs with >= MIN_PARA_LINES lines
    valid = [p for p in paragraphs if len(p) >= MIN_PARA_LINES]
    if len(valid) < 2:
        return {"skipped": True, "reason": "too few valid paragraphs"}

    # Pre-compute word sets per line
    line_sets: list[tuple[int, set]] = []  # (para_idx, word_set)
    for pi, para in enumerate(valid):
        for line in para:
            ws = set(line["words"])
            if ws:
                line_sets.append((pi, ws))

    # Sample intra-paragraph pairs
    intra_jaccards = []
    for _ in range(n_samples):
        # Pick a random paragraph with >= 2 lines
        pi = rng.choice([i for i in range(len(valid)) if len(valid[i]) >= 2])
        para_lines = [(idx, ws) for idx, ws in line_sets if idx == pi]
        if len(para_lines) < 2:
            continue
        l1, l2 = rng.sample(para_lines, 2)
        intra_jaccards.append(jaccard(l1[1], l2[1]))

    # Sample inter-paragraph pairs
    inter_jaccards = []
    for _ in range(n_samples):
        l1, l2 = rng.sample(line_sets, 2)
        while l1[0] == l2[0]:  # ensure different paragraphs
            l2 = rng.choice(line_sets)
        inter_jaccards.append(jaccard(l1[1], l2[1]))

    intra_mean = float(np.mean(intra_jaccards)) if intra_jaccards else 0.0
    inter_mean = float(np.mean(inter_jaccards)) if inter_jaccards else 0.0
    intra_std = float(np.std(intra_jaccards, ddof=1)) if len(intra_jaccards) > 1 else 0.0
    inter_std = float(np.std(inter_jaccards, ddof=1)) if len(inter_jaccards) > 1 else 0.0

    # Effect size: how much more similar are intra pairs?
    pooled_std = np.sqrt((intra_std ** 2 + inter_std ** 2) / 2)
    cohens_d = (intra_mean - inter_mean) / pooled_std if pooled_std > 1e-10 else None

    # Permutation test: shuffle paragraph assignments
    observed_diff = intra_mean - inter_mean
    null_diffs = []
    all_para_indices = [pi for pi, _ in line_sets]
    for _ in range(n_samples):
        shuffled = list(all_para_indices)
        rng.shuffle(shuffled)
        # Re-label line_sets with shuffled indices
        perm_intra = []
        perm_inter = []
        for _ in range(min(200, n_samples)):
            i, j = rng.sample(range(len(line_sets)), 2)
            j_val = jaccard(line_sets[i][1], line_sets[j][1])
            if shuffled[i] == shuffled[j]:
                perm_intra.append(j_val)
            else:
                perm_inter.append(j_val)
        if perm_intra and perm_inter:
            null_diffs.append(np.mean(perm_intra) - np.mean(perm_inter))

    null_mean = float(np.mean(null_diffs)) if null_diffs else 0.0
    null_std = float(np.std(null_diffs, ddof=1)) if len(null_diffs) > 1 else 0.0
    z = (observed_diff - null_mean) / null_std if null_std > 1e-10 else None

    return {
        "intra_mean": round(intra_mean, 4),
        "inter_mean": round(inter_mean, 4),
        "intra_std": round(intra_std, 4),
        "inter_std": round(inter_std, 4),
        "difference": round(observed_diff, 4),
        "cohens_d": round(cohens_d, 3) if cohens_d is not None else None,
        "z": round(z, 3) if z is not None else None,
        "n_valid_paragraphs": len(valid),
        "n_samples": n_samples,
    }


# =====================================================================
# 7c-2 — Character distribution coherence
# =====================================================================

def char_profile(words: list[str]) -> np.ndarray:
    """Compute normalised character frequency vector (19 EVA chars)."""
    chars = "acdefghiklmnopqrsty"
    counts = Counter(ch for w in words for ch in w)
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(chars))
    return np.array([counts.get(ch, 0) / total for ch in chars])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(dot / (na * nb))


def char_coherence_test(paragraphs: list[list[dict]],
                         n_samples: int = N_NULL,
                         seed: int = SEED) -> dict:
    """Compare character profile similarity within vs between paragraphs."""
    rng = random.Random(seed)
    valid = [p for p in paragraphs if len(p) >= MIN_PARA_LINES]
    if len(valid) < 2:
        return {"skipped": True, "reason": "too few paragraphs"}

    # Pre-compute char profiles per line
    profiles: list[tuple[int, np.ndarray]] = []
    for pi, para in enumerate(valid):
        for line in para:
            prof = char_profile(line["words"])
            if np.sum(prof) > 0:
                profiles.append((pi, prof))

    # Sample intra and inter pairs
    intra_sims = []
    inter_sims = []
    for _ in range(n_samples):
        i, j = rng.sample(range(len(profiles)), 2)
        sim = cosine_sim(profiles[i][1], profiles[j][1])
        if profiles[i][0] == profiles[j][0]:
            intra_sims.append(sim)
        else:
            inter_sims.append(sim)

    # Need enough of both
    if len(intra_sims) < 50 or len(inter_sims) < 50:
        # Force sampling
        for _ in range(n_samples):
            pi = rng.choice([k for k in range(len(valid)) if len(valid[k]) >= 2])
            para_profs = [(idx, p) for idx, p in profiles if idx == pi]
            if len(para_profs) >= 2:
                a, b = rng.sample(para_profs, 2)
                intra_sims.append(cosine_sim(a[1], b[1]))
            # Inter
            a, b = rng.sample(profiles, 2)
            while a[0] == b[0]:
                b = rng.choice(profiles)
            inter_sims.append(cosine_sim(a[1], b[1]))

    intra_mean = float(np.mean(intra_sims))
    inter_mean = float(np.mean(inter_sims))

    return {
        "intra_cosine_mean": round(intra_mean, 4),
        "inter_cosine_mean": round(inter_mean, 4),
        "difference": round(intra_mean - inter_mean, 4),
        "n_intra": len(intra_sims),
        "n_inter": len(inter_sims),
    }


# =====================================================================
# 7c-3 — Word-length consistency
# =====================================================================

def word_length_coherence(paragraphs: list[list[dict]]) -> dict:
    """Compare within-paragraph word-length std vs between-paragraph std."""
    valid = [p for p in paragraphs if len(p) >= MIN_PARA_LINES]

    # Within-paragraph: std of mean word length across lines
    within_stds = []
    para_means = []
    for para in valid:
        line_means = []
        for line in para:
            if line["words"]:
                line_means.append(np.mean([len(w) for w in line["words"]]))
        if len(line_means) >= 2:
            within_stds.append(float(np.std(line_means, ddof=1)))
            para_means.append(float(np.mean(line_means)))

    # Between-paragraph: std of paragraph means
    within_mean_std = float(np.mean(within_stds)) if within_stds else 0.0
    between_std = float(np.std(para_means, ddof=1)) if len(para_means) > 1 else 0.0

    # F-ratio: between / within (> 1 means paragraphs differ from each other)
    f_ratio = between_std / within_mean_std if within_mean_std > 1e-10 else None

    return {
        "n_paragraphs": len(valid),
        "within_para_std_mean": round(within_mean_std, 4),
        "between_para_std": round(between_std, 4),
        "f_ratio": round(f_ratio, 3) if f_ratio is not None else None,
    }


# =====================================================================
# 7c-4 — Bigram profile similarity
# =====================================================================

def bigram_profile(words: list[str]) -> Counter:
    bg = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bg[w[i:i + 2]] += 1
    return bg


def bigram_cosine(bg_a: Counter, bg_b: Counter) -> float:
    all_keys = set(bg_a) | set(bg_b)
    if not all_keys:
        return 0.0
    a = np.array([bg_a.get(k, 0) for k in all_keys], dtype=float)
    b = np.array([bg_b.get(k, 0) for k in all_keys], dtype=float)
    return cosine_sim(a, b)


def bigram_coherence_test(paragraphs: list[list[dict]],
                           n_samples: int = N_NULL,
                           seed: int = SEED) -> dict:
    """Compare bigram profile cosine similarity within vs between paragraphs."""
    rng = random.Random(seed)
    valid = [p for p in paragraphs if len(p) >= MIN_PARA_LINES]
    if len(valid) < 2:
        return {"skipped": True, "reason": "too few paragraphs"}

    # Pre-compute bigram profiles per line
    profiles: list[tuple[int, Counter]] = []
    for pi, para in enumerate(valid):
        for line in para:
            bg = bigram_profile(line["words"])
            if bg:
                profiles.append((pi, bg))

    intra_sims = []
    inter_sims = []
    for _ in range(n_samples):
        # Intra
        pi = rng.choice([k for k in range(len(valid)) if len(valid[k]) >= 2])
        para_profs = [(idx, bg) for idx, bg in profiles if idx == pi]
        if len(para_profs) >= 2:
            a, b = rng.sample(para_profs, 2)
            intra_sims.append(bigram_cosine(a[1], b[1]))
        # Inter
        a, b = rng.sample(profiles, 2)
        while a[0] == b[0]:
            b = rng.choice(profiles)
        inter_sims.append(bigram_cosine(a[1], b[1]))

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0

    return {
        "intra_cosine_mean": round(intra_mean, 4),
        "inter_cosine_mean": round(inter_mean, 4),
        "difference": round(intra_mean - inter_mean, 4),
        "n_intra": len(intra_sims),
        "n_inter": len(inter_sims),
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(vocab: dict, char_coh: dict, wlen: dict, bigram: dict,
               db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS paragraph_coherence_test")
    cur.execute("""
        CREATE TABLE paragraph_coherence_test (
            test           TEXT PRIMARY KEY,
            intra_mean     REAL,
            inter_mean     REAL,
            difference     REAL,
            z_score        REAL,
            detail_json    TEXT
        )
    """)

    for test_name, d in [("vocabulary", vocab), ("char_profile", char_coh),
                          ("word_length", wlen), ("bigram_profile", bigram)]:
        if d.get("skipped"):
            continue
        cur.execute("INSERT INTO paragraph_coherence_test VALUES (?,?,?,?,?,?)", (
            test_name,
            d.get("intra_mean") or d.get("intra_cosine_mean") or d.get("within_para_std_mean"),
            d.get("inter_mean") or d.get("inter_cosine_mean") or d.get("between_para_std"),
            d.get("difference") or d.get("f_ratio"),
            d.get("z"),
            json.dumps(d),
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(vocab: dict, char_coh: dict, wlen: dict, bigram: dict,
                   n_paras: int, n_lines: int) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 7c — Paragraph coherence test")
    lines.append("  Question: lines are self-contained (7a), but paragraphs are real (7b).")
    lines.append("  What connects lines within the same paragraph?")
    lines.append("=" * 80)

    lines.append(f"\n  Paragraphs analysed: {n_paras} (>= {MIN_PARA_LINES} lines)")
    lines.append(f"  Total text lines: {n_lines}")

    # 7c-1
    lines.append("\n── 7c-1 — Vocabulary overlap (Jaccard) ──")
    if vocab.get("skipped"):
        lines.append(f"  Skipped: {vocab.get('reason')}")
    else:
        lines.append(f"  Intra-paragraph (same para):   {vocab['intra_mean']:.4f}")
        lines.append(f"  Inter-paragraph (diff paras):  {vocab['inter_mean']:.4f}")
        lines.append(f"  Difference:                    {vocab['difference']:+.4f}")
        if vocab.get("cohens_d") is not None:
            lines.append(f"  Cohen's d:                     {vocab['cohens_d']:.3f}")
        z = vocab.get("z")
        if z is not None:
            lines.append(f"  z (permutation):               {z:+.2f}")
        if vocab["difference"] > 0:
            pct = (vocab["difference"] / vocab["inter_mean"] * 100
                   if vocab["inter_mean"] > 0 else 0)
            lines.append(f"  Lines in the same paragraph share {pct:.0f}% more vocabulary")
        else:
            lines.append("  No vocabulary advantage for intra-paragraph lines")

    # 7c-2
    lines.append("\n── 7c-2 — Character profile similarity (cosine) ──")
    if char_coh.get("skipped"):
        lines.append(f"  Skipped: {char_coh.get('reason')}")
    else:
        lines.append(f"  Intra-paragraph:  {char_coh['intra_cosine_mean']:.4f}")
        lines.append(f"  Inter-paragraph:  {char_coh['inter_cosine_mean']:.4f}")
        lines.append(f"  Difference:       {char_coh['difference']:+.4f}")

    # 7c-3
    lines.append("\n── 7c-3 — Word-length consistency ──")
    lines.append(f"  Within-paragraph std (mean): {wlen['within_para_std_mean']:.4f}")
    lines.append(f"  Between-paragraph std:       {wlen['between_para_std']:.4f}")
    if wlen.get("f_ratio") is not None:
        lines.append(f"  F-ratio (between/within):    {wlen['f_ratio']:.3f}")
        if wlen["f_ratio"] > 1.5:
            lines.append("  Paragraphs differ in word length MORE than lines within them")
        else:
            lines.append("  Word length is fairly uniform across paragraphs")

    # 7c-4
    lines.append("\n── 7c-4 — Bigram profile similarity (cosine) ──")
    if bigram.get("skipped"):
        lines.append(f"  Skipped: {bigram.get('reason')}")
    else:
        lines.append(f"  Intra-paragraph:  {bigram['intra_cosine_mean']:.4f}")
        lines.append(f"  Inter-paragraph:  {bigram['inter_cosine_mean']:.4f}")
        lines.append(f"  Difference:       {bigram['difference']:+.4f}")

    # Overall verdict
    lines.append("\n── Verdict ──")
    diffs = []
    if not vocab.get("skipped"):
        diffs.append(("vocabulary", vocab["difference"]))
    if not char_coh.get("skipped"):
        diffs.append(("char_profile", char_coh["difference"]))
    if not bigram.get("skipped"):
        diffs.append(("bigram_profile", bigram["difference"]))

    positive = sum(1 for _, d in diffs if d > 0.001)

    if positive >= 2:
        lines.append("  PARAGRAPHS HAVE INTERNAL COHERENCE")
        lines.append("  Lines within a paragraph are more similar to each other than")
        lines.append("  to lines from other paragraphs — even though they don't share")
        lines.append("  exact words across line boundaries.")
        lines.append("")
        lines.append("  This is the signature of a STRUCTURED DOCUMENT: each line is a")
        lines.append("  self-contained entry, but lines are grouped by topic/category.")
        lines.append("  Compatible with: register, catalogue, formulary, nomenclature.")
        lines.append("  Incompatible with: continuous prose, random text, pure decoration.")
    elif positive == 0:
        lines.append("  NO INTERNAL COHERENCE DETECTED")
        lines.append("  Lines within a paragraph are no more similar than random lines.")
        lines.append("  Paragraphs may be purely visual groupings.")
    else:
        lines.append("  WEAK COHERENCE")
        lines.append("  Some similarity within paragraphs, but not consistent across metrics.")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 7c: test whether lines within a paragraph share structural properties."""
    report_path = config.stats_dir / "paragraph_coherence_test.json"
    summary_path = config.stats_dir / "paragraph_coherence_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  paragraph_coherence_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 7c — Paragraph Coherence Test")

    # 1. Parse IVTFF
    print_step("Parsing IVTFF file...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    all_lines = parse_ivtff_lines(eva_file)
    text_lines = [l for l in all_lines if l["para_type"] != "label"]
    click.echo(f"    {len(text_lines)} text lines")

    # 2. Build paragraphs
    print_step("Building paragraphs from Stolfi markers...")
    paragraphs = build_paragraphs(all_lines)
    valid = [p for p in paragraphs if len(p) >= MIN_PARA_LINES]
    total_lines_in_valid = sum(len(p) for p in valid)
    click.echo(f"    {len(paragraphs)} total paragraphs, "
               f"{len(valid)} with >= {MIN_PARA_LINES} lines "
               f"({total_lines_in_valid} lines)")

    # Size distribution
    sizes = [len(p) for p in valid]
    click.echo(f"    Paragraph sizes: min={min(sizes)}, max={max(sizes)}, "
               f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}")

    # 3. Test 7c-1 — Vocabulary overlap
    print_step(f"Test 7c-1 — Vocabulary overlap ({N_NULL} samples)...")
    vocab = vocabulary_overlap_test(valid, n_samples=N_NULL, seed=SEED)
    if not vocab.get("skipped"):
        z_str = f"z={vocab['z']:+.2f}" if vocab.get("z") is not None else ""
        click.echo(f"    Intra={vocab['intra_mean']:.4f}  "
                   f"Inter={vocab['inter_mean']:.4f}  "
                   f"Diff={vocab['difference']:+.4f}  {z_str}")

    # 4. Test 7c-2 — Character profile coherence
    print_step(f"Test 7c-2 — Character profile similarity ({N_NULL} samples)...")
    char_coh = char_coherence_test(valid, n_samples=N_NULL, seed=SEED + 1)
    if not char_coh.get("skipped"):
        click.echo(f"    Intra={char_coh['intra_cosine_mean']:.4f}  "
                   f"Inter={char_coh['inter_cosine_mean']:.4f}  "
                   f"Diff={char_coh['difference']:+.4f}")

    # 5. Test 7c-3 — Word-length consistency
    print_step("Test 7c-3 — Word-length consistency...")
    wlen = word_length_coherence(valid)
    click.echo(f"    Within-para std={wlen['within_para_std_mean']:.4f}  "
               f"Between-para std={wlen['between_para_std']:.4f}  "
               f"F={wlen.get('f_ratio', 'n/a')}")

    # 6. Test 7c-4 — Bigram profile similarity
    print_step(f"Test 7c-4 — Bigram profile similarity ({N_NULL} samples)...")
    bigram = bigram_coherence_test(valid, n_samples=N_NULL, seed=SEED + 2)
    if not bigram.get("skipped"):
        click.echo(f"    Intra={bigram['intra_cosine_mean']:.4f}  "
                   f"Inter={bigram['inter_cosine_mean']:.4f}  "
                   f"Diff={bigram['difference']:+.4f}")

    # 7. Save JSON
    print_step("Saving JSON...")
    report = {
        "n_paragraphs_total": len(paragraphs),
        "n_paragraphs_valid": len(valid),
        "vocabulary_overlap": vocab,
        "char_profile_coherence": char_coh,
        "word_length_coherence": wlen,
        "bigram_coherence": bigram,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    click.echo(f"    {report_path}")

    # 8. Save TXT
    summary = format_summary(vocab, char_coh, wlen, bigram,
                              len(valid), total_lines_in_valid)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 9. Save to DB
    print_step("Writing DB table paragraph_coherence_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(vocab, char_coh, wlen, bigram, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB not found — skip DB write")

    click.echo(f"\n{summary}")
