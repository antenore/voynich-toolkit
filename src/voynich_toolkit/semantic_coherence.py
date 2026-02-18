"""Semantic coherence analysis of the decoded Voynich text.

Annotates every decoded word with a 6-level classification (A-F),
discovers high-coherence passages via sliding window, and runs a
permutation test to assess whether coherent stretches exceed chance.

CLI: voynich --force semantic-coherence
"""

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ToolkitConfig
from .full_decode import decode_word
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# ── Annotation levels ────────────────────────────────────────────────
# A  glossed           – dictionary gloss (STEPBible/Jastrow/Klein/curated)
# B  morpho_glossed    – prefix/suffix stripped, stem has gloss (tier-2)
# C  compound_glossed  – compound split, both halves glossed (tier-3)
# D  attested          – Sefaria corpus attestation (no gloss)
# E  resolved_no_gloss – morpho/compound resolved, stem not glossed
# F  unknown           – no match

LEVEL_LABELS = {
    "A": "glossed",
    "B": "morpho_glossed",
    "C": "compound_glossed",
    "D": "attested",
    "E": "resolved_no_gloss",
    "F": "unknown",
}

# Levels that count as "semantically accessible" (translatable)
SEMANTIC_LEVELS = {"A", "B", "C"}


# =====================================================================
# Data loading
# =====================================================================

def load_annotation_data(config: ToolkitConfig):
    """Load classification data from SQLite + JSON.

    Returns (glossed, morpho, compound, attested, resolved) where:
      glossed  : dict {hebrew_word: gloss}        (level A)
      morpho   : dict {hebrew_word: composed_gloss} (level B)
      compound : dict {hebrew_word: composed_gloss} (level C)
      attested : set  {hebrew_word}                (level D)
      resolved : set  {hebrew_word}                (level E)
    """
    db_path = config.output_dir.parent / "voynich.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Level A: glossed (non-Sefaria sources)
    glossed = {}
    for row in conn.execute(
        "SELECT consonantal, gloss FROM glossed_words "
        "WHERE source != 'Sefaria-Corpus'"
    ):
        glossed[row["consonantal"]] = row["gloss"]

    # Level B: morpho_glossed (tier-2, stem has gloss)
    morpho = {}
    for row in conn.execute(
        "SELECT word, prefix_meaning, stem_gloss, suffix_meaning "
        "FROM phrase_resolutions WHERE tier=2 AND stem_in_gloss=1"
    ):
        parts = []
        if row["prefix_meaning"]:
            parts.append(row["prefix_meaning"])
        # Truncate long Jastrow glosses for display
        stem = row["stem_gloss"] or ""
        if len(stem) > 60:
            stem = stem[:57] + "..."
        parts.append(stem)
        if row["suffix_meaning"]:
            parts.append(row["suffix_meaning"])
        morpho[row["word"]] = "+".join(parts)

    # Level C: compound_glossed (tier-3, both halves glossed)
    compound = {}
    for row in conn.execute(
        "SELECT word, left_gloss, right_gloss FROM compound_splits "
        "WHERE rank=0 AND left_gloss != '' AND right_gloss != ''"
    ):
        lg = row["left_gloss"]
        rg = row["right_gloss"]
        lg_short = lg[:40] + "..." if len(lg) > 40 else lg
        rg_short = rg[:40] + "..." if len(rg) > 40 else rg
        compound[row["word"]] = f"{lg_short}+{rg_short}"

    # Level D: attested (Sefaria corpus, no dictionary gloss)
    attested = set()
    for row in conn.execute(
        "SELECT consonantal FROM glossed_words WHERE source = 'Sefaria-Corpus'"
    ):
        attested.add(row["consonantal"])

    # Level E: resolved but no gloss (remaining tier-2/3/4 not in A/B/C)
    resolved = set()
    for row in conn.execute("SELECT word FROM phrase_resolutions"):
        w = row["word"]
        if w not in glossed and w not in morpho and w not in compound:
            resolved.add(w)

    conn.close()
    return glossed, morpho, compound, attested, resolved


# =====================================================================
# Word classification
# =====================================================================

def build_type_classification(all_hebrew_types, glossed, morpho, compound,
                              attested, resolved):
    """Classify every unique Hebrew type into levels A-F.

    Returns dict {hebrew_word: level_letter}.
    """
    classification = {}
    for w in all_hebrew_types:
        if w in glossed:
            classification[w] = "A"
        elif w in morpho:
            classification[w] = "B"
        elif w in compound:
            classification[w] = "C"
        elif w in attested:
            classification[w] = "D"
        elif w in resolved:
            classification[w] = "E"
        else:
            classification[w] = "F"
    return classification


def get_gloss_label(heb_word, level, glossed, morpho, compound):
    """Return a short display label for a classified word."""
    if level == "A":
        g = glossed.get(heb_word, "")
        return g[:50] + "..." if len(g) > 50 else g
    elif level == "B":
        g = morpho.get(heb_word, "")
        return g[:50] + "..." if len(g) > 50 else g
    elif level == "C":
        g = compound.get(heb_word, "")
        return g[:50] + "..." if len(g) > 50 else g
    elif level == "D":
        return "(attested)"
    elif level == "E":
        return "(resolved)"
    return ""


# =====================================================================
# Line & passage metrics
# =====================================================================

def compute_line_metrics(levels):
    """Compute coherence metrics for a single line (list of level letters).

    Returns dict with glossed_ratio, max_consecutive, semantic_density.
    """
    n = len(levels)
    if n == 0:
        return {"glossed_ratio": 0.0, "max_consecutive": 0,
                "semantic_density": 0.0, "n_words": 0, "n_semantic": 0}

    semantic = [lv in SEMANTIC_LEVELS for lv in levels]
    n_sem = sum(semantic)

    # Max consecutive semantic words
    max_consec = 0
    cur = 0
    for s in semantic:
        if s:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    return {
        "glossed_ratio": n_sem / n,
        "max_consecutive": max_consec,
        "semantic_density": n_sem / n,  # same as glossed_ratio for full line
        "n_words": n,
        "n_semantic": n_sem,
    }


def score_passage(lines_metrics, all_semantic_types_in_passage,
                  total_semantic_tokens):
    """Score a multi-line passage for ranking.

    passage_score = glossed_ratio² × max_consecutive × log₂(1+total_words)
                    × √lexical_diversity
    """
    total_words = sum(m["n_words"] for m in lines_metrics)
    total_sem = sum(m["n_semantic"] for m in lines_metrics)
    if total_words == 0:
        return 0.0

    glossed_ratio = total_sem / total_words

    # Max consecutive across the passage (concatenated)
    max_consec = 0
    cur = 0
    for m in lines_metrics:
        # We need per-word level info — use stored semantic flags
        pass

    # Lexical diversity
    n_unique = len(all_semantic_types_in_passage)
    lex_div = n_unique / total_semantic_tokens if total_semantic_tokens > 0 else 0

    # Max consecutive is computed from combined line semantics
    # (passed separately)
    return glossed_ratio


# =====================================================================
# Passage discovery
# =====================================================================

def discover_passages(page_data, classification, glossed, morpho, compound,
                      min_words=6, max_window=10):
    """Find high-coherence passages via sliding window over page lines.

    Returns list of passage dicts sorted by score (descending).
    """
    all_passages = []

    for page in page_data:
        folio = page["folio"]
        section = page["section"]
        line_infos = []  # list of [{eva, heb, level, gloss}] per line

        for line_words_eva in page["line_words"]:
            line_info = []
            for eva_w in line_words_eva:
                _, heb, n_unk = decode_word(eva_w)
                level = classification.get(heb, "F")
                gloss = get_gloss_label(heb, level, glossed, morpho, compound)
                line_info.append({
                    "eva": eva_w,
                    "heb": heb,
                    "level": level,
                    "gloss": gloss,
                })
            if line_info:
                line_infos.append(line_info)

        if not line_infos:
            continue

        # Sliding window: 1 to max_window lines
        for win_size in range(1, min(max_window + 1, len(line_infos) + 1)):
            for start in range(len(line_infos) - win_size + 1):
                window = line_infos[start:start + win_size]
                all_words = []
                for line in window:
                    all_words.extend(line)

                total = len(all_words)
                if total < min_words:
                    continue

                sem_words = [w for w in all_words if w["level"] in SEMANTIC_LEVELS]
                n_sem = len(sem_words)
                glossed_ratio = n_sem / total

                # Max consecutive semantic
                max_consec = 0
                cur = 0
                for w in all_words:
                    if w["level"] in SEMANTIC_LEVELS:
                        cur += 1
                        max_consec = max(max_consec, cur)
                    else:
                        cur = 0

                # Lexical diversity among semantic words
                sem_types = set(w["heb"] for w in sem_words)
                lex_div = len(sem_types) / n_sem if n_sem > 0 else 0

                # Score
                score = (glossed_ratio ** 2
                         * max_consec
                         * math.log2(1 + total)
                         * math.sqrt(lex_div))

                all_passages.append({
                    "folio": folio,
                    "section": section,
                    "start_line": start,
                    "n_lines": win_size,
                    "n_words": total,
                    "n_semantic": n_sem,
                    "glossed_ratio": round(glossed_ratio, 3),
                    "max_consecutive": max_consec,
                    "lex_diversity": round(lex_div, 3),
                    "score": round(score, 4),
                    "words": all_words,
                })

    # Remove overlapping passages (greedy: keep highest score)
    all_passages.sort(key=lambda p: p["score"], reverse=True)
    selected = []
    used = set()  # (folio, line_idx) pairs

    for p in all_passages:
        lines_used = {(p["folio"], p["start_line"] + i)
                      for i in range(p["n_lines"])}
        if lines_used & used:
            continue
        selected.append(p)
        used |= lines_used

    return selected


# =====================================================================
# Permutation test
# =====================================================================

def run_permutation_test(page_data, classification, n_perms=1000, seed=42):
    """Test H₀: semantic labels are randomly assigned to types.

    Shuffles which types are "semantic" (A/B/C) among all types,
    then computes max_consecutive, n_high_lines, mean_glossed_ratio.
    """
    print_step(f"Permutation test ({n_perms} permutations)...")
    rng = np.random.default_rng(seed)

    # Build matrix: for each line, list of type_ids
    type_to_id = {}
    line_type_ids = []  # list of arrays of int
    line_n_words = []   # number of words per line (for min-word filter)

    for page in page_data:
        for line_words_eva in page["line_words"]:
            ids = []
            for eva_w in line_words_eva:
                _, heb, _ = decode_word(eva_w)
                if heb not in type_to_id:
                    type_to_id[heb] = len(type_to_id)
                ids.append(type_to_id[heb])
            if ids:
                line_type_ids.append(np.array(ids, dtype=np.int32))
                line_n_words.append(len(ids))

    n_types = len(type_to_id)
    id_to_type = {v: k for k, v in type_to_id.items()}

    # Real semantic mask
    real_mask = np.zeros(n_types, dtype=bool)
    for w, tid in type_to_id.items():
        if classification.get(w, "F") in SEMANTIC_LEVELS:
            real_mask[tid] = True

    n_semantic_types = int(real_mask.sum())

    def compute_stats(mask):
        """Compute coherence stats given a boolean mask over type IDs."""
        max_consec_global = 0
        n_high_lines = 0
        ratio_sum = 0.0
        n_qualifying = 0

        for ids, nw in zip(line_type_ids, line_n_words):
            sem = mask[ids]
            n_sem = int(sem.sum())
            ratio = n_sem / nw

            # Max consecutive
            mc = 0
            cur = 0
            for s in sem:
                if s:
                    cur += 1
                    if cur > mc:
                        mc = cur
                else:
                    cur = 0
            if mc > max_consec_global:
                max_consec_global = mc

            # High line (≥3 words, ≥50% glossed)
            if nw >= 3:
                n_qualifying += 1
                ratio_sum += ratio
                if ratio >= 0.5:
                    n_high_lines += 1

        mean_ratio = ratio_sum / n_qualifying if n_qualifying > 0 else 0.0
        return max_consec_global, n_high_lines, mean_ratio

    # Observed stats
    obs_mc, obs_hl, obs_mr = compute_stats(real_mask)

    # Permutation distribution
    perm_mc = np.zeros(n_perms, dtype=int)
    perm_hl = np.zeros(n_perms, dtype=int)
    perm_mr = np.zeros(n_perms, dtype=float)

    type_indices = np.arange(n_types)
    for i in range(n_perms):
        # Shuffle: pick n_semantic_types random types as "semantic"
        perm_mask = np.zeros(n_types, dtype=bool)
        chosen = rng.choice(type_indices, size=n_semantic_types, replace=False)
        perm_mask[chosen] = True
        mc, hl, mr = compute_stats(perm_mask)
        perm_mc[i] = mc
        perm_hl[i] = hl
        perm_mr[i] = mr

    def pval_z(obs, perm_arr):
        p = (np.sum(perm_arr >= obs) + 1) / (n_perms + 1)
        std = perm_arr.std()
        z = (obs - perm_arr.mean()) / std if std > 0 else 0.0
        return float(p), float(z), float(perm_arr.mean()), float(std)

    mc_p, mc_z, mc_mean, mc_std = pval_z(obs_mc, perm_mc)
    hl_p, hl_z, hl_mean, hl_std = pval_z(obs_hl, perm_hl)
    mr_p, mr_z, mr_mean, mr_std = pval_z(obs_mr, perm_mr)

    return {
        "n_perms": n_perms,
        "n_types": n_types,
        "n_semantic_types": n_semantic_types,
        "max_consecutive": {
            "observed": int(obs_mc),
            "perm_mean": round(mc_mean, 2),
            "perm_std": round(mc_std, 2),
            "p_value": round(mc_p, 4),
            "z_score": round(mc_z, 2),
        },
        "n_high_lines": {
            "observed": int(obs_hl),
            "perm_mean": round(hl_mean, 2),
            "perm_std": round(hl_std, 2),
            "p_value": round(hl_p, 4),
            "z_score": round(hl_z, 2),
        },
        "mean_glossed_ratio": {
            "observed": round(obs_mr, 4),
            "perm_mean": round(mr_mean, 4),
            "perm_std": round(mr_std, 4),
            "p_value": round(mr_p, 4),
            "z_score": round(mr_z, 2),
        },
    }


# =====================================================================
# Section breakdown
# =====================================================================

def section_breakdown(page_data, classification):
    """Per-section annotation statistics."""
    section_stats = {}
    for page in page_data:
        sec = page["section"]
        if sec not in section_stats:
            section_stats[sec] = Counter()
        for line_words_eva in page["line_words"]:
            for eva_w in line_words_eva:
                _, heb, _ = decode_word(eva_w)
                level = classification.get(heb, "F")
                section_stats[sec][level] += 1

    result = {}
    for sec in sorted(section_stats):
        counts = section_stats[sec]
        total = sum(counts.values())
        sem = sum(counts.get(lv, 0) for lv in SEMANTIC_LEVELS)
        result[sec] = {
            "total_tokens": total,
            "semantic_tokens": sem,
            "semantic_ratio": round(sem / total, 3) if total else 0.0,
            "by_level": {lv: counts.get(lv, 0) for lv in "ABCDEF"},
        }
    return result


# =====================================================================
# Formatting
# =====================================================================

def format_passage(p, rank):
    """Format a single passage for TXT output."""
    lines = []
    lines.append(
        f"#{rank}  {p['folio']} ({p['section']}) "
        f"lines {p['start_line']+1}-{p['start_line']+p['n_lines']}  "
        f"score={p['score']:.3f}  "
        f"{p['n_semantic']}/{p['n_words']} semantic ({p['glossed_ratio']:.0%})  "
        f"max_consec={p['max_consecutive']}"
    )

    # Split words back into lines for display
    eva_line = []
    heb_line = []
    ann_line = []
    for w in p["words"]:
        eva_line.append(w["eva"])
        heb_line.append(w["heb"])
        tag = f"[{w['level']}]"
        if w["level"] in SEMANTIC_LEVELS and w["gloss"]:
            gloss_short = w["gloss"][:25]
            ann_line.append(f"{tag}{gloss_short}")
        else:
            ann_line.append(tag + w["heb"])

    lines.append(f"  EVA:   {' '.join(eva_line)}")
    lines.append(f"  HEB:   {' '.join(heb_line)}")
    lines.append(f"  GLOSS: {' '.join(ann_line)}")
    return "\n".join(lines)


def format_summary(annotation_stats, section_stats, perm_results,
                   top_passages):
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 65)
    lines.append("  SEMANTIC COHERENCE ANALYSIS")
    lines.append("=" * 65)

    # Annotation stats
    lines.append("\n  ANNOTATION LEVELS:")
    total = sum(annotation_stats.values())
    for lv in "ABCDEF":
        n = annotation_stats.get(lv, 0)
        pct = n / total * 100 if total else 0
        label = LEVEL_LABELS[lv]
        lines.append(f"    {lv} ({label:20s}): {n:6d} tokens ({pct:5.1f}%)")
    sem_total = sum(annotation_stats.get(lv, 0) for lv in SEMANTIC_LEVELS)
    lines.append(f"    --- Semantic (A+B+C)  : {sem_total:6d} tokens "
                 f"({sem_total/total*100:.1f}%)" if total else "")

    # Section breakdown
    lines.append("\n  PER-SECTION BREAKDOWN:")
    lines.append(f"    {'Sec':4s} {'Total':>6s} {'Sem':>6s} {'Ratio':>7s}  "
                 f"{'A':>5s} {'B':>5s} {'C':>5s} {'D':>5s} {'E':>5s} {'F':>5s}")
    for sec, s in section_stats.items():
        bl = s["by_level"]
        lines.append(
            f"    {sec:4s} {s['total_tokens']:6d} {s['semantic_tokens']:6d} "
            f"{s['semantic_ratio']:7.1%}  "
            f"{bl['A']:5d} {bl['B']:5d} {bl['C']:5d} "
            f"{bl['D']:5d} {bl['E']:5d} {bl['F']:5d}"
        )

    # Permutation test
    lines.append("\n  PERMUTATION TEST (H₀: random label assignment):")
    for metric_name, data in perm_results.items():
        if isinstance(data, dict) and "observed" in data:
            sig = "*" if data["p_value"] < 0.05 else "ns"
            if data["p_value"] < 0.01:
                sig = "**"
            if data["p_value"] < 0.001:
                sig = "***"
            lines.append(
                f"    {metric_name:25s}: obs={data['observed']:>8}  "
                f"perm={data['perm_mean']:>8}±{data['perm_std']:<6}  "
                f"z={data['z_score']:>6.2f}  p={data['p_value']:.4f} ({sig})"
            )

    # Top passages
    lines.append(f"\n  TOP 20 COHERENT PASSAGES:")
    lines.append("-" * 65)
    for i, p in enumerate(top_passages[:20], 1):
        lines.append(format_passage(p, i))
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False):
    """Run semantic coherence analysis."""
    report_path = config.stats_dir / "semantic_coherence.json"
    text_path = config.stats_dir / "semantic_coherence_summary.txt"

    if report_path.exists() and not force:
        print("  ⏭  semantic_coherence.json exists (use --force)")
        return

    config.ensure_dirs()
    print_header("SEMANTIC COHERENCE ANALYSIS")

    # 1. Load annotation data
    print_step("Loading annotation data from SQLite + JSON...")
    glossed, morpho, compound, attested, resolved = load_annotation_data(config)
    print(f"    Level A (glossed):          {len(glossed):,} types")
    print(f"    Level B (morpho_glossed):   {len(morpho):,} types")
    print(f"    Level C (compound_glossed): {len(compound):,} types")
    print(f"    Level D (attested):         {len(attested):,} types")
    print(f"    Level E (resolved):         {len(resolved):,} types")

    # 2. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_path = config.eva_data_dir / "LSI_ivtff_0d.txt"
    parsed = parse_eva_words(eva_path)
    page_data = parsed["pages"]
    print(f"    {parsed['total_words']:,} words, {len(page_data)} pages")

    # 3. Decode all unique words and classify
    print_step("Decoding and classifying all types...")
    all_hebrew_types = set()
    for page in page_data:
        for line_words_eva in page["line_words"]:
            for eva_w in line_words_eva:
                _, heb, _ = decode_word(eva_w)
                all_hebrew_types.add(heb)

    classification = build_type_classification(
        all_hebrew_types, glossed, morpho, compound, attested, resolved
    )

    # Count tokens per level
    annotation_stats = Counter()
    for page in page_data:
        for line_words_eva in page["line_words"]:
            for eva_w in line_words_eva:
                _, heb, _ = decode_word(eva_w)
                annotation_stats[classification.get(heb, "F")] += 1

    total_tokens = sum(annotation_stats.values())
    sem_tokens = sum(annotation_stats.get(lv, 0) for lv in SEMANTIC_LEVELS)
    print(f"    {len(all_hebrew_types):,} unique types classified")
    print(f"    {sem_tokens:,}/{total_tokens:,} semantic tokens "
          f"({sem_tokens/total_tokens:.1%})")

    # 4. Section breakdown
    print_step("Computing per-section breakdown...")
    sec_stats = section_breakdown(page_data, classification)
    for sec, s in sec_stats.items():
        print(f"    {sec}: {s['semantic_ratio']:.1%} semantic "
              f"({s['semantic_tokens']}/{s['total_tokens']})")

    # 5. Discover passages
    print_step("Discovering high-coherence passages...")
    passages = discover_passages(
        page_data, classification, glossed, morpho, compound,
        min_words=6, max_window=10
    )
    print(f"    {len(passages):,} non-overlapping passages found")
    if passages:
        top = passages[0]
        print(f"    Best: {top['folio']} score={top['score']:.3f} "
              f"({top['glossed_ratio']:.0%} semantic)")

    # 6. Permutation test
    perm_results = run_permutation_test(
        page_data, classification, n_perms=1000, seed=42
    )
    mc = perm_results["max_consecutive"]
    hl = perm_results["n_high_lines"]
    mr = perm_results["mean_glossed_ratio"]
    print(f"    max_consecutive: obs={mc['observed']} "
          f"perm={mc['perm_mean']}±{mc['perm_std']} "
          f"z={mc['z_score']:.2f} p={mc['p_value']:.4f}")
    print(f"    n_high_lines:    obs={hl['observed']} "
          f"perm={hl['perm_mean']}±{hl['perm_std']} "
          f"z={hl['z_score']:.2f} p={hl['p_value']:.4f}")
    print(f"    mean_ratio:      obs={mr['observed']:.4f} "
          f"perm={mr['perm_mean']}±{mr['perm_std']} "
          f"z={mr['z_score']:.2f} p={mr['p_value']:.4f}")

    # 7. Prepare JSON output (strip verbose word data from all passages)
    top_50 = passages[:50]
    passages_json = []
    for p in top_50:
        pj = {k: v for k, v in p.items() if k != "words"}
        pj["words"] = [
            {"eva": w["eva"], "heb": w["heb"],
             "level": w["level"], "gloss": w["gloss"][:80]}
            for w in p["words"]
        ]
        passages_json.append(pj)

    output = {
        "annotation_stats": {
            "by_level": {lv: annotation_stats.get(lv, 0) for lv in "ABCDEF"},
            "total_tokens": total_tokens,
            "semantic_tokens": sem_tokens,
            "semantic_ratio": round(sem_tokens / total_tokens, 4),
            "n_types": len(all_hebrew_types),
            "n_semantic_types": sum(
                1 for w, lv in classification.items() if lv in SEMANTIC_LEVELS
            ),
        },
        "section_breakdown": sec_stats,
        "permutation_test": perm_results,
        "top_passages": passages_json,
        "summary": {
            "n_passages_found": len(passages),
            "best_score": passages[0]["score"] if passages else 0,
            "best_folio": passages[0]["folio"] if passages else "",
        },
    }

    # 8. Save
    print_step("Saving results...")
    with open(report_path, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)
    print(f"    → {report_path}")

    summary_text = format_summary(
        annotation_stats, sec_stats, perm_results, top_50
    )
    with open(text_path, "w") as f:
        f.write(summary_text)
    print(f"    → {text_path}")
