"""
Labelese Crib Test — Phase 19.

Tests the commenter's challenge (Reddit, 2026-04-30):

  "Internal consistency should be higher with a nomenclator and null-free
   encoding. We basically don't have any internal matches between herbal
   and recipes."

If the manuscript is a herbal + pharmacopeia (categorical register
hypothesis), the LABEL words next to plants in the herbal section should
reappear as ingredients in the paragraph TEXT of the pharma section.
A nomenclator with consistent encoding predicts this overlap.

This test parses the IVTFF locus tags directly to separate:
  - Labels (@La, @Lc, @Lf, @Ln, @Lp, @Ls, @Lt, @Lx, @Lz) per section
  - Paragraph text (@P, @P0, @P1, @Pb) per section
  - Circular/radial (@C, @R) per section

Then computes cross-section vocabulary overlap and compares to a null
constructed by shuffling words within sections.
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

SEED = 42
N_PERM = 500

# IVTFF locus type tags
LABEL_TAGS = {"La", "Lc", "Lf", "Ln", "Lp", "Ls", "Lt", "Lx", "Lz"}
PARAGRAPH_TAGS = {"P", "P0", "P1", "P2", "P3", "Pa", "Pb", "Pc"}
# Note: the leading @ is in the file; we strip it during parsing.

# Section codes per IVTFF spec
SECTION_NAMES = {
    "H": "herbal",
    "P": "pharma",
    "B": "balneological",
    "S": "astronomical",
    "Z": "zodiac",
    "C": "cosmological",
    "T": "text",
    "A": "?",
    "?": "unknown",
}


# =====================================================================
# IVTFF parser with locus-type awareness
# =====================================================================

def parse_ivtff_with_locus(filepath: Path, transcriber: str = "H") -> dict:
    """Parse IVTFF preserving locus type (label vs paragraph vs circle/radius)."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    header_re = re.compile(r"^<(f\w+)>\s+<!\s*(.*?)>")
    meta_re = re.compile(r"\$(\w)=(\w+)")
    # locus pattern: <f1r.1,@P0;H>  or  <f88r.2,@Lf;H>
    transcription_re = re.compile(
        r"^<(f\w+)\.\d+[a-z]*,([@+])([A-Za-z][0-9a-z]*);(\w)>\s+(.+)"
    )

    current_meta = {}
    current_folio = None

    # Per (folio, locus_type, section): list of words
    locus_data: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    def extract_words(eva_text: str) -> list[str]:
        """Clean an EVA line and extract words (dot-separated)."""
        clean = re.sub(r"\{[^}]*\}", "", eva_text)
        clean = re.sub(r"<[^>]*>", "", clean)
        clean = re.sub(r"[%!?\[\]*,]", "", clean)
        words = []
        for token in clean.split("."):
            word = re.sub(r"[^a-z]", "", token)
            if word:
                words.append(word)
        return words

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue

        m = header_re.match(line)
        if m:
            current_folio = m.group(1)
            current_meta = dict(meta_re.findall(m.group(2)))
            continue

        m = transcription_re.match(line)
        if not m:
            continue

        folio, locus_prefix, locus_type, trans, eva_text = m.groups()
        if trans != transcriber:
            continue

        section = current_meta.get("I", "?")
        words = extract_words(eva_text)
        locus_data[(folio, locus_type, section)].extend(words)

    return {
        "locus_data": locus_data,
        "n_loci": len(locus_data),
    }


# =====================================================================
# Vocabulary extraction
# =====================================================================

def vocabulary_by_section_and_type(
    locus_data: dict[tuple[str, str, str], list[str]],
) -> dict[str, dict[str, Counter]]:
    """Partition vocabulary by section × locus-type (label vs paragraph)."""
    out: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"labels": Counter(), "paragraphs": Counter(), "other": Counter()}
    )

    for (folio, locus_type, section), words in locus_data.items():
        if locus_type in LABEL_TAGS:
            bucket = "labels"
        elif locus_type in PARAGRAPH_TAGS:
            bucket = "paragraphs"
        else:
            bucket = "other"
        out[section][bucket].update(words)

    return out


# =====================================================================
# Overlap analysis
# =====================================================================

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def overlap_count(a: set, b: set) -> int:
    return len(a & b)


def measure_crib_overlap(
    vocab: dict[str, dict[str, Counter]],
    rng: np.random.Generator,
    n_perm: int,
) -> dict:
    """Compute label-paragraph overlap across sections and permutation null.

    Key tests:
      - Herbal LABELS × Pharma PARAGRAPHS: how many herbal label-words
        also appear as text in pharma paragraphs?
      - Pharma LABELS × Herbal PARAGRAPHS: reverse direction
      - Permutation null: shuffle words across sections, see if observed
        overlap exceeds chance
    """
    # Real overlaps
    sections_present = list(vocab.keys())

    # All label words (across sections)
    all_labels = Counter()
    all_paragraphs = Counter()
    for sec in sections_present:
        all_labels.update(vocab[sec]["labels"])
        all_paragraphs.update(vocab[sec]["paragraphs"])

    real_pairs = {}
    pair_keys = [
        ("H_labels", "P_paragraphs"),  # herbal labels in pharma text
        ("P_labels", "H_paragraphs"),  # pharma labels in herbal text
        ("H_labels", "B_paragraphs"),  # herbal labels in balneo text
        ("B_labels", "H_paragraphs"),  # balneo labels in herbal text
        ("H_labels", "S_paragraphs"),  # herbal labels in astro text
        ("Z_labels", "S_paragraphs"),  # zodiac labels in astro text
    ]
    for src_key, tgt_key in pair_keys:
        src_sec, src_type = src_key.split("_")
        tgt_sec, tgt_type = tgt_key.split("_")
        src_vocab = set(vocab.get(src_sec, {}).get(src_type, Counter()).keys())
        tgt_vocab = set(vocab.get(tgt_sec, {}).get(tgt_type, Counter()).keys())
        if not src_vocab or not tgt_vocab:
            continue
        real_pairs[f"{src_key}__in__{tgt_key}"] = {
            "n_src": len(src_vocab),
            "n_tgt": len(tgt_vocab),
            "overlap_count": overlap_count(src_vocab, tgt_vocab),
            "jaccard": round(jaccard(src_vocab, tgt_vocab), 4),
            "src_in_tgt_pct": round(
                100 * len(src_vocab & tgt_vocab) / len(src_vocab), 2
            ),
        }

    # Permutation null:
    # for each (src, tgt) pair, randomly relabel words to sections
    # while preserving label/paragraph type counts per section.
    # Build a flat pool of all label words and all paragraph words.
    all_label_words: list[tuple[str, str]] = []  # (section, word)
    all_para_words: list[tuple[str, str]] = []
    for sec in sections_present:
        for w, c in vocab[sec]["labels"].items():
            all_label_words.extend([(sec, w)] * c)
        for w, c in vocab[sec]["paragraphs"].items():
            all_para_words.extend([(sec, w)] * c)

    # Sectional counts (preserved during permutation)
    label_sec_counts = Counter(s for s, _ in all_label_words)
    para_sec_counts = Counter(s for s, _ in all_para_words)
    label_words_only = [w for _, w in all_label_words]
    para_words_only = [w for _, w in all_para_words]

    null_distributions: dict[str, list[int]] = {k: [] for k in real_pairs}

    for _ in range(n_perm):
        # Reshuffle section assignments preserving counts
        rng.shuffle(label_words_only)
        rng.shuffle(para_words_only)

        # Reconstruct shuffled vocabularies
        shuffled_labels: dict[str, Counter] = defaultdict(Counter)
        shuffled_paras: dict[str, Counter] = defaultdict(Counter)
        idx = 0
        for sec, n in label_sec_counts.items():
            for w in label_words_only[idx:idx + n]:
                shuffled_labels[sec][w] += 1
            idx += n
        idx = 0
        for sec, n in para_sec_counts.items():
            for w in para_words_only[idx:idx + n]:
                shuffled_paras[sec][w] += 1
            idx += n

        for pair_key in real_pairs:
            src_full, tgt_full = pair_key.split("__in__")
            src_sec, src_type = src_full.split("_")
            tgt_sec, tgt_type = tgt_full.split("_")

            shuffled_vocab = {
                "labels": shuffled_labels,
                "paragraphs": shuffled_paras,
            }
            src_vocab_n = set(shuffled_vocab[src_type].get(src_sec, Counter()).keys())
            tgt_vocab_n = set(shuffled_vocab[tgt_type].get(tgt_sec, Counter()).keys())
            null_distributions[pair_key].append(overlap_count(src_vocab_n, tgt_vocab_n))

    # Compute z-scores
    for pair_key, real_data in real_pairs.items():
        nulls = null_distributions[pair_key]
        if not nulls:
            continue
        null_mean = float(np.mean(nulls))
        null_std = float(np.std(nulls))
        z = (real_data["overlap_count"] - null_mean) / null_std if null_std > 0 else 0
        real_data["null_overlap_mean"] = round(null_mean, 1)
        real_data["null_overlap_std"] = round(null_std, 1)
        real_data["z_score"] = round(z, 2)

    return real_pairs


# =====================================================================
# Specific labels: top label words and their reappearance
# =====================================================================

def trace_top_labels(
    vocab: dict[str, dict[str, Counter]],
    src_section: str,
    n_top: int = 30,
) -> list[dict]:
    """Trace top label words from src_section: where else do they appear?"""
    src_labels = vocab.get(src_section, {}).get("labels", Counter())
    if not src_labels:
        return []

    out = []
    for word, count in src_labels.most_common(n_top):
        appearances = {}
        for sec, vd in vocab.items():
            if sec == src_section:
                continue
            in_labels = vd["labels"].get(word, 0)
            in_paragraphs = vd["paragraphs"].get(word, 0)
            if in_labels > 0 or in_paragraphs > 0:
                appearances[sec] = {
                    "labels": in_labels,
                    "paragraphs": in_paragraphs,
                }
        out.append({
            "word": word,
            "count_in_src_labels": count,
            "appearances_elsewhere": appearances,
            "n_other_sections": len(appearances),
        })
    return out


# =====================================================================
# Reporting
# =====================================================================

def format_summary(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 19 — LABELESE CROSS-SECTION CRIB TEST")
    lines.append("Do labels in one section reappear as paragraph text in another?")
    lines.append("=" * 78)
    lines.append("")

    stats = report["stats"]
    lines.append(f"Total loci parsed: {stats['n_loci']:,}")
    lines.append("")

    lines.append("Per-section vocabulary sizes (types):")
    for sec, vd in sorted(report["per_section_vocab_sizes"].items()):
        sec_name = SECTION_NAMES.get(sec, sec)
        lines.append(f"  {sec} ({sec_name:<14s}): "
                     f"labels={vd['labels']:>5d} types | "
                     f"paragraphs={vd['paragraphs']:>5d} types")
    lines.append("")

    lines.append("--- Cross-section overlap (real vs null) ---")
    lines.append(f"{'Comparison':<40s} {'src':>6s} {'tgt':>6s} "
                 f"{'overlap':>8s} {'src%':>7s} {'null':>8s} {'z':>6s}")
    lines.append("-" * 78)
    for pair_key, data in report["crib_overlaps"].items():
        src_full, tgt_full = pair_key.split("__in__")
        label = f"{src_full} in {tgt_full}"
        lines.append(
            f"  {label:<38s} {data['n_src']:>6d} {data['n_tgt']:>6d} "
            f"{data['overlap_count']:>8d} {data['src_in_tgt_pct']:>6.1f}% "
            f"{data.get('null_overlap_mean', 0):>8.1f} "
            f"{data.get('z_score', 0):>+6.2f}"
        )
    lines.append("")

    # Top label words and their reappearance
    lines.append("--- Top 10 herbal label words: where do they reappear? ---")
    for entry in report["top_herbal_labels"][:10]:
        if entry["n_other_sections"] == 0:
            others = "NONE"
        else:
            parts = []
            for sec, ad in entry["appearances_elsewhere"].items():
                bits = []
                if ad["labels"]:
                    bits.append(f"L:{ad['labels']}")
                if ad["paragraphs"]:
                    bits.append(f"P:{ad['paragraphs']}")
                parts.append(f"{sec}({','.join(bits)})")
            others = ", ".join(parts)
        lines.append(f"  {entry['word']:<14s} (×{entry['count_in_src_labels']}) → {others}")
    lines.append("")

    if report.get("top_pharma_labels"):
        lines.append("--- Top 10 pharma label words: where do they reappear? ---")
        for entry in report["top_pharma_labels"][:10]:
            if entry["n_other_sections"] == 0:
                others = "NONE"
            else:
                parts = []
                for sec, ad in entry["appearances_elsewhere"].items():
                    bits = []
                    if ad["labels"]:
                        bits.append(f"L:{ad['labels']}")
                    if ad["paragraphs"]:
                        bits.append(f"P:{ad['paragraphs']}")
                    parts.append(f"{sec}({','.join(bits)})")
                others = ", ".join(parts)
            lines.append(f"  {entry['word']:<14s} (×{entry['count_in_src_labels']}) → {others}")
        lines.append("")

    # Verdict
    lines.append("=" * 78)
    lines.append("VERDICT")
    lines.append("=" * 78)
    overlaps = report["crib_overlaps"]
    h_in_p = overlaps.get("H_labels__in__P_paragraphs", {})
    p_in_h = overlaps.get("P_labels__in__H_paragraphs", {})

    h_z = h_in_p.get("z_score", 0)
    p_z = p_in_h.get("z_score", 0)
    h_pct = h_in_p.get("src_in_tgt_pct", 0)
    p_pct = p_in_h.get("src_in_tgt_pct", 0)

    if h_z > 2 or p_z > 2:
        lines.append("SIGNAL: significant herbal↔pharma label-text overlap above null.")
        lines.append(f"  H labels in P paragraphs: {h_pct:.1f}% (z={h_z:+.2f})")
        lines.append(f"  P labels in H paragraphs: {p_pct:.1f}% (z={p_z:+.2f})")
        lines.append("Compatible with nomenclator-style cross-referencing.")
    elif h_z > -2 and p_z > -2:
        lines.append("NO SIGNAL: herbal-pharma overlap is at chance level.")
        lines.append(f"  H labels in P paragraphs: {h_pct:.1f}% (z={h_z:+.2f})")
        lines.append(f"  P labels in H paragraphs: {p_pct:.1f}% (z={p_z:+.2f})")
        lines.append("The commenter's challenge is empirically supported:")
        lines.append("herbal plant labels do NOT systematically appear in pharma recipes")
        lines.append("more than chance. Weakens (but does not eliminate) the categorical")
        lines.append("register/nomenclator hypothesis with consistent encoding.")
    else:
        lines.append("ANTI-SIGNAL: herbal-pharma overlap is BELOW chance.")
        lines.append(f"  H labels in P paragraphs: {h_pct:.1f}% (z={h_z:+.2f})")
        lines.append(f"  P labels in H paragraphs: {p_pct:.1f}% (z={p_z:+.2f})")
        lines.append("Sections are MORE differentiated than random would predict.")
        lines.append("Strong evidence against simple nomenclator with shared vocabulary.")
    lines.append("")
    lines.append("Caveat: a verbose cipher with section-specific tables would not")
    lines.append("predict overlap either, so this test specifically targets the")
    lines.append("'simple shared nomenclator' hypothesis, not all cipher variants.")
    lines.append("")

    return "\n".join(lines) + "\n"


def save_to_db(config: ToolkitConfig, report: dict):
    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS labelese_crib_test")
    cur.execute("""
        CREATE TABLE labelese_crib_test (
            comparison TEXT PRIMARY KEY,
            n_src INTEGER,
            n_tgt INTEGER,
            overlap_count INTEGER,
            src_in_tgt_pct REAL,
            null_overlap_mean REAL,
            null_overlap_std REAL,
            z_score REAL
        )
    """)
    rows = []
    for pair_key, data in report["crib_overlaps"].items():
        rows.append((
            pair_key,
            data["n_src"], data["n_tgt"], data["overlap_count"],
            data["src_in_tgt_pct"],
            data.get("null_overlap_mean", 0),
            data.get("null_overlap_std", 0),
            data.get("z_score", 0),
        ))
    cur.executemany(
        "INSERT INTO labelese_crib_test VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Phase 19: Labelese cross-section crib test."""
    report_path = config.stats_dir / "labelese_crib_test.json"
    summary_path = config.stats_dir / "labelese_crib_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Labelese crib test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 19 — Labelese Cross-Section Crib Test")

    print_step("Parsing IVTFF with locus-type awareness...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    parsed = parse_ivtff_with_locus(eva_file, transcriber="H")
    locus_data = parsed["locus_data"]
    click.echo(f"    Loci parsed: {parsed['n_loci']:,}")

    print_step("Partitioning vocabulary by section × locus-type...")
    vocab = vocabulary_by_section_and_type(locus_data)
    per_section_vocab_sizes = {
        sec: {
            "labels": len(vd["labels"]),
            "paragraphs": len(vd["paragraphs"]),
            "other": len(vd["other"]),
        }
        for sec, vd in vocab.items()
    }
    for sec, sizes in per_section_vocab_sizes.items():
        sec_name = SECTION_NAMES.get(sec, sec)
        click.echo(f"    {sec} ({sec_name}): {sizes}")

    print_step(f"Measuring cross-section crib overlaps ({N_PERM} perms)...")
    rng = np.random.default_rng(SEED)
    crib_overlaps = measure_crib_overlap(vocab, rng, N_PERM)
    for pair_key, data in crib_overlaps.items():
        click.echo(f"    {pair_key}: overlap={data['overlap_count']} "
                   f"({data['src_in_tgt_pct']:.1f}% of src), "
                   f"z={data.get('z_score', 0):+.2f}")

    print_step("Tracing top label words across sections...")
    top_herbal_labels = trace_top_labels(vocab, "H", n_top=30)
    top_pharma_labels = trace_top_labels(vocab, "P", n_top=30)

    print_step("Saving results...")
    report = {
        "stats": {"n_loci": parsed["n_loci"]},
        "per_section_vocab_sizes": per_section_vocab_sizes,
        "crib_overlaps": crib_overlaps,
        "top_herbal_labels": top_herbal_labels,
        "top_pharma_labels": top_pharma_labels,
        "parameters": {
            "n_perm": N_PERM,
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
    click.echo(f"    DB:   labelese_crib_test table")

    click.echo(f"\n{summary}")
