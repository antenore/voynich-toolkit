"""
Section specificity analysis: measure whether domain-specific terms
concentrate in their expected manuscript sections.

Phase 8D: computes Specificity Index (SI), runs permutation tests
on section labels, and generates a heatmap.

SI = (observed_in_expected / total) / (words_expected / words_total)
SI > 1 means the term concentrates above baseline expectation.
"""
import json
import random
from collections import Counter, defaultdict

import click
import numpy as np

from .config import ToolkitConfig
from .utils import print_header, print_step


# =====================================================================
# Expected domain → section mapping
# =====================================================================

# Which manuscript sections we expect each anchor category to appear in
EXPECTED_SECTIONS = {
    "botanica_parti": ["herbal", "pharmaceutical"],
    "botanica_azioni": ["herbal", "pharmaceutical"],
    "colori": ["herbal"],
    "corpo_medicina": ["balneological", "pharmaceutical"],
    "liquidi": ["balneological", "pharmaceutical"],
    "misure": ["pharmaceutical"],
    "astrologia": ["zodiac", "astronomical", "cosmological"],
    "pianeti": ["zodiac", "astronomical"],
    "alchimia": ["herbal", "pharmaceutical"],
    "cabala": [],  # No specific section expected
    "numeri": [],  # Everywhere
}


# =====================================================================
# Specificity Index computation
# =====================================================================

def compute_specificity_index(category_hits, section_word_counts,
                              expected_sections):
    """Compute the Specificity Index for a domain category.

    SI = (observed_in_expected / total_hits) / (words_in_expected / total_words)

    Args:
        category_hits: dict {section: n_hits} for this category
        section_word_counts: dict {section: n_words} across whole manuscript
        expected_sections: list of expected section names

    Returns: float SI (>1 = concentrated above expectation)
    """
    if not expected_sections or not category_hits:
        return 0.0

    total_hits = sum(category_hits.values())
    if total_hits == 0:
        return 0.0

    hits_in_expected = sum(
        category_hits.get(sec, 0) for sec in expected_sections)

    total_words = sum(section_word_counts.values())
    if total_words == 0:
        return 0.0

    words_in_expected = sum(
        section_word_counts.get(sec, 0) for sec in expected_sections)

    observed_ratio = hits_in_expected / total_hits
    baseline_ratio = words_in_expected / total_words

    if baseline_ratio == 0:
        return float('inf') if observed_ratio > 0 else 0.0

    return observed_ratio / baseline_ratio


def compute_all_specificity(anchor_report, pages_data):
    """Compute SI for all anchor categories.

    Args:
        anchor_report: dict loaded from anchor_words_report.json
        pages_data: dict {folio: {section, ...}}

    Returns: dict {category: {SI, observed_ratio, baseline_ratio, ...}}
    """
    # Count words per section
    section_word_counts = Counter()
    for folio, pdata in pages_data.items():
        section = pdata.get("section", "unknown")
        section_word_counts[section] += len(pdata.get("words_eva", []))

    results = {}

    section_dists = anchor_report.get("section_distributions", {})
    for cat_id, cat_hits in section_dists.items():
        expected = EXPECTED_SECTIONS.get(cat_id, [])
        if not expected:
            continue

        si = compute_specificity_index(
            cat_hits, section_word_counts, expected)

        total_hits = sum(cat_hits.values())
        hits_expected = sum(cat_hits.get(s, 0) for s in expected)
        total_words = sum(section_word_counts.values())
        words_expected = sum(section_word_counts.get(s, 0) for s in expected)

        results[cat_id] = {
            "specificity_index": round(si, 2),
            "expected_sections": expected,
            "hits_in_expected": hits_expected,
            "total_hits": total_hits,
            "observed_ratio": round(hits_expected / max(total_hits, 1), 3),
            "baseline_ratio": round(
                words_expected / max(total_words, 1), 3),
            "section_distribution": cat_hits,
        }

    return results


# =====================================================================
# Permutation test on section labels
# =====================================================================

def permutation_test_specificity(anchor_report, pages_data,
                                 n_perms=1000, seed=42):
    """Test SI significance by shuffling section labels.

    For each permutation: shuffle section labels → recompute category
    hits per section → recompute SI. The p-value is the fraction of
    permuted SIs >= real SI.
    """
    rng = random.Random(seed)

    # Real section labels
    folios = list(pages_data.keys())
    real_sections = [pages_data[f].get("section", "unknown") for f in folios]

    # Count words per section (doesn't change with shuffling)
    section_word_counts = Counter()
    folio_word_counts = {}
    for folio, pdata in pages_data.items():
        section = pdata.get("section", "unknown")
        n_words = len(pdata.get("words_eva", []))
        section_word_counts[section] += n_words
        folio_word_counts[folio] = n_words

    # Collect per-folio hit counts from the anchor report
    # We need to know: for each category, which folios had hits
    by_category = anchor_report.get("by_category", {})
    category_folio_hits = {}  # {cat_id: {folio: n_hits}}

    for cat_id, cat_data in by_category.items():
        folio_hits = Counter()
        for match_entry in cat_data.get("matches", []):
            for m in match_entry.get("decoded_forms",
                                     match_entry.get("matches", [])):
                for loc in m.get("locations", m.get("sample_pages", [])):
                    if isinstance(loc, dict):
                        folio_hits[loc["folio"]] += loc.get("count", 1)
                    else:
                        folio_hits[loc] += 1
        if folio_hits:
            category_folio_hits[cat_id] = dict(folio_hits)

    # Compute real SI for each category
    real_si = {}
    for cat_id in category_folio_hits:
        expected = EXPECTED_SECTIONS.get(cat_id, [])
        if not expected:
            continue

        # Compute section hits for this category
        cat_section_hits = Counter()
        for folio, count in category_folio_hits[cat_id].items():
            section = pages_data.get(folio, {}).get("section", "unknown")
            cat_section_hits[section] += count

        si = compute_specificity_index(
            cat_section_hits, section_word_counts, expected)
        real_si[cat_id] = si

    # Permutation test for each category
    results = {}
    for cat_id, real_value in real_si.items():
        expected = EXPECTED_SECTIONS.get(cat_id, [])
        perm_si_values = []

        for _ in range(n_perms):
            shuffled = real_sections.copy()
            rng.shuffle(shuffled)
            folio_to_section = dict(zip(folios, shuffled))

            # Recompute section word counts under shuffled labels
            perm_section_words = Counter()
            for folio, section in folio_to_section.items():
                perm_section_words[section] += folio_word_counts.get(folio, 0)

            # Recompute category section hits under shuffled labels
            perm_cat_hits = Counter()
            for folio, count in category_folio_hits.get(cat_id, {}).items():
                section = folio_to_section.get(folio, "unknown")
                perm_cat_hits[section] += count

            perm_si = compute_specificity_index(
                perm_cat_hits, perm_section_words, expected)
            perm_si_values.append(perm_si)

        arr = np.array(perm_si_values, dtype=float)
        n_ge = int(np.sum(arr >= real_value))
        p_value = (n_ge + 1) / (n_perms + 1)

        mean = float(arr.mean())
        std = float(arr.std())
        z_score = (real_value - mean) / std if std > 0 else float('inf')

        results[cat_id] = {
            "real_si": round(real_value, 2),
            "random_mean_si": round(mean, 2),
            "random_std_si": round(std, 2),
            "p_value": round(p_value, 6),
            "z_score": round(z_score, 2),
            "n_perms": n_perms,
            "significant_01": p_value < 0.01,
            "significant_05": p_value < 0.05,
        }

    return results


# =====================================================================
# Heatmap generation
# =====================================================================

def generate_heatmap(specificity_results, output_path):
    """Generate a domains × sections heatmap (matplotlib).

    Shows observed/expected ratios with color coding.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available, skipping heatmap")
        return

    categories = sorted(specificity_results.keys())
    if not categories:
        return

    # Collect all sections
    all_sections = set()
    for cat_data in specificity_results.values():
        all_sections.update(cat_data.get("section_distribution", {}).keys())
    sections = sorted(all_sections)

    if not sections:
        return

    # Build matrix
    matrix = np.zeros((len(categories), len(sections)))
    for i, cat in enumerate(categories):
        cat_data = specificity_results[cat]
        dist = cat_data.get("section_distribution", {})
        total = sum(dist.values()) or 1
        for j, sec in enumerate(sections):
            matrix[i, j] = dist.get(sec, 0) / total * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(sections)))
    ax.set_xticklabels(sections, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(sections)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color=color, fontsize=7)

    ax.set_title("Anchor Word Distribution: Domain × Section (%)")
    fig.colorbar(im, ax=ax, label="% of category hits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Section specificity analysis with permutation test."""
    report_path = config.stats_dir / "section_specificity_report.json"
    heatmap_path = config.stats_dir / "section_specificity_heatmap.png"

    if report_path.exists() and not force:
        click.echo("  Section specificity report exists. "
                   "Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 8D — Section Specificity Analysis")

    # 1. Load anchor words report
    print_step("Loading anchor words report...")
    anchor_path = config.stats_dir / "anchor_words_report.json"
    if not anchor_path.exists():
        raise click.ClickException(
            f"Anchor words report not found: {anchor_path}\n"
            "  Run first: voynich anchor-words"
        )
    with open(anchor_path) as f:
        anchor_report = json.load(f)
    click.echo(f"    {anchor_report.get('total_matched', 0)} anchor "
               "words matched")

    # 2. Load full decode for page data
    print_step("Loading decoded text...")
    full_decode_path = config.stats_dir / "full_decode.json"
    if not full_decode_path.exists():
        raise click.ClickException(
            f"Full decode not found: {full_decode_path}\n"
            "  Run first: voynich full-decode"
        )
    with open(full_decode_path) as f:
        fd = json.load(f)
    pages_data = fd["pages"]
    click.echo(f"    {len(pages_data)} pages loaded")

    # 3. Compute SI for all categories
    print_step("Computing Specificity Index per category...")
    specificity = compute_all_specificity(anchor_report, pages_data)

    for cat_id, cat_data in sorted(specificity.items()):
        si = cat_data["specificity_index"]
        marker = "***" if si > 2 else "**" if si > 1.5 else "*" if si > 1 else ""
        click.echo(f"    {cat_id:25s} SI={si:5.2f} "
                   f"({cat_data['hits_in_expected']}/{cat_data['total_hits']} "
                   f"in expected sections) {marker}")

    # 4. Permutation test
    print_step("Running permutation test (1000 permutations)...")
    perm_results = permutation_test_specificity(
        anchor_report, pages_data, n_perms=1000, seed=42)

    for cat_id, perm_data in sorted(perm_results.items()):
        sig = "***" if perm_data["p_value"] < 0.001 else \
              "**" if perm_data["p_value"] < 0.01 else \
              "*" if perm_data["p_value"] < 0.05 else "ns"
        click.echo(f"    {cat_id:25s} p={perm_data['p_value']:.4f} "
                   f"z={perm_data['z_score']:.1f} "
                   f"SI={perm_data['real_si']:.2f} vs "
                   f"random={perm_data['random_mean_si']:.2f} [{sig}]")

    # 5. Generate heatmap
    print_step("Generating heatmap...")
    generate_heatmap(specificity, heatmap_path)
    if heatmap_path.exists():
        click.echo(f"    Heatmap: {heatmap_path}")

    # 6. Save report
    print_step("Saving report...")
    report = {
        "specificity_indices": specificity,
        "permutation_tests": perm_results,
        "expected_sections": EXPECTED_SECTIONS,
        "n_categories_tested": len(perm_results),
        "n_significant_01": sum(
            1 for r in perm_results.values() if r["significant_01"]),
        "n_significant_05": sum(
            1 for r in perm_results.values() if r["significant_05"]),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")

    # 7. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  SECTION SPECIFICITY RESULTS")
    click.echo(f"{'=' * 60}")

    n_sig = report["n_significant_01"]
    n_tested = report["n_categories_tested"]
    click.echo(f"\n  {n_sig}/{n_tested} categories significant at p < 0.01")

    avg_si = np.mean([r["real_si"] for r in perm_results.values()]) \
        if perm_results else 0
    click.echo(f"  Average SI: {avg_si:.2f} (>1 = concentrated)")

    click.echo(f"\n  {'=' * 40}")
    if n_sig >= n_tested // 2:
        click.echo("  VERDICT: STRONG section specificity")
        click.echo("  Domain terms concentrate in expected sections")
    elif n_sig > 0:
        click.echo("  VERDICT: PARTIAL section specificity")
        click.echo("  Some categories show section concentration")
    else:
        click.echo("  VERDICT: No significant section specificity")
    click.echo(f"  {'=' * 40}")
