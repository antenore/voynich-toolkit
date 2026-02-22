"""Domain-specific lexicon test.

Tests whether decoded words from specific manuscript sections match
domain-specific lexicon entries at a higher rate than expected.

For each domain (botanical, astronomical, medical, balneological):
1. Collect domain-tagged consonantal forms from the curated lexicon
2. Decode all EVA words per manuscript section
3. Count how many decoded words match the domain lexicon
4. Chi-square test + permutation test for section-specificity

CLI: voynich --force domain-lexicon-test
"""

from __future__ import annotations

import json
import random
from collections import Counter

import click
import numpy as np
from scipy.stats import chi2 as chi2_dist

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew
from .full_decode import SECTION_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Domain → expected sections mapping
# =====================================================================

DOMAIN_EXPECTED_SECTIONS = {
    "botanical": ["H", "P"],       # herbal, pharmaceutical
    "astronomical": ["S", "Z", "C"],  # astronomical, zodiac, cosmological
    "medical": ["B", "P"],         # balneological, pharmaceutical
    "balneological": ["B"],        # balneological
}

DOMAIN_LABELS = {
    "botanical": "Botanical",
    "astronomical": "Astronomical",
    "medical": "Medical/Anatomical",
    "balneological": "Balneological",
}


# =====================================================================
# Load domain lexicons
# =====================================================================


def load_domain_lexicons(config: ToolkitConfig) -> dict[str, set[str]]:
    """Load domain-tagged consonantal forms from lexicon.json.

    Returns dict {domain: set(consonantal_forms)}.
    Uses ONLY curated entries (not auto-classified STEPBible) for clean test.
    """
    lex_path = config.lexicon_dir / "lexicon.json"
    if not lex_path.exists():
        raise click.ClickException(f"Lexicon not found: {lex_path}")

    with open(lex_path) as f:
        lex = json.load(f)

    domain_sets = {}
    by_domain = lex.get("by_domain", {})

    for domain in DOMAIN_EXPECTED_SECTIONS:
        entries = by_domain.get(domain, [])
        if not isinstance(entries, list):
            continue
        # Use ALL entries for this domain (curated + auto-classified)
        forms = set()
        for entry in entries:
            cons = entry.get("consonants", "")
            if cons and len(cons) >= 2:
                forms.add(cons)
        domain_sets[domain] = forms

    return domain_sets


def load_curated_only(config: ToolkitConfig) -> dict[str, set[str]]:
    """Load ONLY curated domain entries (manually verified).

    More conservative than load_domain_lexicons: excludes auto-classified
    STEPBible entries for a cleaner test.
    """
    lex_path = config.lexicon_dir / "lexicon.json"
    with open(lex_path) as f:
        lex = json.load(f)

    curated_sources = {"Curato-Botanico", "Curato-IbnEzra",
                       "Curato-Medico", "Curato-Generale",
                       "Curato-Balneologico"}

    domain_sets = {}
    by_domain = lex.get("by_domain", {})

    for domain in DOMAIN_EXPECTED_SECTIONS:
        entries = by_domain.get(domain, [])
        if not isinstance(entries, list):
            continue
        forms = set()
        for entry in entries:
            if entry.get("source", "") in curated_sources:
                cons = entry.get("consonants", "")
                if cons and len(cons) >= 2:
                    forms.add(cons)
        domain_sets[domain] = forms

    return domain_sets


# =====================================================================
# Count domain matches per section
# =====================================================================


def count_domain_matches(pages: list[dict],
                         domain_lexicons: dict[str, set[str]],
                         min_len: int = 3) -> dict:
    """Count domain-specific matches per manuscript section.

    Returns: {
        section: {
            'n_words': int,
            'n_decoded': int,
            'domain_matches': {domain: n_matches},
        }
    }
    """
    by_section: dict[str, dict] = {}

    for page in pages:
        sec = page.get("section", "?")
        if sec not in by_section:
            by_section[sec] = {"words": [], "decoded": []}
        words = page.get("words", [])
        by_section[sec]["words"].extend(words)
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= min_len:
                by_section[sec]["decoded"].append(heb)

    results = {}
    for sec, data in sorted(by_section.items()):
        decoded = data["decoded"]
        domain_matches = {}
        for domain, lex_set in domain_lexicons.items():
            n_match = sum(1 for h in decoded if h in lex_set)
            domain_matches[domain] = n_match
        results[sec] = {
            "n_words": len(data["words"]),
            "n_decoded": len(decoded),
            "domain_matches": domain_matches,
        }

    return results


# =====================================================================
# Chi-square test for domain concentration
# =====================================================================


def chi_square_concentration(section_counts: dict,
                             domain: str,
                             expected_sections: list[str]) -> dict:
    """Chi-square test: is domain concentrated in expected sections?

    H0: domain matches are distributed proportionally to total decoded words.
    Returns: chi2, p-value, observed vs expected, concentration ratio.
    """
    # Build observed and expected
    sections = sorted(section_counts.keys())
    total_decoded = sum(section_counts[s]["n_decoded"] for s in sections)
    total_domain = sum(section_counts[s]["domain_matches"][domain]
                       for s in sections)

    if total_decoded == 0 or total_domain == 0:
        return {"chi2": 0, "p_value": 1.0, "concentration_ratio": 0,
                "total_matches": 0}

    # Observed and expected per section
    obs = []
    exp = []
    section_labels = []
    for sec in sections:
        n_dec = section_counts[sec]["n_decoded"]
        n_match = section_counts[sec]["domain_matches"][domain]
        expected = total_domain * (n_dec / total_decoded)
        if expected > 0:  # skip empty sections
            obs.append(n_match)
            exp.append(expected)
            section_labels.append(sec)

    obs = np.array(obs, dtype=float)
    exp = np.array(exp, dtype=float)

    # Chi-square statistic
    # Use only cells with expected > 0
    mask = exp > 0
    chi2_stat = float(np.sum((obs[mask] - exp[mask])**2 / exp[mask]))
    df = max(int(np.sum(mask)) - 1, 1)
    p_value = float(1 - chi2_dist.cdf(chi2_stat, df))

    # Concentration ratio: observed in expected / expected in expected
    in_expected = sum(section_counts[s]["domain_matches"][domain]
                      for s in sections if s in expected_sections)
    words_in_expected = sum(section_counts[s]["n_decoded"]
                            for s in sections if s in expected_sections)
    baseline_ratio = words_in_expected / total_decoded if total_decoded else 0
    observed_ratio = in_expected / total_domain if total_domain else 0
    concentration = observed_ratio / baseline_ratio if baseline_ratio > 0 else 0

    per_section = {}
    for i, sec in enumerate(section_labels):
        per_section[sec] = {
            "observed": int(obs[i]),
            "expected": round(exp[i], 1),
            "residual": round((obs[i] - exp[i]) / max(np.sqrt(exp[i]), 1), 2),
        }

    return {
        "chi2": round(chi2_stat, 2),
        "df": df,
        "p_value": round(p_value, 6),
        "total_matches": total_domain,
        "in_expected_sections": in_expected,
        "concentration_ratio": round(concentration, 3),
        "observed_ratio": round(observed_ratio, 4),
        "baseline_ratio": round(baseline_ratio, 4),
        "per_section": per_section,
    }


# =====================================================================
# Permutation test for domain concentration
# =====================================================================


def permutation_concentration(pages: list[dict],
                              domain_lexicon: set[str],
                              expected_sections: list[str],
                              n_perms: int = 1000,
                              seed: int = 42,
                              min_len: int = 3) -> dict:
    """Permutation test: shuffle section labels, measure concentration.

    Test statistic: number of domain matches in expected sections.
    H0: section labels are independent of domain match distribution.
    """
    rng = random.Random(seed)

    # Decode all words, track section assignment
    page_decoded = []
    for page in pages:
        sec = page.get("section", "?")
        decoded = []
        for w in page.get("words", []):
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= min_len:
                decoded.append(heb)
        page_decoded.append({"section": sec, "decoded": decoded})

    # Real statistic: matches in expected sections
    real_in_expected = 0
    real_total = 0
    for pd in page_decoded:
        for h in pd["decoded"]:
            if h in domain_lexicon:
                real_total += 1
                if pd["section"] in expected_sections:
                    real_in_expected += 1

    if real_total == 0:
        return {"real_in_expected": 0, "real_total": 0,
                "z_score": 0, "p_value": 1.0}

    # Permutation: shuffle section assignments among pages
    sections = [pd["section"] for pd in page_decoded]
    null_counts = []

    for _ in range(n_perms):
        shuffled = list(sections)
        rng.shuffle(shuffled)
        count = 0
        for i, pd in enumerate(page_decoded):
            if shuffled[i] in expected_sections:
                for h in pd["decoded"]:
                    if h in domain_lexicon:
                        count += 1
        null_counts.append(count)

    null_arr = np.array(null_counts, dtype=float)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))

    z_score = (real_in_expected - null_mean) / null_std if null_std > 0 else 0
    p_value = float(np.mean(null_arr >= real_in_expected))

    return {
        "real_in_expected": real_in_expected,
        "real_total": real_total,
        "real_concentration": round(real_in_expected / real_total, 4),
        "null_mean": round(null_mean, 1),
        "null_std": round(null_std, 2),
        "z_score": round(z_score, 2),
        "p_value": round(max(p_value, 1 / (n_perms + 1)), 4),
        "n_perms": n_perms,
    }


# =====================================================================
# Entry point
# =====================================================================


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Run domain-specific lexicon test."""
    out_json = config.stats_dir / "domain_lexicon_test.json"
    out_txt = config.stats_dir / "domain_lexicon_test.txt"

    if out_json.exists() and not force:
        click.echo(f"  Output exists: {out_json} (use --force)")
        return

    config.ensure_dirs()
    print_header("Domain-Specific Lexicon Test")

    # Parse corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} words, {len(pages)} pages")

    # Load domain lexicons
    print_step("Loading domain lexicons...")
    domain_lexicons = load_domain_lexicons(config)
    for domain, forms in sorted(domain_lexicons.items()):
        click.echo(f"    {domain}: {len(forms)} forms")

    curated_lexicons = load_curated_only(config)
    for domain, forms in sorted(curated_lexicons.items()):
        click.echo(f"    {domain} (curated only): {len(forms)} forms")

    # Count domain matches per section
    print_step("Counting domain matches per section...")
    section_counts = count_domain_matches(pages, domain_lexicons)

    for sec in sorted(section_counts.keys()):
        sc = section_counts[sec]
        sec_name = SECTION_NAMES.get(sec, sec)
        matches_str = ", ".join(
            f"{d}={sc['domain_matches'][d]}"
            for d in sorted(sc["domain_matches"])
            if sc["domain_matches"][d] > 0
        )
        click.echo(f"    {sec} ({sec_name}): {sc['n_decoded']} decoded"
                    f" | {matches_str or 'no matches'}")

    # Chi-square tests
    print_step("Chi-square concentration tests...")
    chi_results = {}
    for domain, expected in DOMAIN_EXPECTED_SECTIONS.items():
        label = DOMAIN_LABELS.get(domain, domain)
        chi = chi_square_concentration(section_counts, domain, expected)
        chi_results[domain] = chi
        sig = "***" if chi["p_value"] < 0.001 else \
              "**" if chi["p_value"] < 0.01 else \
              "*" if chi["p_value"] < 0.05 else "ns"
        click.echo(f"    {label}: chi2={chi['chi2']:.1f} "
                    f"p={chi['p_value']:.4f} {sig} "
                    f"conc={chi['concentration_ratio']:.2f} "
                    f"({chi['in_expected_sections']}/{chi['total_matches']} "
                    f"in expected)")

    # Permutation tests (with curated lexicon for cleaner test)
    n_perms = 1000
    print_step(f"Permutation tests ({n_perms} permutations)...")
    perm_results = {}
    for domain, expected in DOMAIN_EXPECTED_SECTIONS.items():
        label = DOMAIN_LABELS.get(domain, domain)
        lex_set = domain_lexicons[domain]
        if not lex_set:
            click.echo(f"    {label}: no lexicon forms, skip")
            continue
        perm = permutation_concentration(
            pages, lex_set, expected,
            n_perms=n_perms, seed=42)
        perm_results[domain] = perm
        sig = "***" if perm["p_value"] < 0.001 else \
              "**" if perm["p_value"] < 0.01 else \
              "*" if perm["p_value"] < 0.05 else "ns"
        click.echo(f"    {label}: z={perm['z_score']:.2f} "
                    f"p={perm['p_value']:.4f} {sig} "
                    f"({perm['real_in_expected']}/{perm['real_total']} "
                    f"in expected, null mean={perm['null_mean']:.0f})")

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("  DOMAIN LEXICON TEST — SUMMARY")
    click.echo(f"{'='*60}")

    any_significant = False
    for domain in DOMAIN_EXPECTED_SECTIONS:
        label = DOMAIN_LABELS.get(domain, domain)
        chi = chi_results.get(domain, {})
        perm = perm_results.get(domain, {})
        expected = DOMAIN_EXPECTED_SECTIONS[domain]
        exp_names = [SECTION_NAMES.get(s, s) for s in expected]

        click.echo(f"\n  {label} → expected in {', '.join(exp_names)}:")
        click.echo(f"    Matches: {chi.get('total_matches', 0)}")
        click.echo(f"    Chi2: {chi.get('chi2', 0):.1f} "
                    f"(p={chi.get('p_value', 1):.4f})")
        click.echo(f"    Concentration: {chi.get('concentration_ratio', 0):.2f}")
        if perm:
            click.echo(f"    Perm z: {perm.get('z_score', 0):.2f} "
                        f"(p={perm.get('p_value', 1):.4f})")
            if perm.get("p_value", 1) < 0.05:
                any_significant = True

    click.echo(f"\n  Any domain significant: {any_significant}")

    # Save JSON
    report = {
        "domain_sizes": {d: len(s) for d, s in domain_lexicons.items()},
        "curated_sizes": {d: len(s) for d, s in curated_lexicons.items()},
        "section_counts": section_counts,
        "chi_square": chi_results,
        "permutation": perm_results,
        "expected_sections": {d: [SECTION_NAMES.get(s, s) for s in secs]
                              for d, secs in DOMAIN_EXPECTED_SECTIONS.items()},
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"\n  JSON: {out_json}")

    # Save TXT
    lines = [
        "Domain-Specific Lexicon Test",
        "=" * 60,
        "",
    ]
    for domain in DOMAIN_EXPECTED_SECTIONS:
        label = DOMAIN_LABELS.get(domain, domain)
        chi = chi_results.get(domain, {})
        perm = perm_results.get(domain, {})
        expected = DOMAIN_EXPECTED_SECTIONS[domain]
        exp_names = [SECTION_NAMES.get(s, s) for s in expected]

        lines.append(f"{label} → {', '.join(exp_names)}")
        lines.append(f"  Lexicon size: {len(domain_lexicons.get(domain, set()))}")
        lines.append(f"  Total matches: {chi.get('total_matches', 0)}")
        lines.append(f"  In expected: {chi.get('in_expected_sections', 0)}")
        lines.append(f"  Concentration: {chi.get('concentration_ratio', 0):.2f}")
        lines.append(f"  Chi2: {chi.get('chi2', 0):.1f} "
                      f"(p={chi.get('p_value', 1):.4f})")
        if perm:
            lines.append(f"  Perm z: {perm.get('z_score', 0):.2f} "
                          f"(p={perm.get('p_value', 1):.4f})")
        lines.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    click.echo(f"  TXT: {out_txt}")
