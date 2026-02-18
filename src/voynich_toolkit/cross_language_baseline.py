"""
Cross-language baseline: prove Hebrew mapping beats other Semitic languages.

Phase 8C: decode corpus with the real mapping, match against:
  1. Hebrew lexicon (enriched)
  2. Aramaic lexicon (from Jastrow 'Ch.' entries)
  3. Random consonantal strings (null baseline)

If Hebrew >> Aramaic >> Random → strong validation signal.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step, timer


# =====================================================================
# 20-char mapping (from Phase 7 + Phase 9 B3)
# =====================================================================

MAPPING = {
    'a': 'y', 'c': 'A', 'd': 'r', 'e': 'p', 'f': 'l',
    'g': 'X', 'h': 'E', 'k': 't', 'l': 'm', 'm': 'g',
    'n': 'd', 'o': 'w', 'p': 'l', 'r': 'h', 's': 'n',
    't': 'J', 'y': 'S',
}
II_HEBREW = 'h'
I_HEBREW = 'r'
CH_HEBREW = 'k'  # ch digraph → kaf
INITIAL_D_HEBREW = 'b'  # n at Hebrew-initial → bet (Phase 9 B2)
INITIAL_H_HEBREW = 's'  # he at Hebrew-initial → samekh (l/r allography)
DIRECTION = 'rtl'


def preprocess_eva(word):
    """Replace ch→token, ii→token, standalone i→token, strip q-prefix."""
    w = word
    # ch digraph first
    w = w.replace('ch', '\x03')

    prefix = ''
    if w.startswith('qo'):
        prefix = 'qo'
        w = w[2:]
    elif w.startswith('q') and len(w) > 1:
        prefix = 'q'
        w = w[1:]

    w = re.sub(r'i{3,}', lambda m: '\x01' * (len(m.group()) // 2) +
               ('\x02' if len(m.group()) % 2 else ''), w)
    w = w.replace('ii', '\x01')
    w = w.replace('i', '\x02')
    return prefix, w


def decode_to_hebrew(eva_word):
    """Decode EVA word to Hebrew consonantal string."""
    _, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed)) if DIRECTION == 'rtl' else list(processed)

    parts = []
    for ch in chars:
        if ch == '\x01':
            parts.append(II_HEBREW)
        elif ch == '\x02':
            parts.append(I_HEBREW)
        elif ch == '\x03':
            parts.append(CH_HEBREW)
        elif ch in MAPPING:
            parts.append(MAPPING[ch])
        else:
            return None  # unmapped char

    # Positional split: dalet at word-initial → bet (Phase 9 B2)
    if parts and parts[0] == 'd':
        parts[0] = INITIAL_D_HEBREW

    # Positional split: he at word-initial → samekh (l/r allography)
    if parts and parts[0] == 'h':
        parts[0] = INITIAL_H_HEBREW

    return ''.join(parts)


# =====================================================================
# Random lexicon generation
# =====================================================================

def generate_random_lexicon(hebrew_set, seed=42):
    """Generate a random consonantal lexicon matching Hebrew stats.

    Preserves the length distribution and consonant frequency of the
    real Hebrew lexicon, but shuffles to create meaningless strings.
    """
    rng = random.Random(seed)
    forms = list(hebrew_set)

    # Measure length distribution
    lengths = [len(f) for f in forms]

    # Measure consonant frequency
    all_chars = ''.join(forms)
    char_freq = Counter(all_chars)
    total = sum(char_freq.values())
    char_weights = {c: n / total for c, n in char_freq.items()}
    chars = list(char_weights.keys())
    weights = [char_weights[c] for c in chars]

    # Generate random forms with same length distribution
    random_forms = set()
    for length in lengths:
        form = ''.join(rng.choices(chars, weights=weights, k=length))
        random_forms.add(form)

    return random_forms


# =====================================================================
# Match scoring
# =====================================================================

@timer
def score_against_lexicon(decoded_words, lexicon_set, min_len=3):
    """Count how many decoded words match a given lexicon.

    Returns: (n_matched, n_total, match_rate, matched_words_sample)
    """
    n_total = 0
    n_matched = 0
    matched_sample = Counter()

    for heb_word in decoded_words:
        if heb_word is None or len(heb_word) < min_len:
            continue
        n_total += 1
        if heb_word in lexicon_set:
            n_matched += 1
            matched_sample[heb_word] += 1

    rate = n_matched / max(n_total, 1)
    top_matches = matched_sample.most_common(30)

    return n_matched, n_total, rate, top_matches


# =====================================================================
# Statistical comparison
# =====================================================================

def compare_lexicons(results_dict):
    """Compute z-scores between Hebrew and each control lexicon.

    Args:
        results_dict: {lexicon_name: (n_matched, n_total, rate, top)}

    Returns: dict with pairwise z-scores
    """
    comparisons = {}
    hebrew_data = results_dict.get("hebrew")
    if not hebrew_data:
        return comparisons

    heb_matches, heb_total, heb_rate, _ = hebrew_data

    for name, (matches, total, rate, _) in results_dict.items():
        if name == "hebrew":
            continue

        # Two-proportion z-test
        n1, n2 = heb_total, total
        p1, p2 = heb_rate, rate
        if n1 == 0 or n2 == 0:
            continue

        p_pool = (heb_matches + matches) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            z = float('inf') if p1 > p2 else 0.0
        else:
            se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
            z = (p1 - p2) / se if se > 0 else float('inf')

        # One-tailed p-value (Hebrew > other)
        from scipy.stats import norm
        p_value = 1 - norm.cdf(z) if z != float('inf') else 0.0

        comparisons[f"hebrew_vs_{name}"] = {
            "hebrew_rate": round(heb_rate, 4),
            f"{name}_rate": round(rate, 4),
            "rate_ratio": round(heb_rate / max(rate, 1e-6), 2),
            "z_score": round(float(z), 2),
            "p_value": round(float(p_value), 6),
            "significant_001": bool(p_value < 0.001),
        }

    return comparisons


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Cross-language baseline comparison."""
    report_path = config.stats_dir / "cross_language_report.json"

    if report_path.exists() and not force:
        click.echo("  Cross-language report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 8C — Cross-Language Baseline")

    # 1. Parse EVA text
    print_step("Parsing EVA text...")
    from .word_structure import parse_eva_words
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words")

    # 2. Decode all words to Hebrew
    print_step("Decoding all words to Hebrew consonants...")
    decoded_words = []
    for page in eva_data["pages"]:
        for word in page["words"]:
            heb = decode_to_hebrew(word)
            if heb:
                decoded_words.append(heb)
    click.echo(f"    {len(decoded_words)} words decoded")

    # 3. Load Hebrew lexicon (enriched if available, else base)
    print_step("Loading lexicons...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    base_path = config.lexicon_dir / "lexicon.json"

    if enriched_path.exists():
        with open(enriched_path) as f:
            hlex = json.load(f)
        click.echo(f"    Hebrew (enriched): "
                   f"{len(hlex['all_consonantal_forms'])} forms")
    elif base_path.exists():
        with open(base_path) as f:
            hlex = json.load(f)
        click.echo(f"    Hebrew (base): "
                   f"{len(hlex['all_consonantal_forms'])} forms")
    else:
        raise click.ClickException(
            "No Hebrew lexicon found. Run: voynich prepare-lexicon")

    hebrew_set = set(hlex["all_consonantal_forms"])

    # 4. Load Aramaic lexicon (from enrich-lexicon)
    aramaic_path = config.lexicon_dir / "aramaic_lexicon.json"
    aramaic_set = set()
    if aramaic_path.exists():
        with open(aramaic_path) as f:
            aram_data = json.load(f)
        aramaic_set = set(aram_data.get("all_consonantal_forms", []))
        click.echo(f"    Aramaic: {len(aramaic_set)} forms")
    else:
        click.echo("    Aramaic: not available (run enrich-lexicon first)")

    # 5. Generate random lexicon
    print_step("Generating random consonantal lexicon...")
    random_set = generate_random_lexicon(hebrew_set, seed=42)
    click.echo(f"    Random: {len(random_set)} forms")

    # 6. Score against each lexicon
    print_step("Scoring against Hebrew lexicon...")
    heb_result = score_against_lexicon(decoded_words, hebrew_set)
    click.echo(f"    Hebrew: {heb_result[0]}/{heb_result[1]} "
               f"({heb_result[2]*100:.1f}%)")

    results_dict = {"hebrew": heb_result}

    if aramaic_set:
        print_step("Scoring against Aramaic lexicon...")
        aram_result = score_against_lexicon(decoded_words, aramaic_set)
        click.echo(f"    Aramaic: {aram_result[0]}/{aram_result[1]} "
                   f"({aram_result[2]*100:.1f}%)")
        results_dict["aramaic"] = aram_result

    print_step("Scoring against random lexicon...")
    rand_result = score_against_lexicon(decoded_words, random_set)
    click.echo(f"    Random: {rand_result[0]}/{rand_result[1]} "
               f"({rand_result[2]*100:.1f}%)")
    results_dict["random"] = rand_result

    # 7. Statistical comparisons
    print_step("Computing statistical comparisons...")
    comparisons = compare_lexicons(results_dict)

    # 8. Save report
    print_step("Saving report...")
    report = {
        "n_decoded_words": len(decoded_words),
        "lexicons": {},
        "comparisons": comparisons,
    }

    for name, (matches, total, rate, top) in results_dict.items():
        report["lexicons"][name] = {
            "n_forms": len(hebrew_set) if name == "hebrew"
                       else len(aramaic_set) if name == "aramaic"
                       else len(random_set),
            "n_matched": matches,
            "n_total": total,
            "match_rate": round(rate, 4),
            "match_pct": round(rate * 100, 2),
            "top_matches": [
                {"word": w, "count": c} for w, c in top
            ],
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")

    # 9. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  CROSS-LANGUAGE BASELINE RESULTS")
    click.echo(f"{'=' * 60}")

    click.echo(f"\n  {'Lexicon':12s} {'Forms':>8s} {'Matches':>8s} "
               f"{'Rate':>8s}")
    click.echo(f"  {'-'*44}")
    for name in ["hebrew", "aramaic", "random"]:
        if name in results_dict:
            matches, total, rate, _ = results_dict[name]
            n_forms = (len(hebrew_set) if name == "hebrew"
                       else len(aramaic_set) if name == "aramaic"
                       else len(random_set))
            click.echo(f"  {name:12s} {n_forms:8,d} {matches:8,d} "
                       f"{rate*100:7.1f}%")

    click.echo(f"\n  Statistical comparisons:")
    for comp_name, comp_data in comparisons.items():
        sig = "***" if comp_data["significant_001"] else ""
        click.echo(f"    {comp_name}: z={comp_data['z_score']:.1f}, "
                   f"p={comp_data['p_value']:.6f} {sig}")
        click.echo(f"      rate ratio: {comp_data['rate_ratio']:.1f}x")

    # Verdict — use z-score for properly matched controls
    click.echo(f"\n  {'=' * 40}")
    heb_vs_rand = comparisons.get("hebrew_vs_random", {})
    heb_vs_aram = comparisons.get("hebrew_vs_aramaic", {})
    z_rand = heb_vs_rand.get("z_score", 0)
    z_aram = heb_vs_aram.get("z_score", 0)

    if z_rand > 5:
        click.echo("  VERDICT: Hebrew STRONGLY exceeds random baseline "
                   f"(z={z_rand:.1f})")
    elif z_rand > 2:
        click.echo("  VERDICT: Hebrew significantly exceeds random "
                   f"(z={z_rand:.1f})")
    else:
        click.echo("  VERDICT: Hebrew does not clearly exceed random "
                   f"(z={z_rand:.1f})")

    if aramaic_set and z_aram > 5:
        click.echo(f"  Hebrew STRONGLY exceeds Aramaic (z={z_aram:.1f})")
    elif aramaic_set and z_aram > 2:
        click.echo(f"  Hebrew significantly exceeds Aramaic (z={z_aram:.1f})")
    click.echo(f"  {'=' * 40}")
