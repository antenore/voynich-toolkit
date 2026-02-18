"""
Mater Lectionis Tolerance — Phase 9 A2

Hebrew matres lectionis (he, vav, yod) serve as vowel indicators
in plene spelling. The EVA cipher may encode these, causing d=1
mismatches when matching against consonantal lexicons.

Two-directional tolerance:
  A. Strip matres from DECODED words → match original lexicon
     (decoded has extra vowel letters the lexicon doesn't)
  B. Strip matres from LEXICON entries → build stripped index
     (lexicon has vowel letters that decoded text lacks)

If mater tolerance helps Hebrew MORE than Random → real signal.
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew, generate_random_lexicon
from .utils import print_header, print_step, timer
from .word_structure import parse_eva_words


# Hebrew matres lectionis (in ASCII encoding)
MATRES = frozenset({'h', 'w', 'y'})  # he, vav, yod

# Italian equivalents (through HEBREW_TO_ITALIAN mapping)
MATRES_ITALIAN = frozenset({'e', 'o', 'i'})  # he→e, vav→o, yod→i


# =====================================================================
# Variant generation
# =====================================================================

def generate_stripped_variants(word, max_strip=2, matres=MATRES):
    """Generate variants by removing matres from non-initial positions.

    Rules:
    - Never strip initial consonant (position 0)
    - Strip non-initial he(h), vav(w), yod(y)
    - Max `max_strip` removals per word
    - Don't reduce word below 2 characters
    """
    if len(word) < 3:
        return {word}

    strip_positions = [i for i in range(1, len(word))
                       if word[i] in matres]

    if not strip_positions:
        return {word}

    variants = {word}
    for n in range(1, min(max_strip + 1, len(strip_positions) + 1)):
        for combo in combinations(strip_positions, n):
            v = ''.join(c for i, c in enumerate(word) if i not in combo)
            if len(v) >= 2:
                variants.add(v)

    return variants


def build_stripped_index(lexicon_set, max_strip=2):
    """Build reverse index: {stripped_form: set(original_forms)}.

    For each lexicon entry, generate stripped variants and map
    them back to the original form.  This enables direction B:
    matching decoded words against lexicon entries that lost matres.
    """
    index = defaultdict(set)
    for form in lexicon_set:
        for variant in generate_stripped_variants(form, max_strip):
            if variant != form:
                index[variant].add(form)
    return dict(index)


# =====================================================================
# Scoring
# =====================================================================

@timer
def score_bidirectional(decoded_words, lexicon_set, stripped_index,
                        max_strip=2, min_len=3):
    """Score decoded words against lexicon with bidirectional mater tolerance.

    Returns dict with:
      exact: exact match count
      dir_a: matches found by stripping matres from decoded word
      dir_b: matches found via stripped lexicon index
      total: total words scored
      rate_exact: exact / total
      rate_tolerant: (exact + dir_a + dir_b) / total
      recovered_a: sample of direction A recoveries
      recovered_b: sample of direction B recoveries
      mater_stats: Counter of which matres were stripped
    """
    n_total = 0
    n_exact = 0
    n_dir_a = 0
    n_dir_b = 0
    recovered_a = []
    recovered_b = []
    mater_stripped = Counter()  # char → count
    position_stripped = Counter()  # 'medial' or 'final' → count

    for heb_word in decoded_words:
        if heb_word is None or len(heb_word) < min_len:
            continue
        n_total += 1

        if heb_word in lexicon_set:
            n_exact += 1
            continue

        # Direction A: strip matres from decoded
        found_a = False
        variants = generate_stripped_variants(heb_word, max_strip)
        for v in variants:
            if v != heb_word and v in lexicon_set:
                n_dir_a += 1
                found_a = True
                if len(recovered_a) < 50:
                    recovered_a.append((heb_word, v))
                # Track which chars were stripped
                for i, c in enumerate(heb_word):
                    if c in MATRES:
                        stripped_here = c not in v or (
                            heb_word[:i].count(c) + heb_word[i+1:].count(c)
                            >= v.count(c))
                        # Simpler: count char frequencies
                for c in MATRES:
                    diff = heb_word.count(c) - v.count(c)
                    if diff > 0:
                        mater_stripped[c] += diff
                        # Check position
                        for i in range(1, len(heb_word)):
                            if heb_word[i] == c:
                                pos = 'final' if i == len(heb_word)-1 else 'medial'
                                position_stripped[pos] += 1
                break

        if found_a:
            continue

        # Direction B: decoded matches a stripped lexicon form
        if heb_word in stripped_index:
            n_dir_b += 1
            originals = stripped_index[heb_word]
            if len(recovered_b) < 50:
                recovered_b.append((heb_word, list(originals)[:3]))

    rate_exact = n_exact / max(n_total, 1)
    rate_tolerant = (n_exact + n_dir_a + n_dir_b) / max(n_total, 1)

    return {
        'exact': n_exact,
        'dir_a': n_dir_a,
        'dir_b': n_dir_b,
        'total': n_total,
        'rate_exact': rate_exact,
        'rate_tolerant': rate_tolerant,
        'improvement_abs': rate_tolerant - rate_exact,
        'improvement_pct': round((rate_tolerant - rate_exact) * 100, 2),
        'recovered_a': recovered_a,
        'recovered_b': recovered_b,
        'mater_stripped': dict(mater_stripped),
        'position_stripped': dict(position_stripped),
    }


# =====================================================================
# Anchor / Zodiac / Plant impact
# =====================================================================

def _levenshtein_distance(a, b):
    """Simple Levenshtein distance (no external dep needed)."""
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        # Fallback
        if len(a) < len(b):
            return _levenshtein_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]


def analyze_anchor_impact(anchor_report_path, max_strip=2):
    """Check d=1→d=0 conversions in anchor report via mater tolerance."""
    if not anchor_report_path.exists():
        return None

    with open(anchor_report_path) as f:
        report = json.load(f)

    conversions = []
    by_category = report.get("by_category", {})

    for cat_id, cat_data in by_category.items():
        for m in cat_data.get("matches", []):
            target = m["normalized"]
            # Detect Hebrew vs Italian: Hebrew uses ASCII encoding with
            # uppercase (S, X, J, E, A) for special chars
            is_hebrew = any(c in 'SXJEAC' for c in target)
            matres = MATRES if is_hebrew else MATRES_ITALIAN

            for df in m.get("decoded_forms", []):
                if df.get("distance") != 1:
                    continue
                decoded = df["decoded_word"]
                count = df.get("total_count", 0)

                # Direction A: strip from decoded
                for v in generate_stripped_variants(decoded, max_strip, matres):
                    if v != decoded and _levenshtein_distance(v, target) == 0:
                        conversions.append({
                            "category": cat_id,
                            "anchor": m["anchor"],
                            "target": target,
                            "decoded": decoded,
                            "stripped": v,
                            "direction": "A",
                            "count": count,
                        })
                        break
                else:
                    # Direction B: strip from target
                    for v in generate_stripped_variants(target, max_strip, matres):
                        if v != target and _levenshtein_distance(decoded, v) == 0:
                            conversions.append({
                                "category": cat_id,
                                "anchor": m["anchor"],
                                "target": target,
                                "decoded": decoded,
                                "stripped_target": v,
                                "direction": "B",
                                "count": count,
                            })
                            break

    return conversions


def analyze_zodiac_impact(zodiac_report_path, max_strip=2):
    """Check d=1→d=0 conversions in zodiac terms via mater tolerance."""
    if not zodiac_report_path.exists():
        return None

    with open(zodiac_report_path) as f:
        report = json.load(f)

    conversions = []
    results = report.get("results", {})

    for category, cat_data in results.items():
        for match in cat_data.get("matches", []):
            target = match.get("normalized", "")
            if not target:
                continue

            for hit in match.get("zodiac_hits", []):
                decoded = hit.get("decoded", "")
                dist = hit.get("distance", 99)
                if dist != 1 or not decoded:
                    continue

                # Direction A: strip from decoded
                for v in generate_stripped_variants(decoded, max_strip):
                    if v != decoded and _levenshtein_distance(v, target) == 0:
                        conversions.append({
                            "term": match.get("word", ""),
                            "canonical": match.get("canonical", ""),
                            "target": target,
                            "decoded": decoded,
                            "stripped": v,
                            "direction": "A",
                            "folio": hit.get("folio", ""),
                        })
                        break
                else:
                    # Direction B: strip from target
                    for v in generate_stripped_variants(target, max_strip):
                        if v != target and _levenshtein_distance(decoded, v) == 0:
                            conversions.append({
                                "term": match.get("word", ""),
                                "canonical": match.get("canonical", ""),
                                "target": target,
                                "decoded": decoded,
                                "stripped_target": v,
                                "direction": "B",
                                "folio": hit.get("folio", ""),
                            })
                            break

    return conversions


def analyze_plant_impact(plant_report_path, max_strip=2):
    """Check d=1→d=0 conversions in plant matches via mater tolerance."""
    if not plant_report_path.exists():
        return None

    with open(plant_report_path) as f:
        report = json.load(f)

    conversions = []
    for hit in report.get("all_hits", []):
        if hit.get("distance") != 1:
            continue

        decoded = hit.get("decoded", "")
        target = hit.get("plant_match", "")
        if not decoded or not target:
            continue

        # Plants are Italian/Latin — use Italian matres
        for v in generate_stripped_variants(decoded, max_strip, MATRES_ITALIAN):
            if v != decoded and _levenshtein_distance(v, target) == 0:
                conversions.append({
                    "plant": target,
                    "decoded": decoded,
                    "stripped": v,
                    "direction": "A",
                    "folio": hit.get("folio", ""),
                    "gloss": hit.get("gloss", ""),
                })
                break
        else:
            for v in generate_stripped_variants(target, max_strip, MATRES_ITALIAN):
                if v != target and _levenshtein_distance(decoded, v) == 0:
                    conversions.append({
                        "plant": target,
                        "decoded": decoded,
                        "stripped_target": v,
                        "direction": "B",
                        "folio": hit.get("folio", ""),
                        "gloss": hit.get("gloss", ""),
                    })
                    break

    return conversions


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Mater lectionis tolerance analysis."""
    report_path = config.stats_dir / "mater_lectionis_report.json"

    if report_path.exists() and not force:
        click.echo("  Mater lectionis report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 9 A2 — Mater Lectionis Tolerance")

    # 1. Parse EVA and decode to Hebrew
    print_step("Parsing and decoding EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)

    decoded_words = []
    for page in eva_data["pages"]:
        for word in page["words"]:
            heb = decode_to_hebrew(word)
            if heb:
                decoded_words.append(heb)
    click.echo(f"    {len(decoded_words)} words decoded to Hebrew")

    # 2. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    base_path = config.lexicon_dir / "lexicon.json"
    lex_path = enriched_path if enriched_path.exists() else base_path
    if not lex_path.exists():
        raise click.ClickException("No Hebrew lexicon found.")

    with open(lex_path) as f:
        hlex = json.load(f)
    hebrew_set = set(hlex["all_consonantal_forms"])
    click.echo(f"    {len(hebrew_set)} Hebrew forms")

    # 3. Build stripped index for lexicon (direction B)
    print_step("Building mater-stripped lexicon index...")
    hebrew_stripped_idx = build_stripped_index(hebrew_set, max_strip=2)
    click.echo(f"    {len(hebrew_stripped_idx)} stripped forms indexed")

    # 4. Score Hebrew: exact vs mater-tolerant
    print_step("Scoring Hebrew: exact vs mater-tolerant...")
    heb_results = score_bidirectional(
        decoded_words, hebrew_set, hebrew_stripped_idx, max_strip=2)

    click.echo(f"    Exact:    {heb_results['exact']:,d}/{heb_results['total']:,d} "
               f"({heb_results['rate_exact']*100:.1f}%)")
    click.echo(f"    Dir A:    +{heb_results['dir_a']:,d} "
               f"(decoded has extra matres)")
    click.echo(f"    Dir B:    +{heb_results['dir_b']:,d} "
               f"(lexicon has matres decoded lacks)")
    click.echo(f"    Tolerant: {heb_results['exact']+heb_results['dir_a']+heb_results['dir_b']:,d}"
               f"/{heb_results['total']:,d} "
               f"({heb_results['rate_tolerant']*100:.1f}%)")
    click.echo(f"    Improvement: +{heb_results['improvement_pct']:.2f}%")

    # 5. Generate random baseline and score
    print_step("Scoring random baseline with same tolerance...")
    random_set = generate_random_lexicon(hebrew_set, seed=42)
    random_stripped_idx = build_stripped_index(random_set, max_strip=2)
    rand_results = score_bidirectional(
        decoded_words, random_set, random_stripped_idx, max_strip=2)

    click.echo(f"    Random exact:    {rand_results['rate_exact']*100:.1f}%")
    click.echo(f"    Random tolerant: {rand_results['rate_tolerant']*100:.1f}%")
    click.echo(f"    Random improvement: +{rand_results['improvement_pct']:.2f}%")

    # 6. Statistical comparison
    print_step("Statistical comparison...")
    heb_tol = heb_results['rate_tolerant']
    rand_tol = rand_results['rate_tolerant']
    n = heb_results['total']

    # Two-proportion z-test: Hebrew tolerant vs Random tolerant
    p_pool = (heb_tol * n + rand_tol * n) / (2 * n)
    if 0 < p_pool < 1:
        se = (p_pool * (1 - p_pool) * 2 / n) ** 0.5
        z_tolerant = (heb_tol - rand_tol) / se if se > 0 else float('inf')
    else:
        z_tolerant = float('inf')

    # Hebrew exact vs random exact (for comparison)
    heb_ex = heb_results['rate_exact']
    rand_ex = rand_results['rate_exact']
    p_pool_ex = (heb_ex * n + rand_ex * n) / (2 * n)
    if 0 < p_pool_ex < 1:
        se_ex = (p_pool_ex * (1 - p_pool_ex) * 2 / n) ** 0.5
        z_exact = (heb_ex - rand_ex) / se_ex if se_ex > 0 else float('inf')
    else:
        z_exact = float('inf')

    # Signal improvement: does mater tolerance increase the gap?
    delta_heb = heb_results['improvement_pct']
    delta_rand = rand_results['improvement_pct']
    signal_gain = delta_heb - delta_rand

    click.echo(f"    z-score (exact):    {z_exact:.1f}")
    click.echo(f"    z-score (tolerant): {z_tolerant:.1f}")
    click.echo(f"    Hebrew improvement: +{delta_heb:.2f}%")
    click.echo(f"    Random improvement: +{delta_rand:.2f}%")
    click.echo(f"    Signal gain: +{signal_gain:.2f}% "
               f"({'Hebrew benefits MORE' if signal_gain > 0 else 'No differential signal'})")

    # 7. Mater pattern analysis
    print_step("Analyzing mater patterns...")
    ms = heb_results.get('mater_stripped', {})
    ps = heb_results.get('position_stripped', {})
    click.echo(f"    Matres stripped (dir A): "
               f"he(h)={ms.get('h',0)}, vav(w)={ms.get('w',0)}, yod(y)={ms.get('y',0)}")
    click.echo(f"    Positions: final={ps.get('final',0)}, medial={ps.get('medial',0)}")

    # 8. Sample recovered words
    print_step("Sample recovered words (direction A)...")
    for orig, stripped in heb_results['recovered_a'][:15]:
        click.echo(f"    {orig:12s} -> {stripped:10s} (stripped mater)")

    print_step("Sample recovered words (direction B)...")
    for decoded, originals in heb_results['recovered_b'][:15]:
        orig_str = ', '.join(originals) if isinstance(originals, list) else str(originals)
        click.echo(f"    {decoded:12s} <- {orig_str} (lexicon has mater)")

    # 9. Anchor/Zodiac/Plant impact
    print_step("Analyzing anchor word impact...")
    anchor_path = config.stats_dir / "anchor_words_report.json"
    anchor_conv = analyze_anchor_impact(anchor_path)
    if anchor_conv is not None:
        total_occ = sum(c.get('count', 0) for c in anchor_conv)
        dir_a_conv = [c for c in anchor_conv if c['direction'] == 'A']
        dir_b_conv = [c for c in anchor_conv if c['direction'] == 'B']
        click.echo(f"    d=1 → d=0 conversions: {len(anchor_conv)} "
                   f"(A={len(dir_a_conv)}, B={len(dir_b_conv)})")
        click.echo(f"    Recovered occurrences: {total_occ}")
        for c in sorted(anchor_conv, key=lambda x: -x.get('count', 0))[:10]:
            direction = c['direction']
            if direction == 'A':
                click.echo(f"      {c['anchor']:15s} {c['decoded']:8s} -> "
                           f"{c['stripped']:8s} = {c['target']:8s} ({c['count']} occ)")
            else:
                click.echo(f"      {c['anchor']:15s} {c['decoded']:8s} = "
                           f"target-{c.get('stripped_target',''):8s} ({c['count']} occ)")

    print_step("Analyzing zodiac impact...")
    zodiac_path = config.stats_dir / "zodiac_test_report.json"
    zodiac_conv = analyze_zodiac_impact(zodiac_path)
    if zodiac_conv is not None:
        click.echo(f"    d=1 → d=0 conversions: {len(zodiac_conv)}")
        for c in zodiac_conv[:10]:
            click.echo(f"      {c['term']:15s} {c['decoded']:8s} "
                       f"({c['direction']}) folio={c['folio']}")

    print_step("Analyzing plant impact...")
    plant_path = config.stats_dir / "plant_search_report.json"
    plant_conv = analyze_plant_impact(plant_path)
    if plant_conv is not None:
        click.echo(f"    d=1 → d=0 conversions: {len(plant_conv)}")
        for c in plant_conv[:10]:
            direction = c['direction']
            if direction == 'A':
                click.echo(f"      {c['plant']:15s} {c['decoded']:8s} -> "
                           f"{c['stripped']:8s} [{c['folio']}]")
            else:
                click.echo(f"      {c['plant']:15s} {c['decoded']:8s} = "
                           f"target-{c.get('stripped_target',''):8s} [{c['folio']}]")

    # 10. Save report
    print_step("Saving report...")
    report = {
        "max_strip": 2,
        "matres": list(MATRES),
        "hebrew": {
            "exact": heb_results['exact'],
            "dir_a": heb_results['dir_a'],
            "dir_b": heb_results['dir_b'],
            "total": heb_results['total'],
            "rate_exact": round(heb_results['rate_exact'], 4),
            "rate_tolerant": round(heb_results['rate_tolerant'], 4),
            "improvement_pct": heb_results['improvement_pct'],
            "mater_stripped": heb_results['mater_stripped'],
            "position_stripped": heb_results['position_stripped'],
            "recovered_a_sample": [
                {"original": o, "stripped": s}
                for o, s in heb_results['recovered_a']
            ],
            "recovered_b_sample": [
                {"decoded": d, "lexicon_originals": list(o) if isinstance(o, list) else [str(o)]}
                for d, o in heb_results['recovered_b']
            ],
        },
        "random": {
            "exact": rand_results['exact'],
            "dir_a": rand_results['dir_a'],
            "dir_b": rand_results['dir_b'],
            "total": rand_results['total'],
            "rate_exact": round(rand_results['rate_exact'], 4),
            "rate_tolerant": round(rand_results['rate_tolerant'], 4),
            "improvement_pct": rand_results['improvement_pct'],
        },
        "statistical": {
            "z_exact": round(float(z_exact), 2),
            "z_tolerant": round(float(z_tolerant), 2),
            "signal_gain_pct": round(signal_gain, 2),
            "hebrew_benefits_more": signal_gain > 0,
        },
        "anchor_conversions": anchor_conv if anchor_conv else [],
        "zodiac_conversions": zodiac_conv if zodiac_conv else [],
        "plant_conversions": plant_conv if plant_conv else [],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")

    # 11. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  MATER LECTIONIS TOLERANCE — RESULTS")
    click.echo(f"{'=' * 60}")

    click.echo(f"\n  Hebrew lexicon matching:")
    click.echo(f"    Exact:     {heb_results['rate_exact']*100:.1f}%")
    click.echo(f"    Tolerant:  {heb_results['rate_tolerant']*100:.1f}% "
               f"(+{delta_heb:.2f}%)")
    click.echo(f"    Dir A:     +{heb_results['dir_a']} words "
               f"(decoded has extra matres)")
    click.echo(f"    Dir B:     +{heb_results['dir_b']} words "
               f"(lexicon has matres decoded lacks)")

    click.echo(f"\n  Random baseline:")
    click.echo(f"    Exact:     {rand_results['rate_exact']*100:.1f}%")
    click.echo(f"    Tolerant:  {rand_results['rate_tolerant']*100:.1f}% "
               f"(+{delta_rand:.2f}%)")

    click.echo(f"\n  Signal analysis:")
    click.echo(f"    z-score exact:    {z_exact:.1f}")
    click.echo(f"    z-score tolerant: {z_tolerant:.1f}")
    click.echo(f"    Signal gain:      +{signal_gain:.2f}%")

    if anchor_conv:
        n_anchor = len(anchor_conv)
        anchor_occ = sum(c.get('count', 0) for c in anchor_conv)
        click.echo(f"\n  Anchor words: {n_anchor} d=1→d=0 conversions "
                   f"({anchor_occ} occurrences)")

    if zodiac_conv:
        click.echo(f"  Zodiac terms: {len(zodiac_conv)} d=1→d=0 conversions")

    if plant_conv:
        click.echo(f"  Plant terms:  {len(plant_conv)} d=1→d=0 conversions")

    # Verdict
    click.echo(f"\n  {'=' * 40}")
    if signal_gain > 0.5:
        click.echo(f"  VERDICT: Mater tolerance yields DIFFERENTIAL signal")
        click.echo(f"  Hebrew benefits {signal_gain:.2f}% more than random")
    elif signal_gain > 0:
        click.echo(f"  VERDICT: Mater tolerance yields WEAK differential signal")
        click.echo(f"  Hebrew benefits {signal_gain:.2f}% more than random")
    else:
        click.echo(f"  VERDICT: No differential signal from mater tolerance")
        click.echo(f"  Both Hebrew and random benefit equally")
    click.echo(f"  {'=' * 40}")
