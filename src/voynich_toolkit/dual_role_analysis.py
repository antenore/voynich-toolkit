"""
Dual Role Analysis — Phase 9 B2

Investigate whether specific EVA characters map to different Hebrew
letters depending on word position.  The primary anomaly: shin (S from
EVA y) appears in 42.5% of word-initial positions — 6-8× higher than
any Hebrew letter should be.  If EVA y has a dual role (y-final→X,
y-elsewhere→shin), we recover an unmapped letter and fix the anomaly.

Approach:
1. Decode corpus with positional tracking (which EVA char → which position)
2. Compute Hebrew letter frequencies by position, flag anomalies
3. For each anomalous (letter, position), test splits with all 5 unmapped
   Hebrew letters: bet(b), zayin(z), samekh(s), tsade(C), qof(q)
4. Score each split: gained matches − lost matches against Hebrew lexicon
5. Rank by net improvement; verify with random baseline
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .cross_language_baseline import generate_random_lexicon
from .full_decode import (
    FULL_MAPPING, II_HEBREW, I_HEBREW, CH_HEBREW, DIRECTION, preprocess_eva,
)
from .utils import print_header, print_step, timer
from .word_structure import parse_eva_words


# Unmapped Hebrew letters (candidates for freed slots)
UNMAPPED_HEBREW = ['b', 'z', 's', 'C', 'q']
UNMAPPED_NAMES = {
    'b': 'bet', 'z': 'zayin', 's': 'samekh', 'C': 'tsade', 'q': 'qof',
}

# Rough Biblical Hebrew initial-consonant frequencies (%)
# From Andersen-Forbes word-initial counts on the HB
BH_INITIAL_PCT = {
    'w': 11.0, 'l': 8.5, 'm': 8.0, 'b': 7.5, 'h': 7.0, 'A': 6.0,
    'y': 5.5, 'S': 5.0, 'k': 4.5, 'n': 4.0, 'E': 3.5, 'r': 3.5,
    'd': 3.0, 't': 3.0, 'g': 2.5, 'p': 2.5, 'X': 2.0, 'J': 1.5,
    's': 1.5, 'C': 1.5, 'z': 1.0, 'q': 1.0,
}


# =====================================================================
# Decode with tracking
# =====================================================================

def decode_with_tracking(eva_word):
    """Decode EVA word to Hebrew, tracking which EVA char produced each
    Hebrew character.

    Returns:
        hebrew: str or None
        tracking: list of EVA source tokens ('y', 'ii', 'ch', etc.)
    """
    prefix, processed = preprocess_eva(eva_word)
    chars = (list(reversed(processed)) if DIRECTION == 'rtl'
             else list(processed))

    parts = []
    track = []
    for ch in chars:
        if ch == '\x01':
            parts.append(II_HEBREW)
            track.append('ii')
        elif ch == '\x02':
            parts.append(I_HEBREW)
            track.append('i')
        elif ch == '\x03':
            parts.append(CH_HEBREW)
            track.append('ch')
        elif ch in FULL_MAPPING:
            parts.append(FULL_MAPPING[ch])
            track.append(ch)
        else:
            return None, None

    return ''.join(parts), track


def position_type(index, length):
    """Return 'initial', 'medial', or 'final' for a character index."""
    if index == 0:
        return 'initial'
    elif index == length - 1:
        return 'final'
    return 'medial'


# =====================================================================
# Frequency analysis
# =====================================================================

def compute_position_frequencies(corpus_data):
    """Compute Hebrew letter frequencies by position.

    Args:
        corpus_data: list of (hebrew_word, tracking)

    Returns:
        freq: {heb_letter: {position: count}}
        eva_src: {(heb_letter, position): {eva_char: count}}
    """
    freq = defaultdict(lambda: Counter())
    eva_src = defaultdict(lambda: Counter())

    for heb_word, track in corpus_data:
        if heb_word is None or len(heb_word) < 3:
            continue
        for i, (h, ev) in enumerate(zip(heb_word, track)):
            pos = position_type(i, len(heb_word))
            freq[h][pos] += 1
            eva_src[(h, pos)][ev] += 1

    return dict(freq), dict(eva_src)


def identify_anomalies(freq, threshold=3.0):
    """Find (letter, position) combos that are over-represented.

    Compare observed initial frequency against BH expected.
    Returns list of (letter, position, observed_pct, expected_pct, ratio)
    sorted by ratio descending.
    """
    # Compute observed initial percentages
    total_initial = sum(freq.get(h, {}).get('initial', 0) for h in freq)
    if total_initial == 0:
        return []

    anomalies = []
    for h in freq:
        obs = freq[h].get('initial', 0)
        obs_pct = obs / total_initial * 100
        exp_pct = BH_INITIAL_PCT.get(h, 1.0)
        ratio = obs_pct / max(exp_pct, 0.1)
        if ratio >= threshold:
            anomalies.append((h, 'initial', obs_pct, exp_pct, ratio, obs))

    # Also check medial anomalies
    total_medial = sum(freq.get(h, {}).get('medial', 0) for h in freq)
    if total_medial > 0:
        for h in freq:
            obs = freq[h].get('medial', 0)
            obs_pct = obs / total_medial * 100
            # Use BH initial as rough proxy (no good medial data)
            exp_pct = BH_INITIAL_PCT.get(h, 1.0)
            ratio = obs_pct / max(exp_pct, 0.1)
            if ratio >= threshold:
                anomalies.append((h, 'medial', obs_pct, exp_pct, ratio, obs))

    anomalies.sort(key=lambda x: -x[4])
    return anomalies


# =====================================================================
# Split testing
# =====================================================================

@timer
def test_all_splits(corpus_data, lexicon_set, anomalies, max_candidates=5):
    """Test positional splits for all anomalous chars.

    For each anomaly, try replacing the Hebrew char at that position
    with each unmapped Hebrew letter.  Count gained/lost lexicon matches.

    Returns: list of split results sorted by net improvement.
    """
    # Pre-index: for each word, check if it currently matches
    indexed = []
    for heb_word, track in corpus_data:
        if heb_word is None or len(heb_word) < 3:
            continue
        is_match = heb_word in lexicon_set
        indexed.append((heb_word, track, is_match))

    results = []

    # Test splits for top anomalies
    tested_chars = set()
    for letter, pos, obs_pct, exp_pct, ratio, count in anomalies[:max_candidates]:
        key = (letter, pos)
        if key in tested_chars:
            continue
        tested_chars.add(key)

        for new_heb in UNMAPPED_HEBREW:
            gained = 0
            lost = 0
            sample_gained = []
            sample_lost = []
            n_affected = 0

            for heb_word, track, is_match in indexed:
                # Find positions where this letter appears from the
                # relevant position type
                affected = []
                for i, (h, ev) in enumerate(zip(heb_word, track)):
                    if h == letter and position_type(i, len(heb_word)) == pos:
                        affected.append(i)

                if not affected:
                    continue
                n_affected += 1

                # Build modified word
                new_word = list(heb_word)
                for idx in affected:
                    new_word[idx] = new_heb
                new_word = ''.join(new_word)

                new_match = new_word in lexicon_set

                if is_match and not new_match:
                    lost += 1
                    if len(sample_lost) < 5:
                        sample_lost.append((heb_word, new_word))
                elif not is_match and new_match:
                    gained += 1
                    if len(sample_gained) < 10:
                        sample_gained.append((heb_word, new_word))

            net = gained - lost
            results.append({
                'letter': letter,
                'position': pos,
                'new_hebrew': new_heb,
                'new_name': UNMAPPED_NAMES[new_heb],
                'n_affected': n_affected,
                'gained': gained,
                'lost': lost,
                'net': net,
                'obs_pct': round(obs_pct, 1),
                'exp_pct': round(exp_pct, 1),
                'ratio': round(ratio, 1),
                'sample_gained': sample_gained,
                'sample_lost': sample_lost,
            })

    results.sort(key=lambda x: -x['net'])
    return results


@timer
def test_split_with_random(corpus_data, hebrew_set, random_set,
                           letter, pos, new_heb):
    """Test a specific split against both Hebrew and random lexicons.

    Returns: (heb_net, rand_net, differential)
    """
    def score_split(indexed, lexicon_set):
        gained = lost = 0
        for heb_word, track, is_match in indexed:
            affected = [i for i, (h, ev) in enumerate(zip(heb_word, track))
                        if h == letter and position_type(i, len(heb_word)) == pos]
            if not affected:
                continue
            new_word = list(heb_word)
            for idx in affected:
                new_word[idx] = new_heb
            new_word = ''.join(new_word)
            new_match = new_word in lexicon_set
            if is_match and not new_match:
                lost += 1
            elif not is_match and new_match:
                gained += 1
        return gained - lost

    indexed_heb = [(hw, tr, hw in hebrew_set)
                   for hw, tr in corpus_data if hw and len(hw) >= 3]
    indexed_rand = [(hw, tr, hw in random_set)
                    for hw, tr in corpus_data if hw and len(hw) >= 3]

    heb_net = score_split(indexed_heb, hebrew_set)
    rand_net = score_split(indexed_rand, random_set)

    return heb_net, rand_net, heb_net - rand_net


# =====================================================================
# Combined multi-split analysis
# =====================================================================

def test_combined_splits(corpus_data, lexicon_set, splits):
    """Test multiple splits applied simultaneously.

    Args:
        splits: list of (letter, position, new_hebrew) tuples

    Returns: (gained, lost, net, sample_gained)
    """
    gained = lost = 0
    sample_gained = []

    for heb_word, track in corpus_data:
        if heb_word is None or len(heb_word) < 3:
            continue
        is_match = heb_word in lexicon_set

        # Apply all splits
        new_word = list(heb_word)
        any_change = False
        for letter, pos, new_heb in splits:
            for i, (h, ev) in enumerate(zip(heb_word, track)):
                if h == letter and position_type(i, len(heb_word)) == pos:
                    new_word[i] = new_heb
                    any_change = True

        if not any_change:
            continue

        new_word_str = ''.join(new_word)
        new_match = new_word_str in lexicon_set

        if is_match and not new_match:
            lost += 1
        elif not is_match and new_match:
            gained += 1
            if len(sample_gained) < 20:
                sample_gained.append((heb_word, new_word_str))

    return gained, lost, gained - lost, sample_gained


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Dual role analysis: test position-dependent mapping splits."""
    report_path = config.stats_dir / "dual_role_report.json"

    if report_path.exists() and not force:
        click.echo("  Dual role report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 9 B2 — Dual Role Analysis")

    # 1. Parse and decode with tracking
    print_step("Decoding corpus with position tracking...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)

    corpus_data = []
    for page in eva_data["pages"]:
        for word in page["words"]:
            heb, track = decode_with_tracking(word)
            if heb:
                corpus_data.append((heb, track))
    click.echo(f"    {len(corpus_data)} words decoded with tracking")

    # 2. Position frequency analysis
    print_step("Computing positional frequencies...")
    freq, eva_src = compute_position_frequencies(corpus_data)

    # Print initial distribution
    total_init = sum(freq.get(h, {}).get('initial', 0) for h in freq)
    click.echo(f"\n    Hebrew initial-position distribution (top 8):")
    for h in sorted(freq, key=lambda x: -freq[x].get('initial', 0))[:8]:
        ini = freq[h].get('initial', 0)
        pct = ini / total_init * 100
        exp = BH_INITIAL_PCT.get(h, 1.0)
        ratio = pct / max(exp, 0.1)
        flag = " <<<" if ratio > 3 else ""
        click.echo(f"      {h:2s}  {pct:5.1f}% (BH ~{exp:.1f}%, "
                   f"ratio {ratio:.1f}x){flag}")

    # 3. Identify anomalies
    print_step("Identifying positional anomalies...")
    anomalies = identify_anomalies(freq, threshold=2.5)
    click.echo(f"    {len(anomalies)} anomalies found (ratio > 2.5x):")
    for letter, pos, obs_pct, exp_pct, ratio, count in anomalies:
        click.echo(f"      {letter:2s} at {pos:8s}: "
                   f"{obs_pct:.1f}% vs BH {exp_pct:.1f}% "
                   f"(ratio {ratio:.1f}x, {count} occ)")

    # 4. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    base_path = config.lexicon_dir / "lexicon.json"
    lex_path = enriched_path if enriched_path.exists() else base_path
    with open(lex_path) as f:
        hlex = json.load(f)
    hebrew_set = set(hlex["all_consonantal_forms"])
    click.echo(f"    {len(hebrew_set)} forms")

    # 5. Test all splits
    print_step("Testing positional splits against lexicon...")
    split_results = test_all_splits(
        corpus_data, hebrew_set, anomalies, max_candidates=6)

    # Show top 15 results
    click.echo(f"\n    Top 15 splits by net lexicon improvement:")
    click.echo(f"    {'Current':>8s} {'Pos':>8s} {'→New':>8s} "
               f"{'Gained':>7s} {'Lost':>5s} {'Net':>5s} {'Affected':>8s}")
    click.echo(f"    {'-'*55}")
    for r in split_results[:15]:
        click.echo(f"    {r['letter']:>8s} {r['position']:>8s} "
                   f"→{r['new_name']:>7s} "
                   f"{r['gained']:7d} {r['lost']:5d} "
                   f"{r['net']:+5d} {r['n_affected']:8d}")

    # 6. Random baseline for top candidate
    best = split_results[0] if split_results else None
    heb_net = rand_net = diff = 0
    if best and best['net'] > 0:
        print_step(f"Validating best split against random baseline...")
        random_set = generate_random_lexicon(hebrew_set, seed=42)
        heb_net, rand_net, diff = test_split_with_random(
            corpus_data, hebrew_set, random_set,
            best['letter'], best['position'], best['new_hebrew'])
        click.echo(f"    Best: {best['letter']}@{best['position']} → "
                   f"{best['new_name']}")
        click.echo(f"    Hebrew net: {heb_net:+d}")
        click.echo(f"    Random net: {rand_net:+d}")
        click.echo(f"    Differential: {diff:+d} "
                   f"({'Hebrew benefits MORE' if diff > 0 else 'No advantage'})")

    # 7. Test top 2-3 combined
    print_step("Testing combined splits (top non-conflicting)...")
    combined_splits = []
    used_letters = set()
    used_new = set()
    for r in split_results:
        if r['net'] <= 0:
            break
        if r['letter'] in used_letters:
            continue
        if r['new_hebrew'] in used_new:
            continue
        combined_splits.append(
            (r['letter'], r['position'], r['new_hebrew']))
        used_letters.add(r['letter'])
        used_new.add(r['new_hebrew'])
        if len(combined_splits) >= 3:
            break

    combined_result = None
    if len(combined_splits) >= 2:
        c_gained, c_lost, c_net, c_sample = test_combined_splits(
            corpus_data, hebrew_set, combined_splits)
        combined_result = {
            'splits': [
                {'letter': l, 'position': p,
                 'new_hebrew': nh, 'new_name': UNMAPPED_NAMES[nh]}
                for l, p, nh in combined_splits
            ],
            'gained': c_gained,
            'lost': c_lost,
            'net': c_net,
            'sample': [(o, n) for o, n in c_sample],
        }
        click.echo(f"    Combined {len(combined_splits)} splits:")
        for l, p, nh in combined_splits:
            click.echo(f"      {l}@{p} → {UNMAPPED_NAMES[nh]}")
        click.echo(f"    Gained: {c_gained}, Lost: {c_lost}, "
                   f"Net: {c_net:+d}")
        if c_sample:
            click.echo(f"    Sample recovered:")
            for orig, new in c_sample[:10]:
                click.echo(f"      {orig:12s} → {new}")

    # 8. Show sample gained for best single split
    if best and best['sample_gained']:
        print_step(f"Sample words gained by best split "
                   f"({best['letter']}→{best['new_name']}):")
        for orig, new in best['sample_gained'][:10]:
            click.echo(f"    {orig:12s} → {new}")
    if best and best['sample_lost']:
        print_step(f"Sample words LOST by best split:")
        for orig, new in best['sample_lost']:
            click.echo(f"    {orig:12s} → {new}")

    # 9. Save report
    print_step("Saving report...")
    report = {
        "position_frequencies": {
            h: dict(freq[h]) for h in freq
        },
        "bh_expected": BH_INITIAL_PCT,
        "anomalies": [
            {"letter": l, "position": p, "obs_pct": round(o, 1),
             "exp_pct": round(e, 1), "ratio": round(r, 1), "count": c}
            for l, p, o, e, r, c in anomalies
        ],
        "split_results": [
            {k: v for k, v in r.items()
             if k not in ('sample_gained', 'sample_lost')}
            for r in split_results[:30]
        ],
        "best_split": {
            "letter": best['letter'],
            "position": best['position'],
            "new_hebrew": best['new_hebrew'],
            "new_name": best['new_name'],
            "net": best['net'],
            "gained": best['gained'],
            "lost": best['lost'],
            "heb_net_vs_random": diff,
            "sample_gained": [
                {"original": o, "new": n}
                for o, n in best['sample_gained']
            ],
        } if best else None,
        "combined": {
            "splits": combined_result['splits'],
            "gained": combined_result['gained'],
            "lost": combined_result['lost'],
            "net": combined_result['net'],
            "sample": [{"original": o, "new": n}
                       for o, n in combined_result['sample'][:20]],
        } if combined_result else None,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")

    # 10. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  DUAL ROLE ANALYSIS — RESULTS")
    click.echo(f"{'=' * 60}")

    if best:
        click.echo(f"\n  Primary anomaly: {best['letter']} at "
                   f"{best['position']} ({best['obs_pct']}% vs "
                   f"BH {best['exp_pct']}%)")
        click.echo(f"  Best split: {best['letter']}@{best['position']} → "
                   f"{best['new_name']} ({best['new_hebrew']})")
        click.echo(f"    +{best['gained']} gained, "
                   f"−{best['lost']} lost, "
                   f"net {best['net']:+d}")
        if diff != 0:
            click.echo(f"    vs random: Hebrew {heb_net:+d}, "
                       f"Random {rand_net:+d}, diff {diff:+d}")

    if combined_result and combined_result['net'] > 0:
        click.echo(f"\n  Combined ({len(combined_result['splits'])} splits): "
                   f"net {combined_result['net']:+d}")

    # Verdict
    click.echo(f"\n  {'=' * 40}")
    if best and best['net'] > 50 and diff > 0:
        click.echo(f"  VERDICT: STRONG dual-role signal")
        click.echo(f"  {best['letter']}@{best['position']} → "
                   f"{best['new_name']} recovers {best['net']} matches")
    elif best and best['net'] > 20:
        click.echo(f"  VERDICT: Moderate dual-role signal")
        click.echo(f"  {best['letter']}@{best['position']} → "
                   f"{best['new_name']} recovers {best['net']} matches")
    elif best and best['net'] > 0:
        click.echo(f"  VERDICT: Weak dual-role signal")
    else:
        click.echo(f"  VERDICT: No beneficial split found")
    click.echo(f"  {'=' * 40}")
