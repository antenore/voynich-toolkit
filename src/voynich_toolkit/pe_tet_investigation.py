"""
Deep investigation of Pe (p) and Tet (J) medial over-representation.

Phase 9 diagnosis identified:
- Pe at medial ~21% (BH ~2.0%, highly anomalous)
- J@medial→qof: +126 net matches in prior dual role test (pre-B2 mapping)

This module re-investigates with the CURRENT mapping (including d→bet,
h→samekh positional splits already integrated).

3 unmapped Hebrew letters remain: zayin(z), tsade(C), qof(q).

Approach:
1. Decode corpus with full current mapping (all positional splits)
2. Compute positional frequency profiles for pe and J
3. Test pe@medial → each of z, C, q
4. Test J@medial → each of z, C, q
5. Validate best candidates against random baseline
6. Test combined splits (pe+J if both positive)
7. Cross-check: do recovered words make semantic sense?
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import (
    FULL_MAPPING, II_HEBREW, I_HEBREW, CH_HEBREW,
    INITIAL_D_HEBREW, INITIAL_H_HEBREW, DIRECTION, preprocess_eva,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# Unmapped Hebrew letters — candidates for recovered slots
UNMAPPED = {'z': 'zayin', 'C': 'tsade', 'q': 'qof'}

# BH medial frequency estimates (rough, from Andersen-Forbes)
BH_MEDIAL_PCT = {
    'y': 10.0, 'w': 9.0, 'r': 7.0, 'h': 5.0, 'm': 5.0,
    'l': 5.0, 'n': 5.0, 'b': 4.5, 'S': 3.5, 't': 3.0,
    'k': 3.0, 'A': 3.0, 'd': 2.5, 'X': 2.5, 'E': 2.0,
    'p': 2.0, 'g': 1.5, 'J': 1.5, 's': 1.5, 'z': 1.0,
    'C': 0.8, 'q': 0.8,
}


# =====================================================================
# Decode with full current mapping (including all positional splits)
# =====================================================================

def decode_current(eva_word):
    """Decode EVA word with the full current mapping including positional splits.

    Returns: (hebrew_str, eva_tracking_list) or (None, None)
    """
    prefix, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed)) if DIRECTION == 'rtl' else list(processed)

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

    # Apply positional splits (d→bet, h→samekh at word-initial)
    if parts and parts[0] == 'd':
        parts[0] = INITIAL_D_HEBREW
    if parts and parts[0] == 'h':
        parts[0] = INITIAL_H_HEBREW

    return ''.join(parts), track


def position_type(idx, length):
    if idx == 0:
        return 'initial'
    elif idx == length - 1:
        return 'final'
    return 'medial'


# =====================================================================
# Positional frequency analysis
# =====================================================================

def compute_hebrew_positional_freq(corpus):
    """Compute Hebrew letter frequencies by position.

    Args:
        corpus: list of (hebrew_word, track)

    Returns:
        freq: {hebrew_letter: {position: count}}
        totals: {position: total_count}
    """
    freq = defaultdict(lambda: Counter())
    totals = Counter()

    for heb_word, track in corpus:
        if heb_word is None or len(heb_word) < 3:
            continue
        for i, h in enumerate(heb_word):
            pos = position_type(i, len(heb_word))
            freq[h][pos] += 1
            totals[pos] += 1

    return dict(freq), dict(totals)


# =====================================================================
# Split testing
# =====================================================================

def test_split(corpus, lexicon_set, target_letter, position, new_letter,
               min_len=3):
    """Test replacing target_letter at position with new_letter.

    Returns dict with gained, lost, net, samples.
    """
    gained = 0
    lost = 0
    sample_gained = []
    sample_lost = []
    n_affected = 0

    for heb_word, track in corpus:
        if heb_word is None or len(heb_word) < min_len:
            continue

        # Find positions where target_letter appears at the given position type
        affected_indices = []
        for i, h in enumerate(heb_word):
            if h == target_letter and position_type(i, len(heb_word)) == position:
                affected_indices.append(i)

        if not affected_indices:
            continue
        n_affected += 1

        # Build modified word
        new_word = list(heb_word)
        for idx in affected_indices:
            new_word[idx] = new_letter
        new_word_str = ''.join(new_word)

        orig_match = heb_word in lexicon_set
        new_match = new_word_str in lexicon_set

        if not orig_match and new_match:
            gained += 1
            if len(sample_gained) < 15:
                sample_gained.append((heb_word, new_word_str))
        elif orig_match and not new_match:
            lost += 1
            if len(sample_lost) < 10:
                sample_lost.append((heb_word, new_word_str))

    return {
        'target': target_letter,
        'position': position,
        'new_letter': new_letter,
        'new_name': UNMAPPED.get(new_letter, CONSONANT_NAMES.get(new_letter, '?')),
        'n_affected': n_affected,
        'gained': gained,
        'lost': lost,
        'net': gained - lost,
        'sample_gained': sample_gained,
        'sample_lost': sample_lost,
    }


def test_split_random(corpus, hebrew_set, random_set, target_letter,
                      position, new_letter, min_len=3):
    """Validate a split against both Hebrew and Random lexicons.

    Returns (heb_net, rand_net, differential).
    """
    def score(corpus, lex):
        gained = lost = 0
        for heb_word, track in corpus:
            if heb_word is None or len(heb_word) < min_len:
                continue
            affected = [i for i, h in enumerate(heb_word)
                        if h == target_letter and
                        position_type(i, len(heb_word)) == position]
            if not affected:
                continue
            new_word = list(heb_word)
            for idx in affected:
                new_word[idx] = new_letter
            new_word_str = ''.join(new_word)
            om = heb_word in lex
            nm = new_word_str in lex
            if not om and nm:
                gained += 1
            elif om and not nm:
                lost += 1
        return gained - lost

    heb_net = score(corpus, hebrew_set)
    rand_net = score(corpus, random_set)
    return heb_net, rand_net, heb_net - rand_net


# =====================================================================
# Combined splits
# =====================================================================

def test_combined(corpus, lexicon_set, splits, min_len=3):
    """Test multiple splits applied simultaneously.

    Args:
        splits: list of (target_letter, position, new_letter)

    Returns: (gained, lost, net, sample_gained)
    """
    gained = lost = 0
    sample_gained = []

    for heb_word, track in corpus:
        if heb_word is None or len(heb_word) < min_len:
            continue

        new_word = list(heb_word)
        any_change = False
        for target, pos, new_letter in splits:
            for i, h in enumerate(heb_word):
                if h == target and position_type(i, len(heb_word)) == pos:
                    new_word[i] = new_letter
                    any_change = True

        if not any_change:
            continue

        new_word_str = ''.join(new_word)
        om = heb_word in lexicon_set
        nm = new_word_str in lexicon_set

        if not om and nm:
            gained += 1
            if len(sample_gained) < 25:
                sample_gained.append((heb_word, new_word_str))
        elif om and not nm:
            lost += 1

    return gained, lost, gained - lost, sample_gained


# =====================================================================
# Lexicon and random generation
# =====================================================================

def _load_lexicon(config):
    for name in ("lexicon_enriched.json", "lexicon.json"):
        path = config.lexicon_dir / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            forms = data.get("all_consonantal_forms", [])
            if forms:
                return set(forms)
    return set()


def _generate_random(hebrew_set, seed=42):
    rng = random.Random(seed)
    forms = list(hebrew_set)
    lengths = [len(f) for f in forms]
    all_chars = ''.join(forms)
    char_freq = Counter(all_chars)
    total = sum(char_freq.values())
    chars = list(char_freq.keys())
    weights = [char_freq[c] / total for c in chars]
    random_forms = set()
    for length in lengths:
        form = ''.join(rng.choices(chars, weights=weights, k=length))
        random_forms.add(form)
    return random_forms


# =====================================================================
# Gloss lookup (if available)
# =====================================================================

def _load_gloss_map(config):
    """Load form→gloss mapping from enriched lexicon."""
    for name in ("lexicon_enriched.json", "lexicon.json"):
        path = config.lexicon_dir / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data.get("form_to_gloss", {})
    return {}


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Deep investigation of pe and tet medial over-representation."""
    report_path = config.stats_dir / "pe_tet_investigation_report.json"

    if report_path.exists() and not force:
        click.echo("  Pe/Tet investigation report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 9 — Pe(p) & Tet(J) MEDIAL INVESTIGATION")
    click.echo("  Current: EVA e→pe(p), EVA t→tet(J)")
    click.echo("  Unmapped Hebrew: zayin(z), tsade(C), qof(q)")
    click.echo("  Note: using CURRENT mapping (d→bet, h→samekh already integrated)")

    # 1. Parse and decode
    print_step("Parsing and decoding corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)

    corpus = []
    for page in eva_data["pages"]:
        for word in page["words"]:
            heb, track = decode_current(word)
            if heb:
                corpus.append((heb, track))
    click.echo(f"    {len(corpus)} words decoded with current mapping")

    # 2. Load lexicons
    print_step("Loading lexicons...")
    hebrew_set = _load_lexicon(config)
    random_set = _generate_random(hebrew_set, seed=42)
    gloss_map = _load_gloss_map(config)
    click.echo(f"    Hebrew: {len(hebrew_set)} forms, Random: {len(random_set)} forms")

    report = {}

    # ═══════════════════════════════════════════════════════════════
    # 3. POSITIONAL FREQUENCY ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print_header("1. POSITIONAL FREQUENCY PROFILES")

    freq, totals = compute_hebrew_positional_freq(corpus)

    # Focus on pe and J
    for target, target_name in [('p', 'pe'), ('J', 'tet')]:
        tf = freq.get(target, {})
        click.echo(f"\n  {target_name.upper()} ({target}):")
        for pos in ['initial', 'medial', 'final']:
            count = tf.get(pos, 0)
            total = totals.get(pos, 1)
            pct = count / total * 100
            bh_exp = BH_MEDIAL_PCT.get(target, 1.0)
            ratio = pct / max(bh_exp, 0.1)
            flag = " <<<" if ratio > 3 else (" <<" if ratio > 2 else "")
            click.echo(f"    {pos:8s}: {count:6d} / {total:6d} = {pct:5.1f}%"
                       f"  (BH ~{bh_exp:.1f}%, ratio {ratio:.1f}x){flag}")

    # Also show ALL medial letters for context
    click.echo(f"\n  ALL medial letters (top 10):")
    click.echo(f"    {'Letter':8s} {'Count':>7s} {'%':>7s} {'BH%':>6s} {'Ratio':>6s}")
    click.echo(f"    {'-'*40}")
    total_med = totals.get('medial', 1)
    for h in sorted(freq, key=lambda x: -freq[x].get('medial', 0))[:12]:
        count = freq[h].get('medial', 0)
        pct = count / total_med * 100
        bh = BH_MEDIAL_PCT.get(h, 1.0)
        ratio = pct / max(bh, 0.1)
        flag = " <<<" if ratio > 3 else ""
        click.echo(f"    {h:8s} {count:7d} {pct:6.1f}% {bh:5.1f}% "
                   f"{ratio:5.1f}x{flag}")

    report['positional_freq'] = {
        h: dict(freq[h]) for h in freq
    }
    report['totals'] = dict(totals)

    # ═══════════════════════════════════════════════════════════════
    # 4. PE MEDIAL SPLIT TESTS
    # ═══════════════════════════════════════════════════════════════
    print_header("2. PE MEDIAL SPLIT TESTS")
    click.echo("  Testing: pe(p) at medial → each unmapped Hebrew letter\n")

    pe_results = []
    for new_letter, new_name in UNMAPPED.items():
        result = test_split(corpus, hebrew_set, 'p', 'medial', new_letter)
        pe_results.append(result)
        click.echo(f"    p@medial → {new_name:8s} ({new_letter}): "
                   f"gained={result['gained']:+5d}  "
                   f"lost={result['lost']:+5d}  "
                   f"net={result['net']:+5d}  "
                   f"(affected {result['n_affected']} words)")

    # Also test pe at other positions for completeness
    click.echo(f"\n  Also testing pe at initial and final:")
    for pos in ['initial', 'final']:
        for new_letter, new_name in UNMAPPED.items():
            result = test_split(corpus, hebrew_set, 'p', pos, new_letter)
            pe_results.append(result)
            if result['net'] > 10:
                click.echo(f"    p@{pos:8s} → {new_name:8s}: net={result['net']:+5d}")

    report['pe_splits'] = [{k: v for k, v in r.items()
                            if k not in ('sample_gained', 'sample_lost')}
                           for r in pe_results]

    # ═══════════════════════════════════════════════════════════════
    # 5. TET MEDIAL SPLIT TESTS
    # ═══════════════════════════════════════════════════════════════
    print_header("3. TET MEDIAL SPLIT TESTS")
    click.echo("  Testing: tet(J) at medial → each unmapped Hebrew letter\n")

    tet_results = []
    for new_letter, new_name in UNMAPPED.items():
        result = test_split(corpus, hebrew_set, 'J', 'medial', new_letter)
        tet_results.append(result)
        click.echo(f"    J@medial → {new_name:8s} ({new_letter}): "
                   f"gained={result['gained']:+5d}  "
                   f"lost={result['lost']:+5d}  "
                   f"net={result['net']:+5d}  "
                   f"(affected {result['n_affected']} words)")

    # Also test J at other positions
    click.echo(f"\n  Also testing tet at initial and final:")
    for pos in ['initial', 'final']:
        for new_letter, new_name in UNMAPPED.items():
            result = test_split(corpus, hebrew_set, 'J', pos, new_letter)
            tet_results.append(result)
            if result['net'] > 10:
                click.echo(f"    J@{pos:8s} → {new_name:8s}: net={result['net']:+5d}")

    report['tet_splits'] = [{k: v for k, v in r.items()
                             if k not in ('sample_gained', 'sample_lost')}
                            for r in tet_results]

    # ═══════════════════════════════════════════════════════════════
    # 6. RANDOM BASELINE VALIDATION
    # ═══════════════════════════════════════════════════════════════
    print_header("4. RANDOM BASELINE VALIDATION")

    # Find best pe and best J splits
    best_pe = max((r for r in pe_results if r['position'] == 'medial'),
                  key=lambda x: x['net'], default=None)
    best_tet = max((r for r in tet_results if r['position'] == 'medial'),
                   key=lambda x: x['net'], default=None)

    # Also find overall best pe and J across all positions
    best_pe_all = max(pe_results, key=lambda x: x['net'], default=None)
    best_tet_all = max(tet_results, key=lambda x: x['net'], default=None)

    validated = []

    for label, candidate in [('pe_medial', best_pe),
                              ('tet_medial', best_tet),
                              ('pe_best', best_pe_all),
                              ('tet_best', best_tet_all)]:
        if candidate is None or candidate['net'] <= 0:
            click.echo(f"\n  {label}: No positive split found — skipping validation")
            continue

        click.echo(f"\n  Validating: {candidate['target']}@{candidate['position']}"
                   f" → {candidate['new_name']} (net={candidate['net']:+d})")

        heb_net, rand_net, diff = test_split_random(
            corpus, hebrew_set, random_set,
            candidate['target'], candidate['position'], candidate['new_letter'])

        click.echo(f"    Hebrew net:     {heb_net:+d}")
        click.echo(f"    Random net:     {rand_net:+d}")
        click.echo(f"    Differential:   {diff:+d}")

        is_real = diff > 20  # at least 20 more than random
        click.echo(f"    {'REAL SIGNAL' if is_real else 'NOISE (random gains similar)'}")

        validated.append({
            'label': label,
            'target': candidate['target'],
            'position': candidate['position'],
            'new_letter': candidate['new_letter'],
            'new_name': candidate['new_name'],
            'heb_net': heb_net,
            'rand_net': rand_net,
            'differential': diff,
            'is_real': is_real,
        })

    report['random_validation'] = validated

    # ═══════════════════════════════════════════════════════════════
    # 7. COMBINED SPLIT TEST
    # ═══════════════════════════════════════════════════════════════
    print_header("5. COMBINED SPLIT TEST")

    # Select validated candidates (differential > 20)
    real_splits = [v for v in validated if v['is_real']
                   and v['label'] in ('pe_medial', 'tet_medial')]

    # Ensure no conflict (same new_letter used twice)
    used_new = set()
    chosen = []
    for v in sorted(real_splits, key=lambda x: -x['differential']):
        if v['new_letter'] not in used_new:
            chosen.append((v['target'], v['position'], v['new_letter']))
            used_new.add(v['new_letter'])

    if len(chosen) >= 2:
        click.echo(f"\n  Testing combined: "
                   + " + ".join(f"{t}@{p}→{UNMAPPED.get(n, n)}"
                                for t, p, n in chosen))
        c_gained, c_lost, c_net, c_sample = test_combined(
            corpus, hebrew_set, chosen)

        # Also vs random
        _, _, c_net_r, _ = test_combined(corpus, random_set, chosen)

        click.echo(f"    Combined gained: {c_gained}")
        click.echo(f"    Combined lost:   {c_lost}")
        click.echo(f"    Combined net:    {c_net:+d}")
        click.echo(f"    vs Random net:   {c_net_r:+d}")
        click.echo(f"    Differential:    {c_net - c_net_r:+d}")

        report['combined'] = {
            'splits': [{'target': t, 'position': p, 'new_letter': n,
                        'new_name': UNMAPPED.get(n, n)}
                       for t, p, n in chosen],
            'gained': c_gained,
            'lost': c_lost,
            'net': c_net,
            'rand_net': c_net_r,
            'differential': c_net - c_net_r,
        }

        if c_sample:
            click.echo(f"\n    Sample recovered words:")
            for orig, new in c_sample[:15]:
                gloss = gloss_map.get(new, '')
                g_str = f" = {gloss}" if gloss else ""
                click.echo(f"      {orig:12s} → {new:12s}{g_str}")
    elif len(chosen) == 1:
        click.echo(f"\n  Only 1 validated split — no combined test needed.")
    else:
        click.echo(f"\n  No validated medial splits — skipping combined test.")
        report['combined'] = None

    # ═══════════════════════════════════════════════════════════════
    # 8. SAMPLE RECOVERED WORDS WITH GLOSSES
    # ═══════════════════════════════════════════════════════════════
    print_header("6. SAMPLE RECOVERED WORDS")

    for candidate in [best_pe, best_tet]:
        if candidate is None or candidate['net'] <= 0:
            continue

        v = next((v for v in validated
                  if v['target'] == candidate['target']
                  and v['position'] == candidate['position']
                  and v['new_letter'] == candidate['new_letter']
                  and v.get('is_real')), None)
        if not v:
            continue

        click.echo(f"\n  {candidate['target']}@{candidate['position']}"
                   f" → {candidate['new_name']}:")
        click.echo(f"  {'Original':12s} → {'Modified':12s} {'Gloss':s}")
        click.echo(f"  {'-'*50}")
        for orig, new in candidate['sample_gained'][:15]:
            gloss = gloss_map.get(new, '')
            click.echo(f"  {orig:12s} → {new:12s} {gloss}")

    # ═══════════════════════════════════════════════════════════════
    # 9. EVA SOURCE ANALYSIS — what EVA chars produce pe/tet at medial?
    # ═══════════════════════════════════════════════════════════════
    print_header("7. EVA SOURCE ANALYSIS")

    for target, target_name in [('p', 'pe'), ('J', 'tet')]:
        click.echo(f"\n  Which EVA chars produce {target_name}({target}) at medial?")
        eva_sources = Counter()
        for heb_word, track in corpus:
            if heb_word is None or len(heb_word) < 3:
                continue
            for i, (h, ev) in enumerate(zip(heb_word, track)):
                if h == target and position_type(i, len(heb_word)) == 'medial':
                    eva_sources[ev] += 1
        total_src = sum(eva_sources.values())
        for ev, count in eva_sources.most_common(5):
            click.echo(f"    EVA {ev:3s}: {count:5d} ({count/total_src*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print_header("VERDICT")

    click.echo(f"\n  {'='*55}")

    # Summarize pe
    if best_pe and best_pe['net'] > 0:
        v_pe = next((v for v in validated
                     if v['label'] == 'pe_medial'), None)
        if v_pe and v_pe['is_real']:
            click.echo(f"  PE@medial → {best_pe['new_name']}: "
                       f"VALIDATED (net={best_pe['net']:+d}, "
                       f"diff={v_pe['differential']:+d})")
        else:
            click.echo(f"  PE@medial → {best_pe['new_name']}: "
                       f"net={best_pe['net']:+d} but "
                       f"NOT validated vs random")
    else:
        click.echo(f"  PE@medial: no positive split found")

    # Summarize tet
    if best_tet and best_tet['net'] > 0:
        v_tet = next((v for v in validated
                      if v['label'] == 'tet_medial'), None)
        if v_tet and v_tet['is_real']:
            click.echo(f"  TET@medial → {best_tet['new_name']}: "
                       f"VALIDATED (net={best_tet['net']:+d}, "
                       f"diff={v_tet['differential']:+d})")
        else:
            click.echo(f"  TET@medial → {best_tet['new_name']}: "
                       f"net={best_tet['net']:+d} but "
                       f"NOT validated vs random")
    else:
        click.echo(f"  TET@medial: no positive split found")

    click.echo(f"  {'='*55}")

    # Save report
    print_step("Saving report...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")
