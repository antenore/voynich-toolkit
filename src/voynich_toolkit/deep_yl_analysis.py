"""
Deep investigation of EVA y (shin) and EVA l (mem).

Phase 9 — Priority investigations:

Part 1: EVA y → shin (S)
  - Shin at 42.5% word-initial: test the she- (Mishnaic relative pronoun) hypothesis
  - She-prefix stripping: strip initial S from decoded words, check lexicon matches
  - Post-shin distribution: do letters following initial shin look like normal word-initials?
  - Allograph scan: cosine + context overlap of y with all other EVA chars

Part 2: EVA l → mem (m)
  - Positional breakdown: why does mem nearly disappear in medial position?
  - Dual-role test at lower threshold: l@medial → samekh/zayin/tsade/qof
  - Allograph scan: cosine + context overlap of l with all other EVA chars
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step


# =====================================================================
# Mapping (duplicated for self-contained analysis)
# =====================================================================

MAPPING = {
    'a': 'y', 'c': 'A', 'd': 'r', 'e': 'p', 'f': 'l',
    'g': 'X', 'h': 'E', 'k': 't', 'l': 'm', 'm': 'g',
    'n': 'd', 'o': 'w', 'p': 'l', 'r': 'h', 's': 'n',
    't': 'J', 'y': 'S',
}
II_HEBREW = 'h'
I_HEBREW = 'r'
CH_HEBREW = 'k'
INITIAL_D_HEBREW = 'b'

# Unmapped Hebrew letters (candidates for dual-role reassignment)
UNMAPPED_HEBREW = ['z', 's', 'C', 'q']

# Biblical Hebrew approximate initial-consonant frequencies (%)
BH_INITIAL_PCT = {
    'w': 11.0, 'l': 8.5, 'm': 8.0, 'b': 7.5, 'h': 7.0,
    'S': 2.5, 'k': 5.5, 'A': 5.0, 'y': 4.5, 'n': 4.0,
    't': 3.5, 'E': 3.0, 'd': 3.0, 'r': 2.5, 'p': 2.0,
    'X': 2.0, 'g': 1.5, 'J': 1.0, 'C': 1.0, 'q': 1.0,
    'z': 1.0, 's': 1.0,
}

# Mishnaic Hebrew approximate initial-consonant frequencies (%)
# Shin is ~8-12% initial in MH (she- prefix boosts it significantly)
MH_INITIAL_PCT_RANGE = {'S': (8.0, 15.0)}


# =====================================================================
# Preprocessing & decoding (from cross_language_baseline)
# =====================================================================

def preprocess_eva(word):
    """Replace ch→token, ii→token, standalone i→token, strip q-prefix."""
    w = word
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
    """Decode EVA word to Hebrew consonantal string (canonical pipeline)."""
    _, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))
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
            return None
    if parts and parts[0] == 'd':
        parts[0] = INITIAL_D_HEBREW
    return ''.join(parts)


def decode_to_hebrew_parts(eva_word):
    """Decode EVA word returning list of Hebrew chars (for positional analysis)."""
    _, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))
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
            return None
    if parts and parts[0] == 'd':
        parts[0] = INITIAL_D_HEBREW
    return parts


# =====================================================================
# Lexicon loading
# =====================================================================

def load_lexicon_set(config):
    """Load enriched Hebrew lexicon as a set of consonantal strings."""
    for name in ("lexicon_enriched.json", "enriched_hebrew_lexicon.json",
                 "lexicon.json", "hebrew_lexicon.json"):
        path = config.lexicon_dir / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            forms = data.get("all_consonantal_forms", [])
            if forms:
                return set(forms)

    raise click.ClickException("Hebrew lexicon not found. Run: voynich enrich-lexicon")


# =====================================================================
# Part 1: EVA y (shin) — She-prefix hypothesis
# =====================================================================

def analyze_shin_prefix(eva_words, lexicon_set, min_len=3):
    """Test whether initial shin acts as Mishnaic she- prefix.

    If shin-initial words are she-X, then stripping shin should leave
    a valid Hebrew word X more often than chance.

    Returns dict with test results.
    """
    # Decode all words
    decoded = []
    for w in eva_words:
        heb = decode_to_hebrew(w)
        if heb and len(heb) >= min_len:
            decoded.append(heb)

    # Separate shin-initial vs other
    shin_words = [w for w in decoded if w[0] == 'S']
    other_words = [w for w in decoded if w[0] != 'S']

    # Test 1: Strip initial shin → check lexicon match of remainder
    shin_stripped_matches = 0
    shin_stripped_total = 0
    shin_stripped_examples = Counter()
    for w in shin_words:
        remainder = w[1:]
        if len(remainder) < 2:  # too short after stripping
            continue
        shin_stripped_total += 1
        if remainder in lexicon_set:
            shin_stripped_matches += 1
            shin_stripped_examples[remainder] += 1

    shin_strip_rate = shin_stripped_matches / max(shin_stripped_total, 1)

    # Control: strip each other Hebrew letter and compute same rate
    control_rates = {}
    for letter in sorted(set(w[0] for w in decoded)):
        if letter == 'S':
            continue
        words_with_initial = [w for w in decoded if w[0] == letter]
        n_stripped = 0
        n_match = 0
        for w in words_with_initial:
            remainder = w[1:]
            if len(remainder) < 2:
                continue
            n_stripped += 1
            if remainder in lexicon_set:
                n_match += 1
        if n_stripped > 0:
            control_rates[letter] = {
                'count': len(words_with_initial),
                'stripped_total': n_stripped,
                'stripped_matches': n_match,
                'rate': n_match / n_stripped,
            }

    # Control mean/std
    rates = [v['rate'] for v in control_rates.values() if v['stripped_total'] >= 50]
    control_mean = float(np.mean(rates)) if rates else 0
    control_std = float(np.std(rates)) if rates else 0
    shin_z = (shin_strip_rate - control_mean) / control_std if control_std > 0 else 0

    # Test 2: What follows initial shin? (second letter distribution)
    post_shin_dist = Counter()
    for w in shin_words:
        if len(w) >= 2:
            post_shin_dist[w[1]] += 1

    # Compare post-shin distribution to overall word-initial distribution
    all_initial_dist = Counter()
    for w in decoded:
        all_initial_dist[w[0]] += 1

    # Normalize both
    post_shin_total = sum(post_shin_dist.values())
    all_initial_total = sum(all_initial_dist.values())
    post_shin_pct = {k: v / post_shin_total * 100 for k, v in post_shin_dist.items()}
    all_initial_pct = {k: v / all_initial_total * 100 for k, v in all_initial_dist.items()}

    # Cosine similarity between post-shin and all-initial distributions
    all_letters = sorted(set(post_shin_pct) | set(all_initial_pct))
    v1 = np.array([post_shin_pct.get(c, 0) for c in all_letters])
    v2 = np.array([all_initial_pct.get(c, 0) for c in all_letters])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cosine_post_shin_vs_initial = float(
        np.dot(v1, v2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0

    return {
        'total_decoded': len(decoded),
        'shin_initial_count': len(shin_words),
        'shin_initial_pct': len(shin_words) / max(len(decoded), 1) * 100,
        'shin_strip_test': {
            'stripped_total': shin_stripped_total,
            'stripped_matches': shin_stripped_matches,
            'strip_match_rate': shin_strip_rate,
            'top_stripped_matches': shin_stripped_examples.most_common(20),
            'control_mean_rate': control_mean,
            'control_std': control_std,
            'z_score_vs_controls': shin_z,
        },
        'control_strip_rates': control_rates,
        'post_shin_distribution': dict(post_shin_dist.most_common()),
        'post_shin_pct': {k: round(v, 1) for k, v in
                          sorted(post_shin_pct.items(), key=lambda x: -x[1])},
        'all_initial_pct': {k: round(v, 1) for k, v in
                            sorted(all_initial_pct.items(), key=lambda x: -x[1])},
        'cosine_post_shin_vs_initial': cosine_post_shin_vs_initial,
    }


# =====================================================================
# Part 2: EVA l (mem) — Positional analysis & dual role
# =====================================================================

def analyze_mem_positional(eva_words, lexicon_set, min_len=3):
    """Investigate EVA l (mem) positional distribution and dual-role potential.

    Returns dict with positional breakdown and dual-role test results.
    """
    # Decode all words to parts
    all_parts = []  # list of (hebrew_parts, eva_word)
    for w in eva_words:
        parts = decode_to_hebrew_parts(w)
        if parts and len(parts) >= min_len:
            all_parts.append((parts, w))

    # Count mem positions
    mem_pos = Counter()  # 'initial', 'medial', 'final'
    mem_total = 0
    for parts, _ in all_parts:
        for i, ch in enumerate(parts):
            if ch == 'm':
                mem_total += 1
                if i == 0:
                    mem_pos['initial'] += 1
                elif i == len(parts) - 1:
                    mem_pos['final'] += 1
                else:
                    mem_pos['medial'] += 1

    # Compare with all letters
    all_positions = defaultdict(Counter)
    for parts, _ in all_parts:
        for i, ch in enumerate(parts):
            if i == 0:
                all_positions[ch]['initial'] += 1
            elif i == len(parts) - 1:
                all_positions[ch]['final'] += 1
            else:
                all_positions[ch]['medial'] += 1

    # Medial ratio for each letter
    medial_ratios = {}
    for letter, counts in all_positions.items():
        total = sum(counts.values())
        if total >= 50:
            medial_ratios[letter] = {
                'initial': counts['initial'],
                'medial': counts['medial'],
                'final': counts['final'],
                'total': total,
                'medial_pct': counts['medial'] / total * 100,
            }

    # Dual-role test: replace l in medial position with each unmapped letter
    # For words where l appears medially, try replacing that position
    dual_role_results = []
    for candidate_heb in UNMAPPED_HEBREW:
        n_affected = 0
        gained = 0
        lost = 0
        for parts, eva_w in all_parts:
            original = ''.join(parts)
            has_medial_m = False
            for i, ch in enumerate(parts):
                if ch == 'm' and 0 < i < len(parts) - 1:
                    has_medial_m = True
                    break
            if not has_medial_m:
                continue

            # Try replacing each medial m with candidate
            modified_parts = list(parts)
            for i in range(1, len(parts) - 1):
                if modified_parts[i] == 'm':
                    modified_parts[i] = candidate_heb
            modified = ''.join(modified_parts)

            n_affected += 1
            orig_match = original in lexicon_set
            mod_match = modified in lexicon_set
            if mod_match and not orig_match:
                gained += 1
            elif orig_match and not mod_match:
                lost += 1

        net = gained - lost
        dual_role_results.append({
            'candidate': candidate_heb,
            'candidate_name': CONSONANT_NAMES.get(candidate_heb, candidate_heb),
            'n_affected': n_affected,
            'gained': gained,
            'lost': lost,
            'net': net,
        })

    dual_role_results.sort(key=lambda x: -x['net'])

    return {
        'mem_total': mem_total,
        'mem_positions': dict(mem_pos),
        'mem_initial_pct': mem_pos['initial'] / max(mem_total, 1) * 100,
        'mem_medial_pct': mem_pos['medial'] / max(mem_total, 1) * 100,
        'mem_final_pct': mem_pos['final'] / max(mem_total, 1) * 100,
        'all_letter_medial_ratios': medial_ratios,
        'dual_role_tests': dual_role_results,
    }


# =====================================================================
# Part 3: Allograph scan (both y and l vs all EVA chars)
# =====================================================================

def compute_positional_profile(words, char):
    """Compute positional profile for an EVA character."""
    counts = Counter()
    for w in words:
        for i, c in enumerate(w):
            if c == char:
                if len(w) == 1:
                    counts['isolated'] += 1
                elif i == 0:
                    counts['initial'] += 1
                elif i == len(w) - 1:
                    counts['final'] += 1
                else:
                    counts['medial'] += 1
    total = sum(counts.values())
    return {pos: counts.get(pos, 0) for pos in ['initial', 'medial', 'final']}, total


def compute_bigram_context(words, char):
    """Compute before/after character distributions for an EVA char."""
    before = Counter()
    after = Counter()
    for w in words:
        for i, c in enumerate(w):
            if c == char:
                before[w[i-1] if i > 0 else '#'] += 1
                after[w[i+1] if i < len(w) - 1 else '#'] += 1
    return before, after


def cosine_similarity(profile_a, profile_b):
    """Cosine similarity between two positional profiles."""
    keys = ['initial', 'medial', 'final']
    v1 = np.array([profile_a.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([profile_b.get(k, 0) for k in keys], dtype=float)
    # Normalize to percentages
    if v1.sum() > 0:
        v1 = v1 / v1.sum()
    if v2.sum() > 0:
        v2 = v2 / v2.sum()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def context_overlap(counter_a, counter_b):
    """Weighted context overlap between two bigram distributions."""
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    if total_a == 0 or total_b == 0:
        return 0.0
    all_chars = set(counter_a) | set(counter_b)
    overlap = sum(
        min(counter_a.get(c, 0) / total_a, counter_b.get(c, 0) / total_b)
        for c in all_chars
    )
    return float(overlap)


def allograph_scan(eva_words, target_char):
    """Scan target EVA char against all other EVA chars for allograph potential.

    Returns sorted list of (char, cosine, context_overlap) tuples.
    """
    eva_chars = sorted(set(''.join(eva_words)) - {target_char})
    # Filter to meaningful chars (at least 100 occurrences)
    char_counts = Counter(''.join(eva_words))
    eva_chars = [c for c in eva_chars if char_counts.get(c, 0) >= 100
                 and c.isalpha()]

    target_profile, _ = compute_positional_profile(eva_words, target_char)
    target_before, target_after = compute_bigram_context(eva_words, target_char)

    results = []
    for other in eva_chars:
        other_profile, _ = compute_positional_profile(eva_words, other)
        other_before, other_after = compute_bigram_context(eva_words, other)

        cos = cosine_similarity(target_profile, other_profile)
        ctx_before = context_overlap(target_before, other_before)
        ctx_after = context_overlap(target_after, other_after)
        ctx_combined = (ctx_before + ctx_after) / 2

        results.append({
            'char': other,
            'hebrew': MAPPING.get(other, '?'),
            'cosine': round(cos, 3),
            'context_overlap': round(ctx_combined, 3),
            'ctx_before': round(ctx_before, 3),
            'ctx_after': round(ctx_after, 3),
        })

    results.sort(key=lambda x: -x['cosine'])
    return results


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Deep investigation of EVA y (shin) and EVA l (mem)."""
    report_path = config.stats_dir / "deep_yl_analysis.json"

    if report_path.exists() and not force:
        click.echo("  Deep y/l analysis exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("DEEP INVESTIGATION — EVA y (shin) & EVA l (mem)")

    # Load data
    print_step("Loading EVA words and lexicon...")
    from .word_structure import parse_eva_words
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(Path(eva_file))
    eva_words = eva_data['words']
    click.echo(f"    {len(eva_words)} EVA words loaded")

    lexicon_set = load_lexicon_set(config)
    click.echo(f"    {len(lexicon_set)} Hebrew lexicon forms loaded")

    # =====================================================================
    # Part 1: EVA y (shin)
    # =====================================================================
    print_step("Part 1: EVA y (shin) — She-prefix hypothesis...")
    shin_results = analyze_shin_prefix(eva_words, lexicon_set)

    click.echo(f"\n    Shin-initial words: {shin_results['shin_initial_count']} "
               f"({shin_results['shin_initial_pct']:.1f}% of decoded)")

    st = shin_results['shin_strip_test']
    click.echo(f"\n    SHE-PREFIX STRIPPING TEST:")
    click.echo(f"      Strip initial shin → remainder in lexicon:")
    click.echo(f"        Rate:    {st['strip_match_rate']:.3f} "
               f"({st['stripped_matches']}/{st['stripped_total']})")
    click.echo(f"        Control: {st['control_mean_rate']:.3f} ± "
               f"{st['control_std']:.3f}")
    click.echo(f"        z-score: {st['z_score_vs_controls']:.1f}")

    # Show top control rates for comparison
    click.echo(f"\n    Strip rates by initial letter (top 10):")
    controls = shin_results['control_strip_rates']
    ranked = sorted(controls.items(), key=lambda x: -x[1]['rate'])
    for letter, data in ranked[:10]:
        name = CONSONANT_NAMES.get(letter, letter)
        click.echo(f"      {name:8s} ({letter}): {data['rate']:.3f} "
                   f"({data['stripped_matches']}/{data['stripped_total']})")
    # Show shin in context
    click.echo(f"      {'shin':8s} (S): {st['strip_match_rate']:.3f} "
               f"({st['stripped_matches']}/{st['stripped_total']})  ← TARGET")

    # Post-shin distribution
    click.echo(f"\n    POST-SHIN DISTRIBUTION (what follows initial shin):")
    click.echo(f"      Cosine(post-shin vs all-initial): "
               f"{shin_results['cosine_post_shin_vs_initial']:.3f}")
    click.echo(f"\n      {'Letter':8s} {'Post-S%':>8s} {'All-init%':>10s} {'Δ':>8s}")
    click.echo(f"      {'-'*36}")
    ps = shin_results['post_shin_pct']
    ai = shin_results['all_initial_pct']
    for letter in list(ps.keys())[:10]:
        name = CONSONANT_NAMES.get(letter, letter)
        ps_val = ps.get(letter, 0)
        ai_val = ai.get(letter, 0)
        delta = ps_val - ai_val
        click.echo(f"      {name:8s} {ps_val:>7.1f}% {ai_val:>9.1f}% "
                   f"{delta:>+7.1f}")

    # Top stripped matches
    click.echo(f"\n    TOP STRIPPED SHIN MATCHES (S+X → X in lexicon):")
    for form, count in st['top_stripped_matches'][:15]:
        names = ''.join(CONSONANT_NAMES.get(c, c) for c in form)
        click.echo(f"      S+{form:10s} x{count:>4d}  ({names})")

    # =====================================================================
    # Part 2: EVA l (mem)
    # =====================================================================
    print_step("\nPart 2: EVA l (mem) — Positional & dual-role analysis...")
    mem_results = analyze_mem_positional(eva_words, lexicon_set)

    click.echo(f"\n    Mem total occurrences: {mem_results['mem_total']}")
    click.echo(f"      Initial: {mem_results['mem_positions'].get('initial', 0)} "
               f"({mem_results['mem_initial_pct']:.1f}%)")
    click.echo(f"      Medial:  {mem_results['mem_positions'].get('medial', 0)} "
               f"({mem_results['mem_medial_pct']:.1f}%)")
    click.echo(f"      Final:   {mem_results['mem_positions'].get('final', 0)} "
               f"({mem_results['mem_final_pct']:.1f}%)")

    # Compare medial % with other letters
    click.echo(f"\n    MEDIAL % COMPARISON (all letters):")
    click.echo(f"      {'Letter':8s} {'Init%':>7s} {'Med%':>7s} {'Fin%':>7s} {'Total':>7s}")
    click.echo(f"      {'-'*40}")
    ratios = mem_results['all_letter_medial_ratios']
    for letter in sorted(ratios, key=lambda x: ratios[x]['medial_pct']):
        data = ratios[letter]
        name = CONSONANT_NAMES.get(letter, letter)
        init_pct = data['initial'] / data['total'] * 100
        med_pct = data['medial_pct']
        fin_pct = data['final'] / data['total'] * 100
        marker = " ← MEM" if letter == 'm' else ""
        click.echo(f"      {name:8s} {init_pct:>6.1f}% {med_pct:>6.1f}% "
                   f"{fin_pct:>6.1f}% {data['total']:>6d}{marker}")

    # Dual-role test results
    click.echo(f"\n    DUAL-ROLE TEST: l@medial → unmapped Hebrew letter")
    click.echo(f"      {'Candidate':12s} {'Affected':>8s} {'Gained':>7s} "
               f"{'Lost':>6s} {'Net':>6s}")
    click.echo(f"      {'-'*42}")
    for result in mem_results['dual_role_tests']:
        click.echo(f"      {result['candidate_name']:12s} "
                   f"{result['n_affected']:>8d} {result['gained']:>7d} "
                   f"{result['lost']:>6d} {result['net']:>+6d}")

    # =====================================================================
    # Part 3: Allograph scans
    # =====================================================================
    print_step("\nPart 3: Allograph scans...")

    click.echo(f"\n    EVA y (shin) vs all other EVA chars:")
    y_allographs = allograph_scan(eva_words, 'y')
    click.echo(f"      {'Char':6s} {'Hebrew':8s} {'Cosine':>8s} {'Context':>8s}")
    click.echo(f"      {'-'*32}")
    for r in y_allographs[:10]:
        name = CONSONANT_NAMES.get(r['hebrew'], r['hebrew'])
        click.echo(f"      {r['char']:6s} {name:8s} {r['cosine']:>8.3f} "
                   f"{r['context_overlap']:>8.3f}")

    click.echo(f"\n    EVA l (mem) vs all other EVA chars:")
    l_allographs = allograph_scan(eva_words, 'l')
    click.echo(f"      {'Char':6s} {'Hebrew':8s} {'Cosine':>8s} {'Context':>8s}")
    click.echo(f"      {'-'*32}")
    for r in l_allographs[:10]:
        name = CONSONANT_NAMES.get(r['hebrew'], r['hebrew'])
        click.echo(f"      {r['char']:6s} {name:8s} {r['cosine']:>8.3f} "
                   f"{r['context_overlap']:>8.3f}")

    # =====================================================================
    # Verdicts
    # =====================================================================
    click.echo(f"\n{'='*60}")
    click.echo("  VERDICTS")
    click.echo(f"{'='*60}")

    # Shin verdict
    shin_z = st['z_score_vs_controls']
    shin_cosine = shin_results['cosine_post_shin_vs_initial']
    click.echo(f"\n  EVA y (shin):")
    if shin_z > 3 and shin_cosine > 0.8:
        click.echo(f"    She-prefix STRONGLY SUPPORTED (z={shin_z:.1f}, "
                   f"post-shin~initial cosine={shin_cosine:.3f})")
    elif shin_z > 2:
        click.echo(f"    She-prefix moderately supported (z={shin_z:.1f}, "
                   f"cosine={shin_cosine:.3f})")
    elif shin_z > 1:
        click.echo(f"    She-prefix weakly supported (z={shin_z:.1f}, "
                   f"cosine={shin_cosine:.3f})")
    else:
        click.echo(f"    She-prefix NOT supported (z={shin_z:.1f}, "
                   f"cosine={shin_cosine:.3f})")

    # Check allograph potential
    y_top = y_allographs[0] if y_allographs else None
    if y_top and y_top['cosine'] > 0.95 and y_top['context_overlap'] > 0.5:
        click.echo(f"    ALLOGRAPH CANDIDATE: y ~ {y_top['char']} "
                   f"(cos={y_top['cosine']}, ctx={y_top['context_overlap']})")
    else:
        click.echo(f"    No allograph candidates (top: "
                   f"{y_top['char'] if y_top else '?'} "
                   f"cos={y_top['cosine'] if y_top else 0}, "
                   f"ctx={y_top['context_overlap'] if y_top else 0})")

    # Mem verdict
    click.echo(f"\n  EVA l (mem):")
    best_dual = mem_results['dual_role_tests'][0] if mem_results['dual_role_tests'] else None
    if best_dual and best_dual['net'] > 100:
        click.echo(f"    DUAL ROLE CANDIDATE: l@medial → "
                   f"{best_dual['candidate_name']} "
                   f"(net={best_dual['net']:+d})")
    elif best_dual and best_dual['net'] > 0:
        click.echo(f"    Weak dual-role signal: l@medial → "
                   f"{best_dual['candidate_name']} "
                   f"(net={best_dual['net']:+d})")
    else:
        click.echo(f"    No dual-role improvement found")

    l_top = l_allographs[0] if l_allographs else None
    if l_top and l_top['cosine'] > 0.95 and l_top['context_overlap'] > 0.5:
        click.echo(f"    ALLOGRAPH CANDIDATE: l ~ {l_top['char']} "
                   f"(cos={l_top['cosine']}, ctx={l_top['context_overlap']})")
    else:
        click.echo(f"    No allograph candidates (top: "
                   f"{l_top['char'] if l_top else '?'} "
                   f"cos={l_top['cosine'] if l_top else 0}, "
                   f"ctx={l_top['context_overlap'] if l_top else 0})")

    click.echo(f"{'='*60}")

    # =====================================================================
    # Save report
    # =====================================================================
    print_step("Saving report...")
    report = {
        'shin_analysis': shin_results,
        'mem_analysis': mem_results,
        'y_allograph_scan': y_allographs,
        'l_allograph_scan': l_allographs,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")
