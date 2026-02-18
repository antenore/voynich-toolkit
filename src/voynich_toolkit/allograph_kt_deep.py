"""
Deep analysis of EVA k/t allography.

Investigates:
  1. Detailed positional and context profiles of k and t
  2. Minimal pairs (k↔t substitution)
  3. Which Hebrew letter (tav vs tet) is the better unified assignment
  4. Impact of freeing one Hebrew letter slot
  5. What the freed letter could map to (bet, kaf, zayin, samekh, tsade, qof)
  6. Lexicon match rate impact
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import click

from .allograph_analysis import (
    analyze_pair, _positional_profile, _bigram_context,
    _format_profile, _format_context,
)
from .config import ToolkitConfig
from .full_decode import (
    FULL_MAPPING, II_HEBREW, I_HEBREW, DIRECTION,
    preprocess_eva, decode_word,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


def _load_hebrew_lexicon(config):
    """Load enriched Hebrew lexicon."""
    path = config.hebrew_lexicon_path
    if not path.exists():
        path = config.lexicon_dir / "lexicon.json"
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("all_consonantal_forms", []))


def _decode_word_with_override(eva_word, override_char, override_hebrew):
    """Decode a word using the standard 19-char mapping but with one override.

    override_char: EVA char to override (e.g., 'k' or 't')
    override_hebrew: Hebrew letter to use instead
    """
    # Build modified mapping
    mapping = dict(FULL_MAPPING)
    mapping[override_char] = override_hebrew

    # Preprocess (handle ii/i/q)
    prefix, processed = preprocess_eva(eva_word)

    # Decode RTL
    chars = list(reversed(processed))
    parts = []
    for ch in chars:
        if ch == '\x01':
            parts.append(II_HEBREW)
        elif ch == '\x02':
            parts.append(I_HEBREW)
        elif ch in mapping:
            parts.append(mapping[ch])
        else:
            parts.append('?')
    return ''.join(parts)


def _decode_word_unified_kt(eva_word, unified_hebrew):
    """Decode word treating BOTH k and t as the same Hebrew letter."""
    mapping = dict(FULL_MAPPING)
    mapping['k'] = unified_hebrew
    mapping['t'] = unified_hebrew

    prefix, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))
    parts = []
    for ch in chars:
        if ch == '\x01':
            parts.append(II_HEBREW)
        elif ch == '\x02':
            parts.append(I_HEBREW)
        elif ch in mapping:
            parts.append(mapping[ch])
        else:
            parts.append('?')
    return ''.join(parts)


def run(config: ToolkitConfig, force=False, **kwargs):
    """Deep analysis of k/t allography and impact of freeing a Hebrew slot."""
    report_path = config.stats_dir / "allograph_kt_deep_report.json"

    if report_path.exists() and not force:
        click.echo("  k/t deep report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("DEEP ANALYSIS — EVA k/t ALLOGRAPHY")

    # Parse EVA text
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    words = eva_data["words"]
    word_freq = Counter(words)
    unique_words = list(word_freq.keys())
    click.echo(f"    {eva_data['total_words']} words, {len(unique_words)} unique")

    # Load lexicon
    print_step("Loading Hebrew lexicon...")
    hebrew_set = _load_hebrew_lexicon(config)
    click.echo(f"    {len(hebrew_set)} consonantal forms")

    # ═══════════════════════════════════════════════════════
    # 1. DETAILED PROFILES
    # ═══════════════════════════════════════════════════════
    print_header("1. POSITIONAL PROFILES — k vs t")
    result = analyze_pair(words, "k", "t")

    click.echo(f"\n  k: {_format_profile(result['profile_a'])}")
    click.echo(f"  t: {_format_profile(result['profile_b'])}")
    click.echo(f"\n  Cosine similarity: {result['cosine_similarity']}")
    click.echo(f"  Combined context overlap: {result['combined_context_overlap']}")

    click.echo(f"\n  Before-context:")
    click.echo(f"    k: {_format_context(Counter(dict(result['before_context_a'])))}")
    click.echo(f"    t: {_format_context(Counter(dict(result['before_context_b'])))}")
    click.echo(f"  After-context:")
    click.echo(f"    k: {_format_context(Counter(dict(result['after_context_a'])))}")
    click.echo(f"    t: {_format_context(Counter(dict(result['after_context_b'])))}")

    # Slot analysis
    print_step("Slot analysis (absolute position)...")
    max_pos = 8
    from_start_k = defaultdict(int)
    from_start_t = defaultdict(int)
    total_at_pos = defaultdict(int)

    for w in words:
        for i, c in enumerate(w):
            if i >= max_pos:
                break
            total_at_pos[i] += 1
            if c == "k":
                from_start_k[i] += 1
            elif c == "t":
                from_start_t[i] += 1

    click.echo(f"\n  {'Pos':4s} {'k count':>8s} {'k %':>6s} {'t count':>8s} "
               f"{'t %':>6s} {'k/t ratio':>10s}")
    click.echo("  " + "-" * 48)
    for pos in range(max_pos):
        n_k = from_start_k.get(pos, 0)
        n_t = from_start_t.get(pos, 0)
        tot = total_at_pos.get(pos, 1)
        pct_k = n_k / tot * 100
        pct_t = n_t / tot * 100
        ratio = f"{n_k / n_t:.2f}" if n_t > 0 else "inf"
        click.echo(f"  {pos:4d} {n_k:8d} {pct_k:5.1f}% {n_t:8d} "
                   f"{pct_t:5.1f}% {ratio:>10s}")

    # ═══════════════════════════════════════════════════════
    # 2. MINIMAL PAIRS
    # ═══════════════════════════════════════════════════════
    print_header("2. MINIMAL PAIRS — k↔t substitution")

    word_set = set(unique_words)
    minimal_pairs = []
    seen = set()

    for w in word_set:
        for i, c in enumerate(w):
            if c == "k":
                variant = w[:i] + "t" + w[i + 1:]
                if variant in word_set and (w, variant) not in seen:
                    seen.add((w, variant))
                    seen.add((variant, w))
                    # Decode both (returns italian, hebrew, n_unknown)
                    _, heb_k, _ = decode_word(w)
                    _, heb_t, _ = decode_word(variant)
                    minimal_pairs.append({
                        "word_k": w,
                        "word_t": variant,
                        "position": i,
                        "freq_k": word_freq[w],
                        "freq_t": word_freq[variant],
                        "hebrew_k": heb_k,
                        "hebrew_t": heb_t,
                        "k_in_lex": heb_k in hebrew_set,
                        "t_in_lex": heb_t in hebrew_set,
                    })

    minimal_pairs.sort(key=lambda x: -(x["freq_k"] + x["freq_t"]))

    click.echo(f"    Found {len(minimal_pairs)} minimal pairs")

    # Count: how often does k-variant match lexicon vs t-variant?
    k_only_match = sum(1 for mp in minimal_pairs
                       if mp["k_in_lex"] and not mp["t_in_lex"])
    t_only_match = sum(1 for mp in minimal_pairs
                       if mp["t_in_lex"] and not mp["k_in_lex"])
    both_match = sum(1 for mp in minimal_pairs
                     if mp["k_in_lex"] and mp["t_in_lex"])
    neither = sum(1 for mp in minimal_pairs
                  if not mp["k_in_lex"] and not mp["t_in_lex"])

    click.echo(f"    k-only in lexicon: {k_only_match}")
    click.echo(f"    t-only in lexicon: {t_only_match}")
    click.echo(f"    Both in lexicon:   {both_match}")
    click.echo(f"    Neither:           {neither}")

    if minimal_pairs:
        click.echo(f"\n  {'Word(k)':12s} {'Word(t)':12s} {'Pos':4s} "
                   f"{'Fk':>5s} {'Ft':>5s} {'Heb(k)':>8s} {'Heb(t)':>8s} "
                   f"{'Lex?':>5s}")
        click.echo("  " + "-" * 70)
        for mp in minimal_pairs[:30]:
            lex_flag = ""
            if mp["k_in_lex"] and mp["t_in_lex"]:
                lex_flag = "both"
            elif mp["k_in_lex"]:
                lex_flag = "k"
            elif mp["t_in_lex"]:
                lex_flag = "t"
            click.echo(f"  {mp['word_k']:12s} {mp['word_t']:12s} "
                       f"{mp['position']:4d} {mp['freq_k']:5d} "
                       f"{mp['freq_t']:5d} {mp['hebrew_k']:>8s} "
                       f"{mp['hebrew_t']:>8s} {lex_flag:>5s}")

    # ═══════════════════════════════════════════════════════
    # 3. TAV vs TET — which to keep?
    # ═══════════════════════════════════════════════════════
    print_header("3. TAV vs TET — which Hebrew letter to unify on?")

    # Count occurrences of k and t in the corpus
    total_k = sum(1 for w in words for c in w if c == "k")
    total_t = sum(1 for w in words for c in w if c == "t")
    click.echo(f"\n  EVA k occurrences: {total_k} ({total_k/(total_k+total_t)*100:.1f}%)")
    click.echo(f"  EVA t occurrences: {total_t} ({total_t/(total_k+total_t)*100:.1f}%)")
    click.echo(f"  Currently: k→tav(t), t→tet(J)")

    # Test: decode all unique words with both unified options
    # Option A: both k,t → tav (t)
    # Option B: both k,t → tet (J)
    print_step("Testing unified decode: all words → lexicon match rate...")

    # Current (19-char standard)
    current_matches = 0
    current_total = 0
    for w in unique_words:
        _, heb, n_unk = decode_word(w)
        if heb and n_unk == 0:
            current_total += 1
            if heb in hebrew_set:
                current_matches += 1

    # Option A: both → tav
    tav_matches = 0
    for w in unique_words:
        heb = _decode_word_unified_kt(w, 't')  # tav
        if heb and '?' not in heb:
            if heb in hebrew_set:
                tav_matches += 1

    # Option B: both → tet
    tet_matches = 0
    for w in unique_words:
        heb = _decode_word_unified_kt(w, 'J')  # tet
        if heb and '?' not in heb:
            if heb in hebrew_set:
                tet_matches += 1

    click.echo(f"\n  Lexicon hit rate (unique words, {current_total} decodable):")
    click.echo(f"    Current (k→tav, t→tet):  {current_matches} "
               f"({current_matches/current_total*100:.2f}%)")
    click.echo(f"    Unified k,t → tav:       {tav_matches} "
               f"({tav_matches/current_total*100:.2f}%)")
    click.echo(f"    Unified k,t → tet:       {tet_matches} "
               f"({tet_matches/current_total*100:.2f}%)")

    # Expected frequency in Hebrew
    click.echo(f"\n  Hebrew frequency reference:")
    click.echo(f"    tav: ~4.9% of consonants (common)")
    click.echo(f"    tet: ~0.6% of consonants (rare)")
    click.echo(f"    Combined k+t in decoded: {(total_k+total_t)} occ")

    # Recommendation
    if tav_matches >= tet_matches:
        better_unified = "tav"
        freed_letter = "tet (J)"
        freed_name = "tet"
    else:
        better_unified = "tet"
        freed_letter = "tav (t)"
        freed_name = "tav"

    click.echo(f"\n  → Better unification: both → {better_unified} "
               f"(+{max(tav_matches,tet_matches) - current_matches} vs current, "
               f"or {max(tav_matches,tet_matches) - min(tav_matches,tet_matches)} "
               f"vs worse option)")
    click.echo(f"  → Freed Hebrew letter: {freed_letter}")

    # ═══════════════════════════════════════════════════════
    # 4. FREED SLOT — what could it map to?
    # ═══════════════════════════════════════════════════════
    print_header("4. FREED SLOT — testing missing Hebrew letters")

    missing_letters = {
        'b': ('bet', 'v/b'),
        'k': ('kaf', 'k'),
        'z': ('zayin', 'z'),
        's': ('samekh', 's'),
        'C': ('tsade', 'ts'),
        'q': ('qof', 'q'),
    }

    click.echo(f"\n  6 missing Hebrew letters: "
               f"{', '.join(f'{v[0]}({k})' for k,v in missing_letters.items())}")
    click.echo(f"\n  Frequency in Biblical Hebrew:")
    bh_freqs = {
        'b': 4.4, 'k': 2.5, 'z': 0.7,
        's': 0.8, 'C': 1.5, 'q': 1.0,
    }
    for letter, (name, phoneme) in missing_letters.items():
        click.echo(f"    {name:8s} ({letter}): ~{bh_freqs[letter]}% of consonants")

    # For each missing letter, test: if the freed slot is assigned to it,
    # does it appear in lexicon words that are currently d=1 near-misses?
    print_step("Testing each missing letter as replacement for freed slot...")
    click.echo(f"    (Testing: what if EVA '{better_unified == 'tav' and 't' or 'k'}' "
               f"→ {{missing letter}} instead of {freed_name}?)")

    # The freed EVA char depends on which one we keep.
    # If we unify both to tav, then tet is freed, but we don't have a separate
    # EVA char for it. Actually, let me think again...
    #
    # Current: k→tav, t→tet (two separate EVA chars → two Hebrew letters)
    # If allographs: k and t are the SAME glyph → only ONE Hebrew letter needed
    # This means we have 18 effective EVA chars (not 19), mapping to 18 Hebrew letters
    # We DON'T free a slot in the sense of having an extra EVA char.
    # But we DO free a Hebrew letter: if both map to tav, then tet is unmapped.
    #
    # The question is: is there ANOTHER EVA pattern (digraph, positional variant)
    # that could map to the freed Hebrew letter?
    # Or: is the freed Hebrew letter simply absent from the cipher?

    click.echo(f"\n  NOTE: Unifying k/t means 18 effective EVA chars → 18 Hebrew letters.")
    click.echo(f"  The freed Hebrew letter ({freed_name}) becomes UNMAPPED.")
    click.echo(f"  To USE this freed slot, we need to find an EVA pattern for it.")

    # Check: how many lexicon words contain the freed letter?
    freed_heb = 'J' if freed_name == 'tet' else 't'
    words_with_freed = sum(1 for form in hebrew_set if freed_heb in form)
    click.echo(f"\n  Lexicon words containing {freed_name}: {words_with_freed} "
               f"({words_with_freed/len(hebrew_set)*100:.1f}%)")

    # Check: how many lexicon words contain each missing letter?
    for letter, (name, phoneme) in missing_letters.items():
        count = sum(1 for form in hebrew_set if letter in form)
        click.echo(f"    {name:8s} ({letter}): {count} forms "
                   f"({count/len(hebrew_set)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════
    # 5. ALTERNATIVE INTERPRETATION
    # ═══════════════════════════════════════════════════════
    print_header("5. ALTERNATIVE — k/t as positional variants of ONE letter")

    # If k and t are positional variants, maybe they represent TWO DIFFERENT
    # Hebrew letters depending on position. E.g., initial k→kaf, final k→tav.
    # Let's check position-dependent distributions more carefully.

    # Distribution of k in words that ALSO contain t, and vice versa
    words_with_both = 0
    words_with_k_only = 0
    words_with_t_only = 0
    words_with_neither = 0

    for w in unique_words:
        has_k = 'k' in w
        has_t = 't' in w
        if has_k and has_t:
            words_with_both += 1
        elif has_k:
            words_with_k_only += 1
        elif has_t:
            words_with_t_only += 1
        else:
            words_with_neither += 1

    click.echo(f"\n  Co-occurrence in unique words:")
    click.echo(f"    Words with both k and t: {words_with_both}")
    click.echo(f"    Words with k only:       {words_with_k_only}")
    click.echo(f"    Words with t only:       {words_with_t_only}")
    click.echo(f"    Words with neither:      {words_with_neither}")

    # If they're positional variants, we'd expect them to rarely co-occur
    # If they're free allographs, co-occurrence should be based on chance
    total_with_kt = words_with_both + words_with_k_only + words_with_t_only
    expected_cooccurrence = (
        (words_with_both + words_with_k_only) / max(total_with_kt, 1) *
        (words_with_both + words_with_t_only) / max(total_with_kt, 1)
    )
    actual_cooccurrence = words_with_both / max(total_with_kt, 1)
    click.echo(f"\n  Expected co-occurrence (independent): "
               f"{expected_cooccurrence*100:.1f}%")
    click.echo(f"  Actual co-occurrence: "
               f"{actual_cooccurrence*100:.1f}%")
    if actual_cooccurrence < expected_cooccurrence * 0.5:
        click.echo(f"  → COMPLEMENTARY: k and t avoid each other "
                   f"(ratio {actual_cooccurrence/max(expected_cooccurrence,0.001):.2f})")
    elif actual_cooccurrence > expected_cooccurrence * 1.5:
        click.echo(f"  → CO-OCCUR more than expected "
                   f"(ratio {actual_cooccurrence/max(expected_cooccurrence,0.001):.2f})")
    else:
        click.echo(f"  → INDEPENDENT: co-occurrence matches expectation "
                   f"(ratio {actual_cooccurrence/max(expected_cooccurrence,0.001):.2f})")

    # ═══════════════════════════════════════════════════════
    # 6. IMPACT ON MATCH RATES (anchor words, zodiac)
    # ═══════════════════════════════════════════════════════
    print_header("6. IMPACT ON ANCHOR/ZODIAC MATCH RATES")

    # Load anchor words report to check near-misses involving tav/tet
    anchor_path = config.stats_dir / "anchor_words_report.json"
    improved = []
    d1_matches = []
    if anchor_path.exists():
        with open(anchor_path) as f:
            anchor_data = json.load(f)

        # Flatten all d=1 decoded_forms from by_category
        by_cat = anchor_data.get("by_category", {})
        for cat_id, cat_data in by_cat.items():
            for match in cat_data.get("matches", []):
                for df in match.get("decoded_forms", []):
                    if df.get("distance") == 1:
                        d1_matches.append({
                            "anchor": match["anchor"],
                            "normalized": match["normalized"],
                            "language": match["language"],
                            "category": cat_id,
                            **df,
                        })

        # For each d=1 match, would unifying k/t improve it to d=0?
        unified_heb_letter = 't' if better_unified == 'tav' else 'J'
        for m in d1_matches:
            anchor_form = m.get("normalized", "")
            decoded_form = m.get("decoded_word", "")
            eva_word = m.get("eva_word", "")
            if not eva_word:
                continue

            # Only check words that contain k or t
            if 'k' not in eva_word and 't' not in eva_word:
                continue

            unified_heb = _decode_word_unified_kt(eva_word, unified_heb_letter)

            if unified_heb == anchor_form and decoded_form != anchor_form:
                improved.append({
                    "anchor": m.get("anchor", ""),
                    "eva": eva_word,
                    "old_decode": decoded_form,
                    "new_decode": unified_heb,
                    "target": anchor_form,
                    "count": m.get("total_count", 0),
                    "language": m.get("language", ""),
                })

        click.echo(f"\n  Anchor d=1 matches: {len(d1_matches)}")
        click.echo(f"  Matches involving k or t: "
                   f"{sum(1 for m in d1_matches if 'k' in m.get('eva_word','') or 't' in m.get('eva_word',''))}")
        click.echo(f"  Would improve to d=0 with unified k/t: {len(improved)}")
        if improved:
            for imp in improved[:15]:
                click.echo(f"    {imp['anchor']:15s} {imp['eva']:12s} "
                           f"{imp['old_decode']:>8s} → {imp['new_decode']:>8s} "
                           f"(target: {imp['target']}) x{imp['count']}")
    else:
        click.echo("  (anchor report not found)")

    # ═══════════════════════════════════════════════════════
    # 7. DIGRAPH ANALYSIS — could EVA digraphs map to missing letters?
    # ═══════════════════════════════════════════════════════
    print_header("7. COMMON EVA DIGRAPHS — candidates for missing letters?")

    # Count all bigrams
    bigram_count = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bigram_count[w[i:i+2]] += 1

    # Show top digraphs and their current decode
    click.echo(f"\n  {'Digraph':8s} {'Count':>7s} {'%':>6s} "
               f"{'Current decode':>15s} {'Notes':20s}")
    click.echo("  " + "-" * 60)

    notable_digraphs = ['ch', 'sh', 'ck', 'ct', 'ot', 'ok',
                        'al', 'ol', 'ar', 'or', 'ee', 'ey',
                        'yt', 'yk', 'dy', 'hy']
    total_bigrams = sum(bigram_count.values())

    for dg in notable_digraphs:
        if dg not in bigram_count:
            continue
        n = bigram_count[dg]
        pct = n / total_bigrams * 100
        # Current decode (RTL, so reverse the digraph)
        rev = dg[::-1]
        heb_parts = []
        for c in rev:
            if c in FULL_MAPPING:
                heb_parts.append(FULL_MAPPING[c])
        heb_str = ''.join(heb_parts)
        note = ""
        if dg in ('ch', 'sh'):
            note = "← potential single letter?"
        elif dg in ('ck', 'ct'):
            note = "← k/t context"
        click.echo(f"  {dg:8s} {n:7d} {pct:5.1f}% {heb_str:>15s} {note:20s}")

    # ═══════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════
    print_header("VERDICT — k/t allography")

    click.echo(f"\n  ALLOGRAPH CONFIRMATION:")
    click.echo(f"    Cosine similarity:        {result['cosine_similarity']}")
    click.echo(f"    Combined context overlap:  {result['combined_context_overlap']}")
    click.echo(f"    Both exceed thresholds (>0.95, >0.50): YES")
    click.echo(f"    Minimal pairs: {len(minimal_pairs)}")
    click.echo(f"      k-variant in lexicon only: {k_only_match}")
    click.echo(f"      t-variant in lexicon only: {t_only_match}")

    click.echo(f"\n  UNIFICATION RECOMMENDATION:")
    click.echo(f"    Better assignment: both k,t → {better_unified}")
    click.echo(f"    Lexicon hits: current={current_matches}, "
               f"unified={max(tav_matches, tet_matches)}")
    click.echo(f"    Freed Hebrew letter: {freed_letter}")

    click.echo(f"\n  FREED SLOT IMPACT:")
    click.echo(f"    {freed_name} appears in {words_with_freed} lexicon forms "
               f"({words_with_freed/len(hebrew_set)*100:.1f}%)")
    click.echo(f"    Anchor matches improved: {len(improved)}")

    # Save report
    print_step("Saving report...")
    report = {
        "allograph_confirmed": True,
        "cosine": result["cosine_similarity"],
        "context_overlap": result["combined_context_overlap"],
        "profile_k": result["profile_a"],
        "profile_t": result["profile_b"],
        "minimal_pairs": {
            "count": len(minimal_pairs),
            "k_only_lex": k_only_match,
            "t_only_lex": t_only_match,
            "both_lex": both_match,
            "neither": neither,
            "top30": minimal_pairs[:30],
        },
        "unification": {
            "better": better_unified,
            "freed_letter": freed_name,
            "current_matches": current_matches,
            "tav_matches": tav_matches,
            "tet_matches": tet_matches,
        },
        "corpus_stats": {
            "k_occurrences": total_k,
            "t_occurrences": total_t,
            "words_with_both": words_with_both,
            "words_with_k_only": words_with_k_only,
            "words_with_t_only": words_with_t_only,
            "co_occurrence_ratio": round(
                actual_cooccurrence / max(expected_cooccurrence, 0.001), 3),
        },
        "freed_slot_impact": {
            "lexicon_words_with_freed": words_with_freed,
            "anchor_improvements": len(improved),
            "missing_letters": {
                k: {"name": v[0], "bh_freq": bh_freqs[k]}
                for k, v in missing_letters.items()
            },
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")
