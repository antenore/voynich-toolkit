"""
Copyist error analysis in EVA space.

Works entirely in EVA (no decoding needed). Finds word pairs with
small Levenshtein distance, classifies error types, and analyzes
positional patterns to distinguish scribal errors from intentional cipher.

Error types:
  - visual confusion: similar-looking glyph swaps (a/o, e/i, n/r, etc.)
  - dittography: repeated letter/syllable insertion
  - haplography: deleted repeated letter/syllable
  - metathesis: adjacent character swap
  - substitution: single char replacement (not visual)
"""
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import click
from rapidfuzz.distance import Levenshtein

from .config import ToolkitConfig
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Constants
# =====================================================================

# Pairs of EVA chars that look visually similar and could be confused
# by a copyist. Based on EVA glyph shape analysis.
VISUAL_CONFUSION_PAIRS = {
    frozenset({'a', 'o'}),     # both round, differ in closure
    frozenset({'e', 'i'}),     # both short vertical strokes
    frozenset({'n', 'r'}),     # similar short strokes
    frozenset({'c', 'e'}),     # c-shape vs e-shape
    frozenset({'d', 's'}),     # similar height
    frozenset({'k', 't'}),     # tall chars, similar top
    frozenset({'f', 'p'}),     # gallows variants
    frozenset({'l', 'r'}),     # short strokes
    frozenset({'m', 'n'}),     # differ by one stroke
    frozenset({'i', 'n'}),     # both short
    frozenset({'c', 'h'}),     # bench variants
}


@dataclass
class ErrorPair:
    """A pair of words differing by small Levenshtein distance."""
    word_a: str
    word_b: str
    freq_a: int
    freq_b: int
    distance: int
    error_type: str
    detail: str


# =====================================================================
# Error classification
# =====================================================================

def classify_edit(word_a, word_b):
    """Classify the type of edit between two words (Lev distance 1 or 2).

    Returns (error_type, detail_string).
    """
    if len(word_a) == len(word_b):
        # Same length: substitution or metathesis
        diffs = [(i, word_a[i], word_b[i])
                 for i in range(len(word_a)) if word_a[i] != word_b[i]]

        if len(diffs) == 1:
            i, ca, cb = diffs[0]
            pair = frozenset({ca, cb})
            if pair in VISUAL_CONFUSION_PAIRS:
                return "visual_confusion", f"pos {i}: {ca}<->{cb}"
            return "substitution", f"pos {i}: {ca}->{cb}"

        if len(diffs) == 2:
            i1, ca1, cb1 = diffs[0]
            i2, ca2, cb2 = diffs[1]
            # Metathesis: chars swapped at adjacent positions
            if (i2 == i1 + 1 and ca1 == cb2 and ca2 == cb1):
                return "metathesis", f"pos {i1}-{i2}: {ca1}{ca2}<->{ca2}{ca1}"
            # Double visual confusion
            p1 = frozenset({ca1, cb1})
            p2 = frozenset({ca2, cb2})
            if (p1 in VISUAL_CONFUSION_PAIRS
                    and p2 in VISUAL_CONFUSION_PAIRS):
                return "visual_confusion", (f"pos {i1}: {ca1}<->{cb1}, "
                                            f"pos {i2}: {ca2}<->{cb2}")
            # Double substitution
            return "substitution", (f"pos {i1}: {ca1}->{cb1}, "
                                    f"pos {i2}: {ca2}->{cb2}")

    elif abs(len(word_a) - len(word_b)) == 1:
        # Different length by 1: insertion or deletion
        longer = word_a if len(word_a) > len(word_b) else word_b
        shorter = word_b if len(word_a) > len(word_b) else word_a

        # Find the position of the extra char
        for i in range(len(longer)):
            candidate = longer[:i] + longer[i+1:]
            if candidate == shorter:
                extra_ch = longer[i]
                # Dittography: the inserted char is same as neighbor
                is_ditto = False
                if i > 0 and longer[i-1] == extra_ch:
                    is_ditto = True
                if i < len(longer) - 1 and longer[i+1] == extra_ch:
                    is_ditto = True

                if is_ditto:
                    if len(word_a) > len(word_b):
                        return "dittography", f"pos {i}: {extra_ch} repeated"
                    else:
                        return "haplography", f"pos {i}: {extra_ch} dropped"

                if len(word_a) > len(word_b):
                    return "insertion", f"pos {i}: +{extra_ch}"
                else:
                    return "deletion", f"pos {i}: -{extra_ch}"

    elif abs(len(word_a) - len(word_b)) == 2:
        # Different by 2
        return "length_diff_2", f"|{len(word_a)}| vs |{len(word_b)}|"

    return "other", f"dist={Levenshtein.distance(word_a, word_b)}"


# =====================================================================
# Core analysis
# =====================================================================

def find_similar_pairs(word_freq, max_dist=2, min_freq=2):
    """Find all word pairs with Levenshtein distance <= max_dist.

    Uses length-filtering to reduce O(n^2) comparisons.
    Only considers words with frequency >= min_freq.

    Args:
        word_freq: dict {word: count}
        max_dist: maximum Levenshtein distance
        min_freq: minimum word frequency

    Returns:
        list of ErrorPair
    """
    # Filter and bucket by length
    filtered = [(w, c) for w, c in word_freq.items() if c >= min_freq]
    filtered.sort(key=lambda x: -x[1])

    by_length = defaultdict(list)
    for w, c in filtered:
        by_length[len(w)].append((w, c))

    pairs = []
    seen = set()

    lengths = sorted(by_length.keys())
    for i, la in enumerate(lengths):
        words_a = by_length[la]
        # Compare with same length and lengths within max_dist
        for lb in lengths[i:]:
            if lb - la > max_dist:
                break
            words_b = by_length[lb]

            if la == lb:
                # Same-length: compare all pairs within bucket
                for ia in range(len(words_a)):
                    wa, ca = words_a[ia]
                    for ib in range(ia + 1, len(words_b)):
                        wb, cb = words_b[ib]
                        d = Levenshtein.distance(wa, wb,
                                                 score_cutoff=max_dist)
                        if d <= max_dist:
                            key = (min(wa, wb), max(wa, wb))
                            if key not in seen:
                                seen.add(key)
                                etype, detail = classify_edit(wa, wb)
                                pairs.append(ErrorPair(
                                    word_a=wa, word_b=wb,
                                    freq_a=ca, freq_b=cb,
                                    distance=d,
                                    error_type=etype,
                                    detail=detail,
                                ))
            else:
                # Different lengths
                for wa, ca in words_a:
                    for wb, cb in words_b:
                        d = Levenshtein.distance(wa, wb,
                                                 score_cutoff=max_dist)
                        if d <= max_dist:
                            key = (min(wa, wb), max(wa, wb))
                            if key not in seen:
                                seen.add(key)
                                etype, detail = classify_edit(wa, wb)
                                pairs.append(ErrorPair(
                                    word_a=wa, word_b=wb,
                                    freq_a=ca, freq_b=cb,
                                    distance=d,
                                    error_type=etype,
                                    detail=detail,
                                ))

    return pairs


def analyze_error_types(pairs):
    """Aggregate statistics by error type.

    Returns dict {error_type: {count, pct, top_examples}}.
    """
    by_type = defaultdict(list)
    for p in pairs:
        by_type[p.error_type].append(p)

    total = len(pairs) or 1
    result = {}
    for etype, eps in sorted(by_type.items(), key=lambda x: -len(x[1])):
        examples = sorted(eps, key=lambda p: -(p.freq_a + p.freq_b))[:10]
        result[etype] = {
            "count": len(eps),
            "pct": round(100 * len(eps) / total, 1),
            "top_examples": [
                {
                    "word_a": e.word_a,
                    "word_b": e.word_b,
                    "freq_a": e.freq_a,
                    "freq_b": e.freq_b,
                    "detail": e.detail,
                }
                for e in examples
            ],
        }
    return result


def analyze_confusion_matrix(pairs):
    """Build character-level confusion matrix from substitution pairs.

    Returns dict of {(char_a, char_b): count}.
    """
    confusion = Counter()
    for p in pairs:
        if p.error_type in ("visual_confusion", "substitution"):
            if len(p.word_a) == len(p.word_b):
                for i in range(len(p.word_a)):
                    if p.word_a[i] != p.word_b[i]:
                        pair = tuple(sorted([p.word_a[i], p.word_b[i]]))
                        confusion[pair] += 1
    return confusion


def analyze_by_section(eva_data, pairs, word_freq):
    """Analyze error rates by manuscript section.

    Returns dict {section: {n_words, n_unique, n_pairs_involved,
                            error_rate_pct}}.
    """
    # Build set of all words involved in error pairs
    pair_words = set()
    for p in pairs:
        pair_words.add(p.word_a)
        pair_words.add(p.word_b)

    section_stats = defaultdict(lambda: {
        "n_words": 0, "n_unique": 0, "pair_words": set()
    })

    for page in eva_data["pages"]:
        sec = page.get("section", "?")
        words = page["words"]
        section_stats[sec]["n_words"] += len(words)
        unique_here = set(words)
        section_stats[sec]["n_unique"] += len(unique_here)
        for w in unique_here:
            if w in pair_words:
                section_stats[sec]["pair_words"].add(w)

    result = {}
    for sec, s in sorted(section_stats.items()):
        n_involved = len(s["pair_words"])
        n_unique = s["n_unique"]
        result[sec] = {
            "n_words": s["n_words"],
            "n_unique": n_unique,
            "n_pair_words": n_involved,
            "error_rate_pct": round(100 * n_involved / n_unique, 1)
            if n_unique else 0,
        }
    return result


def analyze_positional_bias(pairs):
    """Check if errors concentrate at word start, middle, or end.

    For substitution-type errors at distance 1, records the position
    (normalized 0.0=start, 1.0=end) of the differing character.
    """
    positions = []
    for p in pairs:
        if p.distance != 1 or len(p.word_a) != len(p.word_b):
            continue
        wlen = len(p.word_a)
        if wlen < 2:
            continue
        for i in range(wlen):
            if p.word_a[i] != p.word_b[i]:
                norm_pos = i / (wlen - 1)
                positions.append(norm_pos)
                break

    if not positions:
        return {"n_samples": 0}

    n = len(positions)
    avg = sum(positions) / n
    # Bucket into thirds
    start = sum(1 for p in positions if p < 0.33)
    middle = sum(1 for p in positions if 0.33 <= p <= 0.67)
    end = sum(1 for p in positions if p > 0.67)

    return {
        "n_samples": n,
        "avg_position": round(avg, 3),
        "start_pct": round(100 * start / n, 1),
        "middle_pct": round(100 * middle / n, 1),
        "end_pct": round(100 * end / n, 1),
    }


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, min_freq=2, max_dist=2):
    """Entry point for copyist error analysis."""
    report_path = config.stats_dir / "copyist_analysis.json"

    if report_path.exists() and not force:
        click.echo("  Copyist analysis exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("COPYIST ERROR ANALYSIS — EVA Space")

    # 1. Parse EVA
    print_step("Parsing EVA words...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"EVA file not found: {eva_file}\n  Run first: voynich eva"
        )
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words, "
               f"{eva_data['unique_words']} unique")

    # 2. Word frequencies
    print_step("Computing word frequencies...")
    word_freq = Counter(eva_data["words"])
    n_eligible = sum(1 for w, c in word_freq.items() if c >= min_freq)
    click.echo(f"    {n_eligible} words with freq >= {min_freq}")

    # 3. Find similar pairs
    print_step(f"Finding word pairs with Levenshtein <= {max_dist}...")
    t0 = time.time()
    pairs = find_similar_pairs(word_freq, max_dist=max_dist,
                               min_freq=min_freq)
    elapsed = time.time() - t0
    click.echo(f"    {len(pairs)} pairs found in {elapsed:.1f}s")

    # 4. Error type analysis
    print_step("Classifying error types...")
    type_stats = analyze_error_types(pairs)
    for etype, info in type_stats.items():
        click.echo(f"    {etype:20s} {info['count']:5d} ({info['pct']:.1f}%)")

    # 5. Confusion matrix
    print_step("Building confusion matrix...")
    confusion = analyze_confusion_matrix(pairs)
    top_confusions = confusion.most_common(15)
    for (ca, cb), count in top_confusions[:10]:
        is_visual = frozenset({ca, cb}) in VISUAL_CONFUSION_PAIRS
        marker = " [VISUAL]" if is_visual else ""
        click.echo(f"    {ca} <-> {cb}: {count}{marker}")

    # 6. Section analysis
    print_step("Analyzing by section...")
    section_stats = analyze_by_section(eva_data, pairs, word_freq)
    for sec, info in sorted(section_stats.items()):
        click.echo(f"    Section {sec}: {info['n_pair_words']} pair words "
                   f"/ {info['n_unique']} unique "
                   f"({info['error_rate_pct']:.1f}%)")

    # 7. Positional bias
    print_step("Positional bias analysis...")
    pos_bias = analyze_positional_bias(pairs)
    if pos_bias["n_samples"] > 0:
        click.echo(f"    n={pos_bias['n_samples']}, "
                   f"avg_pos={pos_bias['avg_position']:.3f}")
        click.echo(f"    start={pos_bias['start_pct']:.1f}% "
                   f"middle={pos_bias['middle_pct']:.1f}% "
                   f"end={pos_bias['end_pct']:.1f}%")

    # 8. Save report
    print_step("Saving report...")
    report = {
        "params": {"min_freq": min_freq, "max_dist": max_dist},
        "n_eligible_words": n_eligible,
        "n_pairs": len(pairs),
        "error_types": type_stats,
        "confusion_matrix": [
            {"char_a": ca, "char_b": cb, "count": count,
             "is_visual": frozenset({ca, cb}) in VISUAL_CONFUSION_PAIRS}
            for (ca, cb), count in top_confusions
        ],
        "section_analysis": section_stats,
        "positional_bias": pos_bias,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Report: {report_path}")

    # Summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  COPYIST ERROR ANALYSIS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Pairs found: {len(pairs)} (Lev <= {max_dist})")
    click.echo(f"  Words with freq >= {min_freq}: {n_eligible}")

    n_visual = type_stats.get("visual_confusion", {}).get("count", 0)
    n_meta = type_stats.get("metathesis", {}).get("count", 0)
    n_ditto = type_stats.get("dittography", {}).get("count", 0)
    n_haplo = type_stats.get("haplography", {}).get("count", 0)
    n_sub = type_stats.get("substitution", {}).get("count", 0)

    click.echo(f"\n  Visual confusions: {n_visual}")
    click.echo(f"  Metathesis:        {n_meta}")
    click.echo(f"  Dittography:       {n_ditto}")
    click.echo(f"  Haplography:       {n_haplo}")
    click.echo(f"  Substitution:      {n_sub}")

    # Verdict
    click.echo(f"\n  {'=' * 40}")
    ratio_visual = n_visual / len(pairs) if pairs else 0
    if ratio_visual > 0.25:
        click.echo("  VERDICT: High visual confusion rate suggests")
        click.echo("  copyist error pattern (supports cipher hypothesis)")
    elif n_meta + n_ditto + n_haplo > n_sub:
        click.echo("  VERDICT: Copying errors dominate substitution")
        click.echo("  (supports hand-copy transmission)")
    else:
        click.echo("  VERDICT: Substitution dominates")
        click.echo("  (could be cipher variation or independent coinage)")
    click.echo(f"  {'=' * 40}")
