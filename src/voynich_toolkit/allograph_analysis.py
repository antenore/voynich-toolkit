"""
Allograph analysis: investigate whether two EVA chars are contextual variants.

Methodology (proven in Phase 7 for f/p and ii/i):
  1. Positional profiles — P(initial), P(medial), P(final) for each char
  2. Bigram context — which chars appear before/after each candidate
  3. Cosine similarity of positional profiles
  4. Context overlap (Jaccard + weighted) of bigram neighborhoods
  5. Comparison with known allograph pair (f/p) and known non-pair (i/h)

Thresholds (from Phase 7):
  - Cosine > 0.95 AND context overlap > 0.50 → likely allograph
  - f/p: cosine 0.987, context 0.67 → CONFIRMED
  - i/h: cosine 1.000, context 0.00 → REJECTED (complementary distribution)
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import click

from .config import ToolkitConfig
from .utils import print_header, print_step
from .word_structure import parse_eva_words


def _positional_profile(words, char):
    """Compute P(initial), P(medial), P(final) for a single char."""
    counts = {"initial": 0, "medial": 0, "final": 0, "total": 0}
    for w in words:
        for i, c in enumerate(w):
            if c != char:
                continue
            counts["total"] += 1
            if len(w) == 1:
                counts["initial"] += 1  # treat isolated as initial
            elif i == 0:
                counts["initial"] += 1
            elif i == len(w) - 1:
                counts["final"] += 1
            else:
                counts["medial"] += 1
    return counts


def _bigram_context(words, char):
    """Compute before/after character distributions around a char.

    Returns: (before_counter, after_counter) where each is Counter
    of characters appearing immediately before/after the target char.
    '#' represents word boundary.
    """
    before = Counter()
    after = Counter()
    for w in words:
        for i, c in enumerate(w):
            if c != char:
                continue
            before[w[i - 1] if i > 0 else "#"] += 1
            after[w[i + 1] if i < len(w) - 1 else "#"] += 1
    return before, after


def _cosine_similarity(profile_a, profile_b):
    """Cosine similarity between two positional profiles."""
    keys = ["initial", "medial", "final"]
    tot_a = profile_a["total"] or 1
    tot_b = profile_b["total"] or 1
    vec_a = [profile_a[k] / tot_a for k in keys]
    vec_b = [profile_b[k] / tot_b for k in keys]
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _context_overlap(counter_a, counter_b, mode="weighted"):
    """Overlap between two context distributions.

    mode='jaccard': |intersection| / |union| of character sets
    mode='weighted': sum of min(P(c|a), P(c|b)) for each character c
    """
    if mode == "jaccard":
        set_a = set(counter_a.keys())
        set_b = set(counter_b.keys())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
    else:
        # Weighted overlap (Bhattacharyya-like)
        total_a = sum(counter_a.values()) or 1
        total_b = sum(counter_b.values()) or 1
        all_chars = set(counter_a.keys()) | set(counter_b.keys())
        overlap = 0.0
        for ch in all_chars:
            p_a = counter_a.get(ch, 0) / total_a
            p_b = counter_b.get(ch, 0) / total_b
            overlap += min(p_a, p_b)
        return overlap


def analyze_pair(words, char_a, char_b):
    """Full allograph analysis for a pair of EVA characters."""
    # 1. Positional profiles
    prof_a = _positional_profile(words, char_a)
    prof_b = _positional_profile(words, char_b)

    # 2. Bigram context
    before_a, after_a = _bigram_context(words, char_a)
    before_b, after_b = _bigram_context(words, char_b)

    # 3. Cosine similarity
    cosine = _cosine_similarity(prof_a, prof_b)

    # 4. Context overlaps
    before_jaccard = _context_overlap(before_a, before_b, "jaccard")
    before_weighted = _context_overlap(before_a, before_b, "weighted")
    after_jaccard = _context_overlap(after_a, after_b, "jaccard")
    after_weighted = _context_overlap(after_a, after_b, "weighted")
    combined_weighted = (before_weighted + after_weighted) / 2

    return {
        "char_a": char_a,
        "char_b": char_b,
        "profile_a": prof_a,
        "profile_b": prof_b,
        "cosine_similarity": round(cosine, 4),
        "before_context_a": before_a.most_common(10),
        "before_context_b": before_b.most_common(10),
        "after_context_a": after_a.most_common(10),
        "after_context_b": after_b.most_common(10),
        "before_jaccard": round(before_jaccard, 4),
        "before_weighted": round(before_weighted, 4),
        "after_jaccard": round(after_jaccard, 4),
        "after_weighted": round(after_weighted, 4),
        "combined_context_overlap": round(combined_weighted, 4),
    }


def _format_profile(prof):
    """Format positional profile for display."""
    t = prof["total"] or 1
    return (f"ini={prof['initial']/t*100:5.1f}%  "
            f"med={prof['medial']/t*100:5.1f}%  "
            f"fin={prof['final']/t*100:5.1f}%  "
            f"(n={prof['total']})")


def _format_context(counter, total=None):
    """Format context counter for display."""
    t = total or sum(counter.values()) or 1
    items = counter.most_common(8)
    parts = [f"{ch}:{n/t*100:.0f}%" for ch, n in items]
    return " ".join(parts)


def run(config: ToolkitConfig, force=False, **kwargs):
    """Investigate EVA l/e allography using positional + context analysis."""
    report_path = config.stats_dir / "allograph_le_report.json"

    if report_path.exists() and not force:
        click.echo("  Allograph l/e report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("ALLOGRAPH ANALYSIS — EVA l vs e")

    # Parse EVA text
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    words = eva_data["words"]
    click.echo(f"    {eva_data['total_words']} words, "
               f"{eva_data['unique_words']} unique")

    # ─── Primary analysis: l vs e ───
    print_header("PRIMARY — EVA l vs e")
    le_result = analyze_pair(words, "l", "e")

    click.echo(f"\n  Positional profiles:")
    click.echo(f"    l: {_format_profile(le_result['profile_a'])}")
    click.echo(f"    e: {_format_profile(le_result['profile_b'])}")
    click.echo(f"\n  Cosine similarity: {le_result['cosine_similarity']}")

    click.echo(f"\n  Before-context (what precedes l vs e):")
    click.echo(f"    l: {_format_context(Counter(dict(le_result['before_context_a'])))}")
    click.echo(f"    e: {_format_context(Counter(dict(le_result['before_context_b'])))}")
    click.echo(f"  After-context (what follows l vs e):")
    click.echo(f"    l: {_format_context(Counter(dict(le_result['after_context_a'])))}")
    click.echo(f"    e: {_format_context(Counter(dict(le_result['after_context_b'])))}")

    click.echo(f"\n  Context overlap:")
    click.echo(f"    Before: jaccard={le_result['before_jaccard']}, "
               f"weighted={le_result['before_weighted']}")
    click.echo(f"    After:  jaccard={le_result['after_jaccard']}, "
               f"weighted={le_result['after_weighted']}")
    click.echo(f"    Combined: {le_result['combined_context_overlap']}")

    # ─── Reference pair: f vs p (known allograph) ───
    print_header("REFERENCE — EVA f vs p (known allograph)")
    fp_result = analyze_pair(words, "f", "p")
    click.echo(f"  Positional profiles:")
    click.echo(f"    f: {_format_profile(fp_result['profile_a'])}")
    click.echo(f"    p: {_format_profile(fp_result['profile_b'])}")
    click.echo(f"  Cosine: {fp_result['cosine_similarity']}")
    click.echo(f"  Combined context overlap: {fp_result['combined_context_overlap']}")

    # ─── Reference pair: i vs h (known non-allograph) ───
    print_header("REFERENCE — EVA i vs h (known NON-allograph)")
    ih_result = analyze_pair(words, "i", "h")
    click.echo(f"  Positional profiles:")
    click.echo(f"    i: {_format_profile(ih_result['profile_a'])}")
    click.echo(f"    h: {_format_profile(ih_result['profile_b'])}")
    click.echo(f"  Cosine: {ih_result['cosine_similarity']}")
    click.echo(f"  Combined context overlap: {ih_result['combined_context_overlap']}")

    # ─── Additional pairs for comparison ───
    print_header("COMPARISON TABLE — All candidate pairs")
    pairs_to_test = [
        ("l", "e"),   # primary investigation
        ("f", "p"),   # known allograph (both → lamed)
        ("i", "h"),   # known non-allograph
        ("d", "i"),   # known allograph (both → resh via different paths)
        ("r", "h"),   # r=he, h=ayin — different letters but similar shape?
        ("l", "s"),   # l=mem, s=nun — similar frequency?
        ("e", "d"),   # e=pe, d=resh — are they confused?
        ("k", "t"),   # k=tav, t=tet — both 't' in Italian
    ]

    click.echo(f"\n  {'Pair':8s} {'Cosine':8s} {'Ctx-Bef':8s} {'Ctx-Aft':8s} "
               f"{'Combined':9s} {'Verdict':12s}")
    click.echo("  " + "-" * 60)

    comparison_results = []
    for a, b in pairs_to_test:
        result = analyze_pair(words, a, b)
        cos = result["cosine_similarity"]
        ctx = result["combined_context_overlap"]
        before_w = result["before_weighted"]
        after_w = result["after_weighted"]

        if cos > 0.95 and ctx > 0.50:
            verdict = "ALLOGRAPH"
        elif cos > 0.90 and ctx > 0.40:
            verdict = "possible"
        elif cos > 0.95 and ctx < 0.20:
            verdict = "complementary"
        else:
            verdict = "distinct"

        click.echo(f"  {a}/{b:5s} {cos:8.3f} {before_w:8.3f} {after_w:8.3f} "
                   f"{ctx:9.3f} {verdict:12s}")
        comparison_results.append({
            "pair": f"{a}/{b}",
            "cosine": cos,
            "context_before": before_w,
            "context_after": after_w,
            "combined": ctx,
            "verdict": verdict,
        })

    # ─── Detailed positional slot analysis for l and e ───
    print_header("SLOT ANALYSIS — l and e by absolute position")
    max_pos = 8
    from_start_l = defaultdict(int)
    from_start_e = defaultdict(int)
    total_at_pos = defaultdict(int)

    for w in words:
        for i, c in enumerate(w):
            if i >= max_pos:
                break
            total_at_pos[i] += 1
            if c == "l":
                from_start_l[i] += 1
            elif c == "e":
                from_start_e[i] += 1

    click.echo(f"\n  {'Pos':4s} {'l count':>8s} {'l %':>6s} {'e count':>8s} "
               f"{'e %':>6s} {'l/e ratio':>10s}")
    click.echo("  " + "-" * 48)
    for pos in range(max_pos):
        n_l = from_start_l.get(pos, 0)
        n_e = from_start_e.get(pos, 0)
        tot = total_at_pos.get(pos, 1)
        pct_l = n_l / tot * 100
        pct_e = n_e / tot * 100
        ratio = f"{n_l / n_e:.2f}" if n_e > 0 else "inf"
        click.echo(f"  {pos:4d} {n_l:8d} {pct_l:5.1f}% {n_e:8d} "
                   f"{pct_e:5.1f}% {ratio:>10s}")

    # ─── Word pattern analysis: where do l and e substitute? ───
    print_header("SUBSTITUTION PATTERNS")
    print_step("Finding minimal pairs (words differing only in l↔e)...")

    word_set = set(words)
    word_freq = Counter(words)
    minimal_pairs = []
    seen_pairs = set()

    for w in word_set:
        for i, c in enumerate(w):
            if c == "l":
                variant = w[:i] + "e" + w[i + 1:]
                if variant in word_set and (w, variant) not in seen_pairs:
                    seen_pairs.add((w, variant))
                    seen_pairs.add((variant, w))
                    minimal_pairs.append({
                        "word_l": w,
                        "word_e": variant,
                        "position": i,
                        "word_len": len(w),
                        "freq_l": word_freq[w],
                        "freq_e": word_freq[variant],
                    })

    minimal_pairs.sort(key=lambda x: -(x["freq_l"] + x["freq_e"]))
    click.echo(f"    Found {len(minimal_pairs)} minimal pairs (l↔e at same position)")

    if minimal_pairs:
        click.echo(f"\n  {'Word(l)':12s} {'Word(e)':12s} {'Pos':4s} "
                   f"{'Freq(l)':>8s} {'Freq(e)':>8s}")
        click.echo("  " + "-" * 50)
        for mp in minimal_pairs[:25]:
            click.echo(f"  {mp['word_l']:12s} {mp['word_e']:12s} "
                       f"{mp['position']:4d} {mp['freq_l']:8d} "
                       f"{mp['freq_e']:8d}")

    # ─── Verdict ───
    print_header("VERDICT — l/e allography")
    cos = le_result["cosine_similarity"]
    ctx = le_result["combined_context_overlap"]

    click.echo(f"\n  Cosine similarity:        {cos:.4f} "
               f"(threshold: >0.95)")
    click.echo(f"  Combined context overlap: {ctx:.4f} "
               f"(threshold: >0.50)")
    click.echo(f"  Minimal pairs found:      {len(minimal_pairs)}")
    click.echo(f"\n  Reference benchmarks:")
    click.echo(f"    f/p (confirmed allograph):  "
               f"cos={fp_result['cosine_similarity']}, "
               f"ctx={fp_result['combined_context_overlap']}")
    click.echo(f"    i/h (confirmed non-allo.):  "
               f"cos={ih_result['cosine_similarity']}, "
               f"ctx={ih_result['combined_context_overlap']}")

    if cos > 0.95 and ctx > 0.50:
        verdict = "ALLOGRAPH — l and e are likely contextual variants"
        verdict_code = "allograph"
    elif cos > 0.90 and ctx > 0.40:
        verdict = "POSSIBLE ALLOGRAPH — evidence is suggestive but not conclusive"
        verdict_code = "possible"
    elif cos > 0.95 and ctx < 0.20:
        verdict = "COMPLEMENTARY DISTRIBUTION — same profile but different contexts"
        verdict_code = "complementary"
    else:
        verdict = "DISTINCT — l and e are separate characters"
        verdict_code = "distinct"

    click.echo(f"\n  {'='*50}")
    click.echo(f"  {verdict}")
    click.echo(f"  {'='*50}")

    # ─── Save report ───
    print_step("Saving report...")
    report = {
        "primary_pair": {
            "chars": "l/e",
            "cosine": cos,
            "combined_context_overlap": ctx,
            "verdict": verdict_code,
            "profile_l": le_result["profile_a"],
            "profile_e": le_result["profile_b"],
            "before_context_l": le_result["before_context_a"],
            "before_context_e": le_result["before_context_b"],
            "after_context_l": le_result["after_context_a"],
            "after_context_e": le_result["after_context_b"],
            "context_detail": {
                "before_jaccard": le_result["before_jaccard"],
                "before_weighted": le_result["before_weighted"],
                "after_jaccard": le_result["after_jaccard"],
                "after_weighted": le_result["after_weighted"],
            },
        },
        "reference_pairs": {
            "f_p": {
                "cosine": fp_result["cosine_similarity"],
                "context": fp_result["combined_context_overlap"],
                "known": "allograph",
            },
            "i_h": {
                "cosine": ih_result["cosine_similarity"],
                "context": ih_result["combined_context_overlap"],
                "known": "non-allograph",
            },
        },
        "comparison_table": comparison_results,
        "slot_analysis": {
            "l_by_position": dict(from_start_l),
            "e_by_position": dict(from_start_e),
        },
        "minimal_pairs": {
            "count": len(minimal_pairs),
            "top25": minimal_pairs[:25],
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")
