"""
Digraph analysis: investigate whether EVA digraphs encode single Hebrew letters.

Hypothesis: some EVA bigrams (ch, sh, etc.) are not concatenations of two
independent letters but represent single glyphs → single Hebrew consonants.

Method:
  1. Identify candidate digraphs via:
     a. Frequency (must be plausible as a letter: >0.5% of all chars)
     b. Positional profile (behaves like a single char, not a pair)
     c. Internal cohesion (how predictable is the pair?)
  2. For each candidate, test all 7 possible Hebrew assignments:
     the 6 missing letters (bet, kaf, zayin, samekh, tsade, qof)
     + freed tet (from k/t allograph unification)
  3. Measure lexicon match rate improvement
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import click

from .config import ToolkitConfig
from .full_decode import (
    FULL_MAPPING, II_HEBREW, I_HEBREW, DIRECTION, preprocess_eva,
)
from .prepare_lexicon import CONSONANT_NAMES
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# Missing Hebrew letters + freed tet
CANDIDATE_HEBREW = {
    'b': ('bet', 'v/b', 4.4),
    'k': ('kaf', 'k', 2.5),
    'z': ('zayin', 'z', 0.7),
    's': ('samekh', 's', 0.8),
    'C': ('tsade', 'ts', 1.5),
    'q': ('qof', 'q', 1.0),
    'J': ('tet', 't', 0.6),
}


def _tokenize_with_digraph(word, digraph):
    """Tokenize an EVA word treating a specific digraph as a single unit.

    Returns list of tokens where the digraph is one token.
    E.g., _tokenize_with_digraph("chedy", "ch") → ["ch", "e", "d", "y"]
    """
    tokens = []
    i = 0
    dg_len = len(digraph)
    while i < len(word):
        if word[i:i + dg_len] == digraph:
            tokens.append(digraph)
            i += dg_len
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def _positional_profile_digraph(words, digraph):
    """Compute positional profile for a digraph treated as single unit."""
    counts = {"initial": 0, "medial": 0, "final": 0, "total": 0}
    for w in words:
        tokens = _tokenize_with_digraph(w, digraph)
        n = len(tokens)
        for i, tok in enumerate(tokens):
            if tok != digraph:
                continue
            counts["total"] += 1
            if n == 1:
                counts["initial"] += 1
            elif i == 0:
                counts["initial"] += 1
            elif i == n - 1:
                counts["final"] += 1
            else:
                counts["medial"] += 1
    return counts


def _cohesion_score(words, char_a, char_b):
    """Measure how strongly char_a predicts char_b following it.

    Returns: P(b follows a | a occurs) and P(a precedes b | b occurs)
    """
    total_a = 0
    a_before_b = 0
    total_b = 0
    b_after_a = 0

    for w in words:
        for i, c in enumerate(w):
            if c == char_a:
                total_a += 1
                if i + 1 < len(w) and w[i + 1] == char_b:
                    a_before_b += 1
            if c == char_b:
                total_b += 1
                if i > 0 and w[i - 1] == char_a:
                    b_after_a += 1

    p_b_given_a = a_before_b / total_a if total_a > 0 else 0
    p_a_given_b = b_after_a / total_b if total_b > 0 else 0

    return p_b_given_a, p_a_given_b


def _bigram_context_digraph(words, digraph):
    """Compute before/after context when digraph is treated as single unit."""
    before = Counter()
    after = Counter()
    dg_len = len(digraph)

    for w in words:
        tokens = _tokenize_with_digraph(w, digraph)
        for i, tok in enumerate(tokens):
            if tok != digraph:
                continue
            prev_tok = tokens[i - 1] if i > 0 else "#"
            next_tok = tokens[i + 1] if i < len(tokens) - 1 else "#"
            before[prev_tok] += 1
            after[next_tok] += 1

    return before, after


def _decode_with_digraph(eva_word, digraph, digraph_hebrew):
    """Decode an EVA word treating a digraph as a single Hebrew letter.

    Uses the standard 19-char mapping for everything else.
    The digraph is decoded BEFORE the standard ii/i/q preprocessing.
    """
    # First, replace the digraph with a placeholder
    PLACEHOLDER = '\x03'
    w = eva_word.replace(digraph, PLACEHOLDER)

    # Handle q prefix
    prefix = ''
    if w.startswith('qo') and w[2:3] != '':
        prefix = 'qo'
        w = w[2:]
    elif w.startswith('q') and len(w) > 1 and w[1] != PLACEHOLDER:
        prefix = 'q'
        w = w[1:]

    # Handle ii/i
    import re
    w = re.sub(r'i{3,}',
               lambda m: '\x01' * (len(m.group()) // 2) +
                         ('\x02' if len(m.group()) % 2 else ''), w)
    w = w.replace('ii', '\x01')
    w = w.replace('i', '\x02')

    # Decode RTL
    chars = list(reversed(w))
    parts = []
    n_unknown = 0
    for ch in chars:
        if ch == PLACEHOLDER:
            parts.append(digraph_hebrew)
        elif ch == '\x01':
            parts.append(II_HEBREW)
        elif ch == '\x02':
            parts.append(I_HEBREW)
        elif ch in FULL_MAPPING:
            parts.append(FULL_MAPPING[ch])
        else:
            parts.append('?')
            n_unknown += 1

    return ''.join(parts), n_unknown


def run(config: ToolkitConfig, force=False, **kwargs):
    """Investigate EVA digraphs as single Hebrew letter candidates."""
    report_path = config.stats_dir / "digraph_analysis_report.json"

    if report_path.exists() and not force:
        click.echo("  Digraph analysis report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("DIGRAPH ANALYSIS — EVA digraphs as single letters")

    # Parse EVA text
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    words = eva_data["words"]
    word_freq = Counter(words)
    unique_words = list(word_freq.keys())
    total_chars = sum(len(w) for w in words)
    click.echo(f"    {eva_data['total_words']} words, {len(unique_words)} unique, "
               f"{total_chars} chars")

    # Load lexicon
    print_step("Loading Hebrew lexicon...")
    lex_path = config.hebrew_lexicon_path
    if not lex_path.exists():
        lex_path = config.lexicon_dir / "lexicon.json"
    with open(lex_path) as f:
        lex_data = json.load(f)
    hebrew_set = set(lex_data.get("all_consonantal_forms", []))
    click.echo(f"    {len(hebrew_set)} consonantal forms")

    # ═══════════════════════════════════════════════════════
    # 1. IDENTIFY CANDIDATE DIGRAPHS
    # ═══════════════════════════════════════════════════════
    print_header("1. CANDIDATE DIGRAPHS — frequency + cohesion")

    # Count all bigrams
    bigram_count = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bigram_count[w[i:i + 2]] += 1

    # Candidate threshold: at least 0.3% of total chars
    min_count = int(total_chars * 0.003)
    candidates = [(bg, n) for bg, n in bigram_count.most_common()
                  if n >= min_count]

    click.echo(f"\n  {'Digraph':8s} {'Count':>7s} {'%chars':>7s} "
               f"{'P(b|a)':>7s} {'P(a|b)':>7s} {'Cohesion':>9s} "
               f"{'Profile':>25s}")
    click.echo("  " + "-" * 80)

    digraph_data = {}
    for bg, count in candidates:
        char_a, char_b = bg[0], bg[1]
        p_ba, p_ab = _cohesion_score(words, char_a, char_b)
        profile = _positional_profile_digraph(words, bg)
        t = profile["total"] or 1
        prof_str = (f"I:{profile['initial']/t*100:4.0f}% "
                    f"M:{profile['medial']/t*100:4.0f}% "
                    f"F:{profile['final']/t*100:4.0f}%")
        cohesion = (p_ba + p_ab) / 2

        digraph_data[bg] = {
            "count": count,
            "pct_chars": round(count / total_chars * 100, 2),
            "p_b_given_a": round(p_ba, 3),
            "p_a_given_b": round(p_ab, 3),
            "cohesion": round(cohesion, 3),
            "profile": profile,
        }

        click.echo(f"  {bg:8s} {count:7d} {count/total_chars*100:6.2f}% "
                    f"{p_ba:7.3f} {p_ab:7.3f} {cohesion:9.3f} "
                    f"{prof_str:>25s}")

    # ═══════════════════════════════════════════════════════
    # 2. SELECT TOP CANDIDATES
    # ═══════════════════════════════════════════════════════
    print_header("2. TOP DIGRAPH CANDIDATES — ranked by cohesion")

    # Filter: cohesion > 0.3 AND count > min_count
    top_candidates = sorted(
        [(bg, d) for bg, d in digraph_data.items() if d["cohesion"] > 0.3],
        key=lambda x: -x[1]["cohesion"]
    )

    click.echo(f"\n  {len(top_candidates)} digraphs with cohesion > 0.3:")
    for bg, d in top_candidates:
        click.echo(f"    {bg}: count={d['count']}, cohesion={d['cohesion']}, "
                   f"P(b|a)={d['p_b_given_a']}, P(a|b)={d['p_a_given_b']}")

    # ═══════════════════════════════════════════════════════
    # 3. CONTEXT COMPARISON — do digraphs have their OWN context?
    # ═══════════════════════════════════════════════════════
    print_header("3. CONTEXT ANALYSIS — digraph vs component contexts")

    # For each top candidate: compare context of the digraph (as unit)
    # vs what you'd expect from concatenating the two letters' contexts
    for bg, d in top_candidates[:6]:
        before_dg, after_dg = _bigram_context_digraph(words, bg)
        click.echo(f"\n  {bg} (n={d['count']}):")
        click.echo(f"    Before {bg}: "
                   + " ".join(f"{c}:{n}" for c, n in before_dg.most_common(6)))
        click.echo(f"    After  {bg}: "
                   + " ".join(f"{c}:{n}" for c, n in after_dg.most_common(6)))

    # ═══════════════════════════════════════════════════════
    # 4. LEXICON IMPACT — test each digraph × missing letter
    # ═══════════════════════════════════════════════════════
    print_header("4. LEXICON IMPACT — digraph × Hebrew letter")

    # Baseline: current 19-char match rate
    print_step("Computing baseline match rate...")
    baseline_matches = set()
    for w in unique_words:
        heb, n_unk = _decode_with_digraph(w, "\xff\xff", "X")  # no-op digraph
        # Actually let's just use standard decode
        pass

    # Use standard decode for baseline
    from .full_decode import decode_word
    baseline_count = 0
    baseline_total = 0
    for w in unique_words:
        _, heb, n_unk = decode_word(w)
        if n_unk == 0:
            baseline_total += 1
            if heb in hebrew_set:
                baseline_count += 1
    click.echo(f"    Baseline: {baseline_count}/{baseline_total} "
               f"({baseline_count/baseline_total*100:.2f}%)")

    # Test top candidates
    test_digraphs = [bg for bg, d in top_candidates[:8]]
    # Also always include ch and sh even if cohesion is lower
    for must_have in ["ch", "sh"]:
        if must_have not in test_digraphs and must_have in digraph_data:
            test_digraphs.append(must_have)

    click.echo(f"\n  Testing digraphs: {test_digraphs}")
    click.echo(f"\n  {'Digraph':8s} {'Hebrew':8s} {'Name':8s} "
               f"{'Matches':>8s} {'Delta':>6s} {'New d=0':>7s} {'Top new matches':30s}")
    click.echo("  " + "-" * 85)

    best_results = []

    for bg in test_digraphs:
        for heb_letter, (heb_name, phoneme, bh_freq) in CANDIDATE_HEBREW.items():
            matches = 0
            new_matches = []

            for w in unique_words:
                if bg not in w:
                    continue
                heb_str, n_unk = _decode_with_digraph(w, bg, heb_letter)
                if n_unk > 0:
                    continue
                if heb_str in hebrew_set:
                    matches += 1
                    # Is this a NEW match (not in baseline)?
                    _, baseline_heb, baseline_unk = decode_word(w)
                    if baseline_unk > 0 or baseline_heb not in hebrew_set:
                        freq = word_freq[w]
                        new_matches.append((w, heb_str, freq))

            new_matches.sort(key=lambda x: -x[2])
            delta = len(new_matches)
            top_str = ", ".join(f"{m[1]}({m[2]})" for m in new_matches[:3])

            if delta > 0:
                click.echo(f"  {bg:8s} {heb_letter:8s} {heb_name:8s} "
                           f"{matches:8d} {'+' + str(delta):>6s} "
                           f"{delta:>7d} {top_str:30s}")

                best_results.append({
                    "digraph": bg,
                    "hebrew": heb_letter,
                    "hebrew_name": heb_name,
                    "phoneme": phoneme,
                    "bh_freq": bh_freq,
                    "total_matches": matches,
                    "new_matches": delta,
                    "new_match_details": [
                        {"eva": m[0], "hebrew": m[1], "freq": m[2]}
                        for m in new_matches[:10]
                    ],
                })

    # Sort by new matches
    best_results.sort(key=lambda x: -x["new_matches"])

    # ═══════════════════════════════════════════════════════
    # 5. TOP ASSIGNMENTS RANKED
    # ═══════════════════════════════════════════════════════
    print_header("5. TOP DIGRAPH→HEBREW ASSIGNMENTS")

    if best_results:
        click.echo(f"\n  {'Rank':5s} {'Digraph':8s} {'Hebrew':8s} {'Name':8s} "
                   f"{'New d=0':>8s} {'BH freq':>8s} {'Top examples':30s}")
        click.echo("  " + "-" * 80)
        for i, r in enumerate(best_results[:20]):
            top_ex = ", ".join(
                f"{d['hebrew']}(x{d['freq']})" for d in r["new_match_details"][:3])
            click.echo(f"  {i+1:5d} {r['digraph']:8s} {r['hebrew']:8s} "
                       f"{r['hebrew_name']:8s} {r['new_matches']:>8d} "
                       f"{r['bh_freq']:>7.1f}% {top_ex:30s}")
    else:
        click.echo("  No improvements found.")

    # ═══════════════════════════════════════════════════════
    # 6. GREEDY BEST COMBINATION
    # ═══════════════════════════════════════════════════════
    print_header("6. GREEDY BEST COMBINATION — non-conflicting assignments")

    # Greedy: pick best digraph→hebrew pairs without conflicts
    # (each digraph used once, each hebrew letter used once)
    used_digraphs = set()
    used_hebrews = set()
    selected = []

    for r in best_results:
        if r["digraph"] in used_digraphs:
            continue
        if r["hebrew"] in used_hebrews:
            continue
        if r["new_matches"] < 1:
            continue
        selected.append(r)
        used_digraphs.add(r["digraph"])
        used_hebrews.add(r["hebrew"])

    total_new = sum(r["new_matches"] for r in selected)
    click.echo(f"\n  Selected {len(selected)} non-conflicting assignments "
               f"(+{total_new} new lexicon matches):")
    for r in selected:
        click.echo(f"    {r['digraph']} → {r['hebrew_name']} ({r['hebrew']}) "
                   f"— +{r['new_matches']} matches, "
                   f"BH freq {r['bh_freq']}%")

    # Combined match rate
    print_step("Computing combined match rate with all selected digraphs...")

    combined_count = 0
    for w in unique_words:
        # Apply all selected digraphs in sequence
        heb_str = None
        for r in selected:
            if r["digraph"] in w:
                heb_str, n_unk = _decode_with_digraph(
                    w, r["digraph"], r["hebrew"])
                if n_unk == 0 and heb_str in hebrew_set:
                    combined_count += 1
                    break
        else:
            # No digraph matched — use standard decode
            _, heb, n_unk = decode_word(w)
            if n_unk == 0 and heb in hebrew_set:
                combined_count += 1

    click.echo(f"\n  Baseline match rate:  {baseline_count}/{baseline_total} "
               f"({baseline_count/baseline_total*100:.2f}%)")
    click.echo(f"  Combined match rate:  {combined_count}/{baseline_total} "
               f"({combined_count/baseline_total*100:.2f}%)")
    click.echo(f"  Improvement: +{combined_count - baseline_count} matches "
               f"(+{(combined_count-baseline_count)/baseline_total*100:.2f}%)")

    # ═══════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════
    print_header("VERDICT")

    if selected:
        click.echo(f"\n  {len(selected)} digraph→Hebrew assignments found:")
        for r in selected:
            click.echo(f"    {r['digraph']} → {r['hebrew_name']} "
                       f"({r['hebrew']}, '{r['phoneme']}') "
                       f"— +{r['new_matches']} new d=0 matches")
        click.echo(f"\n  Total new lexicon matches: +{total_new}")
        click.echo(f"  Combined improvement: "
                   f"{baseline_count}→{combined_count} "
                   f"(+{(combined_count-baseline_count)/baseline_total*100:.2f}%)")
    else:
        click.echo("\n  No viable digraph→Hebrew assignments found.")

    # Save report
    print_step("Saving report...")
    report = {
        "digraph_candidates": {
            bg: {
                "count": d["count"],
                "pct_chars": d["pct_chars"],
                "cohesion": d["cohesion"],
                "p_b_given_a": d["p_b_given_a"],
                "p_a_given_b": d["p_a_given_b"],
                "profile": d["profile"],
            }
            for bg, d in digraph_data.items()
        },
        "top_cohesion": [
            {"digraph": bg, **d}
            for bg, d in top_candidates
        ],
        "lexicon_tests": best_results[:30],
        "selected_assignments": selected,
        "baseline_matches": baseline_count,
        "combined_matches": combined_count,
        "baseline_total": baseline_total,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")
