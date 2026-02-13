"""
Fuzzy decode: complete the EVA->Hebrew mapping and fuzzy-match vs Italian.

Loads the 15/19 mapping from champollion_report.json, generates candidate
completions for the 4 unmapped EVA chars (f,g,i,q), decodes all text,
and fuzzy-matches against the Italian lexicon (Lev<=2).

Picks the best completion by total weighted score across all words.
"""
import json
import time
from collections import Counter, defaultdict
from itertools import permutations

import click

from .config import ToolkitConfig
from .fuzzy_utils import LengthBucketedIndex, batch_fuzzy_match, SCORE_WEIGHTS
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import (
    parse_eva_words,
    compute_char_positional_profiles,
)


# =====================================================================
# Constants
# =====================================================================

EVA_CHARS = list("acdefghiklmnopqrsty")
HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"

# Section -> domain
SECTION_TO_DOMAIN = {
    "H": "botanical",
    "S": "astronomical",
    "Z": "astronomical",
    "B": "medical",
    "P": "medical",
}


# =====================================================================
# Loading
# =====================================================================

def load_champollion_mapping(config):
    """Load the extended mapping from champollion_report.json.

    Returns:
        mapping: dict {eva_char: hebrew_char} (15 entries)
        unmapped: list of unmapped EVA chars
    """
    path = config.stats_dir / "champollion_report.json"
    if not path.exists():
        raise click.ClickException(
            f"Champollion report not found: {path}\n"
            "  Run first: voynich champollion"
        )
    with open(path) as f:
        data = json.load(f)

    mapping = {}
    for eva_ch, info in data.get("mapping_extended", {}).items():
        mapping[eva_ch] = info["hebrew"]

    unmapped = [c for c in EVA_CHARS if c not in mapping]
    return mapping, unmapped, data.get("direction", "rtl")


def load_italian_lexicon(config):
    """Load Italian lexicon forms, glosses, and domains.

    Returns:
        all_forms: list of phonemic strings
        form_to_gloss: dict
        form_to_domain: dict
    """
    path = config.lexicon_dir / "italian_lexicon.json"
    if not path.exists():
        raise click.ClickException(
            f"Italian lexicon not found: {path}\n"
            "  Run first: voynich prepare-italian-lexicon"
        )
    with open(path) as f:
        data = json.load(f)

    all_forms = data["all_forms"]
    form_to_gloss = data.get("form_to_gloss", {})
    # Build form_to_domain from by_domain
    form_to_domain = {}
    for domain, entries in data.get("by_domain", {}).items():
        for entry in entries:
            ph = entry.get("phonemic", "")
            if ph and ph not in form_to_domain:
                form_to_domain[ph] = domain

    return all_forms, form_to_gloss, form_to_domain


# =====================================================================
# Completion candidate generation
# =====================================================================

def generate_completions(mapping, unmapped_chars, eva_data,
                         max_candidates=50):
    """Generate candidate completions for unmapped EVA chars.

    Uses permutations of unused Hebrew letters, filtered by positional
    profile compatibility to reduce from 840 to ~50 candidates.

    Args:
        mapping: existing {eva: hebrew} mapping (15 entries)
        unmapped_chars: list of 4 unmapped EVA chars
        eva_data: parsed EVA data (for positional profiles)
        max_candidates: maximum completions to return

    Returns:
        list of dicts, each a full 19-char mapping {eva: hebrew}
    """
    used_hebrew = set(mapping.values())
    unused_hebrew = [h for h in HEBREW_CHARS if h not in used_hebrew]

    n_unmapped = len(unmapped_chars)
    if n_unmapped == 0:
        return [dict(mapping)]

    # Compute positional profiles of unmapped EVA chars
    profiles = compute_char_positional_profiles(eva_data["words"])
    eva_profiles = profiles["profiles"]

    # Score each permutation by positional compatibility
    # For each unmapped EVA char, we want the Hebrew letter whose
    # Italian phoneme has a compatible positional profile
    all_perms = list(permutations(unused_hebrew, n_unmapped))
    click.echo(f"    {len(all_perms)} raw permutations of "
               f"{len(unused_hebrew)} unused Hebrew letters")

    scored = []
    for perm in all_perms:
        candidate = dict(mapping)
        for eva_ch, heb_ch in zip(unmapped_chars, perm):
            candidate[eva_ch] = heb_ch

        # Simple heuristic score: vowel chars should go to
        # EVA chars with high initial/final rates
        score = 0.0
        for eva_ch, heb_ch in zip(unmapped_chars, perm):
            italian_ph = HEBREW_TO_ITALIAN.get(heb_ch, "?")
            is_vowel = italian_ph in ('a', 'e', 'i', 'o', 'u')

            ep = eva_profiles.get(eva_ch, {})
            ini_rate = ep.get("initial", 0)
            fin_rate = ep.get("final", 0)
            med_rate = ep.get("medial", 0)

            if is_vowel:
                # Vowels: prefer chars with high boundary rates
                score += (ini_rate + fin_rate) * 2 + med_rate
            else:
                # Consonants: prefer chars with high medial rate
                score += med_rate * 2 + ini_rate + fin_rate

        scored.append((score, candidate))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:max_candidates]]


# =====================================================================
# Decoding
# =====================================================================

def decode_word(eva_word, mapping, direction):
    """Decode an EVA word to Italian via Hebrew mapping.

    Returns Italian phonemic string, or None if any char unmapped.
    """
    chars = list(eva_word) if direction == "ltr" else list(reversed(eva_word))
    italian_parts = []
    for ch in chars:
        heb = mapping.get(ch)
        if heb is None:
            return None
        it = HEBREW_TO_ITALIAN.get(heb, "?")
        italian_parts.append(it)
    return "".join(italian_parts)


def decode_corpus(eva_data, mapping, direction, min_word_len=3):
    """Decode all EVA words and return (decoded_word, count) pairs.

    Returns:
        list of (italian_str, count) for successfully decoded words
        with length >= min_word_len
    """
    word_freq = Counter(eva_data["words"])
    decoded = []
    for word, count in word_freq.most_common():
        it = decode_word(word, mapping, direction)
        if it and len(it) >= min_word_len:
            decoded.append((it, count))
    return decoded


# =====================================================================
# Scoring
# =====================================================================

def score_completion(decoded_words, index, max_dist=2):
    """Score a completion by total weighted fuzzy matches.

    Returns (total_score, n_matched_types, matched_details).
    """
    total_score = 0
    n_matched = 0
    details = []

    for italian, count in decoded_words:
        best = index.best_match(italian, max_dist)
        if best:
            weighted = best.score * count
            total_score += weighted
            n_matched += 1
            if len(details) < 200:
                details.append({
                    "decoded": italian,
                    "target": best.target,
                    "distance": best.distance,
                    "score": best.score,
                    "count": count,
                    "weighted": weighted,
                    "gloss": best.gloss,
                    "domain": best.domain,
                })

    details.sort(key=lambda d: -d["weighted"])
    return total_score, n_matched, details


# =====================================================================
# Section breakdown
# =====================================================================

def analyze_by_section(eva_data, mapping, direction, index, max_dist=2):
    """Break down fuzzy matches by manuscript section.

    Returns dict {section: {n_words, n_decoded, n_matched, score,
                            expected_domain, top_matches}}.
    """
    section_words = defaultdict(Counter)
    for page in eva_data["pages"]:
        sec = page.get("section", "?")
        for w in page["words"]:
            section_words[sec][w] += 1

    result = {}
    for sec, word_freq in sorted(section_words.items()):
        n_words = sum(word_freq.values())
        decoded = []
        for word, count in word_freq.most_common():
            it = decode_word(word, mapping, direction)
            if it and len(it) >= 3:
                decoded.append((it, count))

        total_score = 0
        n_matched = 0
        top_matches = []
        for italian, count in decoded:
            best = index.best_match(italian, max_dist)
            if best:
                weighted = best.score * count
                total_score += weighted
                n_matched += 1
                if len(top_matches) < 10:
                    top_matches.append({
                        "decoded": italian,
                        "target": best.target,
                        "distance": best.distance,
                        "gloss": best.gloss,
                        "count": count,
                    })

        result[sec] = {
            "n_words": n_words,
            "n_decoded": len(decoded),
            "n_matched": n_matched,
            "score": total_score,
            "expected_domain": SECTION_TO_DOMAIN.get(sec, "general"),
            "top_matches": top_matches,
        }

    return result


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, direction=None,
        min_word_len=3, max_dist=2, n_candidates=50):
    """Entry point: fuzzy decode with mapping completion."""
    json_path = config.stats_dir / "fuzzy_matches.json"
    txt_path = config.stats_dir / "fuzzy_matches_readable.txt"

    if json_path.exists() and not force:
        click.echo("  Fuzzy decode results exist. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("FUZZY DECODE — Complete Mapping & Match Italian Lexicon")

    # 1. Load champollion mapping
    print_step("Loading champollion mapping...")
    mapping, unmapped, champ_direction = load_champollion_mapping(config)
    if direction is None:
        direction = champ_direction
    click.echo(f"    {len(mapping)}/19 chars mapped, "
               f"unmapped: {unmapped}, direction: {direction}")

    # 2. Load Italian lexicon
    print_step("Loading Italian lexicon...")
    all_forms, form_to_gloss, form_to_domain = load_italian_lexicon(config)
    click.echo(f"    {len(all_forms)} phonemic forms")

    # 3. Build fuzzy index
    print_step("Building fuzzy index...")
    index = LengthBucketedIndex(all_forms, form_to_gloss, form_to_domain)
    click.echo(f"    Indexed {index.size} forms, "
               f"buckets: {len(index.bucket_sizes)}")

    # 4. Parse EVA
    print_step("Parsing EVA words...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"EVA file not found: {eva_file}\n  Run first: voynich eva"
        )
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words, "
               f"{eva_data['unique_words']} unique")

    # 5. Generate completions
    print_step("Generating mapping completions...")
    t0 = time.time()

    if unmapped:
        completions = generate_completions(
            mapping, unmapped, eva_data, max_candidates=n_candidates
        )
    else:
        completions = [dict(mapping)]
    click.echo(f"    {len(completions)} candidate completions")

    # 6. Score each completion
    print_step(f"Scoring {len(completions)} completions...")
    best_score = -1
    best_mapping = None
    best_details = None
    best_n_matched = 0
    all_scores = []

    for i, candidate in enumerate(completions):
        decoded = decode_corpus(eva_data, candidate, direction,
                                min_word_len=min_word_len)
        total, n_matched, details = score_completion(
            decoded, index, max_dist=max_dist
        )
        all_scores.append(total)

        if total > best_score:
            best_score = total
            best_mapping = candidate
            best_details = details
            best_n_matched = n_matched

        if (i + 1) % 10 == 0:
            click.echo(f"    Scored {i+1}/{len(completions)}... "
                       f"best so far: {best_score}")

    elapsed = time.time() - t0
    click.echo(f"    Best score: {best_score} "
               f"({best_n_matched} word types matched) "
               f"in {elapsed:.1f}s")

    # 7. Section breakdown
    print_step("Section breakdown...")
    section_stats = analyze_by_section(
        eva_data, best_mapping, direction, index, max_dist
    )
    for sec, info in sorted(section_stats.items()):
        click.echo(f"    {sec}: {info['n_matched']} matches, "
                   f"score={info['score']}")

    # 8. Build readable mapping
    readable_mapping = {}
    for eva_ch in sorted(best_mapping):
        heb_ch = best_mapping[eva_ch]
        source = "champollion" if eva_ch in mapping else "completion"
        readable_mapping[eva_ch] = {
            "hebrew": heb_ch,
            "hebrew_name": CONSONANT_NAMES.get(heb_ch, "?"),
            "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
            "source": source,
        }

    # 9. Save JSON report
    print_step("Saving results...")
    report = {
        "direction": direction,
        "max_dist": max_dist,
        "min_word_len": min_word_len,
        "n_candidates_tested": len(completions),
        "best_score": best_score,
        "best_n_matched_types": best_n_matched,
        "all_scores_sorted": sorted(all_scores, reverse=True)[:20],
        "mapping": readable_mapping,
        "unmapped_chars": unmapped,
        "top_matches": best_details[:100] if best_details else [],
        "section_breakdown": section_stats,
        "score_by_distance": {
            str(d): sum(m["weighted"] for m in (best_details or [])
                        if m["distance"] == d)
            for d in range(max_dist + 1)
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    JSON: {json_path}")

    # 10. Save readable text
    lines = [
        "# Fuzzy Decode Results",
        f"# Direction: {direction}, max_dist={max_dist}",
        f"# Score: {best_score}, matched types: {best_n_matched}",
        "",
        "=" * 60,
        "  COMPLETE MAPPING (EVA -> Hebrew -> Italian)",
        "=" * 60,
        "",
    ]
    for eva_ch, info in sorted(readable_mapping.items()):
        marker = "*" if info["source"] == "completion" else " "
        lines.append(
            f" {marker}{eva_ch} -> {info['hebrew']} "
            f"({info['hebrew_name']:8s}) -> {info['italian']}  "
            f"[{info['source']}]"
        )

    lines.extend(["", "=" * 60,
                   "  TOP 50 FUZZY MATCHES",
                   "=" * 60, ""])
    lines.append(f"{'Decoded':<15s} {'Target':<15s} {'Dist':>4s} "
                 f"{'Count':>6s} {'Score':>7s}  Gloss")
    lines.append("-" * 75)
    for m in (best_details or [])[:50]:
        lines.append(
            f"{m['decoded']:<15s} {m['target']:<15s} {m['distance']:>4d} "
            f"{m['count']:>6d} {m['weighted']:>7d}  {m['gloss']}"
        )

    lines.extend(["", "=" * 60,
                   "  SECTION BREAKDOWN",
                   "=" * 60, ""])
    for sec, info in sorted(section_stats.items()):
        lines.append(f"\n  Section {sec} "
                     f"(expected: {info['expected_domain']}):")
        lines.append(f"    Words: {info['n_words']}, "
                     f"decoded: {info['n_decoded']}, "
                     f"matched: {info['n_matched']}, "
                     f"score: {info['score']}")
        for tm in info["top_matches"][:5]:
            lines.append(f"    {tm['decoded']:<12s} -> {tm['target']:<12s} "
                         f"(d={tm['distance']}) {tm['gloss']}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"    Text: {txt_path}")

    # Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  FUZZY DECODE RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Direction: {direction}")
    click.echo(f"  Completions tested: {len(completions)}")
    click.echo(f"  Best score: {best_score}")
    click.echo(f"  Matched word types: {best_n_matched}")

    # Score by distance
    for d in range(max_dist + 1):
        d_score = report["score_by_distance"].get(str(d), 0)
        click.echo(f"    dist={d}: score={d_score} "
                   f"(weight={SCORE_WEIGHTS.get(d, 0)})")

    click.echo(f"\n  Mapping ({len(best_mapping)}/19 chars):")
    for eva_ch, info in sorted(readable_mapping.items()):
        marker = "*" if info["source"] == "completion" else " "
        click.echo(f"   {marker}{eva_ch} -> {info['hebrew']} "
                   f"({info['hebrew_name']:8s}) -> {info['italian']}")

    click.echo(f"\n  Top 15 matches:")
    for m in (best_details or [])[:15]:
        click.echo(f"    {m['decoded']:<12s} -> {m['target']:<12s} "
                   f"(d={m['distance']}) x{m['count']}  {m['gloss']}")
