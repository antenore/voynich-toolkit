"""
Hebrew decode: alternative decode path via Hebrew lexicon.

Loads 10 homophone groups from cipher_hypothesis.json, anchors
singleton groups using the champollion mapping, then generates
constrained candidates for multi-member groups and scores each
against the Hebrew consonantal lexicon (Lev<=1).

Compares Hebrew path results with Italian path for convergence.
"""
import json
import time
from collections import Counter, defaultdict
from itertools import product

import click

from .config import ToolkitConfig
from .fuzzy_utils import LengthBucketedIndex, SCORE_WEIGHTS
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Constants
# =====================================================================

HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"
EVA_CHARS = list("acdefghiklmnopqrsty")


# =====================================================================
# Loading
# =====================================================================

def load_homophone_groups(config):
    """Load 10 homophone groups from cipher_hypothesis.json.

    Returns dict {group_id: [eva_chars]}, e.g. {"10": ["a"], "1": ["c","s"]}.
    """
    path = config.stats_dir / "cipher_hypothesis.json"
    if not path.exists():
        raise click.ClickException(
            f"Cipher hypothesis not found: {path}\n"
            "  Run first: voynich cipher-hypothesis"
        )
    with open(path) as f:
        data = json.load(f)

    return data["homophone_analysis"]["groups"]


def load_champollion_mapping(config):
    """Load champollion mapping for anchoring singletons."""
    path = config.stats_dir / "champollion_report.json"
    if not path.exists():
        return {}, "rtl"

    with open(path) as f:
        data = json.load(f)

    mapping = {}
    for eva_ch, info in data.get("mapping_extended", {}).items():
        mapping[eva_ch] = info["hebrew"]

    return mapping, data.get("direction", "rtl")


def load_hebrew_lexicon(config):
    """Load Hebrew consonantal lexicon.

    Returns:
        all_forms: list of consonantal strings
        form_to_gloss: dict
        form_to_domain: dict
    """
    path = config.lexicon_dir / "lexicon.json"
    if not path.exists():
        raise click.ClickException(
            f"Hebrew lexicon not found: {path}\n"
            "  Run first: voynich prepare-lexicon"
        )
    with open(path) as f:
        data = json.load(f)

    all_forms = data["all_consonantal_forms"]

    form_to_gloss = {}
    form_to_domain = {}
    for consonants, entries in data.get("by_consonants", {}).items():
        if entries:
            form_to_gloss[consonants] = entries[0].get("gloss", "")
            form_to_domain[consonants] = entries[0].get("domain", "general")

    return all_forms, form_to_gloss, form_to_domain


# =====================================================================
# Anchor singletons
# =====================================================================

def anchor_singletons(groups, champollion_mapping):
    """Anchor singleton groups using champollion mapping.

    Singleton groups with champollion mapping are anchored (fixed).
    Singletons WITHOUT champollion mapping are treated as variable
    (added to multi_groups as 1-member groups).
    Multi-member groups stay as-is.

    Returns:
        anchored: dict {eva_char: hebrew_char} for all anchored chars
        multi_groups: dict {group_id: [eva_chars]} for multi-member groups
    """
    anchored = {}
    multi_groups = {}

    for gid, members in groups.items():
        if len(members) == 1:
            eva_ch = members[0]
            if eva_ch in champollion_mapping:
                anchored[eva_ch] = champollion_mapping[eva_ch]
            else:
                # Unanchored singleton: treat as variable group
                multi_groups[gid] = members
        else:
            multi_groups[gid] = members

    return anchored, multi_groups


# =====================================================================
# Candidate generation for multi-member groups
# =====================================================================

def compute_eva_frequencies(eva_data):
    """Compute frequency of each EVA char in the corpus."""
    counter = Counter()
    for w in eva_data["words"]:
        for ch in w:
            counter[ch] += 1
    total = sum(counter.values()) or 1
    return {ch: counter[ch] / total for ch in counter}


def generate_group_candidates(multi_groups, anchored, champollion_mapping,
                              eva_freqs, max_per_group=5):
    """Generate constrained candidates for multi-member groups.

    For each multi-member group, identifies possible Hebrew letters
    (from champollion mapping or frequency-ordered remaining letters).
    Picks top-N per group by frequency ranking.

    Returns list of full mappings (each a dict {eva: hebrew}).
    """
    # Collect ALL Hebrew letters reserved by champollion (across all
    # groups) so variable assignments never collide with fixed ones.
    all_champ_hebrew = set(anchored.values())
    for gid, members in multi_groups.items():
        for eva_ch in members:
            if eva_ch in champollion_mapping:
                all_champ_hebrew.add(champollion_mapping[eva_ch])

    # For each multi-member group, find candidate Hebrew assignments
    group_options = {}  # gid -> list of possible assignment dicts

    for gid, members in multi_groups.items():
        # Get champollion assignments for this group's members
        champ_assignments = {}
        for eva_ch in members:
            if eva_ch in champollion_mapping:
                champ_assignments[eva_ch] = champollion_mapping[eva_ch]

        if len(champ_assignments) == len(members):
            # All members have champollion assignments; use them
            group_options[gid] = [champ_assignments]
            continue

        # For members without champollion mapping, find unused Hebrew
        # letters ordered by frequency compatibility
        assigned_in_group = set(champ_assignments.values())
        unassigned_members = [m for m in members
                              if m not in champ_assignments]
        # Exclude all champollion-reserved Hebrew letters
        available_hebrew = [h for h in HEBREW_CHARS
                            if h not in all_champ_hebrew]

        if not unassigned_members:
            group_options[gid] = [champ_assignments]
            continue

        # For each unassigned member, pick top-N Hebrew by frequency
        # compatibility (EVA freq should match Italian phoneme freq)
        from .prepare_italian_lexicon import ITALIAN_LETTER_FREQS
        member_options = {}
        for eva_ch in unassigned_members:
            eva_freq = eva_freqs.get(eva_ch, 0)
            # Score by closeness of Italian phoneme freq to EVA freq
            scored = []
            for heb_ch in available_hebrew:
                it_ph = HEBREW_TO_ITALIAN.get(heb_ch, "?")
                it_freq = sum(ITALIAN_LETTER_FREQS.get(c, 0) for c in it_ph)
                diff = abs(eva_freq - it_freq)
                scored.append((diff, heb_ch))
            scored.sort()
            member_options[eva_ch] = [h for _, h in scored[:max_per_group]]

        # Generate all combinations for this group
        options_list = []
        for combo in product(*[member_options[m]
                                for m in unassigned_members]):
            # Check no duplicate Hebrew letters in this combo
            if len(set(combo)) != len(combo):
                continue
            assignment = dict(champ_assignments)
            for eva_ch, heb_ch in zip(unassigned_members, combo):
                assignment[eva_ch] = heb_ch
            options_list.append(assignment)

        group_options[gid] = options_list[:max_per_group * 2]

    # Now combine: for each combo of group assignments, build full mapping
    if not group_options:
        return [dict(anchored)]

    gids = sorted(group_options.keys())
    all_group_combos = list(product(*[group_options[g] for g in gids]))

    # Limit total combinations
    max_total = 300
    all_group_combos = all_group_combos[:max_total]

    candidates = []
    for combo in all_group_combos:
        full_mapping = dict(anchored)
        valid = True
        used = set(anchored.values())
        for group_assignment in combo:
            for eva_ch, heb_ch in group_assignment.items():
                if heb_ch in used:
                    # Conflict: skip this combo
                    valid = False
                    break
                full_mapping[eva_ch] = heb_ch
                used.add(heb_ch)
            if not valid:
                break
        if valid:
            candidates.append(full_mapping)

    return candidates


# =====================================================================
# Scoring
# =====================================================================

def decode_to_hebrew(eva_word, mapping, direction):
    """Decode EVA word to Hebrew consonantal string."""
    chars = list(eva_word) if direction == "ltr" else list(reversed(eva_word))
    hebrew = []
    for ch in chars:
        heb = mapping.get(ch)
        if heb is None:
            return None
        hebrew.append(heb)
    return "".join(hebrew)


def score_candidate(eva_data, mapping, direction, index,
                    max_dist=1, min_word_len=3):
    """Score a candidate mapping by fuzzy matches vs Hebrew lexicon.

    Returns (total_score, n_matched, top_matches).
    """
    word_freq = Counter(eva_data["words"])
    total_score = 0
    n_matched = 0
    matches = []

    for word, count in word_freq.most_common():
        if len(word) < min_word_len:
            continue
        hebrew = decode_to_hebrew(word, mapping, direction)
        if hebrew is None or len(hebrew) < min_word_len:
            continue

        best = index.best_match(hebrew, max_dist)
        if best:
            weighted = best.score * count
            total_score += weighted
            n_matched += 1
            if len(matches) < 100:
                matches.append({
                    "eva": word,
                    "hebrew": hebrew,
                    "target": best.target,
                    "distance": best.distance,
                    "count": count,
                    "weighted": weighted,
                    "gloss": best.gloss,
                    "domain": best.domain,
                })

    matches.sort(key=lambda m: -m["weighted"])
    return total_score, n_matched, matches


# =====================================================================
# Convergence with Italian path
# =====================================================================

def compute_convergence(hebrew_mapping, italian_mapping):
    """Compare Hebrew and Italian decode paths.

    Returns convergence score: fraction of EVA chars that map to the
    same Hebrew letter in both paths.
    """
    common_chars = set(hebrew_mapping.keys()) & set(italian_mapping.keys())
    if not common_chars:
        return {"n_common": 0, "n_agree": 0, "convergence": 0.0}

    n_agree = sum(1 for c in common_chars
                  if hebrew_mapping[c] == italian_mapping[c])

    return {
        "n_common": len(common_chars),
        "n_agree": n_agree,
        "convergence": round(n_agree / len(common_chars), 3),
        "agreements": sorted(c for c in common_chars
                             if hebrew_mapping[c] == italian_mapping[c]),
        "disagreements": sorted({
            c: {"hebrew_path": hebrew_mapping[c],
                "italian_path": italian_mapping[c]}
            for c in common_chars
            if hebrew_mapping[c] != italian_mapping[c]
        }),
    }


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, direction=None,
        n_candidates=243):
    """Entry point: Hebrew decode alternative path."""
    report_path = config.stats_dir / "hebrew_decode_report.json"

    if report_path.exists() and not force:
        click.echo("  Hebrew decode report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("HEBREW DECODE — Alternative Decode Path")

    # 1. Load homophone groups
    print_step("Loading homophone groups...")
    groups = load_homophone_groups(config)
    n_singleton = sum(1 for g in groups.values() if len(g) == 1)
    n_multi = sum(1 for g in groups.values() if len(g) > 1)
    click.echo(f"    {len(groups)} groups: {n_singleton} singleton, "
               f"{n_multi} multi-member")
    for gid, members in sorted(groups.items(), key=lambda x: len(x[1])):
        click.echo(f"      Group {gid}: {', '.join(members)}")

    # 2. Load champollion mapping for anchoring
    print_step("Loading champollion mapping...")
    champ_mapping, champ_direction = load_champollion_mapping(config)
    if direction is None:
        direction = champ_direction
    click.echo(f"    {len(champ_mapping)} chars from champollion, "
               f"direction: {direction}")

    # 3. Anchor singletons
    print_step("Anchoring singleton groups...")
    anchored, multi_groups = anchor_singletons(groups, champ_mapping)
    click.echo(f"    Anchored: {len(anchored)} chars")
    for eva_ch, heb_ch in sorted(anchored.items()):
        click.echo(f"      {eva_ch} -> {heb_ch} "
                   f"({CONSONANT_NAMES.get(heb_ch, '?')})")
    click.echo(f"    Multi-member groups: {len(multi_groups)}")

    # 4. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    heb_forms, heb_gloss, heb_domain = load_hebrew_lexicon(config)
    click.echo(f"    {len(heb_forms)} consonantal forms")

    # 5. Build fuzzy index (Lev<=1 for Hebrew)
    print_step("Building Hebrew fuzzy index...")
    heb_index = LengthBucketedIndex(heb_forms, heb_gloss, heb_domain)
    click.echo(f"    Indexed {heb_index.size} forms")

    # 6. Parse EVA
    print_step("Parsing EVA words...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"EVA file not found: {eva_file}\n  Run first: voynich eva"
        )
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words")

    # 7. Compute EVA frequencies
    eva_freqs = compute_eva_frequencies(eva_data)

    # 8. Generate candidates
    print_step("Generating mapping candidates...")
    t0 = time.time()
    candidates = generate_group_candidates(
        multi_groups, anchored, champ_mapping, eva_freqs,
        max_per_group=3,
    )
    click.echo(f"    {len(candidates)} candidate mappings")

    # 9. Score each candidate
    print_step(f"Scoring {len(candidates)} candidates vs Hebrew lexicon...")
    best_score = -1
    best_mapping = None
    best_matches = None
    best_n_matched = 0
    all_scores = []

    for i, candidate in enumerate(candidates):
        score, n_matched, matches = score_candidate(
            eva_data, candidate, direction, heb_index,
            max_dist=1, min_word_len=3,
        )
        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_mapping = candidate
            best_matches = matches
            best_n_matched = n_matched

        if (i + 1) % 50 == 0:
            click.echo(f"    Scored {i+1}/{len(candidates)}... "
                       f"best so far: {best_score}")

    elapsed = time.time() - t0
    click.echo(f"    Best score: {best_score} "
               f"({best_n_matched} word types matched) "
               f"in {elapsed:.1f}s")

    # 10. Convergence with Italian path
    print_step("Computing convergence with Italian path...")
    # Load Italian path mapping
    italian_mapping = {}
    fuzzy_path = config.stats_dir / "fuzzy_matches.json"
    if fuzzy_path.exists():
        with open(fuzzy_path) as f:
            fd = json.load(f)
        for eva_ch, info in fd.get("mapping", {}).items():
            italian_mapping[eva_ch] = info["hebrew"]
    elif champ_mapping:
        italian_mapping = dict(champ_mapping)

    convergence = compute_convergence(best_mapping or {}, italian_mapping)
    click.echo(f"    Convergence: {convergence['convergence']:.1%} "
               f"({convergence['n_agree']}/{convergence['n_common']} agree)")

    # 11. Build readable mapping
    readable_mapping = {}
    if best_mapping:
        for eva_ch in sorted(best_mapping):
            heb_ch = best_mapping[eva_ch]
            readable_mapping[eva_ch] = {
                "hebrew": heb_ch,
                "hebrew_name": CONSONANT_NAMES.get(heb_ch, "?"),
                "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
                "anchored": eva_ch in anchored,
            }

    # 12. Save report
    print_step("Saving report...")
    report = {
        "direction": direction,
        "n_candidates_tested": len(candidates),
        "best_score": best_score,
        "best_n_matched": best_n_matched,
        "all_scores_sorted": sorted(all_scores, reverse=True)[:20],
        "mapping": readable_mapping,
        "anchored_chars": sorted(anchored.keys()),
        "multi_member_groups": {
            gid: members for gid, members in multi_groups.items()
        },
        "homophone_groups": groups,
        "convergence_with_italian": convergence,
        "top_matches": (best_matches or [])[:80],
        "score_by_distance": {
            str(d): sum(m["weighted"] for m in (best_matches or [])
                        if m["distance"] == d)
            for d in range(2)
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Report: {report_path}")

    # Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  HEBREW DECODE RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Direction: {direction}")
    click.echo(f"  Candidates tested: {len(candidates)}")
    click.echo(f"  Best score: {best_score}")
    click.echo(f"  Matched word types: {best_n_matched}")

    if readable_mapping:
        click.echo(f"\n  Best mapping:")
        for eva_ch, info in sorted(readable_mapping.items()):
            marker = " " if info["anchored"] else "*"
            click.echo(f"   {marker}{eva_ch} -> {info['hebrew']} "
                       f"({info['hebrew_name']:8s}) -> {info['italian']}")

    click.echo(f"\n  Convergence with Italian path: "
               f"{convergence['convergence']:.1%}")
    if convergence.get("agreements"):
        click.echo(f"    Agree on: {', '.join(convergence['agreements'])}")

    click.echo(f"\n  Top 15 Hebrew matches:")
    for m in (best_matches or [])[:15]:
        click.echo(f"    {m['eva']:<12s} -> {m['hebrew']:<8s} "
                   f"~ {m['target']:<8s} (d={m['distance']}) "
                   f"x{m['count']}  {m['gloss']}")

    # Verdict
    click.echo(f"\n  {'=' * 40}")
    conv = convergence["convergence"]
    if conv >= 0.7:
        click.echo("  VERDICT: STRONG convergence between paths")
        click.echo("  (Hebrew and Italian paths agree)")
    elif conv >= 0.4:
        click.echo("  VERDICT: MODERATE convergence")
        click.echo("  (partial agreement, some alternatives)")
    else:
        click.echo("  VERDICT: LOW convergence")
        click.echo("  (paths diverge, multiple interpretations possible)")
    click.echo(f"  {'=' * 40}")
