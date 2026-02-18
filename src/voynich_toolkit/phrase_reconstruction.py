"""Iterative phrase reconstruction from decoded Voynich text.

Uses anchor confidence scoring, contextual candidate generation for
unknown words, cross-context propagation, and gematria analysis.

CLI: voynich --force phrase-reconstruction
"""

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES, decode_word
from .semantic_coherence import (
    SEMANTIC_LEVELS,
    build_type_classification,
    get_gloss_label,
    load_annotation_data,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# ── Constants ────────────────────────────────────────────────────────

MAX_ITER = 5
CANDIDATE_DIST_MAX = 1
CANDIDATE_TOP_K = 5
MIN_PASSAGE_RATIO = 0.3      # minimum glossed_ratio to attempt reconstruction
RESOLVE_THRESHOLD = 0.3       # minimum confidence to accept a candidate
MIN_WORD_LEN = 3              # minimum Hebrew word length for candidate generation

SECTION_TO_DOMAIN = {
    "H": "botanical",
    "P": "medical",
    "B": "medical",
    "S": "astronomical",
    "Z": "astronomical",
    "C": "astronomical",
    "T": "general",
}

# Anchor category → expected manuscript section(s)
CATEGORY_SECTIONS = {
    "botanica_parti": {"H", "P"},
    "botanica_azioni": {"H", "P"},
    "botanica_bos": {"H", "P"},
    "farmaceutica_shem_tov": {"H", "P"},
    "colori": {"H", "P", "B"},
    "corpo_medicina": {"B", "P"},
    "medicina_bos": {"B", "P"},
    "liquidi": {"B", "P", "H"},
    "misure": {"H", "P", "B"},
    "astrologia": {"S", "Z", "C"},
    "astronomia_ibn_ezra": {"S", "Z", "C"},
    "pianeti": {"S", "Z"},
    "numeri": {"S", "Z", "C"},
    "alchimia": {"B", "P"},
    "cabala": {"T", "S"},
}

# Domain keywords for semantic matching
DOMAIN_KEYWORDS = {
    "botanical": {
        "tree", "plant", "herb", "seed", "fruit", "flower", "vine",
        "wheat", "barley", "grain", "leaf", "root", "branch", "thorn",
        "fig", "olive", "palm", "garden", "field", "sow", "harvest",
        "spice", "myrrh", "rose", "lily", "aloe", "resin", "wood",
    },
    "astronomical": {
        "star", "sun", "moon", "heaven", "sky", "constellation",
        "planet", "zodiac", "light", "darkness", "day", "night",
        "year", "month", "season", "time", "degree", "sign",
    },
    "medical": {
        "body", "head", "eye", "ear", "mouth", "tongue", "heart",
        "blood", "skin", "bone", "heal", "sick", "disease", "wound",
        "medicine", "remedy", "ointment", "cure", "oil", "honey",
        "hot", "cold", "moist", "dry",
    },
}

# Standard gematria values for mapped Hebrew letters
GEMATRIA = {
    "A": 1, "b": 2, "g": 3, "d": 4, "h": 5, "w": 6, "z": 7,
    "X": 8, "J": 9, "y": 10, "k": 20, "l": 30, "m": 40, "n": 50,
    "s": 60, "E": 70, "p": 80, "C": 90, "q": 100, "r": 200,
    "S": 300, "t": 400,
}

# Anchor confidence weights
W_LENGTH = 0.25
W_DISTANCE = 0.20
W_DOMAIN = 0.20
W_FREQ = 0.15
W_CONTEXT = 0.20


# =====================================================================
# Corpus pre-decoding (single pass)
# =====================================================================

def predecode_corpus(page_data):
    """Decode entire corpus once. Returns list of decoded pages.

    Each page: {folio, section, decoded_lines: list[list[str]]}
    where each decoded_lines entry is a list of Hebrew words per line.

    Also returns:
      word_freqs: Counter {hebrew_word: token_count}
      all_hebrew_types: set of unique Hebrew types
    """
    decoded_pages = []
    word_freqs = Counter()
    all_hebrew_types = set()

    for page in page_data:
        folio = page.get("folio", "?")
        section = page.get("section", "?")
        decoded_lines = []
        for line_words in page.get("line_words", []):
            heb_line = []
            for ew in line_words:
                _, h, _ = decode_word(ew)
                heb_line.append(h)
                word_freqs[h] += 1
                all_hebrew_types.add(h)
            decoded_lines.append(heb_line)
        decoded_pages.append({
            "folio": folio,
            "section": section,
            "decoded_lines": decoded_lines,
        })

    return decoded_pages, word_freqs, all_hebrew_types


# =====================================================================
# Phase A: Anchor Confidence Scoring
# =====================================================================

def score_anchor_confidence(config, decoded_pages, classification):
    """Score confidence of each anchor match.

    Returns dict {anchor: {score, grade, factors, ...}}.
    """
    print_step("Phase A: Anchor confidence scoring")

    db_path = config.output_dir.parent / "voynich.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    anchors = []
    for row in conn.execute(
        "SELECT anchor, gloss, language, category, category_label, "
        "total_occurrences, best_distance, n_decoded_forms "
        "FROM anchor_matches"
    ):
        anchors.append(dict(row))
    conn.close()

    if not anchors:
        print("  No anchor matches found in DB.")
        return {}

    max_freq = max(a["total_occurrences"] for a in anchors)

    # Build word→sections and word→line_ratios from pre-decoded pages
    word_sections = defaultdict(set)
    word_line_ratios = defaultdict(list)

    for dp in decoded_pages:
        sec = dp["section"]
        for heb_line in dp["decoded_lines"]:
            if not heb_line:
                continue
            n_sem = sum(
                1 for h in heb_line
                if classification.get(h, "F") in SEMANTIC_LEVELS
            )
            ratio = n_sem / len(heb_line)
            for h in heb_line:
                word_sections[h].add(sec)
                word_line_ratios[h].append(ratio)

    # Score each anchor
    results = {}
    for a in anchors:
        name = a["anchor"]
        length = len(name)
        cat = a["category"]

        # 1. Length score
        if length >= 4:
            length_score = 1.0
        elif length == 3:
            length_score = 0.5
        else:
            length_score = 0.1

        # 2. Distance score
        dist = a["best_distance"]
        distance_score = 1.0 if dist == 0 else 0.3

        # 3. Domain score
        expected = CATEGORY_SECTIONS.get(cat, set())
        if expected:
            actual_sections = word_sections.get(name, set())
            domain_score = 1.0 if (actual_sections & expected) else 0.3
        else:
            domain_score = 0.5

        # 4. Frequency score (log-normalized)
        freq = a["total_occurrences"]
        freq_score = (
            math.log(freq + 1) / math.log(max_freq + 1) if max_freq > 0 else 0
        )

        # 5. Context score
        n_forms = a["n_decoded_forms"]
        form_penalty = max(0, 1.0 - math.log(n_forms + 1) / math.log(40))
        ratios = word_line_ratios.get(name, [])
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.3
        context_score = 0.5 * avg_ratio + 0.5 * form_penalty

        # Composite
        score = (
            W_LENGTH * length_score
            + W_DISTANCE * distance_score
            + W_DOMAIN * domain_score
            + W_FREQ * freq_score
            + W_CONTEXT * context_score
        )

        grade = "A" if score > 0.7 else ("B" if score >= 0.4 else "C")

        results[name] = {
            "score": round(score, 3),
            "grade": grade,
            "length_score": round(length_score, 3),
            "distance_score": round(distance_score, 3),
            "domain_score": round(domain_score, 3),
            "freq_score": round(freq_score, 3),
            "context_score": round(context_score, 3),
            "gloss": a["gloss"],
            "language": a["language"],
            "category": a["category"],
            "best_distance": dist,
            "total_occurrences": freq,
            "n_decoded_forms": n_forms,
        }

    n_a = sum(1 for v in results.values() if v["grade"] == "A")
    n_b = sum(1 for v in results.values() if v["grade"] == "B")
    n_c = sum(1 for v in results.values() if v["grade"] == "C")
    print(f"  {len(results)} anchors scored: "
          f"{n_a} grade-A, {n_b} grade-B, {n_c} grade-C")

    return results


# =====================================================================
# Phase B: Candidate Generation
# =====================================================================

def _build_lexicon_index(config):
    """Build lexicon lookup structures.

    Returns:
      lex_dict: dict {consonants: (gloss, domain)} for O(1) lookup
      lex_set: set of all consonantal forms
    """
    db_path = config.output_dir.parent / "voynich.db"
    conn = sqlite3.connect(str(db_path))

    lex_dict = {}
    for row in conn.execute(
        "SELECT consonants, gloss, domain FROM lexicon"
    ):
        c, g, d = row
        # Keep first entry if duplicate (prefer one with gloss)
        if c not in lex_dict or (g and not lex_dict[c][0]):
            lex_dict[c] = (g or "", d or "")
    conn.close()
    return lex_dict, set(lex_dict.keys())


# Hebrew alphabet for edit generation (all 22 letters)
_HEBREW_ALPHA = "AbgdhwzXJyklmnsEpCqrSt"


def _generate_edit1(word):
    """Generate all strings at Levenshtein distance 1 from word.

    Yields (variant, edit_type) where edit_type is 'sub', 'del', 'ins'.
    """
    L = len(word)
    # Substitutions: L * 21
    for i in range(L):
        for c in _HEBREW_ALPHA:
            if c != word[i]:
                yield word[:i] + c + word[i + 1:]
    # Deletions: L
    for i in range(L):
        yield word[:i] + word[i + 1:]
    # Insertions: (L+1) * 22
    for i in range(L + 1):
        for c in _HEBREW_ALPHA:
            yield word[:i] + c + word[i:]


def _classify_gloss_domain(gloss):
    """Classify a gloss string into domain(s) via keyword matching."""
    if not gloss:
        return set()
    gl = gloss.lower()
    domains = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in gl:
                domains.add(domain)
                break
    return domains


def semantic_match(candidate_gloss, context_glosses, section):
    """Score semantic match between a candidate and its context.

    Returns float 0-1.
    """
    score = 0.0
    expected_domain = SECTION_TO_DOMAIN.get(section, "general")

    candidate_domains = _classify_gloss_domain(candidate_gloss)
    if expected_domain in candidate_domains:
        score += 0.5

    context_text = " ".join(context_glosses).lower()
    context_domains = set()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in context_text:
                context_domains.add(domain)
                break
    if candidate_domains & context_domains:
        score += 0.3

    if candidate_gloss and len(candidate_gloss) > 5:
        score += 0.2

    return min(score, 1.0)


def generate_candidates(hebrew_type, section, context_glosses,
                        lex_dict, lex_set):
    """Generate candidate lexicon matches for an unknown Hebrew type.

    Uses edit-variant generation + set lookup instead of brute-force scan.
    Returns list of (form, gloss, distance, semantic_score), max CANDIDATE_TOP_K.
    """
    length = len(hebrew_type)
    if length < MIN_WORD_LEN:
        return []

    candidates = []

    # d=0: O(1) set check
    if hebrew_type in lex_set:
        gloss, domain = lex_dict[hebrew_type]
        sem_score = semantic_match(gloss, context_glosses, section)
        candidates.append((hebrew_type, gloss, 0, sem_score, sem_score + 1.0))

    # d=1: generate all 1-edit variants, check set membership
    seen = {hebrew_type}
    for variant in _generate_edit1(hebrew_type):
        if variant in seen:
            continue
        seen.add(variant)
        if variant in lex_set:
            gloss, domain = lex_dict[variant]
            sem_score = semantic_match(gloss, context_glosses, section)
            candidates.append((variant, gloss, 1, sem_score, sem_score))

    candidates.sort(key=lambda x: (-x[4], x[2]))
    return [(c[0], c[1], c[2], c[3]) for c in candidates[:CANDIDATE_TOP_K]]


def _gather_context_glosses(line_heb_words, classification, glossed, morpho,
                            compound):
    """Collect glosses from known words in a line for context."""
    glosses = []
    for h in line_heb_words:
        level = classification.get(h, "F")
        if level in SEMANTIC_LEVELS:
            g = get_gloss_label(h, level, glossed, morpho, compound)
            if g:
                glosses.append(g)
    return glosses


def run_candidate_generation(decoded_pages, classification, glossed, morpho,
                             compound, resolved_words, lex_dict, lex_set):
    """Generate candidates for unknown types in high-density passages.

    Returns dict {hebrew_type: {candidate, candidate_gloss, distance,
                                semantic_score, confidence, source_passage}}.
    """
    already_resolved = set(resolved_words.keys())
    type_f_contexts = defaultdict(list)

    for dp in decoded_pages:
        folio = dp["folio"]
        section = dp["section"]
        for heb_line in dp["decoded_lines"]:
            if not heb_line:
                continue

            n_sem = sum(
                1 for h in heb_line
                if classification.get(h, "F") in SEMANTIC_LEVELS
            )
            ratio = n_sem / len(heb_line)
            if ratio < MIN_PASSAGE_RATIO:
                continue

            context_glosses = _gather_context_glosses(
                heb_line, classification, glossed, morpho, compound
            )

            for h in heb_line:
                level = classification.get(h, "F")
                if level != "F" or h in already_resolved:
                    continue
                if len(h) < MIN_WORD_LEN:
                    continue
                type_f_contexts[h].append((section, context_glosses, folio))

    new_resolutions = {}
    for heb_type, contexts in type_f_contexts.items():
        best_ctx = max(contexts, key=lambda x: len(x[1]))
        section, ctx_glosses, folio = best_ctx

        candidates = generate_candidates(
            heb_type, section, ctx_glosses, lex_dict, lex_set
        )
        if not candidates:
            continue

        form, gloss, dist, sem_score = candidates[0]

        # Confidence formula
        dist_base = 0.4 if dist == 0 else 0.15
        len_bonus = min(0.15, 0.03 * max(0, len(heb_type) - 3))
        ctx_bonus = min(0.15, 0.01 * len(contexts))
        confidence = (
            dist_base
            + 0.25 * sem_score
            + 0.15 * min(1, len(ctx_glosses) / 8)
            + len_bonus
            + ctx_bonus
        )
        confidence = min(confidence, 1.0)

        if confidence >= RESOLVE_THRESHOLD:
            new_resolutions[heb_type] = {
                "candidate": form,
                "candidate_gloss": gloss,
                "distance": dist,
                "semantic_score": round(sem_score, 3),
                "confidence": round(confidence, 3),
                "source_passage": folio,
                "n_contexts": len(contexts),
            }

    return new_resolutions


# =====================================================================
# Phase C: Cross-Context Propagation
# =====================================================================

def propagate_resolutions(resolved_words, classification):
    """Update classification with newly resolved words.

    Returns number of types newly added.
    """
    n_new = 0
    for heb_type in resolved_words:
        if classification.get(heb_type, "F") == "F":
            classification[heb_type] = "R"  # R = resolved-by-context
            n_new += 1
    return n_new


# =====================================================================
# Phase D: Iterative Loop
# =====================================================================

def run_iterative_loop(decoded_pages, word_freqs, classification, glossed,
                       morpho, compound, lex_dict, lex_set):
    """Run the iterative candidate generation + propagation loop.

    Returns (resolved_words, iteration_log).
    """
    print_step("Phase D: Iterative reconstruction loop")

    resolved_words = {}
    iteration_log = []

    for iteration in range(MAX_ITER):
        new = run_candidate_generation(
            decoded_pages, classification, glossed, morpho, compound,
            resolved_words, lex_dict, lex_set,
        )

        if not new:
            print(f"  Iteration {iteration}: 0 new types → converged")
            iteration_log.append({
                "iteration": iteration,
                "new_types": 0,
                "new_tokens": 0,
                "cumulative_types": len(resolved_words),
            })
            break

        for heb_type, info in new.items():
            info["iteration"] = iteration
            resolved_words[heb_type] = info

        propagate_resolutions(resolved_words, classification)

        # Count tokens using pre-computed freqs
        token_count = sum(word_freqs.get(h, 0) for h in new)

        print(f"  Iteration {iteration}: +{len(new)} types, "
              f"+{token_count} tokens, "
              f"cumulative {len(resolved_words)} types")

        iteration_log.append({
            "iteration": iteration,
            "new_types": len(new),
            "new_tokens": token_count,
            "cumulative_types": len(resolved_words),
        })

        if len(new) < 3:
            print(f"  Diminishing returns → stopping")
            break

    return resolved_words, iteration_log


# =====================================================================
# Phase E: Gematria Analysis
# =====================================================================

def compute_gematria(hebrew_word):
    """Compute standard gematria value for a Hebrew word.

    Returns (value, computable) where computable is True if all letters
    have known gematria values.
    """
    total = 0
    for ch in hebrew_word:
        if ch in GEMATRIA:
            total += GEMATRIA[ch]
        else:
            return 0, False
    return total, True


def analyze_gematria(decoded_pages):
    """Analyze gematria patterns in the decoded corpus.

    Returns dict with distribution, notable values, and section analysis.
    """
    print_step("Phase E: Gematria analysis")

    value_counter = Counter()
    section_values = defaultdict(list)
    incomputable = 0
    total_words = 0

    for dp in decoded_pages:
        section = dp["section"]
        for heb_line in dp["decoded_lines"]:
            for heb in heb_line:
                total_words += 1
                val, ok = compute_gematria(heb)
                if ok and val > 0:
                    value_counter[val] += 1
                    section_values[section].append(val)
                else:
                    incomputable += 1

    computable = total_words - incomputable
    computable_pct = computable / total_words * 100 if total_words else 0

    NOTABLE = {
        7: "7 planets / days of week",
        12: "12 zodiac signs / months",
        28: "28 lunar mansions",
        30: "30 days/month",
        360: "360 degrees",
        365: "365 days/year",
    }
    notable_hits = {}
    for val, label in NOTABLE.items():
        count = value_counter.get(val, 0)
        if count > 0:
            notable_hits[val] = {"label": label, "count": count}

    top_values = value_counter.most_common(20)

    section_stats = {}
    for sec, vals in sorted(section_values.items()):
        if vals:
            sorted_vals = sorted(vals)
            section_stats[sec] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 1),
                "median": sorted_vals[len(vals) // 2],
                "max": max(vals),
            }

    # Zodiac consecutive gematria sequences
    zodiac_vals = section_values.get("Z", [])
    sequences_found = []
    if zodiac_vals:
        sorted_unique = sorted(set(zodiac_vals))
        run_start = sorted_unique[0]
        run_len = 1
        for i in range(1, len(sorted_unique)):
            if sorted_unique[i] == sorted_unique[i - 1] + 1:
                run_len += 1
            else:
                if run_len >= 3:
                    sequences_found.append({
                        "start": run_start,
                        "length": run_len,
                        "end": sorted_unique[i - 1],
                    })
                run_start = sorted_unique[i]
                run_len = 1
        if run_len >= 3:
            sequences_found.append({
                "start": run_start,
                "length": run_len,
                "end": sorted_unique[-1],
            })
    # Sort by length descending
    sequences_found.sort(key=lambda x: -x["length"])

    result = {
        "total_words": total_words,
        "computable": computable,
        "computable_pct": round(computable_pct, 1),
        "incomputable": incomputable,
        "unique_values": len(value_counter),
        "top_values": [{"value": v, "count": c} for v, c in top_values],
        "notable_values": notable_hits,
        "section_stats": section_stats,
        "zodiac_sequences": sequences_found,
    }

    print(f"  {computable}/{total_words} words computable "
          f"({computable_pct:.1f}%)")
    print(f"  {len(value_counter)} unique gematria values")
    if notable_hits:
        for val, info in sorted(notable_hits.items()):
            print(f"    {val} ({info['label']}): {info['count']} occurrences")
    if sequences_found:
        top_seqs = sequences_found[:10]
        print(f"  {len(sequences_found)} zodiac consecutive sequences "
              f"(top {len(top_seqs)} by length):")
        for seq in top_seqs:
            print(f"    {seq['start']}-{seq['end']} (length {seq['length']})")

    return result


# =====================================================================
# Output
# =====================================================================

def write_json_report(output_path, anchor_conf, resolved_words,
                      iteration_log, gematria):
    """Write JSON report."""
    report = {
        "anchor_confidence": anchor_conf,
        "reconstructed_words": resolved_words,
        "iteration_log": iteration_log,
        "gematria": gematria,
        "summary": {
            "n_anchors_scored": len(anchor_conf),
            "n_anchors_grade_a": sum(
                1 for v in anchor_conf.values() if v["grade"] == "A"
            ),
            "n_reconstructed_types": len(resolved_words),
            "n_iterations": len(iteration_log),
            "gematria_computable_pct": gematria.get("computable_pct", 0),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_summary(output_path, anchor_conf, resolved_words,
                  iteration_log, gematria):
    """Write human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PHRASE RECONSTRUCTION — Summary")
    lines.append("=" * 60)

    # Phase A
    lines.append("\n── Phase A: Anchor Confidence ──")
    n_a = sum(1 for v in anchor_conf.values() if v["grade"] == "A")
    n_b = sum(1 for v in anchor_conf.values() if v["grade"] == "B")
    n_c = sum(1 for v in anchor_conf.values() if v["grade"] == "C")
    lines.append(f"  Total anchors: {len(anchor_conf)}")
    lines.append(f"  Grade A (>0.7): {n_a}")
    lines.append(f"  Grade B (0.4-0.7): {n_b}")
    lines.append(f"  Grade C (<0.4): {n_c}")

    sorted_anchors = sorted(
        anchor_conf.items(), key=lambda x: -x[1]["score"]
    )
    lines.append(f"\n  Top-10 most confident:")
    lines.append(f"  {'Anchor':<15} {'Score':>5} {'Gr':>2} "
                 f"{'Dist':>4} {'Occ':>5} {'Gloss':<30}")
    for name, info in sorted_anchors[:10]:
        lines.append(
            f"  {name:<15} {info['score']:5.3f} {info['grade']:>2} "
            f"  d={info['best_distance']:<2} {info['total_occurrences']:5d} "
            f"{info['gloss'][:30]}"
        )

    lines.append(f"\n  Bottom-5 least confident:")
    for name, info in sorted_anchors[-5:]:
        lines.append(
            f"  {name:<15} {info['score']:5.3f} {info['grade']:>2} "
            f"  d={info['best_distance']:<2} {info['total_occurrences']:5d} "
            f"{info['gloss'][:30]}"
        )

    # Phase B-D
    lines.append("\n── Phase B-D: Iterative Reconstruction ──")
    if iteration_log:
        for entry in iteration_log:
            lines.append(
                f"  Iteration {entry['iteration']}: "
                f"+{entry['new_types']} types, "
                f"+{entry.get('new_tokens', 0)} tokens, "
                f"cumulative {entry['cumulative_types']}"
            )
    lines.append(f"\n  Total reconstructed types: {len(resolved_words)}")

    if resolved_words:
        sorted_recon = sorted(
            resolved_words.items(),
            key=lambda x: (-x[1]["confidence"], -x[1].get("n_contexts", 0)),
        )
        lines.append(f"\n  Top-10 reconstructed words:")
        lines.append(
            f"  {'Hebrew':<12} {'Candidate':<12} {'Dist':>4} "
            f"{'Conf':>5} {'Ctx':>3} {'Gloss':<30}"
        )
        for heb, info in sorted_recon[:10]:
            lines.append(
                f"  {heb:<12} {info['candidate']:<12} "
                f"  d={info['distance']:<2} {info['confidence']:5.3f} "
                f"{info.get('n_contexts', 0):3d} "
                f"{info['candidate_gloss'][:30]}"
            )

    # Phase E
    lines.append("\n── Phase E: Gematria Analysis ──")
    gem = gematria
    lines.append(
        f"  Computable: {gem['computable']}/{gem['total_words']} "
        f"({gem['computable_pct']:.1f}%)"
    )
    lines.append(f"  Unique values: {gem['unique_values']}")

    if gem.get("notable_values"):
        lines.append("  Notable values:")
        for val, info in sorted(gem["notable_values"].items(),
                                key=lambda x: int(x[0])):
            val_int = int(val) if isinstance(val, str) else val
            lines.append(
                f"    {val_int:>5} ({info['label']}): {info['count']} occ"
            )

    if gem.get("top_values"):
        lines.append("  Top-10 values by frequency:")
        for entry in gem["top_values"][:10]:
            lines.append(
                f"    value={entry['value']:>5}: {entry['count']} occ"
            )

    if gem.get("section_stats"):
        lines.append("  Per-section gematria:")
        for sec, st in sorted(gem["section_stats"].items()):
            sec_name = SECTION_NAMES.get(sec, sec)
            lines.append(
                f"    {sec} ({sec_name:<14}): n={st['n']:>5}, "
                f"mean={st['mean']:>6.1f}, median={st['median']:>5}"
            )

    if gem.get("zodiac_sequences"):
        seqs = gem["zodiac_sequences"]
        lines.append(f"  Zodiac consecutive sequences: {len(seqs)} total "
                     f"(top 10 by length):")
        for seq in seqs[:10]:
            lines.append(
                f"    {seq['start']}-{seq['end']} (length {seq['length']})"
            )

    lines.append("\n" + "=" * 60)
    text = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False):
    """Run iterative phrase reconstruction pipeline."""
    json_path = config.stats_dir / "phrase_reconstruction.json"
    txt_path = config.stats_dir / "phrase_reconstruction_summary.txt"

    if not force and json_path.exists():
        print(f"  Output exists: {json_path.name} (use --force to re-run)")
        return

    print_header("PHRASE RECONSTRUCTION")

    # ── Load corpus + single-pass decode ─────────────────────────────
    print_step("Loading corpus and pre-decoding")
    eva_path = config.eva_data_dir / "LSI_ivtff_0d.txt"
    parsed = parse_eva_words(eva_path)
    page_data = parsed["pages"]

    decoded_pages, word_freqs, all_hebrew_types = predecode_corpus(page_data)
    print(f"  {parsed['total_words']} words, {len(page_data)} pages, "
          f"{len(all_hebrew_types)} unique types")

    # ── Load annotation data ─────────────────────────────────────────
    print_step("Loading annotation data")
    glossed, morpho, compound, attested, resolved = load_annotation_data(config)

    classification = build_type_classification(
        all_hebrew_types, glossed, morpho, compound, attested, resolved
    )

    n_f = sum(1 for v in classification.values() if v == "F")
    n_sem = sum(1 for v in classification.values() if v in SEMANTIC_LEVELS)
    print(f"  {n_sem} semantic (A/B/C), {n_f} unknown (F)")

    # ── Phase A ──────────────────────────────────────────────────────
    anchor_conf = score_anchor_confidence(
        config, decoded_pages, classification
    )

    # ── Phase B: Build lexicon index ─────────────────────────────────
    print_step("Phase B: Building lexicon index")
    lex_dict, lex_set = _build_lexicon_index(config)
    print(f"  {len(lex_dict)} lexicon entries indexed")

    # ── Phase C+D ────────────────────────────────────────────────────
    resolved_words, iteration_log = run_iterative_loop(
        decoded_pages, word_freqs, classification, glossed, morpho, compound,
        lex_dict, lex_set,
    )

    # ── Phase E ──────────────────────────────────────────────────────
    gematria = analyze_gematria(decoded_pages)

    # ── Output ───────────────────────────────────────────────────────
    print_step("Writing output")
    write_json_report(json_path, anchor_conf, resolved_words,
                      iteration_log, gematria)
    summary = write_summary(txt_path, anchor_conf, resolved_words,
                            iteration_log, gematria)
    print(f"  {json_path.name}")
    print(f"  {txt_path.name}")

    print()
    print(summary)
