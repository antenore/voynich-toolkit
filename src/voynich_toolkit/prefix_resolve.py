"""
Prefix-resolve: resolve the 3 unknown EVA characters (f, i, q).

Hypothesis: the unknown chars may be Hebrew grammatical prefixes
(be-, le-, ve-, ha-, ke-, mi-) attached to words, not phonetic letters.

Pipeline:
  Step 0 — Test if f/i/q are positional prefixes (strip & re-decode)
  Step 1 — Extract constraint words (1 unknown, 4+ known chars)
  Step 2 — Brute-force: try all 22 Hebrew letters for each unknown
  Step 3 — Global consistency: which letter maximizes lexical hits?
  Step 4 — Cross-validation via anchor words
  Step 5 — Cascade: re-process Type C words with newly resolved chars
"""
import json
from collections import Counter, defaultdict

import click
from rapidfuzz.distance import Levenshtein

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES, load_convergent_mapping
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# All 22 Hebrew consonants in ASCII
HEBREW_LETTERS = list("AbgdhwzXJyklmnsEpCqrSt")

# Hebrew single-letter prefixes and their meanings
HEBREW_PREFIXES = {
    "b": ("bet", "be-", "in/with"),
    "l": ("lamed", "le-", "to/for"),
    "m": ("mem", "mi-", "from"),
    "k": ("kaf", "ke-", "like/as"),
    "w": ("vav", "ve-", "and"),
    "h": ("he", "ha-", "the"),
    "S": ("shin", "she-", "that/which"),
}

# Expected prefix frequencies in typical Hebrew text
HEBREW_PREFIX_FREQS = {
    "w": 0.15,   # ve- (and) — most common
    "l": 0.08,   # le- (to/for)
    "b": 0.06,   # be- (in/with)
    "h": 0.05,   # ha- (the)
    "m": 0.04,   # mi- (from)
    "k": 0.02,   # ke- (like)
    "S": 0.01,   # she- (that)
}


def _load_lexicons(config):
    """Load Italian and Hebrew lexicons as sets for fast lookup."""
    italian_path = config.lexicon_dir / "italian_lexicon.json"
    hebrew_path = config.lexicon_dir / "lexicon.json"

    italian_set = set()
    italian_gloss = {}
    if italian_path.exists():
        with open(italian_path) as f:
            data = json.load(f)
        italian_set = set(data.get("all_forms", []))
        italian_gloss = data.get("form_to_gloss", {})

    hebrew_set = set()
    hebrew_by_cons = {}
    if hebrew_path.exists():
        with open(hebrew_path) as f:
            data = json.load(f)
        hebrew_set = set(data.get("all_consonantal_forms", []))
        hebrew_by_cons = data.get("by_consonants", {})

    return italian_set, italian_gloss, hebrew_set, hebrew_by_cons


def _decode_chars(eva_word, mapping, direction):
    """Decode EVA word to Hebrew ASCII using the 16-char mapping.

    Returns (hebrew_str, unknown_positions) where unknown_positions
    is a list of (position_in_hebrew, eva_char) for unmapped chars.
    """
    chars = list(reversed(eva_word)) if direction == "rtl" else list(eva_word)
    result = []
    unknowns = []
    for i, ch in enumerate(chars):
        if ch in mapping:
            result.append(mapping[ch])
        else:
            result.append(None)
            unknowns.append((i, ch))
    return result, unknowns


def _hebrew_to_italian(hebrew_str):
    """Convert Hebrew ASCII string to Italian phonemes."""
    return "".join(HEBREW_TO_ITALIAN.get(ch, "?") for ch in hebrew_str)


def _normalize_italian_for_lookup(word):
    """Normalize Italian word for lexicon lookup (simplified)."""
    w = word.lower()
    # Common medieval Italian normalizations
    w = w.replace("ch", "k").replace("gh", "g")
    w = w.replace("ph", "p").replace("th", "t")
    # Double consonants → single
    prev = ""
    result = []
    for c in w:
        if c != prev or c in "aeiou":
            result.append(c)
        prev = c
    return "".join(result)


# =====================================================================
# Step 0: Prefix hypothesis test
# =====================================================================

def step0_positional_analysis(eva_data, divergent_chars, direction):
    """Compute positional distribution of each unknown char in EVA words.

    Returns dict: {char: {initial, medial, final, standalone, total}}.
    """
    results = {}
    for ch in sorted(divergent_chars):
        counts = {"initial": 0, "medial": 0, "final": 0,
                  "standalone": 0, "total": 0}
        for page in eva_data["pages"]:
            for word in page["words"]:
                if ch not in word:
                    continue
                for i, c in enumerate(word):
                    if c != ch:
                        continue
                    counts["total"] += 1
                    if len(word) == 1:
                        counts["standalone"] += 1
                    elif i == 0:
                        counts["initial"] += 1
                    elif i == len(word) - 1:
                        counts["final"] += 1
                    else:
                        counts["medial"] += 1
        results[ch] = counts
    return results


def step0_digram_analysis(eva_data):
    """Analyze qo digram positional distribution."""
    counts = {"initial": 0, "medial": 0, "final": 0, "total": 0}
    q_before_o = 0
    q_not_before_o = 0

    for page in eva_data["pages"]:
        for word in page["words"]:
            for i in range(len(word) - 1):
                if word[i] == "q" and word[i + 1] == "o":
                    q_before_o += 1
                    counts["total"] += 1
                    if i == 0:
                        counts["initial"] += 1
                    elif i + 1 == len(word) - 1:
                        counts["final"] += 1
                    else:
                        counts["medial"] += 1
            # Count q not followed by o
            for i, c in enumerate(word):
                if c == "q" and (i + 1 >= len(word) or word[i + 1] != "o"):
                    q_not_before_o += 1

    counts["q_before_o"] = q_before_o
    counts["q_not_before_o"] = q_not_before_o
    return counts


def step0_strip_and_match(eva_data, mapping, direction, divergent_chars,
                          italian_set, italian_gloss, hebrew_set,
                          hebrew_by_cons):
    """Strip each unknown char from initial position and re-decode.

    Tests the prefix hypothesis: if removing the initial char produces
    more lexical matches, it's likely a prefix.
    """
    results = {}

    for strip_char in sorted(divergent_chars):
        # Also test "qo" as a unit when stripping q
        strip_patterns = [strip_char]
        if strip_char == "q":
            strip_patterns.append("qo")

        for pattern in strip_patterns:
            matches = []
            tested = 0
            hit_words = []

            for page in eva_data["pages"]:
                section = page.get("section", "?")
                section_name = SECTION_NAMES.get(section, section)

                for word in page["words"]:
                    # Check if word starts with the pattern
                    if not word.startswith(pattern):
                        continue
                    remainder = word[len(pattern):]
                    if len(remainder) < 3:
                        continue

                    # Check remainder has no unknown chars
                    has_unknown = any(
                        c in divergent_chars for c in remainder)
                    if has_unknown:
                        continue

                    tested += 1

                    # Decode remainder
                    chars = (list(reversed(remainder)) if direction == "rtl"
                             else list(remainder))
                    hebrew_parts = []
                    for ch in chars:
                        if ch in mapping:
                            hebrew_parts.append(mapping[ch])
                        else:
                            break
                    else:
                        heb_str = "".join(hebrew_parts)
                        ita_str = _hebrew_to_italian(heb_str)

                        # Search in Italian lexicon
                        ita_match = None
                        ita_norm = _normalize_italian_for_lookup(ita_str)
                        if ita_norm in italian_set:
                            ita_match = ("italian", ita_norm,
                                         italian_gloss.get(ita_norm, ""))
                        # Fuzzy match (d<=1) for Italian, only 4+ chars
                        if not ita_match and len(ita_norm) >= 4:
                            for form in italian_set:
                                if abs(len(form) - len(ita_norm)) > 1:
                                    continue
                                if Levenshtein.distance(
                                        ita_norm, form, score_cutoff=1) <= 1:
                                    ita_match = (
                                        "italian~1", form,
                                        italian_gloss.get(form, ""))
                                    break

                        # Search in Hebrew lexicon
                        heb_match = None
                        if heb_str in hebrew_set:
                            entries = hebrew_by_cons.get(heb_str, [])
                            gloss = (entries[0].get("gloss", "")
                                     if entries else "")
                            heb_match = ("hebrew", heb_str, gloss)

                        if ita_match or heb_match:
                            best = ita_match or heb_match
                            hit_words.append({
                                "eva_word": word,
                                "stripped": remainder,
                                "hebrew": heb_str,
                                "italian": ita_str,
                                "match_lang": best[0],
                                "match_word": best[1],
                                "gloss": best[2],
                                "section": section_name,
                            })

            hit_rate = len(hit_words) / tested * 100 if tested else 0
            results[pattern] = {
                "tested": tested,
                "hits": len(hit_words),
                "hit_rate": round(hit_rate, 1),
                "top_matches": sorted(
                    hit_words,
                    key=lambda x: x["match_lang"].startswith("italian")
                    and "~" not in x["match_lang"],
                    reverse=True,
                )[:20],
            }

    return results


def step0_identify_prefix(positional, strip_results, eva_data,
                          divergent_chars):
    """Identify which Hebrew prefix each unknown char might be.

    Returns dict: {char: {prefix, hebrew_letter, confidence, reason}}.
    """
    total_words = eva_data["total_words"]
    identifications = {}

    for ch in sorted(divergent_chars):
        pos = positional[ch]
        initial_pct = (pos["initial"] / pos["total"] * 100
                       if pos["total"] else 0)
        freq_in_corpus = pos["total"] / total_words

        # Check if strongly initial
        is_initial_dominant = initial_pct > 60

        # Get strip results
        strip_key = ch
        strip_data = strip_results.get(strip_key, {})
        hit_rate = strip_data.get("hit_rate", 0)

        # For q, also check qo stripping
        qo_hit_rate = 0
        if ch == "q" and "qo" in strip_results:
            qo_hit_rate = strip_results["qo"].get("hit_rate", 0)

        # Try to identify the prefix
        best_prefix = None
        confidence = "low"
        reason = ""

        if is_initial_dominant and hit_rate > 10:
            # Compare frequency with known Hebrew prefix frequencies
            best_match_score = 0
            for prefix_heb, (name, form, meaning) in HEBREW_PREFIXES.items():
                expected_freq = HEBREW_PREFIX_FREQS.get(prefix_heb, 0)
                freq_ratio = (min(freq_in_corpus, expected_freq)
                              / max(freq_in_corpus, expected_freq)
                              if expected_freq > 0 else 0)
                score = freq_ratio * hit_rate
                if score > best_match_score:
                    best_match_score = score
                    best_prefix = prefix_heb

            if hit_rate > 20:
                confidence = "high"
            elif hit_rate > 10:
                confidence = "medium"
            reason = (f"initial={initial_pct:.0f}%, "
                      f"strip_hit_rate={hit_rate:.1f}%, "
                      f"corpus_freq={freq_in_corpus:.3f}")

        elif ch == "q" and qo_hit_rate > 10:
            best_prefix = "qo_unit"
            confidence = "medium" if qo_hit_rate > 20 else "low"
            reason = (f"qo digram hit_rate={qo_hit_rate:.1f}%, "
                      f"initial={initial_pct:.0f}%")

        if best_prefix and best_prefix in HEBREW_PREFIXES:
            name, form, meaning = HEBREW_PREFIXES[best_prefix]
            identifications[ch] = {
                "prefix": best_prefix,
                "prefix_name": name,
                "prefix_form": form,
                "meaning": meaning,
                "confidence": confidence,
                "reason": reason,
                "initial_pct": round(initial_pct, 1),
                "strip_hit_rate": hit_rate,
            }
        else:
            identifications[ch] = {
                "prefix": None,
                "confidence": "none",
                "reason": (f"initial={initial_pct:.0f}%, "
                           f"hit_rate={hit_rate:.1f}% — "
                           "not consistent with prefix hypothesis"),
                "initial_pct": round(initial_pct, 1),
                "strip_hit_rate": hit_rate,
            }

    return identifications


# =====================================================================
# Step 1: Extract constraint words
# =====================================================================

def step1_constraint_words(eva_data, mapping, direction, divergent_chars,
                           prefix_resolved):
    """Extract words with exactly 1 or 2 unknown chars.

    prefix_resolved: set of chars resolved as prefixes in Step 0.
    For these, if the char appears at position 0, it's not counted
    as unknown (it's a known prefix).

    Returns lists of Type A, B, C words.
    """
    type_a = []  # 1 unknown, 4+ known
    type_b = []  # 1 unknown, 3 known
    type_c = []  # 2 unknowns

    word_counter = Counter()
    for page in eva_data["pages"]:
        for word in page["words"]:
            word_counter[word] += 1

    seen = set()
    for page in eva_data["pages"]:
        section = page.get("section", "?")
        section_name = SECTION_NAMES.get(section, section)

        for word in page["words"]:
            if word in seen:
                continue
            seen.add(word)

            chars_list, unknowns = _decode_chars(word, mapping, direction)
            n_known = sum(1 for c in chars_list if c is not None)
            n_unknown = len(unknowns)

            # Check if any unknown is a resolved prefix at position 0
            effective_unknowns = []
            for pos, eva_ch in unknowns:
                if (eva_ch in prefix_resolved and pos == 0
                        and direction == "rtl"):
                    # In RTL, position 0 in reversed word = last char in EVA
                    # Actually for prefix: position 0 in decoded = first
                    # decoded char = last EVA char (RTL)
                    continue
                if (eva_ch in prefix_resolved
                        and pos == 0 and direction == "ltr"):
                    continue
                effective_unknowns.append((pos, eva_ch))

            n_eff_unknown = len(effective_unknowns)
            n_eff_known = len(chars_list) - n_eff_unknown

            entry = {
                "eva_word": word,
                "word_len": len(word),
                "decoded_partial": chars_list,
                "unknowns": effective_unknowns,
                "n_known": n_eff_known,
                "n_unknown": n_eff_unknown,
                "count": word_counter[word],
                "section": section_name,
                "unknown_chars": [ch for _, ch in effective_unknowns],
            }

            if n_eff_unknown == 1 and n_eff_known >= 4:
                type_a.append(entry)
            elif n_eff_unknown == 1 and n_eff_known == 3:
                type_b.append(entry)
            elif n_eff_unknown == 2:
                type_c.append(entry)

    # Sort by count descending
    type_a.sort(key=lambda x: -x["count"])
    type_b.sort(key=lambda x: -x["count"])
    type_c.sort(key=lambda x: -x["count"])

    return type_a, type_b, type_c


# =====================================================================
# Step 2 & 3: Brute force + global consistency
# =====================================================================

def step2_brute_force(constraint_words, mapping, direction,
                      italian_set, italian_gloss,
                      hebrew_set, hebrew_by_cons):
    """For each unknown char, try all 22 Hebrew letters.

    Returns per-char results: {char: {letter: {hits, miss, examples}}}.
    """
    # Group constraint words by unknown char
    by_char = defaultdict(list)
    for entry in constraint_words:
        for _, eva_ch in entry["unknowns"]:
            by_char[eva_ch].append(entry)

    results = {}
    for eva_ch in sorted(by_char):
        words = by_char[eva_ch]
        letter_scores = {}

        for heb_letter in HEBREW_LETTERS:
            # Skip letters already in the mapping values
            used_letters = set(mapping.values())
            if heb_letter in used_letters:
                # Still test but note it's already mapped
                pass

            hits = []
            misses = 0
            ita_phoneme = HEBREW_TO_ITALIAN.get(heb_letter, "?")

            for entry in words:
                decoded = list(entry["decoded_partial"])
                # Fill in the unknown position
                for pos, ch in entry["unknowns"]:
                    if ch == eva_ch:
                        decoded[pos] = heb_letter

                if None in decoded:
                    continue  # Another unknown still present

                heb_str = "".join(decoded)
                ita_str = _hebrew_to_italian(heb_str)

                # Check Italian lexicon
                found = False
                match_info = None
                ita_norm = _normalize_italian_for_lookup(ita_str)

                if ita_norm in italian_set:
                    found = True
                    match_info = {
                        "lang": "italian",
                        "match": ita_norm,
                        "distance": 0,
                        "gloss": italian_gloss.get(ita_norm, ""),
                    }

                # Check Hebrew lexicon
                if not found and heb_str in hebrew_set:
                    found = True
                    entries = hebrew_by_cons.get(heb_str, [])
                    gloss = (entries[0].get("gloss", "")
                             if entries else "")
                    match_info = {
                        "lang": "hebrew",
                        "match": heb_str,
                        "distance": 0,
                        "gloss": gloss,
                    }

                # Fuzzy Italian (d<=1, only for 4+ char results)
                if not found and len(ita_norm) >= 4:
                    for form in italian_set:
                        if abs(len(form) - len(ita_norm)) > 1:
                            continue
                        d = Levenshtein.distance(
                            ita_norm, form, score_cutoff=1)
                        if d <= 1:
                            found = True
                            match_info = {
                                "lang": "italian",
                                "match": form,
                                "distance": 1,
                                "gloss": italian_gloss.get(form, ""),
                            }
                            break

                if found:
                    hits.append({
                        "eva_word": entry["eva_word"],
                        "hebrew": heb_str,
                        "italian": ita_str,
                        "count": entry["count"],
                        "section": entry["section"],
                        **match_info,
                    })
                else:
                    misses += 1

            # Score: prioritize exact matches on long words
            hit_5plus = sum(1 for h in hits if len(h["eva_word"]) >= 5
                            and h["distance"] == 0)
            hit_exact = sum(1 for h in hits if h["distance"] == 0)

            letter_scores[heb_letter] = {
                "hebrew_letter": heb_letter,
                "hebrew_name": CONSONANT_NAMES.get(heb_letter, "?"),
                "italian_phoneme": ita_phoneme,
                "total_hits": len(hits),
                "exact_hits": hit_exact,
                "hits_5plus": hit_5plus,
                "misses": misses,
                "total_tested": len(hits) + misses,
                "hit_rate": (round(len(hits) / (len(hits) + misses) * 100, 1)
                             if (len(hits) + misses) > 0 else 0),
                "already_mapped": heb_letter in used_letters,
                "examples": sorted(hits, key=lambda x: -x["count"])[:10],
            }

        # Sort by total_hits descending
        ranked = sorted(letter_scores.values(),
                        key=lambda x: (-x["hits_5plus"],
                                       -x["exact_hits"],
                                       -x["total_hits"]))
        results[eva_ch] = {
            "n_words_tested": len(words),
            "ranked": ranked[:10],  # Top 10 candidates
            "all_scores": letter_scores,
        }

    return results


def step3_consistency(brute_results, mapping):
    """Pick the best letter for each unknown with consistency checks.

    Returns: {char: {best, confidence, margin, sections}}.
    """
    used_letters = set(mapping.values())
    decisions = {}

    for eva_ch, data in brute_results.items():
        ranked = data["ranked"]
        if not ranked:
            decisions[eva_ch] = {
                "best": None,
                "confidence": "none",
                "reason": "no candidates",
            }
            continue

        # Filter: prefer unmapped letters
        unmapped_ranked = [r for r in ranked if not r["already_mapped"]]
        working = unmapped_ranked if unmapped_ranked else ranked

        top = working[0]
        second = working[1] if len(working) > 1 else None

        # Margin analysis
        margin = 0
        if second and second["total_hits"] > 0:
            margin = top["total_hits"] / second["total_hits"]
        elif top["total_hits"] > 0:
            margin = float("inf")

        # Section diversity
        sections = set()
        for ex in top["examples"]:
            sections.add(ex["section"])

        # Confidence
        if (top["total_hits"] >= 5 and margin >= 2.0
                and len(sections) >= 2 and top["hits_5plus"] >= 2):
            confidence = "high"
        elif (top["total_hits"] >= 3 and margin >= 1.5
              and top["hits_5plus"] >= 1):
            confidence = "medium"
        elif top["total_hits"] >= 1:
            confidence = "low"
        else:
            confidence = "none"

        # Check: does it conflict with already-used letters?
        conflicts_with = None
        if top["hebrew_letter"] in used_letters:
            for k, v in mapping.items():
                if v == top["hebrew_letter"]:
                    conflicts_with = k
                    break

        decisions[eva_ch] = {
            "best": top["hebrew_letter"],
            "best_name": top["hebrew_name"],
            "best_italian": top["italian_phoneme"],
            "total_hits": top["total_hits"],
            "exact_hits": top["exact_hits"],
            "hits_5plus": top["hits_5plus"],
            "hit_rate": top["hit_rate"],
            "margin": round(margin, 2) if margin != float("inf") else "inf",
            "second_best": (second["hebrew_letter"]
                            if second else None),
            "second_hits": (second["total_hits"]
                            if second else 0),
            "confidence": confidence,
            "n_sections": len(sections),
            "sections": sorted(sections),
            "conflicts_with": conflicts_with,
            "top_examples": top["examples"][:5],
        }

    return decisions


# =====================================================================
# Step 4: Cross-validation with anchor words
# =====================================================================

def step4_validate(eva_data, mapping, direction, decisions,
                   prefix_identifications):
    """Apply new char values and re-run anchor word search.

    Compares 16-char vs 16+new char anchor matches.
    """
    from .anchor_words import (
        ANCHOR_DICT, build_word_index, normalize_for_match,
        hebrew_to_consonantal, search_anchor_word,
    )

    # Build extended mapping
    extended = dict(mapping)
    new_chars = {}
    for eva_ch, decision in decisions.items():
        if decision["confidence"] in ("high", "medium") and decision["best"]:
            extended[eva_ch] = decision["best"]
            new_chars[eva_ch] = decision["best"]

    if not new_chars:
        return {"delta_matches": 0, "message": "No new chars to validate"}

    # Decode with both mappings
    divergent_base = {"f", "i", "q"}
    divergent_ext = divergent_base - set(new_chars.keys())

    def decode_all(m, div_set):
        pages = {}
        for page in eva_data["pages"]:
            folio = page["folio"]
            section = page.get("section", "?")
            words_eva = page["words"]
            words_dec = []
            words_heb = []
            for w in words_eva:
                chars = (list(reversed(w)) if direction == "rtl"
                         else list(w))
                heb_parts = []
                ita_parts = []
                for ch in chars:
                    if ch in m:
                        heb = m[ch]
                        heb_parts.append(heb)
                        ita_parts.append(
                            HEBREW_TO_ITALIAN.get(heb, "?"))
                    elif ch in div_set:
                        heb_parts.append(ch.upper())
                        ita_parts.append(ch.upper())
                    else:
                        heb_parts.append("?")
                        ita_parts.append("?")
                words_dec.append("".join(ita_parts))
                words_heb.append("".join(heb_parts))
            pages[folio] = {
                "section": SECTION_NAMES.get(section, section),
                "words_eva": words_eva,
                "words_decoded": words_dec,
                "words_hebrew": words_heb,
            }
        return pages

    pages_base = decode_all(mapping, divergent_base)
    pages_ext = decode_all(extended, divergent_ext)

    # Count anchor matches for both
    def count_anchors(pages_data):
        ita_index = build_word_index(pages_data, "words_decoded")
        heb_index = build_word_index(pages_data, "words_hebrew")
        total_matched = 0
        total_occ = 0
        exact_matches = []

        for cat_id, cat_data in ANCHOR_DICT.items():
            for word, gloss in cat_data.get("italian", []):
                norm = normalize_for_match(word)
                if len(norm) < 3:
                    continue
                eff_d = 0 if len(norm) < 4 else 1
                matches = search_anchor_word(norm, ita_index, eff_d)
                if matches:
                    total_matched += 1
                    for m in matches:
                        total_occ += m["total_count"]
                        if m["distance"] == 0:
                            exact_matches.append({
                                "anchor": word,
                                "gloss": gloss,
                                "language": "italian",
                                "decoded": m["decoded_word"],
                                "count": m["total_count"],
                                "category": cat_id,
                            })

            for word, gloss in cat_data.get("hebrew", []):
                cons = hebrew_to_consonantal(word)
                if len(cons) < 3:
                    continue
                eff_d = 0 if len(cons) < 4 else 1
                matches = search_anchor_word(cons, heb_index, eff_d)
                if matches:
                    total_matched += 1
                    for m in matches:
                        total_occ += m["total_count"]
                        if m["distance"] == 0:
                            exact_matches.append({
                                "anchor": word,
                                "gloss": gloss,
                                "language": "hebrew",
                                "decoded": m["decoded_word"],
                                "count": m["total_count"],
                                "category": cat_id,
                            })

            for word, gloss in cat_data.get("latin", []):
                norm = normalize_for_match(word)
                if len(norm) < 3:
                    continue
                eff_d = 0 if len(norm) < 4 else 1
                matches = search_anchor_word(norm, ita_index, eff_d)
                if matches:
                    total_matched += 1
                    for m in matches:
                        total_occ += m["total_count"]
                        if m["distance"] == 0:
                            exact_matches.append({
                                "anchor": word,
                                "gloss": gloss,
                                "language": "latin",
                                "decoded": m["decoded_word"],
                                "count": m["total_count"],
                                "category": cat_id,
                            })

        return total_matched, total_occ, exact_matches

    base_matched, base_occ, base_exact = count_anchors(pages_base)
    ext_matched, ext_occ, ext_exact = count_anchors(pages_ext)

    # Find new exact matches
    base_exact_set = {(e["anchor"], e["language"]) for e in base_exact}
    new_exact = [e for e in ext_exact
                 if (e["anchor"], e["language"]) not in base_exact_set]

    return {
        "base_matches": base_matched,
        "base_occurrences": base_occ,
        "extended_matches": ext_matched,
        "extended_occurrences": ext_occ,
        "delta_matches": ext_matched - base_matched,
        "delta_occurrences": ext_occ - base_occ,
        "new_exact_matches": new_exact,
        "new_chars_tested": new_chars,
    }


# =====================================================================
# Step 5: Cascade — resolve Type C words
# =====================================================================

def step5_cascade(type_c_words, mapping, decisions, direction,
                  italian_set, italian_gloss,
                  hebrew_set, hebrew_by_cons):
    """With newly resolved chars, re-process Type C words."""
    extended = dict(mapping)
    for eva_ch, decision in decisions.items():
        if decision["confidence"] in ("high", "medium") and decision["best"]:
            extended[eva_ch] = decision["best"]

    # Re-classify Type C words with extended mapping
    newly_resolved = []
    for entry in type_c_words:
        decoded = list(entry["decoded_partial"])
        remaining_unknowns = []

        for pos, eva_ch in entry["unknowns"]:
            if eva_ch in extended and eva_ch not in mapping:
                decoded[pos] = extended[eva_ch]
            elif decoded[pos] is None:
                remaining_unknowns.append((pos, eva_ch))

        if len(remaining_unknowns) == 0:
            # Fully resolved
            heb_str = "".join(decoded)
            ita_str = _hebrew_to_italian(heb_str)
            ita_norm = _normalize_italian_for_lookup(ita_str)
            match_info = None

            if ita_norm in italian_set:
                match_info = {
                    "lang": "italian", "match": ita_norm,
                    "gloss": italian_gloss.get(ita_norm, ""),
                }
            elif heb_str in hebrew_set:
                entries = hebrew_by_cons.get(heb_str, [])
                gloss = entries[0].get("gloss", "") if entries else ""
                match_info = {
                    "lang": "hebrew", "match": heb_str, "gloss": gloss,
                }

            newly_resolved.append({
                "eva_word": entry["eva_word"],
                "hebrew": heb_str,
                "italian": ita_str,
                "count": entry["count"],
                "match": match_info,
            })

    return sorted(newly_resolved, key=lambda x: -x["count"])


# =====================================================================
# Main entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Resolve the 3 unknown characters via prefix test + brute force."""
    report_path = config.stats_dir / "prefix_resolve_report.json"

    if report_path.exists() and not force:
        click.echo("  Prefix-resolve report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PREFIX RESOLVE — Identify f, i, q")

    # Load data
    print_step("Loading convergent mapping...")
    mapping, divergent, direction = load_convergent_mapping(config)
    divergent_chars = set(divergent.keys())
    click.echo(f"    {len(mapping)} agreed chars, "
               f"direction={direction}")
    click.echo(f"    Divergent: {sorted(divergent_chars)}")

    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words")

    print_step("Loading lexicons...")
    italian_set, italian_gloss, hebrew_set, hebrew_by_cons = \
        _load_lexicons(config)
    click.echo(f"    Italian: {len(italian_set)} forms")
    click.echo(f"    Hebrew: {len(hebrew_set)} forms")

    # =================================================================
    # STEP 0: PREFIX HYPOTHESIS TEST
    # =================================================================
    print_header("STEP 0 — PREFIX HYPOTHESIS TEST")

    print_step("0a: Positional analysis...")
    positional = step0_positional_analysis(
        eva_data, divergent_chars, direction)
    for ch, pos in sorted(positional.items()):
        total = pos["total"]
        if total == 0:
            continue
        click.echo(
            f"    {ch}: initial={pos['initial']/total*100:.1f}%  "
            f"medial={pos['medial']/total*100:.1f}%  "
            f"final={pos['final']/total*100:.1f}%  "
            f"standalone={pos['standalone']/total*100:.1f}%  "
            f"(n={total})")

    print_step("0a: qo digram analysis...")
    qo_stats = step0_digram_analysis(eva_data)
    if qo_stats["total"] > 0:
        click.echo(
            f"    qo digram: initial={qo_stats['initial']/qo_stats['total']*100:.1f}%  "
            f"medial={qo_stats['medial']/qo_stats['total']*100:.1f}%  "
            f"final={qo_stats['final']/qo_stats['total']*100:.1f}%  "
            f"(n={qo_stats['total']})")
        click.echo(
            f"    q→o predictivity: {qo_stats['q_before_o']}"
            f"/({qo_stats['q_before_o']}+{qo_stats['q_not_before_o']})"
            f" = {qo_stats['q_before_o']/(qo_stats['q_before_o']+qo_stats['q_not_before_o'])*100:.1f}%")

    print_step("0b: Strip and re-decode test...")
    strip_results = step0_strip_and_match(
        eva_data, mapping, direction, divergent_chars,
        italian_set, italian_gloss, hebrew_set, hebrew_by_cons)
    for pattern, data in sorted(strip_results.items()):
        click.echo(
            f"    strip '{pattern}': {data['tested']} tested, "
            f"{data['hits']} hits ({data['hit_rate']}%)")
        for m in data["top_matches"][:5]:
            click.echo(
                f"      {m['eva_word']:12s} → -{pattern}→ "
                f"{m['stripped']:8s} → {m['italian']:8s} "
                f"= {m['match_word']} ({m['gloss']}) [{m['match_lang']}]")

    print_step("0c: Prefix identification...")
    prefix_ids = step0_identify_prefix(
        positional, strip_results, eva_data, divergent_chars)
    prefix_resolved = set()
    for ch, info in sorted(prefix_ids.items()):
        if info["prefix"]:
            prefix_resolved.add(ch)
            click.echo(
                f"    {ch} → {info['prefix_name']} "
                f"({info['prefix_form']}, '{info['meaning']}') "
                f"— confidence: {info['confidence']} — "
                f"{info['reason']}")
        else:
            click.echo(
                f"    {ch} → NOT a prefix — {info['reason']}")

    # =================================================================
    # STEP 1: CONSTRAINT WORDS
    # =================================================================
    print_header("STEP 1 — CONSTRAINT WORDS")
    print_step("Extracting constraint words...")
    type_a, type_b, type_c = step1_constraint_words(
        eva_data, mapping, direction, divergent_chars, prefix_resolved)

    # Count by unknown char
    a_by_char = Counter(ch for e in type_a for ch in e["unknown_chars"])
    click.echo(f"    Type A (1 unk, 4+ known): {len(type_a)} words")
    for ch, n in a_by_char.most_common():
        click.echo(f"      with {ch}: {n}")
    click.echo(f"    Type B (1 unk, 3 known):  {len(type_b)} words")
    click.echo(f"    Type C (2 unk):            {len(type_c)} words")

    # =================================================================
    # STEP 2 & 3: BRUTE FORCE + CONSISTENCY
    # =================================================================
    print_header("STEP 2-3 — BRUTE FORCE + CONSISTENCY")

    # Determine which chars still need brute force
    chars_to_brute = set()
    for ch in sorted(divergent_chars):
        if ch not in prefix_resolved:
            chars_to_brute.add(ch)
        else:
            # Even for prefix chars, try brute force on non-initial
            # occurrences if they exist
            pos = positional[ch]
            non_initial = pos["medial"] + pos["final"]
            if non_initial > pos["total"] * 0.2:
                chars_to_brute.add(ch)
                click.echo(
                    f"    {ch}: also testing as letter "
                    f"({non_initial} non-initial occurrences)")

    if chars_to_brute:
        print_step("Brute-forcing unknown chars against lexicons...")
        # Use Type A + Type B words for brute force
        all_constraint = type_a + type_b
        brute_results = step2_brute_force(
            all_constraint, mapping, direction,
            italian_set, italian_gloss, hebrew_set, hebrew_by_cons)

        print_step("Global consistency analysis...")
        decisions = step3_consistency(brute_results, mapping)

        for eva_ch in sorted(decisions):
            d = decisions[eva_ch]
            click.echo(f"\n    === {eva_ch.upper()} ===")
            click.echo(
                f"    Best: {eva_ch} → {d['best']} "
                f"({d.get('best_name','?')}) → Italian '{d.get('best_italian','?')}'")
            click.echo(
                f"    Hits: {d['total_hits']} total, "
                f"{d['exact_hits']} exact, "
                f"{d['hits_5plus']} on 5+ char words")
            click.echo(
                f"    Hit rate: {d['hit_rate']}%")
            click.echo(
                f"    Margin vs 2nd ({d.get('second_best','—')}): "
                f"{d['margin']}x")
            click.echo(
                f"    Sections: {d['n_sections']} — {d['sections']}")
            click.echo(
                f"    Confidence: {d['confidence']}")
            if d.get("conflicts_with"):
                click.echo(
                    f"    WARNING: conflicts with {d['conflicts_with']}")

            # Show examples
            if d.get("top_examples"):
                click.echo(f"    Top examples:")
                for ex in d["top_examples"]:
                    click.echo(
                        f"      {ex['eva_word']:12s} → "
                        f"{ex['hebrew']:8s} → {ex['italian']:8s} "
                        f"= {ex['match']} ({ex['gloss']}) "
                        f"[{ex['lang']}, d={ex['distance']}]")

            # Show brute force top 5
            if eva_ch in brute_results:
                click.echo(f"\n    Top 5 candidates:")
                for i, r in enumerate(
                        brute_results[eva_ch]["ranked"][:5]):
                    flag = " *MAPPED*" if r["already_mapped"] else ""
                    click.echo(
                        f"    {i+1}. {eva_ch}={r['hebrew_letter']} "
                        f"({r['hebrew_name']}) → "
                        f"'{r['italian_phoneme']}' — "
                        f"{r['total_hits']} hits "
                        f"({r['exact_hits']} exact, "
                        f"{r['hits_5plus']} on 5+){flag}")
    else:
        brute_results = {}
        decisions = {}
        click.echo("    All chars resolved as prefixes, skipping brute force")

    # =================================================================
    # STEP 4: CROSS-VALIDATION
    # =================================================================
    print_header("STEP 4 — CROSS-VALIDATION")
    print_step("Validating with anchor words...")
    validation = step4_validate(
        eva_data, mapping, direction, decisions, prefix_ids)
    click.echo(
        f"    Base (16 char): {validation['base_matches']} matches, "
        f"{validation['base_occurrences']} occurrences")
    click.echo(
        f"    Extended:       {validation['extended_matches']} matches, "
        f"{validation['extended_occurrences']} occurrences")
    click.echo(
        f"    Delta: +{validation['delta_matches']} matches, "
        f"+{validation['delta_occurrences']} occurrences")

    if validation.get("new_exact_matches"):
        click.echo(f"\n    NEW EXACT MATCHES (d=0):")
        for em in validation["new_exact_matches"]:
            click.echo(
                f"      {em['anchor']} ({em['gloss']}, {em['language']}) "
                f"→ {em['decoded']} x{em['count']} [{em['category']}]")

    # =================================================================
    # STEP 5: CASCADE
    # =================================================================
    print_header("STEP 5 — CASCADE")
    print_step("Re-processing Type C words...")
    cascade = step5_cascade(
        type_c, mapping, decisions, direction,
        italian_set, italian_gloss, hebrew_set, hebrew_by_cons)
    matched_cascade = [c for c in cascade if c["match"]]
    click.echo(
        f"    Type C resolved: {len(cascade)}")
    click.echo(
        f"    With lexical match: {len(matched_cascade)}")
    if matched_cascade:
        click.echo(f"\n    Top 20 newly resolved words:")
        for c in matched_cascade[:20]:
            m = c["match"]
            click.echo(
                f"      {c['eva_word']:12s} → {c['hebrew']:8s} → "
                f"{c['italian']:8s} = {m['match']} "
                f"({m['gloss']}) [{m['lang']}] x{c['count']}")

    # =================================================================
    # FINAL MAPPING TABLE
    # =================================================================
    print_header("FINAL MAPPING — 19 CHARACTERS")
    click.echo(
        f"  {'EVA':5s} {'Hebrew':8s} {'Name':10s} "
        f"{'Italian':8s} {'Confidence':12s}")
    click.echo("  " + "-" * 50)

    final_mapping = dict(mapping)
    for eva_ch in sorted(divergent_chars):
        if eva_ch in decisions and decisions[eva_ch]["best"]:
            final_mapping[eva_ch] = decisions[eva_ch]["best"]

    for eva_ch in sorted(final_mapping):
        heb = final_mapping[eva_ch]
        name = CONSONANT_NAMES.get(heb, "?")
        ita = HEBREW_TO_ITALIAN.get(heb, "?")
        if eva_ch in mapping:
            conf = "convergent"
        elif eva_ch in decisions:
            conf = decisions[eva_ch]["confidence"]
            if eva_ch in prefix_ids and prefix_ids[eva_ch]["prefix"]:
                conf += "+prefix"
        else:
            conf = "prefix"
        click.echo(
            f"  {eva_ch:5s} {heb:8s} {name:10s} "
            f"{ita:8s} {conf:12s}")

    # =================================================================
    # VERDICT
    # =================================================================
    print_header("VERDICT")
    high_conf = [ch for ch in divergent_chars
                 if ch in decisions
                 and decisions[ch]["confidence"] == "high"]
    med_conf = [ch for ch in divergent_chars
                if ch in decisions
                and decisions[ch]["confidence"] == "medium"]
    low_conf = [ch for ch in divergent_chars
                if ch in decisions
                and decisions[ch]["confidence"] == "low"]
    unresolved = [ch for ch in divergent_chars
                  if ch not in decisions
                  or decisions[ch]["confidence"] == "none"]

    click.echo(f"  High confidence: {high_conf or '—'}")
    click.echo(f"  Medium confidence: {med_conf or '—'}")
    click.echo(f"  Low confidence: {low_conf or '—'}")
    click.echo(f"  Unresolved: {unresolved or '—'}")

    if validation["delta_matches"] > 0:
        click.echo(f"\n  Anchor validation: POSITIVE "
                   f"(+{validation['delta_matches']} matches)")
    elif validation["delta_matches"] == 0:
        click.echo(f"\n  Anchor validation: NEUTRAL (no change)")
    else:
        click.echo(f"\n  Anchor validation: NEGATIVE "
                   f"({validation['delta_matches']} matches)")

    # =================================================================
    # SAVE REPORT
    # =================================================================
    print_step("Saving report...")
    report = {
        "step0_positional": {
            ch: {k: v for k, v in pos.items()}
            for ch, pos in positional.items()
        },
        "step0_qo_digram": qo_stats,
        "step0_strip_results": {
            pattern: {
                "tested": d["tested"],
                "hits": d["hits"],
                "hit_rate": d["hit_rate"],
                "top_matches": d["top_matches"][:10],
            }
            for pattern, d in strip_results.items()
        },
        "step0_prefix_ids": prefix_ids,
        "step1_counts": {
            "type_a": len(type_a),
            "type_b": len(type_b),
            "type_c": len(type_c),
            "type_a_by_char": dict(a_by_char),
        },
        "step2_brute_force": {
            ch: {
                "n_words": data["n_words_tested"],
                "top5": [
                    {k: v for k, v in r.items() if k != "examples"}
                    for r in data["ranked"][:5]
                ],
            }
            for ch, data in brute_results.items()
        },
        "step3_decisions": decisions,
        "step4_validation": validation,
        "step5_cascade": {
            "total_resolved": len(cascade),
            "with_match": len(matched_cascade),
            "top20": matched_cascade[:20],
        },
        "final_mapping": {
            eva_ch: {
                "hebrew": final_mapping[eva_ch],
                "name": CONSONANT_NAMES.get(final_mapping[eva_ch], "?"),
                "italian": HEBREW_TO_ITALIAN.get(
                    final_mapping[eva_ch], "?"),
                "confidence": ("convergent" if eva_ch in mapping
                               else decisions.get(eva_ch, {}).get(
                                   "confidence", "unknown")),
            }
            for eva_ch in sorted(final_mapping)
        },
        "verdict": {
            "high_confidence": high_conf,
            "medium_confidence": med_conf,
            "low_confidence": low_conf,
            "unresolved": unresolved,
            "delta_anchors": validation["delta_matches"],
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")
