#!/usr/bin/env python3
"""
Analysis: Is EVA 'i' an independent letter or part of composite glyphs?

EVA 'i' appears 99.8% in medial position, almost always between 'a' and 'n'.
This script tests three models:
  A) ii = single char, standalone i = different char
  B) ain/aiin/aiiin = composite units
  C) i is a vowel modifier (nikkud-like) -- strip it and decode remainder
"""

import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

from voynich_toolkit.word_structure import parse_eva_words
from voynich_toolkit.prepare_italian_lexicon import HEBREW_TO_ITALIAN
from voynich_toolkit.prepare_lexicon import CONSONANT_NAMES

# --- Configuration -----------------------------------------------------------

EVA_FILE = Path("eva_data/LSI_ivtff_0d.txt")
ITALIAN_LEX_FILE = Path("output/lexicon/italian_lexicon.json")
HEBREW_LEX_FILE = Path("output/lexicon/lexicon.json")

# 16-char convergent mapping (EVA -> Hebrew ASCII)
CONVERGENT_MAP = {
    "a": "y", "c": "A", "d": "r", "e": "p", "g": "X",
    "h": "E", "k": "t", "l": "m", "m": "g", "n": "d",
    "o": "w", "p": "l", "r": "h", "s": "n", "t": "J",
    "y": "S",
}

# 17-char mapping: convergent + f -> lamed
MAP_17 = dict(CONVERGENT_MAP)
MAP_17["f"] = "l"  # f -> lamed (Hebrew l)

# All 22 Hebrew consonant ASCII codes
ALL_HEBREW = list("AbgdhwzXJyklmnsEpCqrSt")

DIRECTION = "rtl"


def decode_word(eva_word, mapping, direction="rtl"):
    """Decode an EVA word using the given mapping. Returns None if any char unmapped."""
    hebrew_chars = []
    for ch in eva_word:
        if ch not in mapping:
            return None
        hebrew_chars.append(mapping[ch])
    if direction == "rtl":
        hebrew_chars.reverse()
    return "".join(hebrew_chars)


def decode_to_italian(hebrew_str):
    """Convert Hebrew ASCII string to Italian phonemic."""
    return "".join(HEBREW_TO_ITALIAN.get(ch, "?") for ch in hebrew_str)


# --- Load data ---------------------------------------------------------------

def load_data():
    data = parse_eva_words(EVA_FILE)
    all_words = data["words"]
    pages = data["pages"]

    with open(ITALIAN_LEX_FILE) as f:
        it_lex = json.load(f)
    italian_forms = set(it_lex["all_forms"])
    italian_glosses = it_lex["form_to_gloss"]

    with open(HEBREW_LEX_FILE) as f:
        he_lex = json.load(f)
    hebrew_forms = set(he_lex["all_consonantal_forms"])
    hebrew_glosses = he_lex["by_consonants"]

    return all_words, pages, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses


# =============================================================================
# PART 1: Enumerate all i-containing patterns and their frequencies
# =============================================================================

def analyze_i_patterns(all_words):
    print("=" * 78)
    print("PART 1: i-RUN CONTEXT PATTERNS")
    print("=" * 78)

    word_counts = Counter(all_words)
    total_tokens = len(all_words)

    words_with_i = [w for w in all_words if "i" in w]
    unique_with_i = set(words_with_i)
    print(f"\nTotal word tokens: {total_tokens:,}")
    print(f"Tokens containing 'i': {len(words_with_i):,} ({100*len(words_with_i)/total_tokens:.1f}%)")
    print(f"Unique types containing 'i': {len(unique_with_i):,}")

    context_counts = Counter()
    bigram_counts = Counter()
    i_run_lengths = Counter()

    for word in all_words:
        for m in re.finditer(r"i+", word):
            start, end = m.start(), m.end()
            run = m.group()
            run_len = len(run)
            left = word[start - 1] if start > 0 else "^"
            right = word[end] if end < len(word) else "$"
            context_counts[(left, run, right)] += 1
            bigram_counts[(left, run_len, right)] += 1
            i_run_lengths[run_len] += 1

    print(f"\ni-run length distribution:")
    for length in sorted(i_run_lengths):
        print(f"  {'i'*length:6s} (len={length}): {i_run_lengths[length]:>6,} occurrences")

    print(f"\nTop 40 context patterns (left | i-run | right):")
    print(f"  {'Pattern':<20s} {'Count':>8s}  {'%':>6s}")
    print(f"  {'-'*20} {'-'*8}  {'-'*6}")
    total_runs = sum(context_counts.values())
    for (left, run, right), count in context_counts.most_common(40):
        left_s = "^" if left == "^" else left
        right_s = "$" if right == "$" else right
        pct = 100 * count / total_runs
        print(f"  {left_s}|{run}|{right_s:<14s} {count:>8,}  {pct:>5.1f}%")

    lr_counts = Counter()
    for (left, run_len, right), count in bigram_counts.items():
        lr_counts[(left, right)] += count

    print(f"\nContext pairs (left, right) aggregated across all i-run lengths:")
    print(f"  {'L-R Pair':<12s} {'Total':>8s}  {'i':>7s}  {'ii':>7s}  {'iii':>7s}  {'iiii+':>7s}")
    print(f"  {'-'*12} {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for (left, right), total in lr_counts.most_common(25):
        left_s = "^" if left == "^" else left
        right_s = "$" if right == "$" else right
        i1 = bigram_counts.get((left, 1, right), 0)
        i2 = bigram_counts.get((left, 2, right), 0)
        i3 = bigram_counts.get((left, 3, right), 0)
        i4p = sum(bigram_counts.get((left, n, right), 0) for n in range(4, 10))
        print(f"  {left_s}-{right_s:<10s} {total:>8,}  {i1:>7,}  {i2:>7,}  {i3:>7,}  {i4p:>7,}")

    return context_counts, bigram_counts


# =============================================================================
# PART 2: MODEL A -- ii = single char W, standalone i = char V
# =============================================================================

def transform_a(word):
    """Replace ii->W, leftover i->V."""
    result = []
    idx = 0
    while idx < len(word):
        if word[idx] == "i":
            run_start = idx
            while idx < len(word) and word[idx] == "i":
                idx += 1
            run_len = idx - run_start
            result.extend(["W"] * (run_len // 2))
            if run_len % 2 == 1:
                result.append("V")
        else:
            result.append(word[idx])
            idx += 1
    return "".join(result)


def model_a_analysis(all_words, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses):
    print("\n" + "=" * 78)
    print("PART 2: MODEL A -- ii -> W (single char), standalone i -> V")
    print("=" * 78)

    transformed = [transform_a(w) for w in all_words]
    trans_counts = Counter(transformed)

    has_W = sum(1 for w in transformed if "W" in w)
    has_V = sum(1 for w in transformed if "V" in w)
    has_both = sum(1 for w in transformed if "W" in w and "V" in w)
    print(f"\nAfter transformation:")
    print(f"  Tokens with W (=ii): {has_W:,}")
    print(f"  Tokens with V (=i):  {has_V:,}")
    print(f"  Tokens with both:    {has_both:,}")

    orig_lengths = [len(w) for w in all_words]
    trans_lengths = [len(w) for w in transformed]
    avg_orig = sum(orig_lengths) / len(orig_lengths)
    avg_trans = sum(trans_lengths) / len(trans_lengths)
    print(f"\n  Average word length (original EVA chars): {avg_orig:.2f}")
    print(f"  Average word length (after ii->W, i->V):  {avg_trans:.2f}")

    mapped_chars = set(MAP_17.keys()) | {"W", "V"}

    w_only_words = []
    v_only_words = []
    for tw in set(transformed):
        chars = set(tw)
        if not chars.issubset(mapped_chars):
            continue
        if "W" in chars and "V" not in chars:
            w_only_words.append(tw)
        elif "V" in chars and "W" not in chars:
            v_only_words.append(tw)

    w_only_tokens = sum(trans_counts[w] for w in w_only_words)
    v_only_tokens = sum(trans_counts[w] for w in v_only_words)
    print(f"\n  Unique types decodable with W-only: {len(w_only_words):,}")
    print(f"  Unique types decodable with V-only: {len(v_only_words):,}")
    print(f"  Token count for W-only words: {w_only_tokens:,}")
    print(f"  Token count for V-only words: {v_only_tokens:,}")

    # Try all 22 Hebrew letters for W
    print(f"\n--- Trying all 22 Hebrew letters for W (=ii) ---")
    print(f"  Checking against Italian ({len(italian_forms):,}) and Hebrew ({len(hebrew_forms):,}) lexicons...")

    results_w = []
    for heb_ch in ALL_HEBREW:
        test_map = dict(MAP_17)
        test_map["W"] = heb_ch
        hits_it = 0
        hits_he = 0
        matched_words = []
        for tw in w_only_words:
            decoded = decode_word(tw, test_map, DIRECTION)
            if decoded is None:
                continue
            it_form = decode_to_italian(decoded)
            if it_form in italian_forms:
                hits_it += trans_counts[tw]
                matched_words.append((tw, decoded, it_form, trans_counts[tw]))
            if decoded in hebrew_forms:
                hits_he += trans_counts[tw]
        results_w.append((heb_ch, CONSONANT_NAMES.get(heb_ch, "?"), hits_it, hits_he, matched_words))

    results_w.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {'Hebrew':>8s}  {'Name':<10s}  {'IT hits':>8s}  {'HE hits':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for heb_ch, name, hits_it, hits_he, _ in results_w[:10]:
        print(f"  {heb_ch:>8s}  {name:<10s}  {hits_it:>8,}  {hits_he:>8,}")

    if results_w[0][4]:
        top = results_w[0]
        print(f"\n  Top W candidate: {top[0]} ({top[1]}) -- sample Italian matches:")
        sorted_matches = sorted(top[4], key=lambda x: -x[3])[:20]
        for tw, decoded, it_form, cnt in sorted_matches:
            gloss = italian_glosses.get(it_form, "")
            eva_orig = tw.replace("W", "ii").replace("V", "i")
            print(f"    {eva_orig:<18s} -> {decoded:<12s} -> {it_form:<14s} ({cnt:>4}x)  {gloss}")

    # Try all 22 Hebrew letters for V
    print(f"\n--- Trying all 22 Hebrew letters for V (=standalone i) ---")

    results_v = []
    for heb_ch in ALL_HEBREW:
        test_map = dict(MAP_17)
        test_map["V"] = heb_ch
        hits_it = 0
        hits_he = 0
        matched_words = []
        for tw in v_only_words:
            decoded = decode_word(tw, test_map, DIRECTION)
            if decoded is None:
                continue
            it_form = decode_to_italian(decoded)
            if it_form in italian_forms:
                hits_it += trans_counts[tw]
                matched_words.append((tw, decoded, it_form, trans_counts[tw]))
            if decoded in hebrew_forms:
                hits_he += trans_counts[tw]
        results_v.append((heb_ch, CONSONANT_NAMES.get(heb_ch, "?"), hits_it, hits_he, matched_words))

    results_v.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {'Hebrew':>8s}  {'Name':<10s}  {'IT hits':>8s}  {'HE hits':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for heb_ch, name, hits_it, hits_he, _ in results_v[:10]:
        print(f"  {heb_ch:>8s}  {name:<10s}  {hits_it:>8,}  {hits_he:>8,}")

    if results_v[0][4]:
        top = results_v[0]
        print(f"\n  Top V candidate: {top[0]} ({top[1]}) -- sample Italian matches:")
        sorted_matches = sorted(top[4], key=lambda x: -x[3])[:20]
        for tw, decoded, it_form, cnt in sorted_matches:
            gloss = italian_glosses.get(it_form, "")
            eva_orig = tw.replace("W", "ii").replace("V", "i")
            print(f"    {eva_orig:<18s} -> {decoded:<12s} -> {it_form:<14s} ({cnt:>4}x)  {gloss}")

    return results_w, results_v, avg_trans


# =============================================================================
# PART 3: MODEL B -- ain/aiin/aiiin as composite units
# =============================================================================

def model_b_analysis(all_words):
    print("\n" + "=" * 78)
    print("PART 3: MODEL B -- COMPOSITE UNITS (ain->U, aiin->W, aiiin->X)")
    print("=" * 78)

    def transform_b(word):
        w = word
        w = w.replace("aiiin", "X")
        w = w.replace("aiin", "W")
        w = w.replace("ain", "U")
        return w

    transformed = [transform_b(w) for w in all_words]

    has_i = sum(1 for w in transformed if "i" in w)
    no_i = sum(1 for w in transformed if "i" not in w)
    print(f"\nAfter ain/aiin/aiiin -> composite tokens:")
    print(f"  Tokens still containing 'i': {has_i:,} ({100*has_i/len(all_words):.1f}%)")
    print(f"  Tokens without 'i':          {no_i:,} ({100*no_i/len(all_words):.1f}%)")

    u_count = sum(w.count("U") for w in transformed)
    w_count = sum(w.count("W") for w in transformed)
    x_count = sum(w.count("X") for w in transformed)
    print(f"\n  U (=ain)   occurrences: {u_count:,}")
    print(f"  W (=aiin)  occurrences: {w_count:,}")
    print(f"  X (=aiiin) occurrences: {x_count:,}")

    residual_i_words = [w for w in transformed if "i" in w]
    residual_i_types = set(residual_i_words)
    print(f"\n  Residual i: {len(residual_i_words):,} tokens, {len(residual_i_types):,} types")

    residual_context = Counter()
    for word in residual_i_words:
        for m in re.finditer(r"i+", word):
            start, end = m.start(), m.end()
            run = m.group()
            left = word[start - 1] if start > 0 else "^"
            right = word[end] if end < len(word) else "$"
            residual_context[(left, run, right)] += 1

    print(f"\n  Top 20 residual i contexts:")
    for (left, run, right), count in residual_context.most_common(20):
        left_s = "^" if left == "^" else left
        right_s = "$" if right == "$" else right
        print(f"    {left_s}|{run}|{right_s:<14s} {count:>6,}")

    trans_lengths = [len(w) for w in transformed]
    avg_len = sum(trans_lengths) / len(trans_lengths)
    print(f"\n  Average word length after composite substitution: {avg_len:.2f}")

    # Extended: also replace oin/oiin etc.
    print(f"\n--- Extended composite test: also replace oin/oiin/iin/in/ir ---")
    def transform_b2(word):
        w = word
        w = w.replace("aiiin", "1")
        w = w.replace("aiin", "2")
        w = w.replace("oiin", "3")
        w = w.replace("ain", "4")
        w = w.replace("oin", "5")
        w = w.replace("iin", "6")
        w = w.replace("in", "7")
        w = w.replace("ir", "8")
        return w

    transformed2 = [transform_b2(w) for w in all_words]
    has_i2 = sum(1 for w in transformed2 if "i" in w)
    no_i2 = sum(1 for w in transformed2 if "i" not in w)
    print(f"  Tokens still containing 'i': {has_i2:,} ({100*has_i2/len(all_words):.1f}%)")
    print(f"  Tokens without 'i':          {no_i2:,} ({100*no_i2/len(all_words):.1f}%)")

    residual2 = [w for w in transformed2 if "i" in w]
    residual2_ctx = Counter()
    for word in residual2:
        for m in re.finditer(r"i+", word):
            start, end = m.start(), m.end()
            run = m.group()
            left = word[start - 1] if start > 0 else "^"
            right = word[end] if end < len(word) else "$"
            residual2_ctx[(left, run, right)] += 1
    print(f"\n  Top 15 residual i contexts (extended):")
    for (left, run, right), count in residual2_ctx.most_common(15):
        left_s = "^" if left == "^" else left
        right_s = "$" if right == "$" else right
        print(f"    {left_s}|{run}|{right_s:<14s} {count:>6,}")

    return avg_len


# =============================================================================
# PART 4: MODEL C -- i as vowel modifier (strip and decode)
# =============================================================================

def model_c_analysis(all_words, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses):
    print("\n" + "=" * 78)
    print("PART 4: MODEL C -- STRIP 'i' AS VOWEL MODIFIER (NIKKUD-LIKE)")
    print("=" * 78)

    word_counts = Counter(all_words)

    words_with_i = {w for w in word_counts if "i" in w}
    words_no_i = {w for w in word_counts if "i" not in w}

    tokens_with_i = sum(word_counts[w] for w in words_with_i)
    tokens_no_i = sum(word_counts[w] for w in words_no_i)

    print(f"\n  Words containing 'i': {len(words_with_i):,} types, {tokens_with_i:,} tokens")
    print(f"  Words without 'i':    {len(words_no_i):,} types, {tokens_no_i:,} tokens")

    # -- Baseline: decode words WITHOUT 'i' using MAP_17 --
    print(f"\n--- Baseline: words WITHOUT 'i', decoded with 17-char map ---")
    baseline_hits_it = 0
    baseline_hits_he = 0
    baseline_decodable = 0

    for w in words_no_i:
        cnt = word_counts[w]
        decoded = decode_word(w, MAP_17, DIRECTION)
        if decoded is None:
            continue
        baseline_decodable += cnt
        it_form = decode_to_italian(decoded)
        if it_form in italian_forms:
            baseline_hits_it += cnt
        if decoded in hebrew_forms:
            baseline_hits_he += cnt

    baseline_total = tokens_no_i
    print(f"  Decodable tokens: {baseline_decodable:,} / {baseline_total:,}")
    if baseline_decodable > 0:
        print(f"  Italian lexicon hits: {baseline_hits_it:,} ({100*baseline_hits_it/baseline_decodable:.1f}%)")
        print(f"  Hebrew lexicon hits:  {baseline_hits_he:,} ({100*baseline_hits_he/baseline_decodable:.1f}%)")

    # -- Model C: strip 'i', decode remainder with MAP_17 --
    print(f"\n--- Model C: strip 'i' from words, decode remainder with 17-char map ---")
    c_hits_it = 0
    c_hits_he = 0
    c_decodable = 0
    c_total = 0
    c_matches_it = []
    c_matches_he = []

    for w in words_with_i:
        cnt = word_counts[w]
        stripped = w.replace("i", "")
        if not stripped:
            continue
        c_total += cnt
        decoded = decode_word(stripped, MAP_17, DIRECTION)
        if decoded is None:
            continue
        c_decodable += cnt
        it_form = decode_to_italian(decoded)
        if it_form in italian_forms:
            c_hits_it += cnt
            gloss = italian_glosses.get(it_form, "")
            c_matches_it.append((w, stripped, decoded, it_form, cnt, gloss))
        if decoded in hebrew_forms:
            c_hits_he += cnt
            entries = hebrew_glosses.get(decoded, [])
            he_gloss = entries[0]["gloss"][:60] if entries else ""
            c_matches_he.append((w, stripped, decoded, cnt, he_gloss))

    print(f"  Words with 'i' (stripped): {c_total:,} tokens")
    print(f"  Decodable after stripping: {c_decodable:,} ({100*c_decodable/c_total:.1f}%)")
    if c_decodable > 0:
        print(f"  Italian lexicon hits: {c_hits_it:,} ({100*c_hits_it/c_decodable:.1f}%)")
        print(f"  Hebrew lexicon hits:  {c_hits_he:,} ({100*c_hits_he/c_decodable:.1f}%)")

    print(f"\n  Comparison:")
    if baseline_decodable > 0 and c_decodable > 0:
        b_it_rate = 100 * baseline_hits_it / baseline_decodable
        c_it_rate = 100 * c_hits_it / c_decodable
        b_he_rate = 100 * baseline_hits_he / baseline_decodable
        c_he_rate = 100 * c_hits_he / c_decodable
        print(f"    Italian hit rate -- baseline (no i): {b_it_rate:.1f}%  vs  stripped: {c_it_rate:.1f}%")
        print(f"    Hebrew hit rate  -- baseline (no i): {b_he_rate:.1f}%  vs  stripped: {c_he_rate:.1f}%")

    c_matches_it.sort(key=lambda x: -x[4])
    print(f"\n  Top 30 Italian matches (after stripping 'i'):")
    print(f"    {'EVA orig':<18s}  {'stripped':<14s}  {'Hebrew':<10s}  {'Italian':<14s}  {'Count':>5s}  Gloss")
    print(f"    {'-'*18}  {'-'*14}  {'-'*10}  {'-'*14}  {'-'*5}  {'-'*30}")
    for orig, stripped, decoded, it_form, cnt, gloss in c_matches_it[:30]:
        print(f"    {orig:<18s}  {stripped:<14s}  {decoded:<10s}  {it_form:<14s}  {cnt:>5}  {gloss[:40]}")

    c_matches_he.sort(key=lambda x: -x[3])
    print(f"\n  Top 20 Hebrew matches (after stripping 'i'):")
    print(f"    {'EVA orig':<18s}  {'stripped':<14s}  {'Hebrew':<10s}  {'Count':>5s}  Gloss")
    print(f"    {'-'*18}  {'-'*14}  {'-'*10}  {'-'*5}  {'-'*40}")
    for orig, stripped, decoded, cnt, gloss in c_matches_he[:20]:
        print(f"    {orig:<18s}  {stripped:<14s}  {decoded:<10s}  {cnt:>5}  {gloss[:50]}")

    # Word length after stripping
    stripped_lengths = []
    for w in all_words:
        s = w.replace("i", "")
        if s:
            stripped_lengths.append(len(s))
    avg_stripped = sum(stripped_lengths) / len(stripped_lengths) if stripped_lengths else 0
    avg_orig = sum(len(w) for w in all_words) / len(all_words)
    print(f"\n  Average word length (original):    {avg_orig:.2f}")
    print(f"  Average word length (i stripped):   {avg_stripped:.2f}")

    return c_hits_it, c_decodable, avg_stripped


# =============================================================================
# PART 5: WORD LENGTH COMPARISON
# =============================================================================

def word_length_comparison(all_words, avg_model_a, avg_model_b, avg_model_c):
    print("\n" + "=" * 78)
    print("PART 5: WORD LENGTH COMPARISON ACROSS MODELS")
    print("=" * 78)

    avg_orig = sum(len(w) for w in all_words) / len(all_words)

    refs = {
        "Hebrew":  4.5,
        "Italian": 5.1,
        "Latin":   5.5,
        "Arabic":  4.9,
    }

    models = {
        "Original EVA":          avg_orig,
        "Model A (ii->W, i->V)": avg_model_a,
        "Model B (ain->U, etc)": avg_model_b,
        "Model C (strip i)":     avg_model_c,
    }

    print(f"\n  {'Model/Language':<28s}  {'Avg Length':>10s}  {'vs Hebrew':>10s}  {'vs Italian':>10s}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*10}")
    for name, avg in models.items():
        d_heb = abs(avg - refs["Hebrew"])
        d_it = abs(avg - refs["Italian"])
        print(f"  {name:<28s}  {avg:>10.2f}  {d_heb:>10.2f}  {d_it:>10.2f}")
    print()
    for name, avg in refs.items():
        print(f"  {name + ' (reference)':<28s}  {avg:>10.2f}")


# =============================================================================
# PART 6: POSITIONAL ANALYSIS OF 'i'
# =============================================================================

def positional_analysis(all_words):
    print("\n" + "=" * 78)
    print("PART 6: POSITIONAL DISTRIBUTION OF 'i' IN WORDS")
    print("=" * 78)

    pos_counts = Counter()
    total_i = 0

    for word in all_words:
        for idx, ch in enumerate(word):
            if ch == "i":
                total_i += 1
                if idx == 0:
                    pos_counts["initial"] += 1
                elif idx == len(word) - 1:
                    pos_counts["final"] += 1
                else:
                    pos_counts["medial"] += 1

    print(f"\n  Total 'i' occurrences: {total_i:,}")
    for pos in ["initial", "medial", "final"]:
        cnt = pos_counts.get(pos, 0)
        print(f"  {pos:<10s}: {cnt:>7,} ({100*cnt/total_i:.1f}%)")

    print(f"\n  Comparison: positional distribution of frequent letters")
    print(f"    {'Char':<6s}  {'Total':>7s}  {'Initial':>10s}  {'Medial':>10s}  {'Final':>10s}")
    print(f"    {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for target_ch in "aeinoydschkl":
        t = 0
        pos = Counter()
        for word in all_words:
            for idx, ch in enumerate(word):
                if ch == target_ch:
                    t += 1
                    if idx == 0:
                        pos["initial"] += 1
                    elif idx == len(word) - 1:
                        pos["final"] += 1
                    else:
                        pos["medial"] += 1
        if t == 0:
            continue
        init = pos.get("initial", 0)
        med = pos.get("medial", 0)
        fin = pos.get("final", 0)
        print(f"    {target_ch:<6s}  {t:>7,}  {init:>5} {100*init/t:>4.1f}%  {med:>5} {100*med/t:>4.1f}%  {fin:>5} {100*fin/t:>4.1f}%")


# =============================================================================
# PART 7: INFORMATION-THEORETIC CHECK
# =============================================================================

def entropy_analysis(all_words):
    print("\n" + "=" * 78)
    print("PART 7: CHARACTER ENTROPY ANALYSIS")
    print("=" * 78)

    def char_entropy(word_list):
        counts = Counter()
        total = 0
        for w in word_list:
            for ch in w:
                counts[ch] += 1
                total += 1
        if total == 0:
            return 0, 0, counts
        H = 0
        for ch, c in counts.items():
            p = c / total
            H -= p * math.log2(p)
        return H, total, counts

    H_orig, total_orig, counts_orig = char_entropy(all_words)

    trans_a = [transform_a(w) for w in all_words]
    H_a, total_a, counts_a = char_entropy(trans_a)

    trans_c = [w.replace("i", "") for w in all_words]
    trans_c = [w for w in trans_c if w]
    H_c, total_c, counts_c = char_entropy(trans_c)

    print(f"\n  {'Model':<30s}  {'Chars':>7s}  {'Alphabet':>9s}  {'Entropy':>8s}  {'Max H':>6s}  {'Efficiency':>11s}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*11}")

    for name, H, total, counts in [
        ("Original EVA", H_orig, total_orig, counts_orig),
        ("Model A (ii->W, i->V)", H_a, total_a, counts_a),
        ("Model C (i stripped)", H_c, total_c, counts_c),
    ]:
        n = len(counts)
        max_H = math.log2(n) if n > 1 else 0
        eff = H / max_H if max_H > 0 else 0
        print(f"  {name:<30s}  {total:>7,}  {n:>9}  {H:>8.3f}  {max_H:>6.2f}  {100*eff:>10.1f}%")

    print(f"\n  Character frequency (original EVA, top 20):")
    total = sum(counts_orig.values())
    top_count = counts_orig.most_common(1)[0][1]
    for ch, c in counts_orig.most_common(20):
        bar = "#" * int(60 * c / top_count)
        print(f"    {ch}: {c:>7,} ({100*c/total:>5.1f}%) {bar}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("ANALYSIS: Is EVA 'i' an independent letter or part of composite glyphs?")
    print("=" * 78)

    all_words, pages, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses = load_data()
    print(f"Loaded {len(all_words):,} word tokens, {len(set(all_words)):,} types")
    print(f"Italian lexicon: {len(italian_forms):,} forms")
    print(f"Hebrew lexicon:  {len(hebrew_forms):,} forms")

    # Part 1: Pattern analysis
    context_counts, bigram_counts = analyze_i_patterns(all_words)

    # Part 6: Positional analysis (logically goes with Part 1)
    positional_analysis(all_words)

    # Part 2: Model A
    results_w, results_v, avg_model_a = model_a_analysis(
        all_words, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses
    )

    # Part 3: Model B
    avg_model_b = model_b_analysis(all_words)

    # Part 4: Model C
    c_hits_it, c_decodable, avg_model_c = model_c_analysis(
        all_words, italian_forms, italian_glosses, hebrew_forms, hebrew_glosses
    )

    # Part 5: Word length comparison
    word_length_comparison(all_words, avg_model_a, avg_model_b, avg_model_c)

    # Part 7: Entropy
    entropy_analysis(all_words)

    # -- Summary --
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    top_w = results_w[0] if results_w else None
    top_v = results_v[0] if results_v else None
    print(f"\n  MODEL A:")
    if top_w:
        print(f"    Best candidate for W (=ii):  {top_w[0]} ({top_w[1]}) with {top_w[2]:,} Italian hits")
    if top_v:
        print(f"    Best candidate for V (=i):   {top_v[0]} ({top_v[1]}) with {top_v[2]:,} Italian hits")
    same = top_w and top_v and top_w[0] == top_v[0]
    diff = top_w and top_v and top_w[0] != top_v[0]
    if same:
        print(f"    --> SAME letter for both: ii and i may not be distinct")
    if diff:
        print(f"    --> DIFFERENT letters: supports ii != i hypothesis")

    print(f"\n  MODEL B:")
    print(f"    Composite units absorb most 'i' occurrences")
    print(f"    Remaining 'i' are in non-ain contexts (e.g., oir, eir, etc.)")

    print(f"\n  MODEL C:")
    if c_decodable > 0:
        c_it_rate = 100 * c_hits_it / c_decodable
        print(f"    Italian hit rate after stripping i: {c_it_rate:.1f}%")
    avg_orig = sum(len(w) for w in all_words) / len(all_words)
    print(f"    Word length after stripping: {avg_model_c:.2f}")

    print(f"\n  WORD LENGTHS:")
    print(f"    Original:  {avg_orig:.2f}")
    print(f"    Model A:   {avg_model_a:.2f}")
    print(f"    Model B:   {avg_model_b:.2f}")
    print(f"    Model C:   {avg_model_c:.2f}")
    print(f"    Hebrew:    4.50 (reference)")
    print(f"    Italian:   5.10 (reference)")

    closest = min(
        [("Model A", avg_model_a), ("Model B", avg_model_b), ("Model C", avg_model_c)],
        key=lambda x: abs(x[1] - 4.5)
    )
    print(f"\n    Closest to Hebrew: {closest[0]} ({closest[1]:.2f})")

    print()


if __name__ == "__main__":
    main()
