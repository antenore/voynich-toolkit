#!/usr/bin/env python3
"""
Analysis: Extend convergent mapping with f->l (lamed) and q-prefix stripping.

This standalone script:
1. Loads the 16-char convergent mapping
2. Extends it with f -> l (Hebrew lamed, Italian 'l')
3. Treats q in initial position as a prefix (stripped, not decoded)
4. Decodes the entire corpus with the 17-char mapping + q-prefix
5. Computes new statistics and compares to 16-char baseline
6. Runs anchor word validation (16 vs 17 chars)
7. Shows top 30 most frequent fully-decoded words
8. Saves decoded text sample per section
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add the project source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voynich_toolkit.config import ToolkitConfig
from voynich_toolkit.full_decode import load_convergent_mapping, SECTION_NAMES
from voynich_toolkit.prepare_italian_lexicon import HEBREW_TO_ITALIAN
from voynich_toolkit.prepare_lexicon import CONSONANT_NAMES
from voynich_toolkit.word_structure import parse_eva_words
from voynich_toolkit.anchor_words import (
    ANCHOR_DICT,
    build_word_index,
    normalize_for_match,
    hebrew_to_consonantal,
    search_anchor_word,
)


# =====================================================================
# Configuration
# =====================================================================

ROOT = Path(__file__).parent
CONFIG = ToolkitConfig(
    eva_data_dir=ROOT / "eva_data",
    output_dir=ROOT / "output",
)
EVA_FILE = CONFIG.eva_data_dir / "LSI_ivtff_0d.txt"
LEXICON_PATH = ROOT / "output" / "lexicon" / "italian_lexicon.json"
OUTPUT_PATH = ROOT / "output" / "stats" / "analysis_f_lamed_sample.txt"


def hr(char="=", width=70):
    print(char * width)


def decode_word_extended(eva_word, mapping, divergent_set, direction,
                         strip_q_prefix=False):
    q_stripped = False
    word = eva_word

    if strip_q_prefix and word.startswith("q") and len(word) > 1:
        word = word[1:]
        q_stripped = True

    chars = list(reversed(word)) if direction == "rtl" else list(word)
    hebrew_parts = []
    italian_parts = []
    n_unknown = 0

    for ch in chars:
        if ch in mapping:
            heb = mapping[ch]
            hebrew_parts.append(heb)
            italian_parts.append(HEBREW_TO_ITALIAN.get(heb, "?"))
        elif ch in divergent_set:
            placeholder = ch.upper()
            hebrew_parts.append(placeholder)
            italian_parts.append(placeholder)
            n_unknown += 1
        else:
            hebrew_parts.append("?")
            italian_parts.append("?")
            n_unknown += 1

    return "".join(italian_parts), "".join(hebrew_parts), n_unknown, q_stripped


def decode_corpus(eva_data, mapping, divergent_set, direction,
                  strip_q_prefix=False):
    pages = {}
    stats = {
        "total_words": 0,
        "fully_decoded": 0,
        "with_unknowns": 0,
        "q_stripped": 0,
        "unknown_char_counter": Counter(),
    }

    for page in eva_data["pages"]:
        folio = page["folio"]
        section = page.get("section", "?")
        section_name = SECTION_NAMES.get(section, section)
        words_eva = page["words"]

        words_decoded = []
        words_hebrew = []
        n_fully = 0
        n_with_unk = 0

        for w in words_eva:
            ita, heb, n_unk, q_s = decode_word_extended(
                w, mapping, divergent_set, direction, strip_q_prefix)
            words_decoded.append(ita)
            words_hebrew.append(heb)
            if q_s:
                stats["q_stripped"] += 1
            if n_unk == 0:
                n_fully += 1
            else:
                n_with_unk += 1
                actual_word = w[1:] if (strip_q_prefix and w.startswith("q")
                                        and len(w) > 1) else w
                for ch in (reversed(actual_word) if direction == "rtl"
                           else actual_word):
                    if ch in divergent_set:
                        stats["unknown_char_counter"][ch.upper()] += 1

        pages[folio] = {
            "section": section_name,
            "section_code": section,
            "words_eva": words_eva,
            "words_decoded": words_decoded,
            "words_hebrew": words_hebrew,
            "words_with_unknowns": n_with_unk,
            "words_fully_decoded": n_fully,
        }

        stats["total_words"] += len(words_eva)
        stats["fully_decoded"] += n_fully
        stats["with_unknowns"] += n_with_unk

    return pages, stats


def run_anchor_search(pages_data, label):
    word_index = build_word_index(pages_data, "words_decoded")

    results = {}
    total_matches = 0
    exact_matches = []

    for category, cat_data in ANCHOR_DICT.items():
        # Process Italian words
        for word, meaning in cat_data.get("italian", []):
            lang = "IT"
            target = normalize_for_match(word)

            if not target or len(target) < 3:
                continue

            max_d = 0 if len(target) <= 3 else 1
            matches = search_anchor_word(target, word_index, max_dist=max_d)

            if matches:
                total_matches += len(matches)
                total_count = sum(m["total_count"] for m in matches)
                for m in matches:
                    if m["distance"] == 0:
                        exact_matches.append({
                            "word": word,
                            "lang": lang,
                            "meaning": meaning,
                            "category": category,
                            "decoded": m["decoded_word"],
                            "count": m["total_count"],
                        })

                results[f"{word}({lang})"] = {
                    "target": target,
                    "n_matches": len(matches),
                    "total_occ": total_count,
                    "best_dist": matches[0]["distance"],
                    "best_word": matches[0]["decoded_word"],
                }

        # Process Hebrew words
        for word, meaning in cat_data.get("hebrew", []):
            lang = "HE"
            target = hebrew_to_consonantal(word)

            if not target or len(target) < 3:
                continue

            max_d = 0 if len(target) <= 3 else 1
            matches = search_anchor_word(target, word_index, max_dist=max_d)

            if matches:
                total_matches += len(matches)
                total_count = sum(m["total_count"] for m in matches)
                for m in matches:
                    if m["distance"] == 0:
                        exact_matches.append({
                            "word": word,
                            "lang": lang,
                            "meaning": meaning,
                            "category": category,
                            "decoded": m["decoded_word"],
                            "count": m["total_count"],
                        })

                results[f"{word}({lang})"] = {
                    "target": target,
                    "n_matches": len(matches),
                    "total_occ": total_count,
                    "best_dist": matches[0]["distance"],
                    "best_word": matches[0]["decoded_word"],
                }

        # Process Latin words if present
        for word, meaning in cat_data.get("latin", []):
            lang = "LA"
            target = normalize_for_match(word)

            if not target or len(target) < 3:
                continue

            max_d = 0 if len(target) <= 3 else 1
            matches = search_anchor_word(target, word_index, max_dist=max_d)

            if matches:
                total_matches += len(matches)
                total_count = sum(m["total_count"] for m in matches)
                for m in matches:
                    if m["distance"] == 0:
                        exact_matches.append({
                            "word": word,
                            "lang": lang,
                            "meaning": meaning,
                            "category": category,
                            "decoded": m["decoded_word"],
                            "count": m["total_count"],
                        })

                results[f"{word}({lang})"] = {
                    "target": target,
                    "n_matches": len(matches),
                    "total_occ": total_count,
                    "best_dist": matches[0]["distance"],
                    "best_word": matches[0]["decoded_word"],
                }

    return {
        "total_anchor_matches": total_matches,
        "total_anchors_with_hits": len(results),
        "exact_matches": exact_matches,
        "details": results,
    }


def main():
    hr()
    print("  ANALYSIS: f -> lamed (l) extension + q-prefix stripping")
    hr()

    # 1. Load convergent mapping
    print("\n[1] Loading convergent 16-char mapping...")
    mapping_16, divergent, direction = load_convergent_mapping(CONFIG)
    print(f"    Agreed chars: {len(mapping_16)}")
    print(f"    Divergent: {sorted(divergent.keys())}")
    print(f"    Direction: {direction}")

    # 2. Build extended 17-char mapping (f -> l = lamed)
    print("\n[2] Extending mapping: f -> l (lamed, Italian 'l')...")
    mapping_17 = dict(mapping_16)
    mapping_17["f"] = "l"
    heb_name = CONSONANT_NAMES.get("l", "?")
    ita_val = HEBREW_TO_ITALIAN.get("l", "?")
    print(f"    f -> l ({heb_name}) -> {ita_val}")
    print("    Note: EVA 'p' also maps to lamed (f and p are now synonyms)")
    print("    Remaining divergent: i, q")

    div_16 = set(divergent.keys())
    div_17 = set(divergent.keys()) - {"f"}
    div_17_qstrip = div_17

    # 3. Parse EVA text
    print("\n[3] Parsing EVA text...")
    eva_data = parse_eva_words(EVA_FILE)
    n_pages = len(eva_data["pages"])
    n_words = eva_data["total_words"]
    print(f"    {n_pages} pages, {n_words} words")

    # 4. Decode with all three configurations
    print("\n[4] Decoding corpus (three configurations)...")

    pages_16, stats_16 = decode_corpus(
        eva_data, mapping_16, div_16, direction, strip_q_prefix=False)
    pct_16 = stats_16["fully_decoded"] / stats_16["total_words"] * 100

    pages_17, stats_17 = decode_corpus(
        eva_data, mapping_17, div_17, direction, strip_q_prefix=False)
    pct_17 = stats_17["fully_decoded"] / stats_17["total_words"] * 100

    pages_17q, stats_17q = decode_corpus(
        eva_data, mapping_17, div_17_qstrip, direction, strip_q_prefix=True)
    pct_17q = stats_17q["fully_decoded"] / stats_17q["total_words"] * 100

    # 5. Statistics comparison
    print("\n" + "=" * 70)
    print("  DECODE STATISTICS COMPARISON")
    print("=" * 70)

    total = stats_16["total_words"]
    print(f"\n  Total words: {total}")
    hdr = f"  {'Configuration':<40s} {'Fully decoded':>15s} {'Pct':>8s}"
    print(f"\n{hdr}")
    print(f"  {'-'*40} {'-'*15} {'-'*8}")
    fd16 = stats_16["fully_decoded"]
    fd17 = stats_17["fully_decoded"]
    fd17q = stats_17q["fully_decoded"]
    print(f"  {'16-char (baseline)':<40s} {fd16:>15,d} {pct_16:>7.1f}%")
    print(f"  {'17-char (f->lamed)':<40s} {fd17:>15,d} {pct_17:>7.1f}%")
    print(f"  {'17-char + q-prefix strip':<40s} {fd17q:>15,d} {pct_17q:>7.1f}%")

    f_gain = fd17 - fd16
    print(f"\n  Words gaining f->l decoding: +{f_gain}")

    q_gain = fd17q - fd17
    print(f"  Additional words from q-prefix strip: +{q_gain}")
    print(f"  Words with q-prefix stripped: {stats_17q['q_stripped']}")

    print(f"\n  Remaining unknown chars (17-char + q-prefix):")
    for ch, cnt in stats_17q["unknown_char_counter"].most_common():
        print(f"    {ch}: {cnt}")

    print(f"\n  Section breakdown (17-char + q-prefix):")
    section_stats = defaultdict(lambda: {"total": 0, "fully": 0})
    for pdata in pages_17q.values():
        sec = pdata["section"]
        section_stats[sec]["total"] += len(pdata["words_eva"])
        section_stats[sec]["fully"] += pdata["words_fully_decoded"]

    print(f"  {'Section':<18s} {'Total':>8s} {'Decoded':>8s} {'Pct':>8s}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
    for sec in sorted(section_stats,
                      key=lambda s: -section_stats[s]["total"]):
        s = section_stats[sec]
        p = (s["fully"] / s["total"] * 100) if s["total"] else 0
        print(f"  {sec:<18s} {s['total']:>8,d} {s['fully']:>8,d} {p:>7.1f}%")

    # 6. Anchor word validation
    print("\n" + "=" * 70)
    print("  ANCHOR WORD VALIDATION")
    print("=" * 70)

    print("\n  Running anchor search (16-char baseline)...")
    anchor_16 = run_anchor_search(pages_16, "16-char")

    print("  Running anchor search (17-char + q-prefix)...")
    anchor_17q = run_anchor_search(pages_17q, "17+q")

    print(f"\n  {'Metric':<40s} {'16-char':>10s} {'17+q':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    awh16 = anchor_16["total_anchors_with_hits"]
    awh17q = anchor_17q["total_anchors_with_hits"]
    tam16 = anchor_16["total_anchor_matches"]
    tam17q = anchor_17q["total_anchor_matches"]
    ex16 = len(anchor_16["exact_matches"])
    ex17q = len(anchor_17q["exact_matches"])
    print(f"  {'Anchors with hits':<40s} {awh16:>10d} {awh17q:>10d}")
    print(f"  {'Total match entries':<40s} {tam16:>10d} {tam17q:>10d}")
    print(f"  {'Exact matches (d=0)':<40s} {ex16:>10d} {ex17q:>10d}")

    print(f"\n  Exact matches (d=0) in 17-char + q-prefix:")
    if anchor_17q["exact_matches"]:
        for em in sorted(anchor_17q["exact_matches"],
                         key=lambda x: -x["count"]):
            w = em["word"]
            la = em["lang"]
            d = em["decoded"]
            c = em["count"]
            cat = em["category"]
            mn = em["meaning"]
            print(f"    {w:15s} ({la}) = {d:12s} x{c:4d}  [{cat}]  ({mn})")
    else:
        print("    (none)")

    exact_16_set = {(e["word"], e["lang"]) for e in anchor_16["exact_matches"]}
    exact_17q_set = {(e["word"], e["lang"]) for e in anchor_17q["exact_matches"]}
    new_exact = exact_17q_set - exact_16_set

    print(f"\n  NEW exact matches (in 17+q but not 16-char): {len(new_exact)}")
    if new_exact:
        for em in anchor_17q["exact_matches"]:
            if (em["word"], em["lang"]) in new_exact:
                w = em["word"]
                la = em["lang"]
                d = em["decoded"]
                c = em["count"]
                cat = em["category"]
                print(f"    ** {w:15s} ({la}) = {d:12s} x{c:4d}  [{cat}]")

    # 7. Top 30 fully-decoded words
    print("\n" + "=" * 70)
    print("  TOP 30 FULLY-DECODED WORDS (17-char + q-prefix)")
    print("=" * 70)

    with open(LEXICON_PATH) as lf:
        lex_data = json.load(lf)
    form_to_gloss = lex_data["form_to_gloss"]

    word_freq = Counter()
    word_eva = {}
    word_hebrew = {}

    for folio, pdata in pages_17q.items():
        for eva_w, dec_w, heb_w in zip(
                pdata["words_eva"], pdata["words_decoded"],
                pdata["words_hebrew"]):
            if dec_w == dec_w.lower() and "?" not in dec_w:
                word_freq[dec_w] += 1
                if dec_w not in word_eva:
                    word_eva[dec_w] = eva_w
                    word_hebrew[dec_w] = heb_w

    print(f"\n  Unique fully-decoded word types: {len(word_freq)}")
    print(f"\n  {'#':<4s} {'Italian':>12s} {'Hebrew':>10s} "
          f"{'EVA':>12s} {'Count':>7s}  Gloss")
    print(f"  {'-'*4} {'-'*12} {'-'*10} {'-'*12} {'-'*7}  {'-'*25}")

    for i, (dec_w, count) in enumerate(word_freq.most_common(30), 1):
        ew = word_eva.get(dec_w, "?")
        hw = word_hebrew.get(dec_w, "?")
        gloss = form_to_gloss.get(dec_w, "")
        print(f"  {i:<4d} {dec_w:>12s} {hw:>10s} {ew:>12s} {count:>7,d}  {gloss}")

    # 8. Save decoded text sample per section
    print("\n" + "=" * 70)
    print("  DECODED TEXT SAMPLE "
          "(first 50 fully-decoded 5+ char words per section)")
    print("=" * 70)

    section_samples = defaultdict(list)

    for folio, pdata in pages_17q.items():
        sec = pdata["section"]
        if len(section_samples[sec]) >= 50:
            continue
        for eva_w, dec_w, heb_w in zip(
                pdata["words_eva"], pdata["words_decoded"],
                pdata["words_hebrew"]):
            if len(section_samples[sec]) >= 50:
                break
            if (len(dec_w) >= 5 and dec_w == dec_w.lower()
                    and "?" not in dec_w):
                gloss = form_to_gloss.get(dec_w, "")
                section_samples[sec].append({
                    "eva": eva_w,
                    "italian": dec_w,
                    "hebrew": heb_w,
                    "gloss": gloss,
                    "folio": folio,
                })

    output_lines = []
    for sec in sorted(section_samples):
        n = len(section_samples[sec])
        print(f"\n  --- {sec} ({n} words) ---")
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"  {sec}")
        output_lines.append(f"{'='*60}")

        for s in section_samples[sec]:
            fo = s["folio"]
            ev = s["eva"]
            it = s["italian"]
            he = s["hebrew"]
            gl = s["gloss"]
            line = f"  {fo:10s}  {ev:12s} -> {it:12s} ({he:10s})  {gl}"
            print(line)
            output_lines.append(line)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as of:
        of.write("Analysis: f -> lamed (l) extension + q-prefix stripping\n")
        of.write(f"17-char mapping + q-prefix: "
                 f"{fd17q}/{total} "
                 f"({pct_17q:.1f}%) fully decoded\n")
        of.write("\n".join(output_lines))
        of.write("\n")

    print(f"\n  Sample saved to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline (16-char):     {fd16:>6,d} / {total:,d} = {pct_16:.1f}%")
    print(f"  + f->lamed:             {fd17:>6,d} / {total:,d} = {pct_17:.1f}%  (+{f_gain})")
    print(f"  + q-prefix stripping:   {fd17q:>6,d} / {total:,d} = {pct_17q:.1f}%  (+{q_gain} more)")
    net = fd17q - fd16
    print(f"  Net gain:               +{net:,d} words decoded")
    print(f"  Anchor exact (d=0):     {ex16} -> {ex17q} (+{len(new_exact)} new)")
    print()


if __name__ == "__main__":
    main()
