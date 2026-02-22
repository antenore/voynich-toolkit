"""Scribal error correction via visual confusion pairs.

For each EVA word that does NOT match the Hebrew lexicon after decoding,
generate all single-character visual variants (substituting one confusable
glyph at a time), decode each variant, and check if it matches.

A successful correction recovers a word that was likely mis-copied by a scribe.
Permutation control shuffles the confusion pair labels to measure how many
"corrections" arise by chance.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import ToolkitConfig
from .copyist_errors import VISUAL_CONFUSION_PAIRS
from .full_decode import FULL_MAPPING, preprocess_eva
from .hand1_deep_dive import split_corpus_by_hand
from .mapping_audit import decode_hebrew, load_data, load_honest_lexicon
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# ── Confusion map ─────────────────────────────────────────────


def build_confusion_map() -> dict[str, set[str]]:
    """Convert frozenset pairs to char -> {confusable_chars} lookup."""
    cmap: dict[str, set[str]] = defaultdict(set)
    for pair in VISUAL_CONFUSION_PAIRS:
        chars = list(pair)
        if len(chars) == 2:
            a, b = chars
            cmap[a].add(b)
            cmap[b].add(a)
    return dict(cmap)


# ── Variant generation ────────────────────────────────────────


def generate_visual_variants(
    eva_word: str,
    confusion_map: dict[str, set[str]],
) -> list[tuple[str, int, str, str]]:
    """For each position, try substituting confusable chars.

    Returns list of (variant_word, position, original_char, replacement_char).
    Only generates variants where the char actually changes.
    """
    variants = []
    for pos, ch in enumerate(eva_word):
        if ch not in confusion_map:
            continue
        for repl in confusion_map[ch]:
            variant = eva_word[:pos] + repl + eva_word[pos + 1:]
            variants.append((variant, pos, ch, repl))
    return variants


# ── Core correction engine ────────────────────────────────────


def attempt_corrections(
    unmatched_words: dict[str, int],
    lexicon_set: set,
    form_to_gloss: dict,
    mapping: dict,
    confusion_map: dict[str, set[str]],
) -> list[dict]:
    """Try single-char visual corrections on each unmatched word.

    Returns list of correction records sorted by token frequency (desc).
    """
    corrections = []

    for eva_word, freq in unmatched_words.items():
        variants = generate_visual_variants(eva_word, confusion_map)
        for variant, pos, char_from, char_to in variants:
            heb = decode_hebrew(variant, mapping=mapping)
            if heb is None:
                continue
            if heb not in lexicon_set:
                continue

            gloss = form_to_gloss.get(heb, "")
            corrections.append({
                "eva_original": eva_word,
                "eva_corrected": variant,
                "hebrew": heb,
                "gloss": gloss,
                "freq": freq,
                "position": pos,
                "char_from": char_from,
                "char_to": char_to,
                "has_gloss": bool(gloss),
                "word_len": len(eva_word),
            })

    # Deduplicate: keep best correction per (eva_original, eva_corrected)
    seen: dict[tuple[str, str], dict] = {}
    for c in corrections:
        key = (c["eva_original"], c["eva_corrected"])
        if key not in seen or (c["has_gloss"] and not seen[key]["has_gloss"]):
            seen[key] = c
    corrections = sorted(seen.values(), key=lambda c: -c["freq"])

    return corrections


# ── Per-hand analysis ─────────────────────────────────────────


def analyze_by_hand(
    corrections: list[dict],
    pages: list[dict],
    lexicon_set: set,
    mapping: dict,
) -> dict[str, dict]:
    """Group stats by scribe hand."""
    corpus = split_corpus_by_hand(pages)

    # Build per-hand EVA word frequencies
    hand_freqs: dict[str, Counter] = {}
    for hand, info in corpus.items():
        hand_freqs[hand] = Counter(info["words"])

    # For each hand: count total decoded, unmatched, corrected
    corrected_evas = {c["eva_original"] for c in corrections}
    correction_lookup: dict[str, list[dict]] = defaultdict(list)
    for c in corrections:
        correction_lookup[c["eva_original"]].append(c)

    result = {}
    for hand, freqs in hand_freqs.items():
        n_decoded_types = 0
        n_total_tokens = 0
        n_matched_types = 0
        n_matched_tokens = 0
        n_unmatched_types = 0
        n_unmatched_tokens = 0
        n_corrected_types = 0
        n_corrected_tokens = 0
        confusion_counter: Counter = Counter()

        for word, freq in freqs.items():
            heb = decode_hebrew(word, mapping=mapping)
            if heb is None:
                continue
            n_decoded_types += 1
            n_total_tokens += freq
            if heb in lexicon_set:
                n_matched_types += 1
                n_matched_tokens += freq
            else:
                n_unmatched_types += 1
                n_unmatched_tokens += freq
                if word in corrected_evas:
                    n_corrected_types += 1
                    n_corrected_tokens += freq
                    for c in correction_lookup[word]:
                        confusion_counter[(c["char_from"], c["char_to"])] += freq

        result[hand] = {
            "n_decoded_types": n_decoded_types,
            "n_total_tokens": n_total_tokens,
            "n_matched_types": n_matched_types,
            "n_matched_tokens": n_matched_tokens,
            "n_unmatched_types": n_unmatched_types,
            "n_unmatched_tokens": n_unmatched_tokens,
            "n_corrected_types": n_corrected_types,
            "n_corrected_tokens": n_corrected_tokens,
            "correction_rate_types": (
                n_corrected_types / max(1, n_unmatched_types) * 100
            ),
            "top_confusions": confusion_counter.most_common(10),
        }

    return result


# ── Per-section analysis ──────────────────────────────────────


def analyze_by_section(
    corrections: list[dict],
    pages: list[dict],
    lexicon_set: set,
    mapping: dict,
) -> dict[str, dict]:
    """Group stats by manuscript section."""
    section_freqs: dict[str, Counter] = defaultdict(Counter)
    for p in pages:
        sec = p.get("section", "?")
        for w in p["words"]:
            section_freqs[sec][w] += 1

    corrected_evas = {c["eva_original"] for c in corrections}

    result = {}
    for sec, freqs in sorted(section_freqs.items()):
        n_decoded = 0
        n_total = 0
        n_matched = 0
        n_corrected_types = 0
        n_corrected_tokens = 0

        unmatched_types = 0
        for word, freq in freqs.items():
            heb = decode_hebrew(word, mapping=mapping)
            if heb is None:
                continue
            n_decoded += 1
            n_total += freq
            if heb in lexicon_set:
                n_matched += freq
            else:
                unmatched_types += 1
                if word in corrected_evas:
                    n_corrected_types += 1
                    n_corrected_tokens += freq

        result[sec] = {
            "n_decoded_types": n_decoded,
            "n_total_tokens": n_total,
            "n_matched_tokens": n_matched,
            "n_unmatched_types": unmatched_types,
            "n_corrected_types": n_corrected_types,
            "n_corrected_tokens": n_corrected_tokens,
            "correction_rate_types": (
                n_corrected_types / max(1, unmatched_types) * 100
            ),
        }

    return result


# ── Permutation control ───────────────────────────────────────


def permutation_control(
    unmatched_words: dict[str, int],
    lexicon_set: set,
    form_to_gloss: dict,
    mapping: dict,
    n_perms: int = 200,
    seed: int = 42,
) -> dict:
    """Shuffle confusion pair labels to measure chance recovery rate.

    For each permutation: rebuild confusion map with shuffled char labels,
    then run the same correction procedure.
    """
    rng = random.Random(seed)
    confusion_map = build_confusion_map()

    # Real count
    real_corrections = attempt_corrections(
        unmatched_words, lexicon_set, form_to_gloss, mapping, confusion_map
    )
    real_types = len(real_corrections)
    real_tokens = sum(c["freq"] for c in real_corrections)

    # Collect all chars that appear in any confusion pair
    all_chars = set()
    for pair in VISUAL_CONFUSION_PAIRS:
        all_chars.update(pair)
    all_chars_list = sorted(all_chars)

    perm_type_counts = []
    perm_token_counts = []

    for _ in range(n_perms):
        # Shuffle the char labels to create random confusion pairs
        shuffled = list(all_chars_list)
        rng.shuffle(shuffled)
        char_remap = dict(zip(all_chars_list, shuffled))

        # Build shuffled confusion map
        shuffled_cmap: dict[str, set[str]] = defaultdict(set)
        for pair in VISUAL_CONFUSION_PAIRS:
            chars = list(pair)
            if len(chars) == 2:
                a, b = char_remap[chars[0]], char_remap[chars[1]]
                if a != b:
                    shuffled_cmap[a].add(b)
                    shuffled_cmap[b].add(a)

        perm_corr = attempt_corrections(
            unmatched_words, lexicon_set, form_to_gloss, mapping,
            dict(shuffled_cmap),
        )
        perm_type_counts.append(len(perm_corr))
        perm_token_counts.append(sum(c["freq"] for c in perm_corr))

    perm_type_arr = np.array(perm_type_counts)
    perm_token_arr = np.array(perm_token_counts)

    type_mean = float(perm_type_arr.mean())
    type_std = float(perm_type_arr.std())
    type_z = (real_types - type_mean) / type_std if type_std > 0 else 0.0
    type_p = float(np.mean(perm_type_arr >= real_types))

    token_mean = float(perm_token_arr.mean())
    token_std = float(perm_token_arr.std())
    token_z = (real_tokens - token_mean) / token_std if token_std > 0 else 0.0
    token_p = float(np.mean(perm_token_arr >= real_tokens))

    return {
        "real_types": real_types,
        "real_tokens": real_tokens,
        "perm_type_mean": round(type_mean, 1),
        "perm_type_std": round(type_std, 1),
        "type_z_score": round(type_z, 2),
        "type_p_value": round(type_p, 4),
        "perm_token_mean": round(token_mean, 1),
        "perm_token_std": round(token_std, 1),
        "token_z_score": round(token_z, 2),
        "token_p_value": round(token_p, 4),
        "n_perms": n_perms,
    }


# ── Entry point ───────────────────────────────────────────────


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Full scribal error correction analysis."""
    out_json = config.stats_dir / "scribal_error_correction.json"
    out_txt = config.stats_dir / "scribal_error_correction.txt"

    if not force and out_json.exists():
        print(f"Output exists: {out_json} (use --force to rerun)")
        return

    t0 = time.time()
    print_header("Scribal Error Correction — Visual Confusion Analysis")

    # 1. Load data
    print_step("Loading corpus and lexicon")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    data = parse_eva_words(eva_file)
    pages = data["pages"]

    eva_freqs, full_lex, full_gloss = load_data(config)
    honest_lex, honest_gloss = load_honest_lexicon(config)

    mapping = dict(FULL_MAPPING)
    confusion_map = build_confusion_map()

    print(f"  Corpus: {data['total_words']:,} tokens, "
          f"{data['unique_words']:,} types")
    print(f"  Honest lexicon: {len(honest_lex):,} forms")
    print(f"  Full lexicon:   {len(full_lex):,} forms")
    print(f"  Confusion pairs: {len(VISUAL_CONFUSION_PAIRS)} "
          f"({sum(len(v) for v in confusion_map.values()) // 2} directed)")

    # 2. Decode all words, separate matched vs unmatched (honest lexicon)
    print_step("Decoding and classifying words (honest lexicon)")
    matched_types: dict[str, int] = {}
    unmatched_types: dict[str, int] = {}
    total_decoded = 0
    total_matched_tokens = 0

    for word, freq in eva_freqs.items():
        heb = decode_hebrew(word, mapping=mapping)
        if heb is None:
            continue
        total_decoded += freq
        if heb in honest_lex:
            matched_types[word] = freq
            total_matched_tokens += freq
        else:
            unmatched_types[word] = freq

    n_unmatched_types = len(unmatched_types)
    n_unmatched_tokens = sum(unmatched_types.values())
    print(f"  Decoded: {total_decoded:,} tokens, "
          f"{len(matched_types) + n_unmatched_types:,} types")
    print(f"  Matched: {len(matched_types):,} types, "
          f"{total_matched_tokens:,} tokens "
          f"({total_matched_tokens/total_decoded*100:.1f}%)")
    print(f"  Unmatched: {n_unmatched_types:,} types, "
          f"{n_unmatched_tokens:,} tokens "
          f"({n_unmatched_tokens/total_decoded*100:.1f}%)")

    # 3. Attempt corrections
    print_step("Attempting single-char visual corrections")
    corrections = attempt_corrections(
        unmatched_types, honest_lex, honest_gloss, mapping, confusion_map
    )

    n_corrected_types = len(corrections)
    n_corrected_tokens = sum(c["freq"] for c in corrections)
    n_with_gloss = sum(1 for c in corrections if c["has_gloss"])
    print(f"  Corrected: {n_corrected_types:,} types, "
          f"{n_corrected_tokens:,} tokens")
    print(f"  With gloss: {n_with_gloss:,} types")
    print(f"  Recovery rate (types): "
          f"{n_corrected_types/max(1,n_unmatched_types)*100:.1f}%")
    print(f"  Recovery rate (tokens): "
          f"{n_corrected_tokens/max(1,n_unmatched_tokens)*100:.1f}%")

    # 4. Confusion matrix
    print_step("Building confusion matrix")
    confusion_counter: Counter = Counter()
    for c in corrections:
        confusion_counter[(c["char_from"], c["char_to"])] += c["freq"]

    top_confusions = confusion_counter.most_common(20)
    for (cf, ct), count in top_confusions[:10]:
        print(f"  {cf} → {ct}: {count:,} tokens")

    # 5. Per-hand analysis
    print_step("Analyzing by scribe hand")
    by_hand = analyze_by_hand(corrections, pages, honest_lex, mapping)
    for hand in sorted(by_hand.keys()):
        h = by_hand[hand]
        print(f"  Hand {hand}: {h['n_corrected_types']:,} types corrected, "
              f"{h['n_corrected_tokens']:,} tokens "
              f"({h['correction_rate_types']:.1f}% of unmatched types)")

    # 6. Per-section analysis
    print_step("Analyzing by section")
    by_section = analyze_by_section(corrections, pages, honest_lex, mapping)
    for sec in sorted(by_section.keys()):
        s = by_section[sec]
        print(f"  {sec}: {s['n_corrected_types']:,} types, "
              f"{s['n_corrected_tokens']:,} tokens "
              f"({s['correction_rate_types']:.1f}% of unmatched)")

    # 7. Permutation control
    print_step("Running permutation control (200 perms)")
    perm = permutation_control(
        unmatched_types, honest_lex, honest_gloss, mapping,
        n_perms=200, seed=42,
    )
    print(f"  Real: {perm['real_types']} types, {perm['real_tokens']} tokens")
    print(f"  Perm mean: {perm['perm_type_mean']:.1f} ± "
          f"{perm['perm_type_std']:.1f} types, "
          f"{perm['perm_token_mean']:.1f} ± "
          f"{perm['perm_token_std']:.1f} tokens")
    print(f"  Type z-score: {perm['type_z_score']:.2f} "
          f"(p={perm['type_p_value']:.4f})")
    print(f"  Token z-score: {perm['token_z_score']:.2f} "
          f"(p={perm['token_p_value']:.4f})")

    # 8. Also run with full lexicon for comparison
    print_step("Attempting corrections with full lexicon (comparison)")
    full_corrections = attempt_corrections(
        unmatched_types, full_lex, full_gloss, mapping, confusion_map
    )
    # Re-classify unmatched against full lexicon
    unmatched_full: dict[str, int] = {}
    for word, freq in eva_freqs.items():
        heb = decode_hebrew(word, mapping=mapping)
        if heb is None:
            continue
        if heb not in full_lex:
            unmatched_full[word] = freq

    full_corr_on_full = attempt_corrections(
        unmatched_full, full_lex, full_gloss, mapping, confusion_map
    )
    n_full_types = len(full_corr_on_full)
    n_full_tokens = sum(c["freq"] for c in full_corr_on_full)
    print(f"  Full lexicon unmatched: {len(unmatched_full):,} types")
    print(f"  Full lexicon corrected: {n_full_types:,} types, "
          f"{n_full_tokens:,} tokens")

    elapsed = time.time() - t0

    # 9. Build output
    summary = {
        "n_decoded_tokens": total_decoded,
        "n_matched_types_honest": len(matched_types),
        "n_matched_tokens_honest": total_matched_tokens,
        "n_unmatched_types": n_unmatched_types,
        "n_unmatched_tokens": n_unmatched_tokens,
        "n_corrected_types": n_corrected_types,
        "n_corrected_tokens": n_corrected_tokens,
        "n_with_gloss": n_with_gloss,
        "recovery_rate_types": round(
            n_corrected_types / max(1, n_unmatched_types) * 100, 2
        ),
        "recovery_rate_tokens": round(
            n_corrected_tokens / max(1, n_unmatched_tokens) * 100, 2
        ),
        "full_lexicon_corrected_types": n_full_types,
        "full_lexicon_corrected_tokens": n_full_tokens,
        "permutation": perm,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Serialize by_hand (convert Counter tuples)
    by_hand_out = {}
    for hand, h in by_hand.items():
        hout = dict(h)
        hout["top_confusions"] = [
            {"pair": list(pair), "count": cnt}
            for pair, cnt in h["top_confusions"]
        ]
        by_hand_out[hand] = hout

    output = {
        "summary": summary,
        "by_hand": by_hand_out,
        "by_section": by_section,
        "top_confusions": [
            {"from": cf, "to": ct, "tokens": cnt}
            for (cf, ct), cnt in top_confusions
        ],
        "corrections": corrections[:500],  # cap for JSON size
        "n_total_corrections": len(corrections),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # TXT summary
    lines = [
        "=" * 60,
        "SCRIBAL ERROR CORRECTION — VISUAL CONFUSION ANALYSIS",
        "=" * 60,
        "",
        f"Decoded tokens:     {total_decoded:,}",
        f"Matched (honest):   {total_matched_tokens:,} "
        f"({total_matched_tokens/total_decoded*100:.1f}%)",
        f"Unmatched types:    {n_unmatched_types:,}",
        f"Unmatched tokens:   {n_unmatched_tokens:,}",
        "",
        f"Corrected types:    {n_corrected_types:,} "
        f"({summary['recovery_rate_types']:.1f}%)",
        f"Corrected tokens:   {n_corrected_tokens:,} "
        f"({summary['recovery_rate_tokens']:.1f}%)",
        f"With gloss:         {n_with_gloss:,}",
        "",
        "PERMUTATION CONTROL (200 perms, shuffled confusion labels):",
        f"  Type z-score:     {perm['type_z_score']:.2f} "
        f"(p={perm['type_p_value']:.4f})",
        f"  Token z-score:    {perm['token_z_score']:.2f} "
        f"(p={perm['token_p_value']:.4f})",
        f"  Perm type mean:   {perm['perm_type_mean']:.1f} ± "
        f"{perm['perm_type_std']:.1f}",
        "",
        "TOP CONFUSION PAIRS (by tokens recovered):",
    ]
    for (cf, ct), cnt in top_confusions[:15]:
        lines.append(f"  {cf} → {ct}: {cnt:>6,} tokens")

    lines += ["", "BY HAND:"]
    for hand in sorted(by_hand.keys()):
        h = by_hand[hand]
        lines.append(
            f"  Hand {hand}: {h['n_corrected_types']:>4,} types, "
            f"{h['n_corrected_tokens']:>6,} tokens "
            f"({h['correction_rate_types']:.1f}%)"
        )

    lines += ["", "BY SECTION:"]
    for sec in sorted(by_section.keys()):
        s = by_section[sec]
        lines.append(
            f"  {sec}: {s['n_corrected_types']:>4,} types, "
            f"{s['n_corrected_tokens']:>6,} tokens "
            f"({s['correction_rate_types']:.1f}%)"
        )

    lines += [
        "",
        f"Full lexicon corrected: {n_full_types:,} types, "
        f"{n_full_tokens:,} tokens",
        "",
        "TOP CORRECTIONS (by frequency):",
    ]
    for c in corrections[:30]:
        g = f" [{c['gloss']}]" if c["gloss"] else ""
        lines.append(
            f"  {c['eva_original']:>12s} → {c['eva_corrected']:<12s} "
            f"= {c['hebrew']:<8s}{g} "
            f"(freq={c['freq']}, {c['char_from']}→{c['char_to']} @{c['position']})"
        )

    lines += ["", f"Elapsed: {elapsed:.1f}s", ""]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print_step(f"Output saved: {out_json.name}, {out_txt.name}")
    print(f"  Elapsed: {elapsed:.1f}s")
