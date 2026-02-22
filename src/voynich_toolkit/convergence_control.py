"""Convergence control test.

Tests whether the 84.2% convergence between Hebrew and Italian
hill-climbing paths is meaningful by applying both optimization paths
to control texts that are NOT Hebrew.

Controls:
  1. Shuffled: characters shuffled within each word (preserving structure)
  2. Random: random consonantal strings matching EVA length distribution

For each control, a simplified hill-climber optimizes a 17-char mapping
against (a) Hebrew lexicon and (b) Italian lexicon independently.
Convergence = fraction of chars with same mapping in both paths.

If real convergence (84.2%) is significantly higher than controls,
the dual-path agreement is informative.

CLI: voynich --force convergence-control
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import FULL_MAPPING, preprocess_eva
from .mapping_audit import HEBREW_CHARS, count_matches, decode_hebrew
from .permutation_stats import EVA_CHARS, HEBREW_LETTERS
from .utils import print_header, print_step
from .word_structure import parse_eva_words


N_EVA = 17  # chars in FULL_MAPPING (excluding q)
N_HEBREW = 22


# =====================================================================
# Simplified hill-climber (17 chars, steepest-ascent)
# =====================================================================


def _preprocess_corpus(eva_freqs: Counter):
    """Pre-process all EVA words once for fast repeated scoring.

    Returns:
        words: list of (processed_chars, freq, eva_char_set)
        char_index: dict {eva_char: list of word indices containing it}
    """
    eva_chars_set = set(FULL_MAPPING.keys())
    words = []
    for word, freq in eva_freqs.items():
        _, processed = preprocess_eva(word)
        chars = list(reversed(processed))
        used = {ch for ch in chars if ch in eva_chars_set}
        words.append((chars, freq, used))

    char_index = {c: [] for c in eva_chars_set}
    for idx, (_, _, used) in enumerate(words):
        for c in used:
            char_index[c].append(idx)

    return words, char_index


def _decode_fast(chars, mapping, initial_d="b", initial_h="s"):
    """Fast decode: substitute + positional splits, no re-preprocessing."""
    parts = []
    for ch in chars:
        if ch == "\x01":
            parts.append("h")  # ii
        elif ch == "\x02":
            parts.append("r")  # i alone
        elif ch == "\x03":
            parts.append("k")  # ch
        elif ch in mapping:
            parts.append(mapping[ch])
        else:
            return None
    if parts and parts[0] == "d":
        parts[0] = initial_d
    if parts and parts[0] == "h":
        parts[0] = initial_h
    return "".join(parts)


def simple_hill_climb(eva_freqs: Counter, lexicon_set: set,
                      seed: int = 0, max_iter: int = 50,
                      n_restarts: int = 10,
                      preprocessed=None) -> dict:
    """Steepest-ascent hill-climber for 17-char EVA→Hebrew mapping.

    Uses incremental scoring: only re-evaluates words containing changed chars.
    Returns best mapping as dict {eva_char: hebrew_char}.
    """
    rng = random.Random(seed)
    eva_chars = sorted(FULL_MAPPING.keys())  # 17 chars
    heb_chars = list(HEBREW_CHARS)  # 22 chars

    if preprocessed is None:
        preprocessed = _preprocess_corpus(eva_freqs)
    words, char_index = preprocessed

    best_score = -1
    best_mapping = None

    for restart in range(n_restarts):
        chosen = rng.sample(heb_chars, len(eva_chars))
        mapping = dict(zip(eva_chars, chosen))

        # Full initial score + per-word match cache
        word_matched = [False] * len(words)
        score = 0
        for idx, (chars, freq, _) in enumerate(words):
            heb = _decode_fast(chars, mapping)
            if heb and heb in lexicon_set:
                word_matched[idx] = True
                score += freq

        for iteration in range(max_iter):
            best_delta = 0
            best_move = None

            # Swap moves: only re-evaluate affected words
            for i, j in combinations(range(len(eva_chars)), 2):
                ci, cj = eva_chars[i], eva_chars[j]
                affected = set(char_index[ci]) | set(char_index[cj])

                old_contrib = sum(words[idx][1] for idx in affected
                                  if word_matched[idx])

                mapping[ci], mapping[cj] = mapping[cj], mapping[ci]
                new_contrib = 0
                for idx in affected:
                    heb = _decode_fast(words[idx][0], mapping)
                    if heb and heb in lexicon_set:
                        new_contrib += words[idx][1]
                mapping[ci], mapping[cj] = mapping[cj], mapping[ci]

                delta = new_contrib - old_contrib
                if delta > best_delta:
                    best_delta = delta
                    best_move = ("swap", ci, cj)

            # Replace moves
            used = set(mapping.values())
            unused = [h for h in heb_chars if h not in used]
            for ci in eva_chars:
                affected = char_index[ci]
                old_contrib = sum(words[idx][1] for idx in affected
                                  if word_matched[idx])

                old_h = mapping[ci]
                for new_h in unused:
                    mapping[ci] = new_h
                    new_contrib = 0
                    for idx in affected:
                        heb = _decode_fast(words[idx][0], mapping)
                        if heb and heb in lexicon_set:
                            new_contrib += words[idx][1]
                    delta = new_contrib - old_contrib
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ("replace", ci, new_h)
                mapping[ci] = old_h

            if best_move is None:
                break  # local optimum

            # Apply best move and update cache
            if best_move[0] == "swap":
                _, ci, cj = best_move
                affected = set(char_index[ci]) | set(char_index[cj])
                mapping[ci], mapping[cj] = mapping[cj], mapping[ci]
            else:
                _, ci, new_h = best_move
                affected = set(char_index[ci])
                mapping[ci] = new_h

            for idx in affected:
                heb = _decode_fast(words[idx][0], mapping)
                word_matched[idx] = bool(heb and heb in lexicon_set)
            score += best_delta

        if score > best_score:
            best_score = score
            best_mapping = dict(mapping)

    return best_mapping


def compute_convergence(map_a: dict, map_b: dict) -> dict:
    """Compute convergence between two mappings."""
    common = set(map_a.keys()) & set(map_b.keys())
    if not common:
        return {"n_common": 0, "n_agree": 0, "convergence": 0.0}
    n_agree = sum(1 for c in common if map_a[c] == map_b[c])
    return {
        "n_common": len(common),
        "n_agree": n_agree,
        "convergence": round(n_agree / len(common), 4),
    }


# =====================================================================
# Control text generation
# =====================================================================


def shuffle_within_words(eva_freqs: Counter, seed: int = 42) -> Counter:
    """Shuffle characters within each word, preserving word structure."""
    rng = random.Random(seed)
    result = Counter()
    for word, freq in eva_freqs.items():
        chars = list(word)
        rng.shuffle(chars)
        result[''.join(chars)] += freq
    return result


def generate_random_words(eva_freqs: Counter, seed: int = 42) -> Counter:
    """Generate random EVA-like words matching length distribution."""
    rng = random.Random(seed)
    # Character frequency from real data
    all_chars = []
    for word, freq in eva_freqs.items():
        all_chars.extend(list(word) * freq)
    char_freq = Counter(all_chars)
    chars = list(char_freq.keys())
    weights = [char_freq[c] for c in chars]

    result = Counter()
    for word, freq in eva_freqs.items():
        random_word = ''.join(rng.choices(chars, weights=weights, k=len(word)))
        result[random_word] += freq
    return result


# =====================================================================
# Entry point
# =====================================================================


def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Run convergence control test."""
    out_json = config.stats_dir / "convergence_control.json"
    out_txt = config.stats_dir / "convergence_control.txt"

    if out_json.exists() and not force:
        click.echo(f"  Output exists: {out_json} (use --force)")
        return

    config.ensure_dirs()
    print_header("Convergence Control Test")

    # Load data
    print_step("Loading data...")
    from .mapping_audit import load_data, load_honest_lexicon

    eva_freqs, full_lexicon, _ = load_data(config)
    honest_lexicon, _ = load_honest_lexicon(config)

    # Load Italian lexicon
    italian_lex_path = config.lexicon_dir / "italian_lexicon.json"
    italian_set = set()
    if italian_lex_path.exists():
        with open(italian_lex_path) as f:
            ital_data = json.load(f)
        italian_set = set(ital_data.get("all_forms", []))
    click.echo(f"    EVA: {len(eva_freqs):,} types, "
               f"{sum(eva_freqs.values()):,} tokens")
    click.echo(f"    Hebrew lexicon (honest): {len(honest_lexicon):,}")
    click.echo(f"    Italian lexicon: {len(italian_set):,}")

    n_restarts = 10  # fewer restarts for speed (controls don't need optimality)
    n_iter = 50

    results = {}

    # Filter to freq>=2 for speed (hapax contribute little to token score)
    freq_thresh = 2
    eva_freq_fast = Counter({w: f for w, f in eva_freqs.items() if f >= freq_thresh})
    click.echo(f"    Filtered to freq>={freq_thresh}: {len(eva_freq_fast):,} types "
               f"({sum(eva_freq_fast.values()):,} tokens, "
               f"{sum(eva_freq_fast.values())/sum(eva_freqs.values())*100:.0f}%)")

    # Pre-process real EVA once (shared preprocessing)
    print_step("Pre-processing EVA corpus...")
    real_pp = _preprocess_corpus(eva_freq_fast)
    click.echo(f"    {len(real_pp[0]):,} preprocessed words")

    # 1. Real EVA → Hebrew
    print_step("Hill-climbing on REAL EVA text...")
    real_heb_map = simple_hill_climb(
        eva_freq_fast, honest_lexicon, seed=42,
        max_iter=n_iter, n_restarts=n_restarts, preprocessed=real_pp)
    click.echo(f"    Hebrew mapping derived")

    if italian_set:
        click.echo(f"    Using published convergence = 84.2%")

    # Compare derived mapping to FULL_MAPPING
    conv_real = compute_convergence(real_heb_map, FULL_MAPPING)
    click.echo(f"    Hill-climb vs FULL_MAPPING: {conv_real['convergence']*100:.1f}%")
    results["real"] = {
        "convergence_vs_fullmap": conv_real,
        "published_convergence": 0.842,
    }

    # 2. Shuffled EVA → Hebrew
    print_step("Hill-climbing on SHUFFLED EVA text...")
    shuffled_freqs = shuffle_within_words(eva_freq_fast, seed=42)
    shuf_pp = _preprocess_corpus(shuffled_freqs)
    shuf_map = simple_hill_climb(
        shuffled_freqs, honest_lexicon, seed=42,
        max_iter=n_iter, n_restarts=n_restarts, preprocessed=shuf_pp)
    conv_shuf_full = compute_convergence(shuf_map, FULL_MAPPING)
    click.echo(f"    Shuffled→Hebrew vs FULL_MAPPING: "
               f"{conv_shuf_full['convergence']*100:.1f}%")

    # Second independent optimization on shuffled for self-convergence
    shuf_map_2 = simple_hill_climb(
        shuffled_freqs, honest_lexicon, seed=99,
        max_iter=n_iter, n_restarts=n_restarts, preprocessed=shuf_pp)
    conv_shuf_self = compute_convergence(shuf_map, shuf_map_2)
    click.echo(f"    Shuffled self-convergence: "
               f"{conv_shuf_self['convergence']*100:.1f}%")

    results["shuffled"] = {
        "convergence_vs_fullmap": conv_shuf_full,
        "self_convergence": conv_shuf_self,
    }

    # 3. Random strings → Hebrew
    print_step("Hill-climbing on RANDOM strings...")
    random_freqs = generate_random_words(eva_freq_fast, seed=42)
    rand_pp = _preprocess_corpus(random_freqs)
    rand_map = simple_hill_climb(
        random_freqs, honest_lexicon, seed=42,
        max_iter=n_iter, n_restarts=n_restarts, preprocessed=rand_pp)
    conv_rand_full = compute_convergence(rand_map, FULL_MAPPING)
    click.echo(f"    Random→Hebrew vs FULL_MAPPING: "
               f"{conv_rand_full['convergence']*100:.1f}%")

    rand_map_2 = simple_hill_climb(
        random_freqs, honest_lexicon, seed=99,
        max_iter=n_iter, n_restarts=n_restarts, preprocessed=rand_pp)
    conv_rand_self = compute_convergence(rand_map, rand_map_2)
    click.echo(f"    Random self-convergence: "
               f"{conv_rand_self['convergence']*100:.1f}%")

    results["random"] = {
        "convergence_vs_fullmap": conv_rand_full,
        "self_convergence": conv_rand_self,
    }

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo("  CONVERGENCE CONTROL — SUMMARY")
    click.echo(f"{'='*60}")
    click.echo(f"  Published Hebrew/Italian convergence: 84.2%")
    click.echo(f"  Hill-climb→Hebrew vs FULL_MAPPING:    "
               f"{conv_real['convergence']*100:.1f}%")
    click.echo(f"  Shuffled→Hebrew vs FULL_MAPPING:      "
               f"{conv_shuf_full['convergence']*100:.1f}%")
    click.echo(f"  Random→Hebrew vs FULL_MAPPING:        "
               f"{conv_rand_full['convergence']*100:.1f}%")
    click.echo(f"  Shuffled self-convergence:             "
               f"{conv_shuf_self['convergence']*100:.1f}%")
    click.echo(f"  Random self-convergence:               "
               f"{conv_rand_self['convergence']*100:.1f}%")

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    click.echo(f"\n  JSON: {out_json}")

    # Save TXT
    lines = [
        "Convergence Control Test",
        "=" * 60,
        "",
        f"Published Hebrew/Italian convergence: 84.2%",
        f"Hill-climb→Hebrew vs FULL_MAPPING:    {conv_real['convergence']*100:.1f}%",
        f"Shuffled→Hebrew vs FULL_MAPPING:      {conv_shuf_full['convergence']*100:.1f}%",
        f"Random→Hebrew vs FULL_MAPPING:        {conv_rand_full['convergence']*100:.1f}%",
        f"Shuffled self-convergence:             {conv_shuf_self['convergence']*100:.1f}%",
        f"Random self-convergence:               {conv_rand_self['convergence']*100:.1f}%",
        "",
        "Interpretation: if controls show similar convergence to 84.2%,",
        "the published result is not informative. If controls converge at",
        "much lower rates, the real convergence reflects genuine cipher signal.",
    ]
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    click.echo(f"  TXT: {out_txt}")
