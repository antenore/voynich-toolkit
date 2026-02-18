"""Investigation: EVA 'sh' as a digraph for tsade / qof / zayin.

Bigram analysis found EVA 'sh' has strong digraph characteristics:
  - PMI = 3.02 (near confirmed digraph ch = 3.46)
  - Cohesion = 0.431 (above ii = 0.397)
  - 70.5% word-initial in EVA = word-final in Hebrew
  - Low component independence: s=0.391, h_residual=0.343
  - Frequency 2.41% — compatible with tsade (1.5–3.5%) or qof

Currently sh decodes as nun+ayin (nE) — a non-productive Hebrew pair.
This module tests sh → {tsade(C), qof(q), zayin(z)} with:
  1. Lexicon match gain/loss per word
  2. Differential permutation test (real gain vs random gain)
  3. Glossed-word analysis
"""

from __future__ import annotations

import json
import random as _random
import re
from collections import Counter

import numpy as np

from .config import ToolkitConfig
from .full_decode import (
    CH_HEBREW,
    DIRECTION,
    FULL_MAPPING,
    II_HEBREW,
    I_HEBREW,
    INITIAL_D_HEBREW,
    INITIAL_H_HEBREW,
)
from .mapping_audit import load_data
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step

N_PERMUTATIONS = 1000
RNG_SEED = 42

SH_PLACEHOLDER = "\x04"

TARGETS = {
    "C": "tsade",
    "q": "qof",
    "z": "zayin",
}


# ── Preprocessing with sh digraph ────────────────────────────────


def preprocess_with_sh(eva_word: str) -> tuple[str, str]:
    """Preprocess EVA word treating 'sh' as a digraph.

    Order:
      1. ch → \\x03 (existing digraph kaf)
      2. sh → \\x04 (test digraph)
      3. Strip initial q/qo prefix
      4. ii → \\x01, i → \\x02
    """
    w = eva_word

    # 1. ch digraph (existing)
    w = w.replace("ch", "\x03")

    # 2. sh digraph (under test)
    w = w.replace("sh", SH_PLACEHOLDER)

    # 3. q/qo prefix
    prefix = ""
    if w.startswith("qo"):
        prefix = "qo"
        w = w[2:]
    elif w.startswith("q") and len(w) > 1:
        prefix = "q"
        w = w[1:]

    # 4. ii/i composites
    w = re.sub(
        r"i{3,}",
        lambda m: "\x01" * (len(m.group()) // 2)
        + ("\x02" if len(m.group()) % 2 else ""),
        w,
    )
    w = w.replace("ii", "\x01")
    w = w.replace("i", "\x02")

    return prefix, w


def decode_hebrew_sh(
    eva_word: str,
    mapping: dict[str, str],
    sh_hebrew: str,
) -> str | None:
    """Decode EVA word to Hebrew with sh treated as a single letter.

    Returns Hebrew consonantal string, or None if any char is unknown.
    """
    _, processed = preprocess_with_sh(eva_word)
    chars = list(reversed(processed))

    hebrew_parts = []
    for ch in chars:
        if ch == "\x01":
            hebrew_parts.append(II_HEBREW)
        elif ch == "\x02":
            hebrew_parts.append(I_HEBREW)
        elif ch == "\x03":
            hebrew_parts.append(CH_HEBREW)
        elif ch == SH_PLACEHOLDER:
            hebrew_parts.append(sh_hebrew)
        elif ch in mapping:
            hebrew_parts.append(mapping[ch])
        else:
            return None

    # Positional splits (on Hebrew-initial = last after reversal)
    if hebrew_parts and hebrew_parts[0] == "d":
        hebrew_parts[0] = INITIAL_D_HEBREW
    elif hebrew_parts and hebrew_parts[0] == "h":
        hebrew_parts[0] = INITIAL_H_HEBREW

    return "".join(hebrew_parts)


def decode_hebrew_baseline(
    eva_word: str,
    mapping: dict[str, str],
) -> str | None:
    """Decode EVA word with standard pipeline (sh = s + h)."""
    from .full_decode import preprocess_eva

    _, processed = preprocess_eva(eva_word)
    chars = list(reversed(processed))

    hebrew_parts = []
    for ch in chars:
        if ch == "\x01":
            hebrew_parts.append(II_HEBREW)
        elif ch == "\x02":
            hebrew_parts.append(I_HEBREW)
        elif ch == "\x03":
            hebrew_parts.append(CH_HEBREW)
        elif ch in mapping:
            hebrew_parts.append(mapping[ch])
        else:
            return None

    if hebrew_parts and hebrew_parts[0] == "d":
        hebrew_parts[0] = INITIAL_D_HEBREW
    elif hebrew_parts and hebrew_parts[0] == "h":
        hebrew_parts[0] = INITIAL_H_HEBREW

    return "".join(hebrew_parts)


# ── Counting ─────────────────────────────────────────────────────


def count_matches_mode(eva_freqs, lexicon_set, mapping, sh_hebrew=None):
    """Count lexicon matches. If sh_hebrew given, use sh-digraph decode."""
    matched_types = matched_tokens = total_types = total_tokens = 0
    for word, freq in eva_freqs.items():
        if sh_hebrew:
            heb = decode_hebrew_sh(word, mapping, sh_hebrew)
        else:
            heb = decode_hebrew_baseline(word, mapping)
        if heb is None:
            continue
        total_types += 1
        total_tokens += freq
        if heb in lexicon_set:
            matched_types += 1
            matched_tokens += freq
    return matched_types, matched_tokens, total_types, total_tokens


# ── Word-level analysis ──────────────────────────────────────────


def word_level_analysis(eva_freqs, lexicon_set, form_to_gloss, sh_hebrew):
    """Compare baseline (sh=s+h) vs proposed (sh=digraph→sh_hebrew)."""
    mapping = dict(FULL_MAPPING)

    gained = []
    lost = []
    both = []

    # Only analyze words that actually contain "sh"
    sh_words = {w: f for w, f in eva_freqs.items() if "sh" in w}

    for eva_word, freq in sh_words.items():
        heb_base = decode_hebrew_baseline(eva_word, mapping)
        heb_sh = decode_hebrew_sh(eva_word, mapping, sh_hebrew)
        if heb_base is None or heb_sh is None:
            continue

        in_base = heb_base in lexicon_set
        in_sh = heb_sh in lexicon_set

        entry = {
            "eva": eva_word,
            "hebrew_baseline": heb_base,
            "hebrew_sh": heb_sh,
            "freq": freq,
        }

        if in_sh and not in_base:
            entry["gloss"] = form_to_gloss.get(heb_sh, "")
            gained.append(entry)
        elif in_base and not in_sh:
            entry["gloss"] = form_to_gloss.get(heb_base, "")
            lost.append(entry)
        elif in_base and in_sh:
            entry["gloss_base"] = form_to_gloss.get(heb_base, "")
            entry["gloss_sh"] = form_to_gloss.get(heb_sh, "")
            both.append(entry)

    gained.sort(key=lambda x: -x["freq"])
    lost.sort(key=lambda x: -x["freq"])
    both.sort(key=lambda x: -x["freq"])
    return gained, lost, both, len(sh_words)


# ── Differential permutation test ────────────────────────────────


def differential_test(eva_freqs, lexicon_set, sh_hebrew, n_perms=N_PERMUTATIONS):
    """Test if sh→sh_hebrew gains more for real mapping than random."""
    rng = _random.Random(RNG_SEED)
    mapping = dict(FULL_MAPPING)

    # Real mapping: baseline vs proposed
    _, real_base, _, _ = count_matches_mode(eva_freqs, lexicon_set, mapping)
    _, real_sh, _, _ = count_matches_mode(
        eva_freqs, lexicon_set, mapping, sh_hebrew=sh_hebrew
    )
    real_gain = real_sh - real_base

    # Random permutations
    values = list(mapping.values())
    keys = sorted(mapping.keys())
    random_gains = []

    for _ in range(n_perms):
        shuffled = values[:]
        rng.shuffle(shuffled)
        rand_mapping = dict(zip(keys, shuffled))

        _, rand_base, _, _ = count_matches_mode(
            eva_freqs, lexicon_set, rand_mapping
        )
        _, rand_sh, _, _ = count_matches_mode(
            eva_freqs, lexicon_set, rand_mapping, sh_hebrew=sh_hebrew
        )
        random_gains.append(rand_sh - rand_base)

    mean_random = float(np.mean(random_gains))
    std_random = float(np.std(random_gains)) or 1.0
    z_score = (real_gain - mean_random) / std_random
    p_value = sum(1 for g in random_gains if g >= real_gain) / n_perms

    return {
        "real_tokens_base": real_base,
        "real_tokens_sh": real_sh,
        "real_gain": real_gain,
        "random_gain_mean": round(mean_random, 1),
        "random_gain_std": round(std_random, 1),
        "differential": round(real_gain - mean_random, 1),
        "z_score": round(z_score, 2),
        "p_value": p_value,
        "n_perms": n_perms,
    }


# ── Summary formatting ───────────────────────────────────────────


def format_summary(target_results):
    """Format human-readable summary for all targets."""
    lines = []
    lines.append("=" * 60)
    lines.append("  SH DIGRAPH INVESTIGATION")
    lines.append("  EVA 'sh' as single Hebrew letter")
    lines.append("=" * 60)

    for heb_char, info in target_results.items():
        heb_name = CONSONANT_NAMES.get(heb_char, heb_char)
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  sh → {heb_char} ({heb_name})")
        lines.append(f"{'─' * 60}")

        d = info["differential"]
        lines.append(f"\n  Baseline (sh=s+h):   {d['real_tokens_base']:>7,} tokens")
        lines.append(f"  Proposed (sh→{heb_char}):     {d['real_tokens_sh']:>7,} tokens")
        lines.append(f"  Real gain:           {d['real_gain']:>+7,} tokens")
        lines.append(
            f"  Random gain (mean):  {d['random_gain_mean']:>+7} "
            f"± {d['random_gain_std']}"
        )
        lines.append(f"  Differential:        {d['differential']:>+7} tokens")
        lines.append(f"  z-score:             {d['z_score']:>7}")
        lines.append(f"  p-value:             {d['p_value']}")
        lines.append(f"  Verdict:             {info['verdict']}")

        lines.append(f"\n  Words containing 'sh': {info['sh_word_count']:,}")
        lines.append(
            f"  Gained: {info['gained_count']} types, "
            f"{info['gained_tokens']:,} tokens"
        )
        lines.append(
            f"  Lost:   {info['lost_count']} types, "
            f"{info['lost_tokens']:,} tokens"
        )
        lines.append(
            f"  Both:   {info['both_count']} types, "
            f"{info['both_tokens']:,} tokens"
        )

        gained_glossed = info.get("gained_glossed", 0)
        lost_glossed = info.get("lost_glossed", 0)
        lines.append(f"  Gained with gloss: {gained_glossed}")
        lines.append(f"  Lost with gloss:   {lost_glossed}")

        if info["gained"]:
            lines.append(f"\n  Top Gained Words (sh→{heb_char}):")
            lines.append(
                f"  {'EVA':>12s}  {'Base':>8s}  {'New':>8s}  {'Freq':>5s}  Gloss"
            )
            for w in info["gained"][:15]:
                g = (w.get("gloss") or "")[:50]
                lines.append(
                    f"  {w['eva']:>12s}  {w['hebrew_baseline']:>8s}  "
                    f"{w['hebrew_sh']:>8s}  {w['freq']:>5d}  {g}"
                )

        if info["lost"]:
            lines.append(f"\n  Top Lost Words (sh→{heb_char}):")
            lines.append(
                f"  {'EVA':>12s}  {'Base':>8s}  {'New':>8s}  {'Freq':>5s}  Gloss"
            )
            for w in info["lost"][:10]:
                g = (w.get("gloss") or "")[:50]
                lines.append(
                    f"  {w['eva']:>12s}  {w['hebrew_baseline']:>8s}  "
                    f"{w['hebrew_sh']:>8s}  {w['freq']:>5d}  {g}"
                )

    # Final comparison
    lines.append(f"\n{'=' * 60}")
    lines.append("  COMPARISON")
    lines.append(f"{'=' * 60}")
    lines.append(
        f"  {'Target':>10s}  {'Gain':>7s}  {'Diff':>7s}  {'z':>6s}  "
        f"{'p':>8s}  Verdict"
    )
    for heb_char, info in target_results.items():
        d = info["differential"]
        heb_name = CONSONANT_NAMES.get(heb_char, heb_char)
        lines.append(
            f"  {heb_name:>10s}  {d['real_gain']:>+7,}  "
            f"{d['differential']:>+7}  {d['z_score']:>6}  "
            f"{d['p_value']:>8}  {info['verdict']}"
        )

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────


def run(config: ToolkitConfig, force: bool = False):
    """Run sh digraph investigation for tsade, qof, zayin."""
    out_json = config.stats_dir / "digraph_sh_deep.json"
    out_txt = config.stats_dir / "digraph_sh_deep_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    print_header("SH Digraph Investigation — sh → {tsade, qof, zayin}")

    print_step("Loading data...")
    eva_freqs, lexicon_set, form_to_gloss = load_data(config)
    n_sh_words = sum(1 for w in eva_freqs if "sh" in w)
    print(
        f"      EVA words: {len(eva_freqs):,}, lexicon: {len(lexicon_set):,}, "
        f"words with 'sh': {n_sh_words:,}"
    )

    target_results = {}

    for heb_char, heb_name in TARGETS.items():
        print_step(f"Testing sh → {heb_char} ({heb_name})...")

        # Word-level analysis
        gained, lost, both, sh_count = word_level_analysis(
            eva_freqs, lexicon_set, form_to_gloss, heb_char
        )
        net_types = len(gained) - len(lost)
        gained_tokens = sum(w["freq"] for w in gained)
        lost_tokens = sum(w["freq"] for w in lost)
        both_tokens = sum(w["freq"] for w in both)
        net_tokens = gained_tokens - lost_tokens
        print(
            f"      Gained: {len(gained)} types ({gained_tokens:,} tok), "
            f"Lost: {len(lost)} types ({lost_tokens:,} tok), "
            f"Net: {net_types:+d} types, {net_tokens:+,d} tokens"
        )

        # Differential test
        print(f"      Running differential test ({N_PERMUTATIONS} perms)...")
        diff = differential_test(eva_freqs, lexicon_set, heb_char)
        print(
            f"      Real gain: {diff['real_gain']:+,}, "
            f"Random: {diff['random_gain_mean']:+.0f} ± {diff['random_gain_std']:.0f}"
        )
        print(
            f"      Differential: {diff['differential']:+.0f}, "
            f"z={diff['z_score']}, p={diff['p_value']}"
        )

        verdict = "CONFIRMED" if diff["z_score"] > 2 and diff["differential"] > 0 else "REJECTED"
        if 0 < diff["z_score"] <= 2:
            verdict = "MARGINAL"
        print(f"      Verdict: {verdict}")

        gained_glossed = sum(
            1 for w in gained
            if w.get("gloss") and "[attestato" not in w.get("gloss", "")
        )
        lost_glossed = sum(
            1 for w in lost
            if w.get("gloss") and "[attestato" not in w.get("gloss", "")
        )

        target_results[heb_char] = {
            "hebrew_name": heb_name,
            "differential": diff,
            "gained_count": len(gained),
            "lost_count": len(lost),
            "both_count": len(both),
            "gained_tokens": gained_tokens,
            "lost_tokens": lost_tokens,
            "both_tokens": both_tokens,
            "net_types": net_types,
            "net_tokens": net_tokens,
            "sh_word_count": sh_count,
            "gained_glossed": gained_glossed,
            "lost_glossed": lost_glossed,
            "gained": gained[:50],
            "lost": lost[:50],
            "verdict": verdict,
        }

    # Save
    print_step("Writing output...")
    config.ensure_dirs()
    with open(out_json, "w") as f:
        json.dump(target_results, f, indent=1, ensure_ascii=False)

    summary = format_summary(target_results)
    with open(out_txt, "w") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")

    # Print comparison table
    print(f"\n  {'=' * 50}")
    print(f"  {'Target':>10s}  {'Gain':>7s}  {'Diff':>7s}  {'z':>6s}  Verdict")
    for heb_char, info in target_results.items():
        d = info["differential"]
        heb_name = CONSONANT_NAMES.get(heb_char, heb_char)
        print(
            f"  {heb_name:>10s}  {d['real_gain']:>+7,}  "
            f"{d['differential']:>+7}  {d['z_score']:>6}  {info['verdict']}"
        )
