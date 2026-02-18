"""Italian-layer analysis of decoded Hebrew text.

Transliterates Hebrew consonantal forms to Italian phonemes via
HEBREW_TO_ITALIAN, matches against 60K Italian lexicon (TLIO + Dante +
curated), and computes phonotactic plausibility (bigrams, vowel ratio,
letter frequencies). Random baseline via permutation of the mapping.

Cross-layer analysis identifies words matched in BOTH Hebrew and Italian
lexicons — the strongest decipherment evidence.
"""

from __future__ import annotations

import json
import math
import random as _random
from collections import Counter, defaultdict
from itertools import product

import numpy as np

from .config import ToolkitConfig
from .prepare_italian_lexicon import (
    HEBREW_ALTERNATIVES,
    HEBREW_TO_ITALIAN,
    ITALIAN_COMMON_BIGRAMS,
    ITALIAN_LETTER_FREQS,
)
from .utils import print_header, print_step

# ── Constants ───────────────────────────────────────────────────

VOWELS = set("aeiou")
HEBREW_CHARS = list(HEBREW_TO_ITALIAN.keys())  # 22 chars
MAX_VARIANTS_PER_WORD = 200
N_PERMUTATIONS = 200
RNG_SEED = 42


# ── Data loading ────────────────────────────────────────────────


def load_data(config: ToolkitConfig):
    """Load decoded corpus and Italian lexicon.

    Returns (word_freqs, words_by_section, it_lexicon_set, it_gloss, it_domain_sets).
    """
    # Decoded Hebrew words per page/section
    with open(config.stats_dir / "full_decode.json") as f:
        decode_data = json.load(f)

    word_freqs: Counter = Counter()
    words_by_section: dict[str, Counter] = defaultdict(Counter)
    for page_data in decode_data["pages"].values():
        section = page_data.get("section_code", "?")
        for w in page_data.get("words_hebrew", []):
            if w:
                word_freqs[w] += 1
                words_by_section[section][w] += 1

    # Italian lexicon
    with open(config.lexicon_dir / "italian_lexicon.json") as f:
        it_data = json.load(f)

    it_lexicon_set = set(it_data["all_forms"])
    it_gloss = it_data.get("form_to_gloss", {})

    it_domain_sets: dict[str, set] = {}
    for domain, entries in it_data.get("by_domain", {}).items():
        it_domain_sets[domain] = {e["phonemic"] for e in entries}

    # Hebrew lexicon (for cross-layer)
    with open(config.lexicon_dir / "lexicon_enriched.json") as f:
        heb_data = json.load(f)

    heb_form_to_gloss = heb_data["form_to_gloss"]  # 28K

    return (
        word_freqs,
        dict(words_by_section),
        it_lexicon_set,
        it_gloss,
        it_domain_sets,
        heb_form_to_gloss,
    )


# ── Transliteration ────────────────────────────────────────────


def transliterate(hebrew_word: str, mapping: dict[str, str] | None = None) -> str:
    """Convert Hebrew ASCII to Italian phonemes."""
    m = mapping or HEBREW_TO_ITALIAN
    return "".join(m.get(ch, "?") for ch in hebrew_word)


def generate_variants(hebrew_word: str) -> list[str]:
    """Generate Italian phonemic variants via HEBREW_ALTERNATIVES.

    Expands ambiguous chars (b↔v, p↔f, o↔u, etc.) combinatorially,
    capped at MAX_VARIANTS_PER_WORD.
    """
    options_per_pos = []
    for ch in hebrew_word:
        if ch in HEBREW_ALTERNATIVES:
            options_per_pos.append(HEBREW_ALTERNATIVES[ch])
        elif ch in HEBREW_TO_ITALIAN:
            options_per_pos.append([HEBREW_TO_ITALIAN[ch]])
        else:
            options_per_pos.append(["?"])

    # Estimate count before expanding
    n_combos = 1
    for opts in options_per_pos:
        n_combos *= len(opts)
        if n_combos > MAX_VARIANTS_PER_WORD:
            break

    if n_combos > MAX_VARIANTS_PER_WORD:
        # Fallback: base + single substitutions only
        base = transliterate(hebrew_word)
        variants = {base}
        for i, ch in enumerate(hebrew_word):
            if ch in HEBREW_ALTERNATIVES:
                for alt in HEBREW_ALTERNATIVES[ch]:
                    v = list(base)
                    # Replace the phoneme at position i
                    # Need to track position mapping carefully
                    prefix = transliterate(hebrew_word[:i])
                    suffix = transliterate(hebrew_word[i + 1 :])
                    variants.add(prefix + alt + suffix)
        return list(variants)

    results = set()
    for combo in product(*options_per_pos):
        results.add("".join(combo))
        if len(results) >= MAX_VARIANTS_PER_WORD:
            break
    return list(results)


# ── Italian matching ────────────────────────────────────────────


def match_all_words(word_freqs, it_lexicon_set, it_gloss):
    """Match decoded Hebrew words against Italian lexicon.

    Returns dict {hebrew_word: match_info} for matched words.
    """
    matches = {}

    for word in word_freqs:
        base = transliterate(word)

        # Direct match
        if base in it_lexicon_set:
            matches[word] = {
                "italian": base,
                "matched_form": base,
                "gloss": it_gloss.get(base, ""),
                "is_variant": False,
                "freq": word_freqs[word],
            }
            continue

        # Variant match
        variants = generate_variants(word)
        for v in variants:
            if v != base and v in it_lexicon_set:
                matches[word] = {
                    "italian": base,
                    "matched_form": v,
                    "gloss": it_gloss.get(v, ""),
                    "is_variant": True,
                    "freq": word_freqs[word],
                }
                break

    return matches


def count_matches_with_mapping(word_freqs, it_lexicon_set, mapping):
    """Count how many types/tokens match with a given mapping (no variants)."""
    matched_types = 0
    matched_tokens = 0
    for word, freq in word_freqs.items():
        it = "".join(mapping.get(ch, "?") for ch in word)
        if it in it_lexicon_set:
            matched_types += 1
            matched_tokens += freq
    return matched_types, matched_tokens


# ── Phonotactic analysis ───────────────────────────────────────


def analyze_phonotactics(word_freqs):
    """Compute letter frequencies, bigram coverage, vowel ratio from decoded Italian."""
    # Build full text (weighted by frequency)
    letter_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    total_chars = 0
    total_vowels = 0

    for word, freq in word_freqs.items():
        it = transliterate(word)
        if "?" in it:
            continue
        for _ in range(freq):
            for ch in it:
                letter_counts[ch] += 1
                total_chars += 1
                if ch in VOWELS:
                    total_vowels += 1
            for i in range(len(it) - 1):
                bigram_counts[it[i : i + 2]] += 1

    # Letter frequency correlation
    observed = {}
    for ch in ITALIAN_LETTER_FREQS:
        observed[ch] = letter_counts.get(ch, 0) / total_chars if total_chars else 0

    expected_vals = []
    observed_vals = []
    for ch in sorted(ITALIAN_LETTER_FREQS):
        expected_vals.append(ITALIAN_LETTER_FREQS[ch])
        observed_vals.append(observed.get(ch, 0))

    correlation = float(np.corrcoef(expected_vals, observed_vals)[0, 1])

    # Bigram coverage
    total_bigrams = sum(bigram_counts.values()) or 1
    bigram_coverage = 0
    for bg in ITALIAN_COMMON_BIGRAMS:
        bigram_coverage += bigram_counts.get(bg, 0)
    bigram_pct = bigram_coverage / total_bigrams

    # Top decoded bigrams vs Italian top
    decoded_top_20 = [bg for bg, _ in bigram_counts.most_common(20)]
    overlap = len(set(decoded_top_20) & set(ITALIAN_COMMON_BIGRAMS[:20]))

    # Vowel ratio
    vowel_ratio = total_vowels / total_chars if total_chars else 0

    # Letter distribution details
    letter_dist = {}
    for ch in sorted(letter_counts, key=lambda c: -letter_counts[c]):
        letter_dist[ch] = {
            "observed": round(letter_counts[ch] / total_chars, 4),
            "expected": ITALIAN_LETTER_FREQS.get(ch, 0),
        }

    return {
        "letter_freq_correlation": round(correlation, 4),
        "bigram_coverage_pct": round(100 * bigram_pct, 1),
        "bigram_top20_overlap": overlap,
        "vowel_ratio": round(vowel_ratio, 4),
        "vowel_ratio_expected": 0.468,  # Italian baseline
        "total_chars": total_chars,
        "decoded_top20_bigrams": decoded_top_20,
        "letter_distribution": letter_dist,
    }


# ── Domain congruence ──────────────────────────────────────────


def analyze_domains(words_by_section, it_lexicon_set, it_gloss, it_domain_sets):
    """Check if domain-specific Italian words appear in expected sections."""
    # Map MS sections to Italian lexicon domains
    section_to_domain = {
        "H": "botanical",
        "S": "astronomical",
        "Z": "astronomical",
        "B": "medical",
        "P": "medical",
    }

    results = {}
    for section, domain in section_to_domain.items():
        if section not in words_by_section or domain not in it_domain_sets:
            continue

        domain_forms = it_domain_sets[domain]
        section_words = words_by_section[section]

        domain_hits = 0
        domain_hit_words = []
        total_matched = 0

        for word, freq in section_words.items():
            it_base = transliterate(word)
            if it_base in it_lexicon_set:
                total_matched += 1
                if it_base in domain_forms:
                    domain_hits += 1
                    domain_hit_words.append(
                        (word, it_base, it_gloss.get(it_base, ""), freq)
                    )

        results[section] = {
            "domain": domain,
            "total_words": sum(section_words.values()),
            "total_types": len(section_words),
            "matched_types": total_matched,
            "domain_hits": domain_hits,
            "domain_hit_pct": round(100 * domain_hits / total_matched, 1)
            if total_matched
            else 0,
            "top_domain_words": sorted(domain_hit_words, key=lambda x: -x[3])[:10],
        }

    return results


# ── Random baseline (permutation) ──────────────────────────────


def random_baseline(word_freqs, it_lexicon_set, n_perms=N_PERMUTATIONS):
    """Compare real mapping vs random permutations of HEBREW_TO_ITALIAN."""
    rng = _random.Random(RNG_SEED)

    # Real match count (direct only, no variants)
    real_types, real_tokens = count_matches_with_mapping(
        word_freqs, it_lexicon_set, HEBREW_TO_ITALIAN
    )

    # Random permutations
    values = list(HEBREW_TO_ITALIAN.values())
    random_type_counts = []
    random_token_counts = []

    for _ in range(n_perms):
        shuffled = values[:]
        rng.shuffle(shuffled)
        rand_mapping = dict(zip(HEBREW_CHARS, shuffled))
        rt, rk = count_matches_with_mapping(word_freqs, it_lexicon_set, rand_mapping)
        random_type_counts.append(rt)
        random_token_counts.append(rk)

    mean_random_types = np.mean(random_type_counts)
    std_random_types = np.std(random_type_counts) or 1
    z_types = (real_types - mean_random_types) / std_random_types

    mean_random_tokens = np.mean(random_token_counts)
    std_random_tokens = np.std(random_token_counts) or 1
    z_tokens = (real_tokens - mean_random_tokens) / std_random_tokens

    p_types = sum(1 for r in random_type_counts if r >= real_types) / n_perms
    p_tokens = sum(1 for r in random_token_counts if r >= real_tokens) / n_perms

    return {
        "real_types": real_types,
        "real_tokens": real_tokens,
        "random_mean_types": round(float(mean_random_types), 1),
        "random_std_types": round(float(std_random_types), 1),
        "random_mean_tokens": round(float(mean_random_tokens), 1),
        "random_std_tokens": round(float(std_random_tokens), 1),
        "z_types": round(float(z_types), 2),
        "z_tokens": round(float(z_tokens), 2),
        "p_types": p_types,
        "p_tokens": p_tokens,
        "n_perms": n_perms,
    }


# ── Cross-layer analysis ───────────────────────────────────────


def cross_layer_analysis(word_freqs, it_matches, heb_form_to_gloss):
    """Find words matched in BOTH Hebrew and Italian lexicons."""
    cross = []
    for word, freq in word_freqs.most_common():
        heb_gloss = heb_form_to_gloss.get(word)
        it_info = it_matches.get(word)

        if heb_gloss and it_info and it_info["gloss"]:
            cross.append(
                {
                    "hebrew": word,
                    "italian": it_info["matched_form"],
                    "freq": freq,
                    "hebrew_gloss": heb_gloss[:80],
                    "italian_gloss": it_info["gloss"][:80],
                    "is_variant": it_info["is_variant"],
                }
            )

    cross.sort(key=lambda x: -x["freq"])
    return cross


# ── Summary formatting ─────────────────────────────────────────


def format_summary(
    word_freqs, it_matches, phonotactics, domain_results, baseline, cross_layer
):
    """Format human-readable summary."""
    total_types = len(word_freqs)
    total_tokens = sum(word_freqs.values())
    matched_types = len(it_matches)
    matched_tokens = sum(m["freq"] for m in it_matches.values())
    direct = sum(1 for m in it_matches.values() if not m["is_variant"])
    variant = sum(1 for m in it_matches.values() if m["is_variant"])

    lines = []
    lines.append("=" * 60)
    lines.append("  ITALIAN-LAYER ANALYSIS")
    lines.append("=" * 60)

    # ── Match rates ──
    lines.append("\n── Italian Lexicon Matching ──")
    lines.append(f"  Total types:    {total_types:>7,}")
    lines.append(f"  Total tokens:   {total_tokens:>7,}")
    lines.append(
        f"  Matched types:  {matched_types:>7,} ({100*matched_types/total_types:.1f}%)"
    )
    lines.append(
        f"  Matched tokens: {matched_tokens:>7,} ({100*matched_tokens/total_tokens:.1f}%)"
    )
    lines.append(f"    Direct:       {direct:>7,}")
    lines.append(f"    Via variant:  {variant:>7,}")

    # ── Random baseline ──
    lines.append("\n── Random Baseline ──")
    lines.append(
        f"  Real (direct):  {baseline['real_types']} types, {baseline['real_tokens']} tokens"
    )
    lines.append(
        f"  Random mean:    {baseline['random_mean_types']} ± {baseline['random_std_types']} types, "
        f"{baseline['random_mean_tokens']} ± {baseline['random_std_tokens']} tokens"
    )
    lines.append(
        f"  z-score:        types={baseline['z_types']}, tokens={baseline['z_tokens']}"
    )
    lines.append(
        f"  p-value:        types={baseline['p_types']}, tokens={baseline['p_tokens']}"
    )
    lines.append(f"  Permutations:   {baseline['n_perms']}")

    # ── Phonotactics ──
    lines.append("\n── Phonotactic Analysis ──")
    lines.append(
        f"  Letter freq correlation:  {phonotactics['letter_freq_correlation']:.4f}"
    )
    lines.append(f"  Bigram coverage:          {phonotactics['bigram_coverage_pct']}%")
    lines.append(
        f"  Bigram top-20 overlap:    {phonotactics['bigram_top20_overlap']}/20"
    )
    lines.append(
        f"  Vowel ratio:              {phonotactics['vowel_ratio']:.3f} "
        f"(expected: {phonotactics['vowel_ratio_expected']:.3f})"
    )
    lines.append(
        f"  Decoded top bigrams:      {', '.join(phonotactics['decoded_top20_bigrams'])}"
    )

    # ── Letter distribution ──
    lines.append("\n── Letter Distribution (top 15) ──")
    lines.append(f"  {'Char':>4s}  {'Observed':>8s}  {'Expected':>8s}  {'Δ':>8s}")
    ld = phonotactics["letter_distribution"]
    sorted_ld = sorted(ld.items(), key=lambda x: -x[1]["observed"])[:15]
    for ch, info in sorted_ld:
        delta = info["observed"] - info["expected"]
        lines.append(
            f"  {ch:>4s}  {info['observed']:>8.4f}  {info['expected']:>8.4f}  {delta:>+8.4f}"
        )

    # ── Domain congruence ──
    if domain_results:
        lines.append("\n── Domain Congruence ──")
        for section, dr in sorted(domain_results.items()):
            lines.append(
                f"  Section {section} ({dr['domain']:12s}): "
                f"{dr['domain_hits']}/{dr['matched_types']} domain words "
                f"({dr['domain_hit_pct']}%)"
            )
            for word, it_form, gloss, freq in dr["top_domain_words"][:5]:
                lines.append(f"    {word:>10s} → {it_form:12s} x{freq:<4d} {gloss[:40]}")

    # ── Cross-layer ──
    lines.append(f"\n── Cross-layer (Hebrew + Italian) ──")
    lines.append(f"  Words in both lexicons: {len(cross_layer)}")
    if cross_layer:
        lines.append(
            f"\n  {'Hebrew':>10s}  {'Italian':>10s}  {'Freq':>5s}  "
            f"{'Hebrew gloss':30s}  {'Italian gloss':30s}"
        )
        for entry in cross_layer[:25]:
            v = "*" if entry["is_variant"] else " "
            lines.append(
                f"  {entry['hebrew']:>10s}  {entry['italian']:>10s}{v} {entry['freq']:>5d}  "
                f"{entry['hebrew_gloss']:30s}  {entry['italian_gloss']:30s}"
            )

    # ── Top matched words ──
    lines.append("\n── Top 30 Italian Matches ──")
    top_matches = sorted(it_matches.items(), key=lambda x: -x[1]["freq"])[:30]
    for word, info in top_matches:
        v = "*" if info["is_variant"] else " "
        lines.append(
            f"  {word:>12s} → {info['matched_form']:12s}{v} x{info['freq']:<5d} "
            f"{info['gloss'][:50]}"
        )

    return "\n".join(lines)


# ── Main entry point ────────────────────────────────────────────


def run(config: ToolkitConfig, force: bool = False):
    """Run Italian-layer analysis."""
    out_json = config.stats_dir / "italian_layer.json"
    out_txt = config.stats_dir / "italian_layer_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    print_header("Italian-Layer Analysis")

    # ── Load data ──
    print_step("Loading data...")
    (
        word_freqs,
        words_by_section,
        it_lexicon_set,
        it_gloss,
        it_domain_sets,
        heb_form_to_gloss,
    ) = load_data(config)
    print(
        f"      Decoded: {len(word_freqs):,} types, {sum(word_freqs.values()):,} tokens"
    )
    print(f"      Italian lexicon: {len(it_lexicon_set):,} forms")

    # ── Match ──
    print_step("Matching against Italian lexicon (direct + variants)...")
    it_matches = match_all_words(word_freqs, it_lexicon_set, it_gloss)
    direct = sum(1 for m in it_matches.values() if not m["is_variant"])
    variant = sum(1 for m in it_matches.values() if m["is_variant"])
    matched_tokens = sum(m["freq"] for m in it_matches.values())
    print(
        f"      Matched: {len(it_matches):,} types ({direct} direct, {variant} variant), "
        f"{matched_tokens:,} tokens ({100*matched_tokens/sum(word_freqs.values()):.1f}%)"
    )

    # ── Phonotactics ──
    print_step("Phonotactic analysis...")
    phonotactics = analyze_phonotactics(word_freqs)
    print(
        f"      Letter freq r={phonotactics['letter_freq_correlation']:.3f}, "
        f"vowel ratio={phonotactics['vowel_ratio']:.3f}, "
        f"bigram coverage={phonotactics['bigram_coverage_pct']}%"
    )

    # ── Domain congruence ──
    print_step("Domain congruence...")
    domain_results = analyze_domains(
        words_by_section, it_lexicon_set, it_gloss, it_domain_sets
    )
    for section, dr in sorted(domain_results.items()):
        print(
            f"      {section} ({dr['domain']}): {dr['domain_hits']}/{dr['matched_types']} domain words"
        )

    # ── Random baseline ──
    print_step(f"Random baseline ({N_PERMUTATIONS} permutations)...")
    baseline = random_baseline(word_freqs, it_lexicon_set)
    print(
        f"      z-score: types={baseline['z_types']}, tokens={baseline['z_tokens']}"
    )

    # ── Cross-layer ──
    print_step("Cross-layer analysis (Hebrew ∩ Italian)...")
    cross_layer = cross_layer_analysis(word_freqs, it_matches, heb_form_to_gloss)
    print(f"      Words in both lexicons: {len(cross_layer)}")

    # ── Output ──
    print_step("Writing output...")

    output = {
        "stats": {
            "total_types": len(word_freqs),
            "total_tokens": sum(word_freqs.values()),
            "matched_types": len(it_matches),
            "matched_tokens": matched_tokens,
            "direct_types": direct,
            "variant_types": variant,
            "match_rate_types": round(100 * len(it_matches) / len(word_freqs), 1),
            "match_rate_tokens": round(
                100 * matched_tokens / sum(word_freqs.values()), 1
            ),
        },
        "phonotactics": phonotactics,
        "domain_congruence": {
            s: {k: v for k, v in dr.items() if k != "top_domain_words"}
            for s, dr in domain_results.items()
        },
        "random_baseline": baseline,
        "cross_layer_count": len(cross_layer),
        "cross_layer": cross_layer[:100],
        "top_matches": [
            {
                "hebrew": w,
                "italian": m["matched_form"],
                "gloss": m["gloss"][:100],
                "freq": m["freq"],
                "is_variant": m["is_variant"],
            }
            for w, m in sorted(it_matches.items(), key=lambda x: -x[1]["freq"])[:200]
        ],
    }

    config.ensure_dirs()
    with open(out_json, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    summary = format_summary(
        word_freqs, it_matches, phonotactics, domain_results, baseline, cross_layer
    )
    with open(out_txt, "w") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")
    print(
        f"\n      Italian match: {len(it_matches):,} types "
        f"({100*len(it_matches)/len(word_freqs):.1f}%), "
        f"z={baseline['z_types']}"
    )
