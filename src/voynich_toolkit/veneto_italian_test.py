"""
Phase 25: Test Italian/Veneto vocabulary in unmatched decoded forms.

The existing Judeo-Italian test (Phase 14) used standard JI manuscript
transliteration conventions where Italian 'e' -> yod (y). But the cipher
itself maps 'e' -> ayin (E) or he (h). This module uses the cipher's
OWN phonemic mapping to transliterate Italian, fixing that mismatch.

Hypothesis: the 75% unmatched decoded forms are Italian/Veneto words
encoded through the same EVA->Hebrew cipher. If true, cipher-native
transliteration should produce significantly more matches than random.
"""

import json
import random
import time
import unicodedata
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from .config import ToolkitConfig
from .judeo_italian_test import (
    load_decoded_corpus,
    load_hebrew_set,
    load_italian_forms,
    strip_accents,
)
from .permutation_stats import (
    build_full_mapping,
    decode_eva_with_mapping,
    generate_random_mapping,
)
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# =====================================================================
# Cipher-native transliteration: Italian -> Hebrew using the CIPHER's
# own phonemic mapping (not standard JI conventions)
# =====================================================================

# Invert HEBREW_TO_ITALIAN: Italian sound -> list of possible Hebrew letters
# Multiple Hebrew letters can map to the same Italian sound (ambiguity)
def _build_italian_to_hebrew():
    """Build reverse mapping: Italian phoneme -> list of Hebrew ASCII chars."""
    reverse = {}
    for heb, ita in HEBREW_TO_ITALIAN.items():
        if ita not in reverse:
            reverse[ita] = []
        if heb not in reverse[ita]:
            reverse[ita].append(heb)
    return reverse

_ITALIAN_TO_HEBREW = _build_italian_to_hebrew()

# Italian consonant digraphs that need special handling
_DIGRAPHS = {
    'ch': ['k', 'X'],        # /k/ before e,i -> kaf or chet
    'gh': ['g'],              # /g/ before e,i -> gimel
    'gn': ['ny'],             # /ɲ/ -> nun+yod (placeholder, expanded below)
    'qu': ['kw'],             # /kw/ -> kaf+vav (placeholder)
    'sc': None,               # context-dependent: handled in code
}

MAX_VARIANTS = 64  # cap to prevent combinatorial explosion


def cipher_transliterate(word: str) -> list[str]:
    """Transliterate Italian word to Hebrew using the cipher's own mapping.

    Unlike the standard JI transliteration (e->yod), this uses the
    cipher's actual mapping: a->aleph(A), e->ayin(E)/he(h), i->yod(y),
    o->vav(w), u->vav(w).

    Returns list of possible Hebrew forms (due to ambiguities like
    e->{E,h}, t->{t,J}, k->{X,k}, s->{s,S}, l->{l,p}).
    """
    w = strip_accents(word.lower())
    if not w or not w.isalpha():
        return []

    # Build sequence of possible chars at each position
    char_options = []
    i = 0
    n = len(w)

    while i < n:
        # --- Trigraph: gli + vowel ---
        if w[i:i+3] == 'gli' and i + 3 < n and w[i+3] in 'aeiou':
            char_options.append(['l'])
            char_options.append(['y'])
            i += 3
            continue

        # --- Digraphs ---
        if i + 1 < n:
            pair = w[i:i+2]

            if pair == 'ch':
                char_options.append(['k', 'X'])  # kaf or chet
                i += 2
                continue

            if pair == 'gh':
                char_options.append(['g'])
                i += 2
                continue

            if pair == 'gn':
                char_options.append(['n'])
                char_options.append(['y'])
                i += 2
                continue

            if pair == 'sc' and i + 2 < n and w[i+2] in 'ei':
                char_options.append(['S'])  # shin for /ʃ/
                i += 2
                continue

            if pair == 'qu':
                char_options.append(['k', 'X'])
                char_options.append(['w'])
                i += 2
                continue

        c = w[i]

        # --- Context-dependent c ---
        if c == 'c':
            if i + 1 < n and w[i+1] in 'ei':
                char_options.append(['C'])  # tsade for /tʃ/
            else:
                char_options.append(['k', 'X'])  # kaf or chet for /k/
            i += 1
            continue

        # --- Context-dependent g ---
        if c == 'g':
            char_options.append(['g'])
            i += 1
            continue

        # --- Silent h ---
        if c == 'h':
            i += 1
            continue

        # --- j -> yod ---
        if c == 'j':
            char_options.append(['y'])
            i += 1
            continue

        # --- z -> zayin ---
        if c == 'z':
            char_options.append(['z', 'C'])  # zayin or tsade
            i += 1
            continue

        # --- Known consonants and vowels ---
        if c in _ITALIAN_TO_HEBREW:
            options = _ITALIAN_TO_HEBREW[c]
            char_options.append(options)
            i += 1
            continue

        # Unknown char: skip
        i += 1

    if not char_options:
        return []

    # Compute number of variants
    n_variants = 1
    for opts in char_options:
        n_variants *= len(opts)
        if n_variants > MAX_VARIANTS:
            break

    if n_variants > MAX_VARIANTS:
        # Too many: just use first option for each position
        return [''.join(opts[0] for opts in char_options)]

    # Generate all variants
    variants = set()
    for combo in product(*char_options):
        form = ''.join(combo)
        if len(form) >= 3:
            variants.add(form)

    return sorted(variants)


# =====================================================================
# Normalization (collapse known ambiguities)
# =====================================================================

def normalize_cipher(heb: str) -> str:
    """Normalize Hebrew form by collapsing cipher ambiguities.

    J -> t  (tet/tav both = /t/)
    X -> k  (chet/kaf both = /k/ in this cipher)
    h -> E  (he/ayin both = /e/ in this cipher)
    S -> s  (shin/samekh both = /s/ in this cipher... roughly)
    p -> l  (pe/lamed: both map to /l/ in the cipher - EVA f and p)
    """
    return heb.translate(str.maketrans('JXhSp', 'tkEsl'))


# =====================================================================
# Lexicon construction
# =====================================================================

def load_north_italian_forms(config: ToolkitConfig) -> dict[str, list[str]]:
    """Load North Italian dialect forms from downloaded lexicons."""
    path = config.lexicon_dir / "north_italian_lexicons.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for lang_key, lang_data in data.items():
        if lang_key == "combined":
            continue
        if isinstance(lang_data, dict) and "forms" in lang_data:
            result[lang_key] = lang_data["forms"]
    return result


def build_cipher_native_lexicon(
    italian_forms: list[str],
    north_italian: dict[str, list[str]] | None = None,
) -> tuple[set, set, dict]:
    """Build Hebrew lexicon using cipher-native transliteration.

    Returns:
        strict_set: set of exact Hebrew transliterations
        normalized_set: set of normalized Hebrew transliterations
        source_map: dict {hebrew_form: [italian_words]}
    """
    strict_set = set()
    normalized_set = set()
    source_map: dict[str, list[str]] = {}

    all_forms = set(italian_forms)
    if north_italian:
        for forms in north_italian.values():
            all_forms.update(forms)

    for word in all_forms:
        variants = cipher_transliterate(word)
        for heb in variants:
            strict_set.add(heb)
            normalized_set.add(normalize_cipher(heb))
            source_map.setdefault(heb, []).append(word)

    return strict_set, normalized_set, source_map


# =====================================================================
# Match analysis
# =====================================================================

def classify_decoded(
    decoded_heb: list[str],
    italian_strict: set,
    italian_norm: set,
    hebrew_set: set,
    min_len: int = 3,
) -> dict:
    """Classify each decoded type: hebrew-only, italian-only, both, neither."""
    types = Counter()
    for w in decoded_heb:
        if len(w) >= min_len:
            types[w] += 1

    both = []
    ita_only = []
    heb_only = []
    neither = []

    for word, freq in types.items():
        norm = normalize_cipher(word)
        in_ita = word in italian_strict or norm in italian_norm
        in_heb = word in hebrew_set
        entry = (word, freq)
        if in_ita and in_heb:
            both.append(entry)
        elif in_ita:
            ita_only.append(entry)
        elif in_heb:
            heb_only.append(entry)
        else:
            neither.append(entry)

    return {
        "both": sorted(both, key=lambda x: -x[1]),
        "italian_only": sorted(ita_only, key=lambda x: -x[1]),
        "hebrew_only": sorted(heb_only, key=lambda x: -x[1]),
        "neither": sorted(neither, key=lambda x: -x[1]),
        "total_types": len(types),
        "total_tokens": sum(types.values()),
    }


def section_breakdown(
    config: ToolkitConfig,
    italian_strict: set,
    italian_norm: set,
    hebrew_set: set,
    min_len: int = 3,
) -> dict:
    """Compute per-section match rates."""
    decode_path = config.stats_dir / "full_decode.json"
    if not decode_path.exists():
        return {}

    data = json.loads(decode_path.read_text(encoding="utf-8"))
    section_stats = {}

    for folio, page in data.get("pages", {}).items():
        section = page.get("section", "?")
        if section not in section_stats:
            section_stats[section] = {
                "total": 0, "heb": 0, "ita": 0, "both": 0, "neither": 0,
            }
        for w in page.get("words_hebrew", []):
            if len(w) < min_len:
                continue
            s = section_stats[section]
            s["total"] += 1
            norm = normalize_cipher(w)
            in_ita = w in italian_strict or norm in italian_norm
            in_heb = w in hebrew_set
            if in_ita and in_heb:
                s["both"] += 1
            elif in_ita:
                s["ita"] += 1
            elif in_heb:
                s["heb"] += 1
            else:
                s["neither"] += 1

    # Compute rates
    for section, s in section_stats.items():
        t = max(s["total"], 1)
        s["heb_rate"] = (s["heb"] + s["both"]) / t
        s["ita_rate"] = (s["ita"] + s["both"]) / t
        s["combined_rate"] = (s["heb"] + s["ita"] + s["both"]) / t
        s["neither_rate"] = s["neither"] / t

    return section_stats


# =====================================================================
# Permutation test
# =====================================================================

def run_permutation_test(
    eva_words: list[str],
    italian_norm_set: set,
    n_perms: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test: shuffle EVA->Hebrew mapping, measure Italian match rate.

    Uses cipher-native normalization (not JI conventions).
    """
    from .cross_language_baseline import MAPPING

    rng = random.Random(seed)
    real_full = build_full_mapping(MAPPING)
    real_keys = sorted(real_full.keys())

    # Real mapping score
    real_matched = 0
    real_total = 0
    for w in eva_words:
        if len(w) < 3:
            continue
        heb = decode_eva_with_mapping(w, real_full, mode="hebrew")
        if heb:
            real_total += 1
            if normalize_cipher(heb) in italian_norm_set:
                real_matched += 1
    real_rate = real_matched / max(real_total, 1)

    # Random mappings
    sim_rates = []
    t0 = time.time()

    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"      perm {i+1}/{n_perms}  ({elapsed:.1f}s)")

        rand_map = generate_random_mapping(
            eva_chars=real_keys, seed=rng.randint(0, 2**31))

        n_matched = 0
        n_total = 0
        for w in eva_words:
            if len(w) < 3:
                continue
            heb = decode_eva_with_mapping(w, rand_map, mode="hebrew")
            if heb:
                n_total += 1
                if normalize_cipher(heb) in italian_norm_set:
                    n_matched += 1

        sim_rates.append(n_matched / max(n_total, 1))

    sim_arr = np.array(sim_rates)
    sim_mean = float(np.mean(sim_arr))
    sim_std = float(np.std(sim_arr))
    z_score = (real_rate - sim_mean) / sim_std if sim_std > 0 else float("inf")
    n_above = int(np.sum(sim_arr >= real_rate))
    p_value = (n_above + 1) / (n_perms + 1)

    return {
        "real_rate": real_rate,
        "real_matched": real_matched,
        "real_total": real_total,
        "n_perms": n_perms,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_min": float(np.min(sim_arr)),
        "sim_max": float(np.max(sim_arr)),
        "z_score": z_score,
        "p_value": p_value,
        "n_above_real": n_above,
        "elapsed_s": time.time() - t0,
    }


# =====================================================================
# Output formatting
# =====================================================================

def format_summary(
    lexicon_stats: dict,
    classification: dict,
    sections: dict,
    perm_results: dict,
    old_ji_rate: float,
    source_map: dict,
) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 65)
    lines.append("  PHASE 25: ITALIAN/VENETO VOCABULARY TEST")
    lines.append("  (cipher-native transliteration)")
    lines.append("=" * 65)

    # 1. Lexicon
    lines.append("\n  1. LEXICON CONSTRUCTION")
    lines.append("  " + "-" * 61)
    ls = lexicon_stats
    lines.append(f"  Italian forms (TLIO+Dante+kaikki):  {ls['n_existing']:>8,}")
    if ls.get("n_north_italian", 0) > 0:
        lines.append(f"  North Italian dialects (new):       {ls['n_north_italian']:>8,}")
    lines.append(f"  Combined unique Italian forms:       {ls['n_combined']:>8,}")
    lines.append(f"  Cipher-native Hebrew forms (strict): {ls['n_strict']:>8,}")
    lines.append(f"  Cipher-native Hebrew forms (norm):   {ls['n_norm']:>8,}")

    lines.append("\n  Sample cipher-native transliterations:")
    for s in ls.get("samples", []):
        variants = ", ".join(s["hebrew"][:3])
        if len(s["hebrew"]) > 3:
            variants += f" (+{len(s['hebrew'])-3} more)"
        lines.append(f"    {s['italian']:15s} -> {variants}")

    # 2. Classification
    lines.append("\n  2. DECODED FORM CLASSIFICATION")
    lines.append("  " + "-" * 61)
    cl = classification
    tot_types = cl["total_types"]
    tot_tok = cl["total_tokens"]
    for cat, label in [("hebrew_only", "Hebrew-only"),
                       ("italian_only", "Italian-only"),
                       ("both", "Both Heb+Ita"),
                       ("neither", "Neither")]:
        n_types = len(cl[cat])
        n_tok = sum(f for _, f in cl[cat])
        lines.append(
            f"  {label:20s} {n_types:>6,} types ({n_types/max(tot_types,1)*100:5.1f}%)"
            f"  {n_tok:>8,} tokens ({n_tok/max(tot_tok,1)*100:5.1f}%)"
        )

    combined_types = len(cl["hebrew_only"]) + len(cl["italian_only"]) + len(cl["both"])
    combined_tok = (sum(f for _, f in cl["hebrew_only"])
                    + sum(f for _, f in cl["italian_only"])
                    + sum(f for _, f in cl["both"]))
    lines.append(f"\n  Combined coverage:   "
                 f"{combined_types:,} types ({combined_types/max(tot_types,1)*100:.1f}%)"
                 f"  {combined_tok:,} tokens ({combined_tok/max(tot_tok,1)*100:.1f}%)")

    # Top Italian-only matches
    ita_only = cl["italian_only"][:15]
    if ita_only:
        lines.append("\n  Top Italian-only matches:")
        for word, freq in ita_only:
            sources = source_map.get(word, [])
            if not sources:
                norm = normalize_cipher(word)
                for heb_form, src_words in source_map.items():
                    if normalize_cipher(heb_form) == norm:
                        sources = src_words
                        break
            src_str = ", ".join(sources[:3]) if sources else "?"
            lines.append(f"    {word:15s} freq={freq:>4d}  <- {src_str}")

    # 3. Section breakdown
    if sections:
        lines.append("\n  3. SECTION BREAKDOWN")
        lines.append("  " + "-" * 61)
        lines.append(f"  {'Section':10s} {'Total':>6s} {'Heb%':>7s} "
                     f"{'Ita%':>7s} {'Both%':>7s} {'Neither%':>9s}")
        for section in sorted(sections.keys()):
            s = sections[section]
            lines.append(
                f"  {section:10s} {s['total']:6,d} "
                f"{s['heb_rate']*100:6.1f}% "
                f"{s['ita_rate']*100:6.1f}% "
                f"{s['combined_rate']*100:6.1f}% "
                f"{s['neither_rate']*100:8.1f}%"
            )

    # 4. Permutation test
    lines.append("\n  4. PERMUTATION TEST")
    lines.append("  " + "-" * 61)
    pr = perm_results
    lines.append(f"  Permutations:       {pr['n_perms']}")
    lines.append(f"  Italian real:       {pr['real_rate']*100:.1f}% "
                 f"({pr['real_matched']:,}/{pr['real_total']:,})")
    lines.append(f"  Italian random:     {pr['sim_mean']*100:.1f}% "
                 f"+/- {pr['sim_std']*100:.1f}%")
    lines.append(f"  z-score:            {pr['z_score']:.2f}  "
                 f"(p={pr['p_value']:.6f})")
    if old_ji_rate > 0:
        lines.append(f"  Old JI test rate:   {old_ji_rate*100:.1f}% "
                     f"(for comparison)")
    lines.append(f"  Elapsed:            {pr['elapsed_s']:.1f}s")

    # 5. Verdict
    lines.append("\n  5. VERDICT")
    lines.append("  " + "=" * 61)
    z = pr['z_score']
    ita_only_types = len(cl["italian_only"])
    ita_only_tokens = sum(f for _, f in cl["italian_only"])

    if z > 3.0 and ita_only_types > 50:
        verdict = "ITALIAN SIGNAL DETECTED"
        detail = (f"z={z:.1f}, {ita_only_types} Italian-only types "
                  f"({ita_only_tokens:,} tokens). The unmatched gap "
                  f"contains real Italian/Veneto vocabulary.")
    elif z > 2.0:
        verdict = "MARGINAL ITALIAN SIGNAL"
        detail = (f"z={z:.1f}, {ita_only_types} Italian-only types. "
                  f"Suggestive but not conclusive.")
    elif z > 0:
        verdict = "WEAK/NO ITALIAN SIGNAL"
        detail = (f"z={z:.1f}. Italian match rate exceeds random slightly "
                  f"but not significantly.")
    else:
        verdict = "NO ITALIAN SIGNAL"
        detail = (f"z={z:.1f}. The 75% gap is NOT explained by "
                  f"Italian/Veneto vocabulary.")

    lines.append(f"  {verdict}")
    lines.append(f"  {detail}")
    lines.append("  " + "=" * 61)

    return "\n".join(lines)


# =====================================================================
# Main entry point
# =====================================================================

def run(config: ToolkitConfig, force=False):
    """Phase 25: Test Italian/Veneto vocabulary with cipher-native transliteration."""
    out_json = config.stats_dir / "veneto_italian_test.json"
    out_txt = config.stats_dir / "veneto_italian_test_summary.txt"

    if not force and out_json.exists():
        print(f"  Output exists: {out_json.name} (use --force to re-run)")
        return

    config.ensure_dirs()
    print_header("PHASE 25: ITALIAN/VENETO VOCABULARY TEST")

    # 1. Load Italian lexicons
    print_step("Loading Italian lexicons...")
    italian_forms = load_italian_forms(config)
    print(f"      Existing Italian forms: {len(italian_forms):,}")

    north_italian = load_north_italian_forms(config)
    n_north = sum(len(v) for v in north_italian.values())
    if north_italian:
        for lang, forms in north_italian.items():
            print(f"      {lang}: {len(forms):,} forms")
    else:
        print("      No North Italian lexicons found.")
        print("      (run: python scripts/download_veneto_lexicons.py)")

    # 2. Build cipher-native lexicon
    print_step("Building cipher-native Hebrew lexicon...")
    strict_set, norm_set, source_map = build_cipher_native_lexicon(
        italian_forms, north_italian)
    print(f"      Strict forms:     {len(strict_set):,}")
    print(f"      Normalized forms: {len(norm_set):,}")

    # Sample transliterations
    test_words = [
        "acqua", "seta", "rosa", "erba", "bagno",
        "stella", "olio", "sale", "miele", "vino",
        "carne", "pelle", "radice", "fiore", "foglia",
    ]
    samples = []
    for w in test_words:
        heb_forms = cipher_transliterate(w)
        if heb_forms:
            samples.append({"italian": w, "hebrew": heb_forms})
            preview = ", ".join(heb_forms[:3])
            print(f"      {w:15s} -> {preview}")

    # 3. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    heb_set = load_hebrew_set(config)
    print(f"      {len(heb_set):,} Hebrew consonantal forms")

    # 4. Load decoded corpus
    print_step("Loading decoded corpus...")
    decoded_heb, eva_words = load_decoded_corpus(config)
    print(f"      {len(decoded_heb):,} decoded tokens")

    # 5. Classification
    print_step("Classifying decoded forms...")
    classification = classify_decoded(
        decoded_heb, strict_set, norm_set, heb_set)
    n_heb_only = len(classification["hebrew_only"])
    n_ita_only = len(classification["italian_only"])
    n_both = len(classification["both"])
    n_neither = len(classification["neither"])
    print(f"      Hebrew-only:  {n_heb_only:,} types")
    print(f"      Italian-only: {n_ita_only:,} types")
    print(f"      Both:         {n_both:,} types")
    print(f"      Neither:      {n_neither:,} types")

    # 6. Section breakdown
    print_step("Computing section breakdown...")
    sections = section_breakdown(config, strict_set, norm_set, heb_set)
    for s_name in sorted(sections.keys()):
        s = sections[s_name]
        print(f"      {s_name}: heb={s['heb_rate']*100:.1f}% "
              f"ita={s['ita_rate']*100:.1f}% "
              f"combined={s['combined_rate']*100:.1f}%")

    # 7. Permutation test
    print_step("Running permutation test (1000 iterations)...")
    perm_results = run_permutation_test(
        eva_words, norm_set, n_perms=1000, seed=42)
    print(f"      Italian real:   {perm_results['real_rate']*100:.1f}%")
    print(f"      Italian random: {perm_results['sim_mean']*100:.1f}% "
          f"+/- {perm_results['sim_std']*100:.1f}%")
    print(f"      z-score:        {perm_results['z_score']:.2f} "
          f"(p={perm_results['p_value']:.6f})")

    # 8. Load old JI rate for comparison
    old_ji_rate = 0.0
    old_ji_path = config.stats_dir / "judeo_italian_test.json"
    if old_ji_path.exists():
        old_data = json.loads(old_ji_path.read_text(encoding="utf-8"))
        old_ji_rate = (old_data.get("match_rates", {})
                       .get("ji_normalized", {})
                       .get("rate", 0.0))

    # 9. Build lexicon stats for summary
    lexicon_stats = {
        "n_existing": len(italian_forms),
        "n_north_italian": n_north,
        "n_combined": len(set(italian_forms) | set(
            f for forms in north_italian.values() for f in forms
        )) if north_italian else len(set(italian_forms)),
        "n_strict": len(strict_set),
        "n_norm": len(norm_set),
        "samples": samples,
    }

    # 10. Format and save
    summary = format_summary(
        lexicon_stats, classification, sections,
        perm_results, old_ji_rate, source_map)
    print(f"\n{summary}")

    output = {
        "lexicon": lexicon_stats,
        "classification": {
            "hebrew_only_types": n_heb_only,
            "hebrew_only_tokens": sum(f for _, f in classification["hebrew_only"]),
            "italian_only_types": n_ita_only,
            "italian_only_tokens": sum(f for _, f in classification["italian_only"]),
            "both_types": n_both,
            "both_tokens": sum(f for _, f in classification["both"]),
            "neither_types": n_neither,
            "neither_tokens": sum(f for _, f in classification["neither"]),
            "total_types": classification["total_types"],
            "total_tokens": classification["total_tokens"],
            "top_italian_only": [
                {"decoded": w, "freq": f,
                 "italian_source": source_map.get(w, [])[:3]}
                for w, f in classification["italian_only"][:30]
            ],
        },
        "sections": {
            s: {k: v for k, v in data.items()}
            for s, data in sections.items()
        },
        "permutation_test": perm_results,
        "old_ji_normalized_rate": old_ji_rate,
    }
    out_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n  Saved: {out_json.name}")

    out_txt.write_text(summary, encoding="utf-8")
    print(f"  Saved: {out_txt.name}")
