"""
Test Judeo-Italian hypothesis: Italian written in Hebrew script.

The hypothesis: the VMS text is Judeo-Italian — Italian written in Hebrew
consonantal script, a documented practice in 15th-century northern Italian
Jewish communities. This explains why:
- Individual "Hebrew words" match: Hebrew letters represent Italian sounds
- Phrases don't read as Hebrew: the underlying language is Italian
- Vowel ratio (0.302) is low for Italian: vowels are matres lectionis

Key insight: the 3 unmapped Hebrew letters (zayin, tsade, qof) are exactly
those used for specifically Italian sounds (z, c dolce, c dura/qu) in
attested Judeo-Italian manuscripts.

Method:
1. Transliterate 60K Italian forms to Hebrew via JI conventions
2. Compare decoded VMS text against JI lexicon vs Hebrew lexicon
3. Analyze overlap (which Hebrew matches are explained by JI?)
4. Permutation test: shuffle mapping, re-decode, measure JI match rate

Evidence from manuscripts (Bos & Mensching, Vatican Ebr., Paris BnF 1181):
- ha-menton (mento), ha-gotso (gozzo), tsefaliqa (cefalica)
- Conventions: tet for t, qof for c dura, tsade for c dolce/z, samekh for s
"""

import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ToolkitConfig
from .cross_language_baseline import (
    MAPPING,
    decode_to_hebrew,
    preprocess_eva,
)
from .permutation_stats import (
    build_full_mapping,
    decode_eva_with_mapping,
    generate_random_mapping,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Transliteration: Italian → Hebrew (Judeo-Italian conventions)
# =====================================================================

def strip_accents(s: str) -> str:
    """Remove diacritics (à→a, è→e, etc.)."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


# Single consonant map: Italian → Hebrew ASCII
_CONSONANT_MAP = {
    'b': 'b', 'd': 'd', 'f': 'p', 'l': 'l',
    'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r',
    's': 's', 't': 'J', 'v': 'b', 'z': 'C',
}

# Vowel map: Italian vowel → mater lectionis
_VOWEL_MAP = {
    'a': 'A', 'e': 'y', 'i': 'y', 'o': 'w', 'u': 'w',
}


def transliterate_to_hebrew(word: str) -> str:
    """Convert an Italian word to Hebrew ASCII via Judeo-Italian conventions.

    Handles digraphs (ch, gh, gn, gli+V, sc+e/i, qu), context-dependent
    c/g, consonants, vowels as matres lectionis, and silent h.
    """
    w = strip_accents(word.lower())
    result = []
    i = 0
    n = len(w)

    while i < n:
        # --- Trigraph: gli + vowel → lamed + yod ---
        if (w[i:i + 3] == 'gli' and i + 3 < n
                and w[i + 3] in 'aeiou'):
            result.append('l')
            result.append('y')
            i += 3
            continue

        # --- Digraphs ---
        if i + 1 < n:
            pair = w[i:i + 2]

            if pair == 'ch':
                result.append('k')
                i += 2
                continue

            if pair == 'gh':
                result.append('g')
                i += 2
                continue

            if pair == 'gn':
                result.append('n')
                result.append('y')
                i += 2
                continue

            if pair == 'sc' and i + 2 < n and w[i + 2] in 'ei':
                result.append('S')
                i += 2
                continue

            if pair == 'qu':
                result.append('q')
                result.append('w')
                i += 2
                continue

        c = w[i]

        # --- Context-dependent c ---
        if c == 'c':
            if i + 1 < n and w[i + 1] in 'ei':
                result.append('C')   # tsade for /tʃ/
            else:
                result.append('k')   # kaf for /k/
            i += 1
            continue

        # --- Context-dependent g (gimel for both /g/ and /dʒ/) ---
        if c == 'g':
            result.append('g')
            i += 1
            continue

        # --- Consonants ---
        if c in _CONSONANT_MAP:
            result.append(_CONSONANT_MAP[c])
            i += 1
            continue

        # --- Vowels (matres lectionis) ---
        if c in _VOWEL_MAP:
            result.append(_VOWEL_MAP[c])
            i += 1
            continue

        # --- Silent h ---
        if c == 'h':
            i += 1
            continue

        # --- j → yod (rare in medieval Italian) ---
        if c == 'j':
            result.append('y')
            i += 1
            continue

        # --- x → kaf + samekh (rare) ---
        if c == 'x':
            result.append('k')
            result.append('s')
            i += 1
            continue

        # Unknown character — skip
        i += 1

    return ''.join(result)


# =====================================================================
# Normalization for flexible comparison
# =====================================================================

def normalize_hebrew(heb: str) -> str:
    """Normalize Hebrew for flexible comparison.

    J → t  (tet and tav both = /t/)
    q → k  (qof and kaf both = /k/)
    final h → y  (he and yod both = final vowel)
    """
    if not heb:
        return heb
    result = list(heb)
    for i in range(len(result)):
        if result[i] == 'J':
            result[i] = 't'
        elif result[i] == 'q':
            result[i] = 'k'
    if result and result[-1] == 'h':
        result[-1] = 'y'
    return ''.join(result)


# =====================================================================
# Lexicon construction
# =====================================================================

def build_ji_lexicon(italian_forms: list[str]
                     ) -> tuple[set, set, dict]:
    """Transliterate Italian forms to Hebrew and build JI lexicon.

    Returns:
        strict_set: set of raw Hebrew transliterations
        normalized_set: set of normalized Hebrew transliterations
        source_map: dict {hebrew_form: [italian_words]}
    """
    strict_set = set()
    normalized_set = set()
    source_map: dict[str, list[str]] = {}

    for word in italian_forms:
        heb = transliterate_to_hebrew(word)
        if len(heb) < 3:
            continue
        strict_set.add(heb)
        normalized_set.add(normalize_hebrew(heb))
        source_map.setdefault(heb, []).append(word)

    return strict_set, normalized_set, source_map


# =====================================================================
# Data loading
# =====================================================================

def load_italian_forms(config: ToolkitConfig) -> list[str]:
    """Load Italian word forms from italian_lexicon.json."""
    path = config.lexicon_dir / "italian_lexicon.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    forms = data.get("all_forms", [])
    if isinstance(forms, list) and forms:
        return forms
    # Fallback: extract from by_domain
    result = set()
    by_domain = data.get("by_domain", {})
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            word = entry.get("word", "")
            if word:
                result.add(word.lower())
    return list(result)


def load_hebrew_set(config: ToolkitConfig) -> set[str]:
    """Load Hebrew lexicon as set of consonantal forms."""
    enriched = config.lexicon_dir / "lexicon_enriched.json"
    base = config.lexicon_dir / "lexicon.json"
    path = enriched if enriched.exists() else base
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("all_consonantal_forms", []))


def load_decoded_corpus(config: ToolkitConfig
                        ) -> tuple[list[str], list[str]]:
    """Load decoded Hebrew words and EVA words from full_decode.json.

    Returns: (hebrew_words, eva_words) — parallel lists.
    """
    path = config.stats_dir / "full_decode.json"
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    heb_words = []
    eva_words = []
    for folio, page in data.get("pages", {}).items():
        wh = page.get("words_hebrew", [])
        we = page.get("words_eva", [])
        for h, e in zip(wh, we):
            if h:
                heb_words.append(h)
                eva_words.append(e)
    return heb_words, eva_words


# =====================================================================
# Match rate computation
# =====================================================================

def compute_match_rates(decoded_heb: list[str], ji_strict: set,
                        ji_norm: set, heb_set: set,
                        min_len: int = 3) -> dict:
    """Compute match rates for decoded words against JI and Hebrew."""
    ji_strict_matched = 0
    ji_norm_matched = 0
    heb_matched = 0
    n_total = 0

    for word in decoded_heb:
        if len(word) < min_len:
            continue
        n_total += 1
        if word in ji_strict:
            ji_strict_matched += 1
        norm = normalize_hebrew(word)
        if norm in ji_norm:
            ji_norm_matched += 1
        if word in heb_set:
            heb_matched += 1

    return {
        "n_total": n_total,
        "ji_strict": {
            "n_matched": ji_strict_matched,
            "rate": ji_strict_matched / max(n_total, 1),
        },
        "ji_normalized": {
            "n_matched": ji_norm_matched,
            "rate": ji_norm_matched / max(n_total, 1),
        },
        "hebrew": {
            "n_matched": heb_matched,
            "rate": heb_matched / max(n_total, 1),
        },
    }


# =====================================================================
# Overlap analysis
# =====================================================================

def overlap_analysis(decoded_heb: list[str], ji_norm: set,
                     heb_set: set, ji_source_map: dict,
                     min_len: int = 3) -> dict:
    """Analyze overlap between JI and Hebrew matches."""
    decoded_types = Counter()
    for word in decoded_heb:
        if len(word) >= min_len:
            decoded_types[word] += 1

    both_match = []
    ji_only = []
    heb_only = []
    neither = []

    for word, freq in decoded_types.items():
        norm = normalize_hebrew(word)
        in_ji = norm in ji_norm
        in_heb = word in heb_set

        if in_ji and in_heb:
            both_match.append((word, freq))
        elif in_ji and not in_heb:
            ji_only.append((word, freq))
        elif not in_ji and in_heb:
            heb_only.append((word, freq))
        else:
            neither.append((word, freq))

    # For ji_only, find Italian source words
    ji_only_with_source = []
    for word, freq in sorted(ji_only, key=lambda x: -x[1])[:30]:
        norm = normalize_hebrew(word)
        sources = []
        for heb_form, italian_words in ji_source_map.items():
            if normalize_hebrew(heb_form) == norm:
                sources.extend(italian_words[:3])
        ji_only_with_source.append({
            "decoded": word,
            "freq": freq,
            "italian_source": sources[:3],
        })

    total_heb_types = len(both_match) + len(heb_only)
    return {
        "both_types": len(both_match),
        "both_tokens": sum(f for _, f in both_match),
        "ji_only_types": len(ji_only),
        "ji_only_tokens": sum(f for _, f in ji_only),
        "heb_only_types": len(heb_only),
        "heb_only_tokens": sum(f for _, f in heb_only),
        "neither_types": len(neither),
        "neither_tokens": sum(f for _, f in neither),
        "total_types": len(decoded_types),
        "ji_explains_heb_pct": (
            len(both_match) / max(total_heb_types, 1) * 100
        ),
        "top_ji_only": ji_only_with_source,
    }


# =====================================================================
# Permutation test
# =====================================================================

def run_permutation_test(eva_words: list[str], ji_norm_set: set,
                         n_perms: int = 200, seed: int = 42) -> dict:
    """Permutation test: shuffle mapping, re-decode, measure JI match rate.

    For each permutation:
    1. Generate random bijective EVA→Hebrew mapping (20 keys incl. placeholders)
    2. Decode entire EVA corpus
    3. Normalize each decoded word
    4. Count matches against normalized JI set
    """
    rng = random.Random(seed)

    # Build full real mapping (17 standard + 3 preprocessed-char placeholders)
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
            if normalize_hebrew(heb) in ji_norm_set:
                real_matched += 1
    real_rate = real_matched / max(real_total, 1)

    # Random mappings
    sim_rates = []
    t0 = time.time()

    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      perm {i + 1}/{n_perms}  ({elapsed:.1f}s)")

        rand_map = generate_random_mapping(
            eva_chars=real_keys,
            seed=rng.randint(0, 2**31),
        )

        n_matched = 0
        n_total = 0
        for w in eva_words:
            if len(w) < 3:
                continue
            heb = decode_eva_with_mapping(w, rand_map, mode="hebrew")
            if heb:
                n_total += 1
                if normalize_hebrew(heb) in ji_norm_set:
                    n_matched += 1

        rate = n_matched / max(n_total, 1)
        sim_rates.append(rate)

    sim_arr = np.array(sim_rates)
    sim_mean = float(np.mean(sim_arr))
    sim_std = float(np.std(sim_arr))
    z_score = ((real_rate - sim_mean) / sim_std
               if sim_std > 0 else float("inf"))
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

def format_summary(ji_stats, match_rates, overlap, perm_results,
                   heb_z_from_cross_lang) -> str:
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 65)
    lines.append("  JUDEO-ITALIAN HYPOTHESIS TEST")
    lines.append("=" * 65)

    # 1. Lexicon construction
    lines.append("\n  1. LEXICON CONSTRUCTION")
    lines.append("  " + "-" * 61)
    lines.append(f"  Italian forms loaded:         {ji_stats['n_italian']:,}")
    lines.append(f"  JI Hebrew forms (strict):     {ji_stats['n_ji_strict']:,}")
    lines.append(f"  JI Hebrew forms (normalized): {ji_stats['n_ji_norm']:,}")
    lines.append(f"  Hebrew lexicon forms:         {ji_stats['n_hebrew']:,}")

    lines.append("\n  Sample transliterations:")
    for sample in ji_stats.get("samples", []):
        lines.append(f"    {sample['italian']:20s} -> {sample['hebrew']}")

    # 2. Match rates
    lines.append("\n  2. MATCH RATES")
    lines.append("  " + "-" * 61)
    mr = match_rates
    lines.append(f"  {'':25s} {'Matched':>8s} {'Total':>8s} {'Rate':>8s}")
    lines.append(f"  {'JI strict':25s} {mr['ji_strict']['n_matched']:8,d} "
                 f"{mr['n_total']:8,d} {mr['ji_strict']['rate']*100:7.1f}%")
    lines.append(f"  {'JI normalized':25s} {mr['ji_normalized']['n_matched']:8,d} "
                 f"{mr['n_total']:8,d} {mr['ji_normalized']['rate']*100:7.1f}%")
    lines.append(f"  {'Hebrew lexicon':25s} {mr['hebrew']['n_matched']:8,d} "
                 f"{mr['n_total']:8,d} {mr['hebrew']['rate']*100:7.1f}%")

    # 3. Overlap
    lines.append("\n  3. OVERLAP ANALYSIS")
    lines.append("  " + "-" * 61)
    ov = overlap
    total_heb = ov['both_types'] + ov['heb_only_types']
    lines.append(f"  Hebrew matches also in JI:  {ov['both_types']:,} / "
                 f"{total_heb:,} types ({ov['ji_explains_heb_pct']:.1f}%)")
    lines.append(f"  JI-only matches:            {ov['ji_only_types']:,} types "
                 f"({ov['ji_only_tokens']:,} tokens)")
    lines.append(f"  Hebrew-only matches:        {ov['heb_only_types']:,} types "
                 f"({ov['heb_only_tokens']:,} tokens)")
    lines.append(f"  Neither:                    {ov['neither_types']:,} types")

    if ov.get("top_ji_only"):
        lines.append("\n  Top JI-only matches:")
        for entry in ov["top_ji_only"][:10]:
            src = (", ".join(entry["italian_source"][:2])
                   if entry["italian_source"] else "?")
            lines.append(f"    {entry['decoded']:15s} (freq={entry['freq']:3d})"
                         f" <- {src}")

    # 4. Permutation test
    lines.append("\n  4. PERMUTATION TEST")
    lines.append("  " + "-" * 61)
    pr = perm_results
    lines.append(f"  Permutations:    {pr['n_perms']}")
    lines.append(f"  JI norm real:    {pr['real_rate']*100:.1f}% "
                 f"({pr['real_matched']:,}/{pr['real_total']:,})")
    lines.append(f"  JI norm random:  {pr['sim_mean']*100:.1f}% "
                 f"+/- {pr['sim_std']*100:.1f}%")
    lines.append(f"  JI z-score:      {pr['z_score']:.2f}  "
                 f"(p={pr['p_value']:.4f})")
    lines.append(f"  Hebrew z-score:  {heb_z_from_cross_lang:.1f} "
                 f"(from cross-language)")
    lines.append(f"  Elapsed:         {pr['elapsed_s']:.1f}s")

    # 5. Verdict
    lines.append("\n  5. VERDICT")
    lines.append("  " + "=" * 61)

    ji_z = pr['z_score']
    ji_rate = pr['real_rate']
    heb_rate = match_rates['hebrew']['rate']
    ji_explains = ov['ji_explains_heb_pct']

    if ji_z > 3.0 and ji_rate > heb_rate * 0.5:
        verdict = "JI HYPOTHESIS SUPPORTED"
        detail = (
            f"JI match rate ({ji_rate*100:.1f}%) is significant (z={ji_z:.1f})"
            f" and JI explains {ji_explains:.0f}% of Hebrew matches. "
            f"The Hebrew signal may reflect Italian written in Hebrew script."
        )
    elif ji_z > 2.0:
        verdict = "JI HYPOTHESIS PLAUSIBLE"
        detail = (
            f"JI match rate ({ji_rate*100:.1f}%) is marginally significant "
            f"(z={ji_z:.1f}). JI explains {ji_explains:.0f}% of Hebrew "
            f"matches. Further investigation warranted."
        )
    elif ji_z > 0:
        verdict = "JI HYPOTHESIS WEAK"
        detail = (
            f"JI match rate ({ji_rate*100:.1f}%) exceeds random "
            f"(z={ji_z:.1f}) but not strongly significant. "
            f"JI explains {ji_explains:.0f}% of Hebrew matches."
        )
    else:
        verdict = "JI HYPOTHESIS NOT SUPPORTED"
        detail = (
            f"JI match rate ({ji_rate*100:.1f}%) does not exceed random "
            f"(z={ji_z:.1f}). The Hebrew signal is not explained by JI."
        )

    lines.append(f"  {verdict}")
    lines.append(f"  {detail}")
    lines.append("  " + "=" * 61)

    return "\n".join(lines)


# =====================================================================
# Main entry point
# =====================================================================

def run(config: ToolkitConfig, force=False):
    """Test Judeo-Italian hypothesis."""
    out_json = config.stats_dir / "judeo_italian_test.json"
    out_txt = config.stats_dir / "judeo_italian_test_summary.txt"

    if not force and out_json.exists():
        print(f"  Output exists: {out_json.name} (use --force to re-run)")
        return

    config.ensure_dirs()
    print_header("JUDEO-ITALIAN HYPOTHESIS TEST")

    # 1. Load Italian forms
    print_step("Loading Italian lexicon...")
    italian_forms = load_italian_forms(config)
    print(f"      {len(italian_forms):,} Italian forms")

    if len(italian_forms) < 100:
        print("      ERROR: Italian lexicon too small. "
              "Run: voynich prepare-italian-lexicon")
        return

    # 2. Build JI lexicon
    print_step("Transliterating Italian -> Hebrew (JI conventions)...")
    ji_strict, ji_norm, ji_source_map = build_ji_lexicon(italian_forms)
    print(f"      JI strict forms:     {len(ji_strict):,}")
    print(f"      JI normalized forms: {len(ji_norm):,}")

    # Sample transliterations for verification
    test_words = [
        "rosa", "mento", "cefalica", "ginocchio", "acqua",
        "bagno", "stella", "erba", "figlio", "scienza",
    ]
    samples = []
    for w in test_words:
        heb = transliterate_to_hebrew(w)
        samples.append({"italian": w, "hebrew": heb})
        print(f"      {w:15s} -> {heb}")

    # 3. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    heb_set = load_hebrew_set(config)
    print(f"      {len(heb_set):,} Hebrew consonantal forms")

    # 4. Load decoded corpus
    print_step("Loading decoded corpus from full_decode.json...")
    decoded_heb, eva_words = load_decoded_corpus(config)
    print(f"      {len(decoded_heb):,} decoded words")

    # 5. Compute match rates
    print_step("Computing match rates...")
    match_rates = compute_match_rates(
        decoded_heb, ji_strict, ji_norm, heb_set)
    print(f"      JI strict:     "
          f"{match_rates['ji_strict']['rate']*100:.1f}%")
    print(f"      JI normalized: "
          f"{match_rates['ji_normalized']['rate']*100:.1f}%")
    print(f"      Hebrew:        "
          f"{match_rates['hebrew']['rate']*100:.1f}%")

    # 6. Overlap analysis
    print_step("Overlap analysis...")
    overlap = overlap_analysis(
        decoded_heb, ji_norm, heb_set, ji_source_map)
    print(f"      Both JI+Hebrew: {overlap['both_types']:,} types")
    print(f"      JI-only:        {overlap['ji_only_types']:,} types")
    print(f"      Hebrew-only:    {overlap['heb_only_types']:,} types")
    print(f"      JI explains:    "
          f"{overlap['ji_explains_heb_pct']:.1f}% of Hebrew matches")

    # 7. Permutation test
    print_step("Running permutation test (200 iterations)...")
    perm_results = run_permutation_test(
        eva_words, ji_norm, n_perms=200, seed=42)
    print(f"      JI norm real:   {perm_results['real_rate']*100:.1f}%")
    print(f"      JI norm random: {perm_results['sim_mean']*100:.1f}% "
          f"+/- {perm_results['sim_std']*100:.1f}%")
    print(f"      z-score:        {perm_results['z_score']:.2f} "
          f"(p={perm_results['p_value']:.4f})")

    # 8. Load Hebrew z-score from cross-language for comparison
    heb_z = 0.0
    cl_path = config.stats_dir / "cross_language_report.json"
    if cl_path.exists():
        cl_data = json.loads(cl_path.read_text(encoding="utf-8"))
        heb_vs_rand = (cl_data.get("comparisons", {})
                       .get("hebrew_vs_random", {}))
        heb_z = heb_vs_rand.get("z_score", 0.0)

    # 9. Build summary
    ji_stats = {
        "n_italian": len(italian_forms),
        "n_ji_strict": len(ji_strict),
        "n_ji_norm": len(ji_norm),
        "n_hebrew": len(heb_set),
        "samples": samples,
    }

    summary_txt = format_summary(
        ji_stats, match_rates, overlap, perm_results, heb_z)
    print(f"\n{summary_txt}")

    # 10. Save JSON
    output = {
        "ji_lexicon": {
            "n_italian_forms": len(italian_forms),
            "n_ji_strict": len(ji_strict),
            "n_ji_normalized": len(ji_norm),
            "n_hebrew_lexicon": len(heb_set),
            "samples": samples,
        },
        "match_rates": match_rates,
        "overlap": overlap,
        "permutation_test": perm_results,
        "hebrew_z_from_cross_language": heb_z,
    }
    out_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n  Saved: {out_json.name}")

    # Save TXT
    out_txt.write_text(summary_txt, encoding="utf-8")
    print(f"  Saved: {out_txt.name}")
