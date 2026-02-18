"""Phase 10: Multi-tier resolution of unknown decoded Hebrew words.

Tier 1 — Lexicon re-check (unknown vs 491K set)
Tier 2 — Morphological decomposition (prefix/suffix stripping)
Tier 3 — Compound splitting (both halves in lexicon)
Tier 4 — Fuzzy matching (d=1 against glossed forms)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .config import ToolkitConfig
from .fuzzy_utils import LengthBucketedIndex
from .utils import print_header, print_step

# ── Hebrew morphological affixes ────────────────────────────────

SINGLE_PREFIXES = {
    "b": "in",
    "k": "like",
    "l": "to",
    "m": "from",
    "w": "and",
    "S": "that",
}
COMBINED_PREFIXES = {
    "wb": "and-in",
    "wk": "and-like",
    "wl": "and-to",
    "wm": "and-from",
    "wS": "and-that",
}
# Longest first for greedy matching
ALL_PREFIXES = {**COMBINED_PREFIXES, **SINGLE_PREFIXES}

SUFFIXES = {
    "ym": "pl-m",
    "wt": "pl-f",
    "h": "dir/-ah",
    "w": "his",
    "t": "-at",
    "m": "-am",
    "n": "-an",
}

MIN_STEM = 2
MIN_COMPOUND_PART = 2


# ── Data loading ────────────────────────────────────────────────


def load_data(config: ToolkitConfig):
    """Load all data needed for phrase completion.

    Returns (word_freqs, lexicon_set, form_to_gloss, by_consonants, sefaria_freqs).
    """
    # Word frequencies from full decode
    decode_path = config.stats_dir / "full_decode.json"
    with open(decode_path) as f:
        decode_data = json.load(f)

    word_freqs: Counter = Counter()
    for page_data in decode_data["pages"].values():
        for w in page_data.get("words_hebrew", []):
            if w:
                word_freqs[w] += 1

    # Enriched lexicon
    lex_path = config.lexicon_dir / "lexicon_enriched.json"
    with open(lex_path) as f:
        lex_data = json.load(f)

    lexicon_set = set(lex_data["all_consonantal_forms"])
    form_to_gloss = lex_data["form_to_gloss"]  # 28K with real glosses
    by_consonants = lex_data["by_consonants"]  # 491K

    # Sefaria corpus frequencies
    sefaria_freqs: dict[str, int] = {}
    sefaria_path = config.lexicon_dir / "sefaria_corpus.json"
    if sefaria_path.exists():
        with open(sefaria_path) as f:
            sf_data = json.load(f)
        sefaria_freqs = sf_data.get("forms", {})

    return word_freqs, lexicon_set, form_to_gloss, by_consonants, sefaria_freqs


# ── Corpus classification ───────────────────────────────────────


def classify_corpus(word_freqs, lexicon_set, form_to_gloss):
    """Split decoded words into glossed / attested / unknown.

    Returns three dicts {word: freq}.
    """
    glossed = {}
    attested = {}
    unknown = {}

    for word, freq in word_freqs.items():
        if word in form_to_gloss:
            glossed[word] = freq
        elif word in lexicon_set:
            attested[word] = freq
        else:
            unknown[word] = freq

    return glossed, attested, unknown


# ── Tier 1: Lexicon re-check ───────────────────────────────────


def resolve_tier1(unknown, lexicon_set, by_consonants, sefaria_freqs):
    """Re-check unknown words against full lexicon (catch discrepancies)."""
    resolved = {}
    for word in unknown:
        if word in lexicon_set:
            entries = by_consonants.get(word, [])
            source = entries[0]["source"] if entries else "unknown"
            resolved[word] = {
                "tier": 1,
                "source": source,
                "sefaria_freq": sefaria_freqs.get(word, 0),
            }
    return resolved


# ── Tier 2: Morphological decomposition ────────────────────────


def _try_morphology(word, lexicon_set, form_to_gloss, sefaria_freqs):
    """Try prefix stripping, suffix stripping, then both.

    Returns resolution dict or None.
    """
    # Prefix only (longest first)
    for pfx in sorted(ALL_PREFIXES, key=len, reverse=True):
        if word.startswith(pfx):
            stem = word[len(pfx) :]
            if len(stem) >= MIN_STEM and stem in lexicon_set:
                return {
                    "tier": 2,
                    "prefix": pfx,
                    "prefix_meaning": ALL_PREFIXES[pfx],
                    "suffix": "",
                    "suffix_meaning": "",
                    "stem": stem,
                    "stem_gloss": form_to_gloss.get(stem, ""),
                    "stem_in_gloss": stem in form_to_gloss,
                    "sefaria_freq": sefaria_freqs.get(stem, 0),
                }

    # Suffix only (longest first)
    for sfx in sorted(SUFFIXES, key=len, reverse=True):
        if word.endswith(sfx):
            stem = word[: -len(sfx)]
            if len(stem) >= MIN_STEM and stem in lexicon_set:
                return {
                    "tier": 2,
                    "prefix": "",
                    "prefix_meaning": "",
                    "suffix": sfx,
                    "suffix_meaning": SUFFIXES[sfx],
                    "stem": stem,
                    "stem_gloss": form_to_gloss.get(stem, ""),
                    "stem_in_gloss": stem in form_to_gloss,
                    "sefaria_freq": sefaria_freqs.get(stem, 0),
                }

    # Prefix + suffix combined
    for pfx in sorted(ALL_PREFIXES, key=len, reverse=True):
        if not word.startswith(pfx):
            continue
        rest = word[len(pfx) :]
        for sfx in sorted(SUFFIXES, key=len, reverse=True):
            if rest.endswith(sfx):
                stem = rest[: -len(sfx)]
                if len(stem) >= MIN_STEM and stem in lexicon_set:
                    return {
                        "tier": 2,
                        "prefix": pfx,
                        "prefix_meaning": ALL_PREFIXES[pfx],
                        "suffix": sfx,
                        "suffix_meaning": SUFFIXES[sfx],
                        "stem": stem,
                        "stem_gloss": form_to_gloss.get(stem, ""),
                        "stem_in_gloss": stem in form_to_gloss,
                        "sefaria_freq": sefaria_freqs.get(stem, 0),
                    }

    return None


def resolve_tier2(remaining, lexicon_set, form_to_gloss, sefaria_freqs):
    """Morphological decomposition of remaining unknown words."""
    resolved = {}
    for word in remaining:
        result = _try_morphology(word, lexicon_set, form_to_gloss, sefaria_freqs)
        if result:
            resolved[word] = result
    return resolved


# ── Tier 3: Compound splitting ──────────────────────────────────


def resolve_tier3(remaining, lexicon_set, form_to_gloss, sefaria_freqs):
    """Split words into two parts, both in lexicon."""
    resolved = {}

    for word in remaining:
        if len(word) < MIN_COMPOUND_PART * 2:
            continue

        splits = []
        for i in range(MIN_COMPOUND_PART, len(word) - MIN_COMPOUND_PART + 1):
            left, right = word[:i], word[i:]
            if left in lexicon_set and right in lexicon_set:
                # Score: 3 if both glossed, 2 if one, 1 if both attested-only
                left_glossed = left in form_to_gloss
                right_glossed = right in form_to_gloss
                score = (1 + left_glossed) + (1 + right_glossed) - 1
                # Frequency bonus
                freq_l = sefaria_freqs.get(left, 1)
                freq_r = sefaria_freqs.get(right, 1)
                score += math.log10(max(freq_l, 1) * max(freq_r, 1)) * 0.1

                splits.append(
                    {
                        "left": left,
                        "right": right,
                        "left_gloss": form_to_gloss.get(left, ""),
                        "right_gloss": form_to_gloss.get(right, ""),
                        "score": round(score, 3),
                    }
                )

        if splits:
            splits.sort(key=lambda s: s["score"], reverse=True)
            resolved[word] = {"tier": 3, "splits": splits[:3]}

    return resolved


# ── Tier 4: Fuzzy matching (d=1) ────────────────────────────────


def resolve_tier4(remaining, form_to_gloss, sefaria_freqs):
    """Fuzzy match remaining words against glossed forms (d<=1)."""
    print_step("Building fuzzy index on 28K glossed forms...")
    index = LengthBucketedIndex(
        forms=form_to_gloss.keys(), form_to_gloss=form_to_gloss
    )

    resolved = {}
    for word in remaining:
        if len(word) < 3:
            continue
        matches = index.query(word, max_dist=1)
        if matches:
            resolved[word] = {
                "tier": 4,
                "matches": [
                    {
                        "target": m.target,
                        "distance": m.distance,
                        "gloss": m.gloss,
                    }
                    for m in matches[:3]
                ],
            }

    return resolved


# ── Phrase enrichment ───────────────────────────────────────────


def build_enriched_phrases(config, resolutions, glossed, attested):
    """Annotate contextual phrases with tier resolution info.

    Returns list of enriched phrase dicts with old/new known_ratio.
    """
    phrases_path = config.stats_dir / "contextual_phrases.json"
    if not phrases_path.exists():
        return []

    with open(phrases_path) as f:
        phrases = json.load(f)

    # Build combined known set
    known_set = set(glossed) | set(attested)
    resolved_set = set(resolutions)

    enriched = []
    for phrase in phrases:
        context = phrase.get("context_heb", [])
        n_total = len(context)
        if n_total == 0:
            continue

        old_known = sum(1 for w in context if w in known_set)
        new_known = sum(
            1 for w in context if w in known_set or w in resolved_set
        )

        if new_known <= old_known:
            continue

        # Annotate each word
        annotated = []
        for w in context:
            if w in glossed:
                annotated.append({"word": w, "status": "glossed"})
            elif w in attested:
                annotated.append({"word": w, "status": "attested"})
            elif w in resolutions:
                r = resolutions[w]
                annotated.append(
                    {"word": w, "status": f"tier{r['tier']}", "resolution": r}
                )
            else:
                annotated.append({"word": w, "status": "unknown"})

        enriched.append(
            {
                "page": phrase.get("page", ""),
                "section": phrase.get("section", ""),
                "context_eva": phrase.get("context_eva", []),
                "context_heb": context,
                "annotated": annotated,
                "old_known": old_known,
                "new_known": new_known,
                "n_total": n_total,
                "old_ratio": round(old_known / n_total, 3),
                "new_ratio": round(new_known / n_total, 3),
            }
        )

    enriched.sort(key=lambda p: p["new_ratio"], reverse=True)
    return enriched


# ── Stats and summary ───────────────────────────────────────────


def compute_stats(glossed, attested, unknown, tier_results, word_freqs):
    """Compute summary statistics."""
    total_types = len(word_freqs)
    total_tokens = sum(word_freqs.values())

    glossed_tokens = sum(glossed.values())
    attested_tokens = sum(attested.values())
    unknown_tokens = sum(unknown.values())

    # Tier token counts
    tier_token_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for word, res in tier_results.items():
        tier = res["tier"]
        tier_type_counts[tier] += 1
        tier_token_counts[tier] += word_freqs.get(word, 0)

    resolved_types = sum(tier_type_counts.values())
    resolved_tokens = sum(tier_token_counts.values())

    old_known_types = len(glossed) + len(attested)
    old_known_tokens = glossed_tokens + attested_tokens
    new_known_types = old_known_types + resolved_types
    new_known_tokens = old_known_tokens + resolved_tokens

    return {
        "total_types": total_types,
        "total_tokens": total_tokens,
        "glossed_types": len(glossed),
        "glossed_tokens": glossed_tokens,
        "attested_types": len(attested),
        "attested_tokens": attested_tokens,
        "unknown_types": len(unknown),
        "unknown_tokens": unknown_tokens,
        "tier_type_counts": tier_type_counts,
        "tier_token_counts": tier_token_counts,
        "resolved_types": resolved_types,
        "resolved_tokens": resolved_tokens,
        "old_known_types": old_known_types,
        "old_known_tokens": old_known_tokens,
        "old_known_pct": round(100.0 * old_known_tokens / total_tokens, 1),
        "new_known_types": new_known_types,
        "new_known_tokens": new_known_tokens,
        "new_known_pct": round(100.0 * new_known_tokens / total_tokens, 1),
    }


def format_summary(stats, tier_results, enriched_phrases):
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PHRASE COMPLETION — Multi-tier Resolution")
    lines.append("=" * 60)

    lines.append("\n── Corpus Classification ──")
    lines.append(f"  Total types:    {stats['total_types']:>7,}")
    lines.append(f"  Total tokens:   {stats['total_tokens']:>7,}")
    lines.append(
        f"  Glossed:        {stats['glossed_types']:>7,} types "
        f"({stats['glossed_tokens']:,} tokens)"
    )
    lines.append(
        f"  Attested:       {stats['attested_types']:>7,} types "
        f"({stats['attested_tokens']:,} tokens)"
    )
    lines.append(
        f"  Unknown:        {stats['unknown_types']:>7,} types "
        f"({stats['unknown_tokens']:,} tokens)"
    )

    lines.append("\n── Tier Resolution ──")
    for tier in range(1, 5):
        tier_name = {
            1: "Lexicon re-check",
            2: "Morphological",
            3: "Compound split",
            4: "Fuzzy (d=1)",
        }[tier]
        tc = stats["tier_type_counts"][tier]
        tk = stats["tier_token_counts"][tier]
        lines.append(f"  Tier {tier} ({tier_name:16s}): {tc:>5,} types ({tk:,} tokens)")

    lines.append(f"\n  Total resolved:   {stats['resolved_types']:>5,} types "
                 f"({stats['resolved_tokens']:,} tokens)")

    lines.append("\n── Coverage Improvement ──")
    lines.append(
        f"  Before: {stats['old_known_types']:,} types, "
        f"{stats['old_known_tokens']:,} tokens ({stats['old_known_pct']}%)"
    )
    lines.append(
        f"  After:  {stats['new_known_types']:,} types, "
        f"{stats['new_known_tokens']:,} tokens ({stats['new_known_pct']}%)"
    )
    delta = stats["new_known_pct"] - stats["old_known_pct"]
    lines.append(f"  Delta:  +{delta:.1f}pp")

    # Phrase enrichment
    if enriched_phrases:
        full_before = sum(
            1 for p in enriched_phrases if p["old_ratio"] >= 1.0
        )
        full_after = sum(
            1 for p in enriched_phrases if p["new_ratio"] >= 1.0
        )
        lines.append(f"\n── Enriched Phrases ──")
        lines.append(f"  Phrases improved:     {len(enriched_phrases):,}")
        lines.append(f"  Fully known (before): {full_before}")
        lines.append(f"  Fully known (after):  {full_after}")

        # Top 10 best enriched phrases
        lines.append("\n  Top 10 enriched phrases:")
        for p in enriched_phrases[:10]:
            ctx = " ".join(p["context_heb"])
            lines.append(
                f"    [{p['page']:>5s}] {p['old_ratio']:.0%} → {p['new_ratio']:.0%}  {ctx}"
            )

    # Top tier-2 examples
    tier2_examples = [
        (w, r)
        for w, r in tier_results.items()
        if r["tier"] == 2 and r.get("stem_in_gloss")
    ]
    tier2_examples.sort(key=lambda x: -word_freqs_global.get(x[0], 0))
    if tier2_examples:
        lines.append("\n── Top Tier-2 Decompositions (glossed stem) ──")
        for w, r in tier2_examples[:15]:
            pfx = r["prefix"]
            sfx = r["suffix"]
            stem = r["stem"]
            gloss = r["stem_gloss"][:50]
            freq = word_freqs_global.get(w, 0)
            parts = []
            if pfx:
                parts.append(f"{pfx}({r['prefix_meaning']})")
            parts.append(stem)
            if sfx:
                parts.append(f"-{sfx}({r['suffix_meaning']})")
            decomp = "+".join(parts)
            lines.append(f"    {w:>12s} x{freq:<5d} = {decomp:30s} → {gloss}")

    # Top tier-3 examples
    tier3_examples = [
        (w, r) for w, r in tier_results.items() if r["tier"] == 3
    ]
    tier3_examples.sort(key=lambda x: -x[1]["splits"][0]["score"])
    if tier3_examples:
        lines.append("\n── Top Tier-3 Compound Splits ──")
        for w, r in tier3_examples[:15]:
            s = r["splits"][0]
            lg = s["left_gloss"][:25] if s["left_gloss"] else "?"
            rg = s["right_gloss"][:25] if s["right_gloss"] else "?"
            freq = word_freqs_global.get(w, 0)
            lines.append(
                f"    {w:>12s} x{freq:<5d} = {s['left']}({lg}) + {s['right']}({rg})  [{s['score']:.1f}]"
            )

    return "\n".join(lines)


# Module-level ref for format_summary to access
word_freqs_global: Counter = Counter()


# ── Main entry point ────────────────────────────────────────────


def run(config: ToolkitConfig, force: bool = False):
    """Run multi-tier phrase completion."""
    global word_freqs_global

    out_json = config.stats_dir / "phrase_completion.json"
    out_txt = config.stats_dir / "phrase_completion_summary.txt"

    if not force and out_json.exists():
        print(f"  ⏭  {out_json} exists (use --force)")
        return

    print_header("Phrase Completion — Multi-tier Resolution")

    # ── Load data ──
    print_step("Loading data...")
    word_freqs, lexicon_set, form_to_gloss, by_consonants, sefaria_freqs = load_data(
        config
    )
    word_freqs_global = word_freqs
    print(f"      Decoded words: {len(word_freqs):,} types, {sum(word_freqs.values()):,} tokens")
    print(f"      Lexicon: {len(lexicon_set):,} forms, glossed: {len(form_to_gloss):,}")

    # ── Classify ──
    print_step("Classifying corpus...")
    glossed, attested, unknown = classify_corpus(word_freqs, lexicon_set, form_to_gloss)
    print(f"      Glossed: {len(glossed):,}  Attested: {len(attested):,}  Unknown: {len(unknown):,}")

    remaining = set(unknown)
    all_resolutions: dict[str, dict] = {}

    # ── Tier 1 ──
    print_step("Tier 1: Lexicon re-check...")
    tier1 = resolve_tier1(unknown, lexicon_set, by_consonants, sefaria_freqs)
    all_resolutions.update(tier1)
    remaining -= set(tier1)
    print(f"      Resolved: {len(tier1):,} types")

    # ── Tier 2 ──
    print_step("Tier 2: Morphological decomposition...")
    tier2 = resolve_tier2(remaining, lexicon_set, form_to_gloss, sefaria_freqs)
    all_resolutions.update(tier2)
    remaining -= set(tier2)
    glossed_stems = sum(1 for r in tier2.values() if r.get("stem_in_gloss"))
    print(f"      Resolved: {len(tier2):,} types ({glossed_stems} with glossed stem)")

    # ── Tier 3 ──
    print_step("Tier 3: Compound splitting...")
    tier3 = resolve_tier3(remaining, lexicon_set, form_to_gloss, sefaria_freqs)
    all_resolutions.update(tier3)
    remaining -= set(tier3)
    both_glossed = sum(
        1
        for r in tier3.values()
        if r["splits"][0]["left_gloss"] and r["splits"][0]["right_gloss"]
    )
    print(f"      Resolved: {len(tier3):,} types ({both_glossed} both parts glossed)")

    # ── Tier 4 ──
    print_step("Tier 4: Fuzzy matching (d=1)...")
    tier4 = resolve_tier4(remaining, form_to_gloss, sefaria_freqs)
    all_resolutions.update(tier4)
    remaining -= set(tier4)
    print(f"      Resolved: {len(tier4):,} types")

    print(f"\n      Still unknown: {len(remaining):,} types")

    # ── Phrase enrichment ──
    print_step("Enriching phrases...")
    enriched_phrases = build_enriched_phrases(
        config, all_resolutions, glossed, attested
    )
    print(f"      Phrases improved: {len(enriched_phrases):,}")

    # ── Stats ──
    stats = compute_stats(glossed, attested, unknown, all_resolutions, word_freqs)

    # ── Output ──
    print_step("Writing output...")
    output = {
        "stats": stats,
        "resolutions": all_resolutions,
        "enriched_phrases": enriched_phrases[:500],  # cap for file size
        "still_unknown_types": len(remaining),
    }
    config.ensure_dirs()
    with open(out_json, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    summary = format_summary(stats, all_resolutions, enriched_phrases)
    with open(out_txt, "w") as f:
        f.write(summary)

    print(f"\n      → {out_json}")
    print(f"      → {out_txt}")
    print(f"\n      Coverage: {stats['old_known_pct']}% → {stats['new_known_pct']}% "
          f"(+{stats['new_known_pct'] - stats['old_known_pct']:.1f}pp)")
