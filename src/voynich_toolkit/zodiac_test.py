"""
Zodiac test: match zodiac section labels against sign/month/planet names.

Isolates zodiac-section pages, extracts short candidate words (likely
labels near drawings), and fuzzy-matches against zodiac sign names,
month names, and planet names in Hebrew, Italian, and Latin.
"""
import json
from collections import Counter, defaultdict

import click
from rapidfuzz.distance import Levenshtein

from .config import ToolkitConfig
from .full_decode import (
    SECTION_NAMES,
    decode_word,
    load_convergent_mapping,
)
from .anchor_words import hebrew_to_consonantal, normalize_for_match
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Zodiac vocabulary — signs, months, planets, decans
# =====================================================================

# Each entry: (word, canonical_name)
# Italian words are in medieval Italian orthography

ZODIAC_SIGNS = {
    "hebrew": [
        ("taleh", "Aries"), ("shor", "Taurus"),
        ("teomim", "Gemini"), ("sartan", "Cancer"),
        ("aryeh", "Leo"), ("betulah", "Virgo"),
        ("moznayim", "Libra"), ("akrav", "Scorpio"),
        ("keshet", "Sagittarius"), ("gdi", "Capricorn"),
        ("dli", "Aquarius"), ("dagim", "Pisces"),
    ],
    "italian": [
        ("ariete", "Aries"), ("toro", "Taurus"),
        ("gemini", "Gemini"), ("cancro", "Cancer"),
        ("leone", "Leo"), ("vergine", "Virgo"),
        ("bilancia", "Libra"), ("scorpione", "Scorpio"),
        ("sagittario", "Sagittarius"), ("capricorno", "Capricorn"),
        ("aquario", "Aquarius"), ("pesci", "Pisces"),
    ],
    "latin": [
        ("aries", "Aries"), ("taurus", "Taurus"),
        ("gemini", "Gemini"), ("cancer", "Cancer"),
        ("leo", "Leo"), ("virgo", "Virgo"),
        ("libra", "Libra"), ("scorpio", "Scorpio"),
        ("sagittarius", "Sagittarius"), ("capricornus", "Capricorn"),
        ("aquarius", "Aquarius"), ("pisces", "Pisces"),
    ],
}

MONTHS = {
    "hebrew": [
        ("nisan", "Nisan"), ("iyyar", "Iyyar"),
        ("sivan", "Sivan"), ("tammuz", "Tammuz"),
        ("av", "Av"), ("elul", "Elul"),
        ("tishrei", "Tishrei"), ("cheshvan", "Cheshvan"),
        ("kislev", "Kislev"), ("tevet", "Tevet"),
        ("shevat", "Shevat"), ("adar", "Adar"),
    ],
    "italian": [
        ("gennaio", "January"), ("febbraio", "February"),
        ("marzo", "March"), ("aprile", "April"),
        ("maggio", "May"), ("giugno", "June"),
        ("luglio", "July"), ("agosto", "August"),
        ("settembre", "September"), ("ottobre", "October"),
        ("novembre", "November"), ("dicembre", "December"),
    ],
    "latin": [
        ("januarius", "January"), ("februarius", "February"),
        ("martius", "March"), ("aprilis", "April"),
        ("maius", "May"), ("junius", "June"),
        ("julius", "July"), ("augustus", "August"),
        ("september", "September"), ("october", "October"),
        ("november", "November"), ("december", "December"),
    ],
}

PLANETS = {
    "hebrew": [
        ("shemesh", "Sun"), ("yareach", "Moon"),
        ("maadim", "Mars"), ("kokhav", "Mercury"),
        ("tzedek", "Jupiter"), ("nogah", "Venus"),
        ("shabtai", "Saturn"),
    ],
    "italian": [
        ("sole", "Sun"), ("luna", "Moon"),
        ("marte", "Mars"), ("mercurio", "Mercury"),
        ("giove", "Jupiter"), ("venere", "Venus"),
        ("saturno", "Saturn"),
    ],
    "latin": [
        ("sol", "Sun"), ("luna", "Moon"),
        ("mars", "Mars"), ("mercurius", "Mercury"),
        ("jupiter", "Jupiter"), ("venus", "Venus"),
        ("saturnus", "Saturn"),
    ],
}

ASTRO_TERMS = {
    "hebrew": [
        ("mazzal", "constellation/fortune"), ("machberet", "conjunction"),
        ("neged", "opposition"), ("oleh", "ascendant"),
    ],
    "italian": [
        ("ascendente", "ascendant"), ("congiunzione", "conjunction"),
        ("opposizione", "opposition"),
    ],
    "latin": [
        ("ascendens", "ascendant"), ("coniunctio", "conjunction"),
        ("oppositio", "opposition"),
    ],
}

# Hebrew days of the week
HEBREW_DAYS = [
    ("rishon", "Sunday"), ("sheni", "Monday"),
    ("shlishi", "Tuesday"), ("revii", "Wednesday"),
    ("chamishi", "Thursday"), ("shishi", "Friday"),
    ("shabat", "Shabbat"),
]


# =====================================================================
# Matching
# =====================================================================

def normalize_target(word, language):
    """Normalize a target word based on its language."""
    if language == "hebrew":
        return hebrew_to_consonantal(word)
    else:
        return normalize_for_match(word)


def match_word_against_page(target, page_words, max_dist=1):
    """Match a target against all words on a page.

    Returns list of (decoded_word, eva_word, distance).
    """
    hits = []
    for dec_w, eva_w in page_words:
        if len(dec_w) < 3:
            continue
        if abs(len(dec_w) - len(target)) > max_dist:
            continue
        dist = Levenshtein.distance(target, dec_w, score_cutoff=max_dist)
        if dist <= max_dist:
            hits.append((dec_w, eva_w, dist))
    return hits


def search_zodiac_vocabulary(zodiac_pages, all_pages, max_dist=1):
    """Search zodiac vocabulary against zodiac pages and full manuscript.

    Returns structured results.
    """
    results = {
        "signs": {"label": "Zodiac signs", "matches": []},
        "months": {"label": "Month names", "matches": []},
        "planets": {"label": "Planet names", "matches": []},
        "astro_terms": {"label": "Astrological terms", "matches": []},
        "hebrew_days": {"label": "Hebrew days", "matches": []},
    }

    vocab_sets = [
        ("signs", ZODIAC_SIGNS),
        ("months", MONTHS),
        ("planets", PLANETS),
        ("astro_terms", ASTRO_TERMS),
    ]

    for result_key, vocab in vocab_sets:
        for lang, words in vocab.items():
            word_key = ("words_hebrew" if lang == "hebrew"
                        else "words_decoded")
            for word, canonical in words:
                target = normalize_target(word, lang)
                if len(target) < 3:
                    continue

                # Search in zodiac pages
                zodiac_hits = []
                for folio, pdata in zodiac_pages.items():
                    dec_words = pdata[word_key]
                    eva_words = pdata["words_eva"]
                    pairs = list(zip(dec_words, eva_words))
                    hits = match_word_against_page(target, pairs, max_dist)
                    for dec_w, eva_w, dist in hits:
                        zodiac_hits.append({
                            "folio": folio,
                            "decoded": dec_w,
                            "eva": eva_w,
                            "distance": dist,
                        })

                # Search in all pages (for comparison)
                all_hits_count = 0
                all_hits_sections = Counter()
                for folio, pdata in all_pages.items():
                    dec_words = pdata[word_key]
                    eva_words = pdata["words_eva"]
                    pairs = list(zip(dec_words, eva_words))
                    hits = match_word_against_page(target, pairs, max_dist)
                    for dec_w, eva_w, dist in hits:
                        all_hits_count += 1
                        all_hits_sections[pdata.get("section",
                                                     "unknown")] += 1

                if zodiac_hits or all_hits_count > 0:
                    is_exact = any(h["distance"] == 0 for h in zodiac_hits)
                    results[result_key]["matches"].append({
                        "word": word,
                        "canonical": canonical,
                        "language": lang,
                        "normalized": target,
                        "zodiac_hits": zodiac_hits,
                        "zodiac_count": len(zodiac_hits),
                        "total_manuscript_count": all_hits_count,
                        "section_distribution": dict(
                            all_hits_sections.most_common()),
                        "exact_match_in_zodiac": is_exact,
                    })

    # Hebrew days (only Hebrew)
    for word, canonical in HEBREW_DAYS:
        target = hebrew_to_consonantal(word)
        if len(target) < 3:
            continue
        zodiac_hits = []
        for folio, pdata in zodiac_pages.items():
            pairs = list(zip(pdata["words_hebrew"], pdata["words_eva"]))
            hits = match_word_against_page(target, pairs, max_dist)
            for dec_w, eva_w, dist in hits:
                zodiac_hits.append({
                    "folio": folio, "decoded": dec_w,
                    "eva": eva_w, "distance": dist,
                })
        if zodiac_hits:
            results["hebrew_days"]["matches"].append({
                "word": word, "canonical": canonical,
                "language": "hebrew", "normalized": target,
                "zodiac_hits": zodiac_hits,
                "zodiac_count": len(zodiac_hits),
                "exact_match_in_zodiac": any(
                    h["distance"] == 0 for h in zodiac_hits),
            })

    return results


# =====================================================================
# Short-word analysis (potential labels)
# =====================================================================

def extract_label_candidates(zodiac_pages, mapping, divergent_set, direction):
    """Extract short words (3-7 chars) from zodiac pages as label candidates.

    Groups them by page for analysis.
    """
    candidates = {}
    for folio, pdata in zodiac_pages.items():
        page_candidates = []
        for eva_w, dec_w, heb_w in zip(
            pdata["words_eva"], pdata["words_decoded"],
            pdata["words_hebrew"]
        ):
            if 3 <= len(eva_w) <= 7:
                n_unk = sum(1 for c in dec_w if c.isupper())
                page_candidates.append({
                    "eva": eva_w,
                    "decoded_italian": dec_w,
                    "decoded_hebrew": heb_w,
                    "length": len(eva_w),
                    "has_unknowns": n_unk > 0,
                })
        candidates[folio] = page_candidates
    return candidates


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Zodiac section test: match labels against zodiac vocabulary."""
    report_path = config.stats_dir / "zodiac_test_report.json"

    if report_path.exists() and not force:
        click.echo("  Zodiac test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("ZODIAC TEST — Sign, Month & Planet Matching")

    # 1. Load decoded text
    print_step("Loading decoded text...")
    full_decode_path = config.stats_dir / "full_decode.json"

    if full_decode_path.exists():
        with open(full_decode_path) as f:
            fd = json.load(f)
        all_pages = fd["pages"]
        click.echo(f"    Loaded {len(all_pages)} pages from full_decode.json")
    else:
        click.echo("    full_decode.json not found, decoding on the fly...")
        mapping, divergent, direction = load_convergent_mapping(config)
        eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
        eva_data = parse_eva_words(eva_file)
        divergent_set = set(divergent.keys())
        all_pages = {}
        for page in eva_data["pages"]:
            folio = page["folio"]
            section = page.get("section", "?")
            words_eva = page["words"]
            words_dec = []
            words_heb = []
            for w in words_eva:
                ita, heb, _ = decode_word(
                    w, mapping, divergent_set, direction)
                words_dec.append(ita)
                words_heb.append(heb)
            all_pages[folio] = {
                "section": SECTION_NAMES.get(section, section),
                "section_code": section,
                "words_eva": words_eva,
                "words_decoded": words_dec,
                "words_hebrew": words_heb,
            }

    # 2. Isolate zodiac pages
    print_step("Isolating zodiac section pages...")
    zodiac_pages = {}
    for folio, pdata in all_pages.items():
        section = pdata.get("section", "")
        section_code = pdata.get("section_code", "")
        if section == "zodiac" or section_code == "Z":
            zodiac_pages[folio] = pdata

    click.echo(f"    {len(zodiac_pages)} zodiac pages found")
    if zodiac_pages:
        folii = sorted(zodiac_pages.keys())
        click.echo(f"    Folii: {', '.join(folii[:10])}"
                   f"{'...' if len(folii) > 10 else ''}")
        total_words = sum(
            len(p["words_eva"]) for p in zodiac_pages.values())
        click.echo(f"    Total words in zodiac: {total_words}")

    if not zodiac_pages:
        click.echo("    WARNING: No zodiac pages found!")
        click.echo("    Searching all pages instead...")
        zodiac_pages = all_pages

    # 3. Count vocabulary
    total_vocab = (
        sum(len(v) for v in ZODIAC_SIGNS.values())
        + sum(len(v) for v in MONTHS.values())
        + sum(len(v) for v in PLANETS.values())
        + sum(len(v) for v in ASTRO_TERMS.values())
        + len(HEBREW_DAYS)
    )
    click.echo(f"    {total_vocab} zodiac vocabulary terms to search")

    # 4. Search
    print_step("Searching zodiac vocabulary (Levenshtein <= 1)...")
    results = search_zodiac_vocabulary(zodiac_pages, all_pages, max_dist=1)

    # 5. Collect exact matches (most important!)
    exact_matches = []
    all_zodiac_hits = []
    for group_key, group_data in results.items():
        for match in group_data["matches"]:
            if match.get("exact_match_in_zodiac"):
                exact_matches.append(match)
            if match.get("zodiac_count", 0) > 0:
                all_zodiac_hits.append(match)

    # 6. Label candidates
    print_step("Extracting label candidates (short words 3-7 chars)...")
    label_candidates = extract_label_candidates(
        zodiac_pages, {}, set(), "rtl")  # Uses pre-decoded data
    total_labels = sum(len(v) for v in label_candidates.values())
    click.echo(f"    {total_labels} label candidates across "
               f"{len(label_candidates)} pages")

    # 7. Save report
    print_step("Saving report...")
    report = {
        "zodiac_pages": sorted(zodiac_pages.keys()),
        "n_zodiac_pages": len(zodiac_pages),
        "total_zodiac_words": sum(
            len(p["words_eva"]) for p in zodiac_pages.values()),
        "vocabulary_searched": total_vocab,
        "results": {},
        "exact_matches_in_zodiac": [],
        "all_zodiac_section_hits": [],
        "label_candidates": {},
    }

    for group_key, group_data in results.items():
        report["results"][group_key] = {
            "label": group_data["label"],
            "n_matched": len(group_data["matches"]),
            "matches": group_data["matches"],
        }

    for match in exact_matches:
        report["exact_matches_in_zodiac"].append({
            "word": match["word"],
            "canonical": match["canonical"],
            "language": match["language"],
            "normalized": match["normalized"],
            "hits": match["zodiac_hits"],
        })

    for match in all_zodiac_hits:
        report["all_zodiac_section_hits"].append({
            "word": match["word"],
            "canonical": match["canonical"],
            "language": match["language"],
            "zodiac_count": match["zodiac_count"],
            "total_count": match.get("total_manuscript_count", 0),
        })

    # Top label candidates per page
    for folio, candidates in label_candidates.items():
        report["label_candidates"][folio] = candidates[:20]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")

    # 8. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  ZODIAC TEST RESULTS")
    click.echo(f"{'=' * 60}")

    # EXACT MATCHES — highlighted prominently
    if exact_matches:
        click.echo(f"\n  *** EXACT MATCHES IN ZODIAC SECTION ***")
        for match in exact_matches:
            click.echo(
                f"    {match['canonical']:15s} = "
                f"{match['word']} ({match['language']}) "
                f"-> {match['normalized']}")
            for hit in match["zodiac_hits"]:
                if hit["distance"] == 0:
                    click.echo(
                        f"      EXACT on {hit['folio']}: "
                        f"EVA={hit['eva']} -> {hit['decoded']}")
    else:
        click.echo(f"\n  No exact matches in zodiac section")

    # All zodiac hits
    click.echo(f"\n  Zodiac section matches (exact + fuzzy):")
    for match in sorted(all_zodiac_hits,
                        key=lambda m: -m["zodiac_count"]):
        best_d = min(h["distance"] for h in match["zodiac_hits"])
        marker = "EXACT" if best_d == 0 else f"d={best_d}"
        click.echo(
            f"    {match['canonical']:15s} ({match['language'][:2]:2s}) "
            f"x{match['zodiac_count']} in zodiac, "
            f"x{match.get('total_manuscript_count', '?')} total  "
            f"[{marker}]")

    # Per-group summary
    for group_key, group_data in results.items():
        if group_data["matches"]:
            click.echo(f"\n  {group_data['label']}:")
            for match in group_data["matches"]:
                zodiac_n = match.get("zodiac_count", 0)
                total_n = match.get("total_manuscript_count", 0)
                click.echo(
                    f"    {match['word']:15s} ({match['language'][:2]:2s}) "
                    f"-> {match['normalized']:10s}  "
                    f"zodiac={zodiac_n}, total={total_n}")

    # Label candidates summary
    click.echo(f"\n  Label candidates (short words in zodiac pages):")
    for folio in sorted(label_candidates.keys()):
        candidates = label_candidates[folio]
        if candidates:
            sample = candidates[:5]
            parts = [f"{c['eva']}={c['decoded_italian']}" for c in sample]
            click.echo(f"    {folio}: {len(candidates)} candidates — "
                       f"{', '.join(parts)}...")

    # Verdict
    click.echo(f"\n  {'=' * 40}")
    n_exact = len(exact_matches)
    n_zodiac = len(all_zodiac_hits)
    if n_exact > 0:
        click.echo(f"  VERDICT: {n_exact} EXACT MATCH(ES) FOUND!")
        click.echo("  This is a STRONG validation signal.")
    elif n_zodiac > 5:
        click.echo(f"  VERDICT: {n_zodiac} fuzzy matches in zodiac section")
        click.echo("  Moderate support for the mapping.")
    elif n_zodiac > 0:
        click.echo(f"  VERDICT: {n_zodiac} fuzzy matches in zodiac section")
        click.echo("  Weak support — needs more investigation.")
    else:
        click.echo("  VERDICT: No zodiac vocabulary matches found.")
        click.echo("  The zodiac section may use different conventions.")
    click.echo(f"  {'=' * 40}")
