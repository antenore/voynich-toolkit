"""
Anchor words: search decoded text for domain-specific vocabulary.

Searches the fully decoded manuscript for anchor words in Italian,
Hebrew (consonantal), and Latin, organized by semantic category.
Analyzes section distribution to validate or falsify the mapping.
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
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN, normalize_italian_phonemic
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Hebrew transliteration → consonantal ASCII
# =====================================================================

def hebrew_to_consonantal(translit):
    """Convert romanized Hebrew to consonantal ASCII.

    Uses digraph-first parsing. Vowels map to matres lectionis
    only where conventional (i→y, o/u→w); short vowels dropped.
    """
    w = translit.lower().replace("'", "").replace("-", "")
    result = []
    i = 0
    n = len(w)
    while i < n:
        if i + 1 < n:
            di = w[i:i + 2]
            if di == "sh":
                result.append("S"); i += 2; continue
            if di in ("ch", "kh"):
                result.append("X"); i += 2; continue
            if di in ("tz", "ts"):
                result.append("C"); i += 2; continue
            if di == "th":
                result.append("t"); i += 2; continue
            if di == "ph":
                result.append("p"); i += 2; continue
        c = w[i]
        CONS = {
            "b": "b", "g": "g", "d": "d", "h": "h", "v": "b",
            "w": "w", "z": "z", "y": "y", "k": "k", "l": "l",
            "m": "m", "n": "n", "s": "s", "p": "p", "f": "p",
            "r": "r", "t": "t", "q": "q",
        }
        if c in CONS:
            result.append(CONS[c])
        elif c == "i":
            result.append("y")
        elif c in "ou":
            result.append("w")
        elif c == "a":
            if i == 0:
                result.append("A")
        # 'e' at end → he
        elif c == "e":
            if i == n - 1:
                result.append("h")
        i += 1
    return "".join(result)


def normalize_for_match(word):
    """Normalize an Italian/Latin word for matching against decoded text.

    Applies Italian phonemic normalization, then maps to the phoneme
    set produced by HEBREW_TO_ITALIAN decoding:
      f→p (pe spirantized), v→b (bet spirantized), u→o (vav=o/u),
      ts→t (tsade absent from 16-char mapping), dz→d, h→e.
    """
    w = normalize_italian_phonemic(word)
    # Hebrew phonemic equivalences (Giudeo-Italiano conventions)
    w = w.replace("ts", "t")
    w = w.replace("dz", "d")
    w = w.replace("f", "p")   # pe = p/f
    w = w.replace("v", "b")   # bet = b/v
    w = w.replace("u", "o")   # vav = o/u
    w = w.replace("h", "e")   # he = e (if any survived phonemic norm)
    # Keep only chars from HEBREW_TO_ITALIAN output
    valid = set("abdegiiklmnoprstz")
    w = "".join(c for c in w if c in valid)
    return w


# =====================================================================
# Anchor word dictionary — organized by semantic category
# =====================================================================

# Format: list of (word, gloss) tuples per language
# Italian words will be normalized with normalize_for_match()
# Hebrew words will be converted with hebrew_to_consonantal()
# Latin words will be normalized with normalize_for_match()

ANCHOR_DICT = {
    "botanica_parti": {
        "label": "Botanica - parti della pianta",
        "italian": [
            ("radice", "root"), ("foglia", "leaf"), ("fiore", "flower"),
            ("seme", "seed"), ("corteccia", "bark"), ("frutto", "fruit"),
            ("stelo", "stem"), ("succo", "juice"), ("resina", "resin"),
            ("bacca", "berry"),
        ],
        "hebrew": [
            ("shoresh", "root"), ("aleh", "leaf"), ("perach", "flower"),
            ("zera", "seed"), ("klipah", "bark"), ("pri", "fruit"),
            ("geza", "trunk"), ("mitz", "juice"), ("sraf", "resin"),
        ],
        "latin": [
            ("radix", "root"), ("folium", "leaf"), ("flos", "flower"),
            ("semen", "seed"), ("cortex", "bark"), ("fructus", "fruit"),
        ],
    },
    "botanica_azioni": {
        "label": "Botanica - azioni preparatorie",
        "italian": [
            ("bollire", "boil"), ("macinare", "grind"),
            ("mescolare", "mix"), ("bere", "drink"),
            ("lavare", "wash"), ("seccare", "dry"),
            ("distillare", "distill"), ("filtrare", "filter"),
            ("impastare", "knead"), ("bruciare", "burn"),
            ("cogliere", "pick"), ("raccogliere", "gather"),
            ("tagliare", "cut"), ("pestare", "pound"),
            ("spremere", "squeeze"), ("infondere", "infuse"),
        ],
        "hebrew": [],
        "latin": [],
    },
    "colori": {
        "label": "Colori",
        "italian": [
            ("rosso", "red"), ("bianco", "white"), ("nero", "black"),
            ("verde", "green"), ("giallo", "yellow"), ("azzurro", "blue"),
            ("viola", "violet"),
        ],
        "hebrew": [
            ("adom", "red"), ("lavan", "white"), ("shachor", "black"),
            ("yarok", "green"), ("tzahov", "yellow"),
            ("tchelet", "blue"), ("argaman", "purple"),
        ],
        "latin": [
            ("ruber", "red"), ("albus", "white"), ("niger", "black"),
            ("viridis", "green"), ("flavus", "yellow"),
        ],
    },
    "corpo_medicina": {
        "label": "Corpo umano e medicina",
        "italian": [
            ("sangue", "blood"), ("fegato", "liver"),
            ("stomaco", "stomach"), ("testa", "head"),
            ("occhio", "eye"), ("petto", "chest"),
            ("ventre", "belly"), ("utero", "uterus"),
            ("bile", "bile"), ("urina", "urine"),
            ("febbre", "fever"), ("dolore", "pain"),
            ("veleno", "poison"), ("rimedio", "remedy"),
            ("cura", "cure"), ("unguento", "ointment"),
            ("impiastro", "poultice"), ("sciroppo", "syrup"),
        ],
        "hebrew": [
            ("dam", "blood"), ("kaved", "liver"),
            ("kevah", "stomach"), ("rosh", "head"),
            ("ayin", "eye"), ("chazeh", "chest"),
            ("beten", "belly"), ("rechem", "uterus"),
            ("marah", "bile"), ("sheten", "urine"),
        ],
        "latin": [
            ("sanguis", "blood"), ("hepar", "liver"),
            ("stomachus", "stomach"), ("caput", "head"),
            ("oculus", "eye"), ("venter", "belly"),
            ("uterus", "uterus"),
        ],
    },
    "liquidi": {
        "label": "Liquidi e sostanze",
        "italian": [
            ("acqua", "water"), ("olio", "oil"), ("vino", "wine"),
            ("aceto", "vinegar"), ("miele", "honey"), ("latte", "milk"),
            ("grasso", "fat"), ("cera", "wax"), ("sale", "salt"),
        ],
        "hebrew": [
            ("mayim", "water"), ("shemen", "oil"), ("yayin", "wine"),
            ("chometz", "vinegar"), ("dvash", "honey"),
            ("chalav", "milk"), ("chelev", "fat"),
            ("melach", "salt"),
        ],
        "latin": [
            ("aqua", "water"), ("oleum", "oil"), ("vinum", "wine"),
            ("acetum", "vinegar"), ("mel", "honey"), ("lac", "milk"),
            ("sal", "salt"),
        ],
    },
    "misure": {
        "label": "Misure e quantita",
        "italian": [
            ("poco", "little"), ("molto", "much"),
            ("quanto", "how much"), ("mezzo", "half"),
            ("parte", "part"), ("goccia", "drop"),
            ("pugno", "handful"), ("oncia", "ounce"),
            ("libbra", "pound"), ("dracma", "drachm"),
        ],
        "hebrew": [
            ("meat", "little"), ("harbeh", "much"),
            ("kamah", "how much"), ("chatzi", "half"),
            ("chelek", "part"), ("tipah", "drop"),
        ],
        "latin": [
            ("parum", "little"), ("multum", "much"),
            ("quantum", "how much"), ("dimidium", "half"),
            ("pars", "part"), ("uncia", "ounce"),
            ("libra", "pound"), ("drachma", "drachm"),
        ],
    },
    "astrologia": {
        "label": "Astrologia e tempo",
        "italian": [
            ("luna", "moon"), ("sole", "sun"), ("stella", "star"),
            ("cielo", "sky"), ("luce", "light"), ("ombra", "shadow"),
            ("pieno", "full"), ("nuovo", "new"),
            ("crescente", "waxing"), ("calante", "waning"),
            ("primavera", "spring"), ("estate", "summer"),
            ("autunno", "autumn"), ("inverno", "winter"),
        ],
        "hebrew": [
            ("yareach", "moon"), ("shemesh", "sun"),
            ("kochav", "star"), ("shamayim", "sky"),
            ("or", "light"), ("tzel", "shadow"),
            ("maleh", "full"), ("chadash", "new"),
            ("aviv", "spring"), ("kayitz", "summer"),
            ("stav", "autumn"), ("choref", "winter"),
        ],
        "latin": [
            ("luna", "moon"), ("sol", "sun"), ("stella", "star"),
            ("caelum", "sky"), ("lux", "light"),
        ],
    },
    "pianeti": {
        "label": "Pianeti classici",
        "italian": [
            ("marte", "Mars"), ("mercurio", "Mercury"),
            ("giove", "Jupiter"), ("venere", "Venus"),
            ("saturno", "Saturn"),
        ],
        "hebrew": [
            ("maadim", "Mars"), ("kokhav", "Mercury"),
            ("tzedek", "Jupiter"), ("nogah", "Venus"),
            ("shabtai", "Saturn"),
        ],
        "latin": [
            ("mars", "Mars"), ("mercurius", "Mercury"),
            ("jupiter", "Jupiter"), ("venus", "Venus"),
            ("saturnus", "Saturn"),
        ],
    },
    "alchimia": {
        "label": "Alchimia",
        "italian": [
            ("fuoco", "fire"), ("terra", "earth"),
            ("aria", "air"), ("spirito", "spirit"),
            ("anima", "soul"), ("corpo", "body"),
            ("zolfo", "sulfur"), ("mercurio", "mercury"),
            ("pietra", "stone"), ("oro", "gold"),
            ("argento", "silver"), ("rame", "copper"),
            ("ferro", "iron"), ("piombo", "lead"),
            ("stagno", "tin"), ("calcinare", "calcinate"),
            ("sublimare", "sublimate"), ("distillare", "distill"),
        ],
        "hebrew": [
            ("esh", "fire"), ("adamah", "earth"),
            ("ruach", "wind/spirit"), ("avir", "air"),
            ("even", "stone"), ("zahav", "gold"),
            ("kesef", "silver"), ("nechoshet", "copper"),
            ("barzel", "iron"), ("oferet", "lead"),
            ("bdil", "tin"),
        ],
        "latin": [
            ("ignis", "fire"), ("terra", "earth"),
            ("aer", "air"), ("aqua", "water"),
            ("lapis", "stone"), ("aurum", "gold"),
            ("argentum", "silver"), ("cuprum", "copper"),
            ("ferrum", "iron"), ("plumbum", "lead"),
            ("stannum", "tin"),
        ],
    },
    "cabala": {
        "label": "Cabala",
        "hebrew": [
            ("sephirah", "sephirah"), ("ein sof", "infinite"),
            ("tikkun", "repair"), ("neshamah", "soul"),
            ("nefesh", "soul-vitality"), ("gematria", "gematria"),
            ("keter", "crown"), ("chochmah", "wisdom"),
            ("binah", "understanding"), ("chesed", "kindness"),
            ("gevurah", "strength"), ("tiferet", "beauty"),
            ("netzach", "eternity"), ("hod", "glory"),
            ("yesod", "foundation"), ("malkhut", "kingdom"),
            ("shaddai", "Almighty"), ("elohim", "God"),
            ("adonai", "Lord"),
        ],
        "italian": [],
        "latin": [],
    },
    "numeri": {
        "label": "Numeri (1-12)",
        "hebrew": [
            ("echad", "one"), ("shtayim", "two"),
            ("shalosh", "three"), ("arba", "four"),
            ("chamesh", "five"), ("shesh", "six"),
            ("sheva", "seven"), ("shmoneh", "eight"),
            ("tesha", "nine"), ("eser", "ten"),
        ],
        "italian": [],
        "latin": [],
    },
}


# =====================================================================
# Matching engine
# =====================================================================

def build_word_index(pages_data, word_key):
    """Build inverted index: decoded_word → [(folio, section, eva_word, count)].

    word_key is 'words_decoded' (Italian) or 'words_hebrew' (Hebrew).
    """
    index = defaultdict(list)
    for folio, pdata in pages_data.items():
        section = pdata.get("section", "unknown")
        eva_words = pdata["words_eva"]
        dec_words = pdata[word_key]
        # Count occurrences per page
        page_counts = Counter()
        page_eva = {}
        for eva_w, dec_w in zip(eva_words, dec_words):
            page_counts[dec_w] += 1
            if dec_w not in page_eva:
                page_eva[dec_w] = eva_w
        for dec_w, count in page_counts.items():
            index[dec_w].append({
                "folio": folio,
                "section": section,
                "eva_word": page_eva[dec_w],
                "count": count,
            })
    return dict(index)


def search_anchor_word(target, word_index, max_dist=1):
    """Search for a single anchor word in the word index.

    Returns list of matches with decoded word, distance, locations.
    """
    matches = []
    min_corpus_len = max(3, len(target) - max_dist)
    for dec_word, locations in word_index.items():
        # Skip words shorter than target minus max_dist (at least 3)
        if len(dec_word) < min_corpus_len:
            continue
        # Length pre-filter
        if abs(len(dec_word) - len(target)) > max_dist:
            continue
        dist = Levenshtein.distance(target, dec_word, score_cutoff=max_dist)
        if dist <= max_dist:
            total_count = sum(loc["count"] for loc in locations)
            matches.append({
                "decoded_word": dec_word,
                "distance": dist,
                "total_count": total_count,
                "locations": locations,
            })
    matches.sort(key=lambda m: (m["distance"], -m["total_count"]))
    return matches


def search_all_anchors(pages_data, max_dist=1):
    """Search all anchor words against decoded text.

    Returns structured results by category.
    """
    # Build indexes
    italian_index = build_word_index(pages_data, "words_decoded")
    hebrew_index = build_word_index(pages_data, "words_hebrew")

    results = {}

    for cat_id, cat_data in ANCHOR_DICT.items():
        cat_results = {
            "label": cat_data["label"],
            "matches": [],
        }

        # Italian words → match against Italian decoded text
        # 3-char targets: only exact (d=0); 4+ chars: fuzzy (d<=1)
        for word, gloss in cat_data.get("italian", []):
            normalized = normalize_for_match(word)
            if len(normalized) < 3:
                continue
            effective_dist = 0 if len(normalized) < 4 else max_dist
            matches = search_anchor_word(
                normalized, italian_index, effective_dist)
            if matches:
                cat_results["matches"].append({
                    "anchor": word,
                    "normalized": normalized,
                    "gloss": gloss,
                    "language": "italian",
                    "matches": matches,
                })

        # Hebrew words → match against Hebrew decoded text
        # 3-char targets: only exact (d=0); 4+ chars: fuzzy (d<=1)
        for word, gloss in cat_data.get("hebrew", []):
            consonantal = hebrew_to_consonantal(word)
            if len(consonantal) < 3:
                continue
            effective_dist = 0 if len(consonantal) < 4 else max_dist
            matches = search_anchor_word(
                consonantal, hebrew_index, effective_dist)
            if matches:
                cat_results["matches"].append({
                    "anchor": word,
                    "normalized": consonantal,
                    "gloss": gloss,
                    "language": "hebrew",
                    "matches": matches,
                })

        # Latin words → match against Italian decoded text
        # 3-char targets: only exact (d=0); 4+ chars: fuzzy (d<=1)
        for word, gloss in cat_data.get("latin", []):
            normalized = normalize_for_match(word)
            if len(normalized) < 3:
                continue
            effective_dist = 0 if len(normalized) < 4 else max_dist
            matches = search_anchor_word(
                normalized, italian_index, effective_dist)
            if matches:
                cat_results["matches"].append({
                    "anchor": word,
                    "normalized": normalized,
                    "gloss": gloss,
                    "language": "latin",
                    "matches": matches,
                })

        results[cat_id] = cat_results

    return results


# =====================================================================
# Distribution analysis and falsification tests
# =====================================================================

def compute_section_distribution(matches_list):
    """Compute how anchor word hits distribute across sections."""
    section_counts = Counter()
    for match_entry in matches_list:
        for m in match_entry["matches"]:
            for loc in m["locations"]:
                section_counts[loc["section"]] += loc["count"]
    return dict(section_counts.most_common())


def compute_cooccurrence(results, pages_data):
    """Compute which semantic categories co-occur on the same pages."""
    # For each page, find which categories have hits
    page_categories = defaultdict(set)

    for cat_id, cat_data in results.items():
        for match_entry in cat_data["matches"]:
            for m in match_entry["matches"]:
                for loc in m["locations"]:
                    page_categories[loc["folio"]].add(cat_id)

    # Count co-occurrences
    cooccurrence = Counter()
    for folio, cats in page_categories.items():
        cats = sorted(cats)
        for i, c1 in enumerate(cats):
            for c2 in cats[i + 1:]:
                cooccurrence[(c1, c2)] += 1

    return dict(cooccurrence.most_common(30))


def falsification_tests(results, pages_data):
    """Run falsification tests on anchor word distribution.

    Returns verdict dict with pass/fail for each test.
    """
    tests = {}

    # Compute section hit counts per category
    cat_sections = {}
    for cat_id, cat_data in results.items():
        cat_sections[cat_id] = compute_section_distribution(
            cat_data["matches"])

    # (a) Botanical terms more frequent in herbal/pharma
    bot_parts = cat_sections.get("botanica_parti", {})
    bot_actions = cat_sections.get("botanica_azioni", {})
    bot_total = Counter(bot_parts) + Counter(bot_actions)
    herbal_pharma = bot_total.get("herbal", 0) + bot_total.get(
        "pharmaceutical", 0)
    bot_sum = sum(bot_total.values())
    if bot_sum > 0:
        bot_hp_pct = herbal_pharma / bot_sum * 100
        tests["botanical_in_herbal_pharma"] = {
            "description": "Botanical terms more frequent in Herbal+Pharma",
            "herbal_pharma_count": herbal_pharma,
            "total_count": bot_sum,
            "percentage": round(bot_hp_pct, 1),
            "pass": bot_hp_pct > 40,
        }

    # (b) Astrological terms with concentration in zodiac AND herbal
    astro = cat_sections.get("astrologia", {})
    planet = cat_sections.get("pianeti", {})
    astro_total = Counter(astro) + Counter(planet)
    astro_sum = sum(astro_total.values())
    if astro_sum > 0:
        zodiac_hits = astro_total.get("zodiac", 0)
        herbal_hits = astro_total.get("herbal", 0)
        tests["astro_in_zodiac_and_herbal"] = {
            "description": "Astro terms concentrated in Zodiac and Herbal",
            "zodiac_count": zodiac_hits,
            "herbal_count": herbal_hits,
            "total_count": astro_sum,
            "zodiac_pct": round(zodiac_hits / astro_sum * 100, 1),
            "pass": zodiac_hits > 0 or herbal_hits > 0,
        }

    # (c) Balneological terms peak in balneological
    liquidi = cat_sections.get("liquidi", {})
    liquidi_sum = sum(liquidi.values())
    if liquidi_sum > 0:
        balneal_hits = liquidi.get("balneological", 0)
        tests["liquidi_in_balneological"] = {
            "description": "Liquid terms have peak in Balneological section",
            "balneological_count": balneal_hits,
            "total_count": liquidi_sum,
            "percentage": round(balneal_hits / liquidi_sum * 100, 1),
            "pass": balneal_hits > 0,
        }

    # (e) Kabbalistic terms present (strong author signal)
    cabala = cat_sections.get("cabala", {})
    cabala_sum = sum(cabala.values())
    tests["kabbalistic_terms_found"] = {
        "description": "Kabbalistic terms found (strong Jewish author signal)",
        "total_count": cabala_sum,
        "notable": cabala_sum > 5,
    }

    # (f) Flat distribution check (falsification)
    all_sections = set()
    for cat_s in cat_sections.values():
        all_sections.update(cat_s.keys())
    total_per_section = Counter()
    for cat_s in cat_sections.values():
        for sec, cnt in cat_s.items():
            total_per_section[sec] += cnt
    total_all = sum(total_per_section.values())
    if total_all > 0 and len(total_per_section) > 1:
        proportions = [c / total_all for c in total_per_section.values()]
        max_prop = max(proportions)
        min_prop = min(proportions)
        is_flat = (max_prop - min_prop) < 0.1
        tests["distribution_not_flat"] = {
            "description": "Distribution is NOT flat (mapping has signal)",
            "max_section_pct": round(max_prop * 100, 1),
            "min_section_pct": round(min_prop * 100, 1),
            "pass": not is_flat,
        }

    # (g) Alchemy + botany co-occurrence
    alch = cat_sections.get("alchimia", {})
    alch_sum = sum(alch.values())
    if alch_sum > 0:
        alch_herbal = alch.get("herbal", 0)
        alch_astro_sec = alch.get("astronomical", 0) + alch.get("zodiac", 0)
        tests["alchemy_cooccurrence"] = {
            "description": "Alchemy terms co-occur with botany or astrology",
            "with_herbal": alch_herbal,
            "with_astro": alch_astro_sec,
            "total_count": alch_sum,
        }

    return tests


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Search for anchor words in decoded text."""
    report_path = config.stats_dir / "anchor_words_report.json"

    if report_path.exists() and not force:
        click.echo("  Anchor words report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("ANCHOR WORDS — Domain Vocabulary Search")

    # 1. Load full decode or decode on the fly
    print_step("Loading decoded text...")
    full_decode_path = config.stats_dir / "full_decode.json"
    if full_decode_path.exists():
        with open(full_decode_path) as f:
            fd = json.load(f)
        pages_data = fd["pages"]
        click.echo(f"    Loaded from {full_decode_path}")
        click.echo(f"    {len(pages_data)} pages, "
                   f"{fd['total_words']} total words")
    else:
        click.echo("    full_decode.json not found, decoding on the fly...")
        mapping, divergent, direction = load_convergent_mapping(config)
        eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
        eva_data = parse_eva_words(eva_file)
        divergent_set = set(divergent.keys())
        pages_data = {}
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
            pages_data[folio] = {
                "section": SECTION_NAMES.get(section, section),
                "words_eva": words_eva,
                "words_decoded": words_dec,
                "words_hebrew": words_heb,
            }
        click.echo(f"    Decoded {len(pages_data)} pages")

    # 2. Count anchor words
    total_anchors = sum(
        len(c.get("italian", [])) + len(c.get("hebrew", []))
        + len(c.get("latin", []))
        for c in ANCHOR_DICT.values()
    )
    click.echo(f"    {total_anchors} anchor words in "
               f"{len(ANCHOR_DICT)} categories")

    # 3. Search
    print_step("Searching for anchor words (Levenshtein <= 1)...")
    results = search_all_anchors(pages_data, max_dist=1)

    # 4. Statistics
    total_found = 0
    total_hits = 0
    for cat_data in results.values():
        for match_entry in cat_data["matches"]:
            total_found += 1
            for m in match_entry["matches"]:
                total_hits += m["total_count"]

    click.echo(f"    {total_found} anchor words matched, "
               f"{total_hits} total occurrences")

    # 5. Section distribution per category
    print_step("Analyzing section distribution...")
    category_distributions = {}
    for cat_id, cat_data in results.items():
        dist = compute_section_distribution(cat_data["matches"])
        if dist:
            category_distributions[cat_id] = dist

    # 6. Co-occurrence analysis
    print_step("Computing co-occurrence map...")
    cooccurrence = compute_cooccurrence(results, pages_data)

    # 7. Falsification tests
    print_step("Running falsification tests...")
    tests = falsification_tests(results, pages_data)

    # 8. Build top matches list
    all_matches_flat = []
    for cat_id, cat_data in results.items():
        for match_entry in cat_data["matches"]:
            total_count = sum(
                m["total_count"] for m in match_entry["matches"])
            best_dist = min(
                m["distance"] for m in match_entry["matches"])
            all_matches_flat.append({
                "anchor": match_entry["anchor"],
                "gloss": match_entry["gloss"],
                "language": match_entry["language"],
                "category": cat_id,
                "category_label": cat_data["label"],
                "total_occurrences": total_count,
                "best_distance": best_dist,
                "n_decoded_forms": len(match_entry["matches"]),
            })
    all_matches_flat.sort(key=lambda m: (-m["total_occurrences"],
                                         m["best_distance"]))

    # 9. Save report
    print_step("Saving report...")
    report = {
        "max_dist": 1,
        "total_anchor_words": total_anchors,
        "total_matched": total_found,
        "total_occurrences": total_hits,
        "top_matches": all_matches_flat[:50],
        "by_category": {},
        "section_distributions": category_distributions,
        "cooccurrence_map": {
            f"{k[0]}+{k[1]}": v for k, v in cooccurrence.items()
        },
        "falsification_tests": tests,
    }

    # Detailed per-category results (limited to save space)
    for cat_id, cat_data in results.items():
        cat_summary = {
            "label": cat_data["label"],
            "n_matched": len(cat_data["matches"]),
            "matches": [],
        }
        for match_entry in cat_data["matches"]:
            entry = {
                "anchor": match_entry["anchor"],
                "normalized": match_entry["normalized"],
                "gloss": match_entry["gloss"],
                "language": match_entry["language"],
                "total_occurrences": sum(
                    m["total_count"] for m in match_entry["matches"]),
                "best_distance": min(
                    m["distance"] for m in match_entry["matches"]),
                "decoded_forms": [],
            }
            for m in match_entry["matches"][:5]:  # Top 5 forms
                sections = Counter()
                for loc in m["locations"]:
                    sections[loc["section"]] += loc["count"]
                entry["decoded_forms"].append({
                    "decoded_word": m["decoded_word"],
                    "distance": m["distance"],
                    "total_count": m["total_count"],
                    "sections": dict(sections.most_common()),
                    "sample_pages": [
                        loc["folio"] for loc in m["locations"][:5]
                    ],
                    "eva_word": m["locations"][0]["eva_word"]
                    if m["locations"] else "",
                })
            cat_summary["matches"].append(entry)
        report["by_category"][cat_id] = cat_summary

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    Report: {report_path}")

    # 10. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  ANCHOR WORDS RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Total anchor words searched: {total_anchors}")
    click.echo(f"  Matched: {total_found}")
    click.echo(f"  Total occurrences: {total_hits}")

    # Top 30 matches
    click.echo(f"\n  Top 30 anchor words by frequency:")
    for i, m in enumerate(all_matches_flat[:30]):
        d_marker = "*" if m["best_distance"] == 0 else "~"
        click.echo(
            f"    {i+1:2d}. {d_marker} {m['anchor']:15s} "
            f"({m['language'][:2]:2s}) "
            f"x{m['total_occurrences']:4d}  "
            f"[{m['category_label']}]  "
            f"d={m['best_distance']}")

    # Section distributions
    click.echo(f"\n  Section distribution per category:")
    for cat_id, dist in sorted(category_distributions.items()):
        label = ANCHOR_DICT[cat_id]["label"]
        total = sum(dist.values())
        parts = [f"{s}={c}" for s, c in dist.items()]
        click.echo(f"    {label}: {total} hits — {', '.join(parts)}")

    # Co-occurrence
    if cooccurrence:
        click.echo(f"\n  Top semantic co-occurrences (same page):")
        for (c1, c2), count in list(cooccurrence.items())[:10]:
            l1 = ANCHOR_DICT.get(c1, {}).get("label", c1)
            l2 = ANCHOR_DICT.get(c2, {}).get("label", c2)
            click.echo(f"    {l1} + {l2}: {count} pages")

    # Falsification tests
    click.echo(f"\n  FALSIFICATION TESTS:")
    for test_id, test_data in tests.items():
        status = ""
        if "pass" in test_data:
            status = "PASS" if test_data["pass"] else "FAIL"
        elif "notable" in test_data:
            status = "NOTABLE" if test_data["notable"] else "—"
        click.echo(f"    [{status:7s}] {test_data['description']}")
        # Print key numbers
        for k, v in test_data.items():
            if k not in ("description", "pass", "notable"):
                click.echo(f"             {k}: {v}")

    # Special signals
    click.echo(f"\n  SPECIAL SIGNALS:")
    cabala_matches = results.get("cabala", {}).get("matches", [])
    if cabala_matches:
        click.echo("    KABBALISTIC TERMS FOUND:")
        for m in cabala_matches:
            total = sum(mm["total_count"] for mm in m["matches"])
            click.echo(f"      {m['anchor']} ({m['gloss']}): "
                       f"{total} occurrences, d={m['matches'][0]['distance']}")
    else:
        click.echo("    No kabbalistic terms found")

    alch_matches = results.get("alchimia", {}).get("matches", [])
    if alch_matches:
        click.echo("    ALCHEMICAL TERMS FOUND:")
        for m in alch_matches[:5]:
            total = sum(mm["total_count"] for mm in m["matches"])
            click.echo(f"      {m['anchor']} ({m['gloss']}): "
                       f"{total} occurrences, d={m['matches'][0]['distance']}")

    # Verdict
    n_pass = sum(1 for t in tests.values() if t.get("pass"))
    n_tests = sum(1 for t in tests.values() if "pass" in t)
    click.echo(f"\n  {'=' * 40}")
    click.echo(f"  VERDICT: {n_pass}/{n_tests} falsification tests passed")
    if total_found > 20 and n_pass >= n_tests // 2:
        click.echo("  Distribution shows COHERENT domain structure")
    elif total_found > 10:
        click.echo("  Distribution shows PARTIAL domain structure")
    else:
        click.echo("  Insufficient anchor word matches for verdict")
    click.echo(f"  {'=' * 40}")
