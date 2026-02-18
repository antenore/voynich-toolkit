"""
Plant name search: decode full text and fuzzy-match vs botanical lexicon.

Uses an extended botanical lexicon (~400 entries) including:
  - Curated Italian botanical terms (~80)
  - Latin-vulgar italianized plant names (~120)
  - Hebrew transliterated botanical terms (~50)
  - Judeo-Italian variants (~50)
  - Common medieval herbal terms (~100)

Section analysis: expects H (Herbal) >> S/Z (Astronomical), B/P should
have some botanical terms (pharmacological). Cluster detection: pages
with 2+ distinct plant matches. Validation: check FOLIO_PLANTS expected
plants against actual fuzzy hits.
"""
import json
from collections import Counter, defaultdict

import click

from .champollion import FOLIO_PLANTS
from .config import ToolkitConfig
from .fuzzy_utils import LengthBucketedIndex
from .prepare_italian_lexicon import (
    HEBREW_TO_ITALIAN,
    normalize_italian_phonemic,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Extended botanical lexicon
# =====================================================================

def _build_botanical_lexicon():
    """Build extended botanical lexicon with ~400 entries.

    Each entry: (italian_name, phonemic, gloss, sub_domain)
    The phonemic form is what we match against decoded text.
    """
    entries = []

    def add(name, gloss, sub_domain="botanical"):
        ph = normalize_italian_phonemic(name)
        if len(ph) >= 3:
            entries.append((name, ph, gloss, sub_domain))

    # --- Group 1: Core herbal plants (from FOLIO_PLANTS + common herbs) ---
    herbs = [
        ("viola", "viola, violetta"),
        ("ruta", "ruta graveolens"),
        ("timo", "timo, thymus"),
        ("malva", "malva"),
        ("menta", "menta"),
        ("croco", "croco, zafferano"),
        ("salvia", "salvia"),
        ("sedano", "sedano, apium"),
        ("origano", "origano"),
        ("nigella", "nigella sativa"),
        ("lavanda", "lavanda"),
        ("valeriana", "valeriana"),
        ("rosmarino", "rosmarino"),
        ("ramerino", "rosmarino (toscano)"),
        ("coriandolo", "coriandolo"),
        ("mandragora", "mandragora"),
        ("elleboro", "elleboro"),
        ("borragine", "borragine"),
        ("artemisia", "artemisia"),
        ("ninfea", "ninfea"),
        ("piantagine", "piantagine"),
        ("calendula", "calendula"),
        ("papavero", "papavero"),
        ("sambuco", "sambuco"),
        ("edera", "edera"),
        ("centaurea", "centaurea"),
        ("genziana", "genziana"),
        ("felce", "felce"),
        ("betonica", "betonica"),
        ("aconito", "aconito"),
        ("verbena", "verbena"),
        ("consolida", "consolida, symphytum"),
        ("achillea", "achillea"),
        ("assenzio", "assenzio, artemisia"),
        ("enula", "enula campana"),
        ("cardo", "cardo"),
        ("nepeta", "nepeta, gattaia"),
        ("belladonna", "belladonna, atropa"),
        ("cicoria", "cicoria"),
        ("finocchio", "finocchio"),
        ("aneto", "aneto"),
        ("cumino", "cumino"),
        ("basilico", "basilico"),
        ("prezzemolo", "prezzemolo"),
        ("camomilla", "camomilla"),
        ("issopo", "issopo"),
        ("santoreggia", "santoreggia"),
        ("ricino", "ricino"),
        ("mirto", "mirto"),
        ("ginepro", "ginepro"),
        ("alloro", "alloro, lauro"),
        ("cipresso", "cipresso"),
        ("quercia", "quercia"),
        ("aloe", "aloe"),
        ("rosa", "rosa"),
        ("pepe", "pepe"),
        ("giglio", "giglio"),
        ("ranuncolo", "ranuncolo"),
    ]
    for name, gloss in herbs:
        add(name, gloss, "herbal")

    # --- Group 2: Latin-vulgar italianized plant names ---
    latin_vulgar = [
        ("absintio", "assenzio (lat. volg.)"),
        ("apio", "sedano (lat. volg.)"),
        ("abrotano", "abrotano, southernwood"),
        ("agrimonia", "agrimonia"),
        ("altea", "altea, marshmallow"),
        ("amaranto", "amaranto"),
        ("anemone", "anemone"),
        ("appio", "sedano (var.)"),
        ("aristolochia", "aristolochia"),
        ("asparago", "asparago"),
        ("aster", "aster"),
        ("atripice", "atripice, atriplice"),
        ("balsamo", "balsamo"),
        ("bardana", "bardana"),
        ("bieta", "bieta, bietola"),
        ("brionia", "brionia"),
        ("buglossa", "buglossa"),
        ("calaminta", "calaminta"),
        ("caprifoglio", "caprifoglio"),
        ("carlina", "carlina"),
        ("cassia", "cassia"),
        ("celidonia", "celidonia"),
        ("centauro", "centaurea (var.)"),
        ("cicuta", "cicuta"),
        ("cimino", "cumino (var.)"),
        ("cinnamomo", "cannella"),
        ("cipolla", "cipolla"),
        ("citiso", "citiso"),
        ("coloquintide", "coloquintide"),
        ("comino", "cumino (var.)"),
        ("dittamo", "dittamo"),
        ("dragoncello", "dragoncello"),
        ("ebbio", "ebbio, sambucus ebulus"),
        ("eliotropio", "eliotropio"),
        ("erba", "erba"),
        ("eruca", "rucola"),
        ("eufragia", "eufragia"),
        ("eupatorio", "eupatorio"),
        ("farfara", "farfara, tussilago"),
        ("fieno", "fieno"),
        ("fragola", "fragola"),
        ("fumaria", "fumaria"),
        ("galanga", "galanga"),
        ("garofano", "garofano"),
        ("gelsomino", "gelsomino"),
        ("gialappa", "gialappa"),
        ("gramigna", "gramigna"),
        ("iride", "iride, iris"),
        ("lattuga", "lattuga"),
        ("lino", "lino"),
        ("liquirizia", "liquirizia"),
        ("luppolo", "luppolo"),
        ("maggiorana", "maggiorana"),
        ("malvavisco", "malvavisco, altea"),
        ("marrubio", "marrubio"),
        ("melissa", "melissa"),
        ("melone", "melone"),
        ("melograno", "melograno"),
        ("millefoglie", "millefoglie, achillea"),
        ("mirra", "mirra"),
        ("mortella", "mortella, mirto"),
        ("nardo", "nardo"),
        ("nasturzio", "nasturzio"),
        ("noce", "noce"),
        ("olivo", "olivo"),
        ("ortica", "ortica"),
        ("palma", "palma"),
        ("pastinaca", "pastinaca"),
        ("pelosella", "pelosella"),
        ("peonia", "peonia"),
        ("pervinca", "pervinca"),
        ("pimpinella", "pimpinella"),
        ("piretro", "piretro"),
        ("platano", "platano"),
        ("poligono", "poligono"),
        ("polipodio", "polipodio"),
        ("portulaca", "portulaca"),
        ("primula", "primula"),
        ("psillio", "psillio"),
        ("pulegio", "pulegio, mentuccia"),
        ("rabarbaro", "rabarbaro"),
        ("rafano", "rafano"),
        ("romice", "romice, rumex"),
        ("rosetta", "rosa selvatica"),
        ("ruchetta", "rucola"),
        ("rovo", "rovo, mora"),
        ("salsapariglia", "salsapariglia"),
        ("salicone", "salice"),
        ("sambuco", "sambuco"),
        ("saponaria", "saponaria"),
        ("sassifraga", "sassifraga"),
        ("scabiosa", "scabiosa"),
        ("scolopendrio", "scolopendrio"),
        ("scorzonera", "scorzonera"),
        ("senape", "senape"),
        ("serpillo", "serpillo, timo selvatico"),
        ("solatro", "solatro, solanum"),
        ("spigo", "lavanda spigo"),
        ("stramonio", "stramonio"),
        ("tanaceto", "tanaceto"),
        ("tarassaco", "tarassaco"),
        ("terebinto", "terebinto"),
        ("tormentilla", "tormentilla"),
        ("tussilago", "tussilago, farfara"),
        ("valeriana", "valeriana"),
        ("verbasco", "verbasco"),
        ("veronica", "veronica"),
        ("vetriolo", "vetriolo"),
        ("vincetossico", "vincetossico"),
        ("viburno", "viburno"),
        ("vinca", "pervinca"),
        ("zafferano", "zafferano"),
        ("zenzero", "zenzero"),
        ("zucca", "zucca"),
    ]
    for name, gloss in latin_vulgar:
        add(name, gloss, "latin_vulgar")

    # --- Group 3: Hebrew transliterated plant names ---
    # These are phonemic approximations of Hebrew botanical terms
    # as they'd appear after Hebrew->Italian decode
    hebrew_plants = [
        ("ered", "rosa (heb. wrd)"),
        ("azob", "issopo (heb. Azwb)"),
        ("karkam", "zafferano (heb. krkm)"),
        ("kinamon", "cannella (heb. qnmwn)"),
        ("gepen", "vite (heb. gpn)"),
        ("tamer", "palma (heb. tmr)"),
        ("rimon", "melograno (heb. rmwn)"),
        ("tapoak", "melo (heb. tpwX)"),
        ("erez", "cedro (heb. Arz)"),
        ("laana", "assenzio (heb. lEnh)"),
        ("sosan", "giglio (heb. SwSn)"),
        ("kamon", "cumino (heb. kmn)"),
        ("gad", "coriandolo (heb. gd)"),
        ("ketsa", "cumino nero (heb. qCX)"),
        ("peri", "frutto (heb. pry)"),
        ("zera", "seme (heb. zrE)"),
        ("ale", "foglia (heb. Elh)"),
        ("pera", "fiore (heb. prX)"),
        ("sores", "radice (heb. SrS)"),
        ("eseb", "erba (heb. ESb)"),
        ("mor", "mirra (heb. mr)"),
        ("sam", "droga (heb. sm)"),
        ("ikar", "radice med. (heb. Eqr)"),
        ("tsema", "germoglio (heb. CmX)"),
        ("lebona", "incenso (heb. lbwnh)"),
        ("nerd", "nardo (heb. nrd)"),
        ("elon", "quercia (heb. Alwn)"),
        ("tena", "fico (heb. tAnh)"),
        ("zait", "olivo (heb. zyt)"),
        ("saked", "mandorlo (heb. Sqd)"),
        ("berot", "cipresso (heb. brwS)"),
        ("egoz", "noce (heb. Agwz)"),
        ("sita", "grano (heb. XJh)"),
        ("seora", "orzo (heb. SErh)"),
        ("dokhan", "miglio (heb. dXn)"),
        ("kana", "canna (heb. qnh)"),
        ("som", "aglio (heb. Swm)"),
        ("batsal", "cipolla (heb. bCl)"),
        ("kresa", "porro (heb. krSh)"),
        # --- Shem Tov / medieval Hebrew botanical (XIII sec.) ---
        ("astis", "guado (heb. Astys)"),
        ("aibrata", "ginepro (heb. AybrAtA)"),
        ("apsantin", "assenzio med. (heb. Apsntyn)"),
        ("atstrublin", "pinoli (heb. ACtrwblyn)"),
        ("asa", "mirto med. (heb. AsA)"),
        ("airus", "iris (heb. Ayrws)"),
        ("askare", "bosso (heb. ASkrE)"),
        ("ailen abream", "agnocasto (heb. Ayln Abrhm)"),
        ("erdupni", "oleandro (heb. hrdwpny)"),
        ("elilgim", "mirabolano (heb. hlylgym)"),
        ("dartsin", "cannella med. (heb. drCyn)"),
        ("dabdaniot", "ciliegie (heb. dbdbnywt)"),
        ("erni", "malva (heb. hrny)"),
        ("erzpa", "piretro (heb. hrzpA)"),
        ("kaltit", "assafetida (heb. Xltyt)"),
        ("ebanim", "ebano (heb. hbnym)"),
        ("gopnan", "finocchio med. (heb. gwpnn)"),
        ("burit", "liscivia veg. (heb. bwryt)"),
        ("batnim", "pistacchi (heb. btnym)"),
        ("kelil emelk", "meliloto (heb. klyl hmlk)"),
        ("lapsan", "senape selv. (heb. lpsn)"),
        ("leson etspur", "frassino fr. (heb. lSwn hCpwr)"),
        ("ordi ekmurim", "peonia (heb. wrdy hXmwrym)"),
        ("batsal sade", "scilla (heb. bCl Sdh)"),
        # --- Talmudico/mishnaico ---
        ("morika", "cartamo (heb. mwryqA)"),
        ("kalbana", "galbano (heb. Xlbnh)"),
        ("pilpel", "pepe (heb. plpl)"),
        ("eolsin", "cicoria (heb. EwlSyn)"),
        ("sos", "liquirizia (heb. SwS)"),
        ("rikhan", "basilico (heb. ryXn)"),
        ("kalamut", "borragine (heb. Xlmwt)"),
        ("sumak", "sommacco (heb. swmq)"),
        ("narkis", "narciso (heb. nrqys)"),
        ("dalat", "zucca (heb. dlEt)"),
        ("lupa", "luffa (heb. lwph)"),
        ("turmus", "lupino (heb. twrmws)"),
        ("kazeret", "lattuga (heb. Xzrt)"),
        ("kruv", "cavolo (heb. krwb)"),
        ("lefet", "rapa (heb. lpt)"),
        ("tsanun", "ravanello (heb. Cnwn)"),
        ("sakalim", "crescione (heb. SXlym)"),
        ("gargir", "rucola (heb. grgyr)"),
        ("knista", "carciofo (heb. knyStA)"),
        ("kasa", "lattuga aram. (heb. XsA)"),
        ("kesut", "cuscuta (heb. kSwt)"),
        ("siloa", "bietola (heb. sylwA)"),
        ("kornit", "camomilla (heb. qwrnyt)"),
        ("kusbar", "coriandolo talm. (heb. kwsbr)"),
        ("peigam", "ruta (heb. pygm)"),
        ("tiltan", "fieno greco (heb. tltn)"),
        ("sumsem", "sesamo (heb. SwmSm)"),
        ("dapna", "alloro (heb. dpnh)"),
        ("kilba", "fieno greco var. (heb. Xylbh)"),
        ("mastika", "mastice (heb. msJykA)"),
        ("kapor", "canfora (heb. kpwr)"),
        ("limon", "limone (heb. lymwn)"),
        # --- Piante bibliche aggiuntive ---
        ("tsori", "balsamo Galaad (heb. Cry)"),
        ("natap", "storace (heb. nJp)"),
        ("atad", "licio (heb. AJd)"),
        ("agmon", "giunco (heb. Agmwn)"),
        ("gome", "papiro (heb. gmA)"),
        ("abiuna", "cappero (heb. Abywnh)"),
        ("bedulak", "bdellio (heb. bdwlX)"),
        ("bakaim", "gelso (heb. bkAym)"),
        ("dardar", "cardo stellato (heb. drdr)"),
        ("kabatselet", "colchico (heb. XbClt)"),
        ("kadak", "solano (heb. Xdq)"),
        ("arar", "ginepro bibl. (heb. ErEr)"),
        ("pol", "fava (heb. pwl)"),
        ("tsaelim", "giuggiolo (heb. CAlym)"),
        ("tsaptsapa", "salice (heb. CpCph)"),
        ("karuv", "carruba (heb. Xrwb)"),
        ("etrog", "cedro frutto (heb. Atrwg)"),
        ("kardal", "senape (heb. Xrdl)"),
        ("koper", "henné (heb. kpr)"),
        ("sikma", "sicomoro (heb. Sqmh)"),
        ("dekel", "palma aram. (heb. dql)"),
    ]
    for name, gloss in hebrew_plants:
        add(name, gloss, "hebrew_translit")

    # --- Group 4: Judeo-Italian botanical variants ---
    judeo_it = [
        ("savina", "savina, ginepro sabina"),
        ("ramerino", "rosmarino (toscano/veneto)"),
        ("spigo", "lavanda (var.)"),
        ("erba stella", "plantago"),
        ("erbasanta", "verbena"),
        ("cocomero", "cocomero"),
        ("cetriolo", "cetriolo"),
        ("aglio", "aglio"),
        ("porro", "porro"),
        ("cavolo", "cavolo"),
        ("rapa", "rapa"),
        ("fava", "fava"),
        ("lente", "lenticchia"),
        ("canapa", "canapa"),
        ("senape", "senape"),
        ("orzo", "orzo"),
        ("grano", "grano"),
        ("miglio", "miglio"),
        ("spelta", "spelta"),
        ("pomo", "mela (giudeo-it.)"),
        ("pero", "pero"),
        ("melo", "melo"),
        ("cedro", "cedro"),
        ("arancio", "arancio"),
        ("limone", "limone"),
        ("pistacchio", "pistacchio"),
        ("dattero", "dattero"),
        ("mandorla", "mandorla"),
        ("nocciola", "nocciola"),
        ("castagna", "castagna"),
        ("mora", "mora"),
        ("fico", "fico"),
        ("uva", "uva"),
        ("oliva", "oliva"),
        ("cotogno", "cotogno"),
        ("nespola", "nespola"),
        ("pesca", "pesca (frutto)"),
        ("susina", "susina, prugna"),
        ("ciliegia", "ciliegia"),
        ("amarena", "amarena"),
        ("corbezzolo", "corbezzolo"),
        ("melagrana", "melograno (var.)"),
    ]
    for name, gloss in judeo_it:
        add(name, gloss, "judeo_italian")

    # --- Group 5: Plant parts and herbal terms ---
    herbal_terms = [
        ("foglia", "foglia"),
        ("fiore", "fiore"),
        ("radice", "radice"),
        ("seme", "seme"),
        ("frutto", "frutto"),
        ("corteccia", "corteccia"),
        ("ramo", "ramo"),
        ("stelo", "stelo"),
        ("bocciolo", "bocciolo"),
        ("spiga", "spiga"),
        ("bacca", "bacca"),
        ("erba", "erba"),
        ("pianta", "pianta"),
        ("giardino", "giardino"),
        ("orto", "orto"),
        ("succo", "succo"),
        ("olio", "olio"),
        ("resina", "resina"),
        ("decotto", "decotto"),
        ("infusione", "infusione"),
        ("polvere", "polvere"),
        ("tintura", "tintura"),
        ("balsamo", "balsamo"),
        ("unguento", "unguento"),
        ("impiastro", "impiastro"),
        ("sciroppo", "sciroppo"),
        ("estratto", "estratto"),
        ("distillato", "distillato"),
        ("pomata", "pomata"),
        ("cataplasma", "cataplasma"),
        ("fumigazione", "fumigazione"),
        ("suffumigio", "suffumigio"),
        ("collirio", "collirio"),
        ("gargarismo", "gargarismo"),
        ("virtude", "virtu' (della pianta)"),
        ("proprieta", "proprieta'"),
        ("qualita", "qualita'"),
        ("complessione", "complessione"),
        ("natura", "natura"),
        ("caldo", "caldo (qualita')"),
        ("freddo", "freddo (qualita')"),
        ("umido", "umido (qualita')"),
        ("secco", "secco (qualita')"),
    ]
    for name, gloss in herbal_terms:
        add(name, gloss, "herbal_terms")

    return entries


# =====================================================================
# Core search
# =====================================================================

def decode_word(eva_word, mapping, direction):
    """Decode EVA word to Italian via Hebrew mapping.

    Uses the canonical full_decode.decode_word for proper ch/ii/i/q
    handling plus positional splits (d@init→bet, h@init→samekh).
    """
    from .full_decode import decode_word as _canonical_decode

    italian, hebrew, n_unknown = _canonical_decode(
        eva_word, mapping=mapping, direction=direction
    )
    if n_unknown > 0:
        return None
    return italian


def search_plants_in_text(eva_data, mapping, direction, plant_index,
                          max_dist=2, min_word_len=3):
    """Decode all text and search for plant names.

    Returns list of {folio, section, eva_word, decoded, plant_match,
                     distance, gloss, domain}.
    """
    results = []
    for page in eva_data["pages"]:
        folio = page["folio"]
        section = page.get("section", "?")
        seen_on_page = set()

        for word in page["words"]:
            if len(word) < min_word_len:
                continue
            decoded = decode_word(word, mapping, direction)
            if not decoded or len(decoded) < min_word_len:
                continue

            matches = plant_index.query(decoded, max_dist)
            for m in matches:
                key = (folio, m.target)
                if key in seen_on_page:
                    continue
                seen_on_page.add(key)
                results.append({
                    "folio": folio,
                    "section": section,
                    "eva_word": word,
                    "decoded": decoded,
                    "plant_match": m.target,
                    "distance": m.distance,
                    "gloss": m.gloss,
                    "domain": m.domain,
                })

    return results


def analyze_section_distribution(hits):
    """Check section distribution of plant hits.

    Expected: H (herbal) >> S/Z (astronomical), B/P has some.
    """
    by_section = defaultdict(list)
    for h in hits:
        by_section[h["section"]].append(h)

    result = {}
    total = len(hits) or 1
    for sec, sec_hits in sorted(by_section.items()):
        unique_plants = len(set(h["plant_match"] for h in sec_hits))
        result[sec] = {
            "n_hits": len(sec_hits),
            "pct": round(100 * len(sec_hits) / total, 1),
            "unique_plants": unique_plants,
        }
    return result


def find_plant_clusters(hits):
    """Find pages with 2+ distinct plant matches.

    These are strong candidates for herbal pages.
    """
    by_folio = defaultdict(set)
    for h in hits:
        by_folio[h["folio"]].add(h["plant_match"])

    clusters = []
    for folio, plants in sorted(by_folio.items()):
        if len(plants) >= 2:
            clusters.append({
                "folio": folio,
                "n_plants": len(plants),
                "plants": sorted(plants),
            })

    clusters.sort(key=lambda c: -c["n_plants"])
    return clusters


def validate_folio_plants(hits, folio_plants):
    """Check if expected plants from FOLIO_PLANTS are found.

    Returns per-folio validation results.
    """
    hits_by_folio = defaultdict(set)
    for h in hits:
        hits_by_folio[h["folio"]].add(h["plant_match"])

    results = []
    n_found = 0
    for folio, expected_names in sorted(folio_plants.items()):
        expected_phonemic = [
            normalize_italian_phonemic(n) for n in expected_names
        ]
        actual = hits_by_folio.get(folio, set())
        found = [ep for ep in expected_phonemic if ep in actual]
        if found:
            n_found += 1

        results.append({
            "folio": folio,
            "expected": expected_names,
            "expected_phonemic": expected_phonemic,
            "actual_hits": sorted(actual),
            "found": found,
            "match": len(found) > 0,
        })

    return results, n_found


# =====================================================================
# Mapping loading
# =====================================================================

def load_best_mapping(config):
    """Load the best available mapping (fuzzy_decode > champollion).

    Returns (mapping_dict, direction).
    """
    # Try fuzzy_matches.json first (has completed mapping)
    fuzzy_path = config.stats_dir / "fuzzy_matches.json"
    if fuzzy_path.exists():
        with open(fuzzy_path) as f:
            data = json.load(f)
        mapping = {}
        for eva_ch, info in data.get("mapping", {}).items():
            mapping[eva_ch] = info["hebrew"]
        if len(mapping) == 19:
            return mapping, data.get("direction", "rtl")

    # Fall back to champollion
    champ_path = config.stats_dir / "champollion_report.json"
    if champ_path.exists():
        with open(champ_path) as f:
            data = json.load(f)
        mapping = {}
        for eva_ch, info in data.get("mapping_extended", {}).items():
            mapping[eva_ch] = info["hebrew"]
        return mapping, data.get("direction", "rtl")

    raise click.ClickException(
        "No mapping found. Run: voynich champollion (or fuzzy-decode)"
    )


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, direction=None,
        min_word_len=3, max_dist=2):
    """Entry point: plant name search in decoded text."""
    report_path = config.stats_dir / "plant_search_report.json"

    if report_path.exists() and not force:
        click.echo("  Plant search report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PLANT NAME SEARCH — Botanical Fuzzy Matching")

    # 1. Load mapping
    print_step("Loading best available mapping...")
    mapping, map_direction = load_best_mapping(config)
    if direction is None:
        direction = map_direction
    click.echo(f"    {len(mapping)}/19 chars mapped, direction: {direction}")

    # 2. Build botanical lexicon
    print_step("Building extended botanical lexicon...")
    raw_entries = _build_botanical_lexicon()
    # Deduplicate by phonemic form
    seen = set()
    unique_entries = []
    for name, ph, gloss, sub_dom in raw_entries:
        if ph not in seen:
            seen.add(ph)
            unique_entries.append((name, ph, gloss, sub_dom))

    click.echo(f"    {len(unique_entries)} unique botanical forms "
               f"(from {len(raw_entries)} raw entries)")

    # Sub-domain stats
    sub_counts = Counter(e[3] for e in unique_entries)
    for sd, cnt in sub_counts.most_common():
        click.echo(f"      {sd}: {cnt}")

    # 3. Build fuzzy index
    print_step("Building botanical fuzzy index...")
    forms = [ph for _, ph, _, _ in unique_entries]
    ftg = {ph: gloss for _, ph, gloss, _ in unique_entries}
    ftd = {ph: sub_dom for _, ph, _, sub_dom in unique_entries}
    plant_index = LengthBucketedIndex(forms, ftg, ftd)
    click.echo(f"    Indexed {plant_index.size} forms")

    # 4. Parse EVA
    print_step("Parsing EVA words...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"EVA file not found: {eva_file}\n  Run first: voynich eva"
        )
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} words, "
               f"{len(eva_data['pages'])} pages")

    # 5. Search
    print_step("Searching for plant names in decoded text...")
    hits = search_plants_in_text(
        eva_data, mapping, direction, plant_index,
        max_dist=max_dist, min_word_len=min_word_len,
    )
    click.echo(f"    {len(hits)} plant hits found")

    # 6. Section distribution
    print_step("Section distribution analysis...")
    sec_dist = analyze_section_distribution(hits)
    for sec, info in sorted(sec_dist.items()):
        click.echo(f"    Section {sec}: {info['n_hits']} hits "
                   f"({info['pct']:.1f}%), "
                   f"{info['unique_plants']} unique plants")

    # 7. Cluster detection
    print_step("Plant cluster detection...")
    clusters = find_plant_clusters(hits)
    click.echo(f"    {len(clusters)} folios with 2+ plant matches")
    for c in clusters[:10]:
        click.echo(f"    {c['folio']}: {c['n_plants']} plants "
                   f"({', '.join(c['plants'][:5])})")

    # 8. Folio validation
    print_step("Validating against FOLIO_PLANTS database...")
    folio_val, n_found = validate_folio_plants(hits, FOLIO_PLANTS)
    click.echo(f"    {n_found}/{len(FOLIO_PLANTS)} expected plants found")
    for fv in folio_val:
        if fv["match"]:
            click.echo(f"    [OK] {fv['folio']}: expected "
                       f"{fv['expected']}, found {fv['found']}")

    # 9. Distance breakdown
    dist_counts = Counter(h["distance"] for h in hits)
    click.echo(f"\n    By distance: {dict(sorted(dist_counts.items()))}")

    # 10. Top plants
    plant_freq = Counter(h["plant_match"] for h in hits)
    top_plants = plant_freq.most_common(20)

    # 11. Save report
    print_step("Saving report...")
    report = {
        "direction": direction,
        "max_dist": max_dist,
        "min_word_len": min_word_len,
        "n_botanical_forms": len(unique_entries),
        "n_hits": len(hits),
        "n_unique_plants_found": len(set(h["plant_match"] for h in hits)),
        "distance_breakdown": dict(sorted(dist_counts.items())),
        "section_distribution": sec_dist,
        "clusters": clusters[:30],
        "folio_validation": {
            "n_expected": len(FOLIO_PLANTS),
            "n_found": n_found,
            "details": folio_val[:30],
        },
        "top_plants": [
            {"plant": p, "count": c, "gloss": ftg.get(p, "")}
            for p, c in top_plants
        ],
        "all_hits": hits[:200],
        "mapping_used": {
            eva_ch: {
                "hebrew": heb_ch,
                "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
            }
            for eva_ch, heb_ch in sorted(mapping.items())
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Report: {report_path}")

    # Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  PLANT SEARCH RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Total hits: {len(hits)}")
    click.echo(f"  Unique plants: "
               f"{len(set(h['plant_match'] for h in hits))}")
    click.echo(f"  FOLIO_PLANTS validated: {n_found}/{len(FOLIO_PLANTS)}")
    click.echo(f"  Clusters (2+ plants/page): {len(clusters)}")

    click.echo(f"\n  Top 15 plants found:")
    for p, c in top_plants[:15]:
        gloss = ftg.get(p, "")
        click.echo(f"    {p:<20s} {c:4d} hits  {gloss}")

    # 12. Separate d=0 and d<=1 reporting
    d0_hits = [h for h in hits if h["distance"] == 0]
    d1_hits = [h for h in hits if h["distance"] <= 1]
    d0_plants = set(h["plant_match"] for h in d0_hits)
    d1_plants = set(h["plant_match"] for h in d1_hits)

    click.echo(f"\n  Distance breakdown:")
    click.echo(f"    d=0 (exact): {len(d0_hits)} hits, "
               f"{len(d0_plants)} unique plants")
    click.echo(f"    d<=1 (fuzzy): {len(d1_hits)} hits, "
               f"{len(d1_plants)} unique plants")
    click.echo(f"    d<=2 (loose): {len(hits)} hits, "
               f"{len(set(h['plant_match'] for h in hits))} unique plants")

    if d0_hits:
        click.echo(f"\n  Top d=0 exact plant matches:")
        d0_freq = Counter(h["plant_match"] for h in d0_hits)
        for p, c in d0_freq.most_common(15):
            gloss = ftg.get(p, "")
            click.echo(f"    {p:<20s} {c:4d} hits  {gloss}")

    report["distance_report"] = {
        "d0_hits": len(d0_hits),
        "d0_unique_plants": len(d0_plants),
        "d0_top_plants": [
            {"plant": p, "count": c}
            for p, c in Counter(
                h["plant_match"] for h in d0_hits).most_common(20)
        ],
        "d1_hits": len(d1_hits),
        "d1_unique_plants": len(d1_plants),
    }

    # 13. Permutation test
    click.echo(f"\n  --- Permutation Test ---")
    try:
        from .permutation_stats import (
            build_full_mapping,
            decode_eva_with_mapping,
            permutation_test_mapping,
        )

        # Collect all EVA words
        all_eva_words = []
        for page in eva_data["pages"]:
            all_eva_words.extend(page["words"])

        # Normalized plant set for exact match scoring
        plant_forms = set(ph for _, ph, _, _ in unique_entries)

        def plant_score_fn(test_mapping):
            """Count unique plant types matched (not tokens).

            Using unique types rather than total occurrences reduces
            noise from short forms (e.g. 'mor' matching 225 pages).
            """
            matched_types = set()
            for eva_w in all_eva_words:
                if len(eva_w) < min_word_len:
                    continue
                decoded = decode_eva_with_mapping(
                    eva_w, test_mapping, mode="italian", direction=direction
                )
                if decoded and decoded in plant_forms:
                    matched_types.add(decoded)
            return len(matched_types)

        full_map = build_full_mapping(mapping)
        perm_result = permutation_test_mapping(
            plant_score_fn, full_map, n_perms=1000, seed=42)

        click.echo(f"    Real score:   {perm_result['real_score']}")
        click.echo(f"    Random mean:  {perm_result['random_mean']} "
                   f"± {perm_result['random_std']}")
        click.echo(f"    p-value:      {perm_result['p_value']:.6f}")
        click.echo(f"    z-score:      {perm_result['z_score']:.1f}")
        sig = "***" if perm_result["significant_001"] else \
              "**" if perm_result["significant_01"] else \
              "*" if perm_result["significant_05"] else "ns"
        click.echo(f"    Significance: {sig}")
        report["permutation_test"] = perm_result
    except Exception as e:
        click.echo(f"    Permutation test error: {e}")

    # Re-save report with permutation and distance data
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Verdict
    h_pct = sec_dist.get("H", {}).get("pct", 0)
    click.echo(f"\n  {'=' * 40}")
    if h_pct > 50:
        click.echo("  VERDICT: Herbal section dominates plant hits")
        click.echo("  (consistent with botanical content)")
    elif len(hits) > 50:
        click.echo("  VERDICT: Substantial plant vocabulary found")
        click.echo(f"  ({len(hits)} hits across sections)")
    else:
        click.echo("  VERDICT: Limited plant matches")
        click.echo("  (mapping may need refinement)")
    click.echo(f"  {'=' * 40}")
