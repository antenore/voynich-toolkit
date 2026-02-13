"""
Preparazione lessico italiano medievale per tentativo di decifrazione giudeo-italiana.

Ipotesi: il Voynich fu scritto da un medico ebreo dell'area padovano-veneziana
(1404-1438) usando scrittura corsiva ebraica per codificare testo in italiano /
giudeo-italiano.  A differenza dell'ebraico puro (abjad consonantico), il
giudeo-italiano scrive le vocali usando matres lectionis.

Fonti lessicali:
  - Curato: ~302 termini per dominio (botanico, astronomico, medico, generale)
  - TLIO: ~64.800 lemmi dal Tesoro della Lingua Italiana delle Origini
  - Dante: ~12.800 forme dalla Divina Commedia
  - Kaikki: ~4.900 forme arcaiche/obsolete da Wiktionary

Struttura parallela a prepare_lexicon.py.
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import click

from .config import ToolkitConfig
from .utils import print_header, print_step


# =====================================================================
# Costanti: mappatura ebraico → fonemi italiani (giudeo-italiano)
# =====================================================================

# Mappatura fissa lettere ebraiche → fonemi italiani.
# Le matres lectionis codificano vocali: aleph=a, ayin=e, vav=o/u, yod=i.
# he=e (a fine parola, tipico del giudeo-italiano).
# Usiamo gli stessi codici ASCII del modulo prepare_lexicon.py.
HEBREW_TO_ITALIAN = {
    'A': 'a',    # א aleph → a
    'b': 'b',    # ב bet → b (anche v)
    'g': 'g',    # ג gimel → g
    'd': 'd',    # ד dalet → d
    'h': 'e',    # ה he → e (mater lectionis)
    'w': 'o',    # ו vav → o (anche u)
    'z': 'z',    # ז zayin → z
    'X': 'k',    # ח chet → k (gutturale → velare)
    'J': 't',    # ט tet → t
    'y': 'i',    # י yod → i
    'k': 'k',    # כ kaf → k
    'l': 'l',    # ל lamed → l
    'm': 'm',    # מ mem → m
    'n': 'n',    # נ nun → n
    's': 's',    # ס samekh → s
    'E': 'e',    # ע ayin → e
    'p': 'p',    # פ pe → p (anche f)
    'C': 'ts',   # צ tsade → ts
    'q': 'k',    # ק qof → k
    'r': 'r',    # ר resh → r
    'S': 's',    # ש shin → s (anche sh)
    't': 't',    # ת tav → t
}

# Varianti ambigue: lettere che possono rappresentare più fonemi
HEBREW_ALTERNATIVES = {
    'b': ['b', 'v'],     # bet = b / v
    'p': ['p', 'f'],     # pe = p / f
    'w': ['o', 'u'],     # vav = o / u
    'S': ['s', 'sh'],    # shin = s / sh (sci, sce)
    'h': ['e', ''],      # he = e / muta a fine parola
    'X': ['k', 'ch'],    # chet = k / ch
}

# Frequenze di riferimento dall'italiano medievale (normalizzate)
ITALIAN_LETTER_FREQS = {
    'a': 0.118, 'b': 0.010, 'c': 0.045, 'd': 0.037,
    'e': 0.118, 'f': 0.011, 'g': 0.016, 'h': 0.006,
    'i': 0.104, 'k': 0.001, 'l': 0.065, 'm': 0.025,
    'n': 0.069, 'o': 0.098, 'p': 0.031, 'q': 0.005,
    'r': 0.064, 's': 0.050, 't': 0.056, 'u': 0.030,
    'v': 0.021, 'z': 0.012,
}

# Bigrammi più frequenti in italiano
ITALIAN_COMMON_BIGRAMS = [
    'er', 're', 'on', 'an', 'in', 'el', 'al', 'di', 'le', 'ra',
    'en', 'ne', 'la', 'de', 'co', 'ri', 'ar', 'or', 'te', 'se',
    'ta', 'to', 'ti', 'no', 'ro', 'li', 'lo', 'si', 'na', 'at',
    'ni', 'ia', 'io', 'it', 'ol', 'so', 'il', 'pe', 'me', 'ma',
    'un', 'st', 'pr', 'ca', 'ch', 'tr', 'nt', 'nd', 'ss', 'tt',
]


# =====================================================================
# Sezione 1 — Lessico curato per dominio
# =====================================================================

def get_curated_botanical_it():
    """~80 termini botanici in italiano medievale.

    Fonti: erbari padovani XIV-XV sec., Matthaeus Platearius,
    Dioscoride volgarizzato.
    """
    terms = [
        # Piante aromatiche e medicinali
        ("rosa", "rosa, fiore"),
        ("salvia", "salvia"),
        ("rosmarino", "rosmarino"),
        ("mandragora", "mandragora"),
        ("ruta", "ruta"),
        ("finocchio", "finocchio"),
        ("menta", "menta"),
        ("basilico", "basilico"),
        ("origano", "origano"),
        ("timo", "timo"),
        ("camomilla", "camomilla"),
        ("lavanda", "lavanda"),
        ("borragine", "borragine"),
        ("valeriana", "valeriana"),
        ("verbena", "verbena"),
        ("assenzio", "assenzio"),
        ("anice", "anice"),
        ("cumino", "cumino"),
        ("coriandolo", "coriandolo"),
        ("zafferano", "zafferano"),
        ("cannella", "cannella"),
        ("pepe", "pepe"),
        ("zenzero", "zenzero"),
        # Alberi e frutti
        ("olivo", "olivo"),
        ("fico", "fico"),
        ("vite", "vite"),
        ("melograno", "melograno"),
        ("noce", "noce"),
        ("mandorlo", "mandorlo"),
        ("cedro", "cedro"),
        ("melo", "melo"),
        ("pero", "pero"),
        ("ciliegio", "ciliegio"),
        ("pino", "pino"),
        ("quercia", "quercia"),
        ("cipresso", "cipresso"),
        ("palma", "palma"),
        # Cereali e ortaggi
        ("grano", "grano"),
        ("orzo", "orzo"),
        ("miglio", "miglio"),
        ("fava", "fava"),
        ("lente", "lente, lenticchia"),
        ("aglio", "aglio"),
        ("cipolla", "cipolla"),
        ("porro", "porro"),
        ("lattuga", "lattuga"),
        ("cavolo", "cavolo"),
        ("rapa", "rapa"),
        ("zucca", "zucca"),
        # Parti di pianta
        ("foglia", "foglia"),
        ("fiore", "fiore"),
        ("radice", "radice"),
        ("seme", "seme"),
        ("frutto", "frutto"),
        ("corteccia", "corteccia"),
        ("ramo", "ramo"),
        ("tronco", "tronco"),
        ("stelo", "stelo"),
        ("bocciolo", "bocciolo"),
        ("spiga", "spiga"),
        ("bacca", "bacca"),
        # Concetti erbari
        ("erba", "erba"),
        ("pianta", "pianta"),
        ("giardino", "giardino"),
        ("orto", "orto"),
        ("campo", "campo"),
        ("bosco", "bosco"),
        ("succo", "succo"),
        ("olio", "olio"),
        ("resina", "resina"),
        ("balsamo", "balsamo"),
        ("decotto", "decotto"),
        ("infusione", "infusione"),
        ("polvere", "polvere"),
        ("mirra", "mirra"),
        ("incenso", "incenso"),
        ("aloe", "aloe"),
        ("issopo", "issopo"),
    ]
    return _curated_to_entries(terms, "botanical")


def get_curated_astronomical_it():
    """~50 termini astronomici in italiano medievale.

    Fonti: Sacrobosco volgarizzato, Dante (Convivio),
    trattati astrologici padovani.
    """
    terms = [
        # Corpi celesti
        ("stella", "stella"),
        ("luna", "luna"),
        ("sole", "sole"),
        ("pianeta", "pianeta"),
        ("cielo", "cielo"),
        ("firmamento", "firmamento"),
        ("sfera", "sfera"),
        ("cerchio", "cerchio"),
        ("orbita", "orbita"),
        # Zodiaco
        ("ariete", "ariete (segno)"),
        ("toro", "toro (segno)"),
        ("gemelli", "gemelli (segno)"),
        ("cancro", "cancro (segno)"),
        ("leone", "leone (segno)"),
        ("vergine", "vergine (segno)"),
        ("bilancia", "bilancia (segno)"),
        ("scorpione", "scorpione (segno)"),
        ("sagittario", "sagittario (segno)"),
        ("capricorno", "capricorno (segno)"),
        ("acquario", "acquario (segno)"),
        ("pesci", "pesci (segno)"),
        # Concetti astronomici
        ("eclissi", "eclissi"),
        ("equinozio", "equinozio"),
        ("solstizio", "solstizio"),
        ("oriente", "oriente, est"),
        ("occidente", "occidente, ovest"),
        ("tramontana", "tramontana, nord"),
        ("mezzogiorno", "mezzogiorno, sud"),
        ("grado", "grado"),
        ("segno", "segno zodiacale"),
        ("congiunzione", "congiunzione"),
        ("opposizione", "opposizione"),
        ("ascendente", "ascendente"),
        # Tempo
        ("giorno", "giorno"),
        ("notte", "notte"),
        ("ora", "ora"),
        ("anno", "anno"),
        ("mese", "mese"),
        ("stagione", "stagione"),
        ("primavera", "primavera"),
        ("estate", "estate"),
        ("autunno", "autunno"),
        ("inverno", "inverno"),
        # Luce
        ("luce", "luce"),
        ("ombra", "ombra"),
        ("alba", "alba"),
        ("tramonto", "tramonto"),
        ("aurora", "aurora"),
    ]
    return _curated_to_entries(terms, "astronomical")


def get_curated_medical_it():
    """~80 termini medici/anatomici in italiano medievale.

    Fonti: Trotula, Articella, trattati galenisti padovani.
    """
    terms = [
        # Anatomia
        ("sangue", "sangue"),
        ("cuore", "cuore"),
        ("fegato", "fegato"),
        ("polmone", "polmone"),
        ("stomaco", "stomaco"),
        ("milza", "milza"),
        ("rene", "rene"),
        ("cervello", "cervello"),
        ("testa", "testa"),
        ("occhio", "occhio"),
        ("orecchio", "orecchio"),
        ("naso", "naso"),
        ("bocca", "bocca"),
        ("lingua", "lingua"),
        ("dente", "dente"),
        ("gola", "gola"),
        ("collo", "collo"),
        ("spalla", "spalla"),
        ("braccio", "braccio"),
        ("mano", "mano"),
        ("dito", "dito"),
        ("petto", "petto"),
        ("ventre", "ventre"),
        ("osso", "osso"),
        ("carne", "carne"),
        ("pelle", "pelle"),
        ("nervo", "nervo"),
        ("vena", "vena"),
        ("gamba", "gamba"),
        ("piede", "piede"),
        ("ginocchio", "ginocchio"),
        ("costola", "costola"),
        ("schiena", "schiena"),
        # Patologia
        ("febbre", "febbre"),
        ("dolore", "dolore"),
        ("malattia", "malattia"),
        ("ferita", "ferita"),
        ("piaga", "piaga"),
        ("tumore", "tumore"),
        ("apostema", "apostema, ascesso"),
        ("catarro", "catarro"),
        ("tosse", "tosse"),
        ("gonfiore", "gonfiore"),
        ("infiammazione", "infiammazione"),
        # Terapia
        ("medicina", "medicina"),
        ("rimedio", "rimedio"),
        ("unguento", "unguento"),
        ("impiastro", "impiastro"),
        ("sciroppo", "sciroppo"),
        ("pillola", "pillola"),
        ("purga", "purga"),
        ("salasso", "salasso"),
        ("clistere", "clistere"),
        ("cura", "cura"),
        ("guarigione", "guarigione"),
        # Umori (teoria galenica)
        ("caldo", "caldo (qualita')"),
        ("freddo", "freddo (qualita')"),
        ("umido", "umido (qualita')"),
        ("secco", "secco (qualita')"),
        ("bile", "bile"),
        ("flegma", "flegma"),
        ("melancolia", "melancolia"),
        ("complessione", "complessione"),
        ("temperamento", "temperamento"),
        # Farmaceutica
        ("acqua", "acqua"),
        ("miele", "miele"),
        ("aceto", "aceto"),
        ("sale", "sale"),
        ("zucchero", "zucchero"),
        ("vino", "vino"),
        ("latte", "latte"),
        ("grasso", "grasso"),
        ("cera", "cera"),
        ("dose", "dose"),
        ("mistura", "mistura"),
        ("distillato", "distillato"),
        ("tintura", "tintura"),
        ("estratto", "estratto"),
    ]
    return _curated_to_entries(terms, "medical")


def get_curated_general_it():
    """~100 parole funzionali in italiano medievale.

    Articoli, preposizioni, congiunzioni, pronomi, numeri, verbi comuni.
    """
    terms = [
        # Articoli
        ("il", "il (articolo)"),
        ("lo", "lo (articolo)"),
        ("la", "la (articolo)"),
        ("le", "le (articolo)"),
        ("li", "li (articolo)"),
        ("un", "un (articolo)"),
        ("una", "una (articolo)"),
        # Preposizioni
        ("di", "di"),
        ("da", "da"),
        ("in", "in"),
        ("con", "con"),
        ("per", "per"),
        ("tra", "tra"),
        ("fra", "fra"),
        ("sopra", "sopra"),
        ("sotto", "sotto"),
        ("dentro", "dentro"),
        ("fuori", "fuori"),
        # Congiunzioni
        ("e", "e (congiunzione)"),
        ("o", "o (congiunzione)"),
        ("ma", "ma"),
        ("che", "che"),
        ("se", "se"),
        ("come", "come"),
        ("quando", "quando"),
        ("perche", "perche'"),
        ("poi", "poi"),
        ("anche", "anche"),
        # Pronomi e dimostrativi
        ("questo", "questo"),
        ("quello", "quello"),
        ("questa", "questa"),
        ("quella", "quella"),
        ("ogni", "ogni"),
        ("tutto", "tutto"),
        ("altro", "altro"),
        ("stesso", "stesso"),
        ("molto", "molto"),
        ("poco", "poco"),
        ("primo", "primo"),
        ("secondo", "secondo"),
        # Numeri
        ("uno", "uno"),
        ("due", "due"),
        ("tre", "tre"),
        ("quattro", "quattro"),
        ("cinque", "cinque"),
        ("sei", "sei"),
        ("sette", "sette"),
        ("otto", "otto"),
        ("nove", "nove"),
        ("dieci", "dieci"),
        ("venti", "venti"),
        ("trenta", "trenta"),
        ("cento", "cento"),
        ("mille", "mille"),
        # Verbi comuni (infinito + participio)
        ("fare", "fare"),
        ("dire", "dire"),
        ("dare", "dare"),
        ("prendere", "prendere"),
        ("mettere", "mettere"),
        ("essere", "essere"),
        ("avere", "avere"),
        ("potere", "potere"),
        ("dovere", "dovere"),
        ("volere", "volere"),
        ("sapere", "sapere"),
        ("vedere", "vedere"),
        ("trovare", "trovare"),
        ("usare", "usare"),
        ("bere", "bere"),
        ("mangiare", "mangiare"),
        ("lavare", "lavare"),
        ("bollire", "bollire"),
        ("cuocere", "cuocere"),
        ("seccare", "seccare"),
        ("pestare", "pestare"),
        ("tagliare", "tagliare"),
        ("macinare", "macinare"),
        ("mescolare", "mescolare"),
        # Sostantivi generici
        ("uomo", "uomo"),
        ("donna", "donna"),
        ("corpo", "corpo"),
        ("natura", "natura"),
        ("cosa", "cosa"),
        ("parte", "parte"),
        ("nome", "nome"),
        ("libro", "libro"),
        ("acqua", "acqua"),
        ("fuoco", "fuoco"),
        ("terra", "terra"),
        ("aria", "aria"),
        ("casa", "casa"),
        ("modo", "modo"),
        ("tempo", "tempo"),
        ("virtute", "virtu'"),
        ("proprieta", "proprieta'"),
        ("colore", "colore"),
        ("forma", "forma"),
        ("materia", "materia"),
    ]
    return _curated_to_entries(terms, "general")


def _curated_to_entries(terms, domain):
    """Converte lista curata (parola, glossa) in formato entry standard."""
    entries = []
    for word, gloss in terms:
        normalized = normalize_italian_phonemic(word)
        entries.append({
            "word": word,
            "phonemic": normalized,
            "gloss": gloss,
            "domain": domain,
        })
    return entries


# =====================================================================
# Sezione 2 — Normalizzazione fonemica
# =====================================================================

def normalize_italian_phonemic(word):
    """Normalizza una parola italiana alla forma fonemica attesa nel giudeo-italiano.

    Il giudeo-italiano scrive i suoni, non l'ortografia latina.
    Trasformazioni principali:
    - ch, gh → k, g
    - gl(i) → li
    - gn → ni
    - sc(i/e) → s
    - ci/ce → ts
    - c(a/o/u) → k
    - qu → ku
    - x → ks
    - doppia → singola (la scrittura ebraica non distingue)
    """
    w = word.lower().strip()

    # Trigrammi
    w = re.sub(r'gli', 'li', w)
    w = re.sub(r'sch', 'sk', w)
    w = re.sub(r'sci', 'si', w)
    w = re.sub(r'sce', 'se', w)

    # Digrammi
    w = re.sub(r'ch', 'k', w)
    w = re.sub(r'gh', 'g', w)
    w = re.sub(r'gn', 'ni', w)
    w = re.sub(r'qu', 'ku', w)
    w = re.sub(r'ci(?=[aeou])', 'ts', w)
    w = re.sub(r'ce', 'tse', w)
    w = re.sub(r'c([aou])', r'k\1', w)
    w = re.sub(r'c$', 'k', w)
    w = re.sub(r'c([^eiaoults])', r'k\1', w)
    w = re.sub(r'gi(?=[aeou])', 'dz', w)
    w = re.sub(r'ge', 'dze', w)
    w = re.sub(r'g([aou])', r'g\1', w)
    w = re.sub(r'ph', 'f', w)
    w = re.sub(r'th', 't', w)

    # Semplificazioni giudeo-italiane
    w = w.replace('x', 'ks')
    w = w.replace('j', 'i')

    # Rimuovi consonanti doppie (non distinte nella scrittura ebraica)
    w = re.sub(r'(.)\1+', r'\1', w)

    # h muta interna (dopo consonante) viene eliminata
    # ma h iniziale è spesso omessa in giudeo-italiano
    if w.startswith('h'):
        w = w[1:]
    w = re.sub(r'([^aeiou])h', r'\1', w)

    return w


def generate_spelling_variants(phonemic):
    """Genera varianti ortografiche medievali per una forma fonemica.

    Varianti comuni in toscano/veneto del XIV-XV sec.:
    - ct/tt (facto→fatto)
    - u/v interscambio
    - -tione/-zione
    - doppie/scempie
    - e/i oscillazione atona
    - o/u oscillazione atona
    """
    variants = {phonemic}

    # u ↔ o (oscillazione veneta)
    if 'u' in phonemic:
        variants.add(phonemic.replace('u', 'o', 1))
    if 'o' in phonemic:
        variants.add(phonemic.replace('o', 'u', 1))

    # e ↔ i (oscillazione atona)
    if 'e' in phonemic:
        variants.add(phonemic.replace('e', 'i', 1))
    if 'i' in phonemic:
        variants.add(phonemic.replace('i', 'e', 1))

    # -one → -one / -one (variante veneta -on)
    if phonemic.endswith('one'):
        variants.add(phonemic[:-1])  # -on

    # b ↔ v (lenizione tipica del giudeo-italiano)
    if 'b' in phonemic:
        variants.add(phonemic.replace('b', 'v', 1))
    if 'v' in phonemic:
        variants.add(phonemic.replace('v', 'b', 1))

    # f ↔ p (ambiguita' pe/fe)
    if 'f' in phonemic:
        variants.add(phonemic.replace('f', 'p', 1))
    if 'p' in phonemic:
        variants.add(phonemic.replace('p', 'f', 1))

    return variants


# =====================================================================
# Sezione 3 — Caricamento fonti esterne
# =====================================================================

def _strip_accents(s):
    """Rimuovi accenti mantenendo lettere base (à→a, è→e, etc.)."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def _is_valid_italian_word(word):
    """Verifica che la parola contenga solo lettere latine minuscole."""
    return bool(word) and all(c.isalpha() and c.isascii() for c in word)


def load_tlio_lemmas(lexicon_dir):
    """Carica lemmi dal TLIO lemmario.

    Filtri applicati:
    - Solo parole singole (no locuzioni con spazi)
    - Solo lettere latine (no numeri, simboli)
    - Lunghezza 2-12 caratteri (range utile per matching EVA)
    """
    path = Path(lexicon_dir) / "tlio_lemmario.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    seen = set()
    for item in data["lemmas"]:
        lemma = item["lemma"].strip().lower()
        # Salta locuzioni (con spazi)
        if ' ' in lemma:
            continue
        # Rimuovi accenti
        lemma = _strip_accents(lemma)
        # Solo lettere latine
        if not _is_valid_italian_word(lemma):
            continue
        # Lunghezza utile
        if not (2 <= len(lemma) <= 12):
            continue
        if lemma in seen:
            continue
        seen.add(lemma)

        phonemic = normalize_italian_phonemic(lemma)
        if len(phonemic) < 2:
            continue

        # Assegna dominio da POS se possibile
        pos = item.get("pos", "")
        entries.append({
            "word": lemma,
            "phonemic": phonemic,
            "gloss": f"{lemma} ({pos})" if pos else lemma,
            "domain": "tlio",
        })
    return entries


def load_dante_words(lexicon_dir):
    """Carica parole dalla Divina Commedia.

    Filtri: solo lettere latine, lunghezza 2-12, frequenza >= 2.
    """
    path = Path(lexicon_dir) / "dante_wordlist.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    seen = set()
    for item in data["words"]:
        word = item["word"].strip().lower()
        count = item.get("count", 1)
        # Solo parole con frequenza >= 2
        if count < 2:
            continue
        word = _strip_accents(word)
        if not _is_valid_italian_word(word):
            continue
        if not (2 <= len(word) <= 12):
            continue
        if word in seen:
            continue
        seen.add(word)

        phonemic = normalize_italian_phonemic(word)
        if len(phonemic) < 2:
            continue

        entries.append({
            "word": word,
            "phonemic": phonemic,
            "gloss": f"{word} (Dante, freq={count})",
            "domain": "dante",
        })
    return entries


def load_kaikki_archaic(lexicon_dir):
    """Carica forme arcaiche/obsolete da Kaikki.org (Wiktionary).

    Tutte le entries sono gia' filtrate per tag archaic/obsolete/historical.
    """
    path = Path(lexicon_dir) / "kaikki_archaic_italian.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    seen = set()
    for item in data["words"]:
        word = item["word"].strip().lower()
        word = _strip_accents(word)
        if not _is_valid_italian_word(word):
            continue
        if not (2 <= len(word) <= 12):
            continue
        if word in seen:
            continue
        seen.add(word)

        phonemic = normalize_italian_phonemic(word)
        if len(phonemic) < 2:
            continue

        pos = item.get("pos", "")
        tags = item.get("tags", [])
        entries.append({
            "word": word,
            "phonemic": phonemic,
            "gloss": f"{word} ({pos}, {'/'.join(tags[:2])})",
            "domain": "kaikki",
        })
    return entries


# =====================================================================
# Sezione 4 — Assemblaggio lessico
# =====================================================================

def build_italian_lexicon(curated_entries, external_entries=None):
    """Assembla lessico italiano con varianti fonemiche.

    Ogni entry curata viene:
    1. Normalizzata fonemicamente
    2. Espansa in varianti ortografiche
    3. Organizzata per dominio

    Le entry esterne (TLIO, Dante, Kaikki) vengono aggiunte senza varianti
    (sono gia' forme attestate), ma solo se la forma fonemica non e' gia'
    coperta dai termini curati (priorita' ai curati per le glosse).

    Returns: {
        "by_domain": {domain: [entries]},
        "all_forms": sorted list of all phonemic forms,
        "form_to_gloss": {form: gloss},
        "form_to_domain": {form: domain},
        "stats": {...}
    }
    """
    by_domain = defaultdict(list)
    form_to_gloss = {}
    form_to_domain = {}
    all_forms = set()

    # --- Fase 1: termini curati (con varianti, priorita' alta) ---
    for entry in curated_entries:
        base = entry["phonemic"]
        domain = entry["domain"]
        gloss = entry["gloss"]
        word = entry["word"]

        # Genera varianti
        variants = generate_spelling_variants(base)

        for variant in variants:
            if len(variant) < 2:
                continue
            all_forms.add(variant)
            if variant not in form_to_gloss:
                form_to_gloss[variant] = gloss
                form_to_domain[variant] = domain
            by_domain[domain].append({
                "word": word,
                "phonemic": variant,
                "gloss": gloss,
                "is_variant": variant != base,
            })

    curated_forms = set(all_forms)

    # --- Fase 2: fonti esterne (senza varianti, solo forme nuove) ---
    if external_entries:
        for entry in external_entries:
            phonemic = entry["phonemic"]
            domain = entry["domain"]
            gloss = entry["gloss"]
            word = entry["word"]

            if len(phonemic) < 2:
                continue
            # Aggiungi solo se non gia' nel curato
            all_forms.add(phonemic)
            if phonemic not in form_to_gloss:
                form_to_gloss[phonemic] = gloss
                form_to_domain[phonemic] = domain
            by_domain[domain].append({
                "word": word,
                "phonemic": phonemic,
                "gloss": gloss,
                "is_variant": False,
            })

    # Deduplica per dominio
    for domain in by_domain:
        seen = set()
        unique = []
        for entry in by_domain[domain]:
            key = entry["phonemic"]
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        by_domain[domain] = unique

    sorted_forms = sorted(all_forms)

    # Statistiche
    lengths = Counter(len(f) for f in sorted_forms)
    initial_chars = Counter(f[0] for f in sorted_forms if f)

    # Frequenze lettere nel lessico
    all_chars = "".join(sorted_forms)
    char_freq = Counter(all_chars)
    total_c = len(all_chars) or 1
    char_dist = {
        k: round(v / total_c, 4)
        for k, v in sorted(char_freq.items(), key=lambda x: -x[1])
    }

    stats = {
        "total_forms": len(sorted_forms),
        "curated_forms": len(curated_forms),
        "external_forms": len(all_forms) - len(curated_forms),
        "by_domain_count": {d: len(entries) for d, entries in by_domain.items()},
        "length_distribution": dict(sorted(lengths.items())),
        "initial_char_distribution": dict(
            sorted(initial_chars.items(), key=lambda x: -x[1])
        ),
        "letter_frequencies": char_dist,
    }

    return {
        "by_domain": dict(by_domain),
        "all_forms": sorted_forms,
        "form_to_gloss": form_to_gloss,
        "form_to_domain": form_to_domain,
        "stats": stats,
    }


# =====================================================================
# Sezione 5 — Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per preparazione lessico italiano medievale."""
    print_header("VOYNICH TOOLKIT - Preparazione Lessico Italiano Medievale")
    config.ensure_dirs()

    lexicon_path = config.lexicon_dir / "italian_lexicon.json"
    report_path = config.stats_dir / "italian_lexicon_report.json"

    if lexicon_path.exists() and not force:
        click.echo("  Lessico italiano gia' presente, skip (usa --force)")
        return

    # 1. Vocabolario curato
    print_step("Caricamento vocabolario italiano curato...")
    curated = []

    botanical = get_curated_botanical_it()
    curated.extend(botanical)
    click.echo(f"    Botanico: {len(botanical)} termini")

    astronomical = get_curated_astronomical_it()
    curated.extend(astronomical)
    click.echo(f"    Astronomico: {len(astronomical)} termini")

    medical = get_curated_medical_it()
    curated.extend(medical)
    click.echo(f"    Medico/anatomico: {len(medical)} termini")

    general = get_curated_general_it()
    curated.extend(general)
    click.echo(f"    Generale: {len(general)} termini")
    click.echo(f"    Totale curato: {len(curated)} termini")

    # 2. Fonti esterne
    print_step("Caricamento fonti esterne...")
    external = []

    tlio = load_tlio_lemmas(config.lexicon_dir)
    external.extend(tlio)
    click.echo(f"    TLIO lemmario: {len(tlio)} lemmi" if tlio
               else "    TLIO lemmario: non disponibile (esegui scraping)")

    dante = load_dante_words(config.lexicon_dir)
    external.extend(dante)
    click.echo(f"    Divina Commedia: {len(dante)} forme" if dante
               else "    Divina Commedia: non disponibile")

    kaikki = load_kaikki_archaic(config.lexicon_dir)
    external.extend(kaikki)
    click.echo(f"    Kaikki arcaico: {len(kaikki)} forme" if kaikki
               else "    Kaikki arcaico: non disponibile")

    click.echo(f"    Totale esterno: {len(external)} entries")

    # 3. Assemblaggio con varianti
    print_step("Assemblaggio lessico con varianti fonemiche...")
    lexicon = build_italian_lexicon(curated, external_entries=external)
    stats = lexicon["stats"]
    click.echo(f"    {stats['total_forms']} forme fonemiche uniche "
               f"({stats['curated_forms']} curate + "
               f"{stats['external_forms']} da fonti esterne)")

    # 4. Salvataggio
    print_step("Salvataggio lessico...")

    save_data = {
        "by_domain": lexicon["by_domain"],
        "all_forms": lexicon["all_forms"],
        "form_to_gloss": lexicon["form_to_gloss"],
        "hebrew_to_italian": HEBREW_TO_ITALIAN,
        "hebrew_alternatives": {k: v for k, v in HEBREW_ALTERNATIVES.items()},
        "stats": stats,
    }
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    click.echo(f"    Lessico: {lexicon_path}")

    report = {
        "stats": stats,
        "italian_letter_freqs": ITALIAN_LETTER_FREQS,
        "italian_common_bigrams": ITALIAN_COMMON_BIGRAMS,
        "hebrew_to_italian": HEBREW_TO_ITALIAN,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report:  {report_path}")

    # === Sommario ===
    click.echo(f"\n{'=' * 60}")
    click.echo("  LESSICO ITALIANO MEDIEVALE")
    click.echo(f"{'=' * 60}")

    click.echo(f"\n  Per dominio:")
    for domain, count in sorted(stats["by_domain_count"].items(),
                                 key=lambda x: -x[1]):
        click.echo(f"    {domain:15s} {count:5d} forme")

    click.echo(f"\n  Totale: {stats['total_forms']} forme fonemiche uniche")
    click.echo(f"    Curate:  {stats['curated_forms']}")
    click.echo(f"    Esterne: {stats['external_forms']}")

    click.echo(f"\n  Distribuzione lunghezze:")
    for length, count in sorted(stats["length_distribution"].items()):
        bar = "#" * min(count, 40)
        click.echo(f"    {length} lettere: {count:4d} {bar}")

    click.echo(f"\n  Top 10 lettere nel lessico:")
    for char, freq in list(stats["letter_frequencies"].items())[:10]:
        bar = "#" * int(freq * 100)
        click.echo(f"    {char}  {freq:.3f} {bar}")

    click.echo(f"\n  Mappatura ebraico → italiano:")
    for heb, ita in sorted(HEBREW_TO_ITALIAN.items()):
        from .prepare_lexicon import CONSONANT_NAMES
        name = CONSONANT_NAMES.get(heb, "?")
        click.echo(f"    {heb} ({name:8s}) → {ita}")
