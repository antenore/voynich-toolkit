"""
Preparazione lessico ebraico medievale per tentativo di decifrazione.

Scarica e processa vocabolari ebraici strutturati (STEPBible TBESH),
integra con glossari medievali curati (ibn Ezra, lessico botanico-medico),
e produce un lessico consonantale organizzato per dominio
(botanico, farmaceutico, astronomico, anatomico).
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import click
import requests

from .config import ToolkitConfig
from .utils import print_header, print_step, timer


# =====================================================================
# Costanti: mappatura consonanti ebraiche
# =====================================================================

# Ogni consonante ebraica → singolo carattere ASCII univoco.
# Lowercase = lettere "normali", uppercase = lettere enfatiche/distinte.
# Questo permette confronto diretto con stringhe EVA.
HEBREW_TO_ASCII = {
    '\u05D0': 'A',   # א aleph
    '\u05D1': 'b',   # ב bet
    '\u05D2': 'g',   # ג gimel
    '\u05D3': 'd',   # ד dalet
    '\u05D4': 'h',   # ה he
    '\u05D5': 'w',   # ו vav
    '\u05D6': 'z',   # ז zayin
    '\u05D7': 'X',   # ח chet (X per distinguere da h)
    '\u05D8': 'J',   # ט tet (J per distinguere da t)
    '\u05D9': 'y',   # י yod
    '\u05DA': 'k',   # ך kaf sofit
    '\u05DB': 'k',   # כ kaf
    '\u05DC': 'l',   # ל lamed
    '\u05DD': 'm',   # ם mem sofit
    '\u05DE': 'm',   # מ mem
    '\u05DF': 'n',   # ן nun sofit
    '\u05E0': 'n',   # נ nun
    '\u05E1': 's',   # ס samekh
    '\u05E2': 'E',   # ע ayin (E per gutturale)
    '\u05E3': 'p',   # ף pe sofit
    '\u05E4': 'p',   # פ pe
    '\u05E5': 'C',   # ץ tsade sofit
    '\u05E6': 'C',   # צ tsade
    '\u05E7': 'q',   # ק qof
    '\u05E8': 'r',   # ר resh
    '\u05E9': 'S',   # ש shin/sin (S per distinguere da samekh)
    '\u05EA': 't',   # ת tav
}

# Nomi leggibili per i 22 fonemi consonantali
CONSONANT_NAMES = {
    'A': 'aleph', 'b': 'bet', 'g': 'gimel', 'd': 'dalet',
    'h': 'he', 'w': 'vav', 'z': 'zayin', 'X': 'chet',
    'J': 'tet', 'y': 'yod', 'k': 'kaf', 'l': 'lamed',
    'm': 'mem', 'n': 'nun', 's': 'samekh', 'E': 'ayin',
    'p': 'pe', 'C': 'tsade', 'q': 'qof', 'r': 'resh',
    'S': 'shin', 't': 'tav',
}

# Range unicode per nikkud (punti vocalici) e segni cantillazione
NIKKUD_RANGE = set(range(0x0591, 0x05BE)) | {0x05BF} | \
               set(range(0x05C1, 0x05C3)) | set(range(0x05C4, 0x05C8))

# URL risorse
STEPBIBLE_URLS = [
    "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/"
    "Lexicons/TBESH%20-%20Translators%20Brief%20lexicon%20of%20Extended"
    "%20Strongs%20for%20Hebrew%20-%20STEPBible.org%20CC%20BY.txt",
    "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master/"
    "Older%20Formats/TBESH%20-%20Translators%20Brief%20lexicon%20of%20"
    "Extended%20Strongs%20for%20Hebrew%20-%20STEPBible.org%20CC%20BY.txt",
]


# =====================================================================
# Sezione 1 — Elaborazione testo ebraico
# =====================================================================

def strip_nikkud(text: str) -> str:
    """Rimuove punti vocalici e segni cantillazione dal testo ebraico."""
    return "".join(ch for ch in text if ord(ch) not in NIKKUD_RANGE)


def hebrew_to_consonants(text: str) -> str:
    """Converte testo ebraico in stringa consonantale ASCII.

    Rimuove nikkud, mappa consonanti (incluse forme finali),
    scarta qualsiasi carattere non-ebraico.
    """
    stripped = strip_nikkud(text)
    return "".join(HEBREW_TO_ASCII.get(ch, "") for ch in stripped)


def is_hebrew_char(ch: str) -> bool:
    """True se il carattere e' una lettera ebraica (U+05D0-U+05EA)."""
    return '\u05D0' <= ch <= '\u05EA'


# =====================================================================
# Sezione 2 — Download e parsing STEPBible
# =====================================================================

@timer
def download_stepbible(dest_dir: Path) -> Path:
    """Scarica il lessico STEPBible TBESH."""
    dest = dest_dir / "stepbible_tbesh.txt"
    if dest.exists():
        print(f"    Cache trovata: {dest}")
        return dest

    for url in STEPBIBLE_URLS:
        try:
            print(f"    Download da STEPBible...")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 1000:
                dest.write_text(resp.text, encoding="utf-8")
                print(f"    Salvato: {dest} ({len(resp.text)} bytes)")
                return dest
        except requests.RequestException:
            continue

    raise click.ClickException(
        "Impossibile scaricare STEPBible TBESH. Controlla la connessione."
    )


@timer
def parse_stepbible(filepath: Path) -> list[dict]:
    """Parsa il lessico TBESH in lista di entry strutturate."""
    entries = []
    text = filepath.read_text(encoding="utf-8")

    # Il formato TBESH usa tab come separatore.
    # Le righe dati hanno tipicamente:
    # Key\tExtended\tHebrew\tTranslit\tPOS\tRoot\tGloss...
    current_entry = {}
    for line in text.split("\n"):
        line = line.strip()

        # Skip commenti e righe vuote
        if not line or line.startswith("#") or line.startswith("="):
            continue

        # Righe header con $ definiscono la struttura
        if line.startswith("$"):
            continue

        # Cerchiamo righe con numeri Strong (H0001, H0002, ...)
        # e/o testo ebraico
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        # Cerca il campo con testo ebraico e il campo con la glossa
        hebrew_text = ""
        gloss = ""
        strong_num = ""
        transliteration = ""

        for part in parts:
            part = part.strip()
            # Numero Strong
            if re.match(r'^[HAG]\d{4}', part):
                strong_num = part
            # Testo ebraico (contiene almeno una lettera ebraica)
            elif any(is_hebrew_char(ch) for ch in part) and not hebrew_text:
                hebrew_text = part
            # Translitterazione (ASCII con segni diacritici)
            elif (re.match(r'^[a-zA-Z]', part) and len(part) > 1
                  and not strong_num == part):
                if not transliteration:
                    transliteration = part
                elif not gloss:
                    gloss = part
                else:
                    # Aggiungi al gloss se e' piu' lungo (probabilmente la definizione)
                    if len(part) > len(gloss):
                        gloss = part

        if hebrew_text and (gloss or transliteration):
            consonants = hebrew_to_consonants(hebrew_text)
            if consonants:  # almeno una consonante
                entries.append({
                    "hebrew": hebrew_text,
                    "consonants": consonants,
                    "gloss": gloss or transliteration,
                    "strong": strong_num,
                    "source": "STEPBible",
                })

    return entries


# =====================================================================
# Sezione 3 — Vocabolario medievale curato
# =====================================================================

def get_curated_botanical() -> list[dict]:
    """Termini botanici dall'ebraico biblico e medievale.

    Fonti: STEPBible domini semantici, Jewish Encyclopedia,
    Melbourne Plant Names, glossari Bos & Mensching.
    """
    # (ebraico, consonanti, significato)
    terms = [
        # Alberi e piante principali
        ("עֵץ", "EC", "albero, legno"),
        ("פְּרִי", "pry", "frutto"),
        ("זֶרַע", "zrE", "seme"),
        ("עָלֶה", "Elh", "foglia"),
        ("שֹׁרֶשׁ", "SrS", "radice"),
        ("עָנָף", "Enp", "ramo"),
        ("פֶּרַח", "prX", "fiore"),
        ("נֵץ", "nC", "bocciolo, fiore"),
        # Alberi da frutto
        ("גֶּפֶן", "gpn", "vite"),
        ("תְּאֵנָה", "tAnh", "fico"),
        ("זַיִת", "zyt", "olivo"),
        ("תָּמָר", "tmr", "palma da dattero"),
        ("רִמּוֹן", "rmwn", "melograno"),
        ("תַּפּוּחַ", "tpwX", "melo"),
        ("אֶגוֹז", "Agwz", "noce"),
        ("שָׁקֵד", "Sqd", "mandorlo"),
        ("אֵלָה", "Alh", "terebinto"),
        ("אַלּוֹן", "Alwn", "quercia"),
        ("אֶרֶז", "Arz", "cedro"),
        ("בְּרוֹשׁ", "brwS", "cipresso"),
        # Cereali e colture
        ("חִטָּה", "XJh", "grano"),
        ("שְׂעֹרָה", "SErh", "orzo"),
        ("כֻּסֶּמֶת", "ksmt", "spelta"),
        ("שִׁבֹּלֶת", "Sblt", "spiga"),
        ("דֹּחַן", "dXn", "miglio"),
        # Erbe e piante medicinali
        ("עֵשֶׂב", "ESb", "erba"),
        ("חָצִיר", "XCyr", "erba, fieno"),
        ("שׁוּשַׁן", "SwSn", "giglio, iris"),
        ("וֶרֶד", "wrd", "rosa"),
        ("לַעֲנָה", "lEnh", "assenzio"),
        ("אֵזוֹב", "Azwb", "issopo"),
        ("מֹר", "mr", "mirra"),
        ("לְבוֹנָה", "lbwnh", "incenso"),
        ("כַּרְכֹּם", "krkm", "zafferano, croco"),
        ("קִנָּמוֹן", "qnmwn", "cannella"),
        ("נֵרְדְּ", "nrd", "nardo"),
        ("אָהֳלִים", "Ahlym", "aloe"),
        ("קִדָּה", "qdh", "cassia"),
        ("גַּד", "gd", "coriandolo"),
        ("כַּמֹּן", "kmn", "cumino"),
        ("קֶצַח", "qCX", "cumino nero"),
        # Verdure e ortaggi
        ("שׁוּם", "Swm", "aglio"),
        ("בָּצָל", "bCl", "cipolla"),
        ("כְּרֵשָׁה", "krSh", "porro"),
        ("קִשֻּׁאָה", "qSwAh", "cetriolo"),
        ("אֲבַטִּיחַ", "AbJyX", "cocomero"),
        # Parti di pianta
        ("קְלִפָּה", "qlph", "corteccia, buccia"),
        ("עָלִים", "Elym", "foglie"),
        ("גִּזְעָה", "gzEh", "tronco"),
        ("קָנֶה", "qnh", "canna, stelo"),
        ("נִצָּן", "nCn", "bocciolo"),
        # Termini medievali (glossari italiani Bos & Mensching)
        ("סַם", "sm", "droga, ingrediente"),
        ("עִקָּר", "Eqr", "radice (medicinale)"),
        ("צֶמַח", "CmX", "pianta, germoglio"),
    ]
    return _curated_to_entries(terms, "botanical", "Curato-Botanico")


def get_curated_astronomical() -> list[dict]:
    """Termini astronomici da ibn Ezra e tradizione medievale.

    Fonte primaria: Shlomo Sela, Abraham ibn Ezra's Introductions to Astrology.
    """
    terms = [
        # Pianeti
        ("שַׁבְּתַאי", "SbtAy", "Saturno"),
        ("צֶדֶק", "Cdq", "Giove"),
        ("מַאְדִּים", "mAdym", "Marte"),
        ("חַמָּה", "Xmh", "Sole"),
        ("נוֹגַהּ", "nwgh", "Venere"),
        ("כּוֹכָב", "kwkb", "Mercurio"),
        ("לְבָנָה", "lbnh", "Luna"),
        # Zodiaco
        ("מַזָּלוֹת", "mzlwt", "segni zodiacali"),
        ("מַזָּל", "mzl", "segno zodiacale"),
        ("טָלֶה", "Jlh", "Ariete"),
        ("שׁוֹר", "Swr", "Toro"),
        ("תְּאוֹמִים", "tAwmym", "Gemelli"),
        ("סַרְטָן", "srJn", "Cancro"),
        ("אַרְיֵה", "Aryh", "Leone"),
        ("בְּתוּלָה", "btwlh", "Vergine"),
        ("מֹאזְנַיִם", "mAznym", "Bilancia"),
        ("עַקְרָב", "Eqrb", "Scorpione"),
        ("קֶשֶׁת", "qSt", "Sagittario"),
        ("גְּדִי", "gdy", "Capricorno"),
        ("דְּלִי", "dly", "Acquario"),
        ("דָּגִים", "dgym", "Pesci"),
        # Stelle e costellazioni
        ("כּוֹכָבִים", "kwkbym", "stelle"),
        ("כִּימָה", "kymh", "Pleiadi"),
        ("כְּסִיל", "ksyl", "Orione"),
        ("עָשׁ", "ES", "Orsa Maggiore"),
        ("מַזָּרוֹת", "mzrwt", "costellazioni"),
        # Concetti astronomici
        ("גַּלְגַּל", "glgl", "sfera, ciclo"),
        ("רָקִיעַ", "rqyE", "firmamento"),
        ("תְּקוּפָה", "tqwph", "solstizio, equinozio"),
        ("מוֹלָד", "mwld", "novilunio"),
        ("לִקּוּי", "lqwy", "eclisse"),
        ("מַעֲלָה", "mElh", "grado (astronomico)"),
        ("מֶרְחָק", "mrXq", "distanza"),
        ("בַּיִת", "byt", "casa (astrologica)"),
        ("גֹּבַהּ", "gbh", "altezza, elevazione"),
        ("נְטִיָּה", "nJyh", "declinazione"),
        ("מַחֲזוֹר", "mXzwr", "ciclo"),
        # Direzioni
        ("מִזְרָח", "mzrX", "est"),
        ("מַעֲרָב", "mErb", "ovest"),
        ("צָפוֹן", "Cpwn", "nord"),
        ("דָּרוֹם", "drwm", "sud"),
        # Tempo
        ("יוֹם", "ywm", "giorno"),
        ("לַיְלָה", "lylh", "notte"),
        ("שָׁעָה", "SEh", "ora"),
        ("שָׁנָה", "Snh", "anno"),
        ("חֹדֶשׁ", "XdS", "mese"),
    ]
    return _curated_to_entries(terms, "astronomical", "Curato-IbnEzra")


def get_curated_medical() -> list[dict]:
    """Termini medici e anatomici dall'ebraico biblico e medievale.

    Fonti: Jastrow, Bos Concise Dictionary, Donnolo Sefer Hakhmoni.
    """
    terms = [
        # Corpo umano
        ("רֹאשׁ", "rAS", "testa"),
        ("פָּנִים", "pnym", "viso"),
        ("עַיִן", "Eyn", "occhio"),
        ("אֹזֶן", "Azn", "orecchio"),
        ("אַף", "Ap", "naso"),
        ("פֶּה", "ph", "bocca"),
        ("לָשׁוֹן", "lSwn", "lingua"),
        ("שֵׁן", "Sn", "dente"),
        ("גָּרוֹן", "grwn", "gola"),
        ("צַוָּאר", "CwAr", "collo"),
        ("כָּתֵף", "ktp", "spalla"),
        ("זְרוֹעַ", "zrwE", "braccio"),
        ("יָד", "yd", "mano"),
        ("אֶצְבַּע", "ACbE", "dito"),
        ("חָזֶה", "Xzh", "petto"),
        ("לֵב", "lb", "cuore"),
        ("רֵאָה", "rAh", "polmone"),
        ("כָּבֵד", "kbd", "fegato"),
        ("כִּלְיָה", "klyh", "rene"),
        ("מֵעֶה", "mEh", "intestino"),
        ("קֵבָה", "qbh", "stomaco"),
        ("טְחוֹל", "JXwl", "milza"),
        ("בֶּטֶן", "bJn", "ventre"),
        ("גָּב", "gb", "schiena"),
        ("עֶצֶם", "ECm", "osso"),
        ("בָּשָׂר", "bSr", "carne"),
        ("עוֹר", "Ewr", "pelle"),
        ("דָּם", "dm", "sangue"),
        ("גִּיד", "gyd", "tendine, nervo"),
        ("רֶגֶל", "rgl", "piede, gamba"),
        ("בֶּרֶךְ", "brk", "ginocchio"),
        ("יָרֵךְ", "yrk", "coscia"),
        ("מֹחַ", "mX", "cervello"),
        ("גֻּלְגֹּלֶת", "glglt", "cranio"),
        ("צֶלַע", "ClE", "costola"),
        # Termini medici
        ("רְפוּאָה", "rpwAh", "medicina, cura"),
        ("חֹלִי", "Xly", "malattia"),
        ("רוֹפֵא", "rwpA", "medico"),
        ("מַכָּה", "mkh", "ferita"),
        ("כְּאֵב", "kAb", "dolore"),
        ("חֹם", "Xm", "febbre, calore"),
        ("קַר", "qr", "freddo"),
        ("לַח", "lX", "umido"),
        ("יָבֵשׁ", "ybS", "secco"),
        # Quattro umori (medicina medievale)
        ("דָּם", "dm", "sangue (umore)"),
        ("מָרָה", "mrh", "bile"),
        ("לֵחָה", "lXh", "flegma"),
        ("שְׁחוֹרָה", "SXwrh", "bile nera, melancolia"),
        # Farmaceutica
        ("סַם", "sm", "farmaco, droga"),
        ("תְּרוּפָה", "trwph", "rimedio"),
        ("מִרְקַחַת", "mrqXt", "unguento"),
        ("מַשְׁחָה", "mSXh", "unguento, olio"),
        ("רְטִיָּה", "rJyh", "impiastro"),
        ("שֶׁמֶן", "Smn", "olio"),
        ("מַיִם", "mym", "acqua"),
        ("דְּבַשׁ", "dbS", "miele"),
        ("חֹמֶץ", "XmC", "aceto"),
        ("מֶלַח", "mlX", "sale"),
    ]
    return _curated_to_entries(terms, "medical", "Curato-Medico")


def get_curated_general() -> list[dict]:
    """Parole funzionali e generiche ebraiche ad alta frequenza.

    Articoli, preposizioni, pronomi — importanti perche' le parole piu'
    frequenti nel testo Voynich sono probabilmente parole funzionali.
    """
    terms = [
        # Articoli e preposizioni (prefissi in ebraico)
        ("הַ", "h", "il/la (articolo)"),
        ("בְּ", "b", "in"),
        ("לְ", "l", "a, per"),
        ("מִ", "m", "da, di"),
        ("כְּ", "k", "come"),
        ("וְ", "w", "e (congiunzione)"),
        ("שֶׁ", "S", "che (relativo)"),
        # Pronomi
        ("הוּא", "hwA", "egli"),
        ("הִיא", "hyA", "ella"),
        ("אֲנִי", "Any", "io"),
        ("אַתָּה", "Ath", "tu (m.)"),
        ("הֵם", "hm", "essi"),
        # Parole comuni
        ("כֹּל", "kl", "tutto"),
        ("אֶחָד", "AXd", "uno"),
        ("שְׁנַיִם", "Snym", "due"),
        ("שָׁלֹשׁ", "SlS", "tre"),
        ("אַרְבַּע", "ArbE", "quattro"),
        ("חָמֵשׁ", "XmS", "cinque"),
        ("שֵׁשׁ", "SS", "sei"),
        ("שֶׁבַע", "SbE", "sette"),
        ("שְׁמֹנֶה", "Smnh", "otto"),
        ("תֵּשַׁע", "tSE", "nove"),
        ("עֶשֶׂר", "ESr", "dieci"),
        ("מֵאָה", "mAh", "cento"),
        ("אֶלֶף", "Alp", "mille"),
        # Verbi comuni
        ("עָשָׂה", "ESh", "fare"),
        ("אָמַר", "Amr", "dire"),
        ("נָתַן", "ntn", "dare"),
        ("לָקַח", "lqX", "prendere"),
        ("הָלַךְ", "hlk", "andare"),
        ("בָּא", "bA", "venire"),
        ("יָדַע", "ydE", "sapere"),
        ("רָאָה", "rAh", "vedere"),
        ("שָׁמַע", "SmE", "sentire"),
        ("כָּתַב", "ktb", "scrivere"),
        ("קָרָא", "qrA", "leggere, chiamare"),
        # Sostantivi generici
        ("אִישׁ", "AyS", "uomo"),
        ("אִשָּׁה", "ASh", "donna"),
        ("בֵּן", "bn", "figlio"),
        ("בַּת", "bt", "figlia"),
        ("אָב", "Ab", "padre"),
        ("אֵם", "Am", "madre"),
        ("בַּיִת", "byt", "casa"),
        ("עִיר", "Eyr", "citta'"),
        ("אֶרֶץ", "ArC", "terra"),
        ("שָׁמַיִם", "Smym", "cielo"),
        ("מַיִם", "mym", "acqua"),
        ("אֵשׁ", "AS", "fuoco"),
        ("רוּחַ", "rwX", "vento, spirito"),
        ("אוֹר", "Awr", "luce"),
        ("חֹשֶׁךְ", "XSk", "oscurita'"),
        ("דֶּרֶךְ", "drk", "strada, via"),
        ("דָּבָר", "dbr", "parola, cosa"),
        ("שֵׁם", "Sm", "nome"),
        ("סֵפֶר", "spr", "libro"),
    ]
    return _curated_to_entries(terms, "general", "Curato-Generale")


def _curated_to_entries(terms, domain, source):
    """Converte lista curata in formato entry standard."""
    entries = []
    for hebrew, consonants, gloss in terms:
        # Ricalcola consonanti dal testo ebraico per verifica
        computed = hebrew_to_consonants(hebrew)
        entries.append({
            "hebrew": strip_nikkud(hebrew),
            "consonants": computed if computed else consonants,
            "gloss": gloss,
            "strong": "",
            "source": source,
            "domain": domain,
        })
    return entries


# =====================================================================
# Sezione 4 — Classificazione per dominio
# =====================================================================

# Keywords per classificazione automatica delle glosse STEPBible
DOMAIN_KEYWORDS = {
    "botanical": [
        "tree", "plant", "herb", "seed", "fruit", "flower", "vine",
        "wheat", "barley", "grain", "leaf", "root", "branch", "thorn",
        "fig", "olive", "palm", "cedar", "oak", "pomegranate", "almond",
        "spice", "incense", "myrrh", "cinnamon", "hyssop", "reed",
        "garden", "field", "sow", "harvest", "crop", "grass", "weed",
        "blossom", "sprout", "wood", "timber", "forest", "bush",
        "garlic", "onion", "leek", "cucumber", "melon", "vegetable",
        "rose", "lily", "aloe", "balm", "gum", "resin",
    ],
    "astronomical": [
        "star", "sun", "moon", "heaven", "sky", "constellation",
        "planet", "zodiac", "eclipse", "equinox", "solstice",
        "east", "west", "north", "south", "firmament", "sphere",
        "light", "darkness", "day", "night", "year", "month",
        "season", "time", "cycle", "orbit", "degree", "sign",
        "morning", "evening", "dawn", "dusk", "hour",
    ],
    "medical": [
        "body", "head", "eye", "ear", "nose", "mouth", "tongue",
        "tooth", "throat", "neck", "shoulder", "arm", "hand", "finger",
        "chest", "heart", "lung", "liver", "kidney", "stomach",
        "belly", "bone", "flesh", "skin", "blood", "vein", "nerve",
        "foot", "leg", "knee", "thigh", "brain", "skull", "rib",
        "heal", "sick", "disease", "wound", "pain", "fever",
        "medicine", "remedy", "ointment", "physician", "cure",
        "bile", "phlegm", "humor", "hot", "cold", "moist", "dry",
        "drug", "poison", "oil", "honey", "vinegar", "salt",
    ],
}


def classify_domain(gloss: str) -> list[str]:
    """Classifica un'entry per dominio basandosi sulla glossa inglese."""
    if not gloss:
        return ["general"]

    gloss_lower = gloss.lower()
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in gloss_lower:
                domains.append(domain)
                break

    return domains if domains else ["general"]


# =====================================================================
# Sezione 5 — Assemblaggio lessico
# =====================================================================

@timer
def build_lexicon(stepbible_entries: list[dict],
                  curated_entries: list[dict]) -> dict:
    """Assembla il lessico finale organizzato per dominio.

    Returns: {
        "by_domain": { domain: [entries] },
        "by_consonants": { consonantal_string: [entries] },
        "all_consonantal_forms": [sorted list],
        "stats": { counts, unique forms, etc. }
    }
    """
    # Classifica le entry STEPBible
    for entry in stepbible_entries:
        if "domain" not in entry:
            entry["domain"] = classify_domain(entry.get("gloss", ""))[0]

    all_entries = stepbible_entries + curated_entries

    # Deduplica per (consonanti, dominio)
    seen = set()
    unique_entries = []
    for entry in all_entries:
        key = (entry["consonants"], entry.get("domain", "general"))
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    # Organizza per dominio
    by_domain = defaultdict(list)
    for entry in unique_entries:
        domain = entry.get("domain", "general")
        by_domain[domain].append({
            "hebrew": entry["hebrew"],
            "consonants": entry["consonants"],
            "gloss": entry["gloss"],
            "source": entry["source"],
        })

    # Organizza per stringa consonantale (reverse lookup)
    by_consonants = defaultdict(list)
    for entry in unique_entries:
        by_consonants[entry["consonants"]].append({
            "hebrew": entry["hebrew"],
            "gloss": entry["gloss"],
            "domain": entry.get("domain", "general"),
            "source": entry["source"],
        })

    # Set di tutte le forme consonantali (per lookup rapido)
    all_forms = sorted(set(entry["consonants"] for entry in unique_entries))

    # Distribuzione lunghezze consonantali
    lengths = Counter(len(c) for c in all_forms)

    # Distribuzione consonanti iniziali
    initial_consonants = Counter(c[0] for c in all_forms if c)

    stats = {
        "total_entries": len(unique_entries),
        "unique_consonantal_forms": len(all_forms),
        "by_domain_count": {d: len(entries) for d, entries in by_domain.items()},
        "length_distribution": dict(sorted(lengths.items())),
        "initial_consonant_distribution": dict(
            sorted(initial_consonants.items(), key=lambda x: -x[1])
        ),
        "sources": dict(Counter(e["source"] for e in unique_entries).most_common()),
    }

    return {
        "by_domain": dict(by_domain),
        "by_consonants": dict(by_consonants),
        "all_consonantal_forms": all_forms,
        "consonant_map": CONSONANT_NAMES,
        "stats": stats,
    }


# =====================================================================
# Sezione 6 — Analisi compatibilita' con EVA
# =====================================================================

def analyze_eva_compatibility(lexicon: dict) -> dict:
    """Analizza la compatibilita' strutturale tra lessico ebraico e testo EVA.

    Confronta distribuzioni di lunghezza parole, frequenze consonanti,
    e pattern strutturali.
    """
    forms = lexicon["all_consonantal_forms"]

    # Distribuzione lunghezze
    heb_lengths = Counter(len(f) for f in forms)
    total = sum(heb_lengths.values())
    heb_length_dist = {
        k: round(v / total, 4) for k, v in sorted(heb_lengths.items())
    }

    # Frequenze consonanti nel lessico
    all_consonants = "".join(forms)
    consonant_freq = Counter(all_consonants)
    total_c = len(all_consonants)
    consonant_dist = {
        k: round(v / total_c, 4)
        for k, v in sorted(consonant_freq.items(), key=lambda x: -x[1])
    }

    # Lunghezza media (comparare con 5.17 EVA e 3.14 radice media)
    avg_length = round(sum(len(f) for f in forms) / len(forms), 2) if forms else 0

    # Pattern bigrammi consonantali piu' comuni
    bigrams = Counter()
    for f in forms:
        for i in range(len(f) - 1):
            bigrams[f[i:i+2]] += 1

    return {
        "hebrew_avg_word_length": avg_length,
        "eva_avg_word_length": 5.17,
        "eva_avg_root_length": 3.14,
        "hebrew_length_distribution": heb_length_dist,
        "hebrew_consonant_frequencies": consonant_dist,
        "top_hebrew_bigrams": [
            {"bigram": bg, "count": c}
            for bg, c in bigrams.most_common(20)
        ],
        "n_hebrew_consonants": 22,
        "n_eva_functional_groups": 10,
        "compression_ratio": round(22 / 10, 2),
    }


# =====================================================================
# Sezione 7 — Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per preparazione lessico ebraico."""
    print_header("VOYNICH TOOLKIT - Preparazione Lessico Ebraico")
    config.ensure_dirs()

    report_path = config.stats_dir / "lexicon_report.json"
    lexicon_path = config.lexicon_dir / "lexicon.json"

    if lexicon_path.exists() and not force:
        print("  Lessico gia' presente, skip (usa --force per rieseguire)")
        return

    config.lexicon_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download STEPBible
    print_step("Download lessico STEPBible TBESH...")
    try:
        stepbible_file = download_stepbible(config.lexicon_dir)
        stepbible_entries = parse_stepbible(stepbible_file)
        print(f"    Estratte {len(stepbible_entries)} entry da STEPBible")
    except Exception as e:
        print(f"    ATTENZIONE: STEPBible non disponibile ({e})")
        print(f"    Uso solo vocabolario curato")
        stepbible_entries = []

    # 2. Vocabolario curato medievale
    print_step("Caricamento vocabolario medievale curato...")
    curated = []
    botanical = get_curated_botanical()
    curated.extend(botanical)
    print(f"    Botanico: {len(botanical)} termini")

    astronomical = get_curated_astronomical()
    curated.extend(astronomical)
    print(f"    Astronomico: {len(astronomical)} termini")

    medical = get_curated_medical()
    curated.extend(medical)
    print(f"    Medico/anatomico: {len(medical)} termini")

    general = get_curated_general()
    curated.extend(general)
    print(f"    Generale: {len(general)} termini")
    print(f"    Totale curato: {len(curated)} termini")

    # 3. Assemblaggio lessico
    print_step("Assemblaggio lessico unificato...")
    lexicon = build_lexicon(stepbible_entries, curated)

    # 4. Analisi compatibilita' EVA
    print_step("Analisi compatibilita' strutturale con EVA...")
    compat = analyze_eva_compatibility(lexicon)

    # 5. Salvataggio
    print_step("Salvataggio lessico...")
    with open(lexicon_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)

    report = {
        "stats": lexicon["stats"],
        "eva_compatibility": compat,
        "consonant_map": CONSONANT_NAMES,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # === Sommario ===
    print("\n" + "=" * 60)
    print("  LESSICO EBRAICO MEDIEVALE")
    print("=" * 60)

    stats = lexicon["stats"]
    print(f"\n  Fonti:")
    for source, count in stats["sources"].items():
        print(f"    {source}: {count} entry")

    print(f"\n  Per dominio:")
    for domain, count in sorted(stats["by_domain_count"].items(),
                                 key=lambda x: -x[1]):
        print(f"    {domain:15s} {count:5d} termini")

    print(f"\n  Totale: {stats['total_entries']} entry, "
          f"{stats['unique_consonantal_forms']} forme consonantali uniche")

    print(f"\n  Compatibilita' strutturale EVA:")
    print(f"    Lunghezza media parole ebraiche:  {compat['hebrew_avg_word_length']}")
    print(f"    Lunghezza media parole EVA:       {compat['eva_avg_word_length']}")
    print(f"    Lunghezza media radici EVA:       {compat['eva_avg_root_length']}")
    print(f"    Consonanti ebraiche:              {compat['n_hebrew_consonants']}")
    print(f"    Gruppi funzionali EVA:            {compat['n_eva_functional_groups']}")

    print(f"\n  Top 10 consonanti nel lessico:")
    for cons, freq in list(compat["hebrew_consonant_frequencies"].items())[:10]:
        name = CONSONANT_NAMES.get(cons, cons)
        bar = "#" * int(freq * 100)
        print(f"    {cons} ({name:8s}) {freq:.3f} {bar}")

    print(f"\n  Distribuzione lunghezze:")
    for length, pct in list(compat["hebrew_length_distribution"].items())[:8]:
        bar = "#" * int(pct * 100)
        print(f"    {length} lettere: {pct:.3f} {bar}")

    print(f"\n  Lessico:  {lexicon_path}")
    print(f"  Report:   {report_path}")
