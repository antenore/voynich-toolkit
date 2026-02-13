"""
Approccio Champollion: decifrazione vincolata dai nomi di piante.

Ispirato al metodo di Champollion (cartigli di Cleopatra/Tolomeo),
usa i nomi delle piante illustrate nel Voynich come "cartigli" per
ricostruire il cifrario EVA -> ebraico.

Catena: EVA -> Ebraico (cercato) -> Italiano (fisso, via HEBREW_TO_ITALIAN)

Algoritmo:
1. Database ~55 coppie (folio, nome_pianta) con identificazioni botaniche note
2. Per ogni coppia, genera ipotesi di mapping parziale EVA->ebraico
3. Constraint satisfaction: vincoli supportati da >=2 folii indipendenti
4. Greedy selection del mapping parziale piu' consistente
5. Validazione: quanti nomi di piante decodificati correttamente?

Fonti identificazioni botaniche:
  Sherwood & Petersen (1968), Bax (2014), Tucker & Janick (2013),
  Zandbergen, voynichbotany.wordpress.com
"""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import click

from .config import ToolkitConfig
from .prepare_italian_lexicon import (
    HEBREW_TO_ITALIAN,
    normalize_italian_phonemic,
)
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Costanti
# =====================================================================

HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"

# Folio -> nomi italiani ortografici (normalizzati a runtime)
# Confidence: U=universale, S=forte, B=Bax, M=moderata
FOLIO_PLANTS = {
    # --- Tier 1: corte (4-5 lettere ebraiche), massima confidenza ---
    "f9v":   ["viola"],                         # U
    "f48v":  ["ruta"],                          # U
    "f25r":  ["timo"],                          # S
    "f5v":   ["malva"],                         # S
    "f37r":  ["menta"],                         # S
    "f39r":  ["croco", "zafferano"],            # S
    # --- Tier 2: medie (6-7), forte confidenza ---
    "f51v":  ["salvia"],                        # S
    "f44v":  ["sedano"],                        # S
    "f41r":  ["origano"],                       # S
    "f29v":  ["nigella"],                       # B
    "f45v":  ["lavanda"],                       # S
    "f31v":  ["valeriana"],                     # S
    # --- Tier 3: lunghe (8+), vincoli forti ---
    "f11r":  ["rosmarino", "ramerino"],         # U
    "f41v":  ["coriandolo"],                    # B
    "f44r":  ["mandragora"],                    # U
    # --- Addizionali: confidenza moderata-alta ---
    "f2r":   ["elleboro"],                      # M
    "f4r":   ["borragine"],                     # M
    "f6r":   ["artemisia"],                     # M
    "f13r":  ["ninfea"],                        # M
    "f14r":  ["piantagine"],                    # M
    "f15r":  ["calendula"],                     # M
    "f17r":  ["papavero"],                      # M
    "f22r":  ["sambuco"],                       # M
    "f23r":  ["edera"],                         # M
    "f25v":  ["centaurea"],                     # M
    "f26r":  ["genziana"],                      # M
    "f27r":  ["felce"],                         # M
    "f28r":  ["betonica"],                      # M
    "f30r":  ["aconito"],                       # M
    "f32r":  ["verbena"],                       # M
    "f33r":  ["consolida"],                     # M
    "f34r":  ["achillea"],                      # M
    "f35r":  ["assenzio"],                      # M
    "f36r":  ["enula"],                         # M
    "f38r":  ["cardo"],                         # M
    "f40r":  ["nepeta"],                        # M
    "f42r":  ["belladonna"],                    # M
    "f43r":  ["cicoria"],                       # M
    "f46r":  ["finocchio"],                     # M
    "f47r":  ["aneto"],                         # M
    "f49r":  ["cumino"],                        # M
    "f50r":  ["basilico"],                      # M
    "f52r":  ["prezzemolo"],                    # M
    "f54r":  ["camomilla"],                     # M
    "f55r":  ["issopo"],                        # M
    "f56r":  ["santoreggia"],                   # M
    "f65r":  ["ricino"],                        # M
    "f87r":  ["ninfea"],                        # M
    "f90r1": ["mirto"],                         # M
    "f93r":  ["ginepro"],                       # M
    "f94r":  ["alloro"],                        # M
    "f95r1": ["cipresso"],                      # M
    "f96r":  ["quercia"],                       # M
    "f99r":  ["aloe"],                          # M
    "f100r": ["rosa"],                          # M
    "f101r": ["pepe"],                          # M
    "f16v":  ["giglio"],                        # M
    "f3r":   ["ranuncolo"],                     # M
}

# Top parole funzionali H-section (da escludere come candidati)
FUNCTIONAL_WORDS = {
    "daiin", "aiin", "chedy", "chol", "chor", "dain", "shedy",
    "or", "ar", "ol", "al", "dy", "qokeedy", "qokeey", "chey",
    "shey", "chy", "shy", "dal", "dar", "okeey", "okeedy", "ain",
    "dol", "ykedy", "ytedy", "otar", "okedy", "otedy", "lchedy",
}

# Mapping inverso: fonema italiano -> possibili lettere ebraiche
# Derivato da HEBREW_TO_ITALIAN + HEBREW_ALTERNATIVES
ITALIAN_TO_HEBREW = {
    'a':  ['A'],
    'b':  ['b'],
    'v':  ['b'],            # bet spirantizzato
    'g':  ['g'],
    'd':  ['d'],
    'e':  ['E', 'h'],       # ayin o he
    'o':  ['w'],
    'u':  ['w'],            # vav = o/u
    'z':  ['z'],
    'k':  ['X', 'k', 'q'],  # chet, kaf, qof
    't':  ['J', 't'],       # tet o tav
    'i':  ['y'],
    'l':  ['l'],
    'm':  ['m'],
    'n':  ['n'],
    's':  ['S', 's'],       # shin o samekh
    'p':  ['p'],
    'f':  ['p'],            # pe spirantizzato
    'ts': ['C'],            # tsade
    'r':  ['r'],
    'sh': ['S'],            # shin
    'ch': ['X'],            # chet
}

# Fonemi multi-char che mappano a singola lettera ebraica
MULTI_CHAR_PHONEMES = {'ts', 'sh', 'ch'}

MAX_HEBREW_COMBOS = 200
MIN_SUPPORT_DEFAULT = 2


# =====================================================================
# Fase 1 — Preparazione dati
# =====================================================================

def parse_phonemes(phonemic):
    """Spezza stringa fonemica in lista di fonemi singoli.

    'ts', 'sh', 'ch' -> singolo fonema (= 1 lettera ebraica).
    Tutto il resto -> 1 char = 1 fonema.
    """
    phonemes = []
    i = 0
    while i < len(phonemic):
        if (i + 1 < len(phonemic)
                and phonemic[i:i+2] in ITALIAN_TO_HEBREW):
            phonemes.append(phonemic[i:i+2])
            i += 2
        elif phonemic[i] in ITALIAN_TO_HEBREW:
            phonemes.append(phonemic[i])
            i += 1
        else:
            # Fonema sconosciuto (es. 'w' in parole straniere): skip
            i += 1
    return phonemes


def italian_to_hebrew_options(phonemes):
    """Genera tutte le sequenze ebraiche possibili per una lista di fonemi.

    Gestisce ambiguita' (e->{E,h}, k->{X,k,q}, s->{S,s}, t->{J,t}).
    Tronca a MAX_HEBREW_COMBOS per evitare esplosione combinatoria.
    """
    options_per_pos = []
    for ph in phonemes:
        opts = ITALIAN_TO_HEBREW.get(ph)
        if opts is None:
            return []
        options_per_pos.append(opts)

    # Stima cardinalita'
    total = 1
    for opts in options_per_pos:
        total *= len(opts)

    if total <= MAX_HEBREW_COMBOS:
        return list(product(*options_per_pos))

    # Troppi: riduci posizioni con 3+ opzioni a sola prima opzione
    reduced = []
    for opts in options_per_pos:
        reduced.append(opts if len(opts) <= 2 else opts[:1])
    return list(product(*reduced))[:MAX_HEBREW_COMBOS]


@dataclass
class FolioProfile:
    folio: str
    section: str
    words: list
    line1_words: list
    distinctive_words: list
    word_counts: dict


def build_folio_profiles(eva_data):
    """Costruisce profili per ogni pagina H-section.

    Per ogni folio: parole, prima riga, parole distintive.
    Distintive = freq_locale / freq_globale > 3.
    """
    h_pages = [p for p in eva_data["pages"] if p.get("section") == "H"]

    # Frequenza globale su tutte le pagine H
    global_counts = Counter()
    for page in h_pages:
        global_counts.update(page["words"])
    total_h_words = sum(global_counts.values()) or 1

    profiles = {}
    for page in h_pages:
        folio = page["folio"]
        words = page["words"]
        local_counts = Counter(words)
        n_local = len(words) or 1

        # Prima riga
        line_words_all = page.get("line_words", [])
        line1 = line_words_all[0] if line_words_all else []
        line1_clean = [w for w in line1 if w not in FUNCTIONAL_WORDS]

        # Parole distintive
        distinctive = []
        for w, lc in local_counts.items():
            if w in FUNCTIONAL_WORDS:
                continue
            if lc < 2:
                continue
            global_freq = global_counts[w] / total_h_words
            local_freq = lc / n_local
            if global_freq > 0 and local_freq / global_freq > 3:
                distinctive.append(w)

        profiles[folio] = FolioProfile(
            folio=folio,
            section=page.get("section", "?"),
            words=words,
            line1_words=line1_clean,
            distinctive_words=distinctive,
            word_counts=dict(local_counts),
        )

    return profiles


def select_candidates(profile, target_len, max_n=10):
    """Seleziona parole EVA candidate con lunghezza = target_len.

    Priorita': 1=prima riga, 2=distintiva, 3=qualsiasi.
    """
    candidates = []
    seen = set()

    # P1: prima riga
    for w in profile.line1_words:
        if len(w) == target_len and w not in seen:
            candidates.append((w, 1))
            seen.add(w)

    # P2: distintive
    for w in profile.distinctive_words:
        if len(w) == target_len and w not in seen:
            candidates.append((w, 2))
            seen.add(w)

    # P3: tutte le altre
    for w in profile.word_counts:
        if (len(w) == target_len
                and w not in FUNCTIONAL_WORDS
                and w not in seen):
            candidates.append((w, 3))
            seen.add(w)

    return candidates[:max_n]


# =====================================================================
# Fase 2 — Generazione ipotesi
# =====================================================================

@dataclass
class Hypothesis:
    folio: str
    plant: str
    eva_word: str
    hebrew_seq: str
    partial_mapping: dict
    priority: int
    direction: str


def generate_hypotheses(folio_plants, folio_profiles, direction="rtl"):
    """Genera ipotesi di mapping parziale da allineamento piante-EVA.

    Per ogni (folio, pianta):
    1. Calcola fonemi e lunghezza target (= n. lettere ebraiche)
    2. Genera tutte le sequenze ebraiche possibili
    3. Per RTL: inverte la sequenza ebraica
    4. Seleziona parole EVA candidate con lunghezza giusta
    5. Per ogni (parola_EVA, seq_ebraica): verifica consistenza mapping
    """
    hypotheses = []

    for folio, plants in folio_plants.items():
        if folio not in folio_profiles:
            continue
        profile = folio_profiles[folio]

        for plant_phonemic in plants:
            phonemes = parse_phonemes(plant_phonemic)
            target_len = len(phonemes)
            if target_len < 3:
                continue

            hebrew_seqs = italian_to_hebrew_options(phonemes)
            if not hebrew_seqs:
                continue

            candidates = select_candidates(profile, target_len)
            if not candidates:
                continue

            for eva_word, priority in candidates:
                for heb_seq in hebrew_seqs:
                    # Per RTL: inverti sequenza ebraica
                    if direction == "rtl":
                        aligned = list(reversed(heb_seq))
                    else:
                        aligned = list(heb_seq)

                    # Calcola mapping parziale e verifica consistenza
                    partial = {}
                    reverse_map = {}  # heb -> eva (per iniettivita')
                    consistent = True

                    for i, eva_ch in enumerate(eva_word):
                        heb_ch = aligned[i]
                        # Stesso char EVA deve mappare alla stessa lettera
                        if eva_ch in partial:
                            if partial[eva_ch] != heb_ch:
                                consistent = False
                                break
                        else:
                            partial[eva_ch] = heb_ch
                        # Iniettivita': char EVA diversi -> lettere diverse
                        if heb_ch in reverse_map:
                            if reverse_map[heb_ch] != eva_ch:
                                consistent = False
                                break
                        else:
                            reverse_map[heb_ch] = eva_ch

                    if consistent:
                        hypotheses.append(Hypothesis(
                            folio=folio,
                            plant=plant_phonemic,
                            eva_word=eva_word,
                            hebrew_seq="".join(heb_seq),
                            partial_mapping=dict(partial),
                            priority=priority,
                            direction=direction,
                        ))

    return hypotheses


# =====================================================================
# Fase 3 — Constraint Satisfaction
# =====================================================================

def extract_and_vote_constraints(hypotheses):
    """Estrae vincoli (eva_char, hebrew_letter) e li vota.

    Supporto = numero di folii INDIPENDENTI che producono il vincolo.
    Peso = somma pesi priorita' (P1=3, P2=2, P3=1).
    """
    constraint_folii = defaultdict(set)
    constraint_weight = defaultdict(float)
    priority_weight = {1: 3.0, 2: 2.0, 3: 1.0}

    for hyp in hypotheses:
        w = priority_weight.get(hyp.priority, 1.0)
        for eva_ch, heb_ch in hyp.partial_mapping.items():
            key = (eva_ch, heb_ch)
            constraint_folii[key].add(hyp.folio)
            constraint_weight[key] += w

    return {
        key: {
            "n_folii": len(constraint_folii[key]),
            "weight": constraint_weight[key],
            "folii": sorted(constraint_folii[key]),
        }
        for key in constraint_folii
    }


def greedy_select(constraints, min_support=2):
    """Selezione greedy dei vincoli piu' supportati.

    1. Filtra vincoli con >= min_support folii
    2. Ordina per (n_folii, weight) decrescente
    3. Accetta se non conflittuale (consistenza + iniettivita')
    """
    valid = {
        k: v for k, v in constraints.items()
        if v["n_folii"] >= min_support
    }

    mapping = {}     # eva_char -> hebrew_letter
    reverse = {}     # hebrew_letter -> eva_char

    sorted_constraints = sorted(
        valid.items(),
        key=lambda x: (x[1]["n_folii"], x[1]["weight"]),
        reverse=True,
    )

    accepted = []
    for (eva_ch, heb_ch), info in sorted_constraints:
        # Consistenza
        if eva_ch in mapping:
            if mapping[eva_ch] != heb_ch:
                continue
            else:
                continue  # gia' fissato
        # Iniettivita'
        if heb_ch in reverse:
            if reverse[heb_ch] != eva_ch:
                continue

        mapping[eva_ch] = heb_ch
        reverse[heb_ch] = eva_ch
        accepted.append({
            "eva": eva_ch,
            "hebrew": heb_ch,
            "hebrew_name": CONSONANT_NAMES.get(heb_ch, "?"),
            "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
            "n_folii": info["n_folii"],
            "weight": info["weight"],
            "folii": info["folii"],
        })

    return mapping, accepted


def decode_word(eva_word, mapping, direction):
    """Decodifica parola EVA con mapping (parziale o completo).

    Returns stringa italiana o None se qualche char non e' mappato.
    """
    chars = list(eva_word) if direction == "ltr" else list(reversed(eva_word))
    hebrew = []
    for ch in chars:
        if ch not in mapping:
            return None
        hebrew.append(mapping[ch])
    return "".join(HEBREW_TO_ITALIAN.get(h, "?") for h in hebrew)


def propagate(partial_mapping, folio_plants, folio_profiles, direction):
    """Verifica quanti folii producono il nome pianta atteso.

    Applica il mapping parziale a tutte le parole del folio e controlla
    se qualcuna decodifica esattamente nel nome della pianta.
    Controlla sia il decode diretto sia il match posizionale con
    tutte le alternative ebraiche.
    """
    hits = []

    for folio, plants in folio_plants.items():
        if folio not in folio_profiles:
            continue
        profile = folio_profiles[folio]

        for plant_phonemic in plants:
            phonemes = parse_phonemes(plant_phonemic)
            target_len = len(phonemes)

            # Tutte le alternative ebraiche per ogni posizione
            heb_options = [ITALIAN_TO_HEBREW.get(ph, []) for ph in phonemes]
            if direction == "rtl":
                heb_aligned = list(reversed(heb_options))
            else:
                heb_aligned = list(heb_options)

            for word in set(profile.words):
                if len(word) != target_len:
                    continue

                # Verifica: ogni posizione deve avere il mapping
                all_match = True
                for i, eva_ch in enumerate(word):
                    if eva_ch not in partial_mapping:
                        all_match = False
                        break
                    heb = partial_mapping[eva_ch]
                    if heb not in heb_aligned[i]:
                        all_match = False
                        break

                if all_match:
                    decoded = decode_word(word, partial_mapping, direction)
                    hits.append({
                        "folio": folio,
                        "plant": plant_phonemic,
                        "eva_word": word,
                        "decoded": decoded or "?",
                    })

    return hits


def propagate_partial(partial_mapping, folio_plants, folio_profiles,
                      direction):
    """Come propagate, ma riporta anche match parziali.

    Per ogni (folio, pianta), trova la parola EVA con il maggior
    numero di posizioni che coincidono col nome pianta atteso.
    Controlla TUTTE le alternative ebraiche per ogni posizione.
    """
    results = []

    for folio, plants in folio_plants.items():
        if folio not in folio_profiles:
            continue
        profile = folio_profiles[folio]

        for plant_phonemic in plants:
            phonemes = parse_phonemes(plant_phonemic)
            target_len = len(phonemes)

            # Tutte le opzioni ebraiche per ogni posizione
            heb_options = [ITALIAN_TO_HEBREW.get(ph, []) for ph in phonemes]
            if direction == "rtl":
                heb_options_aligned = list(reversed(heb_options))
            else:
                heb_options_aligned = list(heb_options)

            best_word = None
            best_score = -1
            best_detail = None

            for word in set(profile.words):
                if len(word) != target_len:
                    continue

                n_mapped = 0
                n_match = 0
                detail_chars = []
                for i, eva_ch in enumerate(word):
                    opts = heb_options_aligned[i]
                    if eva_ch in partial_mapping:
                        n_mapped += 1
                        heb = partial_mapping[eva_ch]
                        if heb in opts:
                            n_match += 1
                            detail_chars.append(
                                f"{eva_ch}->OK"
                            )
                        else:
                            expected_it = HEBREW_TO_ITALIAN.get(
                                opts[0], '?') if opts else '?'
                            detail_chars.append(
                                f"{eva_ch}->MISS(att:{expected_it})"
                            )
                    else:
                        detail_chars.append(f"{eva_ch}->?")

                if n_mapped == 0:
                    continue
                score = n_match / target_len

                if score > best_score:
                    best_score = score
                    best_word = word
                    best_detail = {
                        "eva_word": word,
                        "n_mapped": n_mapped,
                        "n_match": n_match,
                        "n_total": target_len,
                        "match_ratio": round(score, 3),
                        "chars": detail_chars,
                    }

            if best_detail and best_detail["n_match"] > 0:
                results.append({
                    "folio": folio,
                    "plant": plant_phonemic,
                    "best": best_detail,
                })

    results.sort(key=lambda x: -x["best"]["match_ratio"])
    return results


def try_complete_from_plants(partial_mapping, folio_plants, folio_profiles,
                              hypotheses, direction):
    """Tenta di completare il mapping usando le ipotesi compatibili.

    Per ogni char EVA non ancora mappato, cerca se esiste un'ipotesi
    compatibile col mapping attuale che propone un valore per quel char.
    Accetta solo proposte unanimi (tutte le ipotesi compatibili concordano).
    """
    extended = dict(partial_mapping)
    reverse = {v: k for k, v in extended.items()}
    additions = []

    # Filtra ipotesi compatibili col mapping attuale
    compatible = []
    for hyp in hypotheses:
        ok = True
        for eva_ch, heb_ch in hyp.partial_mapping.items():
            if eva_ch in extended and extended[eva_ch] != heb_ch:
                ok = False
                break
            if heb_ch in reverse and reverse[heb_ch] != eva_ch:
                ok = False
                break
        if ok:
            compatible.append(hyp)

    # Per ogni char non mappato, cerca proposte disponibili
    unmapped = [c for c in "acdefghiklmnopqrsty" if c not in extended]

    for eva_ch in unmapped:
        proposals = defaultdict(set)  # heb_ch -> set of folii
        for hyp in compatible:
            if eva_ch in hyp.partial_mapping:
                heb_ch = hyp.partial_mapping[eva_ch]
                proposals[heb_ch].add(hyp.folio)

        if not proposals:
            continue

        # Ordina per n. folii decrescente, prendi il primo disponibile
        sorted_props = sorted(
            proposals.items(), key=lambda x: len(x[1]), reverse=True
        )
        for heb_ch, folii in sorted_props:
            if heb_ch not in reverse:
                extended[eva_ch] = heb_ch
                reverse[heb_ch] = eva_ch
                additions.append({
                    "eva": eva_ch,
                    "hebrew": heb_ch,
                    "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
                    "n_folii": len(folii),
                    "source": "completion",
                })
                break

    # Se restano char non mappati, assegna da vincoli globali (non solo
    # ipotesi compatibili) come ultima risorsa
    still_unmapped = [c for c in "acdefghiklmnopqrsty"
                      if c not in extended]
    if still_unmapped:
        # Raccogli proposte da TUTTE le ipotesi (non solo compatibili)
        for eva_ch in still_unmapped:
            all_proposals = defaultdict(set)
            for hyp in hypotheses:
                if eva_ch in hyp.partial_mapping:
                    heb_ch = hyp.partial_mapping[eva_ch]
                    all_proposals[heb_ch].add(hyp.folio)

            sorted_props = sorted(
                all_proposals.items(), key=lambda x: len(x[1]),
                reverse=True,
            )
            for heb_ch, folii in sorted_props:
                if heb_ch not in reverse:
                    extended[eva_ch] = heb_ch
                    reverse[heb_ch] = eva_ch
                    additions.append({
                        "eva": eva_ch,
                        "hebrew": heb_ch,
                        "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
                        "n_folii": len(folii),
                        "source": "fallback",
                    })
                    break

    return extended, additions


# =====================================================================
# Fase 4 — Validazione e output
# =====================================================================

def validate_mapping(partial_mapping):
    """Valida il mapping parziale: copertura, rapporto vocali, etc."""
    vowel_hebrew = {h for h, it in HEBREW_TO_ITALIAN.items()
                    if it in ('a', 'e', 'i', 'o')}
    # 'u' non e' diretto ma vav->o, aggiungiamo w
    vowel_hebrew.add('w')

    n_mapped = len(partial_mapping)
    n_vowel = sum(1 for h in partial_mapping.values() if h in vowel_hebrew)
    vowel_ratio = n_vowel / n_mapped if n_mapped else 0

    return {
        "n_chars_mapped": n_mapped,
        "n_chars_total": 19,
        "coverage_pct": round(100 * n_mapped / 19, 1),
        "vowel_mapped": n_vowel,
        "consonant_mapped": n_mapped - n_vowel,
        "vowel_ratio_mapped": round(vowel_ratio, 3),
        "expected_vowel_ratio": 0.47,
        "plausible": 0.20 <= vowel_ratio <= 0.65 if n_mapped >= 3 else None,
    }


def _build_report(direction, hypotheses, constraints, mapping, accepted,
                   hits, hits_extended, partial_matches, extended_mapping,
                   completion_additions, validation, validation_ext,
                   folio_plants, folio_profiles):
    """Costruisce report JSON completo."""
    # Statistiche vincoli per char EVA
    constraints_by_eva = defaultdict(list)
    for (eva_ch, heb_ch), info in constraints.items():
        constraints_by_eva[eva_ch].append({
            "hebrew": heb_ch,
            "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
            "n_folii": info["n_folii"],
            "weight": info["weight"],
        })

    # Mapping leggibile
    readable_mapping = {}
    for eva_ch, heb_ch in sorted(mapping.items()):
        readable_mapping[eva_ch] = {
            "hebrew": heb_ch,
            "hebrew_name": CONSONANT_NAMES.get(heb_ch, "?"),
            "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
        }

    extended_readable = {}
    for eva_ch, heb_ch in sorted(extended_mapping.items()):
        extended_readable[eva_ch] = {
            "hebrew": heb_ch,
            "hebrew_name": CONSONANT_NAMES.get(heb_ch, "?"),
            "italian": HEBREW_TO_ITALIAN.get(heb_ch, "?"),
            "source": ("greedy" if eva_ch in mapping else "completion"),
        }

    n_matched_folii = sum(
        1 for f in folio_plants if f in folio_profiles
    )

    return {
        "direction": direction,
        "n_folio_plant_pairs": sum(
            len(v) for f, v in folio_plants.items()
            if f in folio_profiles
        ),
        "n_folii_in_eva": n_matched_folii,
        "n_folii_total": len(folio_profiles),
        "n_hypotheses": len(hypotheses),
        "n_constraints_total": len(constraints),
        "n_constraints_valid": len(accepted),
        "mapping_greedy": readable_mapping,
        "mapping_extended": extended_readable,
        "completion_additions": completion_additions,
        "accepted_constraints": accepted,
        "plant_hits_greedy": hits,
        "plant_hits_extended": hits_extended,
        "n_plant_hits_greedy": len(hits),
        "n_plant_hits_extended": len(hits_extended),
        "partial_matches": partial_matches[:20],
        "validation_greedy": validation,
        "validation_extended": validation_ext,
        "constraints_by_eva": {
            k: sorted(v, key=lambda x: -x["n_folii"])
            for k, v in sorted(constraints_by_eva.items())
        },
    }


def _save_matches(report, path):
    """Salva match e mapping come testo leggibile."""
    lines = [
        "# Approccio Champollion: risultati",
        f"# Direzione: {report['direction']}",
        f"# Ipotesi generate: {report['n_hypotheses']}",
        f"# Vincoli accettati: {report['n_constraints_valid']}",
        f"# Piante (greedy): {report['n_plant_hits_greedy']}",
        f"# Piante (esteso): {report['n_plant_hits_extended']}",
        "",
        "=" * 60,
        "  MAPPING GREEDY (EVA -> Ebraico -> Italiano)",
        "=" * 60,
        "",
    ]

    for eva_ch, info in sorted(report["mapping_greedy"].items()):
        lines.append(
            f"  {eva_ch} -> {info['hebrew']} "
            f"({info['hebrew_name']:8s}) -> {info['italian']}"
        )

    lines.extend(["", "=" * 60,
                   "  MAPPING ESTESO (greedy + completamento)",
                   "=" * 60, ""])

    for eva_ch, info in sorted(report["mapping_extended"].items()):
        src = info.get("source", "?")
        marker = "*" if src == "completion" else " "
        lines.append(
            f" {marker}{eva_ch} -> {info['hebrew']} "
            f"({info['hebrew_name']:8s}) -> {info['italian']}  [{src}]"
        )

    lines.extend(["", "=" * 60,
                   "  VINCOLI ACCETTATI (ordinati per supporto)",
                   "=" * 60, ""])

    for c in report["accepted_constraints"]:
        folii_str = ", ".join(c["folii"][:5])
        lines.append(
            f"  {c['eva']} -> {c['hebrew']} ({c['italian']})  "
            f"folii={c['n_folii']}  peso={c['weight']:.1f}  [{folii_str}]"
        )

    lines.extend(["", "=" * 60,
                   "  PIANTE DECODIFICATE (mapping esteso)",
                   "=" * 60, ""])

    if report["plant_hits_extended"]:
        for h in report["plant_hits_extended"]:
            lines.append(
                f"  {h['folio']:<8s} {h['eva_word']:<15s} "
                f"-> {h['decoded']:<15s} (atteso: {h['plant']})"
            )
    else:
        lines.append("  Nessuna pianta decodificata.")

    lines.extend(["", "=" * 60,
                   "  MATCH PARZIALI (top 20)",
                   "=" * 60, ""])

    for pm in report.get("partial_matches", [])[:20]:
        b = pm["best"]
        lines.append(
            f"  {pm['folio']:<8s} {pm['plant']:<15s} "
            f"best={b['eva_word']:<12s} "
            f"match={b['n_match']}/{b['n_total']} "
            f"({b['match_ratio']:.0%})"
        )

    lines.extend(["", "=" * 60,
                   "  VINCOLI PER CHAR EVA",
                   "=" * 60, ""])

    for eva_ch, opts in sorted(report["constraints_by_eva"].items()):
        top = opts[:3]
        opts_str = "  ".join(
            f"{o['hebrew']}({o['italian']}):{o['n_folii']}f"
            for o in top
        )
        lines.append(f"  {eva_ch}: {opts_str}")

    path.write_text("\n".join(lines), encoding="utf-8")


def _print_console(report):
    """Stampa risultati su console."""
    click.echo(f"\n{'=' * 60}")
    click.echo("  RISULTATO APPROCCIO CHAMPOLLION")
    click.echo(f"{'=' * 60}")

    click.echo(f"\n  Direzione: {report['direction'].upper()}")
    click.echo(f"  Folii piante nel testo EVA: {report['n_folii_in_eva']}")
    click.echo(f"  Ipotesi generate: {report['n_hypotheses']}")
    click.echo(f"  Vincoli totali: {report['n_constraints_total']}")
    click.echo(f"  Vincoli accettati (>= min_support): "
               f"{report['n_constraints_valid']}")

    v = report["validation_greedy"]
    click.echo(f"\n  Mapping greedy: {v['n_chars_mapped']}/19 char EVA "
               f"({v['coverage_pct']}%)")
    click.echo(f"  Vocali: {v['vowel_mapped']}, "
               f"consonanti: {v['consonant_mapped']}, "
               f"ratio: {v['vowel_ratio_mapped']:.1%} (atteso ~47%)")

    if report["mapping_greedy"]:
        click.echo("\n  Mapping greedy:")
        for eva_ch, info in sorted(report["mapping_greedy"].items()):
            click.echo(f"    {eva_ch} -> {info['hebrew']} "
                        f"({info['hebrew_name']:8s}) -> {info['italian']}")

    # Mapping esteso
    if report["completion_additions"]:
        click.echo(f"\n  Completamento: "
                    f"+{len(report['completion_additions'])} char")
        for a in report["completion_additions"]:
            click.echo(f"    {a['eva']} -> {a['hebrew']}({a['italian']})  "
                        f"[{a['n_folii']} folii]")

    ve = report["validation_extended"]
    click.echo(f"\n  Mapping esteso: {ve['n_chars_mapped']}/19 char EVA "
               f"({ve['coverage_pct']}%)")

    # Piante decodificate
    n_greedy = report["n_plant_hits_greedy"]
    n_ext = report["n_plant_hits_extended"]
    click.echo(f"\n  Piante decodificate (greedy): {n_greedy}")
    click.echo(f"  Piante decodificate (esteso): {n_ext}")

    if report["plant_hits_extended"]:
        for h in report["plant_hits_extended"]:
            click.echo(f"    [OK] {h['folio']:<8s} {h['eva_word']:<12s} "
                        f"-> {h['decoded']}")

    # Match parziali
    partial = report.get("partial_matches", [])
    if partial:
        click.echo(f"\n  Top match parziali ({len(partial)} totali):")
        for pm in partial[:10]:
            b = pm["best"]
            click.echo(
                f"    {pm['folio']:<8s} {pm['plant']:<12s} "
                f"best={b['eva_word']:<10s} "
                f"{b['n_match']}/{b['n_total']} pos "
                f"({b['match_ratio']:.0%})"
            )

    # Top vincoli
    click.echo("\n  Top vincoli per supporto:")
    all_constraints = []
    for eva_ch, opts in report["constraints_by_eva"].items():
        for o in opts:
            all_constraints.append((eva_ch, o))
    all_constraints.sort(key=lambda x: -x[1]["n_folii"])
    for eva_ch, o in all_constraints[:10]:
        click.echo(f"    {eva_ch} -> {o['hebrew']}({o['italian']})  "
                    f"{o['n_folii']} folii  peso={o['weight']:.0f}")

    # Verdetto
    n_hits = max(n_greedy, n_ext)
    n_constraints = report["n_constraints_valid"]
    click.echo(f"\n  {'=' * 40}")
    if n_hits >= 5:
        click.echo("  VERDETTO: PROMETTENTE — "
                    f"{n_hits} piante decodificate!")
    elif n_hits >= 2:
        click.echo("  VERDETTO: INCORAGGIANTE — "
                    f"{n_hits} piante decodificate")
    elif n_constraints >= 5:
        click.echo("  VERDETTO: VINCOLI TROVATI — "
                    f"{n_constraints} vincoli, {n_hits} piante")
    else:
        click.echo("  VERDETTO: NON CONCLUSIVO — "
                    "vincoli insufficienti")
    click.echo(f"  {'=' * 40}")


# =====================================================================
# Entry point
# =====================================================================

def run(config, force=False, direction="both", min_support=2):
    """Entry point: approccio Champollion dai nomi di piante."""
    report_path = config.stats_dir / "champollion_report.json"

    if report_path.exists() and not force:
        click.echo("  Champollion report esistente. "
                    "Usa --force per rieseguire.")
        return

    config.ensure_dirs()
    print_header("APPROCCIO CHAMPOLLION — Decifrazione dai nomi di piante")

    # --- 1. Carica dati EVA ---
    print_step("Parsing parole EVA...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"File EVA non trovato: {eva_file}\n"
            "  Esegui prima: voynich eva"
        )
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']} parole, "
               f"{eva_data['unique_words']} uniche, "
               f"{len(eva_data['pages'])} pagine")

    # --- 2. Normalizza nomi piante ---
    print_step("Normalizzazione nomi piante...")
    folio_plants_norm = {}
    for folio, names in FOLIO_PLANTS.items():
        folio_plants_norm[folio] = [
            normalize_italian_phonemic(n) for n in names
        ]
    n_pairs = sum(len(v) for v in folio_plants_norm.values())
    click.echo(f"    {len(folio_plants_norm)} folii, "
               f"{n_pairs} coppie folio-pianta")

    # Mostra alcuni esempi di normalizzazione
    examples = [("viola", "viola"), ("croco", "croco"),
                ("nigella", "nigella"), ("coriandolo", "coriandolo")]
    for orig, _ in examples[:3]:
        norm = normalize_italian_phonemic(orig)
        ph = parse_phonemes(norm)
        click.echo(f"    {orig} -> {norm} -> {len(ph)} fonemi")

    # --- 3. Profili folio ---
    print_step("Costruzione profili folio H-section...")
    profiles = build_folio_profiles(eva_data)
    click.echo(f"    {len(profiles)} pagine H con profilo")

    matched_folii = [f for f in folio_plants_norm if f in profiles]
    click.echo(f"    {len(matched_folii)}/{len(folio_plants_norm)} "
               f"folii piante trovati nel testo EVA")

    # --- 4. Genera ipotesi e seleziona per ogni direzione ---
    directions = ["rtl", "ltr"] if direction == "both" else [direction]
    best_report = None

    for d in directions:
        print_step(f"Direzione {d.upper()}...")

        # 4a. Genera ipotesi
        click.echo(f"    Generazione ipotesi...")
        hypotheses = generate_hypotheses(
            folio_plants_norm, profiles, direction=d
        )
        click.echo(f"    {len(hypotheses)} ipotesi consistenti")

        if not hypotheses:
            click.echo(f"    Nessuna ipotesi! Salto {d}.")
            continue

        # 4b. Vota vincoli
        click.echo(f"    Votazione vincoli...")
        constraints = extract_and_vote_constraints(hypotheses)
        n_valid = sum(1 for v in constraints.values()
                      if v["n_folii"] >= min_support)
        click.echo(f"    {len(constraints)} vincoli totali, "
                   f"{n_valid} con >= {min_support} folii")

        # 4c. Selezione greedy
        click.echo(f"    Selezione greedy mapping...")
        mapping, accepted = greedy_select(constraints,
                                           min_support=min_support)
        click.echo(f"    {len(mapping)} char EVA fissati")

        # 4d. Propagazione (greedy)
        click.echo(f"    Propagazione (greedy)...")
        hits = propagate(mapping, folio_plants_norm, profiles, d)
        click.echo(f"    {len(hits)} nomi piante (greedy)")

        # 4e. Completamento mapping
        click.echo(f"    Completamento mapping da ipotesi compatibili...")
        extended, additions = try_complete_from_plants(
            mapping, folio_plants_norm, profiles, hypotheses, d
        )
        click.echo(f"    {len(mapping)} -> {len(extended)} char EVA "
                   f"(+{len(additions)} completati)")

        # 4f. Propagazione (estesa)
        hits_ext = propagate(extended, folio_plants_norm, profiles, d)
        click.echo(f"    {len(hits_ext)} nomi piante (esteso)")

        # 4g. Match parziali
        click.echo(f"    Analisi match parziali...")
        partial_matches = propagate_partial(
            extended, folio_plants_norm, profiles, d
        )
        n_good = sum(1 for pm in partial_matches
                     if pm["best"]["match_ratio"] >= 0.5)
        click.echo(f"    {len(partial_matches)} match parziali, "
                   f"{n_good} con >= 50% posizioni")

        # 4h. Validazione
        validation = validate_mapping(mapping)
        validation_ext = validate_mapping(extended)

        # 4i. Report
        report = _build_report(
            d, hypotheses, constraints, mapping, accepted,
            hits, hits_ext, partial_matches, extended, additions,
            validation, validation_ext, folio_plants_norm, profiles,
        )

        n_total = report["n_plant_hits_extended"]
        if (best_report is None
                or n_total > best_report["n_plant_hits_extended"]
                or (n_total == best_report["n_plant_hits_extended"]
                    and report["n_constraints_valid"]
                    > best_report["n_constraints_valid"])):
            best_report = report

    if best_report is None:
        click.echo("\n  Nessun risultato ottenuto.")
        return

    # --- 5. Salvataggio ---
    print_step("Salvataggio risultati...")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(best_report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Report: {report_path}")

    matches_path = config.stats_dir / "champollion_matches.txt"
    _save_matches(best_report, matches_path)
    click.echo(f"    Matches: {matches_path}")

    # --- 6. Output console ---
    _print_console(best_report)
