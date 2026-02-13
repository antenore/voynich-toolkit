"""
Decoder giudeo-italiano per il manoscritto Voynich.

Ipotesi: cifrario 1:1 con 19 caratteri EVA individuali → 22 lettere ebraiche
corsive → fonemi italiani (via convenzione giudeo-italiana).

Catena: Glifo EVA → Lettera ebraica (cercata) → Fonema italiano (fisso)

Metodo: hill-climbing con multi-restart, validazione da lessico italiano curato.
"""
import json
import random
import time
from collections import Counter, defaultdict
from itertools import combinations

import click
import numpy as np

from .config import ToolkitConfig
from .prepare_lexicon import CONSONANT_NAMES
from .prepare_italian_lexicon import (
    HEBREW_TO_ITALIAN,
    HEBREW_ALTERNATIVES,
    ITALIAN_LETTER_FREQS,
    ITALIAN_COMMON_BIGRAMS,
)
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Costanti
# =====================================================================

HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"  # 22 consonanti
N_HEBREW = len(HEBREW_CHARS)

# 19 caratteri EVA individuali (dall'analisi cipher_hypothesis)
EVA_CHARS = list("acdefghiklmnopqrsty")
N_EVA = len(EVA_CHARS)

# Pesi per lunghezza (calibrati sulla probabilità di match casuale)
# len 3: alta prob casuale → peso basso; len 8+: estremamente improbabile
LENGTH_WEIGHTS = {3: 1, 4: 3, 5: 10, 6: 30, 7: 80, 8: 100}

# Sezione manoscritto → dominio lessicale
SECTION_TO_DOMAIN = {
    "H": "botanical",
    "S": "astronomical",
    "Z": "astronomical",
    "B": "medical",
    "P": "medical",
}

# Top N parole uniche per scoring
TOP_N_WORDS = 1000


# =====================================================================
# Sezione 1 — Preparazione dati
# =====================================================================

def load_eva_charset(config):
    """Estrae i 19 char EVA da cipher_hypothesis.json.

    Returns:
        eva_to_index: {eva_char → index 0..18}
        eva_chars: list dei 19 char EVA ordinati
    """
    path = config.stats_dir / "cipher_hypothesis.json"
    with open(path) as f:
        data = json.load(f)

    raw = data["homophone_analysis"]["groups"]
    chars = set()
    for gid in raw:
        for ch in raw[gid]:
            chars.add(ch)

    sorted_chars = sorted(chars)
    eva_to_index = {ch: i for i, ch in enumerate(sorted_chars)}
    return eva_to_index, sorted_chars


def load_italian_lexicon(config):
    """Carica lessico italiano da italian_lexicon.json.

    Returns:
        lexicon_set: set di forme fonemiche
        gloss_lookup: {form → gloss}
        domain_sets: {domain → set di forme}
    """
    path = config.lexicon_dir / "italian_lexicon.json"
    with open(path) as f:
        data = json.load(f)

    lexicon_set = set(data["all_forms"])
    gloss_lookup = data.get("form_to_gloss", {})

    domain_sets = defaultdict(set)
    for domain, entries in data["by_domain"].items():
        for entry in entries:
            domain_sets[domain].add(entry["phonemic"])

    return lexicon_set, gloss_lookup, dict(domain_sets)


def prepare_char_words(words_data, eva_to_index, top_n=TOP_N_WORDS):
    """Converte parole EVA in rappresentazione a indici char.

    A differenza del decoder ebraico (che usa 10 gruppi omofonici),
    qui ogni char EVA è individuale (19 possibilità).

    Returns:
        list di (char_indices_tuple, eva_word, count, sections_frozenset)
    """
    word_counter = Counter(words_data["words"])

    word_sections = defaultdict(set)
    for page in words_data["pages"]:
        section = page.get("section", "?")
        for w in page["words"]:
            word_sections[w].add(section)

    result = []
    for word, count in word_counter.most_common():
        char_indices = []
        valid = True
        for ch in word:
            if ch in eva_to_index:
                char_indices.append(eva_to_index[ch])
            else:
                valid = False
                break

        if not valid or len(char_indices) < 3:
            continue

        result.append((
            tuple(char_indices),
            word,
            count,
            frozenset(word_sections[word]),
        ))

        if len(result) >= top_n:
            break

    return result


def compute_char_freqs(words_data, eva_to_index):
    """Frequenza di ogni char EVA nel corpus."""
    char_counter = Counter()
    for w in words_data["words"]:
        for ch in w:
            char_counter[ch] += 1

    total = sum(char_counter.values()) or 1
    freqs = []
    sorted_chars = sorted(eva_to_index.keys(), key=lambda c: eva_to_index[c])
    for ch in sorted_chars:
        freqs.append(char_counter.get(ch, 0) / total)
    return freqs


# =====================================================================
# Sezione 2 — Conversione mapping EVA → ebraico → italiano
# =====================================================================

def hebrew_to_italian_phonemes(hebrew_str):
    """Converte stringa ebraica in fonemi italiani usando HEBREW_TO_ITALIAN.

    Gestisce fonemi multi-char come tsade→"ts".
    """
    return "".join(HEBREW_TO_ITALIAN.get(ch, "?") for ch in hebrew_str)


def apply_mapping_to_italian(char_indices, mapping, direction):
    """Pipeline completa: indici EVA → ebraico → italiano.

    Args:
        char_indices: tuple di indici EVA (0..18)
        mapping: list[int] di 19 indici in HEBREW_CHARS
        direction: "rtl" o "ltr"

    Returns:
        (hebrew_str, italian_str)
    """
    indices = char_indices if direction == "ltr" else char_indices[::-1]
    hebrew = "".join(HEBREW_CHARS[mapping[i]] for i in indices)
    italian = hebrew_to_italian_phonemes(hebrew)
    return hebrew, italian


# =====================================================================
# Sezione 3 — Scoring
# =====================================================================

def score_mapping(mapping, char_words, lexicon_set, direction):
    """Scora un mapping contando match pesati nel lessico italiano.

    Returns:
        (score, matches) dove matches = [(eva_word, hebrew, italian, count, sections)]
    """
    score = 0
    matches = []
    for char_indices, word, count, sections in char_words:
        hebrew, italian = apply_mapping_to_italian(
            char_indices, mapping, direction
        )
        ilen = len(italian)
        if ilen >= 3 and italian in lexicon_set:
            weight = LENGTH_WEIGHTS.get(min(ilen, 8), 100)
            score += count * weight
            matches.append((word, hebrew, italian, count, sections))
    return score, matches


def _word_score(char_indices, mapping, direction, lexicon_set, count):
    """Score singola parola (helper per hill-climbing)."""
    _, italian = apply_mapping_to_italian(char_indices, mapping, direction)
    if len(italian) >= 3 and italian in lexicon_set:
        weight = LENGTH_WEIGHTS.get(min(len(italian), 8), 100)
        return count * weight
    return 0


def score_with_ngrams(mapping, char_words, direction, bigram_set):
    """Score secondario basato su bigrammi italiani.

    Conta quanti bigrammi del testo decodificato compaiono
    tra i bigrammi comuni dell'italiano.
    """
    bigram_hits = 0
    total_bigrams = 0
    for char_indices, word, count, sections in char_words:
        _, italian = apply_mapping_to_italian(
            char_indices, mapping, direction
        )
        for i in range(len(italian) - 1):
            bg = italian[i:i+2]
            total_bigrams += count
            if bg in bigram_set:
                bigram_hits += count
    return bigram_hits / total_bigrams if total_bigrams else 0


# =====================================================================
# Sezione 4 — Baseline casuale
# =====================================================================

def compute_random_baseline(char_words, lexicon_set, n_trials=200):
    """Score di 200 mapping casuali → soglia di significatività.

    Ogni mapping: 19 indici scelti da 22 senza ripetizione.
    """
    scores = []
    for _ in range(n_trials):
        perm = random.sample(range(N_HEBREW), N_EVA)
        for direction in ("ltr", "rtl"):
            s, _ = score_mapping(perm, char_words, lexicon_set, direction)
            scores.append(s)

    arr = np.array(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    mx = float(arr.max())
    threshold = mean + 4 * std

    return {
        "mean": round(mean, 1),
        "std": round(std, 1),
        "max": round(mx, 1),
        "threshold": round(threshold, 1),
        "n_trials": n_trials,
    }


# =====================================================================
# Sezione 5 — Inizializzazione mapping
# =====================================================================

def frequency_guided_mapping(eva_freqs):
    """Mapping iniziale: char EVA più frequente → lettera ebraica
    il cui fonema italiano è più frequente.

    Ordina i 19 char EVA per frequenza decrescente,
    e li mappa sulle 22 lettere ebraiche ordinate per frequenza
    del fonema italiano corrispondente.
    """
    # Ordina char EVA per frequenza
    sorted_eva = sorted(range(N_EVA), key=lambda i: eva_freqs[i], reverse=True)

    # Ordina lettere ebraiche per frequenza del fonema italiano
    heb_italian_freq = []
    for i, hc in enumerate(HEBREW_CHARS):
        phoneme = HEBREW_TO_ITALIAN.get(hc, "?")
        # Somma frequenze di tutti i char nel fonema
        freq = sum(ITALIAN_LETTER_FREQS.get(c, 0) for c in phoneme)
        heb_italian_freq.append((i, freq))
    sorted_heb = sorted(heb_italian_freq, key=lambda x: x[1], reverse=True)

    mapping = [0] * N_EVA
    for rank, eva_idx in enumerate(sorted_eva):
        mapping[eva_idx] = sorted_heb[rank][0]

    return mapping


def positional_guided_mapping(words_data, eva_to_index):
    """Mapping basato su profili posizionali.

    Char prevalenti a inizio/fine parola → vocali (a, e, i, o)
    Char prevalenti in posizione mediana → consonanti frequenti
    """
    # Calcola profili posizionali
    initial_counts = Counter()
    final_counts = Counter()
    total_counts = Counter()
    n_initial = 0
    n_final = 0

    for w in words_data["words"]:
        if len(w) < 2:
            continue
        if w[0] in eva_to_index:
            initial_counts[eva_to_index[w[0]]] += 1
            n_initial += 1
        if w[-1] in eva_to_index:
            final_counts[eva_to_index[w[-1]]] += 1
            n_final += 1
        for ch in w:
            if ch in eva_to_index:
                total_counts[eva_to_index[ch]] += 1

    # Calcola "vocalità": char con alta probabilità iniziale + finale
    # sono probabilmente vocali in italiano
    vocality = {}
    for i in range(N_EVA):
        ini_rate = initial_counts.get(i, 0) / n_initial if n_initial else 0
        fin_rate = final_counts.get(i, 0) / n_final if n_final else 0
        vocality[i] = ini_rate + fin_rate

    # Indici ebraici delle vocali (matres lectionis)
    vowel_hebrews = []
    consonant_hebrews = []
    for i, hc in enumerate(HEBREW_CHARS):
        phoneme = HEBREW_TO_ITALIAN.get(hc, "?")
        if phoneme in ('a', 'e', 'i', 'o', 'u'):
            vowel_hebrews.append(i)
        else:
            consonant_hebrews.append(i)

    # Ordina char EVA per vocalità
    sorted_by_vocality = sorted(range(N_EVA),
                                 key=lambda i: vocality.get(i, 0),
                                 reverse=True)

    # Ordina vocali e consonanti ebraiche per frequenza in italiano
    def italian_freq(heb_idx):
        ph = HEBREW_TO_ITALIAN.get(HEBREW_CHARS[heb_idx], "?")
        return sum(ITALIAN_LETTER_FREQS.get(c, 0) for c in ph)

    vowel_hebrews.sort(key=italian_freq, reverse=True)
    consonant_hebrews.sort(key=italian_freq, reverse=True)

    mapping = [0] * N_EVA
    n_vowels = len(vowel_hebrews)
    used = set()

    # Assegna le prime N_vowels posizioni ai char più "vocalici"
    for rank in range(min(n_vowels, N_EVA)):
        eva_idx = sorted_by_vocality[rank]
        heb_idx = vowel_hebrews[rank] if rank < n_vowels else consonant_hebrews[0]
        mapping[eva_idx] = heb_idx
        used.add(heb_idx)

    # Assegna il resto alle consonanti
    cons_idx = 0
    for rank in range(n_vowels, N_EVA):
        eva_idx = sorted_by_vocality[rank]
        while cons_idx < len(consonant_hebrews) and consonant_hebrews[cons_idx] in used:
            cons_idx += 1
        if cons_idx < len(consonant_hebrews):
            mapping[eva_idx] = consonant_hebrews[cons_idx]
            used.add(consonant_hebrews[cons_idx])
            cons_idx += 1

    return mapping


# =====================================================================
# Sezione 6 — Hill-climbing
# =====================================================================

def _build_char_index(char_words):
    """Precomputa char → lista indici parole che lo contengono."""
    index = [[] for _ in range(N_EVA)]
    for widx, (cids, _, _, _) in enumerate(char_words):
        for c in set(cids):
            index[c].append(widx)
    return index


def perturb_mapping(mapping, n_swaps):
    """Perturba mapping con n swap casuali."""
    m = list(mapping)
    for _ in range(n_swaps):
        i, j = random.sample(range(N_EVA), 2)
        m[i], m[j] = m[j], m[i]
    return m


def hill_climb(mapping, char_words, lexicon_set, direction,
               char_index, max_iter=300):
    """Hill-climbing steepest-ascent con delta scoring ottimizzato.

    Mosse: C(19,2)=171 swap + 19×(22-19)=57 replace = 228 vicini/iterazione.
    """
    mapping = list(mapping)

    # Score iniziali per parola
    word_scores = []
    total_score = 0
    for cids, word, count, sections in char_words:
        s = _word_score(cids, mapping, direction, lexicon_set, count)
        word_scores.append(s)
        total_score += s

    n_iters = 0

    for iteration in range(max_iter):
        n_iters = iteration + 1
        best_move = None
        best_delta = 0

        # --- Swap moves: C(19,2) = 171 ---
        for i, j in combinations(range(N_EVA), 2):
            affected = set(char_index[i]) | set(char_index[j])
            mapping[i], mapping[j] = mapping[j], mapping[i]
            delta = 0
            for widx in affected:
                cids, _, count, _ = char_words[widx]
                new_s = _word_score(cids, mapping, direction,
                                    lexicon_set, count)
                delta += new_s - word_scores[widx]
            mapping[i], mapping[j] = mapping[j], mapping[i]

            if delta > best_delta:
                best_move = ("swap", i, j)
                best_delta = delta

        # --- Replace moves: 19 × |unused| ---
        used = set(mapping)
        unused = [c for c in range(N_HEBREW) if c not in used]
        for i in range(N_EVA):
            old = mapping[i]
            for c in unused:
                mapping[i] = c
                delta = 0
                for widx in char_index[i]:
                    cids, _, count, _ = char_words[widx]
                    new_s = _word_score(cids, mapping, direction,
                                        lexicon_set, count)
                    delta += new_s - word_scores[widx]
                if delta > best_delta:
                    best_move = ("replace", i, old, c)
                    best_delta = delta
            mapping[i] = old

        if best_move is None or best_delta <= 0:
            break

        # Applica mossa migliore
        if best_move[0] == "swap":
            _, i, j = best_move
            affected = set(char_index[i]) | set(char_index[j])
            mapping[i], mapping[j] = mapping[j], mapping[i]
        else:
            _, i, _old, new_val = best_move
            affected = set(char_index[i])
            mapping[i] = new_val

        # Aggiorna score parole affette
        for widx in affected:
            cids, _, count, _ = char_words[widx]
            new_s = _word_score(cids, mapping, direction, lexicon_set, count)
            total_score += new_s - word_scores[widx]
            word_scores[widx] = new_s

    return mapping, total_score, n_iters


def search(char_words, lexicon_set, eva_freqs, words_data, eva_to_index,
           n_restarts=30):
    """Ricerca multi-restart: n_restarts per ciascuna direzione.

    Due strategie di inizializzazione:
    1. frequency_guided_mapping (allinea frequenze)
    2. positional_guided_mapping (usa profili posizionali)

    Returns: dict {"ltr": {...}, "rtl": {...}}
    """
    freq_mapping = frequency_guided_mapping(eva_freqs)
    pos_mapping = positional_guided_mapping(words_data, eva_to_index)
    char_index = _build_char_index(char_words)

    results = {}

    for direction in ("ltr", "rtl"):
        best_mapping = None
        best_score = -1
        all_scores = []

        for restart in range(n_restarts):
            if restart == 0:
                m = list(freq_mapping)
            elif restart == 1:
                m = list(pos_mapping)
            elif restart % 2 == 0:
                m = perturb_mapping(freq_mapping, random.randint(2, 6))
            else:
                m = perturb_mapping(pos_mapping, random.randint(2, 6))

            m, score, iters = hill_climb(
                m, char_words, lexicon_set, direction, char_index
            )
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_mapping = list(m)

        # Re-score il migliore per ottenere matches dettagliati
        _, best_matches = score_mapping(
            best_mapping, char_words, lexicon_set, direction
        )

        results[direction] = {
            "mapping": best_mapping,
            "score": best_score,
            "matches": best_matches,
            "all_scores": sorted(all_scores, reverse=True),
        }

        top5 = sorted(all_scores, reverse=True)[:5]
        click.echo(f"    {direction.upper()}: best={best_score}  "
                    f"top5={top5}  ({n_restarts} restart)")

    return results


# =====================================================================
# Sezione 7 — Validazione
# =====================================================================

def validate_domains(matches, domain_sets):
    """Congruenza dominio: parole botaniche da pagine H, ecc.

    Riusa pattern di decipher.py.
    """
    stats = defaultdict(lambda: {"congruent": 0, "total": 0, "words": []})

    for eva_word, hebrew, italian, count, sections in matches:
        word_domains = [d for d, forms in domain_sets.items()
                        if italian in forms]
        if not word_domains:
            continue

        expected = {SECTION_TO_DOMAIN[s] for s in sections
                    if s in SECTION_TO_DOMAIN}

        for wd in word_domains:
            stats[wd]["total"] += 1
            if wd in expected:
                stats[wd]["congruent"] += 1
                stats[wd]["words"].append((eva_word, italian))

    result = {}
    for domain, s in stats.items():
        total = s["total"]
        result[domain] = {
            "congruent": s["congruent"],
            "total": total,
            "ratio": round(s["congruent"] / total, 3) if total else 0.0,
            "sample_words": s["words"][:10],
        }
    return result


def validate_letter_frequencies(matches, direction):
    """Correlazione frequenze lettere decodificate vs italiano medievale."""
    decoded_chars = Counter()
    for _, _, italian, count, _ in matches:
        for ch in italian:
            decoded_chars[ch] += count

    total = sum(decoded_chars.values()) or 1
    decoded_freqs = {ch: decoded_chars[ch] / total for ch in decoded_chars}

    # Calcola correlazione con frequenze di riferimento
    all_chars = set(decoded_freqs) | set(ITALIAN_LETTER_FREQS)
    x = [decoded_freqs.get(c, 0) for c in sorted(all_chars)]
    y = [ITALIAN_LETTER_FREQS.get(c, 0) for c in sorted(all_chars)]

    if len(x) > 1:
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        correlation = 0.0

    return {
        "correlation": round(correlation, 4),
        "decoded_top10": dict(
            sorted(decoded_freqs.items(), key=lambda x: -x[1])[:10]
        ),
    }


def validate_vowel_ratio(mapping, char_words, direction):
    """Rapporto vocali nel testo decodificato.

    Italiano atteso: ~47% vocali.
    """
    vowels = set('aeiou')
    vowel_count = 0
    total_count = 0

    for char_indices, word, count, sections in char_words:
        _, italian = apply_mapping_to_italian(char_indices, mapping, direction)
        for ch in italian:
            total_count += count
            if ch in vowels:
                vowel_count += count

    ratio = vowel_count / total_count if total_count else 0
    return {
        "vowel_ratio": round(ratio, 4),
        "expected_italian": 0.47,
        "plausible": 0.35 <= ratio <= 0.55,
    }


def validate_bigrams(matches):
    """Correlazione bigrammi decodificati vs italiano."""
    bigram_set = set(ITALIAN_COMMON_BIGRAMS)
    decoded_bigrams = Counter()
    total = 0

    for _, _, italian, count, _ in matches:
        for i in range(len(italian) - 1):
            bg = italian[i:i+2]
            decoded_bigrams[bg] += count
            total += count

    hit_count = sum(decoded_bigrams.get(bg, 0) for bg in bigram_set)
    hit_ratio = hit_count / total if total else 0

    return {
        "bigram_hit_ratio": round(hit_ratio, 4),
        "top_decoded_bigrams": [
            {"bigram": bg, "count": c}
            for bg, c in decoded_bigrams.most_common(20)
        ],
    }


# =====================================================================
# Sezione 8 — Report + Output
# =====================================================================

def _build_report(results, baseline, eva_chars, gloss_lookup, domain_sets,
                  char_words):
    """Costruisce report completo con verdetto."""
    best_dir = max(results, key=lambda d: results[d]["score"])
    best = results[best_dir]

    # Criteri di falsificazione
    is_significant = best["score"] > baseline["threshold"]
    n_distinct = len(best["matches"])
    enough_matches = n_distinct >= 50  # soglia più bassa perché lessico più piccolo

    domain_val = validate_domains(best["matches"], domain_sets)
    freq_val = validate_letter_frequencies(best["matches"], best_dir)
    vowel_val = validate_vowel_ratio(best["mapping"], char_words, best_dir)
    bigram_val = validate_bigrams(best["matches"])

    verdict = "SIGNIFICATIVO" if (is_significant and enough_matches) \
        else "NON SIGNIFICATIVO"

    # Top matches con glosse
    sorted_matches = sorted(best["matches"],
                            key=lambda m: m[3], reverse=True)
    top_matches = []
    for eva_word, hebrew, italian, count, sections in sorted_matches[:50]:
        top_matches.append({
            "eva": eva_word,
            "hebrew": hebrew,
            "italian": italian,
            "gloss": gloss_lookup.get(italian, "?"),
            "count": count,
            "sections": sorted(sections),
        })

    # Mapping leggibile: EVA → ebraico → fonema italiano
    readable_mapping = {}
    for i, eva_ch in enumerate(eva_chars):
        heb_idx = best["mapping"][i]
        hc = HEBREW_CHARS[heb_idx]
        name = CONSONANT_NAMES.get(hc, "?")
        phoneme = HEBREW_TO_ITALIAN.get(hc, "?")
        readable_mapping[eva_ch] = {
            "hebrew_char": hc,
            "hebrew_name": name,
            "italian_phoneme": phoneme,
        }

    return {
        "verdict": verdict,
        "best_direction": best_dir,
        "best_score": best["score"],
        "baseline": baseline,
        "score_vs_threshold": (round(best["score"] / baseline["threshold"], 2)
                               if baseline["threshold"] > 0 else 0),
        "n_distinct_matches": n_distinct,
        "n_words_scored": len(char_words),
        "mapping": {
            "indices": best["mapping"],
            "readable": readable_mapping,
        },
        "criteria": {
            "significant_vs_random": is_significant,
            "enough_matches_50": enough_matches,
            "domain_congruence": domain_val,
            "letter_frequency_correlation": freq_val,
            "vowel_ratio": vowel_val,
            "bigram_analysis": bigram_val,
        },
        "ltr_score": results["ltr"]["score"],
        "rtl_score": results["rtl"]["score"],
        "ltr_n_matches": len(results["ltr"]["matches"]),
        "rtl_n_matches": len(results["rtl"]["matches"]),
        "ltr_all_scores": results["ltr"]["all_scores"],
        "rtl_all_scores": results["rtl"]["all_scores"],
        "top_matches": top_matches,
    }


def _print_report(report):
    """Stampa report su console."""
    click.echo(f"\n{'=' * 60}")
    click.echo("  RISULTATO DECIFRAZIONE GIUDEO-ITALIANA")
    click.echo(f"{'=' * 60}")

    bl = report["baseline"]
    click.echo(f"\n  Baseline casuale ({bl['n_trials']} mapping):")
    click.echo(f"    Media={bl['mean']:.1f}  Std={bl['std']:.1f}  "
               f"Max={bl['max']:.1f}")
    click.echo(f"    Soglia (mean + 4\u03c3): {bl['threshold']:.1f}")

    click.echo(f"\n  Score migliori:")
    click.echo(f"    LTR: {report['ltr_score']} "
               f"({report['ltr_n_matches']} match distinti)")
    click.echo(f"    RTL: {report['rtl_score']} "
               f"({report['rtl_n_matches']} match distinti)")

    click.echo(f"\n  Direzione migliore: {report['best_direction'].upper()}")
    click.echo(f"  Score: {report['best_score']}")
    click.echo(f"  Rapporto vs soglia: {report['score_vs_threshold']}x")
    click.echo(f"  Match distinti: {report['n_distinct_matches']}")

    click.echo("\n  Mapping trovato (EVA \u2192 ebraico \u2192 italiano):")
    for eva_ch, info in sorted(report["mapping"]["readable"].items()):
        click.echo(f"    {eva_ch} \u2192 {info['hebrew_char']} "
                    f"({info['hebrew_name']:8s}) \u2192 {info['italian_phoneme']}")

    click.echo("\n  Top 20 parole riconosciute:")
    for m in report["top_matches"][:20]:
        sec = ",".join(m["sections"])
        click.echo(f"    {m['eva']:12s} \u2192 {m['hebrew']:8s} \u2192 "
                    f"{m['italian']:12s} ({m['count']:4d}x) [{sec}]  "
                    f"{m['gloss']}")

    # Validazioni
    c = report["criteria"]

    click.echo("\n  Congruenza dominio:")
    for domain, s in c.get("domain_congruence", {}).items():
        click.echo(f"    {domain}: {s['congruent']}/{s['total']} "
                    f"({s['ratio']:.1%})")

    freq = c.get("letter_frequency_correlation", {})
    click.echo(f"\n  Correlazione frequenze lettere: "
               f"{freq.get('correlation', 0):.4f}")

    vowel = c.get("vowel_ratio", {})
    click.echo(f"  Rapporto vocali: {vowel.get('vowel_ratio', 0):.1%} "
               f"(atteso ~{vowel.get('expected_italian', 0.47):.0%}, "
               f"plausibile: {vowel.get('plausible', False)})")

    bigram = c.get("bigram_analysis", {})
    click.echo(f"  Hit ratio bigrammi: {bigram.get('bigram_hit_ratio', 0):.1%}")

    click.echo(f"\n  {'=' * 40}")
    click.echo(f"  VERDETTO: {report['verdict']}")
    click.echo(f"  {'=' * 40}")

    click.echo("\n  Criteri di falsificazione:")
    ok1 = "\u2713" if c["significant_vs_random"] else "\u2717"
    ok2 = "\u2713" if c["enough_matches_50"] else "\u2717"
    ok3 = "\u2713" if vowel.get("plausible", False) else "\u2717"
    click.echo(f"    {ok1} Score > soglia casuale")
    click.echo(f"    {ok2} \u226550 match distinti")
    click.echo(f"    {ok3} Rapporto vocali plausibile (35-55%)")


def _save_matches_txt(matches, gloss_lookup, output_path):
    """Salva elenco match come testo leggibile."""
    sorted_m = sorted(matches, key=lambda m: m[3], reverse=True)
    lines = [
        "# Parole Voynichesi riconosciute nel lessico italiano",
        f"# Totale: {len(matches)} parole distinte",
        "",
        f"{'EVA':<15s} {'Ebraico':<10s} {'Italiano':<15s} {'Count':>6s}  "
        f"{'Sezioni':<12s} Glossa",
        "-" * 85,
    ]

    for eva_word, hebrew, italian, count, sections in sorted_m:
        gloss = gloss_lookup.get(italian, "?")
        sec = ",".join(sorted(sections))
        lines.append(f"{eva_word:<15s} {hebrew:<10s} {italian:<15s} "
                     f"{count:>6d}  {sec:<12s} {gloss}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_decoded_text(words_data, mapping, direction, eva_to_index,
                       output_path):
    """Salva testo decodificato pagina per pagina.

    Formato: folio | EVA originale | testo italiano decodificato
    """
    lines = [
        "# Testo Voynich decodificato (ipotesi giudeo-italiana)",
        f"# Direzione: {direction.upper()}",
        f"# Formato: folio | EVA | italiano",
        "",
    ]

    for page in words_data["pages"]:
        folio = page["folio"]
        section = page.get("section", "?")
        lines.append(f"=== {folio} [sezione {section}] ===")

        for line_words in page.get("line_words", []):
            eva_line = " ".join(line_words)
            decoded_words = []
            for w in line_words:
                char_indices = []
                valid = True
                for ch in w:
                    if ch in eva_to_index:
                        char_indices.append(eva_to_index[ch])
                    else:
                        valid = False
                        break
                if valid and char_indices:
                    _, italian = apply_mapping_to_italian(
                        tuple(char_indices), mapping, direction
                    )
                    decoded_words.append(italian)
                else:
                    decoded_words.append(f"[{w}]")

            italian_line = " ".join(decoded_words)
            lines.append(f"  {eva_line}")
            lines.append(f"  → {italian_line}")
            lines.append("")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Sezione 9 — Entry point
# =====================================================================

def run(config, force=False, n_restarts=30, direction="both"):
    """Entry point: tentativo di decifrazione giudeo-italiana."""
    report_path = config.stats_dir / "italian_decipher_report.json"

    if report_path.exists() and not force:
        click.echo("  Italian decipher report esistente. Usa --force per rieseguire.")
        return

    config.ensure_dirs()
    print_header("DECIFRAZIONE GIUDEO-ITALIANA — Ipotesi medico padovano")

    # --- 1. Carica dati ---
    print_step("Caricamento charset EVA...")
    eva_to_index, eva_chars = load_eva_charset(config)
    click.echo(f"    {len(eva_chars)} char EVA: {' '.join(eva_chars)}")

    print_step("Caricamento lessico italiano...")
    lexicon_set, gloss_lookup, domain_sets = load_italian_lexicon(config)
    click.echo(f"    {len(lexicon_set)} forme fonemiche in lessico")

    print_step("Parsing parole EVA...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    words_data = parse_eva_words(eva_file)
    click.echo(f"    {words_data['total_words']} parole, "
               f"{words_data['unique_words']} uniche")

    print_step("Preparazione parole per scoring...")
    char_words = prepare_char_words(words_data, eva_to_index)
    eva_freqs = compute_char_freqs(words_data, eva_to_index)
    total_tokens = sum(w[2] for w in char_words)
    click.echo(f"    {len(char_words)} forme uniche (len \u2265 3), "
               f"{total_tokens} token coperti")

    # --- 2. Baseline casuale ---
    print_step("Baseline casuale (200 mapping random)...")
    random.seed(42)
    baseline = compute_random_baseline(char_words, lexicon_set)
    click.echo(f"    Media={baseline['mean']:.1f}  Std={baseline['std']:.1f}  "
               f"Max={baseline['max']:.1f}")
    click.echo(f"    Soglia significativit\u00e0 (mean + 4\u03c3): "
               f"{baseline['threshold']:.1f}")

    # --- 3. Ricerca ---
    print_step(f"Hill-climbing ({n_restarts} restart \u00d7 2 direzioni)...")
    t0 = time.time()
    results = search(char_words, lexicon_set, eva_freqs, words_data,
                     eva_to_index, n_restarts=n_restarts)
    elapsed = time.time() - t0
    click.echo(f"    Completato in {elapsed:.1f}s")

    # --- 4. Report ---
    print_step("Costruzione report...")
    report = _build_report(results, baseline, eva_chars, gloss_lookup,
                           domain_sets, char_words)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Salvato: {report_path}")

    best_dir = report["best_direction"]
    matches_path = config.stats_dir / "italian_decipher_matches.txt"
    _save_matches_txt(results[best_dir]["matches"], gloss_lookup, matches_path)
    click.echo(f"    Salvato: {matches_path}")

    # Salva testo decodificato
    print_step("Salvataggio testo decodificato...")
    text_path = config.stats_dir / "italian_decoded_text.txt"
    _save_decoded_text(words_data, results[best_dir]["mapping"], best_dir,
                       eva_to_index, text_path)
    click.echo(f"    Salvato: {text_path}")

    _print_report(report)
