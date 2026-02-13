"""
Decoder vincolato per il manoscritto Voynich.

Ipotesi: cifrario omofonico con 10 gruppi funzionali → consonanti ebraiche.
Metodo: hill-climbing con multi-restart, validazione da dizionario accademico.
Criteri di falsificazione definiti prima della ricerca.
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
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Costanti
# =====================================================================

HEBREW_CHARS = "AbgdhwzXJyklmnsEpCqrSt"  # 22 consonanti
N_HEBREW = len(HEBREW_CHARS)
N_GROUPS = 10

# Pesi per lunghezza (basati su probabilità match casuale ≈ n_forms / 22^k)
# len 3: ~20% match casuale → peso 1
# len 4: ~1% → peso 5
# len 5: ~0.02% → peso 25
# len 6+: ~0.0005% → peso 100
LENGTH_WEIGHTS = {3: 1, 4: 5, 5: 25, 6: 100, 7: 100}

# Sezione manoscritto → dominio lessicale
SECTION_TO_DOMAIN = {
    "H": "botanical",
    "S": "astronomical",
    "Z": "astronomical",
    "B": "medical",
    "P": "medical",
}

# Top N parole uniche per scoring (tradeoff velocità/copertura)
TOP_N_WORDS = 1000


# =====================================================================
# Sezione 1 — Preparazione dati
# =====================================================================

def load_homophone_groups(config):
    """Carica gruppi omofonici da cipher_hypothesis.json.

    Returns:
        eva_to_group: {eva_char → group_index 0..9}
        groups: list[list[str]], groups[i] = chars EVA nel gruppo i
    """
    path = config.stats_dir / "cipher_hypothesis.json"
    with open(path) as f:
        data = json.load(f)

    raw = data["homophone_analysis"]["groups"]
    sorted_ids = sorted(raw.keys(), key=int)

    groups = []
    eva_to_group = {}
    for idx, gid in enumerate(sorted_ids):
        chars = raw[gid]
        groups.append(chars)
        for ch in chars:
            eva_to_group[ch] = idx

    return eva_to_group, groups


def load_lexicon(config):
    """Carica lessico da lexicon.json.

    Returns:
        lexicon_set: set di forme consonantali
        gloss_lookup: {consonantal_form → gloss}
        domain_sets: {domain → set di forme}
    """
    path = config.lexicon_dir / "lexicon.json"
    with open(path) as f:
        data = json.load(f)

    lexicon_set = set()
    gloss_lookup = {}
    domain_sets = defaultdict(set)

    for domain, entries in data["by_domain"].items():
        for entry in entries:
            form = entry["consonants"]
            lexicon_set.add(form)
            if form not in gloss_lookup:
                gloss_lookup[form] = entry["gloss"]
            domain_sets[domain].add(form)

    return lexicon_set, gloss_lookup, dict(domain_sets)


def load_hebrew_freqs(config):
    """Carica frequenze consonanti ebraiche da lexicon_report.json."""
    path = config.stats_dir / "lexicon_report.json"
    with open(path) as f:
        data = json.load(f)
    return data["eva_compatibility"]["hebrew_consonant_frequencies"]


def prepare_group_words(words_data, eva_to_group, top_n=TOP_N_WORDS):
    """Converte parole EVA in rappresentazione a gruppi.

    Filtra: chars sconosciuti, lunghezza < 3.
    Prende le top_n più frequenti per velocità.

    Returns:
        list di (group_indices_tuple, eva_word, count, sections_frozenset)
    """
    word_counter = Counter(words_data["words"])

    word_sections = defaultdict(set)
    for page in words_data["pages"]:
        section = page.get("section", "?")
        for w in page["words"]:
            word_sections[w].add(section)

    result = []
    for word, count in word_counter.most_common():
        group_indices = []
        valid = True
        for ch in word:
            if ch in eva_to_group:
                group_indices.append(eva_to_group[ch])
            else:
                valid = False
                break

        if not valid or len(group_indices) < 3:
            continue

        result.append((
            tuple(group_indices),
            word,
            count,
            frozenset(word_sections[word]),
        ))

        if len(result) >= top_n:
            break

    return result


def compute_group_freqs(words_data, eva_to_group, groups):
    """Frequenza combinata di ogni gruppo EVA nel corpus."""
    char_counter = Counter()
    for w in words_data["words"]:
        for ch in w:
            char_counter[ch] += 1

    total = sum(char_counter.values())
    group_freqs = []
    for chars in groups:
        freq = sum(char_counter.get(ch, 0) for ch in chars) / total
        group_freqs.append(freq)

    return group_freqs


# =====================================================================
# Sezione 2 — Scoring
# =====================================================================

def apply_mapping_to_word(group_indices, mapping, direction):
    """Converte indici di gruppo in stringa ebraica consonantale."""
    indices = group_indices if direction == "ltr" else group_indices[::-1]
    return "".join(HEBREW_CHARS[mapping[g]] for g in indices)


def score_mapping(mapping, group_words, lexicon_set, direction):
    """Scora un mapping contando match pesati nel lessico.

    Returns:
        (score, matches) dove matches = [(eva_word, hebrew, count, sections)]
    """
    score = 0
    matches = []
    for group_indices, word, count, sections in group_words:
        hebrew = apply_mapping_to_word(group_indices, mapping, direction)
        hlen = len(hebrew)
        if hlen >= 3 and hebrew in lexicon_set:
            weight = LENGTH_WEIGHTS.get(min(hlen, 7), 100)
            score += count * weight
            matches.append((word, hebrew, count, sections))
    return score, matches


def _word_score(group_indices, mapping, direction, lexicon_set, count):
    """Score singola parola (helper per hill-climbing)."""
    hebrew = apply_mapping_to_word(group_indices, mapping, direction)
    if len(hebrew) >= 3 and hebrew in lexicon_set:
        weight = LENGTH_WEIGHTS.get(min(len(hebrew), 7), 100)
        return count * weight
    return 0


# =====================================================================
# Sezione 3 — Baseline casuale
# =====================================================================

def compute_random_baseline(group_words, lexicon_set, n_trials=200):
    """Score di 200 mapping casuali → soglia di significatività.

    Returns:
        dict con mean, std, max, threshold (mean + 4σ)
    """
    scores = []
    for _ in range(n_trials):
        perm = random.sample(range(N_HEBREW), N_GROUPS)
        for direction in ("ltr", "rtl"):
            s, _ = score_mapping(perm, group_words, lexicon_set, direction)
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
# Sezione 4 — Mapping iniziale frequency-guided
# =====================================================================

def frequency_guided_mapping(group_freqs, hebrew_freqs):
    """Mapping iniziale: gruppo più frequente → consonante più frequente.

    Returns: list di 10 indici in HEBREW_CHARS (iniettiva).
    """
    sorted_groups = sorted(range(N_GROUPS),
                           key=lambda i: group_freqs[i], reverse=True)

    hebrew_sorted = sorted(hebrew_freqs.keys(),
                           key=lambda c: hebrew_freqs[c], reverse=True)

    mapping = [0] * N_GROUPS
    for rank, group_idx in enumerate(sorted_groups):
        char = hebrew_sorted[rank]
        mapping[group_idx] = HEBREW_CHARS.index(char)

    return mapping


# =====================================================================
# Sezione 5 — Ricerca hill-climbing
# =====================================================================

def _build_group_index(group_words):
    """Precomputa gruppo → lista indici parole che lo contengono."""
    index = [[] for _ in range(N_GROUPS)]
    for widx, (gids, _, _, _) in enumerate(group_words):
        for g in set(gids):
            index[g].append(widx)
    return index


def perturb_mapping(mapping, n_swaps):
    """Perturba mapping con n swap casuali tra gruppi."""
    m = list(mapping)
    for _ in range(n_swaps):
        i, j = random.sample(range(N_GROUPS), 2)
        m[i], m[j] = m[j], m[i]
    return m


def hill_climb(mapping, group_words, lexicon_set, direction,
               group_index, max_iter=200):
    """Hill-climbing steepest-ascent con delta scoring ottimizzato.

    Mosse: 45 swap + ~120 replace = ~165 vicini per iterazione.
    Solo le parole contenenti i gruppi modificati vengono ri-scorate.

    Returns: (mapping, score, n_iterations)
    """
    mapping = list(mapping)

    # Score iniziali per parola
    word_scores = []
    total_score = 0
    for gids, word, count, sections in group_words:
        s = _word_score(gids, mapping, direction, lexicon_set, count)
        word_scores.append(s)
        total_score += s

    n_iters = 0

    for iteration in range(max_iter):
        n_iters = iteration + 1
        best_move = None
        best_delta = 0

        # --- Swap moves: C(10,2) = 45 ---
        for i, j in combinations(range(N_GROUPS), 2):
            affected = set(group_index[i]) | set(group_index[j])
            mapping[i], mapping[j] = mapping[j], mapping[i]
            delta = 0
            for widx in affected:
                gids, _, count, _ = group_words[widx]
                new_s = _word_score(gids, mapping, direction,
                                    lexicon_set, count)
                delta += new_s - word_scores[widx]
            mapping[i], mapping[j] = mapping[j], mapping[i]

            if delta > best_delta:
                best_move = ("swap", i, j)
                best_delta = delta

        # --- Replace moves: 10 × |unused| ---
        used = set(mapping)
        unused = [c for c in range(N_HEBREW) if c not in used]
        for i in range(N_GROUPS):
            old = mapping[i]
            for c in unused:
                mapping[i] = c
                delta = 0
                for widx in group_index[i]:
                    gids, _, count, _ = group_words[widx]
                    new_s = _word_score(gids, mapping, direction,
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
            affected = set(group_index[i]) | set(group_index[j])
            mapping[i], mapping[j] = mapping[j], mapping[i]
        else:
            _, i, _old, new_val = best_move
            affected = set(group_index[i])
            mapping[i] = new_val

        # Aggiorna score parole affette
        for widx in affected:
            gids, _, count, _ = group_words[widx]
            new_s = _word_score(gids, mapping, direction, lexicon_set, count)
            total_score += new_s - word_scores[widx]
            word_scores[widx] = new_s

    return mapping, total_score, n_iters


def search(group_words, lexicon_set, group_freqs, hebrew_freqs,
           n_restarts=20):
    """Ricerca multi-restart: n_restarts per ciascuna direzione.

    Returns: dict {"ltr": {...}, "rtl": {...}} con mapping, score, matches.
    """
    base_mapping = frequency_guided_mapping(group_freqs, hebrew_freqs)
    group_index = _build_group_index(group_words)

    results = {}

    for direction in ("ltr", "rtl"):
        best_mapping = None
        best_score = -1
        best_matches = []
        all_scores = []

        for restart in range(n_restarts):
            if restart == 0:
                m = list(base_mapping)
            else:
                m = perturb_mapping(base_mapping, random.randint(2, 4))

            m, score, iters = hill_climb(
                m, group_words, lexicon_set, direction, group_index
            )
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_mapping = list(m)

        # Re-score il migliore per ottenere matches dettagliati
        _, best_matches = score_mapping(
            best_mapping, group_words, lexicon_set, direction
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
# Sezione 6 — Validazione dominio
# =====================================================================

def validate_domains(matches, domain_sets):
    """Congruenza dominio: parole botaniche da pagine H, ecc.

    Returns: dict {domain: {congruent, total, ratio, sample_words}}
    """
    stats = defaultdict(lambda: {"congruent": 0, "total": 0, "words": []})

    for eva_word, hebrew, count, sections in matches:
        word_domains = [d for d, forms in domain_sets.items()
                        if hebrew in forms]
        if not word_domains:
            continue

        expected = {SECTION_TO_DOMAIN[s] for s in sections
                    if s in SECTION_TO_DOMAIN}

        for wd in word_domains:
            stats[wd]["total"] += 1
            if wd in expected:
                stats[wd]["congruent"] += 1
                stats[wd]["words"].append((eva_word, hebrew))

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


# =====================================================================
# Sezione 7 — Report
# =====================================================================

def _build_report(results, baseline, groups, gloss_lookup, domain_sets,
                  group_words):
    """Costruisce report completo con verdetto."""
    best_dir = max(results, key=lambda d: results[d]["score"])
    best = results[best_dir]

    # Criteri di falsificazione
    is_significant = best["score"] > baseline["threshold"]
    n_distinct = len(best["matches"])
    enough_matches = n_distinct >= 100

    domain_val = validate_domains(best["matches"], domain_sets)

    verdict = "SIGNIFICATIVO" if (is_significant and enough_matches) \
        else "NON SIGNIFICATIVO"

    # Top matches con glosse
    sorted_matches = sorted(best["matches"],
                            key=lambda m: m[2], reverse=True)
    top_matches = []
    for eva_word, hebrew, count, sections in sorted_matches[:50]:
        top_matches.append({
            "eva": eva_word,
            "hebrew": hebrew,
            "gloss": gloss_lookup.get(hebrew, "?"),
            "count": count,
            "sections": sorted(sections),
        })

    readable_mapping = {}
    for i, chars in enumerate(groups):
        hc = HEBREW_CHARS[best["mapping"][i]]
        name = CONSONANT_NAMES.get(hc, "?")
        readable_mapping[",".join(chars)] = f"{hc} ({name})"

    return {
        "verdict": verdict,
        "best_direction": best_dir,
        "best_score": best["score"],
        "baseline": baseline,
        "score_vs_threshold": (round(best["score"] / baseline["threshold"], 2)
                               if baseline["threshold"] > 0 else 0),
        "n_distinct_matches": n_distinct,
        "n_words_scored": len(group_words),
        "mapping": {
            "indices": best["mapping"],
            "readable": readable_mapping,
        },
        "criteria": {
            "significant_vs_random": is_significant,
            "enough_matches_100": enough_matches,
            "domain_congruence": domain_val,
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
    click.echo("  RISULTATO DECIFRAZIONE")
    click.echo(f"{'=' * 60}")

    bl = report["baseline"]
    click.echo(f"\n  Baseline casuale ({bl['n_trials']} mapping):")
    click.echo(f"    Media={bl['mean']:.1f}  Std={bl['std']:.1f}  "
               f"Max={bl['max']:.1f}")
    click.echo(f"    Soglia (mean + 4σ): {bl['threshold']:.1f}")

    click.echo(f"\n  Score migliori:")
    click.echo(f"    LTR: {report['ltr_score']} "
               f"({report['ltr_n_matches']} match distinti)")
    click.echo(f"    RTL: {report['rtl_score']} "
               f"({report['rtl_n_matches']} match distinti)")

    click.echo(f"\n  Direzione migliore: {report['best_direction'].upper()}")
    click.echo(f"  Score: {report['best_score']}")
    click.echo(f"  Rapporto vs soglia: {report['score_vs_threshold']}x")
    click.echo(f"  Match distinti: {report['n_distinct_matches']}")

    click.echo("\n  Mapping trovato:")
    for group_label, hebrew_label in report["mapping"]["readable"].items():
        click.echo(f"    {{{group_label}}} → {hebrew_label}")

    click.echo("\n  Top 20 parole riconosciute:")
    for m in report["top_matches"][:20]:
        sec = ",".join(m["sections"])
        click.echo(f"    {m['eva']:12s} → {m['hebrew']:8s}  "
                    f"({m['count']:4d}x) [{sec}]  {m['gloss']}")

    click.echo("\n  Congruenza dominio:")
    for domain, s in report["criteria"]["domain_congruence"].items():
        click.echo(f"    {domain}: {s['congruent']}/{s['total']} "
                    f"({s['ratio']:.1%})")

    click.echo(f"\n  {'=' * 40}")
    click.echo(f"  VERDETTO: {report['verdict']}")
    click.echo(f"  {'=' * 40}")

    c = report["criteria"]
    click.echo("\n  Criteri di falsificazione:")
    ok1 = "\u2713" if c["significant_vs_random"] else "\u2717"
    ok2 = "\u2713" if c["enough_matches_100"] else "\u2717"
    click.echo(f"    {ok1} Score > soglia casuale")
    click.echo(f"    {ok2} \u2265100 match distinti")


def _save_matches_txt(matches, gloss_lookup, output_path):
    """Salva elenco match come testo leggibile."""
    sorted_m = sorted(matches, key=lambda m: m[2], reverse=True)
    lines = [
        "# Parole Voynichesi riconosciute nel lessico ebraico",
        f"# Totale: {len(matches)} parole distinte",
        "",
        f"{'EVA':<15s} {'Ebraico':<10s} {'Count':>6s}  "
        f"{'Sezioni':<12s} Glossa",
        "-" * 75,
    ]

    for eva_word, hebrew, count, sections in sorted_m:
        gloss = gloss_lookup.get(hebrew, "?")
        sec = ",".join(sorted(sections))
        lines.append(f"{eva_word:<15s} {hebrew:<10s} {count:>6d}  "
                     f"{sec:<12s} {gloss}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# Sezione 8 — Entry point
# =====================================================================

def run(config, force=False, n_restarts=20):
    """Entry point: tentativo di decifrazione vincolata."""
    report_path = config.stats_dir / "decipher_report.json"

    if report_path.exists() and not force:
        click.echo("  Decipher report esistente. Usa --force per rieseguire.")
        return

    config.ensure_dirs()
    print_header("DECIFRAZIONE VINCOLATA — Ipotesi ebraica omofonica")

    # --- 1. Carica dati ---
    print_step("Caricamento gruppi omofonici...")
    eva_to_group, groups = load_homophone_groups(config)
    click.echo("    " + "  ".join(f"{{{','.join(g)}}}" for g in groups))

    print_step("Caricamento lessico ebraico...")
    lexicon_set, gloss_lookup, domain_sets = load_lexicon(config)
    click.echo(f"    {len(lexicon_set)} forme consonantali in lessico")

    print_step("Caricamento frequenze ebraiche...")
    hebrew_freqs = load_hebrew_freqs(config)

    print_step("Parsing parole EVA...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    words_data = parse_eva_words(eva_file)
    click.echo(f"    {words_data['total_words']} parole, "
               f"{words_data['unique_words']} uniche")

    print_step("Preparazione parole per scoring...")
    group_words = prepare_group_words(words_data, eva_to_group)
    group_freqs = compute_group_freqs(words_data, eva_to_group, groups)
    total_tokens = sum(w[2] for w in group_words)
    click.echo(f"    {len(group_words)} forme uniche (len \u2265 3), "
               f"{total_tokens} token coperti")
    click.echo("    Freq gruppi: " +
               "  ".join(f"g{i}={f:.3f}" for i, f in enumerate(group_freqs)))

    # --- 2. Baseline casuale ---
    print_step("Baseline casuale (200 mapping random)...")
    random.seed(42)
    baseline = compute_random_baseline(group_words, lexicon_set)
    click.echo(f"    Media={baseline['mean']:.1f}  Std={baseline['std']:.1f}  "
               f"Max={baseline['max']:.1f}")
    click.echo(f"    Soglia significatività (mean + 4σ): "
               f"{baseline['threshold']:.1f}")

    # --- 3. Ricerca ---
    print_step(f"Hill-climbing ({n_restarts} restart × 2 direzioni)...")
    t0 = time.time()
    results = search(group_words, lexicon_set, group_freqs, hebrew_freqs,
                     n_restarts=n_restarts)
    elapsed = time.time() - t0
    click.echo(f"    Completato in {elapsed:.1f}s")

    # --- 4. Report ---
    print_step("Costruzione report...")
    report = _build_report(results, baseline, groups, gloss_lookup,
                           domain_sets, group_words)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    click.echo(f"    Salvato: {report_path}")

    best_dir = report["best_direction"]
    matches_path = config.stats_dir / "decipher_matches.txt"
    _save_matches_txt(results[best_dir]["matches"], gloss_lookup, matches_path)
    click.echo(f"    Salvato: {matches_path}")

    _print_report(report)
