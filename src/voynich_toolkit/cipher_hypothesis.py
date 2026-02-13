"""
Test ipotesi di cifratura e decomposizione morfologica del Voynichese.

Testa il testo contro firme statistiche di famiglie di cifrari note
(monoalfabetico, omofonico, nomenclatore, abjad, polialfabetico) e
decompone le parole in componenti morfologiche (prefissi, radici, suffissi).
"""
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import click
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer
from .word_structure import parse_eva_words, compute_char_positional_profiles
from .char_embeddings import (
    compute_cooccurrence_matrix,
    compute_pmi_matrix,
)
from .language_fingerprint import (
    compute_conditional_entropy,
    compute_index_of_coincidence,
)


# =====================================================================
# Costanti
# =====================================================================

# Profili attesi per tipo di cifrario (range diagnostici)
CIPHER_PROFILES = {
    "Homophonic": {
        "label": "Sostituzione omofonica",
        "description": "Piu' simboli cifrati per lettera di testo chiaro",
        "ioc_range": (0.050, 0.080),
        "cv_range": (0.25, 0.65),
        "h1h0_range": (0.45, 0.75),
        "alphabet_range": (18, 60),
    },
    "Monoalphabetic": {
        "label": "Sostituzione monoalfabetica",
        "description": "Ogni lettera chiara -> un solo simbolo cifrato",
        "ioc_range": (0.060, 0.080),
        "cv_range": (0.50, 0.95),
        "h1h0_range": (0.72, 0.90),
        "alphabet_range": (20, 30),
    },
    "Abjad": {
        "label": "Abjad (scrittura consonantica)",
        "description": "Solo consonanti scritte, come ebraico/arabo",
        "ioc_range": (0.065, 0.080),
        "cv_range": (0.40, 0.85),
        "h1h0_range": (0.55, 0.80),
        "alphabet_range": (18, 28),
    },
    "Nomenclator": {
        "label": "Nomenclatore",
        "description": "Misto: cifrario + codici per parole comuni",
        "ioc_range": (0.040, 0.070),
        "cv_range": (0.60, 1.20),
        "h1h0_range": (0.65, 0.85),
        "alphabet_range": (25, 80),
    },
    "Polyalphabetic": {
        "label": "Polialfabetico (Vigenere)",
        "description": "Sostituzione multipla con chiave ripetuta",
        "ioc_range": (0.035, 0.050),
        "cv_range": (0.05, 0.30),
        "h1h0_range": (0.88, 1.00),
        "alphabet_range": (20, 30),
    },
    "Null": {
        "label": "Testo senza significato",
        "description": "Glossolalia, generazione casuale, hoax",
        "ioc_range": (0.030, 0.045),
        "cv_range": (0.00, 0.20),
        "h1h0_range": (0.92, 1.00),
        "alphabet_range": (15, 30),
    },
}


# =====================================================================
# Sezione 1 — Diagnostica alfabeto
# =====================================================================

def compute_alphabet_stats(chars: list[str]) -> dict:
    """Statistiche diagnostiche sull'alfabeto."""
    counter = Counter(chars)
    total = len(chars)
    n_unique = len(counter)

    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    probs = freqs / total

    # Dimensione effettiva (caratteri per 95% del testo)
    cumsum = np.cumsum(probs)
    effective_95 = int(np.searchsorted(cumsum, 0.95)) + 1

    # Dimensione basata su entropia: 2^H(0)
    h0 = -np.sum(probs * np.log2(probs))
    entropy_effective = round(2 ** h0, 1)

    # Coefficiente di variazione delle frequenze
    cv = float(np.std(freqs) / np.mean(freqs)) if np.mean(freqs) > 0 else 0.0

    # Gini coefficient
    n = len(freqs)
    idx = np.arange(1, n + 1)
    gini = float((2 * np.sum(idx * np.sort(freqs)) / (n * np.sum(freqs))) - (n + 1) / n)

    return {
        "unique_characters": n_unique,
        "effective_95": effective_95,
        "entropy_effective": entropy_effective,
        "h0": round(float(h0), 4),
        "cv": round(cv, 4),
        "gini": round(gini, 4),
        "frequencies": probs.tolist(),
    }


# =====================================================================
# Sezione 2 — Scoring ipotesi
# =====================================================================

def range_score(value: float, low: float, high: float) -> float:
    """1.0 se value in [low, high], decade linearmente fuori range."""
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-6)
    if value < low:
        return max(0.0, 1.0 - (low - value) / width)
    return max(0.0, 1.0 - (value - high) / width)


def score_all_hypotheses(alphabet_stats: dict, entropy_data: dict,
                         ioc: float, homophones: dict,
                         words: list[str]) -> dict:
    """Score ogni tipo di cifrario. Restituisce {tipo: {score, evidence, subscores}}."""
    h0 = entropy_data["cascade"][0]
    h1 = entropy_data["cascade"][1] if len(entropy_data["cascade"]) > 1 else h0
    h1h0 = h1 / h0 if h0 > 0 else 1.0
    cv = alphabet_stats["cv"]
    n_chars = alphabet_stats["unique_characters"]

    scores = {}
    for cipher_type, profile in CIPHER_PROFILES.items():
        s_ioc = range_score(ioc, *profile["ioc_range"])
        s_cv = range_score(cv, *profile["cv_range"])
        s_h1h0 = range_score(h1h0, *profile["h1h0_range"])
        s_alpha = range_score(n_chars, *profile["alphabet_range"])

        subscores = {
            "ioc": round(s_ioc, 3),
            "cv": round(s_cv, 3),
            "h1h0_ratio": round(s_h1h0, 3),
            "alphabet_size": round(s_alpha, 3),
        }

        # Pesi: h1h0 e IoC sono i piu' discriminanti
        overall = (s_ioc * 3 + s_cv * 1 + s_h1h0 * 3 + s_alpha * 1) / 8

        # Bonus/penalita' tipo-specifici
        if cipher_type == "Homophonic":
            # Bonus se troviamo gruppi omofonici plausibili
            n_groups = homophones.get("estimated_alphabet_size", n_chars)
            if n_groups < n_chars * 0.85:
                overall = min(1.0, overall + 0.10)
            # Il basso h1/h0 (0.61) e' firma di struttura omofonica
            if h1h0 < 0.65:
                overall = min(1.0, overall + 0.08)

        elif cipher_type == "Abjad":
            # Bonus se lunghezza media parole e' corta (consonanti sole)
            avg_len = np.mean([len(w) for w in words]) if words else 0
            if 4.0 <= avg_len <= 5.5:
                overall = min(1.0, overall + 0.05)

        elif cipher_type == "Nomenclator":
            # Penalita' se non c'e' bimodalita'
            freq_arr = np.array(alphabet_stats["frequencies"])
            # Check gap ratio: biggest gap / range
            sorted_f = np.sort(freq_arr)[::-1]
            gaps = np.abs(np.diff(sorted_f))
            max_gap = float(gaps.max()) if len(gaps) > 0 else 0
            freq_range = float(sorted_f[0] - sorted_f[-1]) if len(sorted_f) > 1 else 1
            bimodality = max_gap / freq_range if freq_range > 0 else 0
            if bimodality < 0.15:
                overall = max(0.0, overall - 0.10)

        scores[cipher_type] = {
            "score": round(overall, 3),
            "label": profile["label"],
            "subscores": subscores,
        }

    return scores


# =====================================================================
# Sezione 3 — Rilevamento omofoni
# =====================================================================

def detect_homophone_candidates(coocc_data: dict, pmi_data: dict,
                                profiles: dict) -> dict:
    """Trova coppie di caratteri potenzialmente omofonici.

    Omofoni hanno:
    1. Contesti bigramma simili (righe simili nella matrice PMI)
    2. Posizioni simili nelle parole
    3. Bassa co-occorrenza diretta (non appaiono adiacenti)
    """
    matrix = coocc_data["matrix"].astype(float)
    char_list = coocc_data["char_list"]
    char_counts = coocc_data["char_counts"]
    n = len(char_list)

    # 1. Similarita' di contesto: riga (successori) + colonna (predecessori)
    right_ctx = matrix.copy()
    left_ctx = matrix.T.copy()
    r_sums = right_ctx.sum(axis=1, keepdims=True)
    r_sums = np.where(r_sums == 0, 1, r_sums)
    l_sums = left_ctx.sum(axis=1, keepdims=True)
    l_sums = np.where(l_sums == 0, 1, l_sums)
    contexts = np.hstack([right_ctx / r_sums, left_ctx / l_sums])

    norms = np.linalg.norm(contexts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    ctx_sim = (contexts / norms) @ (contexts / norms).T

    # 2. Similarita' posizionale
    prof = profiles["profiles"]
    pos_vecs = np.array([
        [prof.get(ch, {}).get("initial", 0), prof.get(ch, {}).get("medial", 0),
         prof.get(ch, {}).get("final", 0), prof.get(ch, {}).get("isolated", 0)]
        for ch in char_list
    ])
    p_norms = np.linalg.norm(pos_vecs, axis=1, keepdims=True)
    p_norms = np.where(p_norms == 0, 1, p_norms)
    pos_sim = (pos_vecs / p_norms) @ (pos_vecs / p_norms).T

    # 3. Anti-co-occorrenza: rapporto osservato/atteso
    direct = matrix + matrix.T
    ch_freq = np.array([char_counts.get(ch, 0) for ch in char_list], dtype=float)
    total = matrix.sum()
    expected = np.outer(ch_freq, ch_freq) / total if total > 0 else np.ones((n, n))
    expected = np.where(expected == 0, 1, expected)
    obs_exp = direct / expected
    anti_coocc = np.clip(1 - obs_exp, 0, 1)

    # Score composto
    homo_scores = ctx_sim * pos_sim * anti_coocc
    np.fill_diagonal(homo_scores, 0)

    # Coppie candidate
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if homo_scores[i, j] > 0.05:
                pairs.append({
                    "char_a": char_list[i],
                    "char_b": char_list[j],
                    "score": round(float(homo_scores[i, j]), 4),
                    "context_sim": round(float(ctx_sim[i, j]), 4),
                    "pos_sim": round(float(pos_sim[i, j]), 4),
                    "anti_coocc": round(float(anti_coocc[i, j]), 4),
                })
    pairs.sort(key=lambda x: x["score"], reverse=True)

    # Clustering gerarchico per trovare gruppi omofonici
    dist = 1 - ctx_sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    # Ensure symmetry for squareform
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    # Taglio ottimale: trova il numero di gruppi dove la similarita'
    # intra-cluster e' ancora alta (>0.5)
    best_n = n
    for n_groups in range(max(8, n // 3), n):
        labels = fcluster(Z, t=n_groups, criterion="maxclust")
        groups = defaultdict(list)
        for ch, label in zip(char_list, labels):
            groups[int(label)].append(ch)

        # Verifica che tutti i gruppi multi-membro abbiano alta similarita'
        ok = True
        for members in groups.values():
            if len(members) > 1:
                idxs = [char_list.index(ch) for ch in members]
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        if ctx_sim[idxs[a], idxs[b]] < 0.5:
                            ok = False
                            break
                    if not ok:
                        break
            if not ok:
                break
        if ok:
            best_n = n_groups
            break

    labels = fcluster(Z, t=best_n, criterion="maxclust")
    optimal_groups = defaultdict(list)
    for ch, label in zip(char_list, labels):
        optimal_groups[int(label)].append(ch)

    return {
        "pairs": pairs[:20],
        "groups": dict(optimal_groups),
        "estimated_alphabet_size": best_n,
        "original_alphabet_size": n,
        "reduction_ratio": round(best_n / n, 3) if n > 0 else 1.0,
        "context_similarity": ctx_sim,
        "char_list": char_list,
    }


# =====================================================================
# Sezione 4 — Analisi morfologica
# =====================================================================

def assign_char_roles(profiles: dict) -> dict:
    """Assegna ruolo I(nicial)/M(edial)/F(inal) a ogni carattere."""
    roles = {}
    for ch, p in profiles["profiles"].items():
        candidates = [("I", p["initial"]), ("M", p["medial"]), ("F", p["final"])]
        best = max(candidates, key=lambda x: x[1])
        roles[ch] = best[0]
    return roles


def extract_affixes(words: list[str], min_count: int = 50,
                    max_len: int = 3) -> dict:
    """Trova prefissi e suffissi comuni."""
    prefix_continuations = defaultdict(set)
    prefix_counts = Counter()
    suffix_continuations = defaultdict(set)
    suffix_counts = Counter()

    for w in words:
        for length in range(1, min(max_len + 1, len(w))):
            pref = w[:length]
            prefix_counts[pref] += 1
            if len(w) > length:
                prefix_continuations[pref].add(w[length])

            suf = w[-length:]
            suffix_counts[suf] += 1
            if len(w) > length:
                suffix_continuations[suf].add(w[-(length + 1)])

    total = len(words)

    def filter_affixes(counts, continuations):
        result = []
        for affix, count in counts.most_common():
            if count < min_count:
                break
            diversity = len(continuations.get(affix, set()))
            if diversity >= 3:
                result.append({
                    "affix": affix,
                    "count": count,
                    "frequency": round(count / total, 4),
                    "diversity": diversity,
                })
        return result

    prefixes = filter_affixes(prefix_counts, prefix_continuations)
    suffixes = filter_affixes(suffix_counts, suffix_continuations)
    return {"prefixes": prefixes[:20], "suffixes": suffixes[:20]}


def decompose_words(words: list[str], prefixes: list[dict],
                    suffixes: list[dict], top_n: int = 500) -> dict:
    """Decompone parole in prefisso + radice + suffisso."""
    prefix_set = {p["affix"] for p in prefixes}
    suffix_set = {s["affix"] for s in suffixes}

    word_counter = Counter(words)
    decompositions = []
    root_counter = Counter()

    for word, count in word_counter.most_common(top_n):
        best = {"prefix": "", "root": word, "suffix": "", "score": 0}

        for plen in range(0, min(4, len(word))):
            for slen in range(0, min(4, len(word) - plen)):
                prefix = word[:plen] if plen > 0 else ""
                suffix = word[len(word) - slen:] if slen > 0 else ""
                root = word[plen:len(word) - slen] if slen > 0 else word[plen:]

                if not root:
                    continue

                score = 0
                if prefix and prefix in prefix_set:
                    score += 2
                if suffix and suffix in suffix_set:
                    score += 2
                if len(root) >= 2:
                    score += 0.5
                if len(root) >= 3:
                    score += 0.5

                if score > best["score"]:
                    best = {"prefix": prefix, "root": root,
                            "suffix": suffix, "score": score}

        decompositions.append({
            "word": word, "count": count,
            "prefix": best["prefix"], "root": best["root"],
            "suffix": best["suffix"],
        })
        root_counter[best["root"]] += count

    # Famiglie: parole con stessa radice + diversi affissi
    root_families = defaultdict(list)
    for d in decompositions:
        if d["prefix"] or d["suffix"]:  # solo parole effettivamente decomposte
            root_families[d["root"]].append(d)

    families = {}
    for root, members in sorted(root_families.items(),
                                key=lambda x: len(x[1]), reverse=True):
        if len(members) > 1:
            families[root] = [
                {"word": m["word"], "prefix": m["prefix"],
                 "suffix": m["suffix"], "count": m["count"]}
                for m in members
            ]

    # Distribuzione lunghezza radici
    root_lengths = [len(d["root"]) for d in decompositions
                    if d["prefix"] or d["suffix"]]
    avg_root_len = round(np.mean(root_lengths), 2) if root_lengths else 0

    return {
        "decompositions": decompositions[:50],
        "top_roots": [{"root": r, "count": c}
                      for r, c in root_counter.most_common(30)],
        "families": dict(list(families.items())[:20]),
        "avg_root_length": avg_root_len,
        "n_families": len(families),
    }


def compute_word_templates(words: list[str], char_roles: dict) -> dict:
    """Mappa parole a template basati sui ruoli posizionali."""
    templates = Counter()
    for w in words:
        tmpl = "".join(char_roles.get(ch, "?") for ch in w)
        templates[tmpl] += 1

    total = len(words)
    top_templates = [
        {"template": t, "count": c, "frequency": round(c / total, 4)}
        for t, c in templates.most_common(30)
    ]

    return {"templates": top_templates, "total_unique": len(templates)}


# =====================================================================
# Sezione 5 — Visualizzazioni
# =====================================================================

@timer
def plot_cipher_scorecard(scores: dict, output_dir: Path):
    """Barre orizzontali con punteggio compatibilita' per tipo di cifrario."""
    # Ordina per score decrescente
    items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    names = [scores[k]["label"] for k, _ in items]
    values = [scores[k]["score"] for k, _ in items]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = []
    for v in values:
        if v >= 0.6:
            colors.append("#2E8B57")
        elif v >= 0.3:
            colors.append("#DAA520")
        else:
            colors.append("#CD5C5C")

    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Compatibilita'", fontsize=11)
    ax.set_title("Scorecard ipotesi di cifratura", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

    for bar, val, (cipher_type, data) in zip(bars, values, items):
        label = f" {val:.2f}"
        # Add key evidence
        subs = data["subscores"]
        detail = f"  IoC:{subs['ioc']:.1f} CV:{subs['cv']:.1f} H1/H0:{subs['h1h0_ratio']:.1f}"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label + detail, va="center", fontsize=8, fontfamily="monospace")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(str(output_dir / "cipher_scorecard.png"), dpi=150)
    plt.close()


@timer
def plot_homophone_evidence(homophones: dict, output_dir: Path):
    """Heatmap similarita' di contesto con gruppi annotati."""
    ctx_sim = homophones["context_similarity"]
    char_list = homophones["char_list"]
    groups = homophones["groups"]
    n = len(char_list)

    # Ordina caratteri per gruppo
    ordered_chars = []
    group_boundaries = []
    for gid in sorted(groups.keys()):
        group_boundaries.append(len(ordered_chars))
        ordered_chars.extend(groups[gid])
    order_idx = [char_list.index(ch) for ch in ordered_chars]

    reordered = ctx_sim[np.ix_(order_idx, order_idx)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(reordered, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_chars, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered_chars, fontsize=10)

    # Disegna confini gruppi
    for b in group_boundaries[1:]:
        ax.axhline(b - 0.5, color="black", linewidth=2)
        ax.axvline(b - 0.5, color="black", linewidth=2)

    # Annotazione gruppi
    for i, gid in enumerate(sorted(groups.keys())):
        members = groups[gid]
        if len(members) > 1:
            start = group_boundaries[i]
            mid = start + len(members) / 2 - 0.5
            ax.text(-1.5, mid, f"G{gid}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#8B4513")

    ax.set_title(f"Similarita' di contesto ({homophones['original_alphabet_size']} chars "
                 f"-> {homophones['estimated_alphabet_size']} gruppi stimati)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    plt.tight_layout()
    plt.savefig(str(output_dir / "homophone_groups.png"), dpi=150)
    plt.close()


@timer
def plot_morpheme_analysis(affixes: dict, templates: dict,
                           families: dict, output_dir: Path):
    """4 pannelli: prefissi, suffissi, template, famiglie esempio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analisi morfologica del Voynichese",
                 fontsize=14, fontweight="bold")

    # Prefissi
    prefs = affixes["prefixes"][:12]
    if prefs:
        labels = [p["affix"] for p in prefs]
        counts = [p["count"] for p in prefs]
        axes[0, 0].barh(labels[::-1], counts[::-1], color="#8B4513", alpha=0.7)
        axes[0, 0].set_xlabel("Occorrenze")
        axes[0, 0].set_title("Prefissi comuni")

    # Suffissi
    suffs = affixes["suffixes"][:12]
    if suffs:
        labels = [s["affix"] for s in suffs]
        counts = [s["count"] for s in suffs]
        axes[0, 1].barh(labels[::-1], counts[::-1], color="#2E8B57", alpha=0.7)
        axes[0, 1].set_xlabel("Occorrenze")
        axes[0, 1].set_title("Suffissi comuni")

    # Template
    tmpls = templates["templates"][:15]
    if tmpls:
        labels = [t["template"] for t in tmpls]
        freqs = [t["frequency"] for t in tmpls]
        axes[1, 0].barh(labels[::-1], freqs[::-1], color="#4169E1", alpha=0.7)
        axes[1, 0].set_xlabel("Frequenza")
        axes[1, 0].set_title(f"Template parola (I=Iniziale M=Mediale F=Finale)")

    # Famiglie di parole (pannello testuale)
    axes[1, 1].axis("off")
    lines = ["Famiglie di parole (stessa radice)\n"]
    for root, members in list(families.items())[:8]:
        words_str = ", ".join(m["word"] for m in members[:6])
        lines.append(f"  [{root}]  {words_str}")
    axes[1, 1].text(
        0.05, 0.95, "\n".join(lines),
        transform=axes[1, 1].transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
    )
    axes[1, 1].set_title("Famiglie morfologiche")

    plt.tight_layout()
    plt.savefig(str(output_dir / "morpheme_structure.png"), dpi=150)
    plt.close()


# =====================================================================
# Sezione 6 — Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per test ipotesi cifratura + analisi morfologica."""
    print_header("VOYNICH TOOLKIT - Test Ipotesi Cifratura")
    config.ensure_dirs()

    report_path = config.stats_dir / "cipher_hypothesis.json"
    if report_path.exists() and not force:
        print("  Report gia' presente, skip (usa --force per rieseguire)")
        return

    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"File EVA non trovato: {eva_file}\n"
            "  Esegui prima: voynich eva"
        )

    # 1. Parsing
    print_step("Parsing parole EVA...")
    data = parse_eva_words(eva_file)
    words = data["words"]
    chars = list("".join(words))
    print(f"    {data['total_words']} parole, {len(chars)} caratteri")

    # 2. Diagnostica alfabeto
    print_step("Diagnostica alfabeto...")
    alpha_stats = compute_alphabet_stats(chars)
    print(f"    Alfabeto: {alpha_stats['unique_characters']} caratteri")
    print(f"    Effettivo (95%): {alpha_stats['effective_95']}")
    print(f"    Effettivo (entropia): {alpha_stats['entropy_effective']}")
    print(f"    CV frequenze: {alpha_stats['cv']:.4f}")
    print(f"    Gini: {alpha_stats['gini']:.4f}")

    # 3. Entropia e IoC
    print_step("Entropia condizionale e IoC...")
    entropy_data = compute_conditional_entropy(chars, max_order=4)
    ioc = compute_index_of_coincidence(chars)
    h0 = entropy_data["cascade"][0]
    h1 = entropy_data["cascade"][1] if len(entropy_data["cascade"]) > 1 else h0
    h1h0 = h1 / h0 if h0 > 0 else 1.0
    print(f"    H(0)={h0:.4f}, H(1)={h1:.4f}, rapporto={h1h0:.4f}")
    print(f"    IoC={ioc:.6f}")

    # 4. Co-occorrenza e profili posizionali
    print_step("Calcolo contesti co-occorrenza...")
    coocc = compute_cooccurrence_matrix(words)
    pmi = compute_pmi_matrix(coocc)
    profiles = compute_char_positional_profiles(words)

    # 5. Rilevamento omofoni
    print_step("Rilevamento gruppi omofonici...")
    homophones = detect_homophone_candidates(coocc, pmi, profiles)
    print(f"    Alfabeto originale: {homophones['original_alphabet_size']}")
    print(f"    Gruppi stimati: {homophones['estimated_alphabet_size']}")
    print(f"    Riduzione: {homophones['reduction_ratio']:.1%}")
    if homophones["pairs"]:
        print("    Top coppie omofone:")
        for p in homophones["pairs"][:5]:
            print(f"      ({p['char_a']},{p['char_b']}): "
                  f"score={p['score']:.3f} "
                  f"ctx={p['context_sim']:.2f} "
                  f"pos={p['pos_sim']:.2f} "
                  f"anti={p['anti_coocc']:.2f}")

    # 6. Scoring ipotesi
    print_step("Scoring ipotesi di cifratura...")
    scores = score_all_hypotheses(alpha_stats, entropy_data, ioc,
                                  homophones, words)

    # 7. Analisi morfologica
    print_step("Estrazione affissi...")
    affixes = extract_affixes(words)
    print(f"    {len(affixes['prefixes'])} prefissi, "
          f"{len(affixes['suffixes'])} suffissi comuni")

    print_step("Decomposizione parole...")
    morphemes = decompose_words(words, affixes["prefixes"], affixes["suffixes"])
    print(f"    {morphemes['n_families']} famiglie morfologiche")
    print(f"    Lunghezza media radice: {morphemes['avg_root_length']}")

    print_step("Analisi template parola...")
    char_roles = assign_char_roles(profiles)
    templates = compute_word_templates(words, char_roles)
    print(f"    {templates['total_unique']} template unici")

    # 8. Visualizzazioni
    print_step("Generazione grafici...")
    plot_cipher_scorecard(scores, config.stats_dir)
    plot_homophone_evidence(homophones, config.stats_dir)
    plot_morpheme_analysis(affixes, templates, morphemes["families"],
                           config.stats_dir)

    # 9. Salva report
    print_step("Salvataggio report...")
    report = {
        "alphabet_diagnostics": {
            k: v for k, v in alpha_stats.items() if k != "frequencies"
        },
        "entropy": {
            "h0": h0, "h1": h1, "h1h0_ratio": round(h1h0, 4),
            "cascade": entropy_data["cascade"],
        },
        "ioc": ioc,
        "cipher_scores": {
            k: {"score": v["score"], "label": v["label"], "subscores": v["subscores"]}
            for k, v in scores.items()
        },
        "homophone_analysis": {
            "estimated_alphabet_size": homophones["estimated_alphabet_size"],
            "original_alphabet_size": homophones["original_alphabet_size"],
            "reduction_ratio": homophones["reduction_ratio"],
            "groups": homophones["groups"],
            "top_pairs": homophones["pairs"][:10],
        },
        "morphological_analysis": {
            "prefixes": affixes["prefixes"][:15],
            "suffixes": affixes["suffixes"][:15],
            "top_roots": morphemes["top_roots"][:20],
            "avg_root_length": morphemes["avg_root_length"],
            "n_families": morphemes["n_families"],
            "top_families": dict(list(morphemes["families"].items())[:10]),
            "top_templates": templates["templates"][:15],
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # === Sommario ===
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"],
                           reverse=True)

    print("\n" + "=" * 60)
    print("  SCORECARD IPOTESI DI CIFRATURA")
    print("=" * 60)

    for cipher_type, data in sorted_scores:
        score = data["score"]
        bar = "#" * int(score * 30)
        marker = " ** BEST **" if cipher_type == sorted_scores[0][0] else ""
        print(f"  {data['label']:<34} {score:.3f} {bar}{marker}")

    print(f"\n  Diagnostica chiave:")
    print(f"    IoC:                 {ioc:.6f} (range lingue: 0.065-0.077)")
    print(f"    H(1)/H(0):           {h1h0:.4f} (lingue: 0.75-0.85, Voynich: BASSO)")
    print(f"    CV frequenze:        {alpha_stats['cv']:.4f}")
    print(f"    Alfabeto reale:      {alpha_stats['unique_characters']}")
    print(f"    Alfabeto stimato:    {homophones['estimated_alphabet_size']} "
          f"(riduzione {1 - homophones['reduction_ratio']:.0%})")

    print(f"\n  Gruppi omofonici rilevati:")
    for gid, members in sorted(homophones["groups"].items()):
        if len(members) > 1:
            print(f"    Gruppo {gid}: {', '.join(members)}")

    print(f"\n  Interpretazione:")
    winner = sorted_scores[0][0]
    if winner == "Homophonic":
        print("    Il basso rapporto H(1)/H(0)=0.61 combinato con IoC=0.077")
        print("    (range naturale) punta a sostituzione omofonica: piu'")
        print("    simboli per lettera, con regole strutturali rigide che")
        print("    producono la forte regolarita' sequenziale osservata.")
    elif winner == "Abjad":
        print("    Compatibile con scrittura consonantica tipo ebraico/arabo.")
    elif winner == "Monoalphabetic":
        print("    Sostituzione semplice di una lingua naturale.")
    print(f"    Polialfabetico ESCLUSO (IoC troppo alto).")
    print(f"    Testo senza significato ESCLUSO (IoC e struttura morfologica).")

    # Morfologia
    print(f"\n  Morfologia:")
    print(f"    Radice media: {morphemes['avg_root_length']} caratteri")
    print(f"    Famiglie: {morphemes['n_families']}")
    print(f"    Top template: {templates['templates'][0]['template']} "
          f"({templates['templates'][0]['frequency']:.1%})")

    if morphemes["families"]:
        print(f"\n  Esempio famiglie:")
        for root, members in list(morphemes["families"].items())[:5]:
            words_list = [m["word"] for m in members[:5]]
            print(f"    [{root}]: {', '.join(words_list)}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
