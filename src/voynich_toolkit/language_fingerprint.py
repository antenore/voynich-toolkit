"""
Fingerprinting linguistico del Voynichese.

Confronta proprieta' statistiche (entropia condizionale, IoC, crescita
vocabolario) con lingue note per restringere le ipotesi sulla natura
del sistema scrittorio.
"""
import json
import math
from pathlib import Path
from collections import Counter

import click
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer
from .word_structure import parse_eva_words


# Valori di riferimento per lingue note
REFERENCE_LANGUAGES = {
    "Latin":   {"ioc": 0.0725, "h0": 3.90, "h1": 3.20, "h2": 2.70, "avg_word_len": 5.5},
    "Italian": {"ioc": 0.0738, "h0": 3.88, "h1": 3.10, "h2": 2.50, "avg_word_len": 5.1},
    "English": {"ioc": 0.0667, "h0": 4.00, "h1": 3.30, "h2": 2.80, "avg_word_len": 4.7},
    "German":  {"ioc": 0.0762, "h0": 3.85, "h1": 3.05, "h2": 2.45, "avg_word_len": 6.3},
    "Arabic":  {"ioc": 0.0758, "h0": 3.95, "h1": 3.15, "h2": 2.55, "avg_word_len": 4.9},
    "Hebrew":  {"ioc": 0.0768, "h0": 3.82, "h1": 3.10, "h2": 2.50, "avg_word_len": 4.5},
    "Random":  {"ioc": 0.0385, "h0": 4.70, "h1": 4.70, "h2": 4.70, "avg_word_len": None},
}

# Distribuzioni di frequenza rank-ordered per lingue note (top 10 per rank)
# Usate per KL divergence sulla *forma* della curva
REFERENCE_RANKED_FREQS = {
    "Latin":   [0.115, 0.102, 0.082, 0.081, 0.079, 0.072, 0.064, 0.063, 0.052, 0.037],
    "Italian": [0.118, 0.112, 0.104, 0.098, 0.069, 0.065, 0.064, 0.056, 0.050, 0.045],
    "English": [0.127, 0.091, 0.082, 0.075, 0.070, 0.067, 0.063, 0.061, 0.060, 0.043],
    "German":  [0.164, 0.098, 0.076, 0.073, 0.070, 0.065, 0.061, 0.051, 0.047, 0.043],
    "Arabic":  [0.126, 0.109, 0.072, 0.062, 0.059, 0.054, 0.051, 0.044, 0.036, 0.034],
    "Hebrew":  [0.107, 0.095, 0.080, 0.065, 0.055, 0.052, 0.050, 0.048, 0.040, 0.038],
}


def compute_conditional_entropy(chars: list[str], max_order: int = 4) -> dict:
    """Cascata di entropia condizionale H(0)..H(max_order).

    H_joint(n) = -sum(p(ngram) * log2(p(ngram))) su tutti gli n-grammi.
    H(0) = H_joint(1)  (entropia unigram)
    H(k) = H_joint(k+1) - H_joint(k)  per k >= 1
    """
    n = len(chars)
    if n == 0:
        return {"orders": [], "h_joint": []}

    h_joints = []

    for order in range(1, max_order + 2):  # need up to max_order+1 for diffs
        if n < order:
            break
        ngram_counts = Counter()
        for i in range(n - order + 1):
            ngram_counts[tuple(chars[i:i + order])] += 1
        total = sum(ngram_counts.values())
        h = 0.0
        for count in ngram_counts.values():
            p = count / total
            h -= p * math.log2(p)
        h_joints.append(round(h, 4))

    # H(0) = H_joint(1), H(k) = H_joint(k+1) - H_joint(k)
    cascade = [h_joints[0]]  # H(0)
    for k in range(1, len(h_joints)):
        cascade.append(round(h_joints[k] - h_joints[k - 1], 4))

    orders = list(range(len(cascade)))

    return {
        "orders": orders,
        "cascade": cascade,
        "h_joint": h_joints,
    }


def compute_index_of_coincidence(chars: list[str]) -> float:
    """IoC = sum(n_i*(n_i-1)) / (N*(N-1))."""
    n = len(chars)
    if n < 2:
        return 0.0
    counter = Counter(chars)
    numerator = sum(c * (c - 1) for c in counter.values())
    return round(numerator / (n * (n - 1)), 6)


def compute_word_level_stats(words: list[str]) -> dict:
    """Type-token ratio, hapax ratio, legge di Heaps, curva crescita vocabolario."""
    total = len(words)
    if total == 0:
        return {}

    counter = Counter(words)
    unique = len(counter)
    hapax = sum(1 for c in counter.values() if c == 1)

    ttr = round(unique / total, 4)
    hapax_ratio = round(hapax / unique, 4) if unique > 0 else 0.0

    # Curva crescita vocabolario campionata a 100 punti
    sample_points = np.linspace(1, total, min(100, total), dtype=int)
    vocab_sizes = []
    seen = set()
    word_idx = 0
    for target in sample_points:
        while word_idx < target:
            seen.add(words[word_idx])
            word_idx += 1
        vocab_sizes.append(len(seen))

    # Heaps law fit: log(V) = log(K) + beta * log(n)
    log_n = np.log10(sample_points.astype(float))
    log_v = np.log10(np.array(vocab_sizes, dtype=float))
    mask = np.isfinite(log_n) & np.isfinite(log_v)
    heaps_k = None
    heaps_beta = None
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_n[mask], log_v[mask], 1)
        heaps_beta = round(float(coeffs[0]), 4)
        heaps_k = round(10 ** float(coeffs[1]), 4)

    return {
        "type_token_ratio": ttr,
        "hapax_ratio": hapax_ratio,
        "hapax_count": hapax,
        "unique_words": unique,
        "total_words": total,
        "heaps_k": heaps_k,
        "heaps_beta": heaps_beta,
        "growth_curve": {
            "n": sample_points.tolist(),
            "vocab_size": vocab_sizes,
        },
    }


def compute_kl_divergence(voynich_freqs: list[float],
                          ref_freqs: list[float],
                          epsilon: float = 1e-10) -> float:
    """KL divergence su distribuzioni rank-ordered.

    Confronta la *forma* della curva, non le lettere specifiche.
    Padda a stessa lunghezza, normalizza, calcola KL(voynich || ref).
    """
    max_len = max(len(voynich_freqs), len(ref_freqs))
    p = np.array(voynich_freqs + [epsilon] * (max_len - len(voynich_freqs)))
    q = np.array(ref_freqs + [epsilon] * (max_len - len(ref_freqs)))

    # Normalizza
    p = p / p.sum()
    q = q / q.sum()

    # KL(P || Q)
    kl = float(np.sum(p * np.log2(p / q)))
    return round(kl, 6)


def compute_repeat_density(pages: list[dict]) -> dict:
    """Per ogni pagina: rapporto parole ripetute / totale. Overlap righe adiacenti."""
    page_densities = []
    line_overlaps_all = []

    for page in pages:
        words = page["words"]
        if not words:
            continue

        counter = Counter(words)
        repeated = sum(1 for w in words if counter[w] > 1)
        density = round(repeated / len(words), 4) if words else 0.0
        page_densities.append({
            "folio": page["folio"],
            "density": density,
            "total_words": len(words),
        })

        # Overlap tra righe adiacenti
        line_words = page.get("line_words", [])
        for i in range(len(line_words) - 1):
            set_a = set(line_words[i])
            set_b = set(line_words[i + 1])
            if set_a and set_b:
                overlap = len(set_a & set_b) / len(set_a | set_b)
                line_overlaps_all.append(round(overlap, 4))

    avg_density = round(np.mean([p["density"] for p in page_densities]), 4) if page_densities else 0.0
    avg_line_overlap = round(np.mean(line_overlaps_all), 4) if line_overlaps_all else 0.0

    return {
        "avg_repeat_density": avg_density,
        "avg_line_overlap": avg_line_overlap,
        "pages": page_densities[:20],  # sample
    }


# --- Visualizzazioni ---

@timer
def plot_entropy_cascade(entropy_data: dict, output_dir: Path):
    """Line plot H(k) per k=0..4: Voynichese vs lingue di riferimento."""
    orders = entropy_data["orders"]
    cascade = entropy_data["cascade"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Voynichese - linea spessa
    ax.plot(orders, cascade, "o-", color="#8B4513", linewidth=3,
            markersize=8, label="Voynichese", zorder=5)

    # Lingue di riferimento
    lang_colors = {
        "Latin": "#2E8B57", "Italian": "#4169E1", "English": "#DC143C",
        "German": "#FF8C00", "Arabic": "#9370DB", "Hebrew": "#20B2AA",
        "Random": "#888888",
    }

    for lang, ref in REFERENCE_LANGUAGES.items():
        ref_values = [ref.get(f"h{k}", None) for k in orders]
        ref_values = [v for v in ref_values if v is not None]
        ref_orders = list(range(len(ref_values)))

        style = "--" if lang == "Random" else "-"
        lw = 1.5 if lang == "Random" else 1.0
        alpha = 0.7 if lang == "Random" else 0.6
        ax.plot(ref_orders, ref_values, style, color=lang_colors[lang],
                linewidth=lw, alpha=alpha, label=lang)

    ax.set_xlabel("Ordine k")
    ax.set_ylabel("H(k) bit/simbolo")
    ax.set_title("Cascata di entropia condizionale")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(orders)
    plt.tight_layout()
    plt.savefig(str(output_dir / "entropy_cascade.png"), dpi=150)
    plt.close()


@timer
def plot_language_distances(kl_results: dict, ioc_voynich: float, output_dir: Path):
    """2 pannelli: barre KL divergence + dot plot IoC con linea Random."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Distanza linguistica del Voynichese", fontsize=14, fontweight="bold")

    # Panel 1: KL divergence bars
    languages = list(kl_results.keys())
    kl_values = [kl_results[lang] for lang in languages]
    colors = ["#2E8B57", "#4169E1", "#DC143C", "#FF8C00", "#9370DB", "#20B2AA"]

    bars = axes[0].barh(languages, kl_values, color=colors[:len(languages)], alpha=0.7)
    axes[0].set_xlabel("KL Divergence (bit)")
    axes[0].set_title("KL divergence dalla distribuzione Voynichese")
    for bar, val in zip(bars, kl_values):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    # Panel 2: IoC comparison
    lang_names = ["Voynichese"] + list(REFERENCE_LANGUAGES.keys())
    ioc_values = [ioc_voynich] + [REFERENCE_LANGUAGES[l]["ioc"] for l in REFERENCE_LANGUAGES]
    dot_colors = ["#8B4513"] + colors[:6] + ["#888888"]

    y_pos = range(len(lang_names))
    axes[1].scatter(ioc_values, y_pos, s=100, c=dot_colors[:len(lang_names)], zorder=3)
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(lang_names, fontsize=9)
    axes[1].axvline(REFERENCE_LANGUAGES["Random"]["ioc"], color="#888888",
                    linestyle="--", alpha=0.7, label="Random")
    axes[1].set_xlabel("Index of Coincidence")
    axes[1].set_title("Confronto IoC")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="x")

    for i, val in enumerate(ioc_values):
        axes[1].annotate(f"{val:.4f}", (val, i), textcoords="offset points",
                         xytext=(8, 0), fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_dir / "language_distances.png"), dpi=150)
    plt.close()


@timer
def plot_vocabulary_growth(word_stats: dict, output_dir: Path):
    """Curva crescita vocabolario + fit Heaps + annotazioni."""
    growth = word_stats.get("growth_curve", {})
    n_values = growth.get("n", [])
    v_values = growth.get("vocab_size", [])

    if not n_values:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_values, v_values, "-", color="#8B4513", linewidth=2, label="Vocabolario osservato")

    # Heaps fit
    k = word_stats.get("heaps_k")
    beta = word_stats.get("heaps_beta")
    if k is not None and beta is not None:
        n_arr = np.array(n_values, dtype=float)
        heaps_fit = k * n_arr ** beta
        ax.plot(n_values, heaps_fit, "--", color="#DC143C", linewidth=1.5,
                label=f"Heaps: V = {k:.1f} * n^{beta:.3f}")

    # Annotazioni
    ttr = word_stats.get("type_token_ratio", 0)
    hapax_r = word_stats.get("hapax_ratio", 0)
    annotation = (f"TTR: {ttr:.4f}\n"
                  f"Hapax ratio: {hapax_r:.4f}\n"
                  f"K = {k}\n"
                  f"beta = {beta}")
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Numero di token (n)")
    ax.set_ylabel("Tipi unici V(n)")
    ax.set_title("Curva crescita vocabolario (legge di Heaps)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "vocabulary_growth.png"), dpi=150)
    plt.close()


# --- Entry point ---

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per il fingerprinting linguistico."""
    print_header("VOYNICH TOOLKIT - Fingerprinting Linguistico")
    config.ensure_dirs()

    report_path = config.stats_dir / "language_fingerprint.json"
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

    # 2. Entropia condizionale
    print_step("Calcolo entropia condizionale H(0)..H(4)...")
    entropy_data = compute_conditional_entropy(chars, max_order=4)
    for k, h in zip(entropy_data["orders"], entropy_data["cascade"]):
        print(f"    H({k}) = {h:.4f}")

    # 3. Index of Coincidence
    print_step("Calcolo Index of Coincidence...")
    ioc = compute_index_of_coincidence(chars)
    print(f"    IoC = {ioc:.6f}")

    # 4. Statistiche a livello parola
    print_step("Statistiche a livello parola...")
    word_stats = compute_word_level_stats(words)
    print(f"    TTR: {word_stats['type_token_ratio']}")
    print(f"    Hapax ratio: {word_stats['hapax_ratio']}")
    print(f"    Heaps: K={word_stats['heaps_k']}, beta={word_stats['heaps_beta']}")

    # 5. KL divergence
    print_step("Calcolo KL divergence vs lingue di riferimento...")
    char_counter = Counter(chars)
    total_chars = len(chars)
    voynich_ranked = sorted([c / total_chars for c in char_counter.values()], reverse=True)

    kl_results = {}
    for lang, ref_freqs in REFERENCE_RANKED_FREQS.items():
        kl = compute_kl_divergence(voynich_ranked[:len(ref_freqs)], ref_freqs)
        kl_results[lang] = kl
        print(f"    KL(Voynich || {lang}) = {kl:.6f}")

    # 6. Repeat density
    print_step("Calcolo densita' ripetizioni...")
    repeat_data = compute_repeat_density(data["pages"])
    print(f"    Repeat density media: {repeat_data['avg_repeat_density']}")
    print(f"    Line overlap medio: {repeat_data['avg_line_overlap']}")

    # 7. Visualizzazioni
    print_step("Generazione grafici...")
    plot_entropy_cascade(entropy_data, config.stats_dir)
    plot_language_distances(kl_results, ioc, config.stats_dir)
    plot_vocabulary_growth(word_stats, config.stats_dir)

    # 8. Salva report
    print_step("Salvataggio report...")
    report = {
        "entropy_cascade": {
            "orders": entropy_data["orders"],
            "voynichese": entropy_data["cascade"],
            "references": {
                lang: [ref.get(f"h{k}", None) for k in entropy_data["orders"]]
                for lang, ref in REFERENCE_LANGUAGES.items()
            },
        },
        "index_of_coincidence": {
            "voynichese": ioc,
            "references": {lang: ref["ioc"] for lang, ref in REFERENCE_LANGUAGES.items()},
        },
        "kl_divergence": kl_results,
        "word_level_stats": {
            k: v for k, v in word_stats.items() if k != "growth_curve"
        },
        "repeat_density": repeat_data,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Sommario
    print("\n" + "=" * 60)
    print("  REPORT FINGERPRINTING LINGUISTICO")
    print("=" * 60)

    print(f"\n  Entropia condizionale:")
    print(f"    {'Ordine':<8}", end="")
    print(f"{'Voynich':>10}", end="")
    for lang in ["Latin", "English", "Random"]:
        print(f"{lang:>10}", end="")
    print()
    for k in entropy_data["orders"]:
        print(f"    H({k}){' ' * 4}", end="")
        print(f"{entropy_data['cascade'][k]:>10.4f}", end="")
        for lang in ["Latin", "English", "Random"]:
            ref_val = REFERENCE_LANGUAGES[lang].get(f"h{k}")
            print(f"{ref_val or 'N/A':>10}", end="")
        print()

    print(f"\n  IoC Voynichese: {ioc:.6f}")
    print(f"  IoC Random:     {REFERENCE_LANGUAGES['Random']['ioc']}")

    closest = min(kl_results, key=kl_results.get)
    print(f"\n  Lingua piu' vicina (KL): {closest} ({kl_results[closest]:.6f})")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
