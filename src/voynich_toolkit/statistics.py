"""
Step 4: Analisi statistica dei glifi estratti.

Calcola:
- Frequenze dei glifi (basate su clustering visuale)
- Distribuzione di Zipf
- Entropia di Shannon
- N-grammi (bigrammi, trigrammi)
- Statistiche dimensionali
"""
import csv
import json
from pathlib import Path
from collections import Counter

import click
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from .config import ToolkitConfig
from .utils import print_header, print_step, timer


def load_glyph_catalog(catalog_path: Path) -> list[dict]:
    """Carica il catalogo dei glifi."""
    with open(catalog_path) as f:
        return json.load(f)


def load_normalized_glyphs(
    catalog: list[dict],
    glyphs_dir: Path,
) -> list[tuple[dict, np.ndarray]]:
    """Carica le immagini normalizzate dei glifi."""
    loaded = []
    for entry in tqdm(catalog, desc="  Caricamento glifi", unit="glifo"):
        norm_path = glyphs_dir / entry.get("normalized_file", "")
        if norm_path.exists():
            img = cv2.imread(str(norm_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                loaded.append((entry, img))
    return loaded


@timer
def cluster_glyphs(
    glyphs: list[tuple[dict, np.ndarray]],
    distance_threshold: float = 0.6,
    max_cluster_glyphs: int = 5000,
) -> dict:
    """
    Raggruppa i glifi visualmente simili usando clustering gerarchico.

    Returns:
        Dict con cluster_id -> lista di glyph entries
    """
    if len(glyphs) < 2:
        return {0: [g[0] for g in glyphs]}

    # Converti i glifi in vettori piatti
    vectors = []
    for _, img in glyphs:
        flat = img.flatten().astype(np.float32) / 255.0
        vectors.append(flat)

    vectors = np.array(vectors)
    print(f"    Clustering {len(vectors)} glifi ({vectors.shape[1]} features)...")

    # Limita il numero per praticita' computazionale
    if len(vectors) > max_cluster_glyphs:
        print(f"    Limitando a {max_cluster_glyphs} glifi per il clustering")
        indices = np.random.choice(len(vectors), max_cluster_glyphs, replace=False)
        vectors = vectors[indices]
        glyphs = [glyphs[i] for i in indices]

    # Clustering gerarchico
    distances = pdist(vectors, metric="cosine")
    Z = linkage(distances, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Organizza per cluster
    clusters = {}
    for label, (entry, _) in zip(labels, glyphs):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(entry)

    return clusters


@timer
def compute_frequencies(clusters: dict) -> list[tuple[str, int, float]]:
    """
    Calcola le frequenze di ciascun tipo di glifo (cluster).

    Returns:
        Lista di (cluster_label, count, frequency)
    """
    total = sum(len(members) for members in clusters.values())
    freqs = []

    for cluster_id, members in sorted(clusters.items(),
                                        key=lambda x: len(x[1]),
                                        reverse=True):
        count = len(members)
        freq = count / total if total > 0 else 0
        label = f"G{cluster_id:03d}"
        freqs.append((label, count, freq))

    return freqs


@timer
def zipf_analysis(frequencies: list[tuple[str, int, float]],
                   output_dir: Path):
    """
    Verifica se la distribuzione segue la legge di Zipf.
    In un linguaggio naturale, freq(r) ~ C / r^alpha dove r e' il rank.
    """
    ranks = np.arange(1, len(frequencies) + 1)
    counts = np.array([f[1] for f in frequencies])

    # Plot log-log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Distribuzione di frequenza
    ax1.bar(range(min(50, len(frequencies))),
            [f[1] for f in frequencies[:50]],
            color="#8B4513", alpha=0.7)
    ax1.set_xlabel("Tipo di glifo (rank)")
    ax1.set_ylabel("Frequenza")
    ax1.set_title("Distribuzione di frequenza dei glifi (top 50)")

    # Plot Zipf (log-log)
    ax2.scatter(np.log10(ranks), np.log10(counts),
                s=15, alpha=0.6, color="#8B4513")

    # Fit lineare in log-log
    log_ranks = np.log10(ranks)
    log_counts = np.log10(counts)
    mask = np.isfinite(log_counts)

    zipf_slope = None
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_ranks[mask], log_counts[mask], 1)
        fit_line = np.poly1d(coeffs)
        ax2.plot(log_ranks, fit_line(log_ranks), "r--", linewidth=2,
                 label=f"Fit: slope = {coeffs[0]:.2f}")
        ax2.legend()
        zipf_slope = coeffs[0]

    ax2.set_xlabel("log10(Rank)")
    ax2.set_ylabel("log10(Frequenza)")
    ax2.set_title("Analisi di Zipf (log-log)")

    plt.tight_layout()
    plot_path = output_dir / "zipf_analysis.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()

    return zipf_slope


@timer
def shannon_entropy(frequencies: list[tuple[str, int, float]]) -> float:
    """
    Calcola l'entropia di Shannon della distribuzione dei glifi.

    Per confronto:
    - Inglese: ~4.0-4.5 bit/carattere
    - Voynichese (da EVA): ~4.0-4.5 bit/carattere (simile!)
    - Random: ~log2(N) bit/carattere
    """
    probs = np.array([f[2] for f in frequencies])
    probs = probs[probs > 0]  # Rimuovi zeri

    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def build_sequence_from_clusters(catalog: list[dict],
                                  clusters: dict) -> list[str]:
    """
    Costruisce una sequenza di simboli basata sui cluster assegnati.
    """
    # Mappa glyph -> cluster label
    glyph_to_label = {}
    for cluster_id, members in clusters.items():
        label = f"G{cluster_id:03d}"
        for member in members:
            key = (member["region"], member["glyph_id"])
            glyph_to_label[key] = label

    # Ordina per posizione
    sorted_catalog = sorted(catalog, key=lambda g: (
        g["region"], g.get("line", 0), g["x"]
    ))

    sequence = []
    for entry in sorted_catalog:
        key = (entry["region"], entry["glyph_id"])
        label = glyph_to_label.get(key)
        if label:
            sequence.append(label)

    return sequence


@timer
def ngram_analysis(sequence: list[str], output_dir: Path,
                    top_n: int = 30, max_n: int = 4) -> dict:
    """Calcola n-grammi (bigrammi, trigrammi, etc.) dalla sequenza di glifi."""
    results = {}

    for n in range(2, max_n + 1):
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngrams.append(ngram)

        counter = Counter(ngrams)
        top = counter.most_common(top_n)

        results[f"{n}-gram"] = [
            {"ngram": " ".join(ng), "count": count}
            for ng, count in top
        ]

        # Salva come CSV
        csv_path = output_dir / f"{n}gram_top{top_n}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ngram", "count"])
            for ng, count in top:
                writer.writerow([" ".join(ng), count])

    return results


@timer
def dimensional_stats(catalog: list[dict], output_dir: Path):
    """Statistiche sulle dimensioni fisiche dei glifi."""
    if not catalog:
        return

    widths = [g["w"] for g in catalog]
    heights = [g["h"] for g in catalog]
    areas = [g["area"] for g in catalog]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(widths, bins=30, color="#8B4513", alpha=0.7)
    axes[0].set_title("Distribuzione larghezze glifi")
    axes[0].set_xlabel("Larghezza (px)")

    axes[1].hist(heights, bins=30, color="#8B4513", alpha=0.7)
    axes[1].set_title("Distribuzione altezze glifi")
    axes[1].set_xlabel("Altezza (px)")

    axes[2].scatter(widths, heights, s=5, alpha=0.3, color="#8B4513")
    axes[2].set_title("Larghezza vs Altezza")
    axes[2].set_xlabel("Larghezza (px)")
    axes[2].set_ylabel("Altezza (px)")

    plt.tight_layout()
    plt.savefig(str(output_dir / "glyph_dimensions.png"), dpi=150)
    plt.close()

    stats = {
        "total_glyphs": len(catalog),
        "width": {"mean": np.mean(widths), "std": np.std(widths),
                   "min": min(widths), "max": max(widths)},
        "height": {"mean": np.mean(heights), "std": np.std(heights),
                    "min": min(heights), "max": max(heights)},
        "area": {"mean": np.mean(areas), "std": np.std(areas),
                  "min": min(areas), "max": max(areas)}
    }
    return stats


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per lo step di analisi statistica."""
    print_header("VOYNICH TOOLKIT - Step 4: Analisi Statistica")
    config.ensure_dirs()

    catalog_path = config.glyphs_dir / "glyphs_catalog.json"
    if not catalog_path.exists():
        raise click.ClickException(
            "Catalogo glifi non trovato! Esegui prima: voynich segment-glyphs"
        )

    report_path = config.stats_dir / "analysis_report.json"
    if report_path.exists() and not force:
        print(f"  Report analisi gia' presente, skip (usa --force per rieseguire)")
        return

    catalog = load_glyph_catalog(catalog_path)
    print(f"  Catalogo: {len(catalog)} glifi")

    # 1. Carica glifi normalizzati
    print_step("Caricamento glifi normalizzati...")
    glyphs = load_normalized_glyphs(catalog, config.glyphs_dir)
    print(f"    {len(glyphs)} glifi caricati")

    # 2. Clustering
    print_step("Clustering visuale dei glifi...")
    clusters = cluster_glyphs(
        glyphs,
        distance_threshold=config.cluster_distance_threshold,
        max_cluster_glyphs=config.max_cluster_glyphs,
    )
    n_clusters = len(clusters)
    print(f"    {n_clusters} tipi di glifi distinti identificati")

    # 3. Frequenze
    print_step("Calcolo frequenze...")
    frequencies = compute_frequencies(clusters)

    # 4. Analisi di Zipf
    print_step("Analisi di Zipf...")
    zipf_slope = zipf_analysis(frequencies, config.stats_dir)

    # 5. Entropia di Shannon
    print_step("Entropia di Shannon...")
    entropy = shannon_entropy(frequencies)

    # 6. N-grammi
    print_step("Costruzione sequenza e analisi n-grammi...")
    sequence = build_sequence_from_clusters(catalog, clusters)
    ngrams = ngram_analysis(sequence, config.stats_dir, top_n=config.top_n_ngrams)

    # 7. Statistiche dimensionali
    print_step("Statistiche dimensionali...")
    dim_stats = dimensional_stats(catalog, config.stats_dir)

    # === REPORT FINALE ===
    report = {
        "total_glyphs": len(catalog),
        "unique_glyph_types": n_clusters,
        "zipf_slope": zipf_slope,
        "zipf_interpretation": (
            "Simile a linguaggio naturale (slope ~ -1.0)"
            if zipf_slope and -1.3 < zipf_slope < -0.7
            else "Deviazione dalla legge di Zipf"
            if zipf_slope
            else "Non calcolabile"
        ),
        "shannon_entropy_bits": round(entropy, 3),
        "entropy_interpretation": (
            f"Entropia = {entropy:.3f} bit/simbolo. "
            f"Per confronto: Inglese ~ 4.0-4.5, "
            f"Voynichese (EVA) ~ 4.0-4.5, Random ~ {np.log2(max(n_clusters, 1)):.1f}"
        ),
        "top_10_glyphs": [
            {"label": f[0], "count": f[1], "frequency": round(f[2], 4)}
            for f in frequencies[:10]
        ],
        "top_bigrams": ngrams.get("2-gram", [])[:10],
        "top_trigrams": ngrams.get("3-gram", [])[:10],
        "dimensional_stats": dim_stats,
        "sequence_length": len(sequence)
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Stampa sommario
    print("\n" + "=" * 60)
    print("  REPORT ANALISI STATISTICA")
    print("=" * 60)
    print(f"  Glifi totali:         {report['total_glyphs']}")
    print(f"  Tipi unici (cluster): {report['unique_glyph_types']}")
    print(f"  Zipf slope:           {report['zipf_slope']}")
    print(f"  -> {report['zipf_interpretation']}")
    print(f"  Shannon entropy:      {report['shannon_entropy_bits']} bit/simbolo")
    print(f"  -> {report['entropy_interpretation']}")
    print(f"\n  Top 5 glifi:")
    for g in report["top_10_glyphs"][:5]:
        bar = "#" * int(g["frequency"] * 100)
        print(f"    {g['label']}: {g['count']:5d} ({g['frequency']:.1%}) {bar}")
    print(f"\n  Report completo: {report_path}")
