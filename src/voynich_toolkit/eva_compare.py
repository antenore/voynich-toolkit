"""
Step 5: Confronto con la trascrizione EVA.

Scarica il dataset EVA (European Voynich Alphabet), lo parsifica,
e confronta le proprieta' statistiche con quelle estratte dalla nostra
analisi indipendente.
"""
import re
import csv
import json
from pathlib import Path
from collections import Counter

import click
import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer


_FALLBACK_EVA_URL = (
    "http://www.voynich.nu/data/EVA-Interlinear-Full.txt"
)


@timer
def download_eva(url: str, output_dir: Path) -> Path | None:
    """Scarica il file EVA se non gia' presente."""
    filename = url.split("/")[-1]
    filepath = output_dir / filename

    if filepath.exists():
        print(f"    EVA gia' scaricato: {filepath}")
        return filepath

    headers = {"User-Agent": "voynich-toolkit/0.1.0"}

    for attempt_url in [url, _FALLBACK_EVA_URL]:
        label = "URL alternativo" if attempt_url != url else attempt_url
        print(f"    Scaricamento da {label}...")
        try:
            response = requests.get(attempt_url, headers=headers, timeout=30)
            response.raise_for_status()
            filepath.write_text(response.text, encoding="utf-8")
            print(f"    Salvato: {filepath} ({len(response.text)} bytes)")
            return filepath
        except requests.RequestException as e:
            print(f"    Download fallito: {e}")

    print(f"    Scarica manualmente il file EVA e mettilo in {output_dir}/")
    return None


def parse_eva_file(filepath: Path, transcriber: str = "H") -> list[list[str]]:
    """
    Parsifica un file EVA in formato IVTFF (Interlinear Voynich Text File Format).

    Il formato IVTFF usa righe come:
      <f1r.1,@P0;H>  fachys.ykal.ar.ataiin.shol.shory...
    dove il tag tra <> identifica pagina, riga, e trascrittore (dopo ;).
    Il testo EVA segue dopo il tag.

    Righe header di pagina hanno il formato:
      <f1r>  <! ...>

    Args:
        filepath: Percorso al file IVTFF
        transcriber: Codice trascrittore preferito (H=Takahashi, C=Currier, etc.)

    Returns:
        Lista di "pagine", ciascuna e' una lista di caratteri EVA
    """
    if filepath is None:
        return []

    text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    all_chars = []
    current_page = []

    # Pattern per righe di trascrizione: <fNNx.line,@unit;TRANSCRIBER>  text...
    transcription_re = re.compile(
        r"^<(f\d+[rv]\d?)\.\d+[^;]*;(\w)>\s+(.+)"
    )
    # Pattern per header di pagina: <fNNx>
    page_header_re = re.compile(r"^<(f\d+[rv]\d?)>\s")

    for line in lines:
        line = line.rstrip()

        # Skip commenti
        if not line or line.startswith("#"):
            continue

        # Header di pagina -> nuova pagina
        if page_header_re.match(line):
            if current_page:
                all_chars.append(current_page)
                current_page = []
            continue

        # Riga di trascrizione
        m = transcription_re.match(line)
        if not m:
            continue

        line_transcriber = m.group(2)
        if line_transcriber != transcriber:
            continue

        eva_text = m.group(3)

        # Rimuovi annotazioni inline tra {} e <>
        clean = re.sub(r"\{[^}]*\}", "", eva_text)
        clean = re.sub(r"<[^>]*>", "", clean)
        # Rimuovi marcatori speciali
        clean = re.sub(r"[%!?\[\]*,]", "", clean)

        # Estrai i caratteri EVA (lettere minuscole)
        for char in clean:
            if char.isalpha() and char.islower():
                current_page.append(char)

    if current_page:
        all_chars.append(current_page)

    return all_chars


@timer
def eva_statistics(pages: list[list[str]], top_n: int = 30) -> dict:
    """Calcola le stesse statistiche dell'analisi dei glifi sul testo EVA."""

    # Sequenza piatta
    sequence = [char for page in pages for char in page]
    total = len(sequence)

    if total == 0:
        return {"error": "Nessun carattere EVA trovato"}

    # Frequenze
    counter = Counter(sequence)
    freq_list = counter.most_common()
    frequencies = [
        (char, count, count / total) for char, count in freq_list
    ]

    # Entropia di Shannon
    probs = np.array([f[2] for f in frequencies])
    entropy = -np.sum(probs * np.log2(probs))

    # Zipf
    ranks = np.arange(1, len(frequencies) + 1)
    counts = np.array([f[1] for f in frequencies])
    log_ranks = np.log10(ranks)
    log_counts = np.log10(counts)
    mask = np.isfinite(log_counts)
    zipf_slope = None
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_ranks[mask], log_counts[mask], 1)
        zipf_slope = float(coeffs[0])

    # N-grammi (su caratteri)
    ngrams_results = {}
    for n in range(2, 5):
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngrams.append(tuple(sequence[i:i + n]))
        counter_ng = Counter(ngrams)
        top = counter_ng.most_common(top_n)
        ngrams_results[f"{n}-gram"] = [
            {"ngram": "".join(ng), "count": c} for ng, c in top
        ]

    return {
        "total_characters": total,
        "unique_characters": len(freq_list),
        "alphabet": [f[0] for f in frequencies],
        "frequencies": frequencies,
        "shannon_entropy": round(entropy, 3),
        "zipf_slope": zipf_slope,
        "ngrams": ngrams_results,
        "pages_count": len(pages),
        "avg_chars_per_page": round(total / len(pages), 1) if pages else 0
    }


@timer
def compare_statistics(our_stats_path: Path, eva_stats: dict,
                        output_dir: Path):
    """Confronta le nostre statistiche con quelle EVA e genera un report."""
    our_stats = {}
    if our_stats_path.exists():
        with open(our_stats_path) as f:
            our_stats = json.load(f)

    comparison = {
        "metric": [],
        "our_analysis": [],
        "eva_reference": [],
        "match": []
    }

    def add_comparison(metric, ours, eva):
        comparison["metric"].append(metric)
        comparison["our_analysis"].append(str(ours))
        comparison["eva_reference"].append(str(eva))
        if isinstance(ours, (int, float)) and isinstance(eva, (int, float)):
            diff = abs(ours - eva) / max(abs(eva), 1) * 100
            comparison["match"].append(f"{100 - min(diff, 100):.0f}%")
        else:
            comparison["match"].append("-")

    add_comparison(
        "Entropia Shannon (bit/simbolo)",
        our_stats.get("shannon_entropy_bits", "N/A"),
        eva_stats.get("shannon_entropy", "N/A")
    )
    add_comparison(
        "Zipf slope",
        our_stats.get("zipf_slope", "N/A"),
        eva_stats.get("zipf_slope", "N/A")
    )
    add_comparison(
        "Tipi unici",
        our_stats.get("unique_glyph_types", "N/A"),
        eva_stats.get("unique_characters", "N/A")
    )
    add_comparison(
        "Simboli totali",
        our_stats.get("total_glyphs", "N/A"),
        eva_stats.get("total_characters", "N/A")
    )

    # Salva come CSV
    csv_path = output_dir / "comparison_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrica", "Nostra Analisi", "EVA Reference", "Match"])
        for i in range(len(comparison["metric"])):
            writer.writerow([
                comparison["metric"][i],
                comparison["our_analysis"][i],
                comparison["eva_reference"][i],
                comparison["match"][i]
            ])

    return comparison


@timer
def plot_eva_analysis(eva_stats: dict, output_dir: Path):
    """Genera grafici dell'analisi EVA."""
    freqs = eva_stats.get("frequencies", [])
    if not freqs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analisi della Trascrizione EVA del Manoscritto Voynich",
                 fontsize=14, fontweight="bold")

    # 1. Frequenze caratteri
    chars = [f[0] for f in freqs]
    counts = [f[1] for f in freqs]
    axes[0, 0].bar(chars, counts, color="#8B4513", alpha=0.7)
    axes[0, 0].set_title("Frequenza caratteri EVA")
    axes[0, 0].set_xlabel("Carattere")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].tick_params(axis="x", rotation=0, labelsize=8)

    # 2. Zipf plot
    ranks = np.arange(1, len(freqs) + 1)
    axes[0, 1].scatter(np.log10(ranks), np.log10(counts),
                        s=30, color="#8B4513", alpha=0.7)
    if eva_stats.get("zipf_slope"):
        log_r = np.log10(ranks)
        coeffs = np.polyfit(log_r, np.log10(counts), 1)
        axes[0, 1].plot(log_r, np.poly1d(coeffs)(log_r), "r--",
                         label=f"slope = {coeffs[0]:.2f}")
        axes[0, 1].legend()
    axes[0, 1].set_title("Analisi di Zipf - EVA")
    axes[0, 1].set_xlabel("log10(Rank)")
    axes[0, 1].set_ylabel("log10(Frequenza)")

    # 3. Top bigrammi
    bigrams = eva_stats.get("ngrams", {}).get("2-gram", [])[:15]
    if bigrams:
        bg_labels = [b["ngram"] for b in bigrams]
        bg_counts = [b["count"] for b in bigrams]
        axes[1, 0].barh(bg_labels[::-1], bg_counts[::-1],
                         color="#8B4513", alpha=0.7)
        axes[1, 0].set_title("Top 15 bigrammi EVA")
        axes[1, 0].set_xlabel("Frequenza")

    # 4. Top trigrammi
    trigrams = eva_stats.get("ngrams", {}).get("3-gram", [])[:15]
    if trigrams:
        tg_labels = [t["ngram"] for t in trigrams]
        tg_counts = [t["count"] for t in trigrams]
        axes[1, 1].barh(tg_labels[::-1], tg_counts[::-1],
                         color="#8B4513", alpha=0.7)
        axes[1, 1].set_title("Top 15 trigrammi EVA")
        axes[1, 1].set_xlabel("Frequenza")

    plt.tight_layout()
    plt.savefig(str(output_dir / "eva_analysis.png"), dpi=150)
    plt.close()


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per lo step di confronto EVA."""
    print_header("VOYNICH TOOLKIT - Step 5: Confronto EVA")
    config.ensure_dirs()

    report_path = config.eva_dir / "eva_report.json"
    if report_path.exists() and not force:
        print(f"  Report EVA gia' presente, skip (usa --force per rieseguire)")
        return

    # 1. Download EVA
    print_step("Download trascrizione EVA...")
    eva_file = download_eva(config.eva_url, config.eva_data_dir)

    if eva_file is None:
        raise click.ClickException(
            "Impossibile scaricare il file EVA.\n"
            "  Scarica manualmente e mettilo in eva_data/"
        )

    # 2. Parsing
    print_step("Parsing file EVA...")
    pages = parse_eva_file(eva_file)
    total_chars = sum(len(p) for p in pages)
    print(f"    {len(pages)} pagine, {total_chars} caratteri totali")

    # 3. Statistiche EVA
    print_step("Calcolo statistiche EVA...")
    eva_stats = eva_statistics(pages, top_n=config.top_n_ngrams)

    # 4. Grafici EVA
    print_step("Generazione grafici EVA...")
    plot_eva_analysis(eva_stats, config.eva_dir)

    # 5. Confronto con le nostre statistiche
    print_step("Confronto con la nostra analisi...")
    our_report_path = config.stats_dir / "analysis_report.json"
    comparison = compare_statistics(our_report_path, eva_stats, config.eva_dir)

    # 6. Salva report EVA
    eva_report = {
        "total_characters": eva_stats["total_characters"],
        "unique_characters": eva_stats["unique_characters"],
        "alphabet": eva_stats["alphabet"],
        "shannon_entropy": eva_stats["shannon_entropy"],
        "zipf_slope": eva_stats["zipf_slope"],
        "top_10_characters": [
            {"char": f[0], "count": f[1], "frequency": round(f[2], 4)}
            for f in eva_stats["frequencies"][:10]
        ],
        "top_bigrams": eva_stats["ngrams"].get("2-gram", [])[:10],
        "top_trigrams": eva_stats["ngrams"].get("3-gram", [])[:10],
        "pages_count": eva_stats["pages_count"],
        "avg_chars_per_page": eva_stats["avg_chars_per_page"]
    }

    with open(report_path, "w") as f:
        json.dump(eva_report, f, indent=2)

    # Stampa sommario
    print("\n" + "=" * 60)
    print("  REPORT EVA")
    print("=" * 60)
    print(f"  Caratteri totali:     {eva_stats['total_characters']}")
    print(f"  Caratteri unici:      {eva_stats['unique_characters']}")
    print(f"  Alfabeto EVA:         {''.join(eva_stats['alphabet'])}")
    print(f"  Shannon entropy:      {eva_stats['shannon_entropy']} bit/simbolo")
    print(f"  Zipf slope:           {eva_stats['zipf_slope']}")

    print(f"\n  Top 10 caratteri EVA:")
    for f in eva_stats["frequencies"][:10]:
        bar = "#" * int(f[2] * 200)
        print(f"    '{f[0]}': {f[1]:6d} ({f[2]:.1%}) {bar}")

    print(f"\n  CONFRONTO")
    print(f"  {'Metrica':<30} {'Nostra':>12} {'EVA':>12} {'Match':>8}")
    print(f"  {'-' * 62}")
    for i in range(len(comparison["metric"])):
        print(f"  {comparison['metric'][i]:<30} "
              f"{comparison['our_analysis'][i]:>12} "
              f"{comparison['eva_reference'][i]:>12} "
              f"{comparison['match'][i]:>8}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.eva_dir}/")
