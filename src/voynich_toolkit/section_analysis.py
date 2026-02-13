"""
Analisi per sezione del Manoscritto Voynich.

Usa i dati EVA (IVTFF) raggruppati per sezione ($I=), lingua Currier ($L=)
e mano ($H=). Calcola statistiche identiche per ogni gruppo e confronta.
"""
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import click
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer

# Nomi lunghi delle sezioni
SECTION_NAMES = {
    "H": "Herbal",
    "S": "Stars",
    "B": "Biological",
    "P": "Pharmaceutical",
    "C": "Cosmological",
    "Z": "Zodiac",
    "A": "Astronomical",
    "T": "Text",
}


def parse_eva_by_section(filepath: Path, transcriber: str = "H"):
    """Parsa il file IVTFF e raggruppa caratteri per sezione e lingua.

    Per ogni header <fNNx>  <! $I=X $Q=Y $L=Z $H=W>:
    - Estrai $I, $L, $H dal commento metadata
    - Accumula i caratteri delle righe di trascrizione successive

    Returns:
        sections: dict[str, list[str]]   # sezione -> tutti i caratteri
        languages: dict[str, list[str]]  # lingua -> tutti i caratteri
        pages: list[dict]  # per-page: {folio, section, language, hand, chars}
    """
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    # Pattern per header di pagina con metadata
    header_re = re.compile(r"^<(f\w+)>\s+<!\s*(.*?)>")
    # Pattern per estrarre metadata $K=V
    meta_re = re.compile(r"\$(\w)=(\w+)")
    # Pattern per righe di trascrizione
    transcription_re = re.compile(
        r"^<(f\w+)\.\d+[^;]*;(\w)>\s+(.+)"
    )

    sections = defaultdict(list)
    languages = defaultdict(list)
    hands = defaultdict(list)
    pages = []

    current_meta = {}
    current_folio = None
    current_chars = []

    def flush_page():
        if current_folio and current_chars:
            section = current_meta.get("I", "?")
            language = current_meta.get("L", "?")
            hand = current_meta.get("H", "?")

            pages.append({
                "folio": current_folio,
                "section": section,
                "language": language,
                "hand": hand,
                "chars": list(current_chars),
            })
            sections[section].extend(current_chars)
            if language != "?":
                languages[language].extend(current_chars)
            if hand != "?":
                hands[hand].extend(current_chars)

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue

        # Header di pagina con metadata
        m = header_re.match(line)
        if m:
            flush_page()
            current_folio = m.group(1)
            meta_str = m.group(2)
            current_meta = dict(meta_re.findall(meta_str))
            current_chars = []
            continue

        # Riga di trascrizione
        m = transcription_re.match(line)
        if not m:
            continue

        line_transcriber = m.group(2)
        if line_transcriber != transcriber:
            continue

        eva_text = m.group(3)

        # Rimuovi annotazioni inline
        clean = re.sub(r"\{[^}]*\}", "", eva_text)
        clean = re.sub(r"<[^>]*>", "", clean)
        clean = re.sub(r"[%!?\[\]*,]", "", clean)

        for char in clean:
            if char.isalpha() and char.islower():
                current_chars.append(char)

    flush_page()

    return dict(sections), dict(languages), dict(hands), pages


def compute_group_stats(chars: list[str], top_n: int = 30) -> dict:
    """Calcola statistiche su un gruppo di caratteri EVA.

    Stessa logica di eva_statistics: frequenze, entropia, Zipf, n-grammi.
    """
    total = len(chars)
    if total == 0:
        return {
            "total_characters": 0,
            "unique_characters": 0,
            "vocabulary": [],
            "frequencies": [],
            "shannon_entropy": 0.0,
            "zipf_slope": None,
            "ngrams": {},
        }

    counter = Counter(chars)
    freq_list = counter.most_common()
    frequencies = [
        {"char": char, "count": count, "frequency": round(count / total, 4)}
        for char, count in freq_list
    ]

    # Shannon entropy (usa probabilita' non arrotondate)
    probs = np.array([count / total for _, count in freq_list])
    entropy = float(-np.sum(probs * np.log2(probs)))

    # Zipf slope
    ranks = np.arange(1, len(freq_list) + 1)
    counts = np.array([f["count"] for f in frequencies])
    log_ranks = np.log10(ranks)
    log_counts = np.log10(counts)
    mask = np.isfinite(log_counts)
    zipf_slope = None
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_ranks[mask], log_counts[mask], 1)
        zipf_slope = round(float(coeffs[0]), 4)

    # N-grammi
    ngrams_results = {}
    for n in range(2, 5):
        ngrams = []
        for i in range(len(chars) - n + 1):
            ngrams.append(tuple(chars[i:i + n]))
        counter_ng = Counter(ngrams)
        top = counter_ng.most_common(top_n)
        ngrams_results[f"{n}-gram"] = [
            {"ngram": "".join(ng), "count": c} for ng, c in top
        ]

    return {
        "total_characters": total,
        "unique_characters": len(freq_list),
        "vocabulary": [f["char"] for f in frequencies],
        "frequencies": frequencies,
        "shannon_entropy": round(entropy, 3),
        "zipf_slope": zipf_slope,
        "ngrams": ngrams_results,
    }


@timer
def analyze_sections(sections, languages, hands, pages, top_n):
    """Calcola statistiche per ogni sezione, lingua e mano."""
    result = {
        "by_section": {},
        "by_language": {},
        "by_hand": {},
        "pages_count": len(pages),
    }

    print(f"    Sezioni trovate: {sorted(sections.keys())}")
    for key in sorted(sections.keys()):
        chars = sections[key]
        label = SECTION_NAMES.get(key, key)
        print(f"      {key} ({label}): {len(chars)} caratteri")
        result["by_section"][key] = compute_group_stats(chars, top_n)
        result["by_section"][key]["label"] = label
        # Conta pagine per sezione
        result["by_section"][key]["pages_count"] = sum(
            1 for p in pages if p["section"] == key
        )

    print(f"    Lingue trovate: {sorted(languages.keys())}")
    for key in sorted(languages.keys()):
        chars = languages[key]
        print(f"      {key}: {len(chars)} caratteri")
        result["by_language"][key] = compute_group_stats(chars, top_n)

    print(f"    Mani trovate: {sorted(hands.keys())}")
    for key in sorted(hands.keys()):
        chars = hands[key]
        print(f"      {key}: {len(chars)} caratteri")
        result["by_hand"][key] = compute_group_stats(chars, top_n)

    return result


@timer
def plot_section_heatmap(stats: dict, output_dir: Path):
    """Matrice sezioni x caratteri EVA, colore = frequenza normalizzata."""
    by_section = stats["by_section"]
    section_keys = sorted(by_section.keys())

    # Raccogli tutti i caratteri usati (union)
    all_chars = set()
    for key in section_keys:
        all_chars.update(by_section[key]["vocabulary"])
    all_chars = sorted(all_chars)

    if not all_chars or not section_keys:
        return

    # Matrice frequenze
    matrix = np.zeros((len(all_chars), len(section_keys)))
    for j, sec in enumerate(section_keys):
        freq_map = {
            f["char"]: f["frequency"]
            for f in by_section[sec]["frequencies"]
        }
        for i, ch in enumerate(all_chars):
            matrix[i, j] = freq_map.get(ch, 0.0)

    # Filtra per mostrare solo i top 25 caratteri (per leggibilita')
    char_totals = matrix.sum(axis=1)
    top_idx = np.argsort(char_totals)[-25:][::-1]
    matrix_top = matrix[top_idx, :]
    chars_top = [all_chars[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    section_labels = [
        f"{k} ({SECTION_NAMES.get(k, k)})" for k in section_keys
    ]
    im = ax.imshow(matrix_top, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(section_keys)))
    ax.set_xticklabels(section_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(chars_top)))
    ax.set_yticklabels(chars_top, fontsize=9)
    ax.set_xlabel("Sezione")
    ax.set_ylabel("Carattere EVA")
    ax.set_title("Frequenza caratteri per sezione del manoscritto")
    fig.colorbar(im, ax=ax, label="Frequenza normalizzata")
    plt.tight_layout()
    plt.savefig(str(output_dir / "section_heatmap.png"), dpi=150)
    plt.close()


@timer
def plot_section_comparison(stats: dict, output_dir: Path):
    """2x2 bar chart: entropy, Zipf slope, vocabulary size, chars/page per sezione."""
    by_section = stats["by_section"]
    section_keys = sorted(by_section.keys())
    labels = [f"{k}\n{SECTION_NAMES.get(k, k)}" for k in section_keys]

    entropies = [by_section[k]["shannon_entropy"] for k in section_keys]
    zipf_slopes = [
        by_section[k]["zipf_slope"] if by_section[k]["zipf_slope"] is not None else 0
        for k in section_keys
    ]
    vocab_sizes = [by_section[k]["unique_characters"] for k in section_keys]
    chars_per_page = [
        round(by_section[k]["total_characters"] / max(by_section[k]["pages_count"], 1), 1)
        for k in section_keys
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Confronto statistiche per sezione", fontsize=14, fontweight="bold")
    bar_color = "#8B4513"

    axes[0, 0].bar(labels, entropies, color=bar_color, alpha=0.7)
    axes[0, 0].set_title("Shannon Entropy (bit/simbolo)")
    axes[0, 0].set_ylabel("Entropy")
    for i, v in enumerate(entropies):
        axes[0, 0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    axes[0, 1].bar(labels, zipf_slopes, color=bar_color, alpha=0.7)
    axes[0, 1].set_title("Zipf Slope")
    axes[0, 1].set_ylabel("Slope")
    for i, v in enumerate(zipf_slopes):
        axes[0, 1].text(i, v - 0.05, f"{v:.2f}", ha="center", fontsize=8)

    axes[1, 0].bar(labels, vocab_sizes, color=bar_color, alpha=0.7)
    axes[1, 0].set_title("Vocabulary Size (caratteri unici)")
    axes[1, 0].set_ylabel("Unici")
    for i, v in enumerate(vocab_sizes):
        axes[1, 0].text(i, v + 0.2, str(v), ha="center", fontsize=8)

    axes[1, 1].bar(labels, chars_per_page, color=bar_color, alpha=0.7)
    axes[1, 1].set_title("Caratteri / pagina")
    axes[1, 1].set_ylabel("Char/page")
    for i, v in enumerate(chars_per_page):
        axes[1, 1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_dir / "section_comparison.png"), dpi=150)
    plt.close()


@timer
def plot_language_comparison(stats: dict, output_dir: Path):
    """Frequenze Lingua A vs B side-by-side + entropy/Zipf annotati."""
    by_lang = stats["by_language"]
    if "A" not in by_lang or "B" not in by_lang:
        print("    Lingue A e B non entrambe presenti, skip plot lingua")
        return

    stats_a = by_lang["A"]
    stats_b = by_lang["B"]

    # Union dei top 20 caratteri per frequenza
    freq_a = {f["char"]: f["frequency"] for f in stats_a["frequencies"]}
    freq_b = {f["char"]: f["frequency"] for f in stats_b["frequencies"]}
    all_chars = sorted(
        set(list(freq_a.keys())[:20]) | set(list(freq_b.keys())[:20])
    )

    vals_a = [freq_a.get(c, 0) for c in all_chars]
    vals_b = [freq_b.get(c, 0) for c in all_chars]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confronto Lingua A vs Lingua B (Currier)", fontsize=14, fontweight="bold")

    x = np.arange(len(all_chars))
    w = 0.35
    axes[0].bar(x - w / 2, vals_a, w, label="Lingua A", color="#8B4513", alpha=0.7)
    axes[0].bar(x + w / 2, vals_b, w, label="Lingua B", color="#2E8B57", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_chars, fontsize=8)
    axes[0].set_xlabel("Carattere EVA")
    axes[0].set_ylabel("Frequenza relativa")
    axes[0].set_title("Distribuzione frequenze")
    axes[0].legend()

    # Pannello testuale con statistiche
    axes[1].axis("off")
    text_lines = [
        "Statistiche comparative\n",
        f"{'Metrica':<25} {'Lingua A':>10} {'Lingua B':>10}",
        f"{'-' * 47}",
        f"{'Caratteri totali':<25} {stats_a['total_characters']:>10,} {stats_b['total_characters']:>10,}",
        f"{'Caratteri unici':<25} {stats_a['unique_characters']:>10} {stats_b['unique_characters']:>10}",
        f"{'Shannon entropy':<25} {stats_a['shannon_entropy']:>10.3f} {stats_b['shannon_entropy']:>10.3f}",
        f"{'Zipf slope':<25} {stats_a['zipf_slope'] or 'N/A':>10} {stats_b['zipf_slope'] or 'N/A':>10}",
    ]

    # Top 5 bigrammi per lingua
    text_lines.append(f"\n{'Top 5 bigrammi A':<25} {'Top 5 bigrammi B':>22}")
    bg_a = stats_a.get("ngrams", {}).get("2-gram", [])[:5]
    bg_b = stats_b.get("ngrams", {}).get("2-gram", [])[:5]
    for i in range(5):
        a_str = f"{bg_a[i]['ngram']} ({bg_a[i]['count']})" if i < len(bg_a) else ""
        b_str = f"{bg_b[i]['ngram']} ({bg_b[i]['count']})" if i < len(bg_b) else ""
        text_lines.append(f"  {a_str:<23} {b_str:>22}")

    axes[1].text(
        0.05, 0.95, "\n".join(text_lines),
        transform=axes[1].transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(str(output_dir / "language_comparison.png"), dpi=150)
    plt.close()


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per l'analisi per sezione."""
    print_header("VOYNICH TOOLKIT - Analisi per Sezione")
    config.ensure_dirs()

    report_path = config.stats_dir / "section_analysis.json"
    if report_path.exists() and not force:
        print("  Report gia' presente, skip (usa --force per rieseguire)")
        return

    # Verifica file EVA
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(
            f"File EVA non trovato: {eva_file}\n"
            "  Esegui prima: voynich eva"
        )

    # 1. Parsing con metadata
    print_step("Parsing EVA con metadata per sezione...")
    sections, languages, hands, pages = parse_eva_by_section(eva_file)
    total_chars = sum(len(v) for v in sections.values())
    print(f"    {len(pages)} folii, {total_chars} caratteri totali")

    # 2. Statistiche per gruppo
    print_step("Calcolo statistiche per sezione/lingua/mano...")
    stats = analyze_sections(sections, languages, hands, pages, config.top_n_ngrams)

    # 3. Visualizzazioni
    print_step("Generazione heatmap sezioni...")
    plot_section_heatmap(stats, config.stats_dir)

    print_step("Generazione confronto sezioni...")
    plot_section_comparison(stats, config.stats_dir)

    print_step("Generazione confronto lingue...")
    plot_language_comparison(stats, config.stats_dir)

    # 4. Salva report
    print_step("Salvataggio report...")

    # Prepara report serializzabile (rimuovi i chars grezzi dai pages)
    report = {
        "total_characters": total_chars,
        "total_pages": len(pages),
        "by_section": {},
        "by_language": {},
        "by_hand": {},
        "pages": [
            {
                "folio": p["folio"],
                "section": p["section"],
                "language": p["language"],
                "hand": p["hand"],
                "char_count": len(p["chars"]),
            }
            for p in pages
        ],
    }

    for group_name in ("by_section", "by_language", "by_hand"):
        for key, group_stats in stats[group_name].items():
            # Copia stats senza la lista completa di frequenze (troppo lunga)
            report[group_name][key] = {
                k: v for k, v in group_stats.items()
                if k != "frequencies"
            }
            # Mantieni solo top 15 frequenze
            report[group_name][key]["top_frequencies"] = group_stats["frequencies"][:15]

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Stampa sommario
    print("\n" + "=" * 60)
    print("  REPORT ANALISI PER SEZIONE")
    print("=" * 60)
    print(f"  Caratteri totali:  {total_chars}")
    print(f"  Pagine analizzate: {len(pages)}")

    print(f"\n  {'Sez':<5} {'Nome':<16} {'Pagine':>7} {'Chars':>8} "
          f"{'Unici':>6} {'Entropy':>8} {'Zipf':>7}")
    print(f"  {'-' * 59}")
    for key in sorted(stats["by_section"].keys()):
        s = stats["by_section"][key]
        label = SECTION_NAMES.get(key, key)
        zipf = f"{s['zipf_slope']:.2f}" if s["zipf_slope"] else "N/A"
        print(f"  {key:<5} {label:<16} {s['pages_count']:>7} {s['total_characters']:>8,} "
              f"{s['unique_characters']:>6} {s['shannon_entropy']:>8.3f} {zipf:>7}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
