"""
Analisi della struttura delle parole Voynichesi.

Tokenizer condiviso parse_eva_words() + analisi posizionale dei caratteri,
distribuzione lunghezza parole, Zipf, clustering funzionale.
"""
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import click
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer


# Lunghezze medie parole per lingue di riferimento
REFERENCE_WORD_LENGTHS = {
    "Latin": 5.5,
    "Italian": 5.1,
    "English": 4.7,
    "German": 6.3,
    "Arabic": 4.9,
    "Hebrew": 4.5,
}


def parse_eva_words(filepath: Path, transcriber: str = "H") -> dict:
    """Parsa il file IVTFF estraendo parole (dot-separated).

    Returns: {
        'words': list[str],       # tutte le parole in ordine
        'pages': list[dict],      # {folio, section, language, hand, words, line_words}
        'total_words': int,
        'unique_words': int,
        'total_chars': int
    }
    """
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    header_re = re.compile(r"^<(f\w+)>\s+<!\s*(.*?)>")
    meta_re = re.compile(r"\$(\w)=(\w+)")
    transcription_re = re.compile(r"^<(f\w+)\.\d+[^;]*;(\w)>\s+(.+)")

    all_words = []
    pages = []

    current_meta = {}
    current_folio = None
    current_words = []
    current_line_words = []

    def flush_page():
        nonlocal current_folio, current_words, current_line_words
        if current_folio and current_words:
            pages.append({
                "folio": current_folio,
                "section": current_meta.get("I", "?"),
                "language": current_meta.get("L", "?"),
                "hand": current_meta.get("H", "?"),
                "words": list(current_words),
                "line_words": list(current_line_words),
            })
            all_words.extend(current_words)
        current_words = []
        current_line_words = []

    def extract_words(eva_text: str) -> list[str]:
        """Pulisce una riga EVA ed estrae parole."""
        clean = re.sub(r"\{[^}]*\}", "", eva_text)
        clean = re.sub(r"<[^>]*>", "", clean)
        clean = re.sub(r"[%!?\[\]*,]", "", clean)
        words = []
        for token in clean.split("."):
            word = re.sub(r"[^a-z]", "", token)
            if word:
                words.append(word)
        return words

    for line in lines:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue

        m = header_re.match(line)
        if m:
            flush_page()
            current_folio = m.group(1)
            current_meta = dict(meta_re.findall(m.group(2)))
            continue

        m = transcription_re.match(line)
        if not m:
            continue

        if m.group(2) != transcriber:
            continue

        line_ws = extract_words(m.group(3))
        current_words.extend(line_ws)
        current_line_words.append(line_ws)

    flush_page()

    unique = set(all_words)
    total_chars = sum(len(w) for w in all_words)

    return {
        "words": all_words,
        "pages": pages,
        "total_words": len(all_words),
        "unique_words": len(unique),
        "total_chars": total_chars,
    }


def compute_word_frequencies(words: list[str], top_n: int = 50) -> dict:
    """Frequenze parole + Zipf slope su parole."""
    counter = Counter(words)
    freq_list = counter.most_common()
    total = len(words)

    top_words = [
        {"word": w, "count": c, "frequency": round(c / total, 6)}
        for w, c in freq_list[:top_n]
    ]

    # Zipf slope
    ranks = np.arange(1, len(freq_list) + 1)
    counts = np.array([c for _, c in freq_list])
    log_r = np.log10(ranks)
    log_c = np.log10(counts)
    mask = np.isfinite(log_c)
    zipf_slope = None
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_r[mask], log_c[mask], 1)
        zipf_slope = round(float(coeffs[0]), 4)

    return {
        "total_words": total,
        "unique_words": len(freq_list),
        "top_words": top_words,
        "zipf_slope": zipf_slope,
        "hapax_count": sum(1 for _, c in freq_list if c == 1),
    }


def compute_word_length_distribution(words: list[str]) -> dict:
    """Istogramma lunghezze + confronto con lingue note."""
    lengths = [len(w) for w in words]
    avg_len = round(np.mean(lengths), 2)
    median_len = int(np.median(lengths))
    std_len = round(np.std(lengths), 2)

    counter = Counter(lengths)
    distribution = [
        {"length": k, "count": counter[k]}
        for k in sorted(counter.keys())
    ]

    return {
        "avg_word_length": avg_len,
        "median_word_length": median_len,
        "std_word_length": std_len,
        "distribution": distribution,
        "reference_languages": REFERENCE_WORD_LENGTHS,
    }


def compute_char_positional_profiles(words: list[str]) -> dict:
    """Per ogni carattere: P(initial), P(medial), P(final), P(isolated).

    Usa parole len>=2 per initial/final, len>=3 per medial.
    """
    initial_counts = Counter()
    medial_counts = Counter()
    final_counts = Counter()
    isolated_counts = Counter()
    total_initial = 0
    total_medial = 0
    total_final = 0
    total_isolated = 0

    for w in words:
        if len(w) == 1:
            isolated_counts[w[0]] += 1
            total_isolated += 1
        elif len(w) == 2:
            initial_counts[w[0]] += 1
            final_counts[w[1]] += 1
            total_initial += 1
            total_final += 1
        else:  # len >= 3
            initial_counts[w[0]] += 1
            final_counts[w[-1]] += 1
            total_initial += 1
            total_final += 1
            for ch in w[1:-1]:
                medial_counts[ch] += 1
                total_medial += 1

    all_chars = sorted(set(initial_counts) | set(medial_counts) |
                       set(final_counts) | set(isolated_counts))

    profiles = {}
    for ch in all_chars:
        ini = initial_counts[ch] / total_initial if total_initial else 0
        med = medial_counts[ch] / total_medial if total_medial else 0
        fin = final_counts[ch] / total_final if total_final else 0
        iso = isolated_counts[ch] / total_isolated if total_isolated else 0
        profiles[ch] = {
            "initial": round(ini, 4),
            "medial": round(med, 4),
            "final": round(fin, 4),
            "isolated": round(iso, 4),
        }

    return {"profiles": profiles, "characters": all_chars}


def compute_slot_analysis(words: list[str], max_pos: int = 10) -> dict:
    """Matrice posizione-assoluta x carattere, sia da inizio che da fine."""
    from_start = defaultdict(Counter)
    from_end = defaultdict(Counter)

    for w in words:
        for i, ch in enumerate(w):
            if i < max_pos:
                from_start[i][ch] += 1
            dist_end = len(w) - 1 - i
            if dist_end < max_pos:
                from_end[dist_end][ch] += 1

    all_chars = sorted(set(
        ch for pos_counter in from_start.values() for ch in pos_counter
    ))

    start_matrix = []
    end_matrix = []
    for pos in range(max_pos):
        start_total = sum(from_start[pos].values()) or 1
        end_total = sum(from_end[pos].values()) or 1
        start_matrix.append({
            ch: round(from_start[pos].get(ch, 0) / start_total, 4)
            for ch in all_chars
        })
        end_matrix.append({
            ch: round(from_end[pos].get(ch, 0) / end_total, 4)
            for ch in all_chars
        })

    return {
        "characters": all_chars,
        "from_start": start_matrix,
        "from_end": end_matrix,
        "max_pos": max_pos,
    }


def compute_functional_groups(profiles: dict, n_clusters: int = 4) -> dict:
    """Clustering gerarchico (ward) sui vettori posizionali 4D."""
    chars = profiles["characters"]
    prof = profiles["profiles"]

    if len(chars) < n_clusters:
        return {"clusters": {}, "characters": chars}

    vectors = np.array([
        [prof[ch]["initial"], prof[ch]["medial"],
         prof[ch]["final"], prof[ch]["isolated"]]
        for ch in chars
    ])

    Z = linkage(vectors, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = defaultdict(list)
    for ch, label in zip(chars, labels):
        clusters[int(label)].append(ch)

    return {
        "clusters": dict(clusters),
        "characters": chars,
        "labels": [int(x) for x in labels],
    }


# --- Visualizzazioni ---

@timer
def plot_word_length_distribution(length_stats: dict, output_dir: Path):
    """Istogramma lunghezze + linee verticali per lingue note."""
    dist = length_stats["distribution"]
    lengths = [d["length"] for d in dist]
    counts = [d["count"] for d in dist]
    avg = length_stats["avg_word_length"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lengths, counts, color="#8B4513", alpha=0.7, label="Voynichese")
    ax.axvline(avg, color="#8B4513", linestyle="--", linewidth=2,
               label=f"Voynich media: {avg:.1f}")

    colors = ["#2E8B57", "#4169E1", "#DC143C", "#FF8C00", "#9370DB", "#20B2AA"]
    for (lang, ref_len), color in zip(REFERENCE_WORD_LENGTHS.items(), colors):
        ax.axvline(ref_len, color=color, linestyle=":", linewidth=1.2,
                   alpha=0.8, label=f"{lang}: {ref_len}")

    ax.set_xlabel("Lunghezza parola (caratteri)")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione lunghezza parole Voynichesi")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, max(lengths) + 1)
    plt.tight_layout()
    plt.savefig(str(output_dir / "word_length_distribution.png"), dpi=150)
    plt.close()


@timer
def plot_char_positional_profiles(profiles: dict, output_dir: Path):
    """Heatmap caratteri x {Initial, Medial, Final, Isolated}."""
    chars = profiles["characters"]
    prof = profiles["profiles"]

    # Filter to chars with meaningful presence
    total_presence = []
    for ch in chars:
        p = prof[ch]
        total_presence.append(p["initial"] + p["medial"] + p["final"] + p["isolated"])

    # Keep top characters by total presence
    idx_sorted = np.argsort(total_presence)[::-1]
    max_show = min(25, len(chars))
    top_idx = idx_sorted[:max_show]
    top_chars = [chars[i] for i in top_idx]

    positions = ["Initial", "Medial", "Final", "Isolated"]
    matrix = np.array([
        [prof[ch]["initial"], prof[ch]["medial"],
         prof[ch]["final"], prof[ch]["isolated"]]
        for ch in top_chars
    ])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(4))
    ax.set_xticklabels(positions, fontsize=10)
    ax.set_yticks(range(len(top_chars)))
    ax.set_yticklabels(top_chars, fontsize=9)
    ax.set_xlabel("Posizione nella parola")
    ax.set_ylabel("Carattere EVA")
    ax.set_title("Profili posizionali dei caratteri")
    fig.colorbar(im, ax=ax, label="P(posizione)", shrink=0.8)

    # Annotate cells
    for i in range(len(top_chars)):
        for j in range(4):
            val = matrix[i, j]
            if val > 0.005:
                text_color = "white" if val > 0.5 * matrix.max() else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.tight_layout()
    plt.savefig(str(output_dir / "char_positional_profiles.png"), dpi=150)
    plt.close()


@timer
def plot_word_zipf(word_freqs: dict, output_dir: Path):
    """Log-log scatter frequenza parole + fit lineare."""
    top = word_freqs["top_words"]
    total = word_freqs["total_words"]

    # Use full distribution for Zipf plot
    all_counts = [w["count"] for w in top]
    ranks = np.arange(1, len(all_counts) + 1)
    log_r = np.log10(ranks)
    log_c = np.log10(all_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(log_r, log_c, s=30, color="#8B4513", alpha=0.7, zorder=3)

    # Fit line
    mask = np.isfinite(log_c)
    if np.sum(mask) > 2:
        coeffs = np.polyfit(log_r[mask], log_c[mask], 1)
        fit_line = np.poly1d(coeffs)(log_r)
        ax.plot(log_r, fit_line, "r--", linewidth=1.5,
                label=f"slope = {coeffs[0]:.2f}")
        ax.legend(fontsize=10)

    # Label top words
    for i in range(min(10, len(top))):
        ax.annotate(top[i]["word"], (log_r[i], log_c[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    ax.set_xlabel("log10(Rank)")
    ax.set_ylabel("log10(Frequenza)")
    ax.set_title(f"Legge di Zipf - Parole Voynichesi (n={total})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "word_zipf.png"), dpi=150)
    plt.close()


# --- Entry point ---

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per l'analisi struttura parole."""
    print_header("VOYNICH TOOLKIT - Struttura Parole")
    config.ensure_dirs()

    report_path = config.stats_dir / "word_structure.json"
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
    print(f"    {data['total_words']} parole, {data['unique_words']} uniche, "
          f"{data['total_chars']} caratteri")

    # 2. Frequenze parole
    print_step("Calcolo frequenze parole...")
    word_freqs = compute_word_frequencies(words)
    print(f"    Zipf slope: {word_freqs['zipf_slope']}")
    print(f"    Hapax legomena: {word_freqs['hapax_count']}")

    # 3. Distribuzione lunghezze
    print_step("Distribuzione lunghezza parole...")
    length_stats = compute_word_length_distribution(words)
    print(f"    Media: {length_stats['avg_word_length']}, "
          f"Mediana: {length_stats['median_word_length']}, "
          f"Std: {length_stats['std_word_length']}")

    # 4. Profili posizionali
    print_step("Profili posizionali caratteri...")
    profiles = compute_char_positional_profiles(words)
    print(f"    {len(profiles['characters'])} caratteri analizzati")

    # 5. Slot analysis
    print_step("Analisi slot posizionali...")
    slots = compute_slot_analysis(words)

    # 6. Gruppi funzionali
    print_step("Clustering gruppi funzionali...")
    groups = compute_functional_groups(profiles)
    for cluster_id, members in sorted(groups["clusters"].items()):
        print(f"    Gruppo {cluster_id}: {' '.join(members)}")

    # 7. Visualizzazioni
    print_step("Generazione grafici...")
    plot_word_length_distribution(length_stats, config.stats_dir)
    plot_char_positional_profiles(profiles, config.stats_dir)
    plot_word_zipf(word_freqs, config.stats_dir)

    # 8. Salva report
    print_step("Salvataggio report...")
    report = {
        "total_words": data["total_words"],
        "unique_words": data["unique_words"],
        "total_chars": data["total_chars"],
        "pages_count": len(data["pages"]),
        "word_frequencies": word_freqs,
        "word_length": length_stats,
        "char_positional_profiles": profiles["profiles"],
        "functional_groups": groups["clusters"],
        "slot_analysis": {
            "characters": slots["characters"],
            "from_start": slots["from_start"][:5],
            "from_end": slots["from_end"][:5],
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Sommario
    print("\n" + "=" * 60)
    print("  REPORT STRUTTURA PAROLE")
    print("=" * 60)
    print(f"  Parole totali:   {data['total_words']}")
    print(f"  Parole uniche:   {data['unique_words']}")
    print(f"  Caratteri tot:   {data['total_chars']}")
    print(f"  Lunghezza media: {length_stats['avg_word_length']}")
    print(f"  Zipf slope:      {word_freqs['zipf_slope']}")

    print(f"\n  Top 10 parole:")
    for w in word_freqs["top_words"][:10]:
        bar = "#" * int(w["frequency"] * 500)
        print(f"    {w['word']:<12} {w['count']:>5} ({w['frequency']:.3%}) {bar}")

    print(f"\n  Gruppi funzionali:")
    for cid, members in sorted(groups["clusters"].items()):
        print(f"    Gruppo {cid}: {', '.join(members)}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
