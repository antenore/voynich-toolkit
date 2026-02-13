"""
Embeddings e analisi co-occorrenza caratteri Voynichesi.

Scopre gruppi funzionali di caratteri (vocali vs consonanti?) tramite
pattern di co-occorrenza dentro le parole.
"""
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import click
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from .config import ToolkitConfig
from .utils import print_header, print_step, timer
from .word_structure import parse_eva_words


def compute_cooccurrence_matrix(words: list[str],
                                min_char_count: int = 50) -> dict:
    """Bigrammi adiacenti DENTRO le parole. Filtra caratteri rari."""
    # Count character frequencies
    char_counter = Counter()
    for w in words:
        char_counter.update(w)

    # Filter to frequent characters
    char_list = sorted(ch for ch, c in char_counter.items() if c >= min_char_count)
    char_idx = {ch: i for i, ch in enumerate(char_list)}
    n = len(char_list)

    matrix = np.zeros((n, n), dtype=int)
    for w in words:
        for i in range(len(w) - 1):
            a, b = w[i], w[i + 1]
            if a in char_idx and b in char_idx:
                matrix[char_idx[a], char_idx[b]] += 1

    return {
        "matrix": matrix,
        "char_list": char_list,
        "char_counts": {ch: char_counter[ch] for ch in char_list},
    }


def compute_pmi_matrix(cooccurrence: dict) -> dict:
    """PPMI (Positive Pointwise Mutual Information).

    PPMI = max(0, log2(P(x,y) / (P(x)*P(y))))
    """
    matrix = cooccurrence["matrix"].astype(float)
    char_list = cooccurrence["char_list"]
    n = len(char_list)

    total = matrix.sum()
    if total == 0:
        return {"matrix": np.zeros((n, n)), "char_list": char_list}

    # Marginal probabilities
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    pmi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                p_xy = matrix[i, j] / total
                p_x = row_sums[i] / total
                p_y = col_sums[j] / total
                pmi[i, j] = max(0.0, math.log2(p_xy / (p_x * p_y)))

    return {"matrix": pmi, "char_list": char_list}


def compute_svd_embeddings(pmi_data: dict, n_components: int = 3) -> dict:
    """SVD su matrice PPMI. Embeddings = U[:,:k] * sqrt(S[:k])."""
    matrix = pmi_data["matrix"]
    char_list = pmi_data["char_list"]
    n = len(char_list)

    if n < n_components:
        n_components = max(1, n)

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(n_components, len(S))
    embeddings = U[:, :k] * np.sqrt(S[:k])

    # Variance explained
    total_var = (S ** 2).sum()
    var_explained = [(S[i] ** 2) / total_var for i in range(k)] if total_var > 0 else [0] * k

    return {
        "embeddings": embeddings,
        "char_list": char_list,
        "n_components": k,
        "variance_explained": [round(float(v), 4) for v in var_explained],
    }


def cluster_characters(embedding_data: dict, n_clusters: int = 4) -> dict:
    """Clustering gerarchico (ward) sugli embeddings."""
    embeddings = embedding_data["embeddings"]
    char_list = embedding_data["char_list"]

    if len(char_list) < n_clusters:
        return {"clusters": {}, "labels": [], "char_list": char_list}

    Z = linkage(embeddings, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = defaultdict(list)
    for ch, label in zip(char_list, labels):
        clusters[int(label)].append(ch)

    return {
        "clusters": dict(clusters),
        "labels": [int(x) for x in labels],
        "char_list": char_list,
    }


def compute_transition_analysis(words: list[str], char_list: list[str]) -> dict:
    """Distribuzione successori/predecessori per carattere.

    Predittivita' = 1 - H(next|current) / H_max.
    """
    char_set = set(char_list)
    successors = defaultdict(Counter)
    predecessors = defaultdict(Counter)

    for w in words:
        for i in range(len(w) - 1):
            a, b = w[i], w[i + 1]
            if a in char_set and b in char_set:
                successors[a][b] += 1
                predecessors[b][a] += 1

    h_max = math.log2(len(char_list)) if len(char_list) > 1 else 1.0

    analysis = {}
    for ch in char_list:
        # Successor entropy
        succ = successors[ch]
        total_succ = sum(succ.values())
        h_succ = 0.0
        if total_succ > 0:
            for count in succ.values():
                p = count / total_succ
                if p > 0:
                    h_succ -= p * math.log2(p)

        # Predecessor entropy
        pred = predecessors[ch]
        total_pred = sum(pred.values())
        h_pred = 0.0
        if total_pred > 0:
            for count in pred.values():
                p = count / total_pred
                if p > 0:
                    h_pred -= p * math.log2(p)

        predictivity = round(1 - h_succ / h_max, 4) if h_max > 0 else 0.0

        analysis[ch] = {
            "successor_entropy": round(h_succ, 4),
            "predecessor_entropy": round(h_pred, 4),
            "predictivity": predictivity,
            "top_successors": [
                {"char": c, "count": n}
                for c, n in succ.most_common(5)
            ],
        }

    return analysis


def compute_word_vectors(words: list[str], embedding_data: dict,
                         top_n_pairs: int = 20) -> dict:
    """Media embedding caratteri per parola. Nearest-neighbor pairs (top 500 parole)."""
    embeddings = embedding_data["embeddings"]
    char_list = embedding_data["char_list"]
    char_idx = {ch: i for i, ch in enumerate(char_list)}

    word_counter = Counter(words)
    top_words = [w for w, _ in word_counter.most_common(500)]

    word_vecs = {}
    for w in top_words:
        indices = [char_idx[ch] for ch in w if ch in char_idx]
        if indices:
            vec = embeddings[indices].mean(axis=0)
            word_vecs[w] = vec

    # Cosine similarity nearest neighbors
    word_list = list(word_vecs.keys())
    if len(word_list) < 2:
        return {"pairs": []}

    vecs = np.array([word_vecs[w] for w in word_list])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs_normed = vecs / norms

    sim_matrix = vecs_normed @ vecs_normed.T
    np.fill_diagonal(sim_matrix, -1)

    # Top pairs
    pairs = []
    flat_idx = np.argsort(sim_matrix.ravel())[::-1]
    seen = set()
    for idx in flat_idx[:top_n_pairs * 3]:  # oversample to handle dedup
        i, j = divmod(int(idx), len(word_list))
        if i >= j:
            continue
        key = (min(word_list[i], word_list[j]), max(word_list[i], word_list[j]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "word_a": word_list[i],
            "word_b": word_list[j],
            "similarity": round(float(sim_matrix[i, j]), 4),
        })
        if len(pairs) >= top_n_pairs:
            break

    return {"pairs": pairs}


# --- Visualizzazioni ---

@timer
def plot_cooccurrence_heatmap(pmi_data: dict, output_dir: Path):
    """Heatmap PPMI top 15 caratteri."""
    matrix = pmi_data["matrix"]
    char_list = pmi_data["char_list"]

    # Top 15 by row sum
    row_sums = matrix.sum(axis=1)
    top_idx = np.argsort(row_sums)[-15:][::-1]
    sub_matrix = matrix[np.ix_(top_idx, top_idx)]
    sub_chars = [char_list[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(sub_chars)))
    ax.set_xticklabels(sub_chars, fontsize=10)
    ax.set_yticks(range(len(sub_chars)))
    ax.set_yticklabels(sub_chars, fontsize=10)
    ax.set_title("PPMI co-occorrenza caratteri (top 15)")
    fig.colorbar(im, ax=ax, label="PPMI")

    # Annotate
    for i in range(len(sub_chars)):
        for j in range(len(sub_chars)):
            val = sub_matrix[i, j]
            if val > 0.1:
                text_color = "white" if val > 0.5 * sub_matrix.max() else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.tight_layout()
    plt.savefig(str(output_dir / "char_cooccurrence.png"), dpi=150)
    plt.close()


@timer
def plot_embeddings_scatter(embedding_data: dict, cluster_data: dict,
                            output_dir: Path):
    """Scatter 2D SVD, etichettati, colorati per cluster."""
    embeddings = embedding_data["embeddings"]
    char_list = embedding_data["char_list"]
    labels = cluster_data.get("labels", [0] * len(char_list))
    clusters = cluster_data.get("clusters", {})

    if embeddings.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    cluster_colors = ["#8B4513", "#2E8B57", "#4169E1", "#DC143C",
                      "#FF8C00", "#9370DB", "#20B2AA", "#B22222"]
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        members = clusters.get(label, [])
        color = cluster_colors[(label - 1) % len(cluster_colors)]
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1],
                   s=100, c=color, alpha=0.7, edgecolors="black", linewidth=0.5,
                   label=f"Gruppo {label}: {', '.join(members)}", zorder=3)

    # Label points
    for i, ch in enumerate(char_list):
        ax.annotate(ch, (embeddings[i, 0], embeddings[i, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=11, fontweight="bold")

    ax.set_xlabel(f"SVD dim 1 ({embedding_data['variance_explained'][0]:.1%} var)")
    if len(embedding_data["variance_explained"]) > 1:
        ax.set_ylabel(f"SVD dim 2 ({embedding_data['variance_explained'][1]:.1%} var)")
    ax.set_title("Embeddings caratteri (SVD su PPMI)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / "char_embeddings_scatter.png"), dpi=150)
    plt.close()


@timer
def plot_transition_graph(cooccurrence: dict, transition_data: dict,
                          output_dir: Path):
    """Layout circolare puro matplotlib. Nodi = caratteri, archi = top transizioni."""
    matrix = cooccurrence["matrix"]
    char_list = cooccurrence["char_list"]
    char_counts = cooccurrence["char_counts"]
    n = len(char_list)

    if n == 0:
        return

    # Top 30 transitions
    flat = []
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0 and i != j:
                flat.append((i, j, matrix[i, j]))
    flat.sort(key=lambda x: x[2], reverse=True)
    top_edges = flat[:30]

    if not top_edges:
        return

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 4.0
    positions = {i: (radius * np.cos(a), radius * np.sin(a))
                 for i, a in enumerate(angles)}

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw edges
    max_count = max(e[2] for e in top_edges)
    for src, dst, count in top_edges:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        lw = 0.5 + 3.0 * (count / max_count)
        alpha = 0.3 + 0.5 * (count / max_count)

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.2",
            arrowstyle="->,head_width=4,head_length=3",
            linewidth=lw, alpha=alpha, color="#8B4513",
        )
        ax.add_patch(arrow)

    # Draw nodes
    max_freq = max(char_counts.values()) if char_counts else 1
    for i, ch in enumerate(char_list):
        x, y = positions[i]
        size = 200 + 800 * (char_counts.get(ch, 0) / max_freq)
        ax.scatter(x, y, s=size, c="wheat", edgecolors="#8B4513",
                   linewidth=2, zorder=5)
        ax.text(x, y, ch, ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Grafo transizioni caratteri (top 30)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(str(output_dir / "char_transition_graph.png"), dpi=150)
    plt.close()


# --- Entry point ---

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per embeddings e co-occorrenza."""
    print_header("VOYNICH TOOLKIT - Embeddings Caratteri")
    config.ensure_dirs()

    report_path = config.stats_dir / "char_embeddings.json"
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
    print(f"    {data['total_words']} parole")

    # 2. Co-occurrence
    print_step("Calcolo matrice co-occorrenza...")
    coocc = compute_cooccurrence_matrix(words)
    print(f"    {len(coocc['char_list'])} caratteri (min 50 occorrenze)")

    # 3. PMI
    print_step("Calcolo matrice PPMI...")
    pmi = compute_pmi_matrix(coocc)

    # 4. SVD Embeddings
    print_step("Calcolo SVD embeddings...")
    emb = compute_svd_embeddings(pmi, n_components=3)
    print(f"    Varianza spiegata: {emb['variance_explained']}")

    # 5. Clustering
    print_step("Clustering caratteri...")
    clusters = cluster_characters(emb, n_clusters=4)
    for cid, members in sorted(clusters["clusters"].items()):
        print(f"    Gruppo {cid}: {' '.join(members)}")

    # 6. Transition analysis
    print_step("Analisi transizioni...")
    transitions = compute_transition_analysis(words, coocc["char_list"])
    # Top predictive characters
    sorted_by_pred = sorted(transitions.items(),
                            key=lambda x: x[1]["predictivity"], reverse=True)
    print("    Top 5 predittivi:")
    for ch, t in sorted_by_pred[:5]:
        print(f"      {ch}: predictivity={t['predictivity']:.3f}")

    # 7. Word vectors
    print_step("Calcolo vettori parola...")
    word_vecs = compute_word_vectors(words, emb)
    print(f"    Top {len(word_vecs['pairs'])} coppie simili")

    # 8. Visualizzazioni
    print_step("Generazione grafici...")
    plot_cooccurrence_heatmap(pmi, config.stats_dir)
    plot_embeddings_scatter(emb, clusters, config.stats_dir)
    plot_transition_graph(coocc, transitions, config.stats_dir)

    # 9. Salva report
    print_step("Salvataggio report...")
    report = {
        "characters": coocc["char_list"],
        "char_counts": coocc["char_counts"],
        "pmi_top_pairs": [],
        "svd_variance_explained": emb["variance_explained"],
        "clusters": clusters["clusters"],
        "transitions": {
            ch: {k: v for k, v in t.items() if k != "top_successors"}
            for ch, t in transitions.items()
        },
        "top_transitions": {
            ch: t["top_successors"]
            for ch, t in sorted_by_pred[:15]
        },
        "word_vector_pairs": word_vecs["pairs"],
    }

    # Top PMI pairs
    pmi_matrix = pmi["matrix"]
    char_list = pmi["char_list"]
    pmi_pairs = []
    for i in range(len(char_list)):
        for j in range(i + 1, len(char_list)):
            val = pmi_matrix[i, j] + pmi_matrix[j, i]
            if val > 0:
                pmi_pairs.append({
                    "pair": f"{char_list[i]}{char_list[j]}",
                    "pmi": round(float(val / 2), 4),
                })
    pmi_pairs.sort(key=lambda x: x["pmi"], reverse=True)
    report["pmi_top_pairs"] = pmi_pairs[:30]

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Sommario
    print("\n" + "=" * 60)
    print("  REPORT EMBEDDINGS CARATTERI")
    print("=" * 60)
    print(f"  Caratteri analizzati: {len(coocc['char_list'])}")
    print(f"  Varianza SVD:        {emb['variance_explained']}")

    print(f"\n  Clusters:")
    for cid, members in sorted(clusters["clusters"].items()):
        print(f"    Gruppo {cid}: {', '.join(members)}")

    print(f"\n  Top 5 coppie PMI:")
    for p in report["pmi_top_pairs"][:5]:
        print(f"    {p['pair']}: {p['pmi']:.3f}")

    print(f"\n  Top 5 coppie parole simili:")
    for p in word_vecs["pairs"][:5]:
        print(f"    {p['word_a']} ~ {p['word_b']}: {p['similarity']:.3f}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
