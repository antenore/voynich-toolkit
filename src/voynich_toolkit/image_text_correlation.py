"""
Correlazione immagine-testo per il Manoscritto Voynich.

Per ogni pagina del PDF, calcola features visive (dall'immagine) e features
testuali (dai cataloghi regioni/glifi). Poi correla le due categorie e
visualizza, colorando per sezione EVA se il mapping e' disponibile.
"""
import re
import json
from pathlib import Path
from collections import defaultdict

import click
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ToolkitConfig
from .utils import print_header, print_step, timer, load_grayscale, binarize

# Nomi sezioni (duplicato da section_analysis per indipendenza)
SECTION_NAMES = {
    "H": "Herbal", "S": "Stars", "B": "Biological", "P": "Pharmaceutical",
    "C": "Cosmological", "Z": "Zodiac", "A": "Astronomical", "T": "Text",
}

SECTION_COLORS = {
    "H": "#2E8B57", "S": "#FFD700", "B": "#DC143C", "P": "#4169E1",
    "C": "#FF8C00", "Z": "#8A2BE2", "A": "#00CED1", "T": "#8B4513",
}


def build_page_mapping(eva_file: Path) -> list[dict]:
    """Costruisci mapping folio -> PDF page (best-effort).

    1. Estrai lista ordinata di folii dal file EVA (header <fNNx>)
    2. Raggruppa panel fold-out (f67r1, f67r2 -> stessa pagina fisica)
    3. Assegna pdf_page in ordine sequenziale alle pagine fisiche

    Il mapping e' approssimativo ma sufficiente per colorare scatter plot.
    """
    header_re = re.compile(r"^<(f\w+)>\s+<!\s*(.*?)>")
    meta_re = re.compile(r"\$(\w)=(\w+)")

    text = eva_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")

    folios = []
    for line in lines:
        m = header_re.match(line.rstrip())
        if m:
            folio = m.group(1)
            meta = dict(meta_re.findall(m.group(2)))
            folios.append({"folio": folio, "section": meta.get("I", "?"),
                           "language": meta.get("L", "?"),
                           "hand": meta.get("H", "?")})

    # Raggruppa fold-out panels per pagina fisica
    # f67r1, f67r2 -> base "f67r", f68r1 f68r2 f68r3 -> "f68r"
    # Anche fRos e' un caso speciale (Rosette) - trattato come singola pagina
    def folio_base(folio_name):
        """Estrai la base del folio (senza suffisso numerico per fold-out)."""
        # Casi: f1r, f1v, f67r1, f67r2, f72v3, fRos
        m = re.match(r"(f\d+[rv])\d*$", folio_name)
        if m:
            return m.group(1)
        return folio_name  # fRos, etc.

    # Raggruppa per pagina fisica (ogni recto e verso e' una pagina separata)
    physical_pages = []
    seen_bases = set()
    for f in folios:
        base = folio_base(f["folio"])
        if base not in seen_bases:
            seen_bases.add(base)
            physical_pages.append({
                "base": base,
                "folios": [f["folio"]],
                "section": f["section"],
                "language": f["language"],
            })
        else:
            # Aggiungi folio al gruppo esistente
            for pp in physical_pages:
                if pp["base"] == base:
                    pp["folios"].append(f["folio"])
                    break

    # Assegna pdf_page sequenziale
    mapping = []
    for i, pp in enumerate(physical_pages):
        page_num = i + 1
        mapping.append({
            "pdf_page": page_num,
            "page_id": f"page_{page_num:03d}",
            "base_folio": pp["base"],
            "folios": pp["folios"],
            "section": pp["section"],
            "language": pp["language"],
        })

    return mapping


@timer
def compute_visual_features(pages_dir: Path, page_ids: list[str],
                            threshold: int = 160) -> dict[str, dict]:
    """Calcola features visive per ogni pagina.

    - ink_density: % pixel scuri dopo binarizzazione
    - color_variance: varianza media dei canali BGR
    """
    features = {}
    for page_id in page_ids:
        img_path = pages_dir / f"{page_id}.png"
        if not img_path.exists():
            continue

        gray = load_grayscale(str(img_path))
        binary = binarize(gray, threshold=threshold, invert=True)
        ink_pixels = np.count_nonzero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]
        ink_density = ink_pixels / total_pixels

        # Color variance
        color_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if color_img is not None:
            color_variance = float(np.mean([
                np.var(color_img[:, :, c]) for c in range(3)
            ]))
        else:
            color_variance = 0.0

        features[page_id] = {
            "ink_density": round(ink_density, 4),
            "color_variance": round(color_variance, 2),
            "image_width": int(gray.shape[1]),
            "image_height": int(gray.shape[0]),
        }

    return features


@timer
def compute_region_features(regions_catalog: list[dict]) -> dict[str, dict]:
    """Calcola features per regioni raggruppate per pagina."""
    by_page = defaultdict(list)
    for r in regions_catalog:
        by_page[r["page"]].append(r)

    features = {}
    for page_id, regions in by_page.items():
        areas = [r["area"] for r in regions]
        features[page_id] = {
            "num_text_regions": len(regions),
            "total_text_area": sum(areas),
            "mean_region_size": round(np.mean(areas), 1) if areas else 0,
        }

    return features


@timer
def compute_glyph_features(glyphs_catalog: list[dict]) -> dict[str, dict]:
    """Calcola features per glifi raggruppati per pagina."""
    by_page = defaultdict(list)
    for g in glyphs_catalog:
        # Estrai pagina: page_001_region_02 -> page_001
        parts = g["region"].split("_region_")
        page_id = parts[0] if parts else g["region"]
        by_page[page_id].append(g)

    features = {}
    for page_id, glyphs in by_page.items():
        widths = [g["w"] for g in glyphs]
        heights = [g["h"] for g in glyphs]
        # Conta regioni uniche per questa pagina
        unique_regions = len(set(g["region"] for g in glyphs))

        features[page_id] = {
            "total_glyphs": len(glyphs),
            "glyphs_per_region": round(len(glyphs) / max(unique_regions, 1), 1),
            "mean_glyph_width": round(np.mean(widths), 1) if widths else 0,
            "mean_glyph_height": round(np.mean(heights), 1) if heights else 0,
        }

    return features


def merge_features(visual: dict, region: dict, glyph: dict) -> dict[str, dict]:
    """Combina features visive, di regioni e di glifi per pagina.

    Aggiunge features derivate: illustration_ratio, glyph_density.
    """
    all_pages = set(visual.keys()) | set(region.keys()) | set(glyph.keys())
    merged = {}

    for page_id in sorted(all_pages):
        v = visual.get(page_id, {})
        r = region.get(page_id, {})
        g = glyph.get(page_id, {})

        feat = {}
        feat.update(v)
        feat.update(r)
        feat.update(g)

        # Features derivate
        img_area = v.get("image_width", 1) * v.get("image_height", 1)
        text_area = r.get("total_text_area", 0)
        feat["illustration_ratio"] = round(1 - text_area / max(img_area, 1), 4)
        feat["glyph_density"] = round(
            g.get("total_glyphs", 0) / max(text_area, 1), 6
        )

        merged[page_id] = feat

    return merged


@timer
def compute_correlations(features: dict[str, dict]) -> dict:
    """Calcola matrice di correlazione Pearson tra features numeriche."""
    if not features:
        return {"feature_names": [], "matrix": []}

    # Feature numeriche da correlare
    numeric_keys = [
        "ink_density", "color_variance",
        "num_text_regions", "total_text_area", "mean_region_size",
        "total_glyphs", "glyphs_per_region",
        "mean_glyph_width", "mean_glyph_height",
        "illustration_ratio", "glyph_density",
    ]

    # Filtra pagine che hanno almeno alcune features
    pages = sorted(features.keys())
    available_keys = []
    for key in numeric_keys:
        if any(key in features[p] for p in pages):
            available_keys.append(key)

    if len(available_keys) < 2:
        return {"feature_names": available_keys, "matrix": []}

    # Matrice dati
    data = []
    for p in pages:
        row = [features[p].get(k, 0) for k in available_keys]
        data.append(row)
    data = np.array(data, dtype=float)

    # Correlazione Pearson
    n_features = len(available_keys)
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            xi = data[:, i]
            xj = data[:, j]
            std_i = np.std(xi)
            std_j = np.std(xj)
            if std_i > 0 and std_j > 0:
                corr_matrix[i, j] = np.corrcoef(xi, xj)[0, 1]
            elif i == j:
                corr_matrix[i, j] = 1.0

    return {
        "feature_names": available_keys,
        "matrix": corr_matrix.tolist(),
    }


@timer
def compute_pca(features: dict[str, dict]) -> dict:
    """PCA manuale via eigendecomposition numpy."""
    numeric_keys = [
        "ink_density", "color_variance",
        "num_text_regions", "total_text_area",
        "total_glyphs", "illustration_ratio", "glyph_density",
    ]

    pages = sorted(features.keys())
    available_keys = [k for k in numeric_keys if any(k in features[p] for p in pages)]

    if len(available_keys) < 2 or len(pages) < 3:
        return {"page_ids": pages, "components": [], "explained_variance": []}

    # Matrice dati
    data = np.array([
        [features[p].get(k, 0) for k in available_keys]
        for p in pages
    ], dtype=float)

    # Standardizza (z-score)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds == 0] = 1
    data_std = (data - means) / stds

    # Matrice covarianza e eigendecomposition
    cov_matrix = np.cov(data_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordina per eigenvalue decrescente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Proietta sui primi 2 componenti
    total_var = np.sum(eigenvalues)
    explained = (eigenvalues / total_var).tolist() if total_var > 0 else []
    projected = data_std @ eigenvectors[:, :2]

    return {
        "page_ids": pages,
        "feature_names": available_keys,
        "components": projected.tolist(),
        "explained_variance": [round(v, 4) for v in explained[:5]],
    }


@timer
def compute_anova(features: dict[str, dict], page_mapping: list[dict]) -> dict:
    """ANOVA per sezione: test se features differiscono tra sezioni."""
    from scipy.stats import f_oneway

    # Mapping page_id -> section
    page_to_section = {}
    for m in page_mapping:
        page_to_section[m["page_id"]] = m["section"]

    test_features = [
        "ink_density", "total_glyphs", "illustration_ratio", "glyph_density",
    ]

    results = {}
    for feat_name in test_features:
        groups = defaultdict(list)
        for page_id, feat in features.items():
            section = page_to_section.get(page_id)
            if section and section != "?" and feat_name in feat:
                groups[section].append(feat[feat_name])

        # Almeno 2 gruppi con >= 2 valori
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(valid_groups) >= 2:
            group_values = list(valid_groups.values())
            f_stat, p_value = f_oneway(*group_values)
            results[feat_name] = {
                "f_statistic": round(float(f_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": bool(p_value < 0.05),
                "groups": {k: len(v) for k, v in valid_groups.items()},
            }

    return results


@timer
def plot_correlation_matrix(correlations: dict, output_dir: Path):
    """Matrice di correlazione Pearson tra features."""
    names = correlations["feature_names"]
    matrix = np.array(correlations["matrix"])

    if len(names) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Matrice di correlazione features pagina")

    # Annota valori
    for i in range(len(names)):
        for j in range(len(names)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Correlazione Pearson")
    plt.tight_layout()
    plt.savefig(str(output_dir / "correlation_matrix.png"), dpi=150)
    plt.close()


@timer
def plot_pca_scatter(pca: dict, features: dict, page_mapping: list[dict],
                     output_dir: Path):
    """Scatter 2D delle pagine da PCA, colorato per sezione."""
    components = np.array(pca["components"])
    page_ids = pca["page_ids"]
    explained = pca["explained_variance"]

    if len(components) < 3 or components.shape[1] < 2:
        return

    # Mapping page_id -> section
    page_to_section = {}
    if page_mapping:
        for m in page_mapping:
            page_to_section[m["page_id"]] = m["section"]

    fig, ax = plt.subplots(figsize=(10, 8))

    if page_to_section:
        sections_found = sorted(set(
            page_to_section.get(p, "?") for p in page_ids
        ))
        for sec in sections_found:
            mask = [page_to_section.get(p, "?") == sec for p in page_ids]
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            color = SECTION_COLORS.get(sec, "#888888")
            label = f"{sec} ({SECTION_NAMES.get(sec, sec)})"
            ax.scatter(components[idx, 0], components[idx, 1],
                       c=color, label=label, alpha=0.6, s=30, edgecolors="k",
                       linewidths=0.3)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.scatter(components[:, 0], components[:, 1],
                   c="#8B4513", alpha=0.6, s=30)

    ev0 = f"{explained[0] * 100:.1f}" if explained else "?"
    ev1 = f"{explained[1] * 100:.1f}" if len(explained) > 1 else "?"
    ax.set_xlabel(f"PC1 ({ev0}% varianza)")
    ax.set_ylabel(f"PC2 ({ev1}% varianza)")
    ax.set_title("PCA pagine per features visive/testuali")
    plt.tight_layout()
    plt.savefig(str(output_dir / "page_feature_scatter.png"), dpi=150)
    plt.close()


@timer
def plot_feature_distributions(features: dict, page_mapping: list[dict],
                               output_dir: Path):
    """Box plot per sezione: distribuzione features per sezione."""
    if not page_mapping:
        return

    page_to_section = {}
    for m in page_mapping:
        page_to_section[m["page_id"]] = m["section"]

    plot_features = ["ink_density", "illustration_ratio", "total_glyphs", "glyph_density"]
    plot_labels = ["Ink density", "Illustration ratio", "Total glyphs", "Glyph density"]

    # Sezioni con dati
    sections_found = sorted(set(
        page_to_section.get(p, "?") for p in features.keys()
        if page_to_section.get(p, "?") != "?"
    ))

    if len(sections_found) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribuzione features per sezione", fontsize=14, fontweight="bold")

    for ax, feat_name, feat_label in zip(axes.flat, plot_features, plot_labels):
        data_by_section = []
        labels = []
        for sec in sections_found:
            values = [
                features[p].get(feat_name, 0)
                for p in features.keys()
                if page_to_section.get(p) == sec and feat_name in features[p]
            ]
            if values:
                data_by_section.append(values)
                labels.append(f"{sec}\n{SECTION_NAMES.get(sec, sec)}")

        if data_by_section:
            bp = ax.boxplot(data_by_section, labels=labels, patch_artist=True)
            colors = [SECTION_COLORS.get(sec, "#888888") for sec in sections_found
                      if any(features[p].get(feat_name) is not None
                             for p in features.keys()
                             if page_to_section.get(p) == sec)]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.set_title(feat_label)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(str(output_dir / "feature_distributions.png"), dpi=150)
    plt.close()


def run(config: ToolkitConfig, force: bool = False) -> None:
    """Entry point per la correlazione immagine-testo."""
    print_header("VOYNICH TOOLKIT - Correlazione Immagine-Testo")
    config.ensure_dirs()

    report_path = config.stats_dir / "image_text_correlation.json"
    if report_path.exists() and not force:
        print("  Report gia' presente, skip (usa --force per rieseguire)")
        return

    # Verifica pagine
    if not config.pages_dir.exists() or not list(config.pages_dir.glob("*.png")):
        raise click.ClickException(
            f"Pagine non trovate in {config.pages_dir}\n"
            "  Esegui prima: voynich extract"
        )

    # Elenca pagine disponibili
    page_files = sorted(config.pages_dir.glob("page_*.png"))
    page_ids = [f.stem for f in page_files]
    print(f"  {len(page_ids)} pagine trovate")

    # 1. Page mapping (opzionale)
    print_step("Costruzione mapping folio -> PDF page...")
    mapping_path = config.stats_dir / "page_mapping.json"
    page_mapping = []
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"

    if eva_file.exists():
        page_mapping = build_page_mapping(eva_file)
        with open(mapping_path, "w") as f:
            json.dump(page_mapping, f, indent=2)
        print(f"    Mapping: {len(page_mapping)} pagine fisiche")
    else:
        print("    File EVA non disponibile, skip mapping per sezione")
        if mapping_path.exists():
            with open(mapping_path) as f:
                page_mapping = json.load(f)

    # 2. Features visive
    print_step("Calcolo features visive...")
    visual_features = compute_visual_features(
        config.pages_dir, page_ids, config.binarize_threshold
    )
    print(f"    Features visive per {len(visual_features)} pagine")

    # 3. Features regioni
    print_step("Calcolo features regioni...")
    regions_path = config.text_regions_dir / "regions_catalog.json"
    region_features = {}
    if regions_path.exists():
        with open(regions_path) as f:
            regions_catalog = json.load(f)
        region_features = compute_region_features(regions_catalog)
        print(f"    Features regioni per {len(region_features)} pagine")
    else:
        print("    regions_catalog.json non trovato, skip features regioni")

    # 4. Features glifi
    print_step("Calcolo features glifi...")
    glyphs_path = config.glyphs_dir / "glyphs_catalog.json"
    glyph_features = {}
    if glyphs_path.exists():
        with open(glyphs_path) as f:
            glyphs_catalog = json.load(f)
        glyph_features = compute_glyph_features(glyphs_catalog)
        print(f"    Features glifi per {len(glyph_features)} pagine")
    else:
        print("    glyphs_catalog.json non trovato, skip features glifi")

    # 5. Merge features
    print_step("Merge e calcolo features derivate...")
    all_features = merge_features(visual_features, region_features, glyph_features)
    print(f"    Features totali per {len(all_features)} pagine")

    # 6. Correlazioni
    print_step("Calcolo matrice di correlazione...")
    correlations = compute_correlations(all_features)

    # 7. PCA
    print_step("Calcolo PCA...")
    pca = compute_pca(all_features)

    # 8. ANOVA (se mapping disponibile)
    anova_results = {}
    if page_mapping:
        print_step("Calcolo ANOVA per sezione...")
        anova_results = compute_anova(all_features, page_mapping)

    # 9. Visualizzazioni
    print_step("Generazione matrice correlazione...")
    plot_correlation_matrix(correlations, config.stats_dir)

    print_step("Generazione PCA scatter...")
    plot_pca_scatter(pca, all_features, page_mapping, config.stats_dir)

    if page_mapping:
        print_step("Generazione box plot per sezione...")
        plot_feature_distributions(all_features, page_mapping, config.stats_dir)

    # 10. Salva report
    print_step("Salvataggio report...")
    report = {
        "pages_analyzed": len(all_features),
        "features_per_page": list(all_features.values())[0] if all_features else {},
        "correlation": correlations,
        "pca": {
            "explained_variance": pca["explained_variance"],
            "n_pages": len(pca["page_ids"]),
        },
        "anova": anova_results,
        "mapping_available": bool(page_mapping),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Stampa sommario
    print("\n" + "=" * 60)
    print("  REPORT CORRELAZIONE IMMAGINE-TESTO")
    print("=" * 60)
    print(f"  Pagine analizzate:  {len(all_features)}")
    print(f"  Mapping sezione:    {'Si' if page_mapping else 'No'}")

    if correlations["feature_names"]:
        print(f"\n  Features correlate: {len(correlations['feature_names'])}")
        # Top correlazioni (esclusa diagonale)
        names = correlations["feature_names"]
        matrix = np.array(correlations["matrix"])
        top_corr = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                top_corr.append((names[i], names[j], matrix[i, j]))
        top_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n  Top 5 correlazioni:")
        for a, b, r in top_corr[:5]:
            print(f"    {a} <-> {b}: {r:.3f}")

    if pca["explained_variance"]:
        ev = pca["explained_variance"]
        print(f"\n  PCA varianza spiegata: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}"
              if len(ev) > 1 else f"\n  PCA varianza: PC1={ev[0]:.1%}")

    if anova_results:
        print(f"\n  ANOVA per sezione:")
        for feat, res in anova_results.items():
            sig = "*" if res["significant"] else ""
            print(f"    {feat}: F={res['f_statistic']:.2f}, p={res['p_value']:.4f} {sig}")

    print(f"\n  Report: {report_path}")
    print(f"  Grafici: {config.stats_dir}/")
