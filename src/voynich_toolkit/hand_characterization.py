"""
Fase 0 — caratterizzazione fisica per mano (Davis 5 scribi).

Calcola per ogni mano (1–5, X, Y, ?):
  - n_pages, n_tokens (EVA raw), n_unique_types (EVA), n_hapax, TTR
  - distribuzione sezioni e lingue Currier
  - lunghezza media parola (EVA)
  - top-5 bigrammi EVA
  - entropia Shannon (distribuzione frequenze EVA)
  - slope Zipf (regressione log-log)

NOTA METODOLOGICA: nel file IVTFF il campo $H è assegnato a livello di folio
(riga di intestazione), non di linea. Non esistono "pagine con più mani"
rilevabili da questo dataset — limitazione strutturale dei dati da documentare.

Output:
  hand_characterization.json
  hand_characterization_summary.txt
  DB table: hand_characterization
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .full_decode import SECTION_NAMES
from .scribe_analysis import HAND_NAMES, split_corpus_by_hand
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Core statistics — pure EVA level (no lexicon, no decoding)
# =====================================================================

def eva_profile(words: list[str]) -> dict:
    """Compute structural profile from raw EVA word list.

    Returns dict with:
      n_tokens, n_unique, n_hapax, ttr, hapax_ratio,
      avg_word_length, shannon_entropy, zipf_slope,
      top5_bigrams
    """
    if not words:
        return {
            "n_tokens": 0, "n_unique": 0, "n_hapax": 0,
            "ttr": 0.0, "hapax_ratio": 0.0,
            "avg_word_length": 0.0,
            "shannon_entropy": 0.0, "zipf_slope": 0.0,
            "top5_bigrams": [],
        }

    freq: Counter[str] = Counter(words)
    n_tokens = len(words)
    n_unique = len(freq)
    n_hapax = sum(1 for c in freq.values() if c == 1)

    ttr = n_unique / n_tokens
    hapax_ratio = n_hapax / n_unique if n_unique else 0.0
    avg_len = float(np.mean([len(w) for w in words]))

    # Shannon entropy (bits) on word-type distribution
    probs = np.array([c / n_tokens for c in freq.values()], dtype=float)
    probs = probs[probs > 0]
    shannon = float(-np.sum(probs * np.log2(probs)))

    # Zipf slope (log-log regression on ranked word frequencies)
    sorted_freqs = sorted(freq.values(), reverse=True)
    if len(sorted_freqs) >= 5:
        ranks = np.log10(np.arange(1, len(sorted_freqs) + 1, dtype=float))
        freqs_log = np.log10(np.array(sorted_freqs, dtype=float))
        coeffs = np.polyfit(ranks, freqs_log, 1)
        zipf_slope = float(coeffs[0])
    else:
        zipf_slope = float("nan")

    # EVA character bigrams
    bigrams: Counter[str] = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bigrams[w[i:i+2]] += 1
    top5 = [{"bigram": bg, "count": c} for bg, c in bigrams.most_common(5)]

    return {
        "n_tokens": n_tokens,
        "n_unique": n_unique,
        "n_hapax": n_hapax,
        "ttr": round(ttr, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "avg_word_length": round(avg_len, 3),
        "shannon_entropy": round(shannon, 4),
        "zipf_slope": round(zipf_slope, 4) if math.isfinite(zipf_slope) else None,
        "top5_bigrams": top5,
    }


# =====================================================================
# Build full report dict
# =====================================================================

def build_report(corpus: dict) -> dict:
    """Build characterization report from split corpus.

    Args:
        corpus: output of split_corpus_by_hand()

    Returns:
        dict keyed by hand label with profile + corpus metadata
    """
    report = {}
    for hand in sorted(corpus.keys()):
        c = corpus[hand]
        words = c["words"]
        profile = eva_profile(words)
        report[hand] = {
            "hand_name": HAND_NAMES.get(hand, "?"),
            "n_pages": c["n_pages"],
            "sections": dict(c["sections"]),
            "languages": dict(c["languages"]),
            **profile,
        }
    return report


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(report: dict, db_path: Path) -> None:
    """Write hand_characterization table to SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS hand_characterization")
    cur.execute("""
        CREATE TABLE hand_characterization (
            hand           TEXT PRIMARY KEY,
            hand_name      TEXT,
            n_pages        INTEGER,
            n_tokens       INTEGER,
            n_unique       INTEGER,
            n_hapax        INTEGER,
            ttr            REAL,
            hapax_ratio    REAL,
            avg_word_length REAL,
            shannon_entropy REAL,
            zipf_slope     REAL,
            sections_json  TEXT,
            languages_json TEXT,
            top5_bigrams_json TEXT
        )
    """)

    rows = []
    for hand, d in sorted(report.items()):
        rows.append((
            hand,
            d["hand_name"],
            d["n_pages"],
            d["n_tokens"],
            d["n_unique"],
            d["n_hapax"],
            d["ttr"],
            d["hapax_ratio"],
            d["avg_word_length"],
            d["shannon_entropy"],
            d.get("zipf_slope"),
            json.dumps(d["sections"]),
            json.dumps(d["languages"]),
            json.dumps(d["top5_bigrams"]),
        ))

    cur.executemany("""
        INSERT INTO hand_characterization VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  FASE 0 — Caratterizzazione fisica per mano (Davis 5 scribi)")
    lines.append("=" * 80)

    lines.append("\n── Corpus Overview ──")
    lines.append(
        f"  {'Hand':>5}  {'Name':20}  {'Pages':>5}  {'Tokens':>7}  "
        f"{'Types':>6}  {'Hapax':>5}  {'TTR':>6}  {'Hapax%':>7}  "
        f"{'AvgLen':>6}  {'Langs':12}  Sections"
    )
    lines.append("  " + "-" * 100)

    for hand in sorted(report.keys()):
        d = report[hand]
        langs = " ".join(f"{l}:{n}" for l, n in
                         sorted(d["languages"].items(), key=lambda x: -x[1]))
        secs = " ".join(f"{SECTION_NAMES.get(s, s)[:3]}:{n}" for s, n in
                        sorted(d["sections"].items(), key=lambda x: -x[1]))
        lines.append(
            f"  {hand:>5}  {d['hand_name']:20}  {d['n_pages']:5d}  "
            f"{d['n_tokens']:7,}  {d['n_unique']:6,}  {d['n_hapax']:5,}  "
            f"{d['ttr']:6.3f}  {d['hapax_ratio']*100:6.1f}%  "
            f"{d['avg_word_length']:6.2f}  {langs:12}  {secs}"
        )

    lines.append("\n── Entropia Shannon e Slope Zipf ──")
    lines.append(
        f"  {'Hand':>5}  {'Name':20}  {'H(word)':>8}  {'Zipf slope':>10}  "
        f"  Top-3 bigrammi EVA"
    )
    lines.append("  " + "-" * 70)
    for hand in sorted(report.keys()):
        d = report[hand]
        if d["n_tokens"] < 50:
            continue
        zslope = (f"{d['zipf_slope']:.3f}" if d["zipf_slope"] is not None
                  else "   n/a")
        bgs = " ".join(b["bigram"] for b in d["top5_bigrams"][:3])
        lines.append(
            f"  {hand:>5}  {d['hand_name']:20}  {d['shannon_entropy']:8.4f}  "
            f"{zslope:>10}    {bgs}"
        )

    lines.append("\n── Nota metodologica ──")
    lines.append(
        "  Il campo $H nel file IVTFF è assegnato a livello di folio (riga di\n"
        "  intestazione), non di linea. Non esistono quindi 'pagine con più\n"
        "  mani' rilevabili da questo dataset. Hands X, Y, ? = non attribuiti."
    )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Fase 0: caratterizzazione fisica per mano — hapax, TTR, entropia, Zipf."""
    report_path = config.stats_dir / "hand_characterization.json"
    summary_path = config.stats_dir / "hand_characterization_summary.txt"

    if report_path.exists() and not force:
        click.echo("  hand_characterization report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("FASE 0 — Caratterizzazione Fisica per Mano")

    # 1. Parse EVA corpus
    print_step("Parsing EVA corpus...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} parole, {len(pages)} pagine")

    # 2. Split by hand
    print_step("Suddivisione per mano...")
    corpus = split_corpus_by_hand(pages)
    for hand in sorted(corpus.keys()):
        c = corpus[hand]
        click.echo(
            f"    Mano {hand:>2} ({HAND_NAMES.get(hand,'?')}): "
            f"{c['n_pages']} pagine, {len(c['words']):,} parole  "
            f"langs={dict(c['languages'])}  secs={dict(c['sections'])}"
        )

    # 3. Compute EVA profiles
    print_step("Calcolo profili EVA (hapax, TTR, entropia, Zipf, bigrammi)...")
    report = build_report(corpus)
    for hand in sorted(report.keys()):
        d = report[hand]
        click.echo(
            f"    Mano {hand}: tokens={d['n_tokens']:,}  "
            f"types={d['n_unique']:,}  hapax={d['n_hapax']:,}  "
            f"TTR={d['ttr']:.3f}  H={d['shannon_entropy']:.3f}bits  "
            f"Zipf={d['zipf_slope']}"
        )

    # 4. Save JSON
    print_step("Salvataggio JSON...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    # 5. Save TXT summary
    summary = format_summary(report)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    # 6. Save to DB
    print_step("Scrittura DB table hand_characterization...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(report, db_path)
        click.echo(f"    {db_path} ✓")
    else:
        click.echo(f"    WARN: DB non trovato a {db_path} — skip DB write")

    click.echo(f"\n{summary}")
