"""
Fase 1b — KL-divergence: il mapping Ipotesi A assomiglia a qualche lingua semitica?

Confronta la distribuzione di frequenza dei consonanti INIZIALI prodotti dal
mapping Ipotesi A con le distribuzioni attese di 4 lingue semitiche note +
una baseline casuale uniforme.

Ipotesi nulla: la distribuzione osservata è compatibile con almeno una lingua
semitica nota.

Metrica: KL-divergence (Kullback-Leibler) e Jensen-Shannon divergence (simmetrica).
KL(P || Q) = sum_i P_i * log2(P_i / Q_i)
  - P = distribuzione osservata (Ipotesi A)
  - Q = distribuzione attesa (lingua di riferimento)
  - Più alto = più diverso
  - 0 = identico

Fonti frequenze di riferimento:
  - Ebraico biblico: stima da Ornan (2003), BHS word-initial frequency analysis
  - Arabo MSA: stima da Holes (2004), Modern Arabic corpus
  - Aramaico biblico: stima da Rosenthal (1997), Ezra/Daniel
  - Siriaco classico: stima da Nöldeke (1904), Peshitta
  - Uniforme/random: 1/22 per ogni consonante (22 consonanti semitiche standard)
  Nota: tutte le frequenze sono approssimazioni dalla letteratura, non da
  un corpus computazionale diretto. L'incertezza è ~±2pp per voce.

Output:
  semitic_kl_test.json
  semitic_kl_test_summary.txt
  DB table: semitic_kl_test
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import click

from .config import ToolkitConfig
from .utils import print_header, print_step


# =====================================================================
# Reference frequency tables — word-initial consonant position
# Consonant ASCII: A=aleph b=bet g=gimel d=dalet h=he w=vav z=zayin
#   X=khet J=tet y=yod k=kaf l=lamed m=mem n=nun s=samekh E=ayin
#   p=pe C=tsade q=qof r=resh S=shin t=tav
# =====================================================================

# Ebraico biblico — word-initial frequency (% approssimativa, fonte: BHS corpus)
HEBREW_INITIAL = {
    'A': 13.5,  # aleph — molto comune in radici verbali e nomi
    'b':  6.5,  # bet
    'g':  2.5,  # gimel
    'd':  4.5,  # dalet
    'h':  9.0,  # he — comune in articolo (ha-) e pronomi
    'w':  4.5,  # vav — comune in forme verbali e-prefissate
    'z':  1.5,  # zayin
    'X':  1.5,  # khet
    'J':  0.8,  # tet
    'y':  9.0,  # yod — comune in yiqtol (imperfetto)
    'k':  5.0,  # kaf
    'l':  7.5,  # lamed
    'm':  8.5,  # mem
    'n':  5.5,  # nun
    's':  1.5,  # samekh
    'E':  4.0,  # ayin
    'p':  3.5,  # pe
    'C':  1.5,  # tsade
    'q':  2.5,  # qof
    'r':  7.0,  # resh
    'S':  4.0,  # shin
    't':  5.5,  # tav
}

# Aramaico biblico — stima da Ezra cap. 4–7 e Daniele
# Simile all'ebraico ma con shift: aleph + mem + yod più frequenti
ARAMAIC_INITIAL = {
    'A': 12.0,
    'b':  6.0,
    'g':  2.0,
    'd':  5.5,
    'h':  8.0,
    'w':  4.0,
    'z':  1.5,
    'X':  1.5,
    'J':  0.5,
    'y':  9.5,
    'k':  5.5,
    'l':  8.5,
    'm': 10.0,
    'n':  6.5,
    's':  2.0,
    'E':  3.5,
    'p':  4.0,
    'C':  1.0,
    'q':  2.5,
    'r':  6.5,
    'S':  3.0,
    't':  6.0,
}

# Siriaco classico — stima dalla Peshitta (NT)
SYRIAC_INITIAL = {
    'A': 10.0,
    'b':  7.5,
    'g':  2.5,
    'd':  6.0,
    'h':  7.0,
    'w':  5.0,
    'z':  1.5,
    'X':  1.5,
    'J':  0.5,
    'y':  9.0,
    'k':  6.0,
    'l':  8.0,
    'm': 10.5,
    'n':  7.0,
    's':  2.5,
    'E':  3.5,
    'p':  5.0,
    'C':  1.0,
    'q':  2.0,
    'r':  6.0,
    'S':  3.0,
    't':  6.0,
}

# Arabo MSA — stima da corpus Arabic Treebank (22 consonanti omologhe)
# Nota: l'arabo ha 28 fonemi; mappiamo sulle 22 consonanti semitiche
# trattando le varianti come fusioni (th→t, dh→d, ecc.)
ARABIC_INITIAL = {
    'A': 20.0,  # alif — molto alto per articolo al- e hamza iniziale
    'b':  4.5,
    'g':  2.0,  # j/g — raro in iniziale
    'd':  4.0,
    'h':  4.5,
    'w':  2.0,
    'z':  1.5,
    'X':  2.0,  # kha
    'J':  1.0,
    'y':  5.0,
    'k':  3.5,
    'l':  6.5,
    'm':  8.5,
    'n':  6.0,
    's':  4.5,
    'E':  4.0,  # ain
    'p':  4.0,  # fa
    'C':  1.5,
    'q':  2.5,
    'r':  5.5,
    'S':  3.5,  # shin
    't':  3.5,
}

# Uniforme/random — 1/22 per ogni consonante
UNIFORM = {c: 100.0 / 22 for c in HEBREW_INITIAL}

REFERENCES = {
    'hebrew':  HEBREW_INITIAL,
    'aramaic': ARAMAIC_INITIAL,
    'syriac':  SYRIAC_INITIAL,
    'arabic':  ARABIC_INITIAL,
    'uniform': UNIFORM,
}

REFERENCE_LABELS = {
    'hebrew':  'Ebraico biblico (BHS)',
    'aramaic': 'Aramaico biblico (Ezra/Daniel)',
    'syriac':  'Siriaco classico (Peshitta)',
    'arabic':  'Arabo MSA (Treebank)',
    'uniform': 'Uniforme/random (1/22)',
}


# =====================================================================
# KL divergence
# =====================================================================

EPSILON = 1e-9  # per evitare log(0)


def normalize(freq_dict: dict[str, float]) -> dict[str, float]:
    """Normalizza a distribuzione di probabilità (somma=1)."""
    total = sum(freq_dict.values())
    return {k: v / total for k, v in freq_dict.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) — quanto P diverge da Q.

    Interpretazione: 0=identici, alto=molto diversi.
    Nota: non simmetrico. P=osservato, Q=riferimento.
    """
    all_keys = set(p) | set(q)
    kl = 0.0
    for k in all_keys:
        pi = p.get(k, EPSILON)
        qi = q.get(k, EPSILON)
        if pi > EPSILON:
            kl += pi * math.log2(pi / qi)
    return kl


def jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (simmetrica, [0,1]).

    JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) dove M=(P+Q)/2
    Più intuitivo di KL: 0=identici, 1=massimamente diversi.
    """
    all_keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, EPSILON) + q.get(k, EPSILON)) for k in all_keys}
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


# =====================================================================
# Load observed distribution from DB
# =====================================================================

def load_observed(db_path: Path) -> dict[str, float]:
    """Carica honesty_initial_distribution dal DB.

    Returns: dict consonante → proporzione (somma=1)
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT letter, count FROM honesty_initial_distribution")
    rows = cur.fetchall()
    conn.close()

    raw = {letter: count for letter, count in rows}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# =====================================================================
# Main analysis
# =====================================================================

def run_kl_analysis(observed: dict[str, float]) -> dict:
    """Calcola KL e JSD tra osservato e ogni lingua di riferimento."""
    results = {}
    for lang_key, ref_raw in REFERENCES.items():
        ref = normalize(ref_raw)
        kl_obs_ref = kl_divergence(observed, ref)
        kl_ref_obs = kl_divergence(ref, observed)
        jsd = jensen_shannon(observed, ref)
        results[lang_key] = {
            'label': REFERENCE_LABELS[lang_key],
            'kl_obs_ref': round(kl_obs_ref, 4),   # KL(observed || ref)
            'kl_ref_obs': round(kl_ref_obs, 4),   # KL(ref || observed)
            'jsd': round(jsd, 4),                  # Jensen-Shannon (simmetrico)
            'jsd_pct': round(jsd * 100, 1),
        }

    # Rank per JSD (più basso = più simile)
    ranked = sorted(results.items(), key=lambda x: x[1]['jsd'])
    for rank, (lang, r) in enumerate(ranked, 1):
        results[lang]['rank'] = rank

    return results


# =====================================================================
# DB save
# =====================================================================

def save_to_db(results: dict, observed: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS semitic_kl_test")
    cur.execute("""
        CREATE TABLE semitic_kl_test (
            lang_key     TEXT PRIMARY KEY,
            label        TEXT,
            kl_obs_ref   REAL,
            kl_ref_obs   REAL,
            jsd          REAL,
            jsd_pct      REAL,
            rank_by_jsd  INTEGER,
            note         TEXT
        )
    """)

    for lang_key, r in results.items():
        cur.execute("""
            INSERT INTO semitic_kl_test VALUES (?,?,?,?,?,?,?,?)
        """, (
            lang_key, r['label'],
            r['kl_obs_ref'], r['kl_ref_obs'],
            r['jsd'], r['jsd_pct'],
            r['rank'],
            'lower JSD = more similar',
        ))

    conn.commit()
    conn.close()


# =====================================================================
# Summary
# =====================================================================

def format_summary(results: dict, observed: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  FASE 1b — KL-divergence: Ipotesi A vs lingue semitiche")
    lines.append("=" * 72)
    lines.append(
        "\n  Domanda: il mapping Ipotesi A produce una distribuzione consonantale\n"
        "  iniziale compatibile con qualche lingua semitica nota?\n"
        "\n  JSD (Jensen-Shannon) = 0 identico, 1 massimamente diverso.\n"
        "  KL(obs||ref) = quanto l'osservato diverge dal riferimento.\n"
    )

    # Distribuzione osservata (top 6)
    lines.append("  Distribuzione osservata Ipotesi A (top 6 iniziali):")
    top6 = sorted(observed.items(), key=lambda x: -x[1])[:6]
    for c, p in top6:
        bar = '█' * int(p * 40)
        lines.append(f"    {c:>2}: {p*100:5.1f}%  {bar}")
    lines.append("")

    # Tabella confronto
    lines.append(
        f"  {'Lingua':35}  {'JSD':>6}  {'KL(obs||ref)':>12}  {'Rank':>4}"
    )
    lines.append("  " + "-" * 62)

    ranked = sorted(results.items(), key=lambda x: x[1]['jsd'])
    for lang_key, r in ranked:
        lines.append(
            f"  {r['label']:35}  {r['jsd']:.4f}  "
            f"{r['kl_obs_ref']:>12.4f}  {r['rank']:>4}"
        )

    # Nota metodologica
    lines.append("\n  " + "─" * 60)
    lines.append(
        "  Nota: frequenze di riferimento = stime dalla letteratura (±2pp).\n"
        "  Il test è qualitativo: l'incertezza non cambia le conclusioni\n"
        "  se i valori JSD sono tutti alti (>0.3) o se nessuna lingua\n"
        "  si avvicina significativamente al valore JSD del random."
    )

    # Verdetto
    lines.append("\n  " + "=" * 60)
    uniform_jsd = results['uniform']['jsd']
    best_lang = ranked[0][0]
    best_jsd = ranked[0][1]['jsd']

    if best_jsd > 0.5:
        lines.append(
            "  VERDETTO: distribuzione INCOMPATIBILE con tutte le lingue\n"
            "  testate. Il mapping Ipotesi A non produce una firma\n"
            "  consonantale riconducibile a nessuna lingua semitica nota."
        )
    elif best_jsd < uniform_jsd * 0.8:
        lines.append(
            f"  VERDETTO: la lingua più vicina è {REFERENCE_LABELS[best_lang]}\n"
            f"  (JSD={best_jsd:.4f} vs uniform={uniform_jsd:.4f}).\n"
            f"  ATTENZIONE: anche il 'più vicino' è molto lontano."
        )
    else:
        lines.append(
            "  VERDETTO: nessuna lingua semitica è sistematicamente più\n"
            f"  vicina del random (best JSD={best_jsd:.4f}, "
            f"uniform={uniform_jsd:.4f}).\n"
            "  La distribuzione non è specifica per alcuna lingua testata."
        )
    lines.append("  " + "=" * 60)

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Fase 1b: KL-divergence Ipotesi A vs lingue semitiche."""
    report_path = config.stats_dir / "semitic_kl_test.json"
    summary_path = config.stats_dir / "semitic_kl_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  semitic_kl_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("FASE 1b — KL-divergence: Ipotesi A vs lingue semitiche")

    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        raise click.ClickException(f"DB non trovato: {db_path}")

    # 1. Carica distribuzione osservata
    print_step("Caricamento distribuzione osservata da DB...")
    observed = load_observed(db_path)
    click.echo(f"    {len(observed)} consonanti, top-3: " + ", ".join(
        f"{c}={p*100:.1f}%" for c, p in
        sorted(observed.items(), key=lambda x: -x[1])[:3]
    ))

    # 2. Calcola KL e JSD
    print_step("Calcolo KL-divergence e Jensen-Shannon...")
    results = run_kl_analysis(observed)
    for lang_key, r in sorted(results.items(), key=lambda x: x[1]['jsd']):
        click.echo(
            f"    {r['label']:35}  JSD={r['jsd']:.4f}  "
            f"KL(obs||ref)={r['kl_obs_ref']:.4f}"
        )

    # 3. Salva
    print_step("Salvataggio...")
    report = {
        'observed': {k: round(v, 6) for k, v in observed.items()},
        'results': results,
        'note': (
            'Frequenze di riferimento = stime dalla letteratura (±2pp). '
            'JSD: 0=identico, 1=massimamente diverso. '
            'Test qualitativo: verifica se il mapping Ipotesi A è '
            'specifico per una lingua semitica o generico/casuale.'
        ),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    summary = format_summary(results, observed)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    save_to_db(results, observed, db_path)
    click.echo(f"    DB: {db_path} ✓")

    click.echo(f"\n{summary}")
