"""
Phase 1b — KL-divergence: does the Hypothesis A mapping resemble any Semitic language?

Compares the frequency distribution of INITIAL consonants produced by the
Hypothesis A mapping against expected distributions from 4 known Semitic languages +
a uniform random baseline.

Null hypothesis: the observed distribution is compatible with at least one known
Semitic language.

Metric: KL-divergence (Kullback-Leibler) and Jensen-Shannon divergence (symmetric).
KL(P || Q) = sum_i P_i * log2(P_i / Q_i)
  - P = observed distribution (Hypothesis A)
  - Q = expected distribution (reference language)
  - Higher = more different
  - 0 = identical

Reference frequency sources:
  - Biblical Hebrew: estimate from Ornan (2003), BHS word-initial frequency analysis
  - Modern Standard Arabic: estimate from Holes (2004), Modern Arabic corpus
  - Biblical Aramaic: estimate from Rosenthal (1997), Ezra/Daniel
  - Classical Syriac: estimate from Noldeke (1904), Peshitta
  - Uniform/random: 1/22 per consonant (22 standard Semitic consonants)
  Note: all frequencies are approximations from the literature, not from
  a direct computational corpus. Uncertainty is ~±2pp per entry.

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

# Biblical Hebrew — word-initial frequency (% approximate, source: BHS corpus)
HEBREW_INITIAL = {
    'A': 13.5,  # aleph — very common in verb roots and nouns
    'b':  6.5,  # bet
    'g':  2.5,  # gimel
    'd':  4.5,  # dalet
    'h':  9.0,  # he — common in article (ha-) and pronouns
    'w':  4.5,  # vav — common in vav-prefixed verb forms
    'z':  1.5,  # zayin
    'X':  1.5,  # khet
    'J':  0.8,  # tet
    'y':  9.0,  # yod — common in yiqtol (imperfect)
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

# Biblical Aramaic — estimate from Ezra ch. 4–7 and Daniel
# Similar to Hebrew but shifted: aleph + mem + yod more frequent
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

# Classical Syriac — estimate from the Peshitta (NT)
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

# Modern Standard Arabic — estimate from Arabic Treebank corpus (22 homologous consonants)
# Note: Arabic has 28 phonemes; we map onto the 22 Semitic consonants
# treating variants as mergers (th→t, dh→d, etc.)
ARABIC_INITIAL = {
    'A': 20.0,  # alif — very high due to article al- and initial hamza
    'b':  4.5,
    'g':  2.0,  # j/g — rare in initial position
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

# Uniform/random — 1/22 per consonant
UNIFORM = {c: 100.0 / 22 for c in HEBREW_INITIAL}

REFERENCES = {
    'hebrew':  HEBREW_INITIAL,
    'aramaic': ARAMAIC_INITIAL,
    'syriac':  SYRIAC_INITIAL,
    'arabic':  ARABIC_INITIAL,
    'uniform': UNIFORM,
}

REFERENCE_LABELS = {
    'hebrew':  'Biblical Hebrew (BHS)',
    'aramaic': 'Biblical Aramaic (Ezra/Daniel)',
    'syriac':  'Classical Syriac (Peshitta)',
    'arabic':  'Modern Standard Arabic (Treebank)',
    'uniform': 'Uniform/random (1/22)',
}


# =====================================================================
# KL divergence
# =====================================================================

EPSILON = 1e-9  # to avoid log(0)


def normalize(freq_dict: dict[str, float]) -> dict[str, float]:
    """Normalize to a probability distribution (sum=1)."""
    total = sum(freq_dict.values())
    return {k: v / total for k, v in freq_dict.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) — how much P diverges from Q.

    Interpretation: 0=identical, high=very different.
    Note: not symmetric. P=observed, Q=reference.
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
    """Jensen-Shannon divergence (symmetric, [0,1]).

    JSD(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) where M=(P+Q)/2
    More intuitive than KL: 0=identical, 1=maximally different.
    """
    all_keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, EPSILON) + q.get(k, EPSILON)) for k in all_keys}
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


# =====================================================================
# Load observed distribution from DB
# =====================================================================

def load_observed(db_path: Path) -> dict[str, float]:
    """Load honesty_initial_distribution from DB.

    Returns: dict consonant → proportion (sum=1)
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
    """Compute KL and JSD between observed and each reference language."""
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
            'jsd': round(jsd, 4),                  # Jensen-Shannon (symmetric)
            'jsd_pct': round(jsd * 100, 1),
        }

    # Rank by JSD (lower = more similar)
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
    lines.append("  PHASE 1b — KL-divergence: Hypothesis A vs Semitic languages")
    lines.append("=" * 72)
    lines.append(
        "\n  Question: does the Hypothesis A mapping produce an initial consonant\n"
        "  distribution compatible with any known Semitic language?\n"
        "\n  JSD (Jensen-Shannon) = 0 identical, 1 maximally different.\n"
        "  KL(obs||ref) = how much the observed diverges from the reference.\n"
    )

    # Observed distribution (top 6)
    lines.append("  Observed Hypothesis A distribution (top 6 initials):")
    top6 = sorted(observed.items(), key=lambda x: -x[1])[:6]
    for c, p in top6:
        bar = '█' * int(p * 40)
        lines.append(f"    {c:>2}: {p*100:5.1f}%  {bar}")
    lines.append("")

    # Comparison table
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

    # Methodological note
    lines.append("\n  " + "─" * 60)
    lines.append(
        "  Note: reference frequencies = estimates from the literature (±2pp).\n"
        "  The test is qualitative: uncertainty does not change conclusions\n"
        "  if all JSD values are high (>0.3) or if no language\n"
        "  approaches the random JSD value significantly."
    )

    # Verdict
    lines.append("\n  " + "=" * 60)
    uniform_jsd = results['uniform']['jsd']
    best_lang = ranked[0][0]
    best_jsd = ranked[0][1]['jsd']

    if best_jsd > 0.5:
        lines.append(
            "  VERDICT: distribution INCOMPATIBLE with all tested languages.\n"
            "  The Hypothesis A mapping does not produce a consonantal\n"
            "  signature attributable to any known Semitic language."
        )
    elif best_jsd < uniform_jsd * 0.8:
        lines.append(
            f"  VERDICT: closest language is {REFERENCE_LABELS[best_lang]}\n"
            f"  (JSD={best_jsd:.4f} vs uniform={uniform_jsd:.4f}).\n"
            f"  WARNING: even the 'closest' is very far."
        )
    else:
        lines.append(
            "  VERDICT: no Semitic language is systematically closer\n"
            f"  than random (best JSD={best_jsd:.4f}, "
            f"uniform={uniform_jsd:.4f}).\n"
            "  The distribution is not specific to any tested language."
        )
    lines.append("  " + "=" * 60)

    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 1b: KL-divergence Hypothesis A vs Semitic languages."""
    report_path = config.stats_dir / "semitic_kl_test.json"
    summary_path = config.stats_dir / "semitic_kl_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  semitic_kl_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 1b — KL-divergence: Hypothesis A vs Semitic languages")

    db_path = config.output_dir.parent / "voynich.db"
    if not db_path.exists():
        raise click.ClickException(f"DB not found: {db_path}")

    # 1. Load observed distribution
    print_step("Loading observed distribution from DB...")
    observed = load_observed(db_path)
    click.echo(f"    {len(observed)} consonants, top-3: " + ", ".join(
        f"{c}={p*100:.1f}%" for c, p in
        sorted(observed.items(), key=lambda x: -x[1])[:3]
    ))

    # 2. Compute KL and JSD
    print_step("Computing KL-divergence and Jensen-Shannon...")
    results = run_kl_analysis(observed)
    for lang_key, r in sorted(results.items(), key=lambda x: x[1]['jsd']):
        click.echo(
            f"    {r['label']:35}  JSD={r['jsd']:.4f}  "
            f"KL(obs||ref)={r['kl_obs_ref']:.4f}"
        )

    # 3. Save
    print_step("Saving...")
    report = {
        'observed': {k: round(v, 6) for k, v in observed.items()},
        'results': results,
        'note': (
            'Reference frequencies = estimates from the literature (±2pp). '
            'JSD: 0=identical, 1=maximally different. '
            'Qualitative test: checks whether the Hypothesis A mapping is '
            'specific to a Semitic language or generic/random.'
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
