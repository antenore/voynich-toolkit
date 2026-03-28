"""
Phase 7e — ckh allograph test.

Phase 7d showed that ckh behaves differently from cph/cth/cfh:
  - Only 21.5% prefix vs 45-60% for the others
  - Mean char position 0.48 (near-random) vs 0.22-0.32
  - Slight complementary distribution with k (z=-2.84)
  - 42.3% of all split gallows (908/2149)

This module tests a specific hypothesis: is ckh an allograph of k?

Four discriminating tests:

  7e-1 — Folio/section distribution: correlation between ckh and k rates per folio.
         If allographs, they should appear in the same folios. Compare with
         cph-p, cth-t as controls.

  7e-2 — Bigram context overlap: do ckh and k appear between the same characters?
         If allographs, the left-right character context should be similar.
         Measure Jaccard and cosine on context bigrams.

  7e-3 — Paragraph position profile: compare ckh and k distribution across
         para_start / para_cont / para_end. Compare with cph-p, cth-t.

  7e-4 — Substitution test: replace all ckh with k in the corpus. Re-compute
         key Phase 7a/7b metrics (cross-boundary words, 'm' end-marker, gallows
         paragraph markers). If ckh is truly an allograph, structural properties
         should be preserved or improved.

Output:
  ckh_allograph_test.json
  ckh_allograph_test_summary.txt
  DB table: ckh_allograph_test
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import click
import numpy as np

from .config import ToolkitConfig
from .currier_line_test import SPLIT_GALLOWS, parse_ivtff_lines
from .full_decode import SECTION_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SEED = 42
N_NULL = 500

# All four pairs
SG_TO_SIMPLE = {"cth": "t", "ckh": "k", "cph": "p", "cfh": "f"}

# Section metadata per folio (populated at runtime)
_FOLIO_SECTION: dict[str, str] = {}


# =====================================================================
# Helpers
# =====================================================================

def _count_char_in_words(words: list[str], char: str,
                         exclude_sg: bool = True) -> int:
    """Count occurrences of a simple gallows character in words,
    excluding occurrences that are part of split gallows."""
    count = 0
    for w in words:
        text = w
        if exclude_sg:
            for sg in sorted(SPLIT_GALLOWS, key=len, reverse=True):
                text = text.replace(sg, "\x00" * len(sg))
        count += text.count(char)
    return count


def _get_context_bigrams(lines: list[dict], target: str,
                          is_trigram: bool = False) -> Counter:
    """Extract left-right character context around target in all words.

    For a trigram like 'ckh', context is (char_before, char_after).
    For a single char like 'k', we first mask split gallows, then find
    standalone k and get context.

    Returns Counter of (left_char, right_char) tuples.
    '^' = word start, '$' = word end.
    """
    contexts: Counter = Counter()

    for line in lines:
        for word in line["words"]:
            if is_trigram:
                # Find all occurrences of the trigram
                start = 0
                while True:
                    idx = word.find(target, start)
                    if idx == -1:
                        break
                    left = word[idx - 1] if idx > 0 else "^"
                    right_idx = idx + len(target)
                    right = word[right_idx] if right_idx < len(word) else "$"
                    contexts[(left, right)] += 1
                    start = idx + 1
            else:
                # Mask split gallows first
                masked = word
                for sg in sorted(SPLIT_GALLOWS, key=len, reverse=True):
                    masked = masked.replace(sg, "\x00" * len(sg))
                # Find standalone character
                for i, ch in enumerate(masked):
                    if ch == target:
                        left = word[i - 1] if i > 0 else "^"
                        right = word[i + 1] if i + 1 < len(word) else "$"
                        contexts[(left, right)] += 1

    return contexts


def _cosine_sim(c1: Counter, c2: Counter) -> float:
    """Cosine similarity between two Counters."""
    keys = set(c1.keys()) | set(c2.keys())
    if not keys:
        return 0.0
    a = np.array([c1.get(k, 0) for k in keys], dtype=float)
    b = np.array([c2.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def _jaccard_keys(c1: Counter, c2: Counter) -> float:
    """Jaccard similarity on the key sets (ignoring counts)."""
    s1 = set(c1.keys())
    s2 = set(c2.keys())
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def z_score(observed: float, null_mean: float, null_std: float) -> float | None:
    if null_std < 1e-10:
        return None
    return (observed - null_mean) / null_std


# =====================================================================
# 7e-1 — Folio/section distribution
# =====================================================================

def test_folio_distribution(lines: list[dict], pages: list[dict]) -> dict:
    """Compare per-folio rate of ckh vs k, and do the same for cph-p and cth-t.

    Compute Pearson correlation of rates across folios.
    If ckh is allograph of k, we expect positive correlation.
    """
    # Build folio section map
    for p in pages:
        _FOLIO_SECTION[p["folio"]] = p.get("section", "?")

    # Count per folio: split gallows and simple gallows
    folio_counts: dict[str, dict] = {}

    for line in lines:
        folio = line["folio"]
        if folio not in folio_counts:
            folio_counts[folio] = {
                "n_words": 0,
                "ckh": 0, "k": 0,
                "cth": 0, "t": 0,
                "cph": 0, "p": 0,
                "cfh": 0, "f": 0,
            }
        fc = folio_counts[folio]
        words = line["words"]
        fc["n_words"] += len(words)

        text = " ".join(words)
        for sg in SG_TO_SIMPLE:
            fc[sg] += text.count(sg)

        for word in words:
            fc["k"] += _count_char_in_words([word], "k")
            fc["t"] += _count_char_in_words([word], "t")
            fc["p"] += _count_char_in_words([word], "p")
            fc["f"] += _count_char_in_words([word], "f")

    # Compute rates and correlations for each pair
    results = {}
    for sg, simple in SG_TO_SIMPLE.items():
        sg_rates = []
        simple_rates = []
        folios_used = []

        for folio, fc in sorted(folio_counts.items()):
            if fc["n_words"] < 10:
                continue
            sg_rates.append(fc[sg] / fc["n_words"])
            simple_rates.append(fc[simple] / fc["n_words"])
            folios_used.append(folio)

        sg_arr = np.array(sg_rates)
        simple_arr = np.array(simple_rates)

        if len(sg_arr) > 2 and np.std(sg_arr) > 0 and np.std(simple_arr) > 0:
            corr = float(np.corrcoef(sg_arr, simple_arr)[0, 1])
        else:
            corr = None

        # Section breakdown
        section_counts: dict[str, dict] = {}
        for folio, fc in folio_counts.items():
            sec = _FOLIO_SECTION.get(folio, "?")
            if sec not in section_counts:
                section_counts[sec] = {sg: 0, simple: 0, "n_words": 0}
            section_counts[sec][sg] += fc[sg]
            section_counts[sec][simple] += fc[simple]
            section_counts[sec]["n_words"] += fc["n_words"]

        results[f"{sg}_vs_{simple}"] = {
            "correlation": round(corr, 4) if corr is not None else None,
            "n_folios": len(folios_used),
            "total_sg": int(sum(fc[sg] for fc in folio_counts.values())),
            "total_simple": int(sum(fc[simple] for fc in folio_counts.values())),
            "by_section": {
                sec: {
                    f"rate_{sg}": round(d[sg] / d["n_words"], 4) if d["n_words"] > 0 else 0,
                    f"rate_{simple}": round(d[simple] / d["n_words"], 4) if d["n_words"] > 0 else 0,
                    "n_words": d["n_words"],
                }
                for sec, d in sorted(section_counts.items())
            },
        }

    return results


# =====================================================================
# 7e-2 — Bigram context overlap
# =====================================================================

def test_bigram_context(lines: list[dict]) -> dict:
    """Compare left-right character context of ckh vs k.

    Also compute for cph-p, cth-t, cfh-f as controls.
    """
    results = {}

    for sg, simple in SG_TO_SIMPLE.items():
        ctx_sg = _get_context_bigrams(lines, sg, is_trigram=True)
        ctx_simple = _get_context_bigrams(lines, simple, is_trigram=False)

        cosine = _cosine_sim(ctx_sg, ctx_simple)
        jaccard = _jaccard_keys(ctx_sg, ctx_simple)

        # Top contexts
        top_sg = ctx_sg.most_common(10)
        top_simple = ctx_simple.most_common(10)

        results[f"{sg}_vs_{simple}"] = {
            "cosine_similarity": round(cosine, 4),
            "jaccard_similarity": round(jaccard, 4),
            "n_contexts_sg": len(ctx_sg),
            "n_contexts_simple": len(ctx_simple),
            "n_shared_contexts": len(set(ctx_sg.keys()) & set(ctx_simple.keys())),
            "total_sg_occurrences": sum(ctx_sg.values()),
            "total_simple_occurrences": sum(ctx_simple.values()),
            "top_sg_contexts": [
                {"context": f"{l}_{sg}_{r}", "count": c}
                for (l, r), c in top_sg
            ],
            "top_simple_contexts": [
                {"context": f"{l}_{simple}_{r}", "count": c}
                for (l, r), c in top_simple
            ],
        }

    return results


def null_bigram_context(lines: list[dict], n_perms: int = N_NULL,
                        seed: int = SEED) -> dict:
    """Null model: shuffle which words contain the target (within folio).

    Computes cosine similarity distribution under null for each pair.
    """
    rng = random.Random(seed)

    # Group lines by folio
    by_folio: dict[str, list[dict]] = {}
    for line in lines:
        by_folio.setdefault(line["folio"], []).append(line)

    null_cosines: dict[str, list[float]] = {
        f"{sg}_vs_{simple}": [] for sg, simple in SG_TO_SIMPLE.items()
    }

    for _ in range(n_perms):
        # Shuffle words within each folio
        shuffled_lines = []
        for folio, flines in by_folio.items():
            all_words = [w for line in flines for w in line["words"]]
            rng.shuffle(all_words)
            idx = 0
            for line in flines:
                n = len(line["words"])
                shuffled_lines.append({
                    "folio": folio,
                    "words": all_words[idx:idx + n],
                    "para_type": line.get("para_type", "other"),
                })
                idx += n

        for sg, simple in SG_TO_SIMPLE.items():
            ctx_sg = _get_context_bigrams(shuffled_lines, sg, is_trigram=True)
            ctx_simple = _get_context_bigrams(shuffled_lines, simple, is_trigram=False)
            cos = _cosine_sim(ctx_sg, ctx_simple)
            null_cosines[f"{sg}_vs_{simple}"].append(cos)

    results = {}
    for key, nulls in null_cosines.items():
        results[key] = {
            "null_mean": round(float(np.mean(nulls)), 4),
            "null_std": round(float(np.std(nulls, ddof=1)), 6),
        }

    return results


# =====================================================================
# 7e-3 — Paragraph position profile
# =====================================================================

def test_para_position(lines: list[dict]) -> dict:
    """Compare distribution across para_start/para_cont/para_end for
    ckh vs k, and for control pairs cph-p, cth-t, cfh-f.

    If ckh is allograph of k, their paragraph position profiles should
    be similar (measured by chi-square and cosine).
    """
    results = {}

    for sg, simple in SG_TO_SIMPLE.items():
        sg_by_pos = Counter()
        simple_by_pos = Counter()

        for line in lines:
            pt = line.get("para_type", "other")
            if pt not in ("para_start", "para_cont", "para_end"):
                continue
            words = line["words"]
            text = " ".join(words)
            sg_count = text.count(sg)
            simple_count = _count_char_in_words(words, simple)

            if sg_count > 0:
                sg_by_pos[pt] += sg_count
            if simple_count > 0:
                simple_by_pos[pt] += simple_count

        # Normalize to fractions
        sg_total = sum(sg_by_pos.values())
        simple_total = sum(simple_by_pos.values())

        positions = ["para_start", "para_cont", "para_end"]
        sg_frac = [sg_by_pos.get(p, 0) / sg_total if sg_total > 0 else 0
                   for p in positions]
        simple_frac = [simple_by_pos.get(p, 0) / simple_total if simple_total > 0 else 0
                       for p in positions]

        # Cosine between position profiles
        a = np.array(sg_frac)
        b = np.array(simple_frac)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denom) if denom > 0 else 0.0

        results[f"{sg}_vs_{simple}"] = {
            "sg_distribution": {p: sg_by_pos.get(p, 0) for p in positions},
            "sg_fractions": {p: round(f, 4) for p, f in zip(positions, sg_frac)},
            "simple_distribution": {p: simple_by_pos.get(p, 0) for p in positions},
            "simple_fractions": {p: round(f, 4) for p, f in zip(positions, simple_frac)},
            "sg_total": sg_total,
            "simple_total": simple_total,
            "cosine_position_profile": round(cosine, 4),
        }

    return results


# =====================================================================
# 7e-4 — Substitution test
# =====================================================================

def test_substitution(lines: list[dict]) -> dict:
    """Replace all ckh -> k in corpus. Re-compute key Phase 7a/7b metrics.

    Tests:
    - Cross-boundary words (Phase 7a): count of words spanning line boundaries
    - 'm' line-end rate (Phase 7a)
    - Simple gallows at paragraph start (Phase 7b): with k now including former ckh

    Compare original vs substituted.
    """
    # Original metrics
    orig = _compute_structural_metrics(lines, substitution=None)
    # ckh -> k substitution
    sub_ckh = _compute_structural_metrics(lines, substitution=("ckh", "k"))
    # Control: cph -> p substitution (should change things more since cph is a real prefix)
    sub_cph = _compute_structural_metrics(lines, substitution=("cph", "p"))

    return {
        "original": orig,
        "ckh_to_k": sub_ckh,
        "cph_to_p": sub_cph,
        "ckh_delta": {
            k: round(sub_ckh[k] - orig[k], 6) for k in orig
            if isinstance(orig[k], (int, float))
        },
        "cph_delta": {
            k: round(sub_cph[k] - orig[k], 6) for k in orig
            if isinstance(orig[k], (int, float))
        },
    }


def _compute_structural_metrics(lines: list[dict],
                                 substitution: tuple[str, str] | None = None) -> dict:
    """Compute Phase 7a/7b-like metrics, optionally after substitution."""
    # Apply substitution
    processed = []
    for line in lines:
        words = line["words"]
        if substitution:
            words = [w.replace(substitution[0], substitution[1]) for w in words]
        processed.append({**line, "words": words})

    # 'm' end rate
    n_lines = 0
    n_m_final = 0
    for line in processed:
        words = line["words"]
        if not words:
            continue
        n_lines += 1
        last_word = words[-1]
        if last_word and last_word[-1] == "m":
            n_m_final += 1

    m_rate = n_m_final / n_lines if n_lines > 0 else 0

    # Simple gallows at paragraph start rate
    simple_gallows = {"t", "k", "p", "f"}
    n_para_start = 0
    n_sg_at_start = 0
    n_para_cont = 0
    n_sg_at_cont = 0

    for line in processed:
        pt = line.get("para_type", "other")
        words = line["words"]
        if not words:
            continue
        first_word = words[0]
        has_simple = any(c in simple_gallows for c in first_word)

        if pt == "para_start":
            n_para_start += 1
            if has_simple:
                n_sg_at_start += 1
        elif pt == "para_cont":
            n_para_cont += 1
            if has_simple:
                n_sg_at_cont += 1

    sg_start_rate = n_sg_at_start / n_para_start if n_para_start > 0 else 0
    sg_cont_rate = n_sg_at_cont / n_para_cont if n_para_cont > 0 else 0

    # Vocabulary size
    all_words = [w for line in processed for w in line["words"]]
    n_types = len(set(all_words))
    n_tokens = len(all_words)

    return {
        "n_lines": n_lines,
        "m_end_rate": round(m_rate, 4),
        "sg_start_rate": round(sg_start_rate, 4),
        "sg_cont_rate": round(sg_cont_rate, 4),
        "sg_start_minus_cont": round(sg_start_rate - sg_cont_rate, 4),
        "n_types": n_types,
        "n_tokens": n_tokens,
    }


# =====================================================================
# DB persistence
# =====================================================================

def save_to_db(results: dict, db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS ckh_allograph_test")
    cur.execute("""
        CREATE TABLE ckh_allograph_test (
            test           TEXT,
            key            TEXT,
            value          REAL,
            detail_json    TEXT,
            PRIMARY KEY (test, key)
        )
    """)

    # 7e-1 folio distribution
    folio = results["folio_distribution"]
    for pair_key, d in folio.items():
        cur.execute("INSERT INTO ckh_allograph_test VALUES (?,?,?,?)",
                    ("7e_folio_distribution", pair_key,
                     d.get("correlation"), json.dumps(d)))

    # 7e-2 bigram context
    bigram = results["bigram_context"]
    for pair_key, d in bigram.items():
        z = d.get("z_cosine")
        cur.execute("INSERT INTO ckh_allograph_test VALUES (?,?,?,?)",
                    ("7e_bigram_context", pair_key,
                     d.get("cosine_similarity"), json.dumps(d)))

    # 7e-3 paragraph position
    para = results["para_position"]
    for pair_key, d in para.items():
        cur.execute("INSERT INTO ckh_allograph_test VALUES (?,?,?,?)",
                    ("7e_para_position", pair_key,
                     d.get("cosine_position_profile"), json.dumps(d)))

    # 7e-4 substitution
    sub = results["substitution"]
    cur.execute("INSERT INTO ckh_allograph_test VALUES (?,?,?,?)",
                ("7e_substitution", "ckh_to_k",
                 sub["ckh_delta"].get("sg_start_minus_cont"),
                 json.dumps(sub)))

    conn.commit()
    conn.close()


# =====================================================================
# Console summary
# =====================================================================

def format_summary(results: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PHASE 7e — ckh allograph test: is ckh an allograph of k?")
    lines.append("=" * 80)

    # 7e-1
    lines.append("\n-- 7e-1: Folio/section distribution correlation --")
    lines.append(f"  {'Pair':>12}  {'r':>7}  {'N_sg':>6}  {'N_simple':>8}  {'Folios':>6}")
    lines.append("  " + "-" * 50)
    folio = results["folio_distribution"]
    for pair_key in sorted(folio.keys()):
        d = folio[pair_key]
        r = f"{d['correlation']:+.4f}" if d["correlation"] is not None else "   n/a"
        lines.append(
            f"  {pair_key:>12}  {r:>7}  {d['total_sg']:>6}  "
            f"{d['total_simple']:>8}  {d['n_folios']:>6}"
        )

    # Section rates for ckh vs k
    ckh_data = folio.get("ckh_vs_k", {}).get("by_section", {})
    if ckh_data:
        lines.append("\n  Section rates (ckh vs k):")
        for sec, d in sorted(ckh_data.items()):
            sec_name = SECTION_NAMES.get(sec, sec)[:12]
            lines.append(
                f"    {sec} ({sec_name:>12}): ckh={d['rate_ckh']:.4f}  "
                f"k={d['rate_k']:.4f}  ratio={d['rate_ckh']/d['rate_k']:.2f}"
                if d.get("rate_k", 0) > 0 else
                f"    {sec} ({sec_name:>12}): ckh={d.get('rate_ckh', 0):.4f}  k=0"
            )

    # 7e-2
    lines.append("\n-- 7e-2: Bigram context similarity --")
    bigram = results["bigram_context"]
    lines.append(f"  {'Pair':>12}  {'Cosine':>7}  {'Jaccard':>8}  {'Shared':>6}  {'z_cos':>6}")
    lines.append("  " + "-" * 50)
    for pair_key in sorted(bigram.keys()):
        d = bigram[pair_key]
        z = d.get("z_cosine")
        z_str = f"{z:+.2f}" if z is not None else "  n/a"
        lines.append(
            f"  {pair_key:>12}  {d['cosine_similarity']:>7.4f}  "
            f"{d['jaccard_similarity']:>8.4f}  {d['n_shared_contexts']:>6}  {z_str:>6}"
        )

    # 7e-3
    lines.append("\n-- 7e-3: Paragraph position profile --")
    para = results["para_position"]
    lines.append(f"  {'Pair':>12}  {'cos_prof':>8}  "
                 f"{'sg_start':>8}  {'sg_cont':>8}  {'sg_end':>8}  "
                 f"{'si_start':>8}  {'si_cont':>8}  {'si_end':>8}")
    lines.append("  " + "-" * 80)
    for pair_key in sorted(para.keys()):
        d = para[pair_key]
        sf = d["sg_fractions"]
        sif = d["simple_fractions"]
        lines.append(
            f"  {pair_key:>12}  {d['cosine_position_profile']:>8.4f}  "
            f"{sf.get('para_start', 0):>8.4f}  {sf.get('para_cont', 0):>8.4f}  "
            f"{sf.get('para_end', 0):>8.4f}  "
            f"{sif.get('para_start', 0):>8.4f}  {sif.get('para_cont', 0):>8.4f}  "
            f"{sif.get('para_end', 0):>8.4f}"
        )

    # 7e-4
    lines.append("\n-- 7e-4: Substitution test --")
    sub = results["substitution"]
    lines.append(f"  {'Metric':>22}  {'Original':>10}  {'ckh->k':>10}  {'cph->p':>10}")
    lines.append("  " + "-" * 58)
    for key in ["m_end_rate", "sg_start_rate", "sg_cont_rate",
                "sg_start_minus_cont", "n_types", "n_tokens"]:
        o = sub["original"][key]
        c = sub["ckh_to_k"][key]
        p = sub["cph_to_p"][key]
        fmt = ".4f" if isinstance(o, float) else ","
        lines.append(
            f"  {key:>22}  {o:>10{fmt}}  {c:>10{fmt}}  {p:>10{fmt}}"
        )

    # Verdict
    lines.append("\n-- Interpretation --")
    v = results.get("verdict", {})
    lines.append(f"  {v.get('summary', 'See details above')}")
    for point in v.get("evidence_for", []):
        lines.append(f"    + {point}")
    for point in v.get("evidence_against", []):
        lines.append(f"    - {point}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines) + "\n"


# =====================================================================
# Verdict
# =====================================================================

def compute_verdict(results: dict) -> dict:
    """Synthesize evidence for/against ckh being an allograph of k."""
    evidence_for: list[str] = []
    evidence_against: list[str] = []

    # 7e-1: folio correlation
    folio = results["folio_distribution"]
    ckh_corr = folio.get("ckh_vs_k", {}).get("correlation")
    cph_corr = folio.get("cph_vs_p", {}).get("correlation")
    cth_corr = folio.get("cth_vs_t", {}).get("correlation")

    if ckh_corr is not None:
        if ckh_corr > 0.3:
            evidence_for.append(
                f"Folio correlation ckh-k = {ckh_corr:+.3f} (positive, co-occur on same folios)")
        elif ckh_corr < -0.1:
            evidence_against.append(
                f"Folio correlation ckh-k = {ckh_corr:+.3f} (negative, avoid same folios)")
        else:
            evidence_against.append(
                f"Folio correlation ckh-k = {ckh_corr:+.3f} (weak, no clear co-occurrence)")

    # 7e-2: bigram context
    bigram = results["bigram_context"]
    ckh_cos = bigram.get("ckh_vs_k", {}).get("cosine_similarity", 0)
    cph_cos = bigram.get("cph_vs_p", {}).get("cosine_similarity", 0)
    cth_cos = bigram.get("cth_vs_t", {}).get("cosine_similarity", 0)

    if ckh_cos > cph_cos and ckh_cos > cth_cos:
        evidence_for.append(
            f"Context similarity ckh-k = {ckh_cos:.3f} (higher than cph-p={cph_cos:.3f} "
            f"and cth-t={cth_cos:.3f})")
    else:
        evidence_against.append(
            f"Context similarity ckh-k = {ckh_cos:.3f} (not higher than "
            f"cph-p={cph_cos:.3f} or cth-t={cth_cos:.3f})")

    # 7e-3: paragraph position
    para = results["para_position"]
    ckh_para_cos = para.get("ckh_vs_k", {}).get("cosine_position_profile", 0)
    cph_para_cos = para.get("cph_vs_p", {}).get("cosine_position_profile", 0)

    if ckh_para_cos > 0.99:
        evidence_for.append(
            f"Paragraph position profile ckh ~ k (cosine={ckh_para_cos:.4f})")
    elif ckh_para_cos > cph_para_cos:
        evidence_for.append(
            f"Paragraph position ckh-k cosine={ckh_para_cos:.4f} "
            f"(closer than cph-p={cph_para_cos:.4f})")
    else:
        evidence_against.append(
            f"Paragraph position ckh-k cosine={ckh_para_cos:.4f} "
            f"(not closer than cph-p={cph_para_cos:.4f})")

    # 7e-4: substitution
    sub = results["substitution"]
    delta_m = abs(sub["ckh_delta"].get("m_end_rate", 0))
    delta_sg = abs(sub["ckh_delta"].get("sg_start_minus_cont", 0))
    delta_types = sub["ckh_delta"].get("n_types", 0)

    if delta_m < 0.005 and delta_sg < 0.01:
        evidence_for.append(
            f"Substitution ckh->k preserves structure (delta m_rate={delta_m:.4f}, "
            f"delta gallows_diff={delta_sg:.4f})")
    else:
        evidence_against.append(
            f"Substitution ckh->k changes structure (delta m_rate={delta_m:.4f}, "
            f"delta gallows_diff={delta_sg:.4f})")

    if delta_types < 0:
        evidence_for.append(
            f"Substitution reduces vocabulary by {abs(delta_types)} types "
            "(merging ckh-words with k-words)")

    # Overall
    n_for = len(evidence_for)
    n_against = len(evidence_against)

    if n_for >= 3 and n_against <= 1:
        summary = "ALLOGRAPH SUPPORTED: ckh behaves like a variant of k"
    elif n_against >= 3 and n_for <= 1:
        summary = "ALLOGRAPH REJECTED: ckh is functionally distinct from k"
    else:
        summary = f"INCONCLUSIVE: {n_for} points for, {n_against} against"

    return {
        "summary": summary,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "n_for": n_for,
        "n_against": n_against,
    }


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs) -> None:
    """Phase 7e: is ckh an allograph of k?"""
    report_path = config.stats_dir / "ckh_allograph_test.json"
    summary_path = config.stats_dir / "ckh_allograph_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  ckh_allograph_test report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("PHASE 7e — ckh allograph test: is ckh a variant of k?")

    # 1. Parse
    print_step("Parsing IVTFF lines...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    all_lines = parse_ivtff_lines(eva_file)
    content_lines = [l for l in all_lines
                     if l.get("para_type") != "label" and l.get("words")]
    click.echo(f"    {len(content_lines)} content lines")

    # Also parse pages for section metadata
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]

    # 2. Test 7e-1: Folio/section distribution
    print_step("7e-1: Folio/section distribution correlation...")
    folio_results = test_folio_distribution(content_lines, pages)
    for pair_key, d in sorted(folio_results.items()):
        r = d["correlation"]
        r_str = f"r={r:+.4f}" if r is not None else "n/a"
        click.echo(f"    {pair_key}: {r_str}")

    # 3. Test 7e-2: Bigram context
    print_step("7e-2: Bigram context overlap...")
    bigram_results = test_bigram_context(content_lines)
    for pair_key, d in sorted(bigram_results.items()):
        click.echo(f"    {pair_key}: cosine={d['cosine_similarity']:.4f}, "
                   f"jaccard={d['jaccard_similarity']:.4f}")

    # Null model for bigram context
    print_step("7e-2: Null model (500 permutations)...")
    bigram_null = null_bigram_context(content_lines, n_perms=N_NULL, seed=SEED)
    for pair_key in sorted(bigram_results.keys()):
        obs_cos = bigram_results[pair_key]["cosine_similarity"]
        null = bigram_null[pair_key]
        z = z_score(obs_cos, null["null_mean"], null["null_std"])
        bigram_results[pair_key]["null_cosine_mean"] = null["null_mean"]
        bigram_results[pair_key]["null_cosine_std"] = null["null_std"]
        bigram_results[pair_key]["z_cosine"] = round(z, 3) if z is not None else None
        z_str = f"z={z:+.2f}" if z is not None else "n/a"
        click.echo(f"    {pair_key}: obs={obs_cos:.4f}, "
                   f"null={null['null_mean']:.4f}, {z_str}")

    # 4. Test 7e-3: Paragraph position
    print_step("7e-3: Paragraph position profile...")
    para_results = test_para_position(content_lines)
    for pair_key, d in sorted(para_results.items()):
        click.echo(f"    {pair_key}: cosine={d['cosine_position_profile']:.4f}")

    # 5. Test 7e-4: Substitution
    print_step("7e-4: Substitution test (ckh->k vs cph->p)...")
    sub_results = test_substitution(content_lines)
    click.echo(f"    Original: m_rate={sub_results['original']['m_end_rate']:.4f}, "
               f"gallows_diff={sub_results['original']['sg_start_minus_cont']:.4f}, "
               f"types={sub_results['original']['n_types']}")
    click.echo(f"    ckh->k:   m_rate={sub_results['ckh_to_k']['m_end_rate']:.4f}, "
               f"gallows_diff={sub_results['ckh_to_k']['sg_start_minus_cont']:.4f}, "
               f"types={sub_results['ckh_to_k']['n_types']}")
    click.echo(f"    cph->p:   m_rate={sub_results['cph_to_p']['m_end_rate']:.4f}, "
               f"gallows_diff={sub_results['cph_to_p']['sg_start_minus_cont']:.4f}, "
               f"types={sub_results['cph_to_p']['n_types']}")

    # 6. Verdict
    print_step("Computing verdict...")
    all_results = {
        "folio_distribution": folio_results,
        "bigram_context": bigram_results,
        "para_position": para_results,
        "substitution": sub_results,
    }
    verdict = compute_verdict(all_results)
    all_results["verdict"] = verdict
    click.echo(f"    {verdict['summary']}")

    # 7. Save
    print_step("Saving JSON...")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    click.echo(f"    {report_path}")

    summary = format_summary(all_results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    {summary_path}")

    print_step("Writing DB table ckh_allograph_test...")
    db_path = config.output_dir.parent / "voynich.db"
    if db_path.exists():
        save_to_db(all_results, db_path)
        click.echo(f"    {db_path}")
    else:
        click.echo(f"    WARN: DB not found at {db_path}")

    click.echo(f"\n{summary}")
