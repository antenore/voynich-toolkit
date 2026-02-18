"""
Hand 1 Deep Dive — Analisi approfondita dello scriba principale.

Phase 16: Hand 1 ha il segnale più forte (48.5% full, 26.1% honest,
86 pagine, tutte sezione H, Currier Language A). Questo modulo esplora:

  vocab  — vocabolario esclusivo Hand 1 vs altri scribi
  structure — distribuzione lunghezze EVA, bigrammi/trigrammi
  audit  — mapping audit solo su token Hand 1
  compare — confronto diretto Hand 1 vs Hand 4 (entrambi Lang A)

Key finding da Phase 15 P3:
  - Hand 4 è anch'esso Lang A ma ha solo 35.1% — più basso di Hand 2 (Lang B, 41.5%)
  - Il mapping non cattura "Lang A" in generale, cattura "Hand 1" nello specifico
  - Herbal H1 vs H2 = +9.9pp, z=8.18, p≈0 (controllato per contenuto)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import norm as sp_norm

from .config import ToolkitConfig
from .currier_split import decode_and_match, two_proportion_ztest
from .full_decode import FULL_MAPPING, SECTION_NAMES, preprocess_eva, decode_word
from .mapping_audit import (
    HEBREW_CHARS,
    decode_hebrew,
    load_honest_lexicon,
    audit_letter,
    count_matches,
)
from .permutation_stats import build_full_mapping
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# ── Costanti ────────────────────────────────────────────────────

HAND1_FOLIOS = {
    # f1r–f11v
    "f1r","f1v","f2r","f2v","f3r","f3v","f4r","f4v","f5r","f5v",
    "f6r","f6v","f7r","f7v","f8r","f8v","f9r","f9v","f10r","f10v",
    "f11r","f11v",
    # f13r–f16v (skip f12, annex Sagittarius)
    "f13r","f13v","f14r","f14v","f15r","f15v","f16r","f16v",
    # f17r–f26v
    "f17r","f17v","f18r","f18v","f19r","f19v","f20r","f20v",
    "f21r","f21v","f22r","f22v","f23r","f23v","f24r","f24v",
    "f25r","f25v","f26r","f26v",
    # f27r–f28v
    "f27r","f27v","f28r","f28v",
    # f29r–f32v
    "f29r","f29v","f30r","f30v","f31r","f31v","f32r","f32v",
    # f33r–f40v
    "f33r","f33v","f34r","f34v","f35r","f35v","f36r","f36v",
    "f37r","f37v","f38r","f38v","f39r","f39v","f40r","f40v",
    # f41r–f46v
    "f41r","f41v","f42r","f42v","f43r","f43v","f44r","f44v",
    "f45r","f45v","f46r","f46v",
    # f47r–f49v
    "f47r","f47v","f48r","f48v","f49r","f49v",
    # f54r–f56v
    "f54r","f54v","f55r","f55v","f56r","f56v",
    # f65r–f66v
    "f65r","f65v","f66r","f66v",
}

MIN_LEN = 3  # lunghezza minima token decodificato


# =====================================================================
# Parsing e split del corpus per mano
# =====================================================================

def split_corpus_by_hand(pages: list[dict]) -> dict[str, dict]:
    """Splitta le pagine per mano (Hand 1, Hand 4, altri)."""
    result: dict[str, dict] = {}
    for p in pages:
        hand = p.get("hand", "?")
        if hand not in result:
            result[hand] = {
                "pages": [],
                "words": [],
                "sections": Counter(),
                "languages": Counter(),
                "n_pages": 0,
            }
        result[hand]["pages"].append(p)
        result[hand]["words"].extend(p["words"])
        result[hand]["sections"][p.get("section", "?")] += 1
        result[hand]["languages"][p.get("language", "?")] += 1
        result[hand]["n_pages"] += 1
    return result


# =====================================================================
# Analisi 1 — VOCAB: vocabolario esclusivo Hand 1
# =====================================================================

def vocab_analysis(
    corpus: dict[str, dict],
    lexicon_set: set,
    form_to_gloss: dict,
    n_top: int = 30,
) -> dict:
    """
    Quante parole decodificate matchano il lessico in modo esclusivo per Hand 1?
    Ritorna:
      - tipi ebraici che appaiono SOLO in Hand 1 (esclusivi)
      - tipi condivisi con almeno un altro scriba
      - top glossed words di Hand 1 con glossa e frequenza
      - confronto top-20 Hand 1 vs Hand 2 (token e tipi)
    """
    # Decodifica per ogni mano
    decoded_by_hand: dict[str, Counter] = {}
    for hand, c in corpus.items():
        cnt: Counter = Counter()
        for w in c["words"]:
            heb = decode_word(w)[1]  # (italian, hebrew, n_unk)
            if heb and len(heb) >= MIN_LEN and "?" not in heb:
                cnt[heb] += 1
        decoded_by_hand[hand] = cnt

    h1_types = set(decoded_by_hand.get("1", {}).keys())
    other_types: set = set()
    for hand, cnt in decoded_by_hand.items():
        if hand != "1":
            other_types.update(cnt.keys())

    exclusive_h1 = h1_types - other_types  # appaiono SOLO in Hand 1
    shared = h1_types & other_types         # condivisi con almeno un altro

    # Parole matchate nella lexicon tra esclusive
    exclusive_matched = {t for t in exclusive_h1 if t in lexicon_set}
    shared_matched = {t for t in shared if t in lexicon_set}

    # Top glossed words di Hand 1 (matched, ordinate per frequenza token)
    h1_cnt = decoded_by_hand.get("1", Counter())
    top_glossed = []
    for heb, freq in h1_cnt.most_common():
        if heb in lexicon_set:
            gloss = form_to_gloss.get(heb, "")
            top_glossed.append({
                "hebrew": heb,
                "freq_h1": freq,
                "gloss": gloss,
                "exclusive": heb in exclusive_h1,
            })
        if len(top_glossed) >= n_top:
            break

    # Confronto tipo-token Hand 1 vs Hand 4 (stessa lingua)
    h4_cnt = decoded_by_hand.get("4", Counter())
    h4_types = set(h4_cnt.keys())
    h1_only = h1_types - h4_types
    h4_only = h4_types - h1_types
    shared_14 = h1_types & h4_types
    jaccard_14 = len(shared_14) / len(h1_types | h4_types) if (h1_types | h4_types) else 0

    # Top parole per Hand 1 matchate, con confronto Hand 4 freq
    top_compare_14 = []
    for heb, freq_h1 in h1_cnt.most_common(n_top):
        if heb in lexicon_set:
            top_compare_14.append({
                "hebrew": heb,
                "gloss": form_to_gloss.get(heb, ""),
                "freq_h1": freq_h1,
                "freq_h4": h4_cnt.get(heb, 0),
                "exclusive_h1": heb not in h4_types,
            })

    return {
        "h1_n_types": len(h1_types),
        "h1_n_types_matched": len(h1_types & lexicon_set),
        "exclusive_n_types": len(exclusive_h1),
        "exclusive_matched": len(exclusive_matched),
        "shared_n_types": len(shared),
        "shared_matched": len(shared_matched),
        "top_glossed_h1": top_glossed,
        "h1_vs_h4": {
            "h1_types": len(h1_types),
            "h4_types": len(h4_types),
            "h1_only": len(h1_only),
            "h4_only": len(h4_only),
            "shared_14": len(shared_14),
            "jaccard_14": round(jaccard_14, 4),
        },
        "top_compare_h1_h4": top_compare_14,
        # Confronto Hand 1 vs Hand 2
        "h1_vs_h2": {
            "h2_types": len(set(decoded_by_hand.get("2", {}).keys())),
            "shared_12": len(h1_types & set(decoded_by_hand.get("2", {}).keys())),
            "jaccard_12": round(
                len(h1_types & set(decoded_by_hand.get("2", {}).keys())) /
                len(h1_types | set(decoded_by_hand.get("2", {}).keys()))
                if (h1_types | set(decoded_by_hand.get("2", {}).keys())) else 0,
                4,
            ),
        },
    }


# =====================================================================
# Analisi 2 — STRUCTURE: bigrammi EVA, lunghezza parole
# =====================================================================

def structure_analysis(corpus: dict[str, dict]) -> dict:
    """
    Distribuzione lunghezze parole EVA (per mano) e bigrammi/trigrammi EVA.
    """
    def word_lengths(words: list[str]) -> dict:
        lengths = [len(w) for w in words if w]
        if not lengths:
            return {"mean": 0, "median": 0, "std": 0, "dist": {}}
        arr = np.array(lengths)
        dist = Counter(lengths)
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": float(np.median(arr)),
            "std": round(float(np.std(arr)), 3),
            "dist": dict(sorted(dist.items())),
        }

    def extract_bigrams(words: list[str]) -> Counter:
        bg: Counter = Counter()
        for w in words:
            # Lavora sull'EVA grezzo preprocessato (senza q-prefix)
            _, proc = preprocess_eva(w)
            chars = list(proc)
            for i in range(len(chars) - 1):
                bg[(chars[i], chars[i+1])] += 1
        return bg

    def extract_trigrams(words: list[str]) -> Counter:
        tg: Counter = Counter()
        for w in words:
            _, proc = preprocess_eva(w)
            chars = list(proc)
            for i in range(len(chars) - 2):
                tg[(chars[i], chars[i+1], chars[i+2])] += 1
        return tg

    def char_freq(words: list[str]) -> Counter:
        cf: Counter = Counter()
        for w in words:
            _, proc = preprocess_eva(w)
            cf.update(proc)
        return cf

    result = {}
    for hand in ["1", "4", "2"]:
        if hand not in corpus:
            continue
        words = corpus[hand]["words"]
        bg = extract_bigrams(words)
        tg = extract_trigrams(words)
        cf = char_freq(words)

        result[hand] = {
            "n_words": len(words),
            "lengths": word_lengths(words),
            "char_freq_top20": [
                {"char": c, "count": n}
                for c, n in cf.most_common(20)
            ],
            "bigrams_top20": [
                {"bigram": f"{a}{b}", "count": n}
                for (a, b), n in bg.most_common(20)
            ],
            "trigrams_top15": [
                {"trigram": f"{a}{b}{c}", "count": n}
                for (a, b, c), n in tg.most_common(15)
            ],
        }

    # Confronto distribuzioni lunghezze H1 vs H4 vs H2
    length_comparison = {}
    for hand in ["1", "4", "2"]:
        if hand in result:
            length_comparison[hand] = result[hand]["lengths"]

    # Bigrams esclusivi Hand 1 vs Hand 4
    bg1 = {k for k, v in Counter({}).items()}
    bg4 = {k for k, v in Counter({}).items()}
    if "1" in corpus and "4" in corpus:
        bg1_cnt = extract_bigrams(corpus["1"]["words"])
        bg4_cnt = extract_bigrams(corpus["4"]["words"])
        # Top bigrams in H1 non presenti in H4
        exclusive_bg_h1 = [
            {"bigram": f"{a}{b}", "count_h1": n, "count_h4": bg4_cnt.get((a,b), 0)}
            for (a, b), n in bg1_cnt.most_common(40)
            if bg4_cnt.get((a,b), 0) == 0
        ][:15]
        # Top bigrams H4 non in H1
        exclusive_bg_h4 = [
            {"bigram": f"{a}{b}", "count_h4": n, "count_h1": bg1_cnt.get((a,b), 0)}
            for (a, b), n in bg4_cnt.most_common(40)
            if bg1_cnt.get((a,b), 0) == 0
        ][:15]
        result["bigram_exclusive_h1"] = exclusive_bg_h1
        result["bigram_exclusive_h4"] = exclusive_bg_h4

    result["length_comparison"] = length_comparison
    return result


# =====================================================================
# Analisi 3 — AUDIT: mapping audit solo su token Hand 1
# =====================================================================

def hand1_mapping_audit(
    corpus: dict[str, dict],
    lexicon_set: set,
    base_mapping: dict | None = None,
    n_top_candidates: int = 5,
) -> dict:
    """
    Audit del mapping su soli token di Hand 1.
    Per ogni lettera EVA, testa tutte le 22 alternative ebraiche mantenendo
    fisso il resto e conta quante forme del lessico vengono trovate.

    Usa il lessico "onesto" (45K) per evitare inflazione Sefaria.
    """
    if base_mapping is None:
        base_mapping = FULL_MAPPING

    # Raccoglie frequenze EVA solo per Hand 1
    eva_freqs: Counter = Counter()
    for w in corpus.get("1", {}).get("words", []):
        eva_freqs[w] += 1

    click.echo(f"      Hand 1: {sum(eva_freqs.values()):,} token EVA, "
               f"{len(eva_freqs):,} tipi unici")

    # Calcola baseline (mapping attuale)
    _, base_tokens, _, _ = count_matches(eva_freqs, lexicon_set,
                                         mapping=dict(base_mapping))

    # Audit lettera per lettera
    audit_results = {}
    free_letters: list[str] = []   # lettere non ottimali

    # Calcola "allowed_hebrew": lettere ebraiche non già usate dal mapping
    # (mapping vincolato: non duplicare)
    for eva_ch in sorted(base_mapping.keys()):
        # Per ogni EVA char, mantieni fissi tutti gli altri
        # allowed = tutto tranne le lettere già usate dagli ALTRI chars
        used_by_others = {v for k, v in base_mapping.items() if k != eva_ch}
        allowed = set(HEBREW_CHARS) - used_by_others

        res = audit_letter(eva_ch, eva_freqs, lexicon_set,
                           dict(base_mapping), allowed_hebrew=allowed)
        audit_results[eva_ch] = res
        if res["current_rank"] > 1:
            free_letters.append(eva_ch)

    # Testa anche le 3 lettere non mappate (z, C, q) — prova di assegnamento
    # Le lettere "usate" includono: mapping values + positional (b,s) + digraph (k)
    # b = INITIAL_D_HEBREW (d[0]→bet), s = INITIAL_H_HEBREW (h[0]→samekh)
    # k = CH_HEBREW (ch→kaf)
    # h = II_HEBREW (ii→he), r = I_HEBREW (standalone i→resh)
    unmapped_test = {}
    from .full_decode import (INITIAL_D_HEBREW as _ID, INITIAL_H_HEBREW as _IH,
                               CH_HEBREW as _CH, II_HEBREW as _II, I_HEBREW as _I)
    used_all = (set(base_mapping.values()) |
                {_ID, _IH, _CH, _II, _I})
    free_hebrew = set(HEBREW_CHARS) - used_all  # dovrebbe essere z, C, q
    for free_h in sorted(free_hebrew):
        # Trova quale EVA char guadagnerebbe di più cedendo la propria lettera a free_h
        best_gain = 0
        best_eva = None
        for eva_ch in base_mapping:
            test_map = dict(base_mapping)
            old_heb = test_map[eva_ch]
            test_map[eva_ch] = free_h
            _, new_tok, _, _ = count_matches(eva_freqs, lexicon_set,
                                             mapping=test_map)
            gain = new_tok - base_tokens
            if gain > best_gain:
                best_gain = gain
                best_eva = eva_ch
        unmapped_test[free_h] = {
            "hebrew": free_h,
            "hebrew_name": CONSONANT_NAMES.get(free_h, "?"),
            "best_eva_to_swap": best_eva,
            "token_gain": best_gain,
        }

    # Riassunto ottimalità
    n_optimal = sum(1 for r in audit_results.values() if r["current_rank"] == 1)
    n_total = len(audit_results)

    return {
        "n_optimal": n_optimal,
        "n_total": n_total,
        "base_tokens": base_tokens,
        "audit": audit_results,
        "non_optimal_letters": free_letters,
        "unmapped_test": unmapped_test,
    }


# =====================================================================
# Analisi 4 — COMPARE: Hand 1 vs Hand 4 (entrambi Lang A)
# =====================================================================

def compare_h1_h4(corpus: dict[str, dict], lexicon_set: set,
                  form_to_gloss: dict) -> dict:
    """
    Confronto diretto Hand 1 vs Hand 4 (entrambi Currier Lang A).
    Gap inspiegato: 48.5% vs 35.1% = -13.4pp.

    Analisi:
      1. Proporzione z-test H1 vs H4
      2. Top parole H4 matchate: glosse e frequenze
      3. Frequenze relative dei caratteri EVA (normalizzate)
      4. Bigrammi più divergenti tra H1 e H4
      5. Lunghezza media parole e distribuzione
    """
    if "1" not in corpus or "4" not in corpus:
        return {"error": "mani 1 o 4 non trovate nel corpus"}

    h1_words = corpus["1"]["words"]
    h4_words = corpus["4"]["words"]

    s1 = decode_and_match(h1_words, lexicon_set)
    s4 = decode_and_match(h4_words, lexicon_set)
    zt = two_proportion_ztest(s1["n_matched"], s1["n_decoded"],
                              s4["n_matched"], s4["n_decoded"])

    # Frequenze caratteri EVA normalizzate
    def eva_char_freq(words):
        cf: Counter = Counter()
        for w in words:
            _, proc = preprocess_eva(w)
            cf.update(proc)
        total = sum(cf.values())
        return {c: round(n / total, 5) for c, n in cf.items()} if total else {}

    cf1 = eva_char_freq(h1_words)
    cf4 = eva_char_freq(h4_words)

    all_chars = sorted(set(cf1) | set(cf4))
    char_comparison = []
    for c in all_chars:
        f1 = cf1.get(c, 0.0)
        f4 = cf4.get(c, 0.0)
        char_comparison.append({
            "char": c,
            "freq_h1": round(f1, 5),
            "freq_h4": round(f4, 5),
            "diff": round(f1 - f4, 5),
        })
    char_comparison.sort(key=lambda x: abs(x["diff"]), reverse=True)

    # Bigrammi EVA normalizzati e divergenza
    def eva_bigrams_norm(words):
        bg: Counter = Counter()
        for w in words:
            _, proc = preprocess_eva(w)
            chars = list(proc)
            for i in range(len(chars) - 1):
                bg[(chars[i], chars[i+1])] += 1
        total = sum(bg.values())
        return {k: n/total for k, n in bg.items()} if total else {}

    bg1 = eva_bigrams_norm(h1_words)
    bg4 = eva_bigrams_norm(h4_words)
    all_bg = set(bg1) | set(bg4)
    bg_divergence = []
    for bg in all_bg:
        f1 = bg1.get(bg, 0.0)
        f4 = bg4.get(bg, 0.0)
        bg_divergence.append({
            "bigram": f"{bg[0]}{bg[1]}",
            "freq_h1": round(f1, 6),
            "freq_h4": round(f4, 6),
            "diff": round(f1 - f4, 6),
        })
    bg_divergence.sort(key=lambda x: abs(x["diff"]), reverse=True)
    bg_divergence_top = bg_divergence[:20]

    # Parole matchate di Hand 4 con glossa
    h4_decoded: Counter = Counter()
    for w in h4_words:
        heb = decode_word(w)[1]
        if heb and len(heb) >= MIN_LEN and "?" not in heb and heb in lexicon_set:
            h4_decoded[heb] += 1

    h4_matched_top = [
        {"hebrew": heb, "freq": n, "gloss": form_to_gloss.get(heb, "")}
        for heb, n in h4_decoded.most_common(20)
    ]

    # Confronto distribuzioni di lunghezza parole
    def len_stats(words):
        ls = [len(w) for w in words if w]
        if not ls:
            return {}
        a = np.array(ls)
        return {
            "mean": round(float(np.mean(a)), 3),
            "median": float(np.median(a)),
            "std": round(float(np.std(a)), 3),
            "n": len(ls),
        }

    return {
        "h1_stats": {
            "n_pages": corpus["1"]["n_pages"],
            "n_words": len(h1_words),
            "n_decoded": s1["n_decoded"],
            "n_matched": s1["n_matched"],
            "match_rate": round(s1["match_rate"], 5),
            "n_unique_types": s1["n_unique"],
            "n_unique_matched": s1["n_unique_matched"],
        },
        "h4_stats": {
            "n_pages": corpus["4"]["n_pages"],
            "n_words": len(h4_words),
            "n_decoded": s4["n_decoded"],
            "n_matched": s4["n_matched"],
            "match_rate": round(s4["match_rate"], 5),
            "n_unique_types": s4["n_unique"],
            "n_unique_matched": s4["n_unique_matched"],
        },
        "z_test_h1_vs_h4": zt,
        "char_comparison_top20": char_comparison[:20],
        "bigram_divergence_top20": bg_divergence_top,
        "h4_top_matched": h4_matched_top,
        "length_stats": {
            "h1": len_stats(h1_words),
            "h4": len_stats(h4_words),
        },
    }


# =====================================================================
# Formatting del sommario
# =====================================================================

def format_summary(vocab: dict, structure: dict, audit: dict,
                   compare: dict, honest: bool = True) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("  HAND 1 DEEP DIVE — Analisi Approfondita Scriba Principale")
    lines.append("=" * 68)
    lex_note = "onesto (45K, no Sefaria)" if honest else "pieno (491K)"
    lines.append(f"  Lessico: {lex_note}")

    # ── VOCAB ──────────────────────────────────────────────────
    lines.append("\n── 1. VOCABOLARIO ──")
    lines.append(f"  Tipi ebraici decodificati Hand 1: {vocab['h1_n_types']:,}")
    lines.append(f"  Matchati nel lessico:             {vocab['h1_n_types_matched']:,} "
                 f"({100*vocab['h1_n_types_matched']/max(1,vocab['h1_n_types']):.1f}%)")
    lines.append(f"  Tipi ESCLUSIVI Hand 1:            {vocab['exclusive_n_types']:,} "
                 f"(matchati: {vocab['exclusive_matched']:,})")
    lines.append(f"  Tipi condivisi con altri scribi:  {vocab['shared_n_types']:,} "
                 f"(matchati: {vocab['shared_matched']:,})")

    lines.append(f"\n  Overlap H1 vs H4 (Jaccard): {vocab['h1_vs_h4']['jaccard_14']:.3f}")
    lines.append(f"  Overlap H1 vs H2 (Jaccard): {vocab['h1_vs_h2']['jaccard_12']:.3f}")

    lines.append(f"\n  Top parole matchate Hand 1:")
    lines.append(f"  {'Hebrew':12s}  {'H1 freq':>8s}  {'Excl?':>6s}  Glossa")
    lines.append("  " + "-" * 55)
    for entry in vocab["top_glossed_h1"][:20]:
        excl = "SI" if entry["exclusive"] else "no"
        gloss = entry["gloss"][:35] if entry["gloss"] else "(nessuna)"
        lines.append(f"  {entry['hebrew']:12s}  {entry['freq_h1']:>8d}  "
                     f"{excl:>6s}  {gloss}")

    lines.append(f"\n  Top parole H1 vs H4 confrontate:")
    lines.append(f"  {'Hebrew':12s}  {'H1 freq':>8s}  {'H4 freq':>8s}  {'Solo H1?':>9s}  Glossa")
    lines.append("  " + "-" * 65)
    for entry in vocab["top_compare_h1_h4"][:15]:
        excl = "SI" if entry["exclusive_h1"] else "no"
        gloss = entry["gloss"][:30] if entry["gloss"] else "(nessuna)"
        lines.append(f"  {entry['hebrew']:12s}  {entry['freq_h1']:>8d}  "
                     f"{entry['freq_h4']:>8d}  {excl:>9s}  {gloss}")

    # ── STRUCTURE ──────────────────────────────────────────────
    lines.append("\n── 2. STRUTTURA EVA ──")
    lines.append(f"  {'Mano':>6s}  {'N parole':>9s}  {'Media':>7s}  {'Mediana':>8s}  {'Dev.std.':>9s}")
    lines.append("  " + "-" * 44)
    lc = structure.get("length_comparison", {})
    for hand in ["1", "4", "2"]:
        if hand in lc:
            ls = lc[hand]
            label = {"1": "Hand 1", "4": "Hand 4", "2": "Hand 2"}.get(hand, hand)
            n_words = structure.get(hand, {}).get("n_words", 0)
            lines.append(f"  {label:>6s}  {n_words:>9,}  {ls['mean']:>7.2f}  "
                         f"{ls['median']:>8.1f}  {ls['std']:>9.3f}")

    lines.append(f"\n  Frequenze caratteri EVA — top 10 per Hand 1:")
    lines.append(f"  {'Char':>5s}  {'H1 freq':>8s}  {'H4 freq':>8s}  {'H2 freq':>8s}")
    lines.append("  " + "-" * 35)
    if "1" in structure:
        h1_cf = {e["char"]: e["count"] for e in structure["1"]["char_freq_top20"]}
        h4_cf = {e["char"]: e["count"] for e in structure.get("4", {}).get("char_freq_top20", [])}
        h2_cf = {e["char"]: e["count"] for e in structure.get("2", {}).get("char_freq_top20", [])}
        h4_total = sum(h4_cf.values()) or 1
        h2_total = sum(h2_cf.values()) or 1
        h1_total = sum(h1_cf.values()) or 1
        for entry in structure["1"]["char_freq_top20"][:10]:
            c = entry["char"]
            f1 = entry["count"] / h1_total
            f4 = h4_cf.get(c, 0) / h4_total
            f2 = h2_cf.get(c, 0) / h2_total
            lines.append(f"  {repr(c):>5s}  {f1:8.4f}  {f4:8.4f}  {f2:8.4f}")

    lines.append(f"\n  Top bigrammi EVA di Hand 1 (non presenti in Hand 4):")
    lines.append(f"  {'Bigram':>7s}  {'H1 count':>9s}")
    lines.append("  " + "-" * 20)
    for entry in structure.get("bigram_exclusive_h1", [])[:10]:
        lines.append(f"  {entry['bigram']:>7s}  {entry['count_h1']:>9d}")

    # ── AUDIT ──────────────────────────────────────────────────
    lines.append("\n── 3. MAPPING AUDIT (solo token Hand 1) ──")
    lines.append(f"  Lettere ottimali: {audit['n_optimal']}/{audit['n_total']}")
    lines.append(f"  Token matched baseline: {audit['base_tokens']:,}")

    if audit["non_optimal_letters"]:
        lines.append(f"  Lettere NON ottimali: {', '.join(audit['non_optimal_letters'])}")
        lines.append(f"\n  {'EVA':>5s}  {'Cur.':>5s}  {'Rank':>5s}  {'Gap tok':>8s}  "
                     f"{'Best':>6s}  Gap%")
        lines.append("  " + "-" * 45)
        for eva_ch in audit["non_optimal_letters"]:
            r = audit["audit"][eva_ch]
            lines.append(f"  {eva_ch:>5s}  {r['current']:>5s}  "
                         f"{r['current_rank']:>5d}  {r['gap_tokens']:>+8d}  "
                         f"{r['best']:>6s}  {r['gap_pct']:>+5.1f}%")
    else:
        lines.append("  Tutte le lettere sono ottimali per Hand 1.")

    lines.append(f"\n  Test lettere non mappate (z, C, q) su Hand 1:")
    lines.append(f"  {'Heb':>5s}  {'Nome':15s}  {'Best swap EVA':>14s}  {'Guadagno tok':>13s}")
    lines.append("  " + "-" * 52)
    for heb, r in sorted(audit.get("unmapped_test", {}).items()):
        lines.append(f"  {heb:>5s}  {r['hebrew_name']:15s}  "
                     f"{str(r['best_eva_to_swap'] or 'n/a'):>14s}  "
                     f"{r['token_gain']:>+13d}")

    # ── COMPARE H1 vs H4 ──────────────────────────────────────
    lines.append("\n── 4. CONFRONTO Hand 1 vs Hand 4 (entrambi Lang A) ──")
    h1s = compare["h1_stats"]
    h4s = compare["h4_stats"]
    zt = compare["z_test_h1_vs_h4"]
    lines.append(f"  Hand 1: {h1s['n_pages']} pag, {h1s['n_decoded']:,} dec, "
                 f"{h1s['match_rate']*100:.1f}% match ({h1s['n_matched']:,} tok)")
    lines.append(f"  Hand 4: {h4s['n_pages']} pag, {h4s['n_decoded']:,} dec, "
                 f"{h4s['match_rate']*100:.1f}% match ({h4s['n_matched']:,} tok)")
    diff_pp = (h1s["match_rate"] - h4s["match_rate"]) * 100
    lines.append(f"  Differenza H1-H4: {diff_pp:+.1f}pp  z={zt['z_score']:.2f}  "
                 f"p={zt['p_value']:.4f}  "
                 f"{'***' if zt.get('significant_001') else '**' if zt.get('significant_01') else '*' if zt.get('significant_05') else 'ns'}")

    lines.append(f"\n  Parole matchate top-10 Hand 4:")
    lines.append(f"  {'Hebrew':12s}  {'H4 freq':>8s}  Glossa")
    lines.append("  " + "-" * 40)
    for entry in compare["h4_top_matched"][:10]:
        gloss = entry["gloss"][:30] if entry["gloss"] else "(nessuna)"
        lines.append(f"  {entry['hebrew']:12s}  {entry['freq']:>8d}  {gloss}")

    lines.append(f"\n  Differenze frequenze caratteri EVA (top 15 per |diff|):")
    lines.append(f"  {'Char':>5s}  {'H1 %':>8s}  {'H4 %':>8s}  {'Diff':>10s}")
    lines.append("  " + "-" * 36)
    for entry in compare["char_comparison_top20"][:15]:
        lines.append(f"  {repr(entry['char']):>5s}  {entry['freq_h1']*100:8.3f}%  "
                     f"{entry['freq_h4']*100:8.3f}%  {entry['diff']*100:>+9.3f}pp")

    lines.append(f"\n  Bigrammi EVA più divergenti H1 vs H4 (top 15):")
    lines.append(f"  {'Bigram':>7s}  {'H1 %':>8s}  {'H4 %':>8s}  {'Diff':>10s}")
    lines.append("  " + "-" * 38)
    for entry in compare["bigram_divergence_top20"][:15]:
        lines.append(f"  {entry['bigram']:>7s}  {entry['freq_h1']*100:8.3f}%  "
                     f"{entry['freq_h4']*100:8.3f}%  {entry['diff']*100:>+9.3f}pp")

    lines.append(f"\n  Lunghezza parole EVA:")
    ls1 = compare["length_stats"]["h1"]
    ls4 = compare["length_stats"]["h4"]
    lines.append(f"    Hand 1: media={ls1['mean']:.2f}  mediana={ls1['median']:.1f}  "
                 f"std={ls1['std']:.3f}  n={ls1['n']:,}")
    lines.append(f"    Hand 4: media={ls4['mean']:.2f}  mediana={ls4['median']:.1f}  "
                 f"std={ls4['std']:.3f}  n={ls4['n']:,}")

    lines.append(f"\n{'=' * 68}")
    return "\n".join(lines)


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Hand 1 deep dive — analisi approfondita dello scriba principale."""
    report_path = config.stats_dir / "hand1_deep_dive.json"
    summary_path = config.stats_dir / "hand1_deep_dive_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Hand 1 Deep Dive report esiste. Usa --force per ri-eseguire.")
        return

    config.ensure_dirs()
    print_header("PHASE 16 — Hand 1 Deep Dive")

    # 1. Parse EVA corpus
    print_step("Parsing corpus EVA...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file non trovato: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    pages = eva_data["pages"]
    click.echo(f"    {eva_data['total_words']:,} parole, {len(pages)} pagine")

    # 2. Split per mano
    print_step("Splitting per mano...")
    corpus = split_corpus_by_hand(pages)
    for hand in sorted(corpus.keys()):
        c = corpus[hand]
        click.echo(f"    Hand {hand}: {c['n_pages']} pag, "
                   f"{len(c['words']):,} parole, "
                   f"sez={dict(c['sections'])}, lang={dict(c['languages'])}")

    # 3. Carica lessico onesto
    print_step("Caricando lessico onesto (45K, no Sefaria-Corpus)...")
    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    if not enriched_path.exists():
        raise click.ClickException("Lessico non trovato. Esegui: voynich enrich-lexicon")
    honest_lex, form_to_gloss = load_honest_lexicon(config)
    click.echo(f"    Lessico onesto: {len(honest_lex):,} forme")

    # 4. VOCAB
    print_step("Analisi vocabolario (vocab)...")
    vocab = vocab_analysis(corpus, honest_lex, form_to_gloss)
    click.echo(f"    Tipi H1: {vocab['h1_n_types']:,}  "
               f"matchati: {vocab['h1_n_types_matched']:,}  "
               f"esclusivi: {vocab['exclusive_n_types']:,}")
    click.echo(f"    Jaccard H1-H4: {vocab['h1_vs_h4']['jaccard_14']:.3f}  "
               f"Jaccard H1-H2: {vocab['h1_vs_h2']['jaccard_12']:.3f}")

    # 5. STRUCTURE
    print_step("Analisi struttura EVA (structure)...")
    structure = structure_analysis(corpus)
    for hand in ["1", "4", "2"]:
        if hand in structure and "lengths" in structure[hand]:
            ls = structure[hand]["lengths"]
            click.echo(f"    Hand {hand}: media={ls['mean']:.2f}  "
                       f"std={ls['std']:.3f}  n={structure[hand]['n_words']:,}")

    # 6. AUDIT
    print_step("Mapping audit solo su token Hand 1 (audit)...")
    click.echo("    (potrebbe richiedere qualche secondo...)")
    audit = hand1_mapping_audit(corpus, honest_lex, base_mapping=FULL_MAPPING)
    click.echo(f"    Lettere ottimali: {audit['n_optimal']}/{audit['n_total']}")
    if audit["non_optimal_letters"]:
        click.echo(f"    Non ottimali: {', '.join(audit['non_optimal_letters'])}")
    for heb, r in sorted(audit.get("unmapped_test", {}).items()):
        click.echo(f"    Unmapped {heb} ({r['hebrew_name']}): "
                   f"gain={r['token_gain']:+d} token "
                   f"(swap EVA {r['best_eva_to_swap']})")

    # 7. COMPARE H1 vs H4
    print_step("Confronto Hand 1 vs Hand 4 (compare)...")
    compare = compare_h1_h4(corpus, honest_lex, form_to_gloss)
    if "error" not in compare:
        zt = compare["z_test_h1_vs_h4"]
        diff_pp = (compare["h1_stats"]["match_rate"] - compare["h4_stats"]["match_rate"]) * 100
        click.echo(f"    H1 {compare['h1_stats']['match_rate']*100:.1f}% vs "
                   f"H4 {compare['h4_stats']['match_rate']*100:.1f}% = "
                   f"{diff_pp:+.1f}pp  z={zt['z_score']:.2f}  p={zt['p_value']:.4f}")

    # 8. Salva report
    print_step("Salvataggio report...")
    report = {
        "vocab": vocab,
        "structure": structure,
        "audit": {
            "n_optimal": audit["n_optimal"],
            "n_total": audit["n_total"],
            "base_tokens": audit["base_tokens"],
            "non_optimal_letters": audit["non_optimal_letters"],
            "audit": {
                k: {kk: vv for kk, vv in v.items() if kk != "top5"}
                for k, v in audit["audit"].items()
            },
            "unmapped_test": audit["unmapped_test"],
        },
        "compare": compare,
        "lexicon": "honest_45K",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(vocab, structure, audit, compare, honest=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT: {summary_path}")

    click.echo(f"\n{summary}")
