"""
Null Model Test — Il segnale ebraico è più che lunghezza?

Phase 16B: Il mapping produce z=4.17 (lessico onesto) ma il testo decodificato
NON è leggibile come ebraico. Questo modulo quantifica il gap tra "match statistico"
e "testo casuale" con 3 test:

  Test 1 (match_rate): real match rate vs stringhe sintetiche stessa lunghezza
  Test 2 (gloss_quality): diversità glosse reali vs sintetiche
  Test 3 (bigram_plausibility): struttura bigramma reale vs sintetica

Se tutti e 3 significativi → il segnale è reale, il gap è sintattico non lessicale.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import click
import numpy as np
from scipy.stats import norm as sp_norm

from .config import ToolkitConfig
from .full_decode import decode_word
from .mapping_audit import load_honest_lexicon
from .utils import print_header, print_step
from .word_structure import parse_eva_words

# ── Costanti ────────────────────────────────────────────────────

MIN_LEN = 3       # lunghezza minima token decodificato
N_ITERS = 500     # iterazioni Monte Carlo
SEED = 42
LAPLACE_ALPHA = 1.0  # Laplace smoothing per modello bigramma


# =====================================================================
# Decodifica corpus → token ebraici
# =====================================================================

def decode_corpus(pages: list[dict]) -> list[str]:
    """Decodifica tutte le parole EVA in token ebraici.

    Returns:
        lista di token ebraici (solo quelli con n_unknown == 0, len >= MIN_LEN)
    """
    tokens = []
    for page in pages:
        for word in page["words"]:
            _ita, heb, n_unk = decode_word(word)
            if n_unk == 0 and len(heb) >= MIN_LEN:
                tokens.append(heb)
    return tokens


# =====================================================================
# Distribuzione caratteri dal lessico
# =====================================================================

def build_char_weights(lexicon_set: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Frequenza caratteri nel lessico onesto → array numpy per sampling.

    Returns:
        (chars_array, probs_array) — entrambi ordinati per char
    """
    counts: Counter = Counter()
    for form in lexicon_set:
        for ch in form:
            counts[ch] += 1
    total = sum(counts.values())
    chars = sorted(counts.keys())
    probs = np.array([counts[c] / total for c in chars], dtype=np.float64)
    return np.array(chars), probs


# =====================================================================
# Generazione corpus sintetico (batch per lunghezza)
# =====================================================================

def generate_synthetic_corpus(
    lengths: np.ndarray,
    chars: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> list[str]:
    """Genera stringhe casuali con stesse lunghezze del corpus reale.

    Raggruppa per lunghezza per efficienza batch.
    """
    synthetic = [""] * len(lengths)
    unique_lens, inverse = np.unique(lengths, return_inverse=True)

    for i, length in enumerate(unique_lens):
        mask = inverse == i
        n_words = int(mask.sum())
        # genera matrice (n_words, length) di indici
        idx = rng.choice(len(chars), size=(n_words, int(length)), p=probs)
        # converti in stringhe
        positions = np.where(mask)[0]
        for j, pos in enumerate(positions):
            synthetic[pos] = "".join(chars[idx[j]])

    return synthetic


# =====================================================================
# Test 1: Match Rate — Real vs Synthetic
# =====================================================================

def run_match_rate_test(
    tokens: list[str],
    lexicon_set: set[str],
    chars: np.ndarray,
    probs: np.ndarray,
    n_iters: int = N_ITERS,
) -> dict:
    """Confronto match rate reale vs corpus sintetici."""
    # Match rate reale
    real_matched = sum(1 for t in tokens if t in lexicon_set)
    real_rate = real_matched / len(tokens)

    # Lunghezze per generazione
    lengths = np.array([len(t) for t in tokens])

    rng = np.random.default_rng(SEED)
    synth_rates = []

    for _ in range(n_iters):
        synth = generate_synthetic_corpus(lengths, chars, probs, rng)
        n_match = sum(1 for s in synth if s in lexicon_set)
        synth_rates.append(n_match / len(synth))

    synth_rates = np.array(synth_rates)
    synth_mean = float(synth_rates.mean())
    synth_std = float(synth_rates.std(ddof=1))

    if synth_std > 0:
        z_score = (real_rate - synth_mean) / synth_std
        p_value = float(1 - sp_norm.cdf(z_score))
    else:
        z_score = float("inf") if real_rate > synth_mean else 0.0
        p_value = 0.0

    return {
        "real_match_rate": round(real_rate, 5),
        "real_matched": real_matched,
        "total_tokens": len(tokens),
        "synth_mean": round(synth_mean, 5),
        "synth_std": round(synth_std, 5),
        "synth_min": round(float(synth_rates.min()), 5),
        "synth_max": round(float(synth_rates.max()), 5),
        "z_score": round(z_score, 2),
        "p_value": round(p_value, 6),
        "n_iters": n_iters,
        "ratio": round(real_rate / synth_mean, 2) if synth_mean > 0 else float("inf"),
        "significant": p_value < 0.05,
    }


# =====================================================================
# Test 2: Gloss Quality — Diversità semantica
# =====================================================================

def compute_gloss_metrics(
    matched_tokens: list[str],
    form_to_gloss: dict[str, str],
) -> dict:
    """Calcola metriche di qualità glosse per token matchati."""
    # Conta frequenza per tipo
    type_freq = Counter(matched_tokens)
    glosses = []
    gloss_lens = []

    for form, freq in type_freq.items():
        g = form_to_gloss.get(form, "")
        if g:
            glosses.extend([g] * freq)
            gloss_lens.append(len(g))

    if not glosses:
        return {
            "gloss_entropy": 0.0,
            "mean_gloss_len": 0.0,
            "top5_concentration": 1.0,
            "hapax_ratio": 1.0,
            "n_glossed_types": 0,
        }

    # Shannon entropy delle glosse (per tipo)
    gloss_type_freq = Counter(glosses)
    total = sum(gloss_type_freq.values())
    entropy = 0.0
    for count in gloss_type_freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Top 5 concentration (fraz. token nei 5 tipi più frequenti)
    top5 = sum(c for _, c in type_freq.most_common(5))
    top5_conc = top5 / sum(type_freq.values())

    # Hapax ratio
    n_hapax = sum(1 for c in type_freq.values() if c == 1)
    hapax_ratio = n_hapax / len(type_freq)

    return {
        "gloss_entropy": round(entropy, 4),
        "mean_gloss_len": round(float(np.mean(gloss_lens)) if gloss_lens else 0, 2),
        "top5_concentration": round(top5_conc, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "n_glossed_types": len([f for f in type_freq if f in form_to_gloss]),
    }


def run_gloss_quality_test(
    tokens: list[str],
    lexicon_set: set[str],
    form_to_gloss: dict[str, str],
    chars: np.ndarray,
    probs: np.ndarray,
    n_iters: int = N_ITERS,
) -> dict:
    """Confronto qualità glosse reali vs sintetiche."""
    # Metriche reali
    real_matched = [t for t in tokens if t in lexicon_set]
    real_metrics = compute_gloss_metrics(real_matched, form_to_gloss)

    lengths = np.array([len(t) for t in tokens])
    rng = np.random.default_rng(SEED + 1)

    synth_entropies = []
    synth_gloss_lens = []
    synth_top5s = []
    synth_hapax = []

    for _ in range(n_iters):
        synth = generate_synthetic_corpus(lengths, chars, probs, rng)
        synth_matched = [s for s in synth if s in lexicon_set]
        m = compute_gloss_metrics(synth_matched, form_to_gloss)
        synth_entropies.append(m["gloss_entropy"])
        synth_gloss_lens.append(m["mean_gloss_len"])
        synth_top5s.append(m["top5_concentration"])
        synth_hapax.append(m["hapax_ratio"])

    # z-score helper: raw direction (real - synth_mean) / synth_std
    def _z(real_val, synth_vals):
        arr = np.array(synth_vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        if std > 0:
            z = (real_val - mean) / std
        else:
            z = float("inf") if real_val > mean else (float("-inf") if real_val < mean else 0.0)
        return {
            "real": round(real_val, 4),
            "synth_mean": round(mean, 4),
            "synth_std": round(std, 4),
            "z_raw": round(z, 2),
        }

    # Per ogni metrica: definire direzione "language-like"
    # Entropy: LOWER = più concentrato = Zipfian = language-like → test one-sided z<0
    # Mean gloss len: HIGHER = glosse più ricche → test one-sided z>0
    # Top5 concentration: HIGHER = più concentrato = Zipfian → test one-sided z>0
    # Hapax ratio: LOWER = meno hapax = vocabolario stabile → test one-sided z<0

    def _finalize(raw, direction):
        """Converte z_raw in z_lang e p_value per la direzione language-like."""
        z_raw = raw["z_raw"]
        # z_lang: positivo = più language-like del random
        z_lang = z_raw if direction == "higher" else -z_raw
        if z_lang != float("inf") and z_lang != float("-inf"):
            p = float(1 - sp_norm.cdf(z_lang))
        else:
            p = 0.0 if z_lang == float("inf") else 1.0
        return {
            "real": raw["real"],
            "synth_mean": raw["synth_mean"],
            "synth_std": raw["synth_std"],
            "z_score": round(z_lang, 2),
            "p_value": round(p, 6),
            "significant": p < 0.05,
            "direction": f"{'lower' if direction == 'lower' else 'higher'} = language-like",
        }

    entropy = _finalize(_z(real_metrics["gloss_entropy"], synth_entropies), "lower")
    gloss_len = _finalize(_z(real_metrics["mean_gloss_len"], synth_gloss_lens), "higher")
    top5 = _finalize(_z(real_metrics["top5_concentration"], synth_top5s), "higher")
    hapax = _finalize(_z(real_metrics["hapax_ratio"], synth_hapax), "lower")

    # Significatività composita: quante metriche sono language-like?
    n_sig = sum(m["significant"] for m in [entropy, gloss_len, top5, hapax])
    composite_significant = n_sig >= 2  # majority rule

    return {
        "real_metrics": real_metrics,
        "entropy": entropy,
        "mean_gloss_len": gloss_len,
        "top5_concentration": top5,
        "hapax_ratio": hapax,
        "n_significant_sub": n_sig,
        "composite_significant": composite_significant,
        "n_iters": n_iters,
    }


# =====================================================================
# Test 3: Bigram Plausibility
# =====================================================================

def build_bigram_model(lexicon_set: set[str]) -> tuple[dict, set]:
    """Modello log-probabilità bigramma dal lessico con Laplace smoothing.

    Returns:
        (log_probs, alphabet) dove log_probs[(c1, c2)] = log P(c2|c1)
    """
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    alphabet: set = set()

    for form in lexicon_set:
        for ch in form:
            alphabet.add(ch)
        for i in range(len(form) - 1):
            bigram_counts[(form[i], form[i + 1])] += 1
            unigram_counts[form[i]] += 1

    V = len(alphabet)
    log_probs = {}
    for c1 in alphabet:
        denom = unigram_counts[c1] + LAPLACE_ALPHA * V
        for c2 in alphabet:
            count = bigram_counts.get((c1, c2), 0)
            log_probs[(c1, c2)] = math.log((count + LAPLACE_ALPHA) / denom)

    return log_probs, alphabet


def score_bigrams(token: str, log_probs: dict, alphabet: set) -> float | None:
    """Media log-likelihood per bigramma di un token.

    Returns None se token ha caratteri fuori dall'alfabeto o len < 2.
    """
    if len(token) < 2:
        return None
    scores = []
    for i in range(len(token) - 1):
        key = (token[i], token[i + 1])
        if key in log_probs:
            scores.append(log_probs[key])
        else:
            return None  # char fuori dall'alfabeto del lessico
    return sum(scores) / len(scores) if scores else None


def score_corpus_bigrams(
    tokens: list[str],
    log_probs: dict,
    alphabet: set,
) -> float:
    """Media log-likelihood bigramma per l'intero corpus."""
    scores = []
    for t in tokens:
        s = score_bigrams(t, log_probs, alphabet)
        if s is not None:
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def run_bigram_plausibility_test(
    tokens: list[str],
    lexicon_set: set[str],
    chars: np.ndarray,
    probs: np.ndarray,
    n_iters: int = N_ITERS,
) -> dict:
    """Confronto plausibilità bigramma reale vs sintetica."""
    log_probs, alphabet = build_bigram_model(lexicon_set)

    real_score = score_corpus_bigrams(tokens, log_probs, alphabet)

    lengths = np.array([len(t) for t in tokens])
    rng = np.random.default_rng(SEED + 2)
    synth_scores = []

    for _ in range(n_iters):
        synth = generate_synthetic_corpus(lengths, chars, probs, rng)
        synth_scores.append(score_corpus_bigrams(synth, log_probs, alphabet))

    synth_scores = np.array(synth_scores)
    synth_mean = float(synth_scores.mean())
    synth_std = float(synth_scores.std(ddof=1))

    if synth_std > 0:
        z_score = (real_score - synth_mean) / synth_std
        p_value = float(1 - sp_norm.cdf(z_score))
    else:
        z_score = float("inf") if real_score > synth_mean else 0.0
        p_value = 0.0

    return {
        "real_score": round(real_score, 6),
        "synth_mean": round(synth_mean, 6),
        "synth_std": round(synth_std, 6),
        "z_score": round(z_score, 2),
        "p_value": round(p_value, 6),
        "n_iters": n_iters,
        "significant": p_value < 0.05,
        "note": "higher = more Hebrew-like bigram structure",
    }


# =====================================================================
# Verdict
# =====================================================================

def derive_verdict(t1: dict, t2: dict, t3: dict) -> dict:
    """Verdetto aggregato dai 3 test."""
    sig_count = sum([
        t1["significant"],
        t2["composite_significant"],
        t3["significant"],
    ])

    if sig_count == 3:
        verdict = "ABOVE_NULL_MODEL"
        explanation = (
            "Tutti e 3 i test significativi. Il segnale ebraico è reale e "
            "strutturalmente diverso da stringhe casuali. Il gap verso la "
            "leggibilità è sintattico (ordine parole, morfologia), non lessicale."
        )
    elif sig_count >= 1:
        verdict = "MARGINAL"
        explanation = (
            f"Solo {sig_count}/3 test significativi. Il segnale potrebbe "
            "essere parzialmente un artefatto di distribuzione lunghezze e "
            "frequenze carattere."
        )
    else:
        verdict = "INDISTINGUISHABLE_FROM_NULL"
        explanation = (
            "Nessun test significativo. Il match rate è interamente spiegabile "
            "da stringhe casuali con stessa distribuzione di lunghezze e "
            "frequenze carattere."
        )

    return {
        "verdict": verdict,
        "significant_tests": sig_count,
        "total_tests": 3,
        "explanation": explanation,
    }


# =====================================================================
# Summary TXT
# =====================================================================

def format_summary(t1: dict, t2: dict, t3: dict, verdict: dict) -> str:
    """Formatta summary testuale."""
    lines = []
    lines.append("=" * 60)
    lines.append("NULL MODEL TEST — Phase 16B")
    lines.append("=" * 60)

    lines.append("")
    lines.append("Test 1: Match Rate (real vs synthetic)")
    lines.append(f"  Real:      {t1['real_match_rate']*100:.2f}% "
                 f"({t1['real_matched']:,}/{t1['total_tokens']:,})")
    lines.append(f"  Synthetic: {t1['synth_mean']*100:.2f}% "
                 f"± {t1['synth_std']*100:.2f}%")
    lines.append(f"  Ratio:     {t1['ratio']:.2f}x")
    lines.append(f"  z-score:   {t1['z_score']:.2f}  "
                 f"p={t1['p_value']:.6f}  "
                 f"{'SIGNIFICANT' if t1['significant'] else 'n.s.'}")

    lines.append("")
    lines.append("Test 2: Gloss Quality (language-likeness of matched words)")
    lines.append("  (z>0 = more language-like than random; direction corrected)")
    for metric_name, label in [
        ("entropy", "Gloss entropy (↓)"),
        ("mean_gloss_len", "Mean gloss length (↑)"),
        ("top5_concentration", "Top-5 concentration (↑)"),
        ("hapax_ratio", "Hapax ratio (↓)"),
    ]:
        m = t2[metric_name]
        sig = "SIG" if m["significant"] else "n.s."
        lines.append(f"  {label:26s}: real={m['real']:.4f}  "
                     f"synth={m['synth_mean']:.4f}±{m['synth_std']:.4f}  "
                     f"z={m['z_score']:.2f}  {sig}")
    lines.append(f"  Composite ({t2['n_significant_sub']}/4 sub-metrics): "
                 f"{'SIGNIFICANT' if t2['composite_significant'] else 'n.s.'}")

    lines.append("")
    lines.append("Test 3: Bigram Plausibility (Hebrew-like structure)")
    lines.append(f"  Real:      {t3['real_score']:.6f}")
    lines.append(f"  Synthetic: {t3['synth_mean']:.6f} "
                 f"± {t3['synth_std']:.6f}")
    lines.append(f"  z-score:   {t3['z_score']:.2f}  "
                 f"p={t3['p_value']:.6f}  "
                 f"{'SIGNIFICANT' if t3['significant'] else 'n.s.'}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"VERDICT: {verdict['verdict']} "
                 f"({verdict['significant_tests']}/{verdict['total_tests']} significant)")
    lines.append(f"  {verdict['explanation']}")
    lines.append("-" * 60)

    return "\n".join(lines) + "\n"


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False, **kwargs):
    """Null Model Test — il segnale ebraico è più che lunghezza?"""
    report_path = config.stats_dir / "null_model_test.json"
    summary_path = config.stats_dir / "null_model_test_summary.txt"

    if report_path.exists() and not force:
        click.echo("  Null Model Test report esiste. Usa --force per ri-eseguire.")
        return

    config.ensure_dirs()
    print_header("PHASE 16B — Null Model Test")

    # 1. Parse EVA corpus
    print_step("Parsing corpus EVA...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file non trovato: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {eva_data['total_words']:,} parole, {len(eva_data['pages'])} pagine")

    # 2. Decodifica corpus
    print_step("Decodifica corpus → token ebraici...")
    tokens = decode_corpus(eva_data["pages"])
    click.echo(f"    {len(tokens):,} token decodificati (len >= {MIN_LEN}, 0 unknown)")

    # 3. Carica lessico onesto
    print_step("Caricando lessico onesto (45K, no Sefaria-Corpus)...")
    honest_lex, form_to_gloss = load_honest_lexicon(config)
    click.echo(f"    Lessico onesto: {len(honest_lex):,} forme")

    # 4. Distribuzione caratteri
    print_step("Building character distribution dal lessico...")
    chars, probs = build_char_weights(honest_lex)
    click.echo(f"    {len(chars)} caratteri unici, "
               f"top-3: {', '.join(f'{chars[i]}={probs[i]:.3f}' for i in np.argsort(-probs)[:3])}")

    # 5. Test 1: Match Rate
    print_step(f"Test 1: Match Rate ({N_ITERS} iterazioni)...")
    t1 = run_match_rate_test(tokens, honest_lex, chars, probs)
    click.echo(f"    Real: {t1['real_match_rate']*100:.2f}%  "
               f"Synth: {t1['synth_mean']*100:.2f}%±{t1['synth_std']*100:.2f}%  "
               f"z={t1['z_score']:.1f}  ratio={t1['ratio']:.2f}x")

    # 6. Test 2: Gloss Quality
    print_step(f"Test 2: Gloss Quality ({N_ITERS} iterazioni)...")
    t2 = run_gloss_quality_test(tokens, honest_lex, form_to_gloss, chars, probs)
    click.echo(f"    {t2['n_significant_sub']}/4 sub-metrics significant "
               f"→ {'SIGNIFICANT' if t2['composite_significant'] else 'n.s.'}")
    ent = t2["entropy"]
    click.echo(f"    Entropy: real={ent['real']:.3f}  "
               f"synth={ent['synth_mean']:.3f}  "
               f"z={ent['z_score']:.1f} (lower=language-like)")

    # 7. Test 3: Bigram Plausibility
    print_step(f"Test 3: Bigram Plausibility ({N_ITERS} iterazioni)...")
    t3 = run_bigram_plausibility_test(tokens, honest_lex, chars, probs)
    click.echo(f"    Real: {t3['real_score']:.5f}  "
               f"Synth: {t3['synth_mean']:.5f}±{t3['synth_std']:.5f}  "
               f"z={t3['z_score']:.1f}")

    # 8. Verdict
    verdict = derive_verdict(t1, t2, t3)
    click.echo(f"\n    VERDICT: {verdict['verdict']} "
               f"({verdict['significant_tests']}/{verdict['total_tests']})")

    # 9. Salva report
    print_step("Salvataggio report...")
    report = {
        "test1_match_rate": t1,
        "test2_gloss_quality": t2,
        "test3_bigram_plausibility": t3,
        "verdict": verdict,
        "corpus": {
            "n_tokens": len(tokens),
            "n_types": len(set(tokens)),
            "min_len": MIN_LEN,
        },
        "lexicon": "honest_45K",
        "n_iters": N_ITERS,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    JSON: {report_path}")

    summary = format_summary(t1, t2, t3, verdict)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    click.echo(f"    TXT: {summary_path}")

    click.echo(f"\n{summary}")
