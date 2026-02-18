"""
Test Naibbe-style verbose homophonic cipher hypothesis.

The Naibbe cipher (Greshko 2025, Cryptologia) is a verbose homophonic
substitution that encrypts Latin/Italian into Voynich-like ciphertext.
If the VMS uses this cipher, our Hebrew signal (z=4.0) should be spurious.

This module tests that hypothesis with:
1. Diagnostic statistics (IC, entropy, Gini) vs mono/homo expected ranges
2. Monte Carlo simulation: encrypt Italian with verbose cipher → apply
   Hebrew mapping → measure match rate
3. Aggregated scorecard with verdict
"""

import json
import random
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew
from .utils import print_header, print_step
from .word_structure import parse_eva_words


# =====================================================================
# Expected ranges from literature (mono vs homophonic)
# =====================================================================

DIAGNOSTIC_RANGES = {
    "IC": {
        "mono": (0.060, 0.085),
        "homo": (0.035, 0.050),
    },
    "Gini": {
        "mono": (0.30, 0.55),
        "homo": (0.05, 0.20),
    },
    "H1/H0": {
        "mono": (0.50, 0.75),
        "homo": (0.80, 0.98),
    },
    "hapax_ratio": {
        "mono": (0.45, 0.65),
        "homo": (0.60, 0.85),
    },
}

# EVA characters that are mappable (17 single chars, excluding q/i digraphs)
MAPPABLE_EVA = list("acdefghklmnoprsty")


# =====================================================================
# Load existing diagnostics from JSON
# =====================================================================

def load_existing_diagnostics(config: ToolkitConfig) -> dict:
    """Load pre-computed statistics from existing JSON output files."""
    stats_dir = config.stats_dir
    diag = {}

    # cipher_hypothesis.json → IC, Gini, H1/H0
    path = stats_dir / "cipher_hypothesis.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        diag["IC"] = data.get("ioc")
        ad = data.get("alphabet_diagnostics", {})
        diag["Gini"] = ad.get("gini")
        diag["effective_alphabet"] = ad.get("effective_95")
        ent = data.get("entropy", {})
        diag["H1/H0"] = ent.get("h1h0_ratio")

    # word_structure.json → hapax_ratio, zipf_slope, avg_word_length
    path = stats_dir / "word_structure.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        wf = data.get("word_frequencies", {})
        hapax = wf.get("hapax_count", 0)
        unique = wf.get("unique_words", 1)
        diag["hapax_ratio"] = hapax / unique if unique else 0
        diag["zipf_slope"] = wf.get("zipf_slope")
        wl = data.get("word_length", {})
        diag["avg_word_length"] = wl.get("avg_word_length")
        diag["total_chars"] = data.get("total_chars")
        # word length distribution for segmentation
        diag["word_length_dist"] = {
            int(d["length"]): d["count"]
            for d in wl.get("distribution", [])
            if isinstance(d, dict) and "length" in d and "count" in d
        }

    # currier_split.json → permutation z-scores
    path = stats_dir / "currier_split.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        pt = data.get("permutation_tests", {})
        diag["currier_A_z"] = pt.get("A", {}).get("z_score")
        diag["currier_B_z"] = pt.get("B", {}).get("z_score")

    # semantic_coherence.json → permutation z-scores
    path = stats_dir / "semantic_coherence.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        pt = data.get("permutation_test", {})
        diag["semantic_max_consec_z"] = (
            pt.get("max_consecutive", {}).get("z_score")
        )
        diag["semantic_n_high_z"] = (
            pt.get("n_high_lines", {}).get("z_score")
        )

    # cross_language_report.json → real match rate
    path = stats_dir / "cross_language_report.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        heb = data.get("lexicons", {}).get("hebrew", {})
        diag["real_match_rate"] = heb.get("match_rate")
        diag["real_n_total"] = heb.get("n_total")

    return diag


# =====================================================================
# Classification
# =====================================================================

def classify_statistic(value, mono_range, homo_range):
    """Classify a value vs expected monoalphabetic / homophonic ranges.

    Returns 'mono', 'homo', or 'ambiguous'.
    """
    if value is None:
        return "unknown"
    mono_lo, mono_hi = mono_range
    homo_lo, homo_hi = homo_range
    in_mono = mono_lo <= value <= mono_hi
    in_homo = homo_lo <= value <= homo_hi
    if in_mono and not in_homo:
        return "mono"
    if in_homo and not in_mono:
        return "homo"
    if in_mono and in_homo:
        return "ambiguous"
    # Outside both ranges — classify by distance to nearest range
    d_mono = min(abs(value - mono_lo), abs(value - mono_hi))
    d_homo = min(abs(value - homo_lo), abs(value - homo_hi))
    return "mono" if d_mono < d_homo else "homo"


# =====================================================================
# Verbose cipher simulation
# =====================================================================

def generate_verbose_table(eva_chars, rng, n_homo=(2, 4), seq_len=(1, 2)):
    """Generate a random verbose homophonic table.

    Maps 26 Italian lowercase letters to 2-4 EVA sequences each,
    where each sequence is 1-2 EVA chars long.
    """
    table = {}
    for letter in string.ascii_lowercase:
        n = rng.randint(*n_homo)
        seqs = []
        for _ in range(n):
            slen = rng.randint(*seq_len)
            seq = "".join(rng.choices(eva_chars, k=slen))
            seqs.append(seq)
        table[letter] = seqs
    return table


def generate_italian_text(italian_forms, target_len, rng):
    """Sample Italian words, concatenate characters up to target length.

    Returns a string of lowercase Italian letters (a-z only).
    """
    chars = []
    total = 0
    while total < target_len:
        word = rng.choice(italian_forms)
        # Keep only a-z chars
        clean = "".join(c for c in word.lower() if c in string.ascii_lowercase)
        if clean:
            chars.append(clean)
            total += len(clean)
    return "".join(chars)[:target_len]


def verbose_encrypt(text_chars, table, rng):
    """Encrypt text character-by-character using the verbose table.

    For each character, randomly choose one of its EVA sequences.
    Returns concatenated EVA string.
    """
    parts = []
    for ch in text_chars:
        if ch in table:
            seq = rng.choice(table[ch])
            parts.append(seq)
        # Skip characters not in table (non-alphabetic)
    return "".join(parts)


def segment_into_words(eva_stream, length_dist, rng):
    """Segment an EVA stream into words following a length distribution.

    length_dist: dict {length: count} — used as weights for sampling.
    Returns list of EVA words.
    """
    lengths = list(length_dist.keys())
    weights = [length_dist[l] for l in lengths]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    words = []
    pos = 0
    stream_len = len(eva_stream)
    while pos < stream_len:
        wlen = rng.choices(lengths, weights=probs, k=1)[0]
        if pos + wlen > stream_len:
            wlen = stream_len - pos
        if wlen > 0:
            words.append(eva_stream[pos:pos + wlen])
        pos += wlen
    return words


def run_naibbe_simulation(italian_forms, length_dist, hebrew_set,
                          real_rate, n_sims=200, seed=42):
    """Monte Carlo simulation of Naibbe-style encryption + Hebrew decode.

    For each simulation:
    1. Generate Italian text (~191K chars)
    2. Generate a random verbose table
    3. Encrypt the text
    4. Segment into words
    5. Decode each word to Hebrew
    6. Check match rate vs Hebrew lexicon

    Returns dict with simulation results.
    """
    rng = random.Random(seed)
    target_len = 191_000  # approximate EVA corpus character count

    sim_rates = []
    t0 = time.time()

    for i in range(n_sims):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      sim {i + 1}/{n_sims}  ({elapsed:.1f}s)")

        # 1. Generate Italian text
        text = generate_italian_text(italian_forms, target_len, rng)

        # 2. Generate verbose table
        table = generate_verbose_table(MAPPABLE_EVA, rng)

        # 3. Encrypt
        eva_stream = verbose_encrypt(text, table, rng)

        # 4. Segment into words
        words = segment_into_words(eva_stream, length_dist, rng)

        # 5-6. Decode and match
        n_decoded = 0
        n_matched = 0
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and all(c is not None for c in heb):
                n_decoded += 1
                if heb in hebrew_set:
                    n_matched += 1

        rate = n_matched / n_decoded if n_decoded else 0
        sim_rates.append(rate)

    sim_rates = np.array(sim_rates)
    sim_mean = float(np.mean(sim_rates))
    sim_std = float(np.std(sim_rates))
    z_score = (real_rate - sim_mean) / sim_std if sim_std > 0 else float("inf")

    # One-sided p-value: how often does simulation exceed real rate?
    n_above = int(np.sum(sim_rates >= real_rate))
    p_value = (n_above + 1) / (n_sims + 1)

    return {
        "n_sims": n_sims,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_min": float(np.min(sim_rates)),
        "sim_max": float(np.max(sim_rates)),
        "sim_median": float(np.median(sim_rates)),
        "real_rate": real_rate,
        "z_score": z_score,
        "p_value": p_value,
        "n_above_real": n_above,
        "elapsed_s": time.time() - t0,
    }


# =====================================================================
# Scorecard
# =====================================================================

def build_scorecard(diagnostics, sim_results):
    """Build aggregated scorecard: each piece of evidence classified."""
    rows = []

    # Diagnostic statistics
    for stat_name in ("IC", "Gini", "H1/H0", "hapax_ratio"):
        val = diagnostics.get(stat_name)
        ranges = DIAGNOSTIC_RANGES.get(stat_name, {})
        verdict = classify_statistic(
            val,
            ranges.get("mono", (0, 0)),
            ranges.get("homo", (0, 0)),
        )
        rows.append({
            "metric": stat_name,
            "value": f"{val:.4f}" if val is not None else "N/A",
            "mono_range": f"{ranges.get('mono', ('?','?'))}",
            "homo_range": f"{ranges.get('homo', ('?','?'))}",
            "verdict": verdict,
        })

    # Simulation: does Naibbe explain the Hebrew match rate?
    z = sim_results.get("z_score", 0)
    p = sim_results.get("p_value", 1)
    if z > 2.0 and p < 0.05:
        sim_verdict = "mono"  # real rate exceeds Naibbe → signal is real
    elif z < 1.0:
        sim_verdict = "homo"  # Naibbe explains the rate
    else:
        sim_verdict = "ambiguous"

    rows.append({
        "metric": "naibbe_simulation",
        "value": f"z={z:.2f}, p={p:.4f}",
        "mono_range": "z>2.0",
        "homo_range": "z<1.0",
        "verdict": sim_verdict,
    })

    # Permutation z-scores: high z → mapping carries real structure
    for key, label in [
        ("currier_A_z", "currier_A_perm"),
        ("currier_B_z", "currier_B_perm"),
        ("semantic_max_consec_z", "semantic_max_consec"),
        ("semantic_n_high_z", "semantic_n_high"),
    ]:
        z_val = diagnostics.get(key)
        if z_val is not None:
            # z>2 suggests real structure beyond noise
            v = "mono" if z_val > 3.0 else ("ambiguous" if z_val > 2.0 else "homo")
            rows.append({
                "metric": label,
                "value": f"z={z_val:.2f}",
                "mono_range": "z>3.0",
                "homo_range": "z<2.0",
                "verdict": v,
            })

    return rows


def final_verdict(scorecard):
    """Determine overall verdict from scorecard."""
    counts = Counter(r["verdict"] for r in scorecard)
    n_mono = counts.get("mono", 0)
    n_homo = counts.get("homo", 0)
    n_amb = counts.get("ambiguous", 0)
    total = len(scorecard)

    if n_mono >= total * 0.6:
        verdict = "MONOALPHABETIC FAVORED"
        detail = (f"The Voynich text behaves like a monoalphabetic cipher "
                  f"({n_mono}/{total} indicators favor mono). The Naibbe "
                  f"verbose cipher hypothesis is not supported.")
    elif n_homo >= total * 0.6:
        verdict = "HOMOPHONIC FAVORED"
        detail = (f"The Voynich text is consistent with verbose homophonic "
                  f"encryption ({n_homo}/{total} indicators favor homo). "
                  f"The Hebrew signal may be a Naibbe artifact.")
    else:
        verdict = "INCONCLUSIVE"
        detail = (f"Mixed evidence: {n_mono} mono, {n_homo} homo, "
                  f"{n_amb} ambiguous out of {total}. Neither hypothesis "
                  f"is clearly favored.")

    return {
        "verdict": verdict,
        "detail": detail,
        "n_mono": n_mono,
        "n_homo": n_homo,
        "n_ambiguous": n_amb,
        "n_total": total,
    }


# =====================================================================
# Output formatting
# =====================================================================

def format_summary(diagnostics, sim_results, scorecard, verdict_info):
    """Format human-readable summary."""
    lines = []
    lines.append("=" * 65)
    lines.append("  NAIBBE VERBOSE CIPHER HYPOTHESIS TEST")
    lines.append("=" * 65)

    lines.append("\n  1. DIAGNOSTIC STATISTICS")
    lines.append("  " + "-" * 61)
    lines.append(f"  {'Metric':<20} {'Value':>10} {'Mono range':>16} "
                 f"{'Homo range':>16} {'Verdict':>10}")
    lines.append("  " + "-" * 61)
    for r in scorecard:
        if r["metric"] in ("IC", "Gini", "H1/H0", "hapax_ratio"):
            lines.append(f"  {r['metric']:<20} {r['value']:>10} "
                         f"{r['mono_range']:>16} {r['homo_range']:>16} "
                         f"{r['verdict']:>10}")

    lines.append(f"\n  Additional: avg_word_length = "
                 f"{diagnostics.get('avg_word_length', 'N/A')}, "
                 f"zipf_slope = {diagnostics.get('zipf_slope', 'N/A')}")

    lines.append("\n  2. NAIBBE MONTE CARLO SIMULATION")
    lines.append("  " + "-" * 61)
    sr = sim_results
    lines.append(f"  Simulations:    {sr['n_sims']}")
    lines.append(f"  Sim match rate: {sr['sim_mean']:.4f} "
                 f"+/- {sr['sim_std']:.4f} "
                 f"(range {sr['sim_min']:.4f} - {sr['sim_max']:.4f})")
    lines.append(f"  Real match rate: {sr['real_rate']:.4f}")
    lines.append(f"  z-score:        {sr['z_score']:.2f}")
    lines.append(f"  p-value:        {sr['p_value']:.4f}")
    lines.append(f"  Elapsed:        {sr['elapsed_s']:.1f}s")

    sim_entry = next((r for r in scorecard if r["metric"] == "naibbe_simulation"), None)
    if sim_entry:
        lines.append(f"  Verdict:        {sim_entry['verdict'].upper()}")

    lines.append("\n  3. PERMUTATION EVIDENCE")
    lines.append("  " + "-" * 61)
    for r in scorecard:
        if r["metric"] not in ("IC", "Gini", "H1/H0", "hapax_ratio",
                                "naibbe_simulation"):
            lines.append(f"  {r['metric']:<30} {r['value']:>10}  → "
                         f"{r['verdict']}")

    lines.append("\n  4. AGGREGATE VERDICT")
    lines.append("  " + "=" * 61)
    vi = verdict_info
    lines.append(f"  {vi['verdict']}")
    lines.append(f"  Score: {vi['n_mono']} mono / {vi['n_homo']} homo / "
                 f"{vi['n_ambiguous']} ambiguous (of {vi['n_total']})")
    lines.append(f"\n  {vi['detail']}")
    lines.append("  " + "=" * 61)

    return "\n".join(lines)


# =====================================================================
# Load Italian lexicon
# =====================================================================

def load_italian_forms(config: ToolkitConfig):
    """Load Italian word forms from the Italian lexicon."""
    path = config.lexicon_dir / "italian_lexicon.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    forms = set()
    by_domain = data.get("by_domain", data)
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            word = entry.get("word", "")
            if word:
                forms.add(word.lower())
    return list(forms)


def load_hebrew_set(config: ToolkitConfig):
    """Load Hebrew lexicon as a set of consonantal forms."""
    path = config.hebrew_lexicon_path
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    forms = set()
    by_domain = data.get("by_domain", data)
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            c = entry.get("consonants", "")
            if c:
                forms.add(c)
    return forms


# =====================================================================
# Main entry point
# =====================================================================

def run(config: ToolkitConfig, force=False):
    """Run Naibbe verbose cipher hypothesis test."""
    out_json = config.stats_dir / "naibbe_test.json"
    out_txt = config.stats_dir / "naibbe_test_summary.txt"

    if not force and out_json.exists():
        print(f"  Output exists: {out_json.name} (use --force to re-run)")
        return

    config.ensure_dirs()
    print_header("NAIBBE VERBOSE CIPHER HYPOTHESIS TEST")

    # 1. Load existing diagnostics
    print_step("Loading existing diagnostics from JSON files...")
    diagnostics = load_existing_diagnostics(config)
    for key in ("IC", "Gini", "H1/H0", "hapax_ratio", "real_match_rate"):
        val = diagnostics.get(key)
        print(f"      {key}: {val}")

    # 2. Parse EVA for word length distribution
    print_step("Parsing EVA corpus for word length distribution...")
    eva_path = config.eva_data_dir / "LSI_ivtff_0d.txt"
    parsed = parse_eva_words(eva_path)
    if not diagnostics.get("word_length_dist"):
        wlens = Counter(len(w) for w in parsed["words"])
        diagnostics["word_length_dist"] = dict(wlens)
    length_dist = diagnostics["word_length_dist"]
    print(f"      Word lengths: {len(length_dist)} distinct values, "
          f"corpus {parsed['total_words']} words")

    # 3. Load Italian lexicon
    print_step("Loading Italian lexicon...")
    italian_forms = load_italian_forms(config)
    print(f"      {len(italian_forms)} Italian forms")
    if len(italian_forms) < 100:
        print("      WARNING: Italian lexicon too small, using synthetic fallback")
        # Generate simple Italian-like words as fallback
        rng = random.Random(42)
        vowels = "aeiou"
        consonants = "bcdfglmnprstvz"
        italian_forms = []
        for _ in range(10000):
            wlen = rng.randint(3, 8)
            w = ""
            for j in range(wlen):
                if j % 2 == 0:
                    w += rng.choice(consonants)
                else:
                    w += rng.choice(vowels)
            italian_forms.append(w)

    # 4. Load Hebrew lexicon
    print_step("Loading Hebrew lexicon...")
    hebrew_set = load_hebrew_set(config)
    print(f"      {len(hebrew_set)} Hebrew consonantal forms")

    # 5. Get real match rate
    real_rate = diagnostics.get("real_match_rate", 0.4033)
    print(f"\n      Real Hebrew match rate: {real_rate:.4f}")

    # 6. Run Naibbe simulation
    print_step("Running Naibbe Monte Carlo simulation (200 iterations)...")
    sim_results = run_naibbe_simulation(
        italian_forms=italian_forms,
        length_dist=length_dist,
        hebrew_set=hebrew_set,
        real_rate=real_rate,
        n_sims=200,
        seed=42,
    )
    print(f"      Sim mean: {sim_results['sim_mean']:.4f} "
          f"+/- {sim_results['sim_std']:.4f}")
    print(f"      z-score: {sim_results['z_score']:.2f}, "
          f"p-value: {sim_results['p_value']:.4f}")

    # 7. Build scorecard
    print_step("Building scorecard...")
    scorecard = build_scorecard(diagnostics, sim_results)
    verdict_info = final_verdict(scorecard)
    print(f"      Verdict: {verdict_info['verdict']}")

    # 8. Format and save
    summary_txt = format_summary(diagnostics, sim_results, scorecard,
                                 verdict_info)
    print(f"\n{summary_txt}")

    # Save JSON
    output = {
        "diagnostics": {k: v for k, v in diagnostics.items()
                        if k != "word_length_dist"},
        "word_length_dist": diagnostics.get("word_length_dist", {}),
        "simulation": sim_results,
        "scorecard": [r for r in scorecard],
        "verdict": verdict_info,
    }
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  Saved: {out_json.name}")

    # Save TXT
    out_txt.write_text(summary_txt, encoding="utf-8")
    print(f"  Saved: {out_txt.name}")
