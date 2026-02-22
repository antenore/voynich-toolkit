"""DictaLM Calibration Experiment.

Blinded calibration of DictaLM's Hebrew word validation:
- 100 known valid Hebrew words (STEPBible, high-frequency)
- 100 known invalid forms (random consonantal strings not in any lexicon)
- 100 decoded Voynich forms (mix of DictaLM-valid, possible, invalid)

All 300 forms are shuffled and submitted without origin labels to assess
DictaLM's precision, recall, and false positive rate.

CLI: voynich --force dictalm-calibrate
"""

import json
import random
import time
from pathlib import Path

import click

from .config import ToolkitConfig
from .dictalm_validation import (
    API_KEY_PATH,
    DELAY_BETWEEN_BATCHES,
    HEBREW_UNICODE,
    load_api_key,
    parse_response,
    query_dictalm,
    test_connection,
    to_hebrew_unicode,
)
from .utils import print_header, print_step


# =====================================================================
# Gold standard construction
# =====================================================================


def build_gold_standard(config: ToolkitConfig, seed: int = 42) -> list[dict]:
    """Build blinded gold standard: 100 Hebrew + 100 random + 100 Voynich.

    Returns list of dicts with keys:
        consonantal, origin ('hebrew', 'random', 'voynich'), expected_valid
    """
    rng = random.Random(seed)

    # Load lexicon for Hebrew and Voynich forms
    with open(config.lexicon_dir / "lexicon_enriched.json") as f:
        lex_data = json.load(f)

    all_forms = set(lex_data["all_consonantal_forms"])

    # 1. Known valid Hebrew: STEPBible forms, high-frequency biblical roots
    by_domain = lex_data.get("by_domain", {})
    stepbible_forms = []
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if entry.get("source") == "STEPBible":
                form = entry.get("consonants", "")
                gloss = entry.get("gloss", "")
                if form and 3 <= len(form) <= 6 and gloss:
                    stepbible_forms.append(form)

    # Deduplicate and pick 100 random high-quality forms
    stepbible_unique = list(set(stepbible_forms))
    rng.shuffle(stepbible_unique)
    hebrew_100 = stepbible_unique[:100]

    # 2. Known invalid: random consonantal strings guaranteed NOT in any lexicon
    hebrew_chars = list("AbgdhwzXJyklmnsEpCqrSt")
    random_100 = []
    attempts = 0
    while len(random_100) < 100 and attempts < 10000:
        length = rng.choice([3, 4, 4, 5, 5])
        form = ''.join(rng.choices(hebrew_chars, k=length))
        if form not in all_forms and form not in random_100:
            random_100.append(form)
        attempts += 1

    # 3. Voynich forms: mix from DictaLM validation results
    voynich_path = config.stats_dir / "dictalm_validation.json"
    voynich_100 = []
    if voynich_path.exists():
        with open(voynich_path) as f:
            dictalm_data = json.load(f)
        words = dictalm_data.get("words", [])

        # Pick ~33 valid, ~34 possible, ~33 invalid
        valid = [w for w in words if w.get("valid") == "yes"]
        possible = [w for w in words if w.get("valid") == "possible"]
        invalid = [w for w in words if w.get("valid") == "no"]

        rng.shuffle(valid)
        rng.shuffle(possible)
        rng.shuffle(invalid)

        voynich_100 = (
            [w["consonantal"] for w in valid[:34]] +
            [w["consonantal"] for w in possible[:33]] +
            [w["consonantal"] for w in invalid[:33]]
        )
    else:
        # Fallback: use glossed words
        glossed_path = config.stats_dir / "glossed_words.json"
        if glossed_path.exists():
            with open(glossed_path) as f:
                glossed = json.load(f)
            forms = [w["consonantal"] for w in glossed if w.get("consonantal")]
            rng.shuffle(forms)
            voynich_100 = forms[:100]

    # Build combined gold standard
    gold = []
    for form in hebrew_100:
        gold.append({
            "consonantal": form,
            "origin": "hebrew",
            "expected_valid": True,
        })
    for form in random_100:
        gold.append({
            "consonantal": form,
            "origin": "random",
            "expected_valid": False,
        })
    for form in voynich_100:
        gold.append({
            "consonantal": form,
            "origin": "voynich",
            "expected_valid": None,  # unknown
        })

    # Shuffle to blind
    rng.shuffle(gold)
    return gold


# =====================================================================
# Blinded prompt (no gloss, no origin hint)
# =====================================================================


def build_blinded_prompt(words: list[dict]) -> str:
    """Build blinded validation prompt (no gloss, no frequency, no origin)."""
    lines = []
    for i, w in enumerate(words):
        heb_uni = to_hebrew_unicode(w["consonantal"])
        lines.append(f'{i+1}. {heb_uni} (ASCII: {w["consonantal"]})')

    word_list = "\n".join(lines)

    return f"""I have {len(words)} consonantal sequences (Hebrew alphabet, no nikkud/vowels). For each one, evaluate whether it could be a real Hebrew word.

For each sequence, provide a JSON object with these exact keys:
- "index": the number (1-based)
- "valid": "yes" if a real Hebrew word/root, "no" if not, "possible" if ambiguous
- "meaning": primary meaning in English (brief), or "" if not valid
- "confidence": "high", "medium", or "low"

IMPORTANT:
- These are CONSONANTAL skeletons only
- Be CRITICAL: if a form is just random consonants, say valid="no"
- Do NOT assume these are Hebrew — some may be nonsense
- Give the MOST COMMON reading if multiple are possible

Respond with ONLY a JSON array (no markdown, no explanation).

Sequences:
{word_list}"""


# =====================================================================
# Metrics computation
# =====================================================================


def compute_metrics(gold: list[dict], results: list[dict]) -> dict:
    """Compute precision, recall, FPR, confusion matrix."""
    # Merge results with gold standard by index
    merged = []
    for g, r in zip(gold, results):
        merged.append({**g, **r})

    # Confusion matrix: ground truth (hebrew/random) vs DictaLM prediction
    tp = fp = fn = tn = 0
    voynich_accepted = voynich_rejected = voynich_possible = 0

    for m in merged:
        origin = m.get("origin", "")
        predicted = m.get("valid", "")

        if origin == "hebrew":
            if predicted in ("yes", "possible"):
                tp += 1
            else:
                fn += 1
        elif origin == "random":
            if predicted in ("yes", "possible"):
                fp += 1
            else:
                tn += 1
        elif origin == "voynich":
            if predicted == "yes":
                voynich_accepted += 1
            elif predicted == "no":
                voynich_rejected += 1
            else:
                voynich_possible += 1

    # Strict: only "yes" counts as positive
    tp_strict = sum(1 for m in merged
                    if m["origin"] == "hebrew" and m.get("valid") == "yes")
    fp_strict = sum(1 for m in merged
                    if m["origin"] == "random" and m.get("valid") == "yes")
    fn_strict = sum(1 for m in merged
                    if m["origin"] == "hebrew" and m.get("valid") != "yes")
    tn_strict = sum(1 for m in merged
                    if m["origin"] == "random" and m.get("valid") != "yes")

    n_hebrew = sum(1 for m in merged if m["origin"] == "hebrew")
    n_random = sum(1 for m in merged if m["origin"] == "random")
    n_voynich = sum(1 for m in merged if m["origin"] == "voynich")

    precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) else 0
    recall_strict = tp_strict / n_hebrew if n_hebrew else 0
    fpr_strict = fp_strict / n_random if n_random else 0

    precision_lenient = tp / (tp + fp) if (tp + fp) else 0
    recall_lenient = tp / n_hebrew if n_hebrew else 0
    fpr_lenient = fp / n_random if n_random else 0

    return {
        "n_hebrew": n_hebrew,
        "n_random": n_random,
        "n_voynich": n_voynich,
        "strict": {
            "precision": round(precision_strict, 4),
            "recall": round(recall_strict, 4),
            "fpr": round(fpr_strict, 4),
            "tp": tp_strict,
            "fp": fp_strict,
            "fn": fn_strict,
            "tn": tn_strict,
        },
        "lenient": {
            "note": "yes+possible counted as positive",
            "precision": round(precision_lenient, 4),
            "recall": round(recall_lenient, 4),
            "fpr": round(fpr_lenient, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "voynich": {
            "accepted": voynich_accepted,
            "possible": voynich_possible,
            "rejected": voynich_rejected,
            "acceptance_rate": round(
                voynich_accepted / n_voynich, 4) if n_voynich else 0,
        },
        "merged": merged,
    }


# =====================================================================
# Entry point
# =====================================================================


def run(config: ToolkitConfig, force: bool = False, batch_size: int = 10):
    """Run DictaLM blinded calibration experiment."""
    out_path = config.stats_dir / "dictalm_calibration.json"
    txt_path = config.stats_dir / "dictalm_calibration.txt"

    if out_path.exists() and not force:
        click.echo(f"  Output exists: {out_path} (use --force)")
        return

    config.ensure_dirs()
    print_header("DictaLM Calibration Experiment")

    # Load API key
    print_step("Loading API key")
    api_key = load_api_key()

    # Test connection
    print_step("Testing API connection")
    if not test_connection(api_key):
        raise click.ClickException("Cannot connect to Featherless API.")

    # Build gold standard
    print_step("Building blinded gold standard (300 forms)")
    gold = build_gold_standard(config, seed=42)
    n_heb = sum(1 for g in gold if g["origin"] == "hebrew")
    n_rand = sum(1 for g in gold if g["origin"] == "random")
    n_voyn = sum(1 for g in gold if g["origin"] == "voynich")
    click.echo(f"    Hebrew: {n_heb}, Random: {n_rand}, Voynich: {n_voyn}")

    # Submit in batches (blinded)
    print_step(f"Submitting {len(gold)} forms in batches of {batch_size}")
    all_results = []
    n_batches = (len(gold) + batch_size - 1) // batch_size
    errors = 0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        batch = gold[start:start + batch_size]

        click.echo(f"  Batch {batch_idx + 1}/{n_batches}...", nl=False)

        try:
            prompt = build_blinded_prompt(batch)
            raw = query_dictalm(prompt, api_key, max_tokens=2000,
                                temperature=0.1)
            parsed = parse_response(raw)

            if len(parsed) == 1 and parsed[0].get("parse_error"):
                click.echo(" PARSE ERROR")
                errors += 1
                all_results.extend([{"valid": "error"}] * len(batch))
            else:
                # Pad if fewer results than batch
                while len(parsed) < len(batch):
                    parsed.append({"valid": "error"})
                all_results.extend(parsed[:len(batch)])
                click.echo(f" OK ({len(parsed)} results)")

        except Exception as e:
            click.echo(f" ERROR: {e}")
            errors += 1
            all_results.extend([{"valid": "error"}] * len(batch))

        if batch_idx < n_batches - 1:
            time.sleep(DELAY_BETWEEN_BATCHES)

    click.echo(f"  Done: {len(all_results)} results, {errors} errors")

    # Compute metrics
    print_step("Computing calibration metrics")
    metrics = compute_metrics(gold, all_results)

    click.echo(f"\n  === CALIBRATION RESULTS ===")
    click.echo(f"  Strict (only 'yes' = positive):")
    s = metrics["strict"]
    click.echo(f"    Precision: {s['precision']*100:.1f}%  "
               f"Recall: {s['recall']*100:.1f}%  "
               f"FPR: {s['fpr']*100:.1f}%")
    click.echo(f"    TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}")

    click.echo(f"  Lenient (yes+possible = positive):")
    le = metrics["lenient"]
    click.echo(f"    Precision: {le['precision']*100:.1f}%  "
               f"Recall: {le['recall']*100:.1f}%  "
               f"FPR: {le['fpr']*100:.1f}%")

    click.echo(f"  Voynich forms (blinded):")
    v = metrics["voynich"]
    click.echo(f"    Accepted: {v['accepted']}/{metrics['n_voynich']} "
               f"({v['acceptance_rate']*100:.1f}%)")
    click.echo(f"    Possible: {v['possible']}, Rejected: {v['rejected']}")

    # Save JSON
    output = {
        "experiment": "DictaLM blinded calibration",
        "n_forms": len(gold),
        "batch_size": batch_size,
        "errors": errors,
        "metrics": {k: v for k, v in metrics.items() if k != "merged"},
        "details": [
            {k: v for k, v in m.items() if k != "expected_valid"}
            for m in metrics["merged"]
        ],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    click.echo(f"\n  JSON: {out_path}")

    # Save TXT
    lines = [
        "DictaLM Calibration Experiment",
        "=" * 60,
        f"Forms: {n_heb} Hebrew + {n_rand} random + {n_voyn} Voynich = {len(gold)}",
        f"Errors: {errors}",
        "",
        "STRICT (only 'yes' = positive):",
        f"  Precision:  {s['precision']*100:.1f}%",
        f"  Recall:     {s['recall']*100:.1f}%",
        f"  FPR:        {s['fpr']*100:.1f}%",
        f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}",
        "",
        "LENIENT (yes+possible = positive):",
        f"  Precision:  {le['precision']*100:.1f}%",
        f"  Recall:     {le['recall']*100:.1f}%",
        f"  FPR:        {le['fpr']*100:.1f}%",
        "",
        "VOYNICH (blinded):",
        f"  Accepted:  {v['accepted']}/{metrics['n_voynich']}",
        f"  Possible:  {v['possible']}",
        f"  Rejected:  {v['rejected']}",
    ]

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    click.echo(f"  TXT: {txt_path}")
