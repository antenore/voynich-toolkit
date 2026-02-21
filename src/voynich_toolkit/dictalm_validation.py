"""DictaLM Hebrew validation via Featherless.ai API.

Phase 20: Independent validation of decoded Hebrew forms using DictaLM,
a Hebrew-specialized LLM trained on 200B tokens of Hebrew text.

Sends our decoded consonantal forms to DictaLM for linguistic validation:
- Is this a real Hebrew word?
- What does it mean?
- What period (biblical/mishnaic/medieval)?
- Is our gloss correct?

API: Featherless.ai (OpenAI-compatible), model: dicta-il/DictaLM-3.0-24B-Thinking
Key: ~/.config/featherless/api_key
"""

import json
import time
from pathlib import Path

import click
import requests

from .config import ToolkitConfig
from .utils import print_header, print_step


# =====================================================================
# Constants
# =====================================================================

API_BASE = "https://api.featherless.ai/v1"
# 24B-Thinking is too slow (spends all tokens on reasoning, never outputs JSON)
# 24B-Base has cold start issues on serverless
# 1.7B-Instruct: fast, reliable, good quality with critical prompts
MODEL = "dicta-il/DictaLM-3.0-1.7B-Instruct"
API_KEY_PATH = Path.home() / ".config" / "featherless" / "api_key"

BATCH_SIZE = 10
DELAY_BETWEEN_BATCHES = 2.0  # seconds — be nice to the API

# Hebrew ASCII → Unicode
HEBREW_UNICODE = {
    'A': '\u05d0', 'b': '\u05d1', 'g': '\u05d2', 'd': '\u05d3',
    'h': '\u05d4', 'w': '\u05d5', 'z': '\u05d6', 'X': '\u05d7',
    'J': '\u05d8', 'y': '\u05d9', 'k': '\u05db', 'l': '\u05dc',
    'm': '\u05de', 'n': '\u05e0', 's': '\u05e1', 'E': '\u05e2',
    'p': '\u05e4', 'C': '\u05e6', 'q': '\u05e7', 'r': '\u05e8',
    'S': '\u05e9', 't': '\u05ea',
}


# =====================================================================
# API utilities
# =====================================================================


def load_api_key() -> str:
    """Load Featherless API key from secure local file."""
    if not API_KEY_PATH.exists():
        raise click.ClickException(
            f"API key not found: {API_KEY_PATH}\n"
            "  Save your Featherless.ai key there (chmod 600)."
        )
    return API_KEY_PATH.read_text().strip()


def to_hebrew_unicode(ascii_hebrew: str) -> str:
    """Convert our ASCII Hebrew encoding to Unicode Hebrew."""
    return ''.join(HEBREW_UNICODE.get(c, c) for c in ascii_hebrew)


def query_dictalm(prompt: str, api_key: str,
                   max_tokens: int = 3000,
                   temperature: float = 0.1) -> str:
    """Send a prompt to DictaLM via Featherless OpenAI-compatible API."""
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": (
                    "You are an expert in Hebrew linguistics covering all periods: "
                    "biblical, mishnaic (Talmudic), medieval, and modern Hebrew. "
                    "You also know Aramaic and Judeo-Italian. "
                    "Always respond in valid JSON when asked."
                )},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def test_connection(api_key: str) -> bool:
    """Test API connection with a simple query."""
    try:
        result = query_dictalm(
            "Is the Hebrew word שלום a real word? Reply with just: yes or no.",
            api_key, max_tokens=50, temperature=0.0,
        )
        print(f"  API test response: {result.strip()[:100]}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  API test FAILED: {e}")
        return False


# =====================================================================
# Prompt construction
# =====================================================================


def build_word_batch_prompt(words: list[dict]) -> str:
    """Build validation prompt for a batch of words."""
    lines = []
    for i, w in enumerate(words):
        heb_uni = to_hebrew_unicode(w["consonantal"])
        gloss = w.get("gloss", "")
        # Truncate long glosses
        if len(gloss) > 60:
            gloss = gloss[:57] + "..."
        lines.append(f'{i+1}. {heb_uni} (ASCII: {w["consonantal"]}) '
                     f'freq={w["freq"]}, our_gloss="{gloss}"')

    word_list = "\n".join(lines)

    return f"""I have {len(words)} consonantal Hebrew forms (no nikkud/vowels) from a computational decipherment project. For each one, validate it as a Hebrew linguist.

For each word, provide a JSON object with these exact keys:
- "index": the word number (1-based)
- "valid": "yes" if a real Hebrew word/root, "no" if not, "possible" if ambiguous
- "meaning": primary meaning in English (brief), or "" if not valid
- "period": "biblical", "mishnaic", "medieval", "modern", "aramaic", or "uncertain"
- "our_gloss_correct": "yes", "no", "partial", or "unknown"
- "correct_meaning": the correct meaning if our gloss is wrong, otherwise ""
- "confidence": "high", "medium", or "low"
- "notes": brief notes (alternative readings, homographs, etc.), or ""

IMPORTANT — be CRITICAL:
- These are CONSONANTAL skeletons — multiple vocalizations may be possible
- Give the MOST COMMON reading
- If our gloss is wrong, say our_gloss_correct="no" and give the correct meaning
- If a form is just random consonants that don't form a real word, say valid="no"
- "freq" is how many times it appears in the Voynich Manuscript decoded text
- Do NOT be generous — if a form is questionable, say valid="possible" not "yes"

Respond with ONLY a JSON array (no markdown, no explanation before/after).

Words:
{word_list}"""


# =====================================================================
# Response parsing
# =====================================================================


def parse_response(raw: str) -> list[dict]:
    """Parse DictaLM response, handling thinking tags, loops, and fences.

    The Thinking model often repeats JSON arrays multiple times.
    We extract the FIRST valid JSON array from the response.
    """
    import re

    text = raw

    # Strip ALL <think>...</think> blocks (DictaLM-Thinking model)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also strip unclosed <think> at end
    if '<think>' in text:
        text = text[:text.find('<think>')]

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find the FIRST complete JSON array [...] using bracket matching
    start = text.find("[")
    if start < 0:
        return [{"parse_error": True, "raw": raw[:500]}]

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end < 0:
        # No complete array — try rfind as fallback
        end = text.rfind("]")

    if end > start:
        text = text[start:end + 1]

    # Fix trailing commas
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [{"parse_error": True, "raw": raw[:500]}]


# =====================================================================
# Checkpoint management
# =====================================================================


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_indices": [], "results": []}


def save_checkpoint(path: Path, checkpoint: dict):
    """Save checkpoint to disk."""
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


# =====================================================================
# Main runner
# =====================================================================


def run(config: ToolkitConfig, force: bool = False, batch_size: int = BATCH_SIZE):
    """Run DictaLM validation of decoded Hebrew forms."""
    out_path = config.stats_dir / "dictalm_validation.json"
    txt_path = config.stats_dir / "dictalm_validation.txt"
    checkpoint_path = config.stats_dir / "dictalm_checkpoint.json"

    if out_path.exists() and not force:
        print(f"Output exists: {out_path} (use --force to re-run)")
        return

    print_header("Phase 20: DictaLM Hebrew Validation")

    # Load API key
    print_step("Loading API key")
    api_key = load_api_key()
    print("  Key loaded from ~/.config/featherless/api_key")

    # Test connection
    print_step("Testing API connection")
    if not test_connection(api_key):
        raise click.ClickException("Cannot connect to Featherless API. Check your key.")
    print("  Connection OK")

    # Load glossed words
    print_step("Loading glossed words")
    glossed_path = config.stats_dir / "glossed_words.json"
    if not glossed_path.exists():
        raise click.ClickException(
            f"glossed_words.json not found: {glossed_path}\n"
            "  Run: voynich --force full-decode"
        )

    with open(glossed_path) as f:
        glossed_data = json.load(f)

    # Build word list sorted by frequency (most common first)
    words = []
    for w in glossed_data:
        # Skip Sefaria-Corpus entries with no real gloss
        gloss = w.get("gloss", "")
        source = w.get("source", "")
        if source == "Sefaria-Corpus" and gloss == "[attestato nel corpus]":
            gloss = "(corpus-attested, no dictionary gloss)"
        words.append({
            "consonantal": w["consonantal"],
            "freq": w.get("freq", 0),
            "gloss": gloss,
            "source": source,
        })

    words.sort(key=lambda x: -x["freq"])
    print(f"  {len(words)} glossed words loaded")

    # Load checkpoint for resume
    checkpoint = load_checkpoint(checkpoint_path)
    done_set = set(checkpoint["completed_indices"])
    if done_set:
        print(f"  Resuming: {len(done_set)} words already validated")

    # Filter out already-done words
    todo = [(i, w) for i, w in enumerate(words) if i not in done_set]
    print(f"  Remaining: {len(todo)} words to validate")

    if not todo:
        print("  All words already validated!")
    else:
        # Process in batches
        n_batches = (len(todo) + batch_size - 1) // batch_size
        print_step(f"Validating in {n_batches} batches of ~{batch_size}")

        errors = 0
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_items = todo[batch_start:batch_start + batch_size]
            batch_words = [item[1] for item in batch_items]
            batch_indices = [item[0] for item in batch_items]

            print(f"\n  Batch {batch_idx + 1}/{n_batches} "
                  f"({len(batch_words)} words, "
                  f"top: {batch_words[0]['consonantal']} freq={batch_words[0]['freq']})...",
                  end="", flush=True)

            try:
                prompt = build_word_batch_prompt(batch_words)
                raw = query_dictalm(prompt, api_key)
                parsed = parse_response(raw)

                if len(parsed) == 1 and parsed[0].get("parse_error"):
                    print(f" PARSE ERROR")
                    errors += 1
                    # Save raw response for debugging
                    checkpoint["results"].append({
                        "batch": batch_idx,
                        "error": "parse_error",
                        "raw": raw[:1000],
                        "indices": batch_indices,
                    })
                else:
                    # Merge results with original word data
                    for j, result in enumerate(parsed):
                        if j < len(batch_words):
                            idx = batch_indices[j]
                            merged = {
                                **batch_words[j],
                                "hebrew_unicode": to_hebrew_unicode(
                                    batch_words[j]["consonantal"]),
                                **{k: v for k, v in result.items()
                                   if k != "index"},
                            }
                            checkpoint["results"].append(merged)
                            checkpoint["completed_indices"].append(idx)

                    print(f" OK ({len(parsed)} results)")

            except requests.exceptions.RequestException as e:
                print(f" API ERROR: {e}")
                errors += 1
                # Don't mark as completed — will retry on resume
            except Exception as e:
                print(f" ERROR: {e}")
                errors += 1

            # Save checkpoint after each batch
            save_checkpoint(checkpoint_path, checkpoint)

            # Rate limit
            if batch_idx < n_batches - 1:
                time.sleep(DELAY_BETWEEN_BATCHES)

        print(f"\n  Done: {len(checkpoint['completed_indices'])} validated, "
              f"{errors} errors")

    # Compile final results
    print_step("Compiling results")
    results = [r for r in checkpoint["results"] if "error" not in r]

    # Statistics
    n_total = len(results)
    n_valid = sum(1 for r in results if r.get("valid") == "yes")
    n_possible = sum(1 for r in results if r.get("valid") == "possible")
    n_invalid = sum(1 for r in results if r.get("valid") == "no")
    n_gloss_correct = sum(1 for r in results
                          if r.get("our_gloss_correct") == "yes")
    n_gloss_wrong = sum(1 for r in results
                        if r.get("our_gloss_correct") == "no")
    n_gloss_partial = sum(1 for r in results
                          if r.get("our_gloss_correct") == "partial")

    # By period
    from collections import Counter
    period_counts = Counter(r.get("period", "unknown") for r in results
                            if r.get("valid") in ("yes", "possible"))
    confidence_counts = Counter(r.get("confidence", "unknown")
                                for r in results)

    # Token-weighted stats
    tok_valid = sum(r["freq"] for r in results if r.get("valid") == "yes")
    tok_total = sum(r["freq"] for r in results)

    stats = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_possible": n_possible,
        "n_invalid": n_invalid,
        "valid_rate": round(n_valid / n_total, 4) if n_total else 0,
        "valid_plus_possible_rate": round(
            (n_valid + n_possible) / n_total, 4) if n_total else 0,
        "valid_token_rate": round(tok_valid / tok_total, 4) if tok_total else 0,
        "gloss_correct": n_gloss_correct,
        "gloss_wrong": n_gloss_wrong,
        "gloss_partial": n_gloss_partial,
        "gloss_accuracy": round(
            n_gloss_correct / n_total, 4) if n_total else 0,
        "periods": dict(period_counts.most_common()),
        "confidence": dict(confidence_counts.most_common()),
    }

    print(f"  Valid: {n_valid}/{n_total} ({stats['valid_rate']*100:.1f}%)")
    print(f"  Possible: {n_possible}/{n_total}")
    print(f"  Invalid: {n_invalid}/{n_total}")
    print(f"  Valid (token-weighted): {stats['valid_token_rate']*100:.1f}%")
    print(f"  Gloss correct: {n_gloss_correct}, wrong: {n_gloss_wrong}, "
          f"partial: {n_gloss_partial}")
    print(f"  Periods: {dict(period_counts.most_common())}")

    # Save JSON
    output = {
        "model": MODEL,
        "stats": stats,
        "words": sorted(results, key=lambda x: -x.get("freq", 0)),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {out_path}")

    # Save TXT
    lines = [
        "Phase 20: DictaLM Hebrew Validation",
        "=" * 60,
        f"Model: {MODEL}",
        f"Words validated: {n_total}",
        "",
        f"Valid:    {n_valid:4d} ({stats['valid_rate']*100:.1f}%)",
        f"Possible: {n_possible:4d}",
        f"Invalid:  {n_invalid:4d}",
        f"Valid (token-weighted): {stats['valid_token_rate']*100:.1f}%",
        "",
        f"Gloss correct:  {n_gloss_correct}",
        f"Gloss wrong:    {n_gloss_wrong}",
        f"Gloss partial:  {n_gloss_partial}",
        f"Gloss accuracy: {stats['gloss_accuracy']*100:.1f}%",
        "",
        "Periods (valid+possible):",
    ]
    for period, count in period_counts.most_common():
        lines.append(f"  {period:15s}: {count}")

    lines.append("")
    lines.append("Top words by frequency:")
    lines.append(f"{'Hebrew':>10s}  {'ASCII':>8s}  {'Freq':>5s}  "
                 f"{'Valid':>5s}  {'Gloss OK':>8s}  Meaning")
    lines.append("-" * 80)

    for r in sorted(results, key=lambda x: -x.get("freq", 0))[:50]:
        heb = r.get("hebrew_unicode", "")
        asc = r.get("consonantal", "")
        freq = r.get("freq", 0)
        valid = r.get("valid", "?")
        gloss_ok = r.get("our_gloss_correct", "?")
        meaning = r.get("meaning", "") or r.get("correct_meaning", "")
        lines.append(f"{heb:>10s}  {asc:>8s}  {freq:5d}  "
                     f"{valid:>5s}  {gloss_ok:>8s}  {meaning[:40]}")

    # Wrong glosses section
    wrong = [r for r in results if r.get("our_gloss_correct") == "no"]
    if wrong:
        lines.append("")
        lines.append(f"WRONG GLOSSES ({len(wrong)}):")
        lines.append(f"{'ASCII':>8s}  {'Freq':>5s}  {'Our gloss':30s}  "
                     f"{'Correct meaning'}")
        lines.append("-" * 80)
        for r in sorted(wrong, key=lambda x: -x.get("freq", 0)):
            asc = r.get("consonantal", "")
            freq = r.get("freq", 0)
            our = r.get("gloss", "")[:30]
            correct = r.get("correct_meaning", "") or r.get("meaning", "")
            lines.append(f"{asc:>8s}  {freq:5d}  {our:30s}  {correct}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  TXT: {txt_path}")

    # Clean up checkpoint on success
    if checkpoint_path.exists() and len(todo) == 0 or errors == 0:
        checkpoint_path.unlink(missing_ok=True)
        print("  Checkpoint cleaned up")


if __name__ == "__main__":
    run(ToolkitConfig())
