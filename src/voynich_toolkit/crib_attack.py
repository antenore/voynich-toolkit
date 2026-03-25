"""
Crib attack: known-plaintext attack on the Voynich manuscript.

Uses the Anthropic API (Claude Sonnet) to generate plausible Hebrew medical
texts, encodes them with the cipher, and compares against real manuscript text.

Workflow:
  1. Extract page constraints from the EVA/IVTFF transcription
  2. Identify anchor words (decoded words with glosses in the DB)
  3. Build a prompt for Claude to generate plausible crib candidates
  4. Call the API and parse results
  5. Score and rank cribs against the real EVA text
  6. Save results to output/stats/crib_attack_{folio}.json
"""

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import anthropic


def _load_api_key() -> str:
    """Load Anthropic API key from environment or .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # Try loading from .env files
    for env_path in [
        Path(__file__).parent.parent.parent / ".env",  # voynich-toolkit/.env
        Path.home() / "code" / "decipher" / ".env",    # project root/.env
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("ANTHROPIC_API_KEY", "ANTHROPIC_API"):
                    return v
    return ""

from .crib_encoder import best_variant_match, encode_word, encode_word_variants
from .full_decode import decode_word

# ---------------------------------------------------------------------------
# Path constants (relative to this file's package root)
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).parent.parent.parent  # voynich-toolkit/
_EVA_FILE = _PKG_ROOT / "eva_data" / "LSI_ivtff_0d.txt"
_DB_PATH = _PKG_ROOT / "voynich.db"
_OUTPUT_DIR = _PKG_ROOT / "output" / "stats"

# Section code → human name
SECTION_NAMES: dict[str, str] = {
    "H": "herbal",
    "S": "astronomical",
    "Z": "zodiac",
    "B": "balneological",
    "P": "pharmaceutical",
    "T": "text",
    "C": "cosmological",
    "?": "unknown",
    "": "unknown",
}

# IVTFF hand preference order (best first)
HAND_PREFERENCE = ["H", "C", "F", "U"]

# ---------------------------------------------------------------------------
# 1. Extract page constraints
# ---------------------------------------------------------------------------


def extract_page_constraints(folio: str, eva_file: str | Path = _EVA_FILE) -> dict:
    """Extract constraints for a given folio page.

    Reads the IVTFF transcription file and extracts:
      - eva_words: list of all EVA words on the page
      - word_lengths: list of character lengths for each EVA word
      - line_structure: list of lists (words per line)
      - section: H / B / P / S / Z / T / C

    IVTFF format example:
      <f27r.2,+P0;H>   dy.coain.shol.dain.dar.shokyd<-><!plant>dchol.cthey.ds

    Rules:
      - Use hand 'H' preferentially, then C, F, U.
      - Words are dot-separated; annotations like <!plant>, <$>, <->, {...}
        are stripped before splitting.
      - Only tokens matching ^[a-z]+$ are kept.
    """
    eva_path = Path(eva_file)
    text = eva_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Patterns
    header_re = re.compile(r"^<(f\w+)>\s+<!(.+?)>")
    meta_re = re.compile(r"\$(\w)=(\w+)")
    # Match lines like: <f27r.2,+P0;H>  ...text...
    line_re = re.compile(r"^<(f\w+)\.\d+[^;]*;(\w)>\s+(.*)")

    # Collect per-hand data for the target folio
    # hand → list of (line_index, [word, ...])
    per_hand: dict[str, list[list[str]]] = {h: [] for h in HAND_PREFERENCE}
    section_code = "?"
    in_target = False

    def _extract_words(raw: str) -> list[str]:
        """Strip all IVTFF annotations and return clean EVA words."""
        # Replace '!' (uncertain word boundary) with '.' FIRST
        clean = raw.replace("!", ".")
        # Replace ',' with '.'
        clean = clean.replace(",", ".")
        # Replace <-> and <...> markers with '.' (they separate text blocks)
        clean = re.sub(r"<[^>]*>", ".", clean)
        # Remove {...} comments
        clean = re.sub(r"\{[^}]*\}", ".", clean)
        # Remove leftover punctuation and special chars
        clean = re.sub(r"[%?\[\]*]", "", clean)
        words = []
        for token in clean.split("."):
            word = re.sub(r"[^a-z]", "", token)
            if word:
                words.append(word)
        return words

    for raw_line in lines:
        line = raw_line.rstrip()

        # Skip comments
        if not line or line.startswith("#"):
            continue

        # Page header: <fNNNr>  <! $I=H $Q=... >
        m = header_re.match(line)
        if m:
            page_id = m.group(1)
            if page_id == folio:
                in_target = True
                meta = dict(meta_re.findall(m.group(2)))
                section_code = meta.get("I", "?")
            elif in_target:
                # We've moved past the target folio
                break
            continue

        if not in_target:
            continue

        m = line_re.match(line)
        if not m:
            continue

        page_id = m.group(1)
        hand = m.group(2)
        raw_text = m.group(3)

        # Verify still on the same folio
        if page_id != folio:
            break

        if hand not in per_hand:
            continue

        words = _extract_words(raw_text)
        if words:
            per_hand[hand].append(words)

    # Pick the best available hand
    chosen_hand = None
    for h in HAND_PREFERENCE:
        if per_hand[h]:
            chosen_hand = h
            break

    if chosen_hand is None:
        return {
            "folio": folio,
            "section": section_code,
            "section_name": SECTION_NAMES.get(section_code, "unknown"),
            "eva_words": [],
            "word_lengths": [],
            "line_structure": [],
            "n_words": 0,
            "n_lines": 0,
        }

    line_structure = per_hand[chosen_hand]  # list of lists
    eva_words = [w for line in line_structure for w in line]
    word_lengths = [len(w) for w in eva_words]

    return {
        "folio": folio,
        "section": section_code,
        "section_name": SECTION_NAMES.get(section_code, "unknown"),
        "hand_used": chosen_hand,
        "eva_words": eva_words,
        "word_lengths": word_lengths,
        "line_structure": line_structure,
        "n_words": len(eva_words),
        "n_lines": len(line_structure),
    }


# ---------------------------------------------------------------------------
# 2. Identify anchors
# ---------------------------------------------------------------------------


def identify_anchors(eva_words: list[str], db_path: str | Path = _DB_PATH) -> list[dict]:
    """For each EVA word, decode it and check if it has a gloss in the database.

    Uses full_decode.decode_word() to get the Hebrew consonantal form, then
    queries glossed_words for a matching entry.

    Returns:
        List of dicts for words that have glosses:
        [{"position": int, "eva": str, "hebrew": str, "gloss": str, "freq": int}, ...]
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    anchors = []
    for pos, eva_word in enumerate(eva_words):
        _italian, hebrew, n_unk = decode_word(eva_word)
        if not hebrew or n_unk == len(hebrew):
            continue

        row = cursor.execute(
            "SELECT consonantal, gloss, freq FROM glossed_words WHERE consonantal = ?",
            (hebrew,),
        ).fetchone()

        if row:
            gloss = row[1] or ""
            freq = row[2] or 0
            # Filter: only keep anchors with real, meaningful glosses
            # Skip weak entries: attested-only, grammar codes, Klein stubs
            skip_patterns = [
                "[attestato nel corpus]",
                "H:N-M", "H:N-F", "H:Adv", "H:V",
                "[Klein headword]",
            ]
            is_strong = not any(pat in gloss for pat in skip_patterns)
            anchors.append({
                "position": pos,
                "eva": eva_word,
                "hebrew": hebrew,
                "gloss": gloss,
                "freq": freq,
                "strong": is_strong,
            })

    conn.close()
    return anchors


# ---------------------------------------------------------------------------
# 3. Build the LLM prompt
# ---------------------------------------------------------------------------


def build_crib_prompt(
    constraints: dict,
    anchors: list[dict],
    n_variants: int = 10,
) -> str:
    """Build a prompt for Claude to generate plausible Hebrew medical texts.

    The prompt specifies:
      - Persona: 15th-century Judeo-Italian physician
      - Section type (herbal, balneological, etc.)
      - Exact word count and per-word character-length constraints
      - Anchor words (decoded words with known glosses) as hard constraints
      - Output format: JSON list of lists
      - Hebrew consonantal notation guide and vocabulary
    """
    section = constraints.get("section", "?")
    section_name = constraints.get("section_name", "unknown")
    word_lengths = constraints.get("word_lengths", [])
    n_words = len(word_lengths)
    folio = constraints.get("folio", "unknown")

    # Format anchor constraints — only strong ones as hard constraints
    strong_anchors = [a for a in anchors if a.get("strong", False)]
    weak_anchors = [a for a in anchors if not a.get("strong", False)]

    anchor_lines = []
    if strong_anchors:
        for a in strong_anchors:
            anchor_lines.append(
                f"  - position {a['position']} (0-indexed): '{a['hebrew']}'"
                f"  meaning: '{a['gloss'][:60]}'"
                f"  ({a['freq']} occurrences)"
            )
    anchor_block = "\n".join(anchor_lines) if anchor_lines else "  (none identified)"

    # Weak anchors as soft hints
    hint_lines = []
    if weak_anchors:
        for a in weak_anchors[:10]:  # limit to avoid prompt bloat
            hint_lines.append(
                f"  - position {a['position']}: '{a['hebrew']}' (attested but meaning unclear)"
            )
    hint_block = "\n".join(hint_lines) if hint_lines else "  (none)"

    # Format word-length constraint as a numbered list
    length_list = ", ".join(
        f"word{i}={l}" for i, l in enumerate(word_lengths)
    )

    prompt = f"""You are a specialist in medieval Hebrew medical manuscripts. Your task is to
generate plausible 15th-century Judeo-Italian physician texts in CONSONANTAL HEBREW that
could have been encoded in the Voynich manuscript.

CONTEXT
=======
- Manuscript section: {section_name} (code '{section}')
- Folio: {folio}
- The text is consonantal Hebrew (no vowels) written by an Italian Jewish physician
  describing herbs, recipes, balneological treatments, or pharmaceutical preparations
  depending on the section type.
- The cipher reads RIGHT TO LEFT; you must write the Hebrew in logical (left-to-right)
  consonantal order as it would appear in a decoded transliteration.

HEBREW ASCII NOTATION (MANDATORY)
==================================
Use ONLY these ASCII symbols for Hebrew consonants:
  A = aleph    b = bet      g = gimel    d = dalet    h = he
  w = vav      z = zayin    X = chet     J = tet      y = yod
  k = kaf      l = lamed    m = mem      n = nun      s = samekh
  E = ayin     p = pe       C = tsade    q = qof      r = resh
  S = shin     t = tav

Do NOT use any other characters. No vowels, no dagesh, no niqqud.

WORD-LENGTH CONSTRAINTS (HARD — non-negotiable)
================================================
The text must have EXACTLY {n_words} words.
Each word must have EXACTLY the number of consonants shown below:
  {length_list}

ANCHOR WORDS (HARD — must appear exactly as given)
====================================================
These words are decoded with high confidence. Use them EXACTLY at the given positions:
{anchor_block}

WEAK HINTS (SOFT — you may use these or replace them with better alternatives)
================================================================================
These words are attested in the corpus but their meaning is unclear.
You may keep them OR replace with a word that fits the recipe context better,
as long as the replacement has the correct character count:
{hint_block}

DIVERSITY AND COHERENCE (CRITICAL)
====================================
- Each variant must tell a DIFFERENT, COHERENT medical recipe or herbal description
- Do NOT repeat the same word in multiple non-anchor positions — variety is essential
- The text should read as natural medical Hebrew, not filler
- Use a VARIETY of medical vocabulary: body parts, plants, actions, substances
- Think: what would a real physician write on this page?

MEDICAL HEBREW VOCABULARY (use this as inspiration)
=====================================================
Body parts:
  yd (hand), rgl (foot), Eyn (eye), lb (heart), rAS (head), bJn (belly),
  kbd (liver), klyt (kidney), grwn (throat), ySn (tooth), znb (tail/appendage)

Plants / botanical:
  SrS (root), Elh (leaf), prX (flower), pry (fruit), zrE (seed),
  EnbS (anise), qdX (coriander), lwnX (frankincense), knmwn (cinnamon),
  rzdh (rue), Alh (aloe), Sr (mastic), hnk (henna), krkm (turmeric)

Actions / verbs (imperative / infinitive form):
  swk (anoint), Spk (pour), JXn (grind), bSl (cook), Erb (mix),
  rXC (wash), rkk (soften), lXS (knead), Ebr (pass/filter), qSr (bind),
  ntn (give/apply), sm (place), SrS (uproot), kth (pound/crush)

Substances:
  mym (water), yyn (wine), Smn (oil), dbs (honey), Xms (vinegar),
  mlX (salt), qJrn (resin), gpr (sulphur), hlb (milk), dm (blood),
  zyt (olive), Srp (turpentine), nrd (nard/spikenard)

Qualities / adjectives:
  Xm (hot), qr (cold), ybs (dry), lX (wet/moist), mr (bitter),
  mtq (sweet), Xzq (strong), rk (soft), qdm (old/ancient)

Medical terms:
  rpy (heal), Xly (sick/disease), mkh (wound), kAb (pain), mrgw (salve),
  qrXt (plaster/poultice), mzg (mixture/compound), hwrl (medicine)

Measures / quantities:
  kzyt (olive-size), kbyCh (egg-size), kp (palm/handful), lgh (log measure),
  rbyEyt (quarter-measure), mnh (portion), skl (shekel weight)

Common prepositions / particles:
  b (in/with), l (to/for), m (from/of), k (as/like), E (on/above), tXt (under)

TASK
====
Generate {n_variants} different plausible texts that:
1. Satisfy ALL word-length constraints exactly
2. Include all anchor words at the specified positions exactly
3. Form a coherent medical/herbal instruction, recipe, or description
4. Use only the Hebrew ASCII consonants listed above
5. Are stylistically consistent with medieval Judeo-Italian medical manuscripts

OUTPUT FORMAT (JSON only — no markdown, no explanation)
=======================================================
Output a single JSON array of {n_variants} items. Each item is a list of
{n_words} Hebrew consonantal strings (one per word), strictly in order.

Example structure (do not use this content, generate your own):
[
  ["SrS", "Elh", "Smn", "bSl", "Xm", ...],
  ["pry", "yd", "mym", "rXC", "qr", ...],
  ...
]

UNMAPPED LETTERS — CRITICAL CONSTRAINT
========================================
The cipher has 3 Hebrew letters that are NOT YET MAPPED to EVA script:
  z (zayin), C (tsade), q (qof)

Words containing z, C, or q will produce '?' in the encoded output and CANNOT be
compared against the manuscript. Therefore:
  - STRONGLY PREFER words that use ONLY the 19 mapped letters: AbgdhwXJyklmnsEprSt
  - AVOID z, C, q unless absolutely no alternative exists
  - For example: use "Xm" (hot) instead of "Xzq" (strong), "kth" instead of "qSr"
  - Rethink word choices that would require zayin, tsade, or qof
  - If an anchor word contains z/C/q, keep it unchanged (it's a hard constraint)

This constraint is MORE IMPORTANT than using fancy vocabulary. Simple words with
only mapped letters are far more valuable than elaborate words with unmapped letters.

CRITICAL VALIDATION
===================
Before outputting, verify for EACH variant:
- Exactly {n_words} words in the list
- Word at position i has exactly word_lengths[i] consonants: {word_lengths}
- Anchor words appear at their specified positions unchanged
- Every character is from the allowed Hebrew ASCII set: AbgdhwzXJyklmnsEpCqrSt
- MINIMIZE use of z, C, q (unmapped letters) — count them and try to reduce

Output the JSON array now:"""

    return prompt


# ---------------------------------------------------------------------------
# 4. Call the API and parse results
# ---------------------------------------------------------------------------


def generate_cribs(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> list[list[str]]:
    """Call the Anthropic API to generate crib texts.

    Parses the JSON from the response. Returns a list of lists of Hebrew words.
    Handles JSON parsing errors gracefully (the LLM might wrap in code blocks).
    """
    api_key = _load_api_key()
    if not api_key:
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY env var "
            "or add ANTHROPIC_API=... to .env"
        )
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown code fences if present
    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text)
    if fence_match:
        raw_text = fence_match.group(1).strip()

    # Try to extract a JSON array even if there is surrounding text
    array_match = re.search(r"\[[\s\S]*\]", raw_text)
    if array_match:
        raw_text = array_match.group(0)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Last resort: try to fix common issues (trailing commas)
        cleaned = re.sub(r",\s*([\]}])", r"\1", raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not parse JSON from API response: {exc}\n"
                f"Raw response (first 500 chars):\n{raw_text[:500]}"
            ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array, got {type(data).__name__}: {str(data)[:200]}"
        )

    # Normalise: each element should be a list of strings
    cribs: list[list[str]] = []
    for item in data:
        if isinstance(item, list):
            cribs.append([str(w) for w in item])
        elif isinstance(item, str):
            # Fallback: sometimes the LLM returns a flat list of words for 1 variant
            cribs.append([item])
        else:
            cribs.append([str(w) for w in item] if hasattr(item, "__iter__") else [])

    return cribs


# ---------------------------------------------------------------------------
# 5. Score and rank cribs
# ---------------------------------------------------------------------------


def score_cribs(
    cribs: list[list[str]],
    real_eva: list[str],
    db_path: str | Path = _DB_PATH,
) -> list[dict[str, Any]]:
    """Score each crib against the real EVA text.

    For each crib (list of Hebrew words):
      1. For each word, find the closest EVA variant via best_variant_match().
      2. Count exact matches (d=0), near matches (d=1), failures (d>1).
      3. Compute overall composite score.

    Returns:
        List of dicts sorted by score descending:
        [{
            'rank': int,
            'hebrew_words': list[str],
            'eva_encoded': list[str],  # best EVA variant per word
            'exact_matches': int,
            'near_matches': int,
            'failures': int,
            'score': float,
            'word_details': [{'position': int, 'hebrew': str,
                               'eva_variant': str, 'real_eva': str,
                               'distance': int}, ...]
        }]
    """
    n_real = len(real_eva)
    scored: list[dict[str, Any]] = []

    for crib in cribs:
        n_words = min(len(crib), n_real)
        exact = 0
        near = 0
        failures = 0
        word_details = []
        eva_encoded = []

        for i in range(n_words):
            heb = crib[i]
            real = real_eva[i] if i < n_real else ""

            best_variant, dist = best_variant_match(heb, real)
            eva_encoded.append(best_variant if best_variant else encode_word(heb))

            if dist == 0:
                exact += 1
            elif dist == 1:
                near += 1
            else:
                failures += 1

            word_details.append({
                "position": i,
                "hebrew": heb,
                "eva_variant": best_variant or "",
                "real_eva": real,
                "distance": dist,
            })

        # Words in crib beyond real text are failures
        for i in range(n_words, len(crib)):
            failures += 1
            eva_encoded.append(encode_word(crib[i]))
            word_details.append({
                "position": i,
                "hebrew": crib[i],
                "eva_variant": encode_word(crib[i]),
                "real_eva": "",
                "distance": 999,
            })

        # Composite score:
        #   exact match    → 1.0 per word
        #   near match     → 0.5 per word
        #   total normalised by n_real (so 0–1 range)
        denom = n_real if n_real > 0 else 1
        score = (exact * 1.0 + near * 0.5) / denom

        scored.append({
            "hebrew_words": list(crib),
            "eva_encoded": eva_encoded,
            "exact_matches": exact,
            "near_matches": near,
            "failures": failures,
            "score": round(score, 4),
            "word_details": word_details,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Add rank
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    return scored


# ---------------------------------------------------------------------------
# 6. Word-level crib attack
# ---------------------------------------------------------------------------


def _build_word_context(pos: int, eva_words: list[str], anchors: list[dict],
                        window: int = 3) -> str:
    """Build a context string showing surrounding words with glosses."""
    anchor_map = {a["position"]: a for a in anchors}
    parts = []
    for i in range(max(0, pos - window), min(len(eva_words), pos + window + 1)):
        if i == pos:
            parts.append(f"  >>> TARGET (position {pos}, {len(eva_words[pos])} EVA chars) <<<")
        elif i in anchor_map:
            a = anchor_map[i]
            gl = a["gloss"][:50] if a.get("strong") else "(attested)"
            parts.append(f"  pos {i}: '{a['hebrew']}' = {gl}")
        else:
            _, heb, _ = decode_word(eva_words[i])
            parts.append(f"  pos {i}: '{heb}' (unknown)")
    return "\n".join(parts)


def build_wordlevel_prompt(
    targets: list[dict],
    section_name: str,
    n_candidates: int = 10,
) -> str:
    """Build a prompt for word-level crib generation.

    Each target is a dict with: position, eva, hebrew, length, context_str.
    Asks for n_candidates alternatives per word.
    """
    target_blocks = []
    for t in targets:
        target_blocks.append(
            f"WORD {t['position']} — {t['length']} Hebrew consonants\n"
            f"Current decode: '{t['hebrew']}' (no confirmed meaning)\n"
            f"Context:\n{t['context_str']}\n"
        )

    targets_text = "\n---\n".join(target_blocks)

    prompt = f"""You are an expert in medieval Hebrew medical manuscripts.

TASK: For each unknown word below, generate {n_candidates} plausible Hebrew consonantal
alternatives that:
1. Have EXACTLY the specified number of consonants
2. Fit the surrounding medical/herbal context
3. Use ONLY mapped Hebrew letters: A b g d h w X J y k l m n s E p r S t
   (AVOID z, C, q — these are unmapped and produce errors)
4. Are real Hebrew/Aramaic words or Italian loanwords in Hebrew consonants

SECTION: {section_name} (herbal/pharmaceutical recipe)

HEBREW ASCII: A=aleph b=bet g=gimel d=dalet h=he w=vav X=chet J=tet y=yod
k=kaf l=lamed m=mem n=nun s=samekh E=ayin p=pe r=resh S=shin t=tav

MEDICAL VOCABULARY HINTS:
- Body: yd(hand) rgl(foot) Eyn(eye) lb(heart) rAS(head) bJn(belly) Ewr(skin) dm(blood)
- Plants: SrS(root) Elh(leaf) prX(flower) pry(fruit) Esb(herb) Smn(oil) hlb(milk)
- Actions: swk(anoint) Spk(pour) JXn(grind) bSl(cook) Erb(mix) rXy(wash) ntn(give)
  kth(crush) rkk(soften) snn(filter) ybs(dry) Xmm(heat)
- Substances: mym(water) yyn(wine) dbs(honey) mlX(salt) Smr(yeast/dregs)
- Qualities: Xm(hot) mr(bitter) mtq(sweet) rk(soft) yps(good/beautiful)

UNKNOWN WORDS TO RESOLVE:
=========================
{targets_text}

OUTPUT FORMAT (JSON only — no explanation):
A JSON object mapping position number to a list of {n_candidates} candidate strings.
Example: {{"0": ["SrSh", "Elbm", ...], "13": ["bSlht", "mymEn", ...], ...}}

Each candidate must have EXACTLY the right number of consonants for that position.
Prioritize words that make medical/recipe sense in the given context."""

    return prompt


def run_wordlevel_attack(
    folio: str,
    n_candidates: int = 10,
    model: str = "claude-sonnet-4-20250514",
    eva_file: str | Path = _EVA_FILE,
    db_path: str | Path = _DB_PATH,
    output_dir: str | Path = _OUTPUT_DIR,
) -> dict:
    """Run a word-level crib attack: generate candidates for each unknown word.

    1. Extract page, identify anchors
    2. Find hapax/unknown words (not in anchor set or repeated formulaic)
    3. For each, build context from surrounding words
    4. Ask LLM for N candidates per word
    5. Encode each candidate, compare to real EVA
    6. Save results
    """
    from collections import Counter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wordlevel] Folio: {folio}  n_candidates: {n_candidates}  model: {model}")

    # Step 1: constraints + anchors
    constraints = extract_page_constraints(folio, eva_file=eva_file)
    if not constraints["eva_words"]:
        return {"folio": folio, "error": "No EVA words found"}

    eva_words = constraints["eva_words"]
    anchors = identify_anchors(eva_words, db_path=db_path)
    anchor_positions = {a["position"] for a in anchors if a.get("strong")}

    # Step 2: find target words (not strong anchors)
    eva_freq = Counter(eva_words)
    targets = []
    for pos, eva_word in enumerate(eva_words):
        if pos in anchor_positions:
            continue
        # Skip very short words (1 char) — too ambiguous
        if len(eva_word) <= 1:
            continue
        _, heb, _ = decode_word(eva_word)
        targets.append({
            "position": pos,
            "eva": eva_word,
            "hebrew": heb,
            "length": len(heb),  # Hebrew consonant count
            "context_str": _build_word_context(pos, eva_words, anchors),
        })

    print(f"[wordlevel] {len(targets)} target words (non-anchor, len>1)")

    # Step 3: batch targets into groups to avoid huge prompts
    BATCH_SIZE = 15
    all_results = {}

    for batch_start in range(0, len(targets), BATCH_SIZE):
        batch = targets[batch_start:batch_start + BATCH_SIZE]
        print(f"[wordlevel] Batch {batch_start // BATCH_SIZE + 1}: "
              f"words {batch[0]['position']}-{batch[-1]['position']}")

        prompt = build_wordlevel_prompt(
            batch, constraints["section_name"], n_candidates=n_candidates
        )

        api_key = _load_api_key()
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()

        # Parse JSON
        raw = re.sub(r"```(?:json)?\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)
        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", raw)
            try:
                candidates = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"[wordlevel] WARNING: could not parse batch response")
                continue

        # Score each candidate
        for t in batch:
            pos = t["position"]
            pos_key = str(pos)
            cands = candidates.get(pos_key, [])
            if not cands:
                continue

            real_eva = eva_words[pos]
            scored = []
            for heb_cand in cands:
                if not isinstance(heb_cand, str):
                    continue
                best_var, dist = best_variant_match(heb_cand, real_eva)
                scored.append({
                    "hebrew": heb_cand,
                    "eva_variant": best_var or encode_word(heb_cand),
                    "distance": dist,
                })

            scored.sort(key=lambda x: x["distance"])
            all_results[pos] = {
                "position": pos,
                "real_eva": real_eva,
                "current_hebrew": t["hebrew"],
                "length": t["length"],
                "candidates": scored,
                "best_distance": scored[0]["distance"] if scored else 999,
                "best_candidate": scored[0] if scored else None,
            }

    # Summary
    hits = [r for r in all_results.values() if r["best_distance"] == 0]
    near = [r for r in all_results.values() if r["best_distance"] == 1]
    print(f"\n[wordlevel] === RESULTS ===")
    print(f"[wordlevel] Exact matches (d=0): {len(hits)}")
    print(f"[wordlevel] Near matches (d=1):  {len(near)}")
    for r in sorted(hits + near, key=lambda x: x["best_distance"]):
        bc = r["best_candidate"]
        print(f"  pos {r['position']:3d}: {r['real_eva']:10s} ← "
              f"{bc['hebrew']:10s} (d={bc['distance']}) "
              f"[was: {r['current_hebrew']}]")

    result = {
        "folio": folio,
        "model": model,
        "n_targets": len(targets),
        "n_exact": len(hits),
        "n_near": len(near),
        "word_results": all_results,
        "hits": [{
            "position": r["position"],
            "real_eva": r["real_eva"],
            "old_hebrew": r["current_hebrew"],
            "new_hebrew": r["best_candidate"]["hebrew"],
            "distance": r["best_distance"],
        } for r in sorted(hits + near, key=lambda x: x["best_distance"])],
    }

    output_path = out_dir / f"crib_wordlevel_{folio}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[wordlevel] Results saved → {output_path}")

    return result


# ---------------------------------------------------------------------------
# 7. Main entry points
# ---------------------------------------------------------------------------


def run_crib_attack(
    folio: str,
    n_variants: int = 10,
    model: str = "claude-sonnet-4-20250514",
    eva_file: str | Path = _EVA_FILE,
    db_path: str | Path = _DB_PATH,
    output_dir: str | Path = _OUTPUT_DIR,
) -> dict:
    """Run a full crib attack on a given folio.

    Steps:
      1. Extract page constraints from EVA file
      2. Identify anchor words from the database
      3. Build the LLM prompt
      4. Generate cribs via the Anthropic API
      5. Score and rank all generated cribs
      6. Save results to output/stats/crib_attack_{folio}.json

    Returns:
        Dict with the full results summary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[crib_attack] Folio: {folio}  n_variants: {n_variants}  model: {model}")

    # --- Step 1: page constraints ---
    print("[crib_attack] Step 1: extracting page constraints …")
    constraints = extract_page_constraints(folio, eva_file=eva_file)

    if not constraints["eva_words"]:
        print(f"[crib_attack] WARNING: no EVA words found for folio '{folio}'")
        result: dict = {
            "folio": folio,
            "error": f"No EVA words found for folio '{folio}'",
            "constraints": constraints,
        }
        _save_result(result, folio, out_dir)
        return result

    print(
        f"[crib_attack]   section: {constraints['section_name']} "
        f"({constraints['section']}), {constraints['n_words']} words, "
        f"{constraints['n_lines']} lines"
    )

    # --- Step 2: anchor words ---
    print("[crib_attack] Step 2: identifying anchor words …")
    anchors = identify_anchors(constraints["eva_words"], db_path=db_path)
    print(f"[crib_attack]   {len(anchors)} anchors found")
    for a in anchors[:5]:
        print(f"[crib_attack]     pos {a['position']:3d}: {a['eva']:12s} → "
              f"{a['hebrew']:10s}  '{a['gloss']}'")
    if len(anchors) > 5:
        print(f"[crib_attack]     … and {len(anchors) - 5} more")

    # --- Step 3: build prompt ---
    print("[crib_attack] Step 3: building prompt …")
    prompt = build_crib_prompt(constraints, anchors, n_variants=n_variants)
    print(f"[crib_attack]   prompt length: {len(prompt)} chars")

    # --- Step 4: call API ---
    print(f"[crib_attack] Step 4: calling {model} …")
    cribs = generate_cribs(prompt, model=model)
    print(f"[crib_attack]   received {len(cribs)} cribs")

    # --- Step 5: score ---
    print("[crib_attack] Step 5: scoring cribs …")
    ranked = score_cribs(cribs, constraints["eva_words"], db_path=db_path)

    # --- Step 6: save ---
    result = {
        "folio": folio,
        "model": model,
        "n_variants_requested": n_variants,
        "n_variants_received": len(cribs),
        "section": constraints["section"],
        "section_name": constraints["section_name"],
        "n_real_words": constraints["n_words"],
        "n_lines": constraints["n_lines"],
        "n_anchors": len(anchors),
        "anchors": anchors,
        "real_eva_words": constraints["eva_words"],
        "word_lengths": constraints["word_lengths"],
        "line_structure": constraints["line_structure"],
        "ranked_cribs": ranked,
        "top_score": ranked[0]["score"] if ranked else 0.0,
        "top_exact_matches": ranked[0]["exact_matches"] if ranked else 0,
        "top_near_matches": ranked[0]["near_matches"] if ranked else 0,
    }

    _save_result(result, folio, out_dir)

    # --- Print summary ---
    print("[crib_attack] === RESULTS ===")
    print(f"[crib_attack] Top score: {result['top_score']:.4f}  "
          f"(exact={result['top_exact_matches']}, "
          f"near={result['top_near_matches']})")
    if ranked:
        top = ranked[0]
        print(f"[crib_attack] Best crib: {' '.join(top['hebrew_words'][:8])} …")
        print(f"[crib_attack] Best EVA:  {' '.join(top['eva_encoded'][:8])} …")
        print(f"[crib_attack] Real EVA:  "
              f"{' '.join(constraints['eva_words'][:8])} …")

    return result


def _save_result(result: dict, folio: str, out_dir: Path) -> None:
    """Serialise and save the result dict to JSON."""
    output_path = out_dir / f"crib_attack_{folio}.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"[crib_attack] Results saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "wordlevel"
    folio = sys.argv[2] if len(sys.argv) > 2 else "f27r"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = sys.argv[4] if len(sys.argv) > 4 else "claude-sonnet-4-20250514"

    if mode == "page":
        result = run_crib_attack(folio, n_variants=n, model=model)
        summary = {k: v for k, v in result.items() if k != "ranked_cribs"}
        if "ranked_cribs" in result:
            summary["ranked_cribs"] = [
                {k2: v2 for k2, v2 in item.items() if k2 != "word_details"}
                for item in result["ranked_cribs"]
            ]
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        # Default: word-level attack
        result = run_wordlevel_attack(folio, n_candidates=n, model=model)
