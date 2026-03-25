#!/usr/bin/env python3
"""Decode ALL Voynich manuscript pages using crib attack + interpretation.

Runs word-level crib attack on every folio, then interprets each page
using the merged vocabulary. Parallelized with thread pool.

Usage:
    python scripts/decode_all_pages.py              # all pages
    python scripts/decode_all_pages.py --workers 3  # 3 parallel workers
    python scripts/decode_all_pages.py --skip-crib  # skip crib, interpret only
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from src.voynich_toolkit.crib_attack import (
    extract_page_constraints,
    identify_anchors,
    run_wordlevel_attack,
)
from src.voynich_toolkit.crib_encoder import _scribal_variants
from src.voynich_toolkit.full_decode import decode_word

# Paths
ROOT = Path(__file__).parent.parent
EVA_FILE = ROOT / "eva_data" / "LSI_ivtff_0d.txt"
DB_PATH = ROOT / "voynich.db"
OUTPUT_DIR = ROOT / "output" / "stats"
READINGS_DIR = ROOT / "output" / "readings"


def load_api_key():
    for env_path in [ROOT / ".env", ROOT.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "ANTHROPIC_API" in line and "=" in line:
                    _, v = line.split("=", 1)
                    return v.strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY", "")


VOCAB_PROMPT = """CONFIRMED VOCABULARY (from 5 decoded pages):
- mwk = cotton/soft filtering material (NOT "be poor")
- SEn = to rest/apply medicine (NOT "to lean")
- Spk = to pour (recipe instruction)
- swk = to anoint
- Sk = mixture/preparation
- bhyr = bright/clear/limpid (liquid quality)
- Spt = to pour/set
- gyr = chalk/lite (clarifier ingredient)
- Skr = strong drink/alcohol (solvent)
- mr = bitter (indicator/quality)
- mytq = sweet (completion taste test)
- gmrt = completed (finishing step)
- nr = lamp/light
- mwpt = sign/wonder (diagnostic indicator)
- SrpEn = bathing/soaking (balneological term)
- Srppt = immersion cycle
- ryJ = warm/heating
- ryt b = "in/within" (preposition)
- wdryn = roses (vered)
- nppk = to drip/drop
- swr = to remove/turn aside

SCRIBAL NOTES: t/J(tav/tet) interchangeable, X/E(chet/ayin) may be confused, w/r(vav/resh) may be confused.
SECTIONS: H=herbal, S=astronomical, Z=zodiac, B=balneological, P=pharmaceutical, T=text, C=cosmological.
HEBREW ASCII: A=aleph b=bet g=gimel d=dalet h=he w=vav X=chet J=tet y=yod k=kaf l=lamed m=mem n=nun s=samekh E=ayin p=pe r=resh S=shin t=tav"""


def get_all_folios():
    """Extract all folio IDs from EVA transcription."""
    text = EVA_FILE.read_text(encoding="utf-8", errors="ignore")
    folios = []
    seen = set()
    for m in re.finditer(r"^<(f\w+)>\s+<!", text, re.MULTILINE):
        fid = m.group(1)
        if fid not in seen:
            folios.append(fid)
            seen.add(fid)
    return folios


def build_decoded_text(folio):
    """Build decoded Hebrew text for a folio, using crib results if available."""
    constraints = extract_page_constraints(folio, eva_file=EVA_FILE)
    if not constraints["eva_words"]:
        return None, constraints

    eva_words = constraints["eva_words"]

    # Try loading word-level results
    crib_path = OUTPUT_DIR / f"crib_wordlevel_{folio}.json"
    word_results = {}
    if crib_path.exists():
        with open(crib_path) as f:
            data = json.load(f)
        word_results = data.get("word_results", {})

    lines_heb = []
    pos = 0
    for line_words in constraints["line_structure"]:
        line = []
        for eva in line_words:
            _, heb, _ = decode_word(eva)
            if str(pos) in word_results:
                bc = word_results[str(pos)].get("best_candidate")
                if bc and bc["distance"] == 0:
                    heb = bc["hebrew"]
            line.append(heb)
            pos += 1
        lines_heb.append(line)

    return lines_heb, constraints


def interpret_page(folio, lines_heb, constraints, api_key):
    """Send decoded text to Sonnet for interpretation."""
    section = constraints.get("section_name", "unknown")
    n_words = len(constraints["eva_words"])

    lines_text = "\n".join(
        f"L{i+1:02d}: {' '.join(line)}" for i, line in enumerate(lines_heb)
    )

    section_hint = {
        "herbal": "This is a HERBAL page — expect plant description, extraction recipe, medicinal application.",
        "balneological": "This is a BALNEOLOGICAL page — expect bathing protocol, water temperature, immersion cycles.",
        "pharmaceutical": "This is a PHARMACEUTICAL page — expect ingredient lists, preparation instructions, dosages.",
        "astronomical": "This is an ASTRONOMICAL page — expect celestial observations, calendar, timing.",
        "zodiac": "This is a ZODIAC page — expect zodiac signs, body parts, astrological medicine.",
        "cosmological": "This is a COSMOLOGICAL page — expect cosmic diagrams, elemental theory.",
        "text": "This is a TEXT-ONLY page — expect continuous medical/philosophical text.",
    }.get(section, "Section type unknown.")

    prompt = f"""{VOCAB_PROMPT}

FOLIO: {folio} — Section: {section} — {n_words} words
{section_hint}

DECODED CONSONANTAL HEBREW TEXT:
{lines_text}

Give a CONCISE flowing translation (max 300 words). Then list any NEW vocabulary not in the confirmed list above. Format:

TRANSLATION:
[your translation]

NEW WORDS:
- word = meaning (confidence)"""

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"ERROR: {e}"


def process_folio(folio, api_key, skip_crib=False):
    """Full pipeline for one folio: crib attack + interpretation."""
    result = {"folio": folio, "status": "ok"}

    try:
        # Step 1: Extract constraints
        constraints = extract_page_constraints(folio, eva_file=EVA_FILE)
        if not constraints["eva_words"]:
            result["status"] = "empty"
            result["n_words"] = 0
            return result

        result["section"] = constraints.get("section_name", "unknown")
        result["n_words"] = len(constraints["eva_words"])

        # Step 2: Crib attack (if not skipping and not already done)
        crib_path = OUTPUT_DIR / f"crib_wordlevel_{folio}.json"
        if not skip_crib and not crib_path.exists():
            try:
                run_wordlevel_attack(
                    folio, n_candidates=10, model="claude-sonnet-4-20250514",
                    eva_file=EVA_FILE, db_path=DB_PATH, output_dir=OUTPUT_DIR,
                )
            except Exception as e:
                result["crib_error"] = str(e)

        # Step 3: Build decoded text
        lines_heb, constraints = build_decoded_text(folio)
        if lines_heb is None:
            result["status"] = "no_text"
            return result

        # Step 4: Interpret
        reading = interpret_page(folio, lines_heb, constraints, api_key)
        result["reading"] = reading

        # Save individual reading
        reading_path = READINGS_DIR / f"{folio}.txt"
        with open(reading_path, "w") as f:
            f.write(f"# {folio} — {result['section']} ({result['n_words']} words)\n\n")
            f.write(reading)

        result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Decode all Voynich pages")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--skip-crib", action="store_true", help="Skip crib attack, interpret only")
    parser.add_argument("--folios", nargs="*", help="Specific folios (default: all)")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: No API key found")
        sys.exit(1)

    READINGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folios = args.folios or get_all_folios()
    print(f"Processing {len(folios)} folios with {args.workers} workers")
    print(f"Skip crib: {args.skip_crib}")

    results = []
    done = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_folio, f, api_key, args.skip_crib): f
            for f in folios
        }

        for future in as_completed(futures):
            folio = futures[future]
            try:
                result = future.result()
                results.append(result)
                done += 1
                status = result.get("status", "?")
                section = result.get("section", "?")
                n_words = result.get("n_words", 0)
                elapsed = time.time() - start_time
                rate = done / elapsed * 60 if elapsed > 0 else 0

                if status == "ok":
                    print(f"  [{done}/{len(folios)}] {folio:10s} {section:15s} {n_words:4d}w  OK  ({rate:.1f}/min)")
                elif status == "empty":
                    print(f"  [{done}/{len(folios)}] {folio:10s} (empty)")
                else:
                    errors += 1
                    print(f"  [{done}/{len(folios)}] {folio:10s} ERROR: {result.get('error', '?')[:60]}")
            except Exception as e:
                errors += 1
                done += 1
                print(f"  [{done}/{len(folios)}] {folio:10s} EXCEPTION: {e}")

    elapsed = time.time() - start_time
    print(f"\nDone: {done} pages in {elapsed:.0f}s ({errors} errors)")

    # Save summary
    summary_path = READINGS_DIR / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
