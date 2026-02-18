"""
Full decode: apply complete mapping to entire manuscript.

Uses the Phase 7+9 resolved mapping (19 Hebrew letters):
  - 16 convergent chars (Italian + Hebrew paths agree)
  - f = lamed (allograph of p)
  - ii = he (allograph of r), standalone i = resh (allograph of d)
  - ch = kaf (digraph → single Hebrew letter, Phase 9 B3)
  - n at Hebrew-initial = bet (positional split, Phase 9 B2)
  - q = prefix (stripped)

Output:
  - full_decode.json — structured per-page decode
  - full_decode_readable.txt — human-readable page-by-page text
"""
import json
import re
from collections import Counter, defaultdict

import click

from .config import ToolkitConfig
from .prepare_italian_lexicon import HEBREW_TO_ITALIAN
from .prepare_lexicon import CONSONANT_NAMES
from .utils import print_header, print_step
from .word_structure import parse_eva_words


SECTION_NAMES = {
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

# =====================================================================
# Complete 19-char mapping (Phase 7)
# =====================================================================

# 17-char letter mapping (16 convergent + f=lamed)
FULL_MAPPING = {
    'a': 'y', 'c': 'A', 'd': 'r', 'e': 'p', 'f': 'l',
    'g': 'X', 'h': 'E', 'k': 't', 'l': 'm', 'm': 'g',
    'n': 'd', 'o': 'w', 'p': 'l', 'r': 'h', 's': 'n',
    't': 'J', 'y': 'S',
}

# i as composite glyph: ii→he(h), standalone i→resh(r)
II_HEBREW = 'h'   # he → Italian 'e'
I_HEBREW = 'r'    # resh → Italian 'r'

# ch as digraph → kaf (Phase 9 B3: cohesion 0.722, +72 lexicon matches)
CH_HEBREW = 'k'   # kaf → Italian 'k'

# Positional split: EVA n at Hebrew word-initial → bet instead of dalet
# (Phase 9 B2: +2162 net lexicon matches, differential +2047 vs random)
INITIAL_D_HEBREW = 'b'  # bet → Italian 'b'

# Positional split: EVA r/ii at Hebrew word-initial → samekh instead of he
# (Phase 9 l/r allography: +563 net matches, differential +509 vs random)
INITIAL_H_HEBREW = 's'  # samekh → Italian 'n' (via HEBREW_TO_ITALIAN)

DIRECTION = 'rtl'


# =====================================================================
# Legacy 16-char convergent mapping loader (for backward compat)
# =====================================================================

def load_convergent_mapping(config):
    """Load the 16-char convergent mapping where both paths agree.

    Returns:
        mapping: dict {eva_char: hebrew_char} for agreed chars
        divergent: dict {eva_char: {italian_path, hebrew_path}}
        direction: str
    """
    fuzzy_path = config.stats_dir / "fuzzy_matches.json"
    if not fuzzy_path.exists():
        raise click.ClickException(
            f"Fuzzy decode report not found: {fuzzy_path}\n"
            "  Run first: voynich fuzzy-decode"
        )
    with open(fuzzy_path) as f:
        fuzzy_data = json.load(f)

    italian_mapping = {}
    for eva_ch, info in fuzzy_data["mapping"].items():
        italian_mapping[eva_ch] = info["hebrew"]
    direction = fuzzy_data.get("direction", "rtl")

    hebrew_path = config.stats_dir / "hebrew_decode_report.json"
    if not hebrew_path.exists():
        raise click.ClickException(
            f"Hebrew decode report not found: {hebrew_path}\n"
            "  Run first: voynich hebrew-decode"
        )
    with open(hebrew_path) as f:
        hebrew_data = json.load(f)

    hebrew_mapping = {}
    for eva_ch, info in hebrew_data["mapping"].items():
        hebrew_mapping[eva_ch] = info["hebrew"]

    mapping = {}
    divergent = {}
    for eva_ch in sorted(set(italian_mapping) & set(hebrew_mapping)):
        if italian_mapping[eva_ch] == hebrew_mapping[eva_ch]:
            mapping[eva_ch] = italian_mapping[eva_ch]
        else:
            divergent[eva_ch] = {
                "italian_path": italian_mapping[eva_ch],
                "hebrew_path": hebrew_mapping[eva_ch],
            }

    return mapping, divergent, direction


# =====================================================================
# 19-char decode
# =====================================================================

def preprocess_eva(word):
    """Preprocess EVA word before char-by-char decoding.

    Order:
      1. ch → \\x03 (digraph kaf, before q-prefix to handle "qochedy" etc.)
      2. Strip initial q/qo prefix
      3. ii → \\x01 (he), standalone i → \\x02 (resh)

    Returns: (prefix_stripped, processed_word)
    """
    w = word

    # 1. Replace ch digraph with placeholder (before prefix stripping)
    w = w.replace('ch', '\x03')

    # 2. Strip q/qo prefix
    prefix = ''
    if w.startswith('qo'):
        prefix = 'qo'
        w = w[2:]
    elif w.startswith('q') and len(w) > 1:
        prefix = 'q'
        w = w[1:]

    # 3. Replace runs of i: ii→single token, then standalone i
    w = re.sub(
        r'i{3,}',
        lambda m: '\x01' * (len(m.group()) // 2) +
                  ('\x02' if len(m.group()) % 2 else ''),
        w)
    w = w.replace('ii', '\x01')
    w = w.replace('i', '\x02')
    return prefix, w


def decode_word(eva_word, mapping=None, divergent_chars=None,
                direction=None):
    """Decode a single EVA word to Italian phonemes and Hebrew.

    When called with no arguments (or mapping=None), uses the full
    19-char mapping with ii/i split and q-prefix stripping.

    Legacy mode: when mapping + divergent_chars are provided, uses
    the old 16-char path (for backward compatibility).

    Returns:
        italian: str with Italian phonemes
        hebrew: str with Hebrew ASCII consonants
        n_unknown: count of unmapped chars
    """
    # Legacy 16-char mode
    if mapping is not None and divergent_chars is not None:
        if direction is None:
            direction = "rtl"
        chars = (list(reversed(eva_word)) if direction == "rtl"
                 else list(eva_word))
        hebrew_parts = []
        italian_parts = []
        n_unknown = 0
        for ch in chars:
            if ch in mapping:
                heb = mapping[ch]
                hebrew_parts.append(heb)
                italian_parts.append(HEBREW_TO_ITALIAN.get(heb, "?"))
            elif ch in divergent_chars:
                placeholder = ch.upper()
                hebrew_parts.append(placeholder)
                italian_parts.append(placeholder)
                n_unknown += 1
            else:
                hebrew_parts.append("?")
                italian_parts.append("?")
                n_unknown += 1
        return "".join(italian_parts), "".join(hebrew_parts), n_unknown

    # Full 19-char mode
    use_mapping = mapping if mapping is not None else FULL_MAPPING
    use_direction = direction if direction is not None else DIRECTION

    prefix, processed = preprocess_eva(eva_word)
    chars = (list(reversed(processed)) if use_direction == "rtl"
             else list(processed))

    hebrew_parts = []
    italian_parts = []
    n_unknown = 0

    for ch in chars:
        if ch == '\x01':  # ii → he
            hebrew_parts.append(II_HEBREW)
            italian_parts.append(HEBREW_TO_ITALIAN.get(II_HEBREW, '?'))
        elif ch == '\x02':  # standalone i → resh
            hebrew_parts.append(I_HEBREW)
            italian_parts.append(HEBREW_TO_ITALIAN.get(I_HEBREW, '?'))
        elif ch == '\x03':  # ch → kaf
            hebrew_parts.append(CH_HEBREW)
            italian_parts.append(HEBREW_TO_ITALIAN.get(CH_HEBREW, '?'))
        elif ch in use_mapping:
            heb = use_mapping[ch]
            hebrew_parts.append(heb)
            italian_parts.append(HEBREW_TO_ITALIAN.get(heb, '?'))
        else:
            hebrew_parts.append('?')
            italian_parts.append('?')
            n_unknown += 1

    # Positional split: dalet at word-initial → bet (Phase 9 B2)
    if hebrew_parts and hebrew_parts[0] == 'd':
        hebrew_parts[0] = INITIAL_D_HEBREW
        italian_parts[0] = HEBREW_TO_ITALIAN.get(INITIAL_D_HEBREW, '?')

    # Positional split: he at word-initial → samekh (Phase 9 l/r allography)
    if hebrew_parts and hebrew_parts[0] == 'h':
        hebrew_parts[0] = INITIAL_H_HEBREW
        italian_parts[0] = HEBREW_TO_ITALIAN.get(INITIAL_H_HEBREW, '?')

    return "".join(italian_parts), "".join(hebrew_parts), n_unknown


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Full decode of the manuscript using complete 19-char mapping."""
    report_path = config.stats_dir / "full_decode.json"
    text_path = config.stats_dir / "full_decode_readable.txt"

    if report_path.exists() and not force:
        click.echo("  Full decode report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("FULL DECODE — Complete Mapping (19 Hebrew letters)")

    # 1. Show mapping
    print_step("Using complete mapping (17 chars + ch + ii/i + q + positional split)...")
    click.echo(f"    17 letter-mapped + ch→kaf + ii→he + i→resh + q→prefix")
    click.echo(f"    Positional: n@initial→bet (Phase 9 B2)")
    click.echo(f"    Direction: {DIRECTION}")

    for eva_ch in sorted(FULL_MAPPING):
        heb = FULL_MAPPING[eva_ch]
        it = HEBREW_TO_ITALIAN.get(heb, "?")
        click.echo(f"      {eva_ch} -> {heb} "
                   f"({CONSONANT_NAMES.get(heb, '?'):8s}) -> {it}")
    click.echo(f"      ch -> {CH_HEBREW} "
               f"({CONSONANT_NAMES.get(CH_HEBREW, '?'):8s}) -> "
               f"{HEBREW_TO_ITALIAN.get(CH_HEBREW, '?')}")
    click.echo(f"      ii -> {II_HEBREW} "
               f"({CONSONANT_NAMES.get(II_HEBREW, '?'):8s}) -> "
               f"{HEBREW_TO_ITALIAN.get(II_HEBREW, '?')}")
    click.echo(f"      i  -> {I_HEBREW} "
               f"({CONSONANT_NAMES.get(I_HEBREW, '?'):8s}) -> "
               f"{HEBREW_TO_ITALIAN.get(I_HEBREW, '?')}")
    click.echo(f"      q  -> prefix (stripped)")
    click.echo(f"      n@initial -> {INITIAL_D_HEBREW} "
               f"({CONSONANT_NAMES.get(INITIAL_D_HEBREW, '?'):8s}) -> "
               f"{HEBREW_TO_ITALIAN.get(INITIAL_D_HEBREW, '?')} "
               f"(positional split)")
    click.echo(f"      r/ii@initial -> {INITIAL_H_HEBREW} "
               f"({CONSONANT_NAMES.get(INITIAL_H_HEBREW, '?'):8s}) -> "
               f"{HEBREW_TO_ITALIAN.get(INITIAL_H_HEBREW, '?')} "
               f"(positional split, l/r allography)")

    # 2. Parse EVA
    print_step("Parsing EVA text...")
    eva_file = config.eva_data_dir / "LSI_ivtff_0d.txt"
    if not eva_file.exists():
        raise click.ClickException(f"EVA file not found: {eva_file}")
    eva_data = parse_eva_words(eva_file)
    click.echo(f"    {len(eva_data['pages'])} pages, "
               f"{eva_data['total_words']} words")

    # 3. Decode all pages
    print_step("Decoding all pages...")
    pages = {}
    total_words = 0
    total_fully_decoded = 0
    total_with_unknowns = 0

    text_lines = []

    for page in eva_data["pages"]:
        folio = page["folio"]
        section = page.get("section", "?")
        section_name = SECTION_NAMES.get(section, section)
        words_eva = page["words"]

        words_decoded = []
        words_hebrew = []
        n_fully = 0
        n_with_unk = 0

        for w in words_eva:
            ita, heb, n_unk = decode_word(w)
            words_decoded.append(ita)
            words_hebrew.append(heb)
            if n_unk == 0:
                n_fully += 1
            else:
                n_with_unk += 1

        pages[folio] = {
            "section": section_name,
            "section_code": section,
            "words_eva": words_eva,
            "words_decoded": words_decoded,
            "words_hebrew": words_hebrew,
            "words_with_unknowns": n_with_unk,
            "words_fully_decoded": n_fully,
        }

        total_words += len(words_eva)
        total_fully_decoded += n_fully
        total_with_unknowns += n_with_unk

        # Build readable text
        text_lines.append(f"\n{'=' * 60}")
        text_lines.append(f"  {folio}  [{section_name}]")
        text_lines.append(f"{'=' * 60}")

        for line_words in page.get("line_words", []):
            if not line_words:
                continue
            eva_line = " ".join(line_words)
            decoded_parts = []
            for w in line_words:
                ita, _, _ = decode_word(w)
                decoded_parts.append(ita)
            text_lines.append(f"  EVA: {eva_line}")
            text_lines.append(f"  ITA: {' '.join(decoded_parts)}")
            text_lines.append("")

    fully_pct = (total_fully_decoded / total_words * 100) if total_words else 0

    # 4. Save JSON report
    print_step("Saving reports...")

    # Build mapping info for report
    full_mapping_info = {}
    for eva_ch in sorted(FULL_MAPPING):
        heb = FULL_MAPPING[eva_ch]
        full_mapping_info[eva_ch] = {
            "hebrew": heb,
            "hebrew_name": CONSONANT_NAMES.get(heb, "?"),
            "italian": HEBREW_TO_ITALIAN.get(heb, "?"),
        }
    # Add ch, ii and i entries
    full_mapping_info["ch"] = {
        "hebrew": CH_HEBREW,
        "hebrew_name": CONSONANT_NAMES.get(CH_HEBREW, "?"),
        "italian": HEBREW_TO_ITALIAN.get(CH_HEBREW, "?"),
        "note": "digraph → single kaf (Phase 9 B3)",
    }
    full_mapping_info["ii"] = {
        "hebrew": II_HEBREW,
        "hebrew_name": CONSONANT_NAMES.get(II_HEBREW, "?"),
        "italian": HEBREW_TO_ITALIAN.get(II_HEBREW, "?"),
        "note": "digraph → single phoneme",
    }
    full_mapping_info["i"] = {
        "hebrew": I_HEBREW,
        "hebrew_name": CONSONANT_NAMES.get(I_HEBREW, "?"),
        "italian": HEBREW_TO_ITALIAN.get(I_HEBREW, "?"),
        "note": "standalone (not part of ii)",
    }
    full_mapping_info["q"] = {
        "hebrew": "—",
        "hebrew_name": "prefix",
        "italian": "—",
        "note": "grammatical prefix, stripped",
    }
    full_mapping_info["n@initial"] = {
        "hebrew": INITIAL_D_HEBREW,
        "hebrew_name": CONSONANT_NAMES.get(INITIAL_D_HEBREW, "?"),
        "italian": HEBREW_TO_ITALIAN.get(INITIAL_D_HEBREW, "?"),
        "note": "positional split: n at Hebrew-initial → bet (Phase 9 B2)",
    }
    full_mapping_info["r/ii@initial"] = {
        "hebrew": INITIAL_H_HEBREW,
        "hebrew_name": CONSONANT_NAMES.get(INITIAL_H_HEBREW, "?"),
        "italian": HEBREW_TO_ITALIAN.get(INITIAL_H_HEBREW, "?"),
        "note": "positional split: he at Hebrew-initial → samekh (l/r allography)",
    }

    report = {
        "direction": DIRECTION,
        "mapping_size": 22,
        "mapping_type": "full_20char_plus_2_positional",
        "hebrew_letters_mapped": 19,
        "divergent_chars": [],
        "mapping": full_mapping_info,
        "pages": pages,
        "total_words": total_words,
        "fully_decoded_pct": round(fully_pct, 1),
        "words_fully_decoded": total_fully_decoded,
        "words_with_unknowns": total_with_unknowns,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    click.echo(f"    JSON: {report_path}")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    click.echo(f"    Text: {text_path}")

    # 5. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  FULL DECODE RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Total words: {total_words}")
    click.echo(f"  Fully decoded: {total_fully_decoded} ({fully_pct:.1f}%)")
    click.echo(f"  With unknowns: {total_with_unknowns}")

    # Section breakdown
    section_stats = defaultdict(lambda: {"total": 0, "fully": 0})
    for pdata in pages.values():
        sec = pdata["section"]
        section_stats[sec]["total"] += len(pdata["words_eva"])
        section_stats[sec]["fully"] += pdata["words_fully_decoded"]

    click.echo(f"\n  By section:")
    for sec in sorted(section_stats, key=lambda s: -section_stats[s]["total"]):
        s = section_stats[sec]
        pct = (s["fully"] / s["total"] * 100) if s["total"] else 0
        click.echo(f"    {sec:15s}: {s['total']:6d} words, "
                   f"{s['fully']:6d} fully decoded ({pct:.1f}%)")

    # Show sample decoded words
    click.echo(f"\n  Sample decoded words (first 20 fully-decoded, len>=4):")
    shown = 0
    for page in eva_data["pages"]:
        if shown >= 20:
            break
        for w in page["words"]:
            if shown >= 20:
                break
            if len(w) >= 4:
                ita, heb, n_unk = decode_word(w)
                if n_unk == 0:
                    click.echo(f"    {w:12s} -> {heb:8s} -> {ita}")
                    shown += 1
