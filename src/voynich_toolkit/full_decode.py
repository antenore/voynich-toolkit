"""
Full decode: apply convergent 16-char mapping to entire manuscript.

Loads the convergent mapping (16 chars where Italian and Hebrew paths
agree), decodes all EVA text, and uses F/I/Q placeholders for the
3 divergent characters.

Output:
  - full_decode.json — structured per-page decode
  - full_decode_readable.txt — human-readable page-by-page text
"""
import json
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


def decode_word(eva_word, mapping, divergent_chars, direction):
    """Decode a single EVA word to Italian phonemes and Hebrew.

    Returns:
        italian: str with Italian phonemes + F/I/Q placeholders
        hebrew: str with Hebrew ASCII + F/I/Q placeholders
        n_unknown: count of unknown (divergent) chars
    """
    chars = list(reversed(eva_word)) if direction == "rtl" else list(eva_word)
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


def run(config: ToolkitConfig, force=False, **kwargs):
    """Full decode of the manuscript using convergent 16-char mapping."""
    report_path = config.stats_dir / "full_decode.json"
    text_path = config.stats_dir / "full_decode_readable.txt"

    if report_path.exists() and not force:
        click.echo("  Full decode report exists. Use --force to re-run.")
        return

    config.ensure_dirs()
    print_header("FULL DECODE — Convergent 16-char Mapping")

    # 1. Load convergent mapping
    print_step("Loading convergent mapping...")
    mapping, divergent, direction = load_convergent_mapping(config)
    click.echo(f"    {len(mapping)} agreed chars, "
               f"{len(divergent)} divergent ({', '.join(sorted(divergent))})")
    click.echo(f"    Direction: {direction}")

    for eva_ch in sorted(mapping):
        heb = mapping[eva_ch]
        it = HEBREW_TO_ITALIAN.get(heb, "?")
        click.echo(f"      {eva_ch} -> {heb} "
                   f"({CONSONANT_NAMES.get(heb, '?'):8s}) -> {it}")

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
    unknown_char_counter = Counter()

    text_lines = []
    divergent_set = set(divergent.keys())

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
            ita, heb, n_unk = decode_word(w, mapping, divergent_set, direction)
            words_decoded.append(ita)
            words_hebrew.append(heb)
            if n_unk == 0:
                n_fully += 1
            else:
                n_with_unk += 1
                for ch in (reversed(w) if direction == "rtl" else w):
                    if ch in divergent_set:
                        unknown_char_counter[ch.upper()] += 1

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
                ita, _, _ = decode_word(w, mapping, divergent_set, direction)
                decoded_parts.append(ita)
            text_lines.append(f"  EVA: {eva_line}")
            text_lines.append(f"  ITA: {' '.join(decoded_parts)}")
            text_lines.append("")

    fully_pct = (total_fully_decoded / total_words * 100) if total_words else 0

    # 4. Save JSON report
    print_step("Saving reports...")
    report = {
        "direction": direction,
        "mapping_size": len(mapping),
        "divergent_chars": sorted(divergent.keys()),
        "mapping": {
            eva_ch: {
                "hebrew": mapping[eva_ch],
                "hebrew_name": CONSONANT_NAMES.get(mapping[eva_ch], "?"),
                "italian": HEBREW_TO_ITALIAN.get(mapping[eva_ch], "?"),
            }
            for eva_ch in sorted(mapping)
        },
        "divergent_details": {
            ch: {
                "italian_path": info["italian_path"],
                "italian_path_name": CONSONANT_NAMES.get(
                    info["italian_path"], "?"),
                "hebrew_path": info["hebrew_path"],
                "hebrew_path_name": CONSONANT_NAMES.get(
                    info["hebrew_path"], "?"),
            }
            for ch, info in divergent.items()
        },
        "pages": pages,
        "total_words": total_words,
        "fully_decoded_pct": round(fully_pct, 1),
        "words_fully_decoded": total_fully_decoded,
        "words_with_unknowns": total_with_unknowns,
        "unknown_char_frequency": dict(unknown_char_counter.most_common()),
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

    click.echo(f"\n  Unknown char frequency:")
    for ch, count in unknown_char_counter.most_common():
        click.echo(f"    {ch}: {count}")

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
                ita, heb, n_unk = decode_word(
                    w, mapping, divergent_set, direction)
                if n_unk == 0:
                    click.echo(f"    {w:12s} -> {heb:8s} -> {ita}")
                    shown += 1
