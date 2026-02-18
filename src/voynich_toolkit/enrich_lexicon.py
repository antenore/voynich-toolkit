"""
Enrich Hebrew lexicon: download Jastrow dictionary, filter proper nouns,
merge with STEPBible lexicon.

Phase 8A: triples the Hebrew lexicon (~6,400 → ~18,000 forms), removes
biblical proper nouns that contaminate match rates.
"""
import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import requests

from .config import ToolkitConfig
from .prepare_lexicon import (
    CONSONANT_NAMES,
    HEBREW_TO_ASCII,
    hebrew_to_consonants,
    strip_nikkud,
)
from .utils import print_header, print_step, timer


# =====================================================================
# Jastrow download & parse
# =====================================================================

JASTROW_URL = (
    "https://raw.githubusercontent.com/BraydenKO/Jastrow-Klein-Dicts/"
    "main/Jastrow/data/01-Merged%20XML/Jastrow-full.csv"
)


@timer
def download_jastrow(dest_dir: Path) -> Path:
    """Download Jastrow dictionary CSV (cached)."""
    dest = dest_dir / "jastrow_full.csv"
    if dest.exists():
        print(f"    Cache trovata: {dest}")
        return dest

    print("    Download Jastrow dictionary...")
    resp = requests.get(JASTROW_URL, timeout=60)
    if resp.status_code != 200:
        raise click.ClickException(
            f"Jastrow download failed: HTTP {resp.status_code}"
        )
    dest.write_text(resp.text, encoding="utf-8")
    print(f"    Salvato: {dest} ({len(resp.text)} bytes)")
    return dest


@timer
def parse_jastrow(filepath: Path) -> list[dict]:
    """Parse Jastrow CSV into structured entries.

    Conservative parsing: skip entries without clear Hebrew headword.
    Returns list of {hebrew, consonants, gloss, source, language_tag}.
    """
    text = filepath.read_text(encoding="utf-8")
    entries = []

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        # Try common column names
        hebrew = (row.get("headword") or row.get("Headword")
                  or row.get("word") or row.get("Word")
                  or row.get("entry") or row.get("Entry") or "")
        gloss = (row.get("definition") or row.get("Definition")
                 or row.get("gloss") or row.get("Gloss")
                 or row.get("meaning") or row.get("Meaning") or "")
        lang_tag = (row.get("language") or row.get("Language")
                    or row.get("lang") or "")

        # Must have some Hebrew content
        hebrew = hebrew.strip()
        if not hebrew:
            continue
        if not any('\u05D0' <= ch <= '\u05EA' for ch in hebrew):
            continue

        # Extract consonantal form
        consonants = hebrew_to_consonants(hebrew)
        if not consonants or len(consonants) < 2:
            continue

        entries.append({
            "hebrew": strip_nikkud(hebrew),
            "consonants": consonants,
            "gloss": gloss.strip(),
            "source": "Jastrow",
            "language_tag": lang_tag.strip(),
        })

    return entries


# If the CSV has no headers or different structure, try fallback parsing
@timer
def parse_jastrow_fallback(filepath: Path) -> list[dict]:
    """Fallback parser for Jastrow CSV with non-standard format."""
    text = filepath.read_text(encoding="utf-8")
    entries = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue

        # Find the Hebrew part and the English part
        hebrew = ""
        gloss = ""
        lang_tag = ""
        for part in parts:
            part = part.strip().strip('"')
            if any('\u05D0' <= ch <= '\u05EA' for ch in part):
                if not hebrew:
                    hebrew = part
            elif re.match(r'^[a-zA-Z]', part) and len(part) > 1:
                if part.lower() in ("ch.", "ch", "heb.", "heb",
                                     "aramaic", "hebrew"):
                    lang_tag = part
                elif not gloss:
                    gloss = part
                elif len(part) > len(gloss):
                    gloss = part

        if hebrew:
            consonants = hebrew_to_consonants(hebrew)
            if consonants and len(consonants) >= 2:
                entries.append({
                    "hebrew": strip_nikkud(hebrew),
                    "consonants": consonants,
                    "gloss": gloss,
                    "source": "Jastrow",
                    "language_tag": lang_tag,
                })

    return entries


# =====================================================================
# Proper noun filter
# =====================================================================

# Patterns in English glosses that indicate proper nouns
PROPER_NOUN_PATTERNS = [
    r'\bname of\b',
    r'\bpr\.?\s*n\b',          # "pr. n.", "pr.n."
    r'\bproper name\b',
    r'\bpersonal name\b',
    r'\bplace name\b',
    r'\ba place\b',
    r'\ba city\b',
    r'\ba town\b',
    r'\ba country\b',
    r'\ba river\b',
    r'\ba mountain\b',
    r'\bson of\b',
    r'\bdaughter of\b',
    r'\btribe of\b',
    r'\bking of\b',
    r'\bpriest of\b',
    r'\bprophet\b',
    r'\bpatriarch\b',
    r'\bbiblical\b',
    r'\bIsrael\b',
    r'\bJerusalem\b',
    r'\bEgypt\b',
    r'\bBabylon\b',
    r'\bAssyria\b',
    r'\bCanaan\b',
    r'\bJudah\b',
    r'\bEphraim\b',
    r'\bManasseh\b',
    r'\bBenjamin\b',
    r'\bGad\b(?!.*garden)',     # Gad the tribe, not gad (coriander)
    r'\bDan\b',
    r'\bLevi\b',
    r'\bMoses\b',
    r'\bAaron\b',
    r'\bAbraham\b',
    r'\bIsaac\b',
    r'\bJacob\b',
    r'\bDavid\b',
    r'\bSolomon\b',
    r'\bSamuel\b',
    r'\bElijah\b',
    r'\bElisha\b',
]

_PROPER_NOUN_RE = re.compile(
    "|".join(PROPER_NOUN_PATTERNS), re.IGNORECASE
)


def is_proper_noun(entry: dict) -> bool:
    """Return True if the entry is likely a proper noun."""
    gloss = entry.get("gloss", "")
    if not gloss:
        return False
    return bool(_PROPER_NOUN_RE.search(gloss))


def filter_proper_nouns(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split entries into (kept, filtered_out)."""
    kept = []
    removed = []
    for entry in entries:
        if is_proper_noun(entry):
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


# =====================================================================
# Aramaic extraction from Jastrow
# =====================================================================

def extract_aramaic(jastrow_entries: list[dict]) -> list[dict]:
    """Extract Aramaic entries from Jastrow (marked as 'Ch.' = Chaldaic).

    These are used as a cross-language control in Phase 8C.
    """
    aramaic = []
    for entry in jastrow_entries:
        lang = entry.get("language_tag", "").lower()
        gloss = entry.get("gloss", "").lower()
        if ("ch." in lang or "ch" == lang or "aramaic" in lang
                or "chald" in lang or "ch." in gloss[:20]):
            aramaic.append(entry)
    return aramaic


# =====================================================================
# Merge lexicons
# =====================================================================

@timer
def merge_lexicons(existing_lexicon: dict, jastrow_entries: list[dict],
                   ) -> dict:
    """Merge Jastrow entries into existing lexicon.

    Deduplicates by consonantal form. Keeps both glosses.
    Returns enriched lexicon dict.
    """
    # Start with existing data
    by_consonants = defaultdict(list)
    for cons, entries in existing_lexicon.get("by_consonants", {}).items():
        by_consonants[cons].extend(entries)

    by_domain = defaultdict(list)
    for domain, entries in existing_lexicon.get("by_domain", {}).items():
        by_domain[domain].extend(entries)

    existing_forms = set(existing_lexicon.get("all_consonantal_forms", []))

    # Track what we add
    new_forms = 0
    new_entries = 0

    for entry in jastrow_entries:
        cons = entry["consonants"]
        domain = "general"  # Jastrow entries are general by default
        gloss = entry.get("gloss", "")

        new_entry = {
            "hebrew": entry["hebrew"],
            "gloss": gloss,
            "domain": domain,
            "source": "Jastrow",
        }

        if cons not in existing_forms:
            new_forms += 1
            existing_forms.add(cons)

        # Always add the entry (even if form exists, gloss may differ)
        # But deduplicate by (consonants, source)
        existing_glosses = {
            e.get("source") for e in by_consonants.get(cons, [])
        }
        if "Jastrow" not in existing_glosses:
            by_consonants[cons].append(new_entry)
            by_domain[domain].append({
                "hebrew": entry["hebrew"],
                "consonants": cons,
                "gloss": gloss,
                "source": "Jastrow",
            })
            new_entries += 1

    all_forms = sorted(existing_forms)

    # Build form_to_gloss for bidirectional lookup
    form_to_gloss = {}
    for cons, entries in by_consonants.items():
        glosses = [e["gloss"] for e in entries if e.get("gloss")]
        if glosses:
            form_to_gloss[cons] = glosses[0]  # First gloss as primary

    # Compute stats
    lengths = Counter(len(c) for c in all_forms)
    sources = Counter()
    for cons, entries in by_consonants.items():
        for e in entries:
            sources[e.get("source", "unknown")] += 1

    domain_counts = {d: len(entries) for d, entries in by_domain.items()}

    stats = {
        "total_entries": sum(len(v) for v in by_consonants.values()),
        "unique_consonantal_forms": len(all_forms),
        "by_domain_count": domain_counts,
        "length_distribution": dict(sorted(lengths.items())),
        "sources": dict(sources.most_common()),
        "new_forms_from_jastrow": new_forms,
        "new_entries_from_jastrow": new_entries,
    }

    return {
        "by_domain": dict(by_domain),
        "by_consonants": dict(by_consonants),
        "all_consonantal_forms": all_forms,
        "form_to_gloss": form_to_gloss,
        "consonant_map": CONSONANT_NAMES,
        "stats": stats,
    }


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force: bool = False) -> None:
    """Enrich Hebrew lexicon with Jastrow dictionary + filter proper nouns."""
    print_header("PHASE 8A — Lexicon Enrichment + Proper Noun Filter")
    config.ensure_dirs()

    enriched_path = config.lexicon_dir / "lexicon_enriched.json"
    report_path = config.stats_dir / "lexicon_enrichment_report.json"
    aramaic_path = config.lexicon_dir / "aramaic_lexicon.json"

    if enriched_path.exists() and not force:
        click.echo("  Enriched lexicon exists. Use --force to re-run.")
        return

    # 1. Load existing lexicon
    print_step("Loading existing lexicon...")
    base_lexicon_path = config.lexicon_dir / "lexicon.json"
    if not base_lexicon_path.exists():
        raise click.ClickException(
            f"Base lexicon not found: {base_lexicon_path}\n"
            "  Run first: voynich prepare-lexicon"
        )
    with open(base_lexicon_path) as f:
        base_lexicon = json.load(f)

    n_base = base_lexicon["stats"]["unique_consonantal_forms"]
    click.echo(f"    Base lexicon: {n_base} unique forms")

    # 2. Download and parse Jastrow
    print_step("Downloading Jastrow dictionary...")
    jastrow_file = download_jastrow(config.lexicon_dir)

    print_step("Parsing Jastrow dictionary...")
    jastrow_entries = parse_jastrow(jastrow_file)
    if len(jastrow_entries) < 100:
        click.echo(f"    Standard parse yielded {len(jastrow_entries)} — "
                    "trying fallback parser...")
        jastrow_entries = parse_jastrow_fallback(jastrow_file)

    click.echo(f"    Parsed {len(jastrow_entries)} Jastrow entries")

    # 3. Extract Aramaic subset (for Phase 8C)
    print_step("Extracting Aramaic subset...")
    aramaic_entries = extract_aramaic(jastrow_entries)
    click.echo(f"    {len(aramaic_entries)} Aramaic entries found")

    # Save Aramaic lexicon
    aramaic_forms = sorted(set(e["consonants"] for e in aramaic_entries))
    aramaic_data = {
        "all_consonantal_forms": aramaic_forms,
        "n_forms": len(aramaic_forms),
        "n_entries": len(aramaic_entries),
        "source": "Jastrow-Aramaic",
    }
    with open(aramaic_path, "w", encoding="utf-8") as f:
        json.dump(aramaic_data, f, ensure_ascii=False, indent=2)
    click.echo(f"    Saved: {aramaic_path}")

    # 4. Filter proper nouns from ALL sources
    print_step("Filtering proper nouns from all sources...")

    # Filter Jastrow
    jastrow_clean, jastrow_pn = filter_proper_nouns(jastrow_entries)
    click.echo(f"    Jastrow: {len(jastrow_pn)} proper nouns removed, "
               f"{len(jastrow_clean)} kept")

    # Filter existing lexicon entries (rebuild flat list)
    base_entries = []
    for cons, entries_list in base_lexicon.get("by_consonants", {}).items():
        for e in entries_list:
            base_entries.append({
                "hebrew": e.get("hebrew", ""),
                "consonants": cons,
                "gloss": e.get("gloss", ""),
                "source": e.get("source", "STEPBible"),
                "domain": e.get("domain", "general"),
            })

    base_clean, base_pn = filter_proper_nouns(base_entries)
    click.echo(f"    Base lexicon: {len(base_pn)} proper nouns removed, "
               f"{len(base_clean)} kept")

    total_pn = len(jastrow_pn) + len(base_pn)
    click.echo(f"    Total proper nouns removed: {total_pn}")

    # 5. Rebuild clean base lexicon
    print_step("Rebuilding clean base lexicon...")
    clean_by_consonants = defaultdict(list)
    clean_by_domain = defaultdict(list)
    clean_forms = set()

    for entry in base_clean:
        cons = entry["consonants"]
        clean_forms.add(cons)
        domain = entry.get("domain", "general")
        clean_by_consonants[cons].append({
            "hebrew": entry["hebrew"],
            "gloss": entry["gloss"],
            "domain": domain,
            "source": entry["source"],
        })
        clean_by_domain[domain].append({
            "hebrew": entry["hebrew"],
            "consonants": cons,
            "gloss": entry["gloss"],
            "source": entry["source"],
        })

    clean_base = {
        "by_domain": dict(clean_by_domain),
        "by_consonants": dict(clean_by_consonants),
        "all_consonantal_forms": sorted(clean_forms),
        "stats": {"unique_consonantal_forms": len(clean_forms)},
    }

    # 6. Merge Jastrow into clean base
    print_step("Merging Jastrow into cleaned lexicon...")
    enriched = merge_lexicons(clean_base, jastrow_clean)

    n_after_jastrow = enriched["stats"]["unique_consonantal_forms"]
    click.echo(f"    After Jastrow: {n_after_jastrow} unique forms "
               f"(was {n_base})")

    # 6b. Merge Sefaria corpus (attested forms from 250M-token corpus)
    sefaria_corpus_path = config.lexicon_dir / "sefaria_corpus.json"
    n_sefaria_new = 0
    sefaria_min_freq = 5  # Only include forms attested ≥5 times

    if sefaria_corpus_path.exists():
        print_step(f"Merging Sefaria corpus (freq≥{sefaria_min_freq})...")
        with open(sefaria_corpus_path) as f:
            sefaria_data = json.load(f)

        existing_forms_set = set(enriched["all_consonantal_forms"])
        corpus_forms = sefaria_data.get("forms", {})

        for form, freq in corpus_forms.items():
            if freq >= sefaria_min_freq and form not in existing_forms_set:
                existing_forms_set.add(form)
                n_sefaria_new += 1
                # Add minimal entry to by_consonants
                enriched["by_consonants"][form] = [{
                    "hebrew": "",
                    "gloss": "",
                    "domain": "general",
                    "source": "Sefaria-Corpus",
                }]
                enriched["by_domain"].setdefault("general", []).append({
                    "hebrew": "",
                    "consonants": form,
                    "gloss": "",
                    "source": "Sefaria-Corpus",
                })

        enriched["all_consonantal_forms"] = sorted(existing_forms_set)
        enriched["stats"]["unique_consonantal_forms"] = len(existing_forms_set)
        enriched["stats"]["total_entries"] += n_sefaria_new
        enriched["stats"]["sources"]["Sefaria-Corpus"] = n_sefaria_new
        enriched["stats"]["new_forms_from_sefaria"] = n_sefaria_new
        enriched["stats"]["sefaria_min_freq"] = sefaria_min_freq
        enriched["stats"]["sefaria_corpus_tokens"] = sefaria_data.get(
            "total_tokens", 0)

        click.echo(f"    Sefaria corpus: +{n_sefaria_new:,} new forms "
                   f"(freq≥{sefaria_min_freq})")
    else:
        click.echo("    Sefaria corpus not found — skipping "
                   "(run extraction first)")

    # 6c. Merge Klein Dictionary (etymological headwords)
    klein_path = config.lexicon_dir / "klein_dictionary.json"
    n_klein_new = 0

    if klein_path.exists():
        print_step("Merging Klein Dictionary...")
        with open(klein_path) as f:
            klein_data = json.load(f)

        existing_forms_set = set(enriched["all_consonantal_forms"])
        klein_forms = klein_data.get("forms", {})

        for form, hebrew_hw in klein_forms.items():
            if len(form) >= 2 and form not in existing_forms_set:
                existing_forms_set.add(form)
                n_klein_new += 1
                enriched["by_consonants"][form] = [{
                    "hebrew": hebrew_hw,
                    "gloss": "",
                    "domain": "general",
                    "source": "Klein",
                }]
                enriched["by_domain"].setdefault("general", []).append({
                    "hebrew": hebrew_hw,
                    "consonants": form,
                    "gloss": "",
                    "source": "Klein",
                })

        enriched["all_consonantal_forms"] = sorted(existing_forms_set)
        enriched["stats"]["unique_consonantal_forms"] = len(existing_forms_set)
        enriched["stats"]["total_entries"] += n_klein_new
        enriched["stats"]["sources"]["Klein"] = n_klein_new
        enriched["stats"]["new_forms_from_klein"] = n_klein_new

        click.echo(f"    Klein Dictionary: +{n_klein_new:,} new forms")
    else:
        click.echo("    Klein Dictionary not found — skipping")

    n_enriched = enriched["stats"]["unique_consonantal_forms"]
    click.echo(f"    Enriched total: {n_enriched:,} unique forms")

    # 7. Save enriched lexicon
    print_step("Saving enriched lexicon...")
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    click.echo(f"    Saved: {enriched_path}")

    # 8. Save enrichment report
    report = {
        "base_lexicon_forms": n_base,
        "after_jastrow_forms": n_after_jastrow,
        "sefaria_new_forms": n_sefaria_new,
        "klein_new_forms": n_klein_new,
        "enriched_lexicon_forms": n_enriched,
        "growth_factor": round(n_enriched / max(n_base, 1), 2),
        "jastrow_raw_entries": len(jastrow_entries),
        "jastrow_after_pn_filter": len(jastrow_clean),
        "jastrow_proper_nouns_removed": len(jastrow_pn),
        "base_proper_nouns_removed": len(base_pn),
        "total_proper_nouns_removed": total_pn,
        "aramaic_entries": len(aramaic_entries),
        "aramaic_unique_forms": len(aramaic_forms),
        "enriched_stats": enriched["stats"],
        "proper_noun_sample": [
            {"hebrew": e["hebrew"], "gloss": e["gloss"]}
            for e in (jastrow_pn + base_pn)[:20]
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f"    Report: {report_path}")

    # 9. Console summary
    click.echo(f"\n{'=' * 60}")
    click.echo("  LEXICON ENRICHMENT RESULTS")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Base lexicon:     {n_base:,} unique forms")
    click.echo(f"  After PN filter:  {len(clean_forms):,} forms "
               f"(-{len(base_pn)} proper nouns)")
    click.echo(f"  + Jastrow:        +{enriched['stats']['new_forms_from_jastrow']:,} new forms")
    if n_sefaria_new:
        click.echo(f"  + Sefaria corpus: +{n_sefaria_new:,} new forms "
                   f"(freq≥{sefaria_min_freq})")
    if n_klein_new:
        click.echo(f"  + Klein Dict:     +{n_klein_new:,} new forms")
    click.echo(f"  Enriched total:   {n_enriched:,} unique forms")
    click.echo(f"  Growth:           {report['growth_factor']}x")
    click.echo(f"\n  Proper nouns removed: {total_pn}")
    if report["proper_noun_sample"]:
        click.echo("  Sample proper nouns removed:")
        for pn in report["proper_noun_sample"][:5]:
            click.echo(f"    {pn['hebrew']} — {pn['gloss'][:60]}")

    click.echo(f"\n  Aramaic subset: {len(aramaic_forms)} forms "
               f"(saved for cross-language baseline)")

    click.echo(f"\n  Sources in enriched lexicon:")
    for source, count in enriched["stats"]["sources"].items():
        click.echo(f"    {source}: {count}")

    click.echo(f"\n  {'=' * 40}")
    click.echo(f"  Output: {enriched_path}")
    click.echo(f"  Report: {report_path}")
    click.echo(f"  {'=' * 40}")
