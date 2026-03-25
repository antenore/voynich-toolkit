#!/usr/bin/env python3
"""Download North Italian dialect lexicons from kaikki.org.

Downloads Venetian, Ligurian, Emilian, Lombard, and Friulian
word lists from kaikki.org Wiktionary extracts (JSONL format).

Output: output/lexicon/north_italian_lexicons.json
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

# kaikki.org stores per-language JSONL extracts.
# URL pattern varies: some use language name, some ISO code.
LANGUAGES = {
    "Venetian": "Venetan",      # kaikki.org uses "Venetan" not "Venetian"
    "Ligurian": "Ligurian",
    "Emilian": "Emilian",
    "Lombard": "Lombard",
    "Friulian": "Friulian",
    "Romagnol": "Romagnol",
    "Piedmontese": "Piedmontese",
    "Ladin": "Ladin",
    "Neapolitan": "Neapolitan",
}

# Multiple URL patterns to try (kaikki.org URL structure varies)
URL_PATTERNS = [
    "https://kaikki.org/dictionary/{lang}/kaikki.org-dictionary-{lang}.jsonl",
    "https://kaikki.org/dictionary/{lang}/kaikki.org-dictionary-{lang}.json",
]


def download_language(lang_key: str, lang_name: str) -> list[str]:
    """Download word forms for a language from kaikki.org."""
    forms = set()

    for pattern in URL_PATTERNS:
        url = pattern.format(lang=lang_name)
        print(f"  Trying {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voynich-toolkit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    word = entry.get("word", "").strip().lower()
                    if word and len(word) >= 2 and word.isalpha():
                        forms.add(word)
            print(f"    OK: {len(forms)} forms")
            return sorted(forms)
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code}")
        except urllib.error.URLError as e:
            print(f"    Error: {e.reason}")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"    WARNING: could not download {lang_key}")
    return []


def main():
    output_dir = Path(__file__).resolve().parent.parent / "output" / "lexicon"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "north_italian_lexicons.json"

    result = {}
    total = 0

    for lang_key, lang_name in LANGUAGES.items():
        print(f"\n[{lang_key}]")
        forms = download_language(lang_key, lang_name)
        result[lang_key.lower()] = {
            "source": f"kaikki.org-{lang_name}",
            "n_forms": len(forms),
            "forms": forms,
        }
        total += len(forms)

    # Combined unique forms
    all_forms = set()
    for lang_data in result.values():
        all_forms.update(lang_data["forms"])

    result["combined"] = {
        "n_unique": len(all_forms),
        "n_total": total,
        "forms": sorted(all_forms),
    }

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {output_path}")
    print(f"Total: {total} forms, {len(all_forms)} unique")


if __name__ == "__main__":
    main()
