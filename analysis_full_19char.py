"""
Full 19-char decode: the complete mapping.

16 convergent + f=lamed + ii=he + standalone_i=resh + q=prefix(strip)
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

from voynich_toolkit.word_structure import parse_eva_words
from voynich_toolkit.prepare_italian_lexicon import HEBREW_TO_ITALIAN
from voynich_toolkit.prepare_lexicon import CONSONANT_NAMES
from voynich_toolkit.full_decode import SECTION_NAMES

# =====================================================================
# COMPLETE MAPPING
# =====================================================================

# 17-char letter mapping (16 convergent + f=lamed)
MAPPING = {
    'a': 'y', 'c': 'A', 'd': 'r', 'e': 'p', 'f': 'l',
    'g': 'X', 'h': 'E', 'k': 't', 'l': 'm', 'm': 'g',
    'n': 'd', 'o': 'w', 'p': 'l', 'r': 'h', 's': 'n',
    't': 'J', 'y': 'S',
}

# Model A: ii→he(h), standalone_i→resh(r)
II_HEBREW = 'h'   # he → Italian 'e'
I_HEBREW = 'r'    # resh → Italian 'r'

DIRECTION = 'rtl'


def preprocess_eva(word):
    """Replace ii→\x01, standalone i→\x02, strip initial q/qo."""
    # Strip initial q-prefix
    prefix = ''
    w = word
    if w.startswith('qo'):
        prefix = 'qo'
        w = w[2:]
    elif w.startswith('q') and len(w) > 1:
        prefix = 'q'
        w = w[1:]
    
    # Replace ii→single token, then standalone i
    w = re.sub(r'i{3,}', lambda m: '\x01' * (len(m.group()) // 2) + ('\x02' if len(m.group()) % 2 else ''), w)
    w = w.replace('ii', '\x01')
    w = w.replace('i', '\x02')
    
    return prefix, w


def decode_word(eva_word):
    """Decode EVA word with full 19-char mapping.
    
    Returns: (hebrew_str, italian_str, fully_decoded, prefix)
    """
    prefix, processed = preprocess_eva(eva_word)
    
    # RTL: reverse the processed word
    chars = list(reversed(processed)) if DIRECTION == 'rtl' else list(processed)
    
    heb_parts = []
    ita_parts = []
    fully = True
    
    for ch in chars:
        if ch == '\x01':  # ii → he
            heb_parts.append(II_HEBREW)
            ita_parts.append(HEBREW_TO_ITALIAN.get(II_HEBREW, '?'))
        elif ch == '\x02':  # standalone i → resh
            heb_parts.append(I_HEBREW)
            ita_parts.append(HEBREW_TO_ITALIAN.get(I_HEBREW, '?'))
        elif ch in MAPPING:
            h = MAPPING[ch]
            heb_parts.append(h)
            ita_parts.append(HEBREW_TO_ITALIAN.get(h, '?'))
        else:
            heb_parts.append('?')
            ita_parts.append('?')
            fully = False
    
    return ''.join(heb_parts), ''.join(ita_parts), fully, prefix


def main():
    # Load data
    eva_data = parse_eva_words(Path('eva_data/LSI_ivtff_0d.txt'))
    
    with open('output/lexicon/italian_lexicon.json') as f:
        lex = json.load(f)
    italian_set = set(lex['all_forms'])
    gloss = lex.get('form_to_gloss', {})
    
    with open('output/lexicon/lexicon.json') as f:
        hlex = json.load(f)
    hebrew_set = set(hlex['all_consonantal_forms'])
    hebrew_by_cons = hlex.get('by_consonants', {})

    # =====================================================================
    # DECODE ENTIRE CORPUS
    # =====================================================================
    print("=" * 70)
    print("  FULL 19-CHAR DECODE")
    print("  16 convergent + f=lamed + ii=he + i=resh + q=prefix")
    print("=" * 70)
    
    total = 0
    fully_decoded = 0
    word_freq = Counter()      # italian_word → count
    heb_word_freq = Counter()  # hebrew_word → count
    eva_to_ita = {}            # eva_word → italian decoded
    eva_to_heb = {}
    section_stats = defaultdict(lambda: {'total': 0, 'fully': 0})
    
    # Per-page decoded text for output
    pages_decoded = []
    
    for page in eva_data['pages']:
        folio = page['folio']
        section = page.get('section', '?')
        sec_name = SECTION_NAMES.get(section, section)
        
        page_words = []
        for w in page['words']:
            total += 1
            section_stats[sec_name]['total'] += 1
            
            heb, ita, ok, pfx = decode_word(w)
            
            display = f"{'e-' if pfx else ''}{ita}" if pfx else ita
            
            word_freq[(ita, pfx)] += 1
            heb_word_freq[heb] += 1
            eva_to_ita[w] = display
            eva_to_heb[w] = heb
            
            if ok:
                fully_decoded += 1
                section_stats[sec_name]['fully'] += 1
            
            page_words.append((w, heb, ita, ok, pfx))
        
        pages_decoded.append((folio, sec_name, page.get('line_words', []), page_words))
    
    pct = fully_decoded / total * 100
    print(f"\n  Total words: {total}")
    print(f"  Fully decoded: {fully_decoded} ({pct:.1f}%)")
    print(f"  Remaining unknowns: {total - fully_decoded}")
    
    # Section breakdown
    print(f"\n  By section:")
    for sec in sorted(section_stats, key=lambda s: -section_stats[s]['total']):
        s = section_stats[sec]
        sp = s['fully'] / s['total'] * 100 if s['total'] else 0
        print(f"    {sec:18s} {s['total']:6d} words, {s['fully']:6d} decoded ({sp:.1f}%)")
    
    # =====================================================================
    # LEXICON MATCHING
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  LEXICON MATCHING")
    print(f"{'=' * 70}")
    
    # Match decoded Italian words against lexicon
    ita_matches = 0
    ita_exact = 0
    heb_matches = 0
    matched_words = []
    
    # Unique decoded words with their total frequency
    word_totals = Counter()
    for (ita, pfx), count in word_freq.items():
        word_totals[ita] += count
    
    for ita_word, count in word_totals.most_common():
        if len(ita_word) < 3:
            continue
        
        # Normalize: remove doubled consonants
        norm = re.sub(r'(.)\1+', r'\1', ita_word)
        
        if ita_word in italian_set:
            ita_matches += count
            ita_exact += 1
            matched_words.append((ita_word, count, ita_word, gloss.get(ita_word, ''), 'exact'))
        elif norm in italian_set:
            ita_matches += count
            matched_words.append((ita_word, count, norm, gloss.get(norm, ''), 'norm'))
    
    # Hebrew matches
    for heb_word, count in heb_word_freq.most_common():
        if len(heb_word) < 3 and heb_word not in hebrew_set:
            continue
        if heb_word in hebrew_set:
            heb_matches += count
    
    print(f"\n  Italian lexicon hits (3+ chars):")
    print(f"    Exact form matches: {ita_exact} unique words")
    print(f"    Total occurrences matched: {ita_matches}")
    print(f"    Hit rate: {ita_matches/total*100:.1f}% of corpus")
    
    print(f"\n  Hebrew lexicon hits (3+ chars):")
    print(f"    Total occurrences matched: {heb_matches}")
    print(f"    Hit rate: {heb_matches/total*100:.1f}% of corpus")
    
    # Top matched words
    print(f"\n  Top 40 Italian lexicon matches:")
    print(f"  {'Italian':15s} {'freq':>5s} {'match':15s} {'type':5s} gloss")
    print(f"  {'-'*65}")
    for ita_word, count, match, g, mtype in sorted(matched_words, key=lambda x: -x[1])[:40]:
        print(f"  {ita_word:15s} {count:5d} {match:15s} {mtype:5s} {g[:40]}")
    
    # =====================================================================
    # TOP DECODED WORDS (ALL, including non-lexicon)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  TOP 50 DECODED WORDS (by frequency)")
    print(f"{'=' * 70}")
    
    print(f"  {'#':3s} {'Italian':12s} {'Hebrew':10s} {'freq':>5s} {'prefix':6s} lexicon")
    print(f"  {'-'*60}")
    
    rank = 0
    for (ita, pfx), count in word_freq.most_common(50):
        rank += 1
        pfx_str = f"qo+" if pfx else ""
        lex_hit = ""
        if ita in italian_set:
            lex_hit = gloss.get(ita, 'yes')[:30]
        elif re.sub(r'(.)\1+', r'\1', ita) in italian_set:
            norm = re.sub(r'(.)\1+', r'\1', ita)
            lex_hit = f"~{norm}: {gloss.get(norm, '')[:25]}"
        print(f"  {rank:3d} {pfx_str}{ita:12s} {count:5d}       {lex_hit}")
    
    # =====================================================================
    # SAMPLE DECODED TEXT (first 3 pages per section)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  SAMPLE DECODED TEXT")
    print(f"{'=' * 70}")
    
    shown_per_section = Counter()
    for folio, sec_name, line_words, page_words in pages_decoded:
        if shown_per_section[sec_name] >= 2:
            continue
        shown_per_section[sec_name] += 1
        
        print(f"\n  --- {folio} [{sec_name}] ---")
        
        # Decode line by line
        word_idx = 0
        for line in line_words:
            if not line:
                continue
            eva_line = ' '.join(line)
            ita_parts = []
            for w in line:
                _, ita, _, pfx = decode_word(w)
                display = f"e-{ita}" if pfx else ita
                ita_parts.append(display)
            print(f"  EVA: {eva_line}")
            print(f"  ITA: {' '.join(ita_parts)}")
            print()
    
    # =====================================================================
    # SAVE FULL DECODED TEXT
    # =====================================================================
    outpath = Path('output/stats/full_decode_19char.txt')
    with open(outpath, 'w') as f:
        for folio, sec_name, line_words, page_words in pages_decoded:
            f.write(f"\n{'='*60}\n  {folio}  [{sec_name}]\n{'='*60}\n")
            for line in line_words:
                if not line:
                    continue
                eva_line = ' '.join(line)
                ita_parts = []
                for w in line:
                    _, ita, _, pfx = decode_word(w)
                    display = f"e-{ita}" if pfx else ita
                    ita_parts.append(display)
                f.write(f"  EVA: {eva_line}\n")
                f.write(f"  ITA: {' '.join(ita_parts)}\n\n")
    print(f"\n  Full decoded text saved to: {outpath}")
    
    # =====================================================================
    # KEY WORD ANALYSIS
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  KEY WORD ANALYSIS — frequent decoded words with meanings")
    print(f"{'=' * 70}")
    
    # Find most common words that match Italian lexicon (4+ chars)
    meaningful = []
    for (ita, pfx), count in word_freq.most_common():
        if len(ita) < 4:
            continue
        norm = re.sub(r'(.)\1+', r'\1', ita)
        g = gloss.get(ita, '') or gloss.get(norm, '')
        if g:
            meaningful.append((ita, count, pfx, g))
    
    print(f"\n  Top 30 meaningful words (4+ chars with lexicon gloss):")
    for ita, count, pfx, g in meaningful[:30]:
        pfx_str = f"qo+" if pfx else "    "
        print(f"    {pfx_str}{ita:15s} x{count:4d}  {g[:50]}")


if __name__ == '__main__':
    main()
