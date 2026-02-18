"""Layout-aware analysis: separate labels, paragraphs, and circular text.

Phase 17: The IVTFF transcription encodes layout type in the unit code:
  - P0, P1... = paragraph (continuous text, ~8 words/line)
  - Lz, L0, Ln, Lf... = label (captions on figures, 85% single word)
  - Cc = circular text (ring around diagrams, ~31 words/line)
  - Pt = title (isolated short line)

Previous analyses mixed all types together. Zodiac (Z) has ZERO paragraph
lines — all label + circular. This explains its low match rate (12%).

This module separates text by layout type and analyzes each independently:
match rates, gloss profiles, word length, and cross-tabulation with
section and hand.
"""

import json
import re
from collections import Counter, defaultdict

import click
import numpy as np
from scipy.stats import chi2

from .config import ToolkitConfig
from .cross_language_baseline import decode_to_hebrew
from .currier_split import two_proportion_ztest
from .full_decode import SECTION_NAMES
from .mapping_audit import load_honest_lexicon
from .utils import print_header, print_step


# =====================================================================
# Layout-aware EVA parser
# =====================================================================

LAYOUT_TYPES = {
    'paragraph': 'P',     # P0, P1, P2...
    'label': 'L',         # Lz, L0, Ln, Lf, Lc, Lt...
    'circular': 'C',      # Cc
    'title': 'T',         # Pt (title within paragraph units)
}


def classify_unit_code(unit_code: str) -> str:
    """Classify IVTFF unit code into layout type.

    Unit code examples: P0, P1, Lz, L0, Ln, Cc, Pt, R1, Ro, Ri, Rb
    """
    # Strip position marks (@, &, *, +, =)
    code = re.sub(r'^[@&*+=]+', '', unit_code)

    if not code:
        return 'paragraph'  # default

    first = code[0]
    if first == 'L':
        return 'label'
    elif code == 'Cc' or first == 'R':
        # Cc = circular, R1/Ro/Ri/Rb = ring (also circular)
        return 'circular'
    elif code == 'Pt':
        return 'title'
    elif first == 'P':
        return 'paragraph'
    else:
        return 'paragraph'  # fallback


def parse_eva_with_layout(filepath, transcriber='H'):
    """Parse IVTFF extracting words WITH layout type per line.

    Returns: {
        'words_by_layout': {layout_type: [words]},
        'pages': list[dict] — each page has words_by_layout,
        'lines': list[dict] — each line with folio, layout, section, hand, words
    }
    """
    text = filepath.read_text(encoding='utf-8', errors='ignore')
    raw_lines = text.split('\n')

    header_re = re.compile(r'^<(f\w+)>\s+<!\s*(.*?)>')
    meta_re = re.compile(r'\$(\w)=(\w+)')
    # Capture the position+unit code part between comma and semicolon
    transcription_re = re.compile(
        r'^<(f\w+)\.(\d+),([^;]*);(\w)>\s+(.+)')

    def extract_words(eva_text):
        clean = re.sub(r'\{[^}]*\}', '', eva_text)
        clean = re.sub(r'<[^>]*>', '', clean)
        clean = re.sub(r'[%!?\[\]*,]', '', clean)
        words = []
        for token in clean.split('.'):
            word = re.sub(r'[^a-z]', '', token)
            if word:
                words.append(word)
        return words

    current_meta = {}
    current_folio = None

    all_lines = []  # list of dicts per line
    pages = []
    page_lines = []

    def flush_page():
        nonlocal current_folio, page_lines
        if current_folio and page_lines:
            # Aggregate words by layout for this page
            wbl = defaultdict(list)
            all_pw = []
            for ln in page_lines:
                wbl[ln['layout']].extend(ln['words'])
                all_pw.extend(ln['words'])
            pages.append({
                'folio': current_folio,
                'section': current_meta.get('I', '?'),
                'language': current_meta.get('L', '?'),
                'hand': current_meta.get('H', '?'),
                'words': all_pw,
                'words_by_layout': dict(wbl),
                'line_words': [ln['words'] for ln in page_lines],
            })
        page_lines = []

    for raw_line in raw_lines:
        raw_line = raw_line.rstrip()
        if not raw_line or raw_line.startswith('#'):
            continue

        m = header_re.match(raw_line)
        if m:
            flush_page()
            current_folio = m.group(1)
            current_meta = dict(meta_re.findall(m.group(2)))
            continue

        m = transcription_re.match(raw_line)
        if not m:
            continue

        folio_id = m.group(1)
        line_num = m.group(2)
        pos_unit = m.group(3)  # e.g. "@Lz", "+P0", "@Cc"
        tr = m.group(4)
        eva_text = m.group(5)

        if tr != transcriber:
            continue

        layout = classify_unit_code(pos_unit)
        words = extract_words(eva_text)

        line_data = {
            'folio': folio_id,
            'line_num': int(line_num),
            'layout': layout,
            'unit_code': pos_unit,
            'section': current_meta.get('I', '?'),
            'hand': current_meta.get('H', '?'),
            'language': current_meta.get('L', '?'),
            'words': words,
            'n_words': len(words),
        }
        all_lines.append(line_data)
        page_lines.append(line_data)

    flush_page()

    # Aggregate by layout
    words_by_layout = defaultdict(list)
    for ln in all_lines:
        words_by_layout[ln['layout']].extend(ln['words'])

    return {
        'words_by_layout': dict(words_by_layout),
        'pages': pages,
        'lines': all_lines,
    }


# =====================================================================
# Analysis 1: Match rates per layout type
# =====================================================================

def layout_match_rates(lines, honest_lex, full_lex):
    """Match rate per layout type (label/paragraph/circular/title)."""
    by_layout = defaultdict(list)
    for ln in lines:
        by_layout[ln['layout']].extend(ln['words'])

    results = {}
    for layout, words in sorted(by_layout.items()):
        n_total = len(words)
        if n_total < 5:
            continue

        honest_matched = 0
        full_matched = 0
        decoded_words = []
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= 2:
                decoded_words.append(heb)
                if heb in honest_lex:
                    honest_matched += 1
                if heb in full_lex:
                    full_matched += 1

        n_decoded = len(decoded_words)
        results[layout] = {
            'n_words': n_total,
            'n_decoded': n_decoded,
            'honest_matched': honest_matched,
            'honest_rate': round(honest_matched / n_decoded, 4) if n_decoded else 0,
            'full_matched': full_matched,
            'full_rate': round(full_matched / n_decoded, 4) if n_decoded else 0,
        }

    return results


# =====================================================================
# Analysis 2: Cross-tabulation layout × section
# =====================================================================

def layout_section_cross(lines, honest_lex):
    """Match rate cross-tabulated: layout × section."""
    cells = defaultdict(list)
    for ln in lines:
        key = (ln['layout'], ln['section'])
        cells[key].extend(ln['words'])

    results = {}
    for (layout, section), words in sorted(cells.items()):
        if len(words) < 10:
            continue
        matched = 0
        n_decoded = 0
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= 2:
                n_decoded += 1
                if heb in honest_lex:
                    matched += 1

        results[f'{layout}:{section}'] = {
            'layout': layout,
            'section': section,
            'section_name': SECTION_NAMES.get(section, section),
            'n_words': len(words),
            'n_decoded': n_decoded,
            'n_matched': matched,
            'match_rate': round(matched / n_decoded, 4) if n_decoded else 0,
        }

    return results


# =====================================================================
# Analysis 3: Cross-tabulation layout × hand
# =====================================================================

def layout_hand_cross(lines, honest_lex):
    """Match rate cross-tabulated: layout × hand."""
    cells = defaultdict(list)
    for ln in lines:
        key = (ln['layout'], ln['hand'])
        cells[key].extend(ln['words'])

    results = {}
    for (layout, hand), words in sorted(cells.items()):
        if len(words) < 10:
            continue
        matched = 0
        n_decoded = 0
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= 2:
                n_decoded += 1
                if heb in honest_lex:
                    matched += 1

        results[f'{layout}:H{hand}'] = {
            'layout': layout,
            'hand': hand,
            'n_words': len(words),
            'n_decoded': n_decoded,
            'n_matched': matched,
            'match_rate': round(matched / n_decoded, 4) if n_decoded else 0,
        }

    return results


# =====================================================================
# Analysis 4: Label gloss analysis
# =====================================================================

def label_gloss_analysis(lines, honest_lex, full_lex, form_to_gloss):
    """Detailed analysis of label words: glosses, word length, section."""
    label_words = []
    for ln in lines:
        if ln['layout'] == 'label':
            for w in ln['words']:
                label_words.append({
                    'eva': w,
                    'section': ln['section'],
                    'hand': ln['hand'],
                    'folio': ln['folio'],
                })

    # Decode and gloss each label word
    decoded_labels = []
    for lw in label_words:
        heb = decode_to_hebrew(lw['eva'])
        if not heb or len(heb) < 2:
            continue
        gloss = form_to_gloss.get(heb, '')
        in_honest = heb in honest_lex
        in_full = heb in full_lex
        decoded_labels.append({
            'eva': lw['eva'],
            'hebrew': heb,
            'gloss': gloss,
            'in_honest': in_honest,
            'in_full': in_full,
            'section': lw['section'],
            'hand': lw['hand'],
            'folio': lw['folio'],
            'heb_len': len(heb),
        })

    # Summary stats
    n_total = len(decoded_labels)
    n_honest = sum(1 for d in decoded_labels if d['in_honest'])
    n_full = sum(1 for d in decoded_labels if d['in_full'])
    n_glossed = sum(1 for d in decoded_labels if d['gloss'])

    # Word length distribution
    lengths = [d['heb_len'] for d in decoded_labels]
    len_dist = dict(Counter(lengths).most_common())

    # Top glossed labels
    gloss_freq = Counter()
    gloss_map = {}
    for d in decoded_labels:
        if d['gloss']:
            gloss_freq[d['hebrew']] += 1
            gloss_map[d['hebrew']] = d['gloss']

    top_glossed = [
        {'hebrew': heb, 'count': c, 'gloss': gloss_map[heb]}
        for heb, c in gloss_freq.most_common(20)
    ]

    # Per-section breakdown of labels
    by_section = defaultdict(lambda: {'total': 0, 'matched': 0})
    for d in decoded_labels:
        sec = d['section']
        by_section[sec]['total'] += 1
        if d['in_honest']:
            by_section[sec]['matched'] += 1

    section_rates = {
        sec: {
            'n_labels': s['total'],
            'n_matched': s['matched'],
            'match_rate': round(s['matched'] / s['total'], 4) if s['total'] else 0,
        }
        for sec, s in sorted(by_section.items())
    }

    return {
        'n_decoded_labels': n_total,
        'n_honest_matched': n_honest,
        'honest_rate': round(n_honest / n_total, 4) if n_total else 0,
        'n_full_matched': n_full,
        'full_rate': round(n_full / n_total, 4) if n_total else 0,
        'n_glossed': n_glossed,
        'avg_hebrew_length': round(np.mean(lengths), 2) if lengths else 0,
        'length_distribution': len_dist,
        'top_glossed_labels': top_glossed,
        'section_rates': section_rates,
        'all_labels': decoded_labels,
    }


# =====================================================================
# Analysis 5: Paragraph-only section analysis (decontaminated)
# =====================================================================

def paragraph_only_section(lines, honest_lex):
    """Match rate per section using ONLY paragraph lines.

    This decontaminates the section analysis from label/circular mixing.
    """
    by_section = defaultdict(list)
    for ln in lines:
        if ln['layout'] == 'paragraph':
            by_section[ln['section']].extend(ln['words'])

    results = {}
    for sec, words in sorted(by_section.items()):
        if len(words) < 10:
            continue
        matched = 0
        n_decoded = 0
        for w in words:
            heb = decode_to_hebrew(w)
            if heb and len(heb) >= 2:
                n_decoded += 1
                if heb in honest_lex:
                    matched += 1

        results[sec] = {
            'section_name': SECTION_NAMES.get(sec, sec),
            'n_words': len(words),
            'n_decoded': n_decoded,
            'n_matched': matched,
            'match_rate': round(matched / n_decoded, 4) if n_decoded else 0,
        }

    return results


# =====================================================================
# Analysis 6: Pairwise comparisons (label vs paragraph)
# =====================================================================

def layout_pairwise(layout_rates):
    """Z-test comparing label vs paragraph match rates."""
    comparisons = {}

    if 'label' in layout_rates and 'paragraph' in layout_rates:
        lab = layout_rates['label']
        par = layout_rates['paragraph']
        if lab['n_decoded'] >= 10 and par['n_decoded'] >= 10:
            z = two_proportion_ztest(
                lab['honest_matched'], lab['n_decoded'],
                par['honest_matched'], par['n_decoded'])
            comparisons['label_vs_paragraph'] = {
                'label_rate': lab['honest_rate'],
                'paragraph_rate': par['honest_rate'],
                **z,
            }

    if 'circular' in layout_rates and 'paragraph' in layout_rates:
        cir = layout_rates['circular']
        par = layout_rates['paragraph']
        if cir['n_decoded'] >= 10 and par['n_decoded'] >= 10:
            z = two_proportion_ztest(
                cir['honest_matched'], cir['n_decoded'],
                par['honest_matched'], par['n_decoded'])
            comparisons['circular_vs_paragraph'] = {
                'circular_rate': cir['honest_rate'],
                'paragraph_rate': par['honest_rate'],
                **z,
            }

    return comparisons


# =====================================================================
# Summary formatter
# =====================================================================

def format_summary(layout_rates, cross_section, cross_hand,
                   label_analysis, para_section, pairwise):
    """Format human-readable summary."""
    lines = []
    lines.append('=' * 70)
    lines.append('  LAYOUT-AWARE ANALYSIS — Phase 17')
    lines.append('=' * 70)

    # 1. Match rates by layout type
    lines.append('\n  1. MATCH RATES BY LAYOUT TYPE')
    lines.append(f"  {'Type':>12s} {'Words':>7s} {'Decoded':>7s} "
                 f"{'Honest%':>8s} {'Full%':>7s}")
    lines.append('  ' + '-' * 50)
    for lt in ['paragraph', 'label', 'circular', 'title']:
        if lt not in layout_rates:
            continue
        r = layout_rates[lt]
        lines.append(
            f"  {lt:>12s} {r['n_words']:7d} {r['n_decoded']:7d} "
            f"{r['honest_rate']*100:7.1f}% {r['full_rate']*100:6.1f}%")

    # 2. Pairwise comparisons
    if pairwise:
        lines.append('\n  2. PAIRWISE COMPARISONS (honest lexicon)')
        for name, p in pairwise.items():
            sig = '***' if p.get('p_value', 1) < 0.001 else \
                  '**' if p.get('p_value', 1) < 0.01 else \
                  '*' if p.get('p_value', 1) < 0.05 else 'ns'
            lines.append(
                f"  {name}: diff={p['diff']*100:+.1f}pp, "
                f"z={p['z_score']:.2f}, p={p['p_value']:.6f} {sig}")

    # 3. Layout × section cross-tab
    lines.append('\n  3. LAYOUT × SECTION (honest, ≥10 words)')
    lines.append(f"  {'Layout':>12s} {'Sec':>4s} {'Name':>14s} "
                 f"{'Words':>6s} {'Rate%':>7s}")
    lines.append('  ' + '-' * 55)
    for key in sorted(cross_section,
                      key=lambda k: -cross_section[k]['match_rate']):
        c = cross_section[key]
        lines.append(
            f"  {c['layout']:>12s} {c['section']:>4s} "
            f"{c['section_name']:>14s} {c['n_decoded']:6d} "
            f"{c['match_rate']*100:6.1f}%")

    # 4. Paragraph-only sections (decontaminated)
    if para_section:
        lines.append('\n  4. PARAGRAPH-ONLY SECTION RATES (decontaminated)')
        lines.append(f"  {'Sec':>4s} {'Name':>14s} {'Words':>7s} "
                     f"{'Rate%':>7s}")
        lines.append('  ' + '-' * 40)
        for sec in sorted(para_section,
                          key=lambda s: -para_section[s]['match_rate']):
            s = para_section[sec]
            lines.append(
                f"  {sec:>4s} {s['section_name']:>14s} "
                f"{s['n_decoded']:7d} {s['match_rate']*100:6.1f}%")

    # 5. Label analysis highlights
    if label_analysis:
        la = label_analysis
        lines.append(f"\n  5. LABEL ANALYSIS")
        lines.append(f"  Total decoded labels: {la['n_decoded_labels']}")
        lines.append(f"  Honest match rate: {la['honest_rate']*100:.1f}%")
        lines.append(f"  Full match rate: {la['full_rate']*100:.1f}%")
        lines.append(f"  Avg Hebrew length: {la['avg_hebrew_length']:.1f} chars")
        lines.append(f"  Glossed labels: {la['n_glossed']}")

        if la.get('top_glossed_labels'):
            lines.append(f"\n  TOP GLOSSED LABELS:")
            for g in la['top_glossed_labels'][:15]:
                lines.append(
                    f"    {g['hebrew']:>8s} (n={g['count']:2d}) — "
                    f"{g['gloss'][:40]}")

        if la.get('section_rates'):
            lines.append(f"\n  LABELS PER SECTION (honest):")
            for sec, sr in sorted(la['section_rates'].items(),
                                  key=lambda x: -x[1]['match_rate']):
                lines.append(
                    f"    {sec}: {sr['n_labels']:4d} labels, "
                    f"{sr['match_rate']*100:.1f}% match")

    # 6. Layout × hand
    if cross_hand:
        lines.append(f"\n  6. LAYOUT × HAND (honest, ≥10 words)")
        lines.append(f"  {'Layout':>12s} {'Hand':>5s} {'Words':>6s} "
                     f"{'Rate%':>7s}")
        lines.append('  ' + '-' * 40)
        for key in sorted(cross_hand,
                          key=lambda k: -cross_hand[k]['match_rate']):
            c = cross_hand[key]
            lines.append(
                f"  {c['layout']:>12s} {'H'+c['hand']:>5s} "
                f"{c['n_decoded']:6d} {c['match_rate']*100:6.1f}%")

    # Verdict
    lines.append(f"\n  {'='*60}")
    if 'label' in layout_rates and 'paragraph' in layout_rates:
        lab_r = layout_rates['label']['honest_rate']
        par_r = layout_rates['paragraph']['honest_rate']
        diff = lab_r - par_r
        if abs(diff) > 0.03:
            direction = 'higher' if diff > 0 else 'lower'
            lines.append(
                f"  VERDICT: Labels have {direction} match rate than "
                f"paragraphs ({lab_r*100:.1f}% vs {par_r*100:.1f}%)")
        else:
            lines.append(
                f"  VERDICT: Labels and paragraphs have similar rates "
                f"({lab_r*100:.1f}% vs {par_r*100:.1f}%)")
    lines.append(f"  {'='*60}")

    return '\n'.join(lines) + '\n'


# =====================================================================
# DB export
# =====================================================================

def export_to_db(config, report):
    """Export layout analysis results to SQLite DB."""
    db_path = config.output_dir.parent / 'voynich.db'
    if not db_path.exists():
        return

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute('DROP TABLE IF EXISTS layout_analysis')
    cur.execute('''CREATE TABLE layout_analysis (
        category TEXT,
        key TEXT,
        value REAL,
        detail TEXT,
        PRIMARY KEY (category, key)
    )''')

    # Layout match rates
    for lt, r in report.get('layout_rates', {}).items():
        cur.execute(
            'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
            ('layout_rate', lt,
             r.get('honest_rate', 0),
             json.dumps({k: v for k, v in r.items()
                         if k != 'honest_rate'})))

    # Cross-section
    for key, c in report.get('cross_section', {}).items():
        cur.execute(
            'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
            ('layout_section', key,
             c.get('match_rate', 0),
             json.dumps({k: v for k, v in c.items()
                         if k != 'match_rate'})))

    # Cross-hand
    for key, c in report.get('cross_hand', {}).items():
        cur.execute(
            'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
            ('layout_hand', key,
             c.get('match_rate', 0),
             json.dumps({k: v for k, v in c.items()
                         if k != 'match_rate'})))

    # Pairwise
    for name, p in report.get('pairwise', {}).items():
        cur.execute(
            'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
            ('pairwise', name,
             p.get('z_score', 0),
             json.dumps(p)))

    # Label analysis summary
    la = report.get('label_analysis', {})
    if la:
        for key in ['n_decoded_labels', 'honest_rate', 'full_rate',
                     'avg_hebrew_length', 'n_glossed']:
            cur.execute(
                'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
                ('label_summary', key, la.get(key, 0), ''))

    # Paragraph-only sections
    for sec, s in report.get('para_section', {}).items():
        cur.execute(
            'INSERT INTO layout_analysis VALUES (?, ?, ?, ?)',
            ('para_section', sec,
             s.get('match_rate', 0),
             json.dumps(s)))

    conn.commit()
    conn.close()


# =====================================================================
# Entry point
# =====================================================================

def run(config: ToolkitConfig, force=False, **kwargs):
    """Layout-aware analysis: labels vs paragraphs vs circular text."""
    report_path = config.stats_dir / 'layout_analysis.json'
    summary_path = config.stats_dir / 'layout_analysis_summary.txt'

    if report_path.exists() and not force:
        click.echo('  Layout analysis report exists. Use --force to re-run.')
        return

    config.ensure_dirs()
    print_header('LAYOUT-AWARE ANALYSIS')

    # 1. Parse EVA with layout info
    print_step('Parsing EVA with layout types...')
    eva_file = config.eva_data_dir / 'LSI_ivtff_0d.txt'
    if not eva_file.exists():
        raise click.ClickException(f'EVA file not found: {eva_file}')
    eva_data = parse_eva_with_layout(eva_file)

    n_lines = len(eva_data['lines'])
    for lt, words in sorted(eva_data['words_by_layout'].items()):
        click.echo(f'    {lt}: {len(words)} words')
    click.echo(f'    Total lines: {n_lines}, pages: {len(eva_data["pages"])}')

    # 2. Load lexicons
    print_step('Loading lexicons...')
    enriched_path = config.lexicon_dir / 'lexicon_enriched.json'
    if not enriched_path.exists():
        raise click.ClickException(
            'Enriched lexicon not found. Run: voynich enrich-lexicon')
    with open(enriched_path) as f:
        lex_data = json.load(f)
    full_lex = set(lex_data['all_consonantal_forms'])
    form_to_gloss = lex_data.get('form_to_gloss', {})
    click.echo(f'    Full: {len(full_lex):,} forms')

    honest_lex, _ = load_honest_lexicon(config)
    click.echo(f'    Honest: {len(honest_lex):,} forms')

    all_lines = eva_data['lines']

    # 3. Match rates per layout
    print_step('Computing match rates per layout type...')
    layout_rates = layout_match_rates(all_lines, honest_lex, full_lex)
    for lt in ['paragraph', 'label', 'circular', 'title']:
        if lt in layout_rates:
            r = layout_rates[lt]
            click.echo(
                f'    {lt}: {r["n_decoded"]} decoded, '
                f'honest={r["honest_rate"]*100:.1f}%, '
                f'full={r["full_rate"]*100:.1f}%')

    # 4. Pairwise comparisons
    print_step('Pairwise z-tests...')
    pairwise = layout_pairwise(layout_rates)
    for name, p in pairwise.items():
        sig = '***' if p['p_value'] < 0.001 else \
              '**' if p['p_value'] < 0.01 else \
              '*' if p['p_value'] < 0.05 else 'ns'
        click.echo(
            f'    {name}: diff={p["diff"]*100:+.1f}pp, '
            f'z={p["z_score"]:.2f}, p={p["p_value"]:.6f} {sig}')

    # 5. Cross-tabulation layout × section
    print_step('Cross-tabulation layout × section...')
    cross_section = layout_section_cross(all_lines, honest_lex)
    for key in sorted(cross_section,
                      key=lambda k: -cross_section[k]['match_rate'])[:10]:
        c = cross_section[key]
        click.echo(
            f'    {c["layout"]:>10s}:{c["section"]} '
            f'({c["section_name"]}): {c["match_rate"]*100:.1f}% '
            f'[{c["n_decoded"]} words]')

    # 6. Cross-tabulation layout × hand
    print_step('Cross-tabulation layout × hand...')
    cross_hand = layout_hand_cross(all_lines, honest_lex)
    for key in sorted(cross_hand,
                      key=lambda k: -cross_hand[k]['match_rate'])[:10]:
        c = cross_hand[key]
        click.echo(
            f'    {c["layout"]:>10s}:H{c["hand"]} '
            f'{c["match_rate"]*100:.1f}% [{c["n_decoded"]} words]')

    # 7. Label gloss analysis
    print_step('Analyzing label words...')
    label_analysis = label_gloss_analysis(
        all_lines, honest_lex, full_lex, form_to_gloss)
    click.echo(
        f'    {label_analysis["n_decoded_labels"]} labels decoded, '
        f'{label_analysis["honest_rate"]*100:.1f}% honest match, '
        f'{label_analysis["n_glossed"]} glossed')
    if label_analysis['top_glossed_labels']:
        top3 = label_analysis['top_glossed_labels'][:3]
        for g in top3:
            click.echo(
                f'      {g["hebrew"]} (n={g["count"]}) — {g["gloss"][:30]}')

    # 8. Paragraph-only section rates
    print_step('Paragraph-only section analysis...')
    para_section = paragraph_only_section(all_lines, honest_lex)
    for sec in sorted(para_section,
                      key=lambda s: -para_section[s]['match_rate']):
        s = para_section[sec]
        click.echo(
            f'    {sec} ({s["section_name"]}): {s["match_rate"]*100:.1f}% '
            f'[{s["n_decoded"]} para words]')

    # 9. Save JSON (strip all_labels for size)
    print_step('Saving reports...')
    label_for_json = {k: v for k, v in label_analysis.items()
                      if k != 'all_labels'}
    report = {
        'layout_rates': layout_rates,
        'pairwise': pairwise,
        'cross_section': cross_section,
        'cross_hand': cross_hand,
        'label_analysis': label_for_json,
        'para_section': para_section,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    click.echo(f'    JSON: {report_path}')

    # 10. Save TXT summary
    summary = format_summary(layout_rates, cross_section, cross_hand,
                             label_analysis, para_section, pairwise)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    click.echo(f'    TXT: {summary_path}')

    # 11. Export to DB
    print_step('Exporting to SQLite...')
    try:
        export_to_db(config, report)
        click.echo('    Done.')
    except Exception as e:
        click.echo(f'    DB export failed: {e}')

    click.echo(f'\n{summary}')
