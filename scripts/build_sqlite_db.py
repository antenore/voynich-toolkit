#!/usr/bin/env python3
"""
Build a SQLite database from Voynich Toolkit output files.

Usage:
    python scripts/build_sqlite_db.py [--output voynich.db]

Imports: glossed_words, anchor_matches, phrase_completions,
         contextual_phrases, cipher_mapping, lexicon entries,
         validation scorecard, letter audit.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "output" / "stats"
LEXICON = ROOT / "output" / "lexicon"


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def create_tables(cur: sqlite3.Cursor) -> None:
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS glossed_words (
            consonantal TEXT PRIMARY KEY,
            hebrew TEXT,
            freq INTEGER,
            gloss TEXT,
            source TEXT,
            length INTEGER
        );

        CREATE TABLE IF NOT EXISTS anchor_matches (
            anchor TEXT,
            gloss TEXT,
            language TEXT,
            category TEXT,
            category_label TEXT,
            total_occurrences INTEGER,
            best_distance INTEGER,
            n_decoded_forms INTEGER
        );

        CREATE TABLE IF NOT EXISTS phrase_resolutions (
            word TEXT PRIMARY KEY,
            tier INTEGER,
            prefix TEXT,
            prefix_meaning TEXT,
            suffix TEXT,
            suffix_meaning TEXT,
            stem TEXT,
            stem_gloss TEXT,
            stem_in_gloss INTEGER,
            sefaria_freq INTEGER
        );

        CREATE TABLE IF NOT EXISTS contextual_phrases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page TEXT,
            section TEXT,
            target_word TEXT,
            target_pos INTEGER,
            context_heb TEXT,
            context_eva TEXT,
            context_glossed TEXT,
            n_known INTEGER,
            n_total INTEGER,
            known_ratio REAL
        );

        CREATE TABLE IF NOT EXISTS cipher_mapping (
            eva TEXT PRIMARY KEY,
            hebrew TEXT,
            hebrew_name TEXT,
            italian TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS lexicon (
            consonants TEXT,
            hebrew TEXT,
            gloss TEXT,
            source TEXT,
            domain TEXT
        );

        CREATE TABLE IF NOT EXISTS validation (
            metric TEXT PRIMARY KEY,
            value TEXT,
            target TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS letter_audit (
            eva TEXT PRIMARY KEY,
            current_hebrew TEXT,
            current_name TEXT,
            current_tokens INTEGER,
            current_rank INTEGER,
            best_hebrew TEXT,
            best_tokens INTEGER,
            gap_tokens INTEGER,
            gap_pct REAL
        );

        CREATE TABLE IF NOT EXISTS semantic_passages (
            rank INTEGER PRIMARY KEY,
            folio TEXT,
            section TEXT,
            start_line INTEGER,
            n_lines INTEGER,
            n_words INTEGER,
            n_semantic INTEGER,
            glossed_ratio REAL,
            max_consecutive INTEGER,
            lex_diversity REAL,
            score REAL,
            words_json TEXT
        );

        CREATE TABLE IF NOT EXISTS semantic_stats (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS compound_splits (
            word TEXT NOT NULL,
            rank INTEGER NOT NULL,
            left_part TEXT,
            right_part TEXT,
            left_gloss TEXT,
            right_gloss TEXT,
            score REAL,
            PRIMARY KEY (word, rank)
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_glossed_freq ON glossed_words(freq DESC);
        CREATE INDEX IF NOT EXISTS idx_glossed_source ON glossed_words(source);
        CREATE INDEX IF NOT EXISTS idx_anchor_dist ON anchor_matches(best_distance);
        CREATE INDEX IF NOT EXISTS idx_anchor_lang ON anchor_matches(language);
        CREATE INDEX IF NOT EXISTS idx_anchor_cat ON anchor_matches(category);
        CREATE INDEX IF NOT EXISTS idx_phrase_tier ON phrase_resolutions(tier);
        CREATE INDEX IF NOT EXISTS idx_phrase_stem ON phrase_resolutions(stem);
        CREATE INDEX IF NOT EXISTS idx_ctx_page ON contextual_phrases(page);
        CREATE INDEX IF NOT EXISTS idx_ctx_section ON contextual_phrases(section);
        CREATE INDEX IF NOT EXISTS idx_ctx_ratio ON contextual_phrases(known_ratio DESC);
        CREATE INDEX IF NOT EXISTS idx_ctx_target ON contextual_phrases(target_word);
        CREATE INDEX IF NOT EXISTS idx_lex_consonants ON lexicon(consonants);
        CREATE INDEX IF NOT EXISTS idx_lex_source ON lexicon(source);
        CREATE INDEX IF NOT EXISTS idx_lex_domain ON lexicon(domain);
        CREATE INDEX IF NOT EXISTS idx_sem_folio ON semantic_passages(folio);
        CREATE INDEX IF NOT EXISTS idx_sem_section ON semantic_passages(section);
        CREATE INDEX IF NOT EXISTS idx_sem_score ON semantic_passages(score DESC);
        CREATE INDEX IF NOT EXISTS idx_cs_left ON compound_splits(left_part);
        CREATE INDEX IF NOT EXISTS idx_cs_right ON compound_splits(right_part);
        CREATE INDEX IF NOT EXISTS idx_cs_score ON compound_splits(score DESC);

        CREATE TABLE IF NOT EXISTS anchor_confidence (
            anchor TEXT PRIMARY KEY,
            score REAL,
            grade TEXT,
            length_score REAL,
            distance_score REAL,
            domain_score REAL,
            freq_score REAL,
            context_score REAL
        );

        CREATE TABLE IF NOT EXISTS reconstructed_words (
            hebrew_type TEXT PRIMARY KEY,
            candidate TEXT,
            candidate_gloss TEXT,
            distance INTEGER,
            semantic_score REAL,
            confidence REAL,
            source_passage TEXT,
            iteration INTEGER
        );

        CREATE TABLE IF NOT EXISTS reconstruction_log (
            iteration INTEGER,
            metric TEXT,
            value REAL,
            PRIMARY KEY (iteration, metric)
        );

        CREATE TABLE IF NOT EXISTS currier_split_stats (
            language TEXT NOT NULL,
            metric TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (language, metric)
        );

        CREATE TABLE IF NOT EXISTS currier_section_stats (
            language TEXT NOT NULL,
            section TEXT NOT NULL,
            n_pages INTEGER,
            n_words INTEGER,
            n_matched INTEGER,
            match_rate REAL,
            PRIMARY KEY (language, section)
        );

        CREATE TABLE IF NOT EXISTS naibbe_test (
            metric TEXT PRIMARY KEY,
            value TEXT,
            verdict TEXT
        );

        CREATE TABLE IF NOT EXISTS judeo_italian_test (
            metric TEXT PRIMARY KEY,
            value TEXT,
            detail TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ac_grade ON anchor_confidence(grade);
        CREATE INDEX IF NOT EXISTS idx_rw_conf ON reconstructed_words(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_rw_iter ON reconstructed_words(iteration);

        CREATE TABLE IF NOT EXISTS hand1_dive (
            metric TEXT PRIMARY KEY,
            value TEXT,
            detail TEXT
        );

        CREATE TABLE IF NOT EXISTS null_model_test (
            metric TEXT PRIMARY KEY,
            value TEXT,
            detail TEXT
        );

        CREATE TABLE IF NOT EXISTS section_entropy (
            metric TEXT PRIMARY KEY,
            value TEXT,
            detail TEXT
        );
    """)


def import_glossed_words(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "glossed_words.json")
    if not data:
        return 0
    cur.executemany(
        "INSERT OR REPLACE INTO glossed_words VALUES (?,?,?,?,?,?)",
        [(r["consonantal"], r.get("hebrew", ""), r["freq"],
          r.get("gloss", ""), r.get("source", ""), r.get("length", 0))
         for r in data],
    )
    return len(data)


def import_anchor_matches(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "anchor_words_report.json")
    if not data:
        return 0
    matches = data.get("top_matches", [])
    cur.executemany(
        "INSERT INTO anchor_matches VALUES (?,?,?,?,?,?,?,?)",
        [(r["anchor"], r.get("gloss", ""), r.get("language", ""),
          r.get("category", ""), r.get("category_label", ""),
          r.get("total_occurrences", 0), r.get("best_distance", 99),
          r.get("n_decoded_forms", 0))
         for r in matches],
    )
    return len(matches)


def _normalize_resolution(word: str, r: dict) -> tuple:
    """Normalize tier 2/3/4 resolution to a flat row for phrase_resolutions."""
    tier = r["tier"]
    prefix = r.get("prefix", "")
    prefix_meaning = r.get("prefix_meaning", "")
    suffix = r.get("suffix", "")
    suffix_meaning = r.get("suffix_meaning", "")
    sefaria_freq = r.get("sefaria_freq", 0)

    if tier == 3:
        splits = r.get("splits", [])
        if splits:
            best = splits[0]
            stem = f"{best['left']}+{best['right']}"
            left_g = best.get("left_gloss", "")
            right_g = best.get("right_gloss", "")
            parts = [g for g in (left_g, right_g) if g]
            stem_gloss = " + ".join(parts) if parts else ""
            stem_in_gloss = 1 if (left_g and right_g) else 0
        else:
            stem, stem_gloss, stem_in_gloss = "", "", 0
    elif tier == 4:
        matches = r.get("matches", [])
        if matches:
            best = matches[0]
            stem = best.get("target", "")
            stem_gloss = best.get("gloss", "")
            stem_in_gloss = 1 if stem_gloss else 0
        else:
            stem, stem_gloss, stem_in_gloss = "", "", 0
    else:
        stem = r.get("stem", "")
        stem_gloss = r.get("stem_gloss", "")
        stem_in_gloss = 1 if r.get("stem_in_gloss") else 0

    return (word, tier, prefix, prefix_meaning, suffix, suffix_meaning,
            stem, stem_gloss, stem_in_gloss, sefaria_freq)


def import_phrase_resolutions(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "phrase_completion.json")
    if not data:
        return 0
    resolutions = data.get("resolutions", {})
    cur.executemany(
        "INSERT OR REPLACE INTO phrase_resolutions VALUES (?,?,?,?,?,?,?,?,?,?)",
        [_normalize_resolution(word, r) for word, r in resolutions.items()],
    )
    return len(resolutions)


def import_compound_splits(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "phrase_completion.json")
    if not data:
        return 0
    resolutions = data.get("resolutions", {})
    rows = []
    for word, r in resolutions.items():
        if r["tier"] != 3:
            continue
        for rank, sp in enumerate(r.get("splits", [])):
            rows.append((
                word, rank,
                sp.get("left", ""), sp.get("right", ""),
                sp.get("left_gloss", ""), sp.get("right_gloss", ""),
                sp.get("score", 0.0),
            ))
    cur.executemany(
        "INSERT INTO compound_splits VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def import_contextual_phrases(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "contextual_phrases.json")
    if not data:
        return 0
    cur.executemany(
        """INSERT INTO contextual_phrases
           (page, section, target_word, target_pos, context_heb,
            context_eva, context_glossed, n_known, n_total, known_ratio)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(r["page"], r.get("section", ""),
          r["target_word"], r.get("target_pos", 0),
          json.dumps(r.get("context_heb", []), ensure_ascii=False),
          json.dumps(r.get("context_eva", []), ensure_ascii=False),
          json.dumps(r.get("context_glossed", []), ensure_ascii=False),
          r.get("n_known_in_context", 0), r.get("n_total_in_context", 0),
          r.get("known_ratio", 0.0))
         for r in data],
    )
    return len(data)


def import_cipher_mapping(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "full_decode.json")
    if not data:
        return 0
    mapping = data.get("mapping", {})
    cur.executemany(
        "INSERT OR REPLACE INTO cipher_mapping VALUES (?,?,?,?,?)",
        [(eva, m.get("hebrew", ""), m.get("hebrew_name", ""),
          m.get("italian", ""), m.get("note", ""))
         for eva, m in mapping.items()],
    )
    return len(mapping)


def import_lexicon(cur: sqlite3.Cursor) -> int:
    data = load_json(LEXICON / "lexicon_enriched.json")
    if not data:
        return 0
    count = 0
    by_domain = data.get("by_domain", data)
    rows = []
    for domain, entries in by_domain.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            rows.append((
                entry.get("consonants", ""),
                entry.get("hebrew", ""),
                entry.get("gloss", ""),
                entry.get("source", ""),
                domain,
            ))
    # Batch insert
    cur.executemany("INSERT INTO lexicon VALUES (?,?,?,?,?)", rows)
    count = len(rows)
    return count


def import_validation(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "validation_summary.json")
    if not data:
        return 0
    scorecard = data.get("scorecard", {})
    count = 0
    for key, entry in scorecard.items():
        if isinstance(entry, dict):
            cur.execute(
                "INSERT OR REPLACE INTO validation VALUES (?,?,?,?)",
                (entry.get("metric", key),
                 str(entry.get("value", "")),
                 str(entry.get("target", "")),
                 entry.get("status", "")),
            )
            count += 1
    return count


def import_letter_audit(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "mapping_audit.json")
    if not data:
        return 0
    audits = data.get("letter_audit", [])
    cur.executemany(
        "INSERT OR REPLACE INTO letter_audit VALUES (?,?,?,?,?,?,?,?,?)",
        [(r["eva"], r.get("current", ""), r.get("current_name", ""),
          r.get("current_tokens", 0), r.get("current_rank", 0),
          r.get("best", ""), r.get("best_tokens", 0),
          r.get("gap_tokens", 0), r.get("gap_pct", 0.0))
         for r in audits],
    )
    return len(audits)


def import_semantic_coherence(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "semantic_coherence.json")
    if not data:
        return 0

    # Import top passages
    passages = data.get("top_passages", [])
    cur.executemany(
        "INSERT OR REPLACE INTO semantic_passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(i + 1, p["folio"], p.get("section", ""), p.get("start_line", 0),
          p.get("n_lines", 1), p.get("n_words", 0), p.get("n_semantic", 0),
          p.get("glossed_ratio", 0.0), p.get("max_consecutive", 0),
          p.get("lex_diversity", 0.0), p.get("score", 0.0),
          json.dumps(p.get("words", []), ensure_ascii=False))
         for i, p in enumerate(passages)],
    )

    # Import stats as key-value pairs
    stats_rows = []

    # Annotation stats
    ann = data.get("annotation_stats", {})
    for lv, count in ann.get("by_level", {}).items():
        stats_rows.append((f"level_{lv}_tokens", str(count)))
    for key in ("total_tokens", "semantic_tokens", "semantic_ratio",
                "n_types", "n_semantic_types"):
        if key in ann:
            stats_rows.append((key, str(ann[key])))

    # Permutation test results
    perm = data.get("permutation_test", {})
    for metric_name in ("max_consecutive", "n_high_lines", "mean_glossed_ratio"):
        m = perm.get(metric_name, {})
        if isinstance(m, dict):
            for k, v in m.items():
                stats_rows.append((f"perm_{metric_name}_{k}", str(v)))

    # Section breakdown
    for sec, s in data.get("section_breakdown", {}).items():
        stats_rows.append((f"section_{sec}_ratio", str(s.get("semantic_ratio", 0))))
        stats_rows.append((f"section_{sec}_tokens", str(s.get("total_tokens", 0))))

    cur.executemany(
        "INSERT OR REPLACE INTO semantic_stats VALUES (?,?)",
        stats_rows,
    )

    return len(passages) + len(stats_rows)


def import_phrase_reconstruction(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "phrase_reconstruction.json")
    if not data:
        return 0
    count = 0

    # anchor_confidence
    ac = data.get("anchor_confidence", {})
    if ac:
        cur.executemany(
            "INSERT OR REPLACE INTO anchor_confidence VALUES (?,?,?,?,?,?,?,?)",
            [(name, v["score"], v["grade"], v["length_score"],
              v["distance_score"], v["domain_score"], v["freq_score"],
              v["context_score"])
             for name, v in ac.items()],
        )
        count += len(ac)

    # reconstructed_words
    rw = data.get("reconstructed_words", {})
    if rw:
        cur.executemany(
            "INSERT OR REPLACE INTO reconstructed_words VALUES (?,?,?,?,?,?,?,?)",
            [(heb, v["candidate"], v.get("candidate_gloss", ""),
              v["distance"], v.get("semantic_score", 0),
              v["confidence"], v.get("source_passage", ""),
              v.get("iteration", 0))
             for heb, v in rw.items()],
        )
        count += len(rw)

    # reconstruction_log
    log = data.get("iteration_log", [])
    if log:
        rows = []
        for entry in log:
            it = entry["iteration"]
            for k in ("new_types", "new_tokens", "cumulative_types"):
                if k in entry:
                    rows.append((it, k, entry[k]))
        cur.executemany(
            "INSERT OR REPLACE INTO reconstruction_log VALUES (?,?,?)",
            rows,
        )
        count += len(rows)

    return count


def import_currier_split(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "currier_split.json")
    if not data:
        return 0
    count = 0

    # Flat stats per language
    for lang in ("A", "B"):
        mr = (data.get("match_results") or {}).get(lang, {})
        for metric in ("n_decoded", "n_matched", "match_rate",
                       "n_unique", "n_unique_matched", "type_match_rate"):
            if metric in mr:
                cur.execute(
                    "INSERT OR REPLACE INTO currier_split_stats VALUES (?,?,?)",
                    (lang, metric, str(mr[metric])),
                )
                count += 1
        perm = (data.get("permutation_tests") or {}).get(lang, {})
        for metric in ("z_score", "p_value", "real_score",
                       "random_mean", "random_std"):
            if metric in perm:
                cur.execute(
                    "INSERT OR REPLACE INTO currier_split_stats VALUES (?,?,?)",
                    (lang, f"perm_{metric}", str(perm[metric])),
                )
                count += 1
        corpus = (data.get("corpus") or {}).get(lang, {})
        for metric in ("n_pages", "n_words"):
            if metric in corpus:
                cur.execute(
                    "INSERT OR REPLACE INTO currier_split_stats VALUES (?,?,?)",
                    (lang, metric, str(corpus[metric])),
                )
                count += 1

    # Comparison
    comp = data.get("comparison_a_vs_b", {})
    for metric in ("z_score", "p_value", "diff"):
        if metric in comp:
            cur.execute(
                "INSERT OR REPLACE INTO currier_split_stats VALUES (?,?,?)",
                ("A_vs_B", metric, str(comp[metric])),
            )
            count += 1

    # Section stats
    sec_stats = data.get("section_stats", {})
    for lang, sections in sec_stats.items():
        for sec, s in sections.items():
            cur.execute(
                "INSERT OR REPLACE INTO currier_section_stats VALUES (?,?,?,?,?,?)",
                (lang, sec, s.get("n_pages", 0), s.get("n_decoded", 0),
                 s.get("n_matched", 0), s.get("match_rate", 0)),
            )
            count += 1

    return count


def import_naibbe_test(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "naibbe_test.json")
    if not data:
        return 0
    rows = []
    # Scorecard rows
    for entry in data.get("scorecard", []):
        rows.append((
            entry.get("metric", ""),
            entry.get("value", ""),
            entry.get("verdict", ""),
        ))
    # Verdict
    vi = data.get("verdict", {})
    if vi:
        rows.append(("final_verdict", vi.get("verdict", ""), vi.get("detail", "")))
    # Simulation summary
    sim = data.get("simulation", {})
    if sim:
        rows.append(("sim_mean", str(sim.get("sim_mean", "")), ""))
        rows.append(("sim_std", str(sim.get("sim_std", "")), ""))
        rows.append(("sim_z_score", str(sim.get("z_score", "")), ""))
        rows.append(("sim_p_value", str(sim.get("p_value", "")), ""))
    cur.executemany(
        "INSERT OR REPLACE INTO naibbe_test VALUES (?,?,?)",
        rows,
    )
    return len(rows)


def import_judeo_italian_test(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "judeo_italian_test.json")
    if not data:
        return 0
    rows = []
    # Match rates
    mr = data.get("match_rates", {})
    for key in ("ji_strict", "ji_normalized", "hebrew"):
        sub = mr.get(key, {})
        if sub:
            rows.append((
                f"match_rate_{key}",
                str(sub.get("rate", "")),
                f"n_matched={sub.get('n_matched', 0)}",
            ))
    rows.append(("n_total", str(mr.get("n_total", "")), ""))
    # Overlap
    ov = data.get("overlap", {})
    for key in ("both_types", "ji_only_types", "heb_only_types",
                "ji_explains_heb_pct"):
        if key in ov:
            rows.append((f"overlap_{key}", str(ov[key]), ""))
    # Permutation test
    pt = data.get("permutation_test", {})
    for key in ("real_rate", "sim_mean", "sim_std", "z_score", "p_value"):
        if key in pt:
            rows.append((f"perm_{key}", str(pt[key]), ""))
    # JI lexicon stats
    jl = data.get("ji_lexicon", {})
    for key in ("n_italian_forms", "n_ji_strict", "n_ji_normalized",
                "n_hebrew_lexicon"):
        if key in jl:
            rows.append((f"lexicon_{key}", str(jl[key]), ""))
    # Hebrew z from cross-language
    heb_z = data.get("hebrew_z_from_cross_language")
    if heb_z is not None:
        rows.append(("hebrew_z_cross_language", str(heb_z), ""))
    cur.executemany(
        "INSERT OR REPLACE INTO judeo_italian_test VALUES (?,?,?)",
        rows,
    )
    return len(rows)


def import_hand1_dive(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "hand1_deep_dive.json")
    if not data:
        return 0
    rows = []

    # Vocab section
    vocab = data.get("vocab", {})
    rows.append(("h1_n_types", str(vocab.get("h1_n_types", "")), ""))
    rows.append(("h1_n_types_matched", str(vocab.get("h1_n_types_matched", "")),
                 f"type_rate={100*vocab.get('h1_n_types_matched',0)/max(1,vocab.get('h1_n_types',1)):.1f}%"))
    rows.append(("exclusive_n_types", str(vocab.get("exclusive_n_types", "")),
                 f"matched={vocab.get('exclusive_matched', '')}"))
    rows.append(("shared_n_types", str(vocab.get("shared_n_types", "")),
                 f"matched={vocab.get('shared_matched', '')}"))
    h14 = vocab.get("h1_vs_h4", {})
    rows.append(("jaccard_h1_h4", str(h14.get("jaccard_14", "")),
                 f"h1_only={h14.get('h1_only','')} h4_only={h14.get('h4_only','')}"))
    h12 = vocab.get("h1_vs_h2", {})
    rows.append(("jaccard_h1_h2", str(h12.get("jaccard_12", "")),
                 f"shared={h12.get('shared_12','')}"))

    # Structure section
    struct = data.get("structure", {})
    lc = struct.get("length_comparison", {})
    for hand, ls in lc.items():
        rows.append((f"word_len_mean_h{hand}", str(ls.get("mean", "")),
                     f"std={ls.get('std','')} median={ls.get('median','')}"))

    # Audit section
    audit = data.get("audit", {})
    rows.append(("audit_n_optimal", str(audit.get("n_optimal", "")),
                 f"of {audit.get('n_total', '')} letters"))
    rows.append(("audit_base_tokens", str(audit.get("base_tokens", "")),
                 "token matches at baseline (Hand 1 only, honest lex)"))
    rows.append(("audit_non_optimal", str(audit.get("non_optimal_letters", "")), ""))
    for heb, r in sorted(audit.get("unmapped_test", {}).items()):
        rows.append((f"unmapped_{heb}_gain",
                     str(r.get("token_gain", "")),
                     f"name={r.get('hebrew_name','')} best_swap={r.get('best_eva_to_swap','')}"))

    # Compare H1 vs H4
    compare = data.get("compare", {})
    h1s = compare.get("h1_stats", {})
    h4s = compare.get("h4_stats", {})
    if h1s and h4s:
        rows.append(("h1_match_rate", str(h1s.get("match_rate", "")),
                     f"n_decoded={h1s.get('n_decoded','')} n_matched={h1s.get('n_matched','')}"))
        rows.append(("h4_match_rate", str(h4s.get("match_rate", "")),
                     f"n_decoded={h4s.get('n_decoded','')} n_matched={h4s.get('n_matched','')}"))
        zt = compare.get("z_test_h1_vs_h4", {})
        rows.append(("h1_vs_h4_z", str(zt.get("z_score", "")),
                     f"p={zt.get('p_value','')} diff={zt.get('diff','')}"))
        rows.append(("h1_vs_h4_diff_pp",
                     str(round((h1s.get("match_rate",0)-h4s.get("match_rate",0))*100, 2)),
                     "percentage points"))

    cur.executemany(
        "INSERT OR REPLACE INTO hand1_dive VALUES (?,?,?)",
        rows,
    )
    return len(rows)


def import_null_model_test(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "null_model_test.json")
    if not data:
        return 0
    rows = []

    # Test 1: Match Rate
    t1 = data.get("test1_match_rate", {})
    rows.append(("t1_real_match_rate", str(t1.get("real_match_rate", "")),
                 f"matched={t1.get('real_matched','')}/{t1.get('total_tokens','')}"))
    rows.append(("t1_synth_mean", str(t1.get("synth_mean", "")),
                 f"std={t1.get('synth_std','')}"))
    rows.append(("t1_z_score", str(t1.get("z_score", "")),
                 f"p={t1.get('p_value','')}"))
    rows.append(("t1_ratio", str(t1.get("ratio", "")), "real/synthetic"))
    rows.append(("t1_significant", str(t1.get("significant", "")), ""))

    # Test 2: Gloss Quality
    t2 = data.get("test2_gloss_quality", {})
    for metric_name in ["entropy", "mean_gloss_len", "top5_concentration", "hapax_ratio"]:
        m = t2.get(metric_name, {})
        rows.append((f"t2_{metric_name}_z", str(m.get("z_score", "")),
                     f"real={m.get('real','')} synth={m.get('synth_mean','')}±{m.get('synth_std','')}"))
    rows.append(("t2_n_significant_sub", str(t2.get("n_significant_sub", "")), "of 4"))
    rows.append(("t2_composite_significant", str(t2.get("composite_significant", "")), ""))

    # Test 3: Bigram Plausibility
    t3 = data.get("test3_bigram_plausibility", {})
    rows.append(("t3_real_score", str(t3.get("real_score", "")), "mean log-likelihood"))
    rows.append(("t3_synth_mean", str(t3.get("synth_mean", "")),
                 f"std={t3.get('synth_std','')}"))
    rows.append(("t3_z_score", str(t3.get("z_score", "")),
                 f"p={t3.get('p_value','')}"))
    rows.append(("t3_significant", str(t3.get("significant", "")), ""))

    # Verdict
    verdict = data.get("verdict", {})
    rows.append(("verdict", str(verdict.get("verdict", "")),
                 f"{verdict.get('significant_tests','')}/{verdict.get('total_tests','')} significant"))
    rows.append(("verdict_explanation", str(verdict.get("explanation", "")), ""))

    cur.executemany(
        "INSERT OR REPLACE INTO null_model_test VALUES (?,?,?)",
        rows,
    )
    return len(rows)


def import_section_entropy(cur: sqlite3.Cursor) -> int:
    data = load_json(STATS / "section_entropy.json")
    if not data:
        return 0
    rows = []

    # Match rates per section
    match_rates = data.get("match_rates", {})
    for sec, s in sorted(match_rates.items()):
        rows.append((f"rate_{sec}_honest", str(s.get("honest_rate", "")),
                     f"matched={s.get('honest_matched','')}/{s.get('n_decoded','')} "
                     f"pages={s.get('n_pages','')}"))
        rows.append((f"rate_{sec}_full", str(s.get("full_rate", "")),
                     f"matched={s.get('full_matched','')}/{s.get('n_decoded','')}"))
        rows.append((f"inflate_{sec}", str(s.get("inflation_factor", "")),
                     "full/honest ratio"))

    # Uniformity tests
    for label, key in [("honest", "uniformity_honest"), ("full", "uniformity_full")]:
        u = data.get(key, {})
        rows.append((f"chi2_{label}", str(u.get("chi2", "")),
                     f"df={u.get('df','')}, p={u.get('p_value','')}"))
        rows.append((f"uniform_{label}", str(u.get("significant_05", "")),
                     f"pooled_rate={u.get('pooled_rate','')}"))

    # Pairwise
    pw = data.get("pairwise", {}).get("best_vs_worst", {})
    if pw:
        rows.append(("pairwise_best", str(pw.get("best", "")),
                     f"rate={pw.get('p1','')}"))
        rows.append(("pairwise_worst", str(pw.get("worst", "")),
                     f"rate={pw.get('p2','')}"))
        rows.append(("pairwise_z", str(pw.get("z_score", "")),
                     f"p={pw.get('p_value','')}"))

    cur.executemany(
        "INSERT OR REPLACE INTO section_entropy VALUES (?,?,?)",
        rows,
    )
    return len(rows)


def build(db_path: str) -> None:
    db_path = str(Path(db_path).resolve())
    print(f"Building SQLite database: {db_path}\n")

    # Remove existing DB and stale WAL/SHM files to rebuild from scratch
    Path(db_path).unlink(missing_ok=True)
    Path(db_path + "-wal").unlink(missing_ok=True)
    Path(db_path + "-shm").unlink(missing_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=DELETE")
    cur.execute("PRAGMA synchronous=NORMAL")

    create_tables(cur)
    conn.commit()

    importers = [
        ("glossed_words", import_glossed_words),
        ("anchor_matches", import_anchor_matches),
        ("phrase_resolutions", import_phrase_resolutions),
        ("compound_splits", import_compound_splits),
        ("contextual_phrases", import_contextual_phrases),
        ("cipher_mapping", import_cipher_mapping),
        ("lexicon", import_lexicon),
        ("validation", import_validation),
        ("letter_audit", import_letter_audit),
        ("semantic_coherence", import_semantic_coherence),
        ("phrase_reconstruction", import_phrase_reconstruction),
        ("currier_split", import_currier_split),
        ("naibbe_test", import_naibbe_test),
        ("judeo_italian_test", import_judeo_italian_test),
        ("hand1_dive", import_hand1_dive),
        ("null_model_test", import_null_model_test),
        ("section_entropy", import_section_entropy),
    ]

    for name, fn in importers:
        print(f"  Importing {name}...", end=" ", flush=True)
        n = fn(cur)
        conn.commit()
        print(f"{n:,} rows")

    # Summary
    print()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    total = 0
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM [{t}]")
        n = cur.fetchone()[0]
        total += n
        print(f"  {t}: {n:,} rows")

    print(f"\n  TOTAL: {total:,} rows")

    conn.close()
    size_mb = Path(db_path).stat().st_size / 1024 / 1024
    print(f"  Database size: {size_mb:.1f} MB")
    print(f"\nDone! Database ready at: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Voynich SQLite database")
    parser.add_argument(
        "--output", "-o",
        default=str(ROOT / "voynich.db"),
        help="Output database path (default: voynich-toolkit/voynich.db)",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
