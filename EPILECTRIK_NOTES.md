# Epilectrik/Voynich — Collaboration Notes

## Autore
- **Joe DiPrima** (GitHub: `epilectrik`, Reddit: `aa-epilectrik`)
- Texas, italo-americano 3a generazione (cognome DiPrima)
- Repo: https://github.com/epilectrik/voynich
- Tool: Claude Code + GPT-5 cross-validation
- 2,066 commit, 178MB, 6,357 file, 401 fasi di ricerca, ~988 constraint validate

## La sua ipotesi (radicalmente diversa dalla nostra)

**Claim Tier 0 (frozen fact per lui):**
> Il testo Currier B codifica **programmi di controllo a ciclo chiuso** per mantenere un sistema entro un regime di viabilità — NON è linguaggio naturale.

- Falsifica esplicitamente "è linguaggio" (0.19% reference rate)
- Falsifica "è un cifrario" (trasformazioni DIMINUISCONO la mutual information)
- Le illustrazioni sono **epifenomeniche** (swap invariance p=1.0 — scambiando illustrazioni tra folio la struttura del testo non cambia)
- Interpretazione dominio (Tier 3, speculativa): **distillazione a riflusso** (confronto con Brunschwig 1500)

## Architettura a 4 livelli

| Livello | Sistema | Token | Funzione |
|---------|---------|-------|----------|
| Esecuzione | Currier B | 23,243 (61.9%) | Programmi di controllo adattivo |
| Distinzione | Currier A | 11,415 (30.5%) | Registro di discriminazioni fini |
| Contesto | AZC (astro/zodiac/cosmo) | 3,299 (8.7%) | Tabella di lookup posizionale |
| Orientamento | HT (Human Track) | 7,042 | Specifiche composte ridondanti |

## Risultati chiave del suo lavoro

### Token Morphology
Ogni token Currier B: `[ARTICULATOR] + [PREFIX] + MIDDLE + [SUFFIX]`
- 479 token types → **49 classi di istruzione** (compressione 9.8x, 100% grammar coverage)
- 5 ruoli funzionali: ENERGY_OPERATOR (31.2%), AUXILIARY (16.6%), FREQUENT_OPERATOR (12.5%), CORE_CONTROL (4.4%), FLOW_OPERATOR (4.7%)

### 3 Operatori Kernel (k, h, e)
- **k** = modulazione energetica (inizio riga)
- **h** = gestione transizioni di fase (metà riga)
- **e** = ancora di stabilità (fine riga — 36% di tutti i token B)
- Transizione e→h completamente bloccata

### 17 Transizioni Proibite (5 classi di hazard)
- PHASE_ORDERING (41%), COMPOSITION_JUMP (24%), CONTAINMENT_TIMING (24%), RATE_MISMATCH (6%), ENERGY_OVERSHOOT (6%)

### Direzione di lettura
**LTR confermato** (7-test battery, MI forward bias +0.070 bits, z=17)
- Noi troviamo **RTL** (z=22.97) — non necessariamente in contraddizione se il testo non è linguaggio (la "direzione" misura cose diverse)

### Brunschwig Alignment (28 test, 4 suite)
- Reversibilità 89%, retry medio 1.19 token
- Gradi del fuoco: rho = -0.457, p < 0.0001
- Illustrazioni radici-piante → operazioni POUND/CHOP (r = 0.366, p = 0.0007)

### Falsificazioni permanenti (30+)
- NON è linguaggio (0.19% reference rate)
- NON è cifrario (transform diminuisce MI)
- NON è ricettario step-by-step
- NON è calendario/astrologia (98%+ self-transition rate in AZC)
- NON c'è concatenazione procedurale tra folio
- Illustrazioni NON sono istruttive (swap invariance p=1.0)

## Dati disponibili nel suo repo

| File | Size | Contenuto |
|------|------|-----------|
| `scripts/voynich.py` | 170KB | Libreria principale (Transcript, Morphology, BFolioDecoder) |
| `data/token_dictionary.json` | 8.6MB | ~8,150 token con glosse interpretative |
| `data/middle_dictionary.json` | 320KB | ~1,339 MIDDLE con operazioni |
| `data/brunschwig_curated_v3.json` | 591KB | 245+ ricette curate di Brunschwig |
| `data/decoder_maps.json` | 62KB | Mappe di decodifica |
| `context/CONSTRAINT_TABLE.txt` | 168KB | Tutte le ~988 constraint |
| `context/EXPERT_CONTEXT.md` | 597KB | Contesto consolidato completo |
| `context/STRUCTURAL_CONTRACTS/*.yaml` | ~229KB | Grammatiche formali (B, A, AZC, HT, paragraph) |

## Cosa del suo lavoro ci interessa

1. **Decomposizione morfologica TOKEN → PREFIX+MIDDLE+SUFFIX**
   - 100% coverage, 9.8x compressione → struttura interna forte
   - Potremmo applicare la stessa decomposizione al nostro decoded Hebrew e vedere se la struttura si preserva o si distrugge

2. **Swap invariance delle illustrazioni**
   - p=1.0 — le illustrazioni non predicono il testo
   - Utile per il nostro paper: argomento contro domain-specific content

3. **Entropia condizionale e transizioni proibite**
   - 17 transizioni proibite in 5 classi → grammatica rigida
   - Confronto con le nostre bigram statistics del testo decodificato

4. **Currier A vs B come sistemi qualitativamente diversi**
   - Lui tratta A e B come sistemi DIVERSI (registro vs programmi)
   - Noi li trattiamo come varianti dello stesso fenomeno
   - Il suo A/B split è molto più dettagliato del nostro

5. **Confronto LTR vs RTL**
   - Risultati opposti (lui LTR z=17, noi RTL z=22.97)
   - Entrambi significativi → misurano proprietà diverse?

## Cosa nostra può servire a lui

1. **Database SQLite** (58.7MB, 512K rows) — tutto in un file queryable
2. **Decoded corpus completo** — 37K parole → Hebrew consonantale
3. **Lessico 491K forme** con glosse e frequenze
4. **Permutation framework** riusabile (seed=42, 1000 perms, FDR correction)
5. **Analisi Currier A/B** con match rate per mano (1-5,X,Y,?)
6. **Layout analysis** (IVTFF unit codes: label vs paragraph vs circular)
7. **Paper PDF** — meta-analysis + literature comparison (15 papers)

## Punti di tensione

| Aspetto | Noi | Lui |
|---------|-----|-----|
| Ipotesi | Ebraico consonantico cifrato | Programmi di controllo (non linguaggio) |
| Direzione | RTL (z=22.97) | LTR (z=17) |
| Currier A/B | Varianti dello stesso tipo | Sistemi qualitativamente diversi |
| AZC | Parte del testo decodificato | Tabella di lookup posizionale |
| Illustrazioni | Domain hints (botanical, zodiac) | Epifenomeniche (swap invariance) |

## Punti di convergenza

1. Entrambi usiamo Claude Code come tool primario
2. Entrambi usiamo permutation tests per validazione
3. Entrambi riconosciamo Currier A/B come strutturalmente significativo
4. Entrambi abbiamo sistemi di tier/confidenza per le affermazioni
5. Entrambi siamo rigorosi sulle falsificazioni (non ricicliamo ipotesi fallite)
6. Entrambi troviamo struttura posizionale forte nei token
7. Entrambi notiamo l'asimmetria label vs paragraph

## Test incrociati proposti

1. **Morfologia sul decoded**: applicare la sua decomposizione PREFIX+MIDDLE+SUFFIX al nostro testo Hebrew decodificato → la struttura si preserva? Se sì, la morfologia è nell'EVA (non nell'interpretazione)
2. **Hebrew match su classi**: raggruppare i nostri match rates per le sue 49 classi di istruzione → i "kernel operators" matchano diversamente dagli "auxiliary"?
3. **Swap invariance sul nostro decoded**: le nostre glosse predicono il dominio illustrativo? Se no, conferma la sua swap invariance
4. **Bigram plausibility confronto**: confrontare le nostre bigram z-scores (40.9) con le sue transizioni proibite

## Proposta collaborazione

- **Citarsi reciprocamente** nei rispettivi lavori
- **Condividere dati** in formato comodo (JSON/CSV/SQLite)
- **Test incrociati**: morfologia sua → nostro decoded, Hebrew match → sue classi
- **Pre-registration congiunta** per test incrociati (evita p-hacking)
