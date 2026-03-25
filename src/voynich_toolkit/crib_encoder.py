"""
Crib encoder: reverse the Voynich cipher (Hebrew consonantal → EVA script).

This module implements the reverse of the decryption pipeline in full_decode.py.
Given Hebrew consonantal text, it encodes to EVA script for comparison against
the actual manuscript ("cribs" for cryptanalysis).

The forward mapping (EVA → Hebrew) is inverted, with special handling for:
- Positional rules (bet at initial → n in EVA, samekh at initial → r in EVA)
- Digraphs (k in Hebrew → ch in EVA, he → ii)
- Allographs (lamed has f/p; resh has d/i; he has r/ii)
"""

from itertools import product
from typing import Dict, List, Tuple


# EVA → Hebrew forward mapping (from full_decode.py)
FULL_MAPPING = {
    'a': 'y', 'c': 'A', 'd': 'r', 'e': 'p', 'f': 'l',
    'g': 'X', 'h': 'E', 'k': 't', 'l': 'm', 'm': 'g',
    'n': 'd', 'o': 'w', 'p': 'l', 'r': 'h', 's': 'n',
    't': 'J', 'y': 'S',
}

# Special digraphs and allographs in the forward mapping:
# ch (digraph) → k (kaf)
# ii (digraph) → h (he)
# i (standalone) → r (resh)
# n at Hebrew-initial → b (bet)
# r/ii at Hebrew-initial → s (samekh)


def build_reverse_mapping() -> Dict[str, str]:
    """Build Hebrew → EVA reverse mapping.

    Where multiple EVA chars map to the same Hebrew letter, use the primary one:
    - lamed (l) → p (primary, f is allograph)
    - resh (r) → d (primary, i is allograph)
    - he (h) → r (primary, ii is allograph)

    Special Hebrew letters zayin(z), tsade(C), qof(q) are unmapped → '?'

    Returns:
        Dict[str, str]: Hebrew char → EVA char mapping.
    """
    reverse_map = {}

    # Invert FULL_MAPPING, preferring primary encodings
    for eva_char, hebrew_char in FULL_MAPPING.items():
        # Skip digraphs for now; they'll be handled specially
        if eva_char in ('ch', 'ii', 'i'):
            continue

        # If already mapped, keep the primary one (first occurrence)
        if hebrew_char not in reverse_map:
            reverse_map[hebrew_char] = eva_char

    # Override with primary preferences
    reverse_map['l'] = 'p'  # lamed → p (primary over f)
    reverse_map['r'] = 'd'  # resh → d (primary over i)
    reverse_map['h'] = 'r'  # he → r (primary over ii)

    # Special digraphs
    reverse_map['k'] = 'ch'  # kaf → ch (digraph)

    # Unmapped letters become '?'
    for char in ['z', 'C', 'q']:
        if char not in reverse_map:
            reverse_map[char] = '?'

    return reverse_map


def encode_word(hebrew_consonantal: str) -> str:
    """Encode a single Hebrew consonantal word to EVA.

    Pipeline:
    1. Apply positional rules:
       - bet (b) at position 0 (Hebrew-initial, before RTL reverse) → n in EVA
       - samekh (s) at position 0 (Hebrew-initial, before RTL reverse) → r in EVA
    2. Map each Hebrew char to EVA using reverse mapping
    3. Reverse the string (because decode reads RTL, so encode writes LTR)

    Args:
        hebrew_consonantal (str): Hebrew consonantal word (e.g., "mwk")

    Returns:
        str: EVA encoded word. Unknown chars become '?'.
    """
    if not hebrew_consonantal:
        return ""

    reverse_map = build_reverse_mapping()

    # Apply positional rules BEFORE reversal
    # In Hebrew, initial position is the rightmost character (RTL), but we're
    # working with the input string order, so position 0 is logically "initial"
    eva_chars = []

    for i, hebrew_char in enumerate(hebrew_consonantal):
        eva_char = reverse_map.get(hebrew_char, '?')

        # Apply positional substitutions at initial position (pos 0)
        if i == 0:
            if hebrew_char == 'b':
                eva_char = 'n'
            elif hebrew_char == 's':
                eva_char = 'r'

        eva_chars.append(eva_char)

    # Reverse the string (RTL → LTR in EVA output)
    eva_word = ''.join(reversed(eva_chars))

    return eva_word


# Allograph alternatives: Hebrew char → list of possible EVA encodings
# These capture the manuscript's scribal variation
ALLOGRAPH_MAP = {
    'h': ['r', 'ii'],       # he → r (primary) or ii
    'r': ['d', 'i'],         # resh → d (primary) or i (standalone)
    'l': ['p', 'f'],         # lamed → p (primary) or f
    's': ['r', 'ii'],        # samekh at initial → r or ii
    'b': ['n'],              # bet at initial → n (no alternative)
}


def encode_word_variants(hebrew_consonantal: str, with_q_prefix: bool = True) -> List[str]:
    """Encode a Hebrew word to ALL possible EVA variants (allograph combinations).

    Generates every combination of allograph choices for ambiguous letters:
    - h (he): r or ii
    - r (resh): d or i
    - l (lamed): p or f
    Plus optionally adds q/qo prefix variants.

    Args:
        hebrew_consonantal: Hebrew consonantal word
        with_q_prefix: if True, also generate qo-prefixed variants

    Returns:
        List of all possible EVA encodings (deduplicated).
    """
    if not hebrew_consonantal:
        return ['']

    reverse_map = build_reverse_mapping()

    # Build list of possibilities for each position
    char_options = []
    for i, hch in enumerate(hebrew_consonantal):
        if i == 0 and hch == 'b':
            char_options.append(['n'])
        elif i == 0 and hch == 's':
            # samekh at initial: r or ii
            char_options.append(['r', 'ii'])
        elif hch == 'h':
            char_options.append(['r', 'ii'])
        elif hch == 'r':
            char_options.append(['d', 'i'])
        elif hch == 'l':
            char_options.append(['p', 'f'])
        elif hch == 'k':
            char_options.append(['ch'])
        else:
            eva = reverse_map.get(hch, '?')
            char_options.append([eva])

    variants = set()
    for combo in product(*char_options):
        # Reverse (RTL→LTR)
        eva_word = ''.join(reversed(combo))
        variants.add(eva_word)
        if with_q_prefix:
            variants.add('qo' + eva_word)
            variants.add('q' + eva_word)

    return sorted(variants)


# Scribal equivalence pairs discovered from near-match analysis (Phase 26).
# These are letters that medieval Italian-Hebrew scribes commonly confused,
# either phonetically (gutturals) or visually (similar letterforms).
SCRIBAL_EQUIVALENCES = {
    't': ['t', 'J'],   # tav ↔ tet: phonetic merger in Italian Hebrew
    'J': ['J', 't'],
    'X': ['X', 'E'],   # chet ↔ ayin: guttural confusion (×11 in near-matches)
    'E': ['E', 'X'],
    'w': ['w', 'r'],   # vav ↔ resh: visually near-identical in cursive (×11)
    'r': ['r', 'w'],
}


def _scribal_variants(hebrew_word: str) -> list:
    """Generate variants applying known scribal equivalences.

    Medieval Italian Hebrew scribes commonly confused:
    - t ↔ J (tav ↔ tet): phonetic merger
    - X ↔ E (chet ↔ ayin): guttural confusion
    - w ↔ r (vav ↔ resh): visual similarity in cursive

    Returns list of unique variants including the original.
    """
    # Check if any equivalence applies
    has_equiv = any(ch in SCRIBAL_EQUIVALENCES for ch in hebrew_word)
    if not has_equiv:
        return [hebrew_word]

    from itertools import product as _product
    options = []
    for ch in hebrew_word:
        if ch in SCRIBAL_EQUIVALENCES:
            options.append(SCRIBAL_EQUIVALENCES[ch])
        else:
            options.append([ch])

    # Cap explosion: for very long words with many equivalences, limit combinations
    n_combos = 1
    for opt in options:
        n_combos *= len(opt)
    if n_combos > 512:
        # Too many combinations — only swap one letter at a time
        variants = {hebrew_word}
        for i, ch in enumerate(hebrew_word):
            if ch in SCRIBAL_EQUIVALENCES:
                for alt in SCRIBAL_EQUIVALENCES[ch]:
                    variant = hebrew_word[:i] + alt + hebrew_word[i+1:]
                    variants.add(variant)
        return list(variants)

    return list({''.join(combo) for combo in _product(*options)})


def best_variant_match(hebrew_word: str, real_eva_word: str) -> Tuple[str, int]:
    """Find the allograph variant of a Hebrew word closest to a real EVA word.

    Also tries scribal equivalence swaps (t↔J, X↔E, w↔r), since these
    letters were commonly confused in medieval Italian Hebrew manuscripts.

    Returns:
        (best_variant, edit_distance)
    """
    # Generate scribal variants of the Hebrew word, then EVA variants of each
    heb_variants = _scribal_variants(hebrew_word)
    best = None
    best_dist = 999
    for heb_var in heb_variants:
        for v in encode_word_variants(heb_var):
            d = _edit_distance(v, real_eva_word)
            if d < best_dist:
                best_dist = d
                best = v
            if d == 0:
                return best, 0
    return best, best_dist


def encode_text(hebrew_words: List[str]) -> List[str]:
    """Encode a list of Hebrew consonantal words to EVA words.

    Args:
        hebrew_words (List[str]): List of Hebrew consonantal words

    Returns:
        List[str]: List of EVA encoded words
    """
    return [encode_word(word) for word in hebrew_words]


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        s1 (str): First string
        s2 (str): Second string

    Returns:
        int: Minimum edit distance (insertions, deletions, substitutions)
    """
    len1, len2 = len(s1), len(s2)

    # Create DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                   dp[i][j - 1],      # insertion
                                   dp[i - 1][j - 1])  # substitution

    return dp[len1][len2]


def _char_bigrams(word: str) -> set:
    """Extract character bigrams from a word.

    Args:
        word (str): Input word

    Returns:
        set: Set of character bigrams (2-grams)
    """
    if len(word) < 2:
        return set()
    return {word[i:i+2] for i in range(len(word) - 1)}


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set1 (set): First set
        set2 (set): Second set

    Returns:
        float: Jaccard similarity in [0, 1]
    """
    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def compare_eva(generated_eva: List[str], real_eva: List[str]) -> Dict:
    """Compare generated EVA against real EVA from the manuscript.

    Computes:
    - Exact matches: word-by-word equality
    - Partial matches: edit distance < 2 (1-char difference)
    - Character bigram overlap: Jaccard similarity
    - Best alignment: sliding window to find optimal offset

    Args:
        generated_eva (List[str]): Generated EVA words
        real_eva (List[str]): Real EVA words from manuscript

    Returns:
        Dict with keys:
        - exact_matches (int): Count of exactly matching words
        - partial_matches (List[Tuple]): [(gen_word, real_word, distance), ...]
        - match_ratio (float): exact_matches / min(len(generated), len(real))
        - char_ngram_overlap (float): avg Jaccard bigram similarity
        - alignment_score (float): best alignment score
        - alignment_offset (int): best alignment offset
        - alignment_matches (int): matches in best alignment
    """
    result = {}

    # Exact matches
    exact_matches = 0
    for i in range(min(len(generated_eva), len(real_eva))):
        if generated_eva[i] == real_eva[i]:
            exact_matches += 1

    result['exact_matches'] = exact_matches

    # Match ratio
    min_len = min(len(generated_eva), len(real_eva))
    result['match_ratio'] = exact_matches / min_len if min_len > 0 else 0.0

    # Partial matches (edit distance <= 1)
    partial_matches = []
    for i in range(min(len(generated_eva), len(real_eva))):
        gen_word = generated_eva[i]
        real_word = real_eva[i]
        if gen_word != real_word:
            dist = _edit_distance(gen_word, real_word)
            if dist <= 1:
                partial_matches.append((gen_word, real_word, dist))

    result['partial_matches'] = partial_matches

    # Character bigram overlap (n-gram similarity)
    bigram_scores = []
    for i in range(min(len(generated_eva), len(real_eva))):
        gen_bigrams = _char_bigrams(generated_eva[i])
        real_bigrams = _char_bigrams(real_eva[i])
        similarity = _jaccard_similarity(gen_bigrams, real_bigrams)
        bigram_scores.append(similarity)

    avg_ngram = sum(bigram_scores) / len(bigram_scores) if bigram_scores else 0.0
    result['char_ngram_overlap'] = avg_ngram

    # Best alignment (sliding window)
    longer = generated_eva if len(generated_eva) >= len(real_eva) else real_eva
    shorter = real_eva if len(generated_eva) >= len(real_eva) else generated_eva

    best_score = 0
    best_offset = 0
    best_matches = 0

    for offset in range(len(longer) - len(shorter) + 1):
        matches = 0
        for i in range(len(shorter)):
            if longer[offset + i] == shorter[i]:
                matches += 1

        score = matches / len(shorter) if shorter else 0.0

        if score > best_score or (score == best_score and matches > best_matches):
            best_score = score
            best_offset = offset
            best_matches = matches

    result['alignment_score'] = best_score
    result['alignment_offset'] = best_offset
    result['alignment_matches'] = best_matches

    return result


def score_crib(generated_eva: List[str], real_eva: List[str]) -> float:
    """Compute a single score combining exact matches, partial matches, and n-gram overlap.

    Scoring formula:
    - Exact match: weight 1.0
    - Partial match (dist=1): weight 0.5
    - N-gram overlap: weight 0.3
    - Alignment bonus: weight 0.2

    Args:
        generated_eva (List[str]): Generated EVA words
        real_eva (List[str]): Real EVA words from manuscript

    Returns:
        float: Composite score in [0, 1]
    """
    comparison = compare_eva(generated_eva, real_eva)

    min_len = min(len(generated_eva), len(real_eva))
    if min_len == 0:
        return 0.0

    # Exact match contribution (0-1)
    exact_contrib = comparison['exact_matches'] / min_len

    # Partial match contribution (0-0.5 max)
    partial_contrib = (len(comparison['partial_matches']) * 0.5) / min_len

    # N-gram contribution (0-0.3 max, scaled)
    ngram_contrib = comparison['char_ngram_overlap'] * 0.3

    # Alignment bonus (0-0.2 max)
    alignment_contrib = comparison['alignment_score'] * 0.2

    total_score = min(1.0, exact_contrib + partial_contrib + ngram_contrib + alignment_contrib)

    return total_score
