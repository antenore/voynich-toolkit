"""
Shared fuzzy matching infrastructure for Voynich decipherment.

Provides length-bucketed indexing for O(1) length-filtering before
Levenshtein distance computation, plus common dataclasses and scoring.

Uses rapidfuzz for fast Levenshtein with early-exit (score_cutoff).
"""
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein


# Score weights by Levenshtein distance: exact=10, dist1=3, dist2=1
SCORE_WEIGHTS = {0: 10, 1: 3, 2: 1}


@dataclass
class FuzzyMatch:
    """A single fuzzy match result."""
    query: str
    target: str
    distance: int
    score: int
    gloss: str = ""
    domain: str = ""


class LengthBucketedIndex:
    """Index that buckets lexicon forms by length for fast filtering.

    Before computing Levenshtein distance, filters candidates to
    those with |len(query) - len(candidate)| <= max_dist, since
    Levenshtein distance >= length difference.
    """

    def __init__(self, forms, form_to_gloss=None, form_to_domain=None):
        """Build index from iterable of string forms.

        Args:
            forms: iterable of lexicon strings
            form_to_gloss: optional {form: gloss} dict
            form_to_domain: optional {form: domain} dict
        """
        self._buckets = defaultdict(list)
        self._form_to_gloss = form_to_gloss or {}
        self._form_to_domain = form_to_domain or {}
        for form in forms:
            self._buckets[len(form)].append(form)
        self._min_len = min(self._buckets) if self._buckets else 0
        self._max_len = max(self._buckets) if self._buckets else 0

    def query(self, word, max_dist=2):
        """Find all forms within Levenshtein distance max_dist of word.

        Returns list of FuzzyMatch sorted by (distance, target).
        """
        wlen = len(word)
        matches = []

        lo = max(self._min_len, wlen - max_dist)
        hi = min(self._max_len, wlen + max_dist)

        for bucket_len in range(lo, hi + 1):
            for candidate in self._buckets.get(bucket_len, []):
                d = Levenshtein.distance(word, candidate,
                                         score_cutoff=max_dist)
                if d <= max_dist:
                    score = SCORE_WEIGHTS.get(d, 0)
                    matches.append(FuzzyMatch(
                        query=word,
                        target=candidate,
                        distance=d,
                        score=score,
                        gloss=self._form_to_gloss.get(candidate, ""),
                        domain=self._form_to_domain.get(candidate, ""),
                    ))

        matches.sort(key=lambda m: (m.distance, m.target))
        return matches

    def best_match(self, word, max_dist=2):
        """Return best (lowest distance) match or None."""
        matches = self.query(word, max_dist)
        return matches[0] if matches else None

    @property
    def size(self):
        """Total number of indexed forms."""
        return sum(len(b) for b in self._buckets.values())

    @property
    def bucket_sizes(self):
        """Dict of {length: count}."""
        return {k: len(v) for k, v in sorted(self._buckets.items())}


def batch_fuzzy_match(words, index, max_dist=2, min_word_len=3):
    """Match a batch of (word, count) pairs against an index.

    Args:
        words: list of (word_str, count) tuples
        index: LengthBucketedIndex
        max_dist: maximum Levenshtein distance
        min_word_len: skip words shorter than this

    Returns:
        list of (word, count, [FuzzyMatch, ...]) for words with >=1 match
    """
    results = []
    for word, count in words:
        if len(word) < min_word_len:
            continue
        matches = index.query(word, max_dist)
        if matches:
            results.append((word, count, matches))
    return results
