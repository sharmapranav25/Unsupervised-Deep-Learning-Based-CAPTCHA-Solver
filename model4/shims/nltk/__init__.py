"""Minimal shim: strhub only uses nltk.edit_distance (metrics). Full nltk imports sqlite3, whose DLL is unavailable here."""
from rapidfuzz.distance import Levenshtein as _L


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    assert substitution_cost == 1 and not transpositions
    return _L.distance(s1, s2)
