"""
All tunable knobs live here: thresholds, penalties, fast-path points, and model weights.
"""

# --- Global row-level default (can be overridden by CLI --min-ratio)
MIN_RATIO = 0.70

# --- Heuristics for short/long text (guarding substring/“exact” inside long paragraphs)
LONG_TEXT_CHAR_THRESHOLD   = 95
LONG_TEXT_TOKEN_THRESHOLD  = 15
SHORT_TEXT_CHAR_THRESHOLD  = 35
SHORT_TEXT_TOKEN_THRESHOLD = 4

# When dashboard short is inside source long, require one of these:
MIN_PRECISION_FOR_LONG_TEXT = 0.40   # matched_len / len(source_val)
MIN_PARTIAL_FOR_LONG_TEXT   = 0.90   # partial_ratio
MIN_TFIDF_FOR_LONG_TEXT     = 0.45   # TF-IDF cosine

# --- Fast-path “points” (scores) for early exits
DIRECT_MATCH_SCORE              = 0.99   # pure “single == single”
WRAPPED_LIST_MATCH_SCORE        = 0.94  # member of [A,B] / {A,B} / (A,B)
LISTLIKE_DELIMITED_MATCH_SCORE  = 0.90   # member of unwrapped “A, B, C”

# --- Container penalty applied to BM25/similarity fallback
CONTAINER_PENALTY = {
    "single":         1.00,
    "wrapped_list":   0.94,
    "delimited_list": 0.90,
    "empty":          0.00,
}

# --- Per-tag thresholds: whether a row “counts” as a hit
# The minimum score required for that tag to count as a hit when computing.
TAG_THRESHOLDS = {
    "direct":     0.70,  # will always pass with DIRECT_MATCH_SCORE
    "list":       0.70,
    "listlike":   0.70,
    "bm25":       0.35,  # looser to admit sentence-level matches
    "similarity": 0.45,
    "none":       1.00,
}

# --- Hybrid similarity blend weights (used only in fallback step)
# base score = w_fp*partial_ratio + w_fr*ratio + w_lev*lev + w_tfidf*tfidf
# Logic stays in scoring.py and core.py.
# don’t touch these unless you want new behaviors.
W_FP   = 0.25
W_FR   = 0.25
W_LEV  = 0.20
W_TFIDF= 0.30

# --- TF-IDF vectorizer params (char-ngrams is robust to typos)
TFIDF_ANALYZER     = "char"
TFIDF_NGRAM_RANGE  = (3, 5)

# --- BM25 squashing strength (bigger => smaller normalized scores)
BM25_SQUASH_SCALE = 2.0

# --- Suggestion labels (for best mapping per dashboard col)
HIGH_RATIO = 0.85
HIGH_SCORE = 0.95
