"""Crisis-keyword safety net (deterministic override).

Why this exists
---------------
The TF-IDF + Logistic Regression NLP agent is unreliable on short / misspelled
inputs (see SupervisorAgent warnings). For mental-health *crisis* language we
cannot trust a probabilistic prediction — we need a deterministic rule that
escalates the final status to **Crisis** regardless of model confidence.

Design
------
- Regex patterns tolerant to common misspellings + repeated letters
  (``saaaad`` -> ``saad``, ``sucide`` -> matches the ``suicide`` family).
- No external NLP dependency.
- Returns the matched phrase + category so the UI can explain *why* it fired.

Severity
--------
``CAT_SUICIDE``, ``CAT_SELF_HARM`` and ``CAT_SEVERE`` are *crisis* categories
that force a Crisis override in the Supervisor. ``CAT_HOPELESS`` is a
contributing signal that adds a warning but does not override on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# ----- categories (ordered roughly by severity) ---------------------------
CAT_SUICIDE = "suicidal_ideation"
CAT_SEVERE = "imminent_action"
CAT_SELF_HARM = "self_harm"
CAT_HOPELESS = "hopelessness"

CRISIS_CATEGORIES: frozenset[str] = frozenset({CAT_SUICIDE, CAT_SEVERE, CAT_SELF_HARM})

# ----- patterns -----------------------------------------------------------
# Letter `+` / `*` quantifiers absorb common typos (saad, suuicide, sucide).
# All patterns are compiled with ``re.IGNORECASE``.
_PATTERNS: list[tuple[str, str]] = [
    # ---- Suicidal ideation -------------------------------------------------
    (CAT_SUICIDE, r"\bs+u+i*c+i*d+e*\b"),                       # suicide, sucide, suiicide
    (CAT_SUICIDE, r"\bs+u+i+s+i+d+e*\b"),                       # suiside, suisid (rarer)
    (CAT_SUICIDE, r"\bcommit\s+s+u+i*c+i*d+e*\b"),
    (CAT_SUICIDE, r"\bk+i+l+(?:ing)?\s+(?:my)?\s*s+e+l+f+\b"),  # killing myself / kill myself
    (CAT_SUICIDE, r"\b(?:k\.?m\.?s\.?)\b"),                     # 'kms' shorthand
    (CAT_SUICIDE, r"\bend(?:ing)?\s+(?:my\s+|this\s+)?life\b"),
    (CAT_SUICIDE, r"\btake\s+my\s+(?:own\s+)?life\b"),
    (CAT_SUICIDE, r"\bwant(?:ed)?\s+to\s+die\b"),
    (CAT_SUICIDE, r"\bdon'?t\s+want\s+to\s+(?:live|be\s+here|exist)\b"),
    (CAT_SUICIDE, r"\bbetter\s+off\s+(?:without\s+me|dead)\b"),
    (CAT_SUICIDE, r"\bno\s+(?:one|body)\s+would\s+miss\s+me\b"),
    # ---- "attempt" near a death/kill cue (severe escalator) ---------------
    (
        CAT_SEVERE,
        r"\ba*t+empt(?:ing|ed)?\b\s+(?:to\s+)?"
        r"(?:s+u+i*c+i*d+e*|d+i+e+|k+i+l+\s*(?:my)?\s*self|end\s+(?:my\s+)?life)",
    ),
    (CAT_SEVERE, r"\bplan(?:ning|ned)?\s+to\s+(?:kill|die|end\s+(?:it|my\s+life))"),
    (CAT_SEVERE, r"\bsoon\s+(?:i\s+)?(?:will\s+)?(?:die|end\s+it|kms)\b"),
    # ---- Self-harm --------------------------------------------------------
    (CAT_SELF_HARM, r"\bself[-\s]?harm(?:ing)?\b"),
    (CAT_SELF_HARM, r"\bcut(?:ting)?\s+my\s*self\b"),
    (CAT_SELF_HARM, r"\bhurt(?:ing)?\s+my\s*self\b"),
    (CAT_SELF_HARM, r"\boverdos(?:e|ing|ed)\b"),
    # ---- Hopelessness (does NOT auto-Crisis on its own) -------------------
    (CAT_HOPELESS, r"\b(?:completely\s+|totally\s+|so\s+)?hopeless\b"),
    (CAT_HOPELESS, r"\bno\s+(?:point|reason|hope)\b"),
    (CAT_HOPELESS, r"\bnothing\s+matters\b"),
    (CAT_HOPELESS, r"\bcan'?t\s+go\s+on\b"),
    (CAT_HOPELESS, r"\bgive\s*up(?:\s+on\s+life)?\b"),
]

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (cat, re.compile(pat, re.IGNORECASE)) for cat, pat in _PATTERNS
]

_RE_COLLAPSE_REPEATS = re.compile(r"([a-z])\1{2,}")
_RE_INTRA_PUNCT = re.compile(r"(?<=[a-z])[.\-_*](?=[a-z])")


@dataclass
class CrisisHit:
    category: str
    phrase: str   # the actual matched text from the user input
    pattern: str  # the regex (debug / display)


# ----- public API ---------------------------------------------------------
def normalize_for_match(text: str) -> str:
    """Light pre-clean: lowercase, collapse 3+ char repeats, strip intra-word punctuation."""
    if not text:
        return ""
    t = text.lower()
    t = _RE_COLLAPSE_REPEATS.sub(r"\1\1", t)        # "saaaad" -> "saad"
    t = _RE_INTRA_PUNCT.sub("", t)                   # "s.u.i.c.i.d.e" -> "suicide"
    return t


def find_crisis_signals(*texts: str) -> list[CrisisHit]:
    """Scan one or more text fragments and return deduplicated crisis matches."""
    hits: list[CrisisHit] = []
    seen: set[tuple[str, str]] = set()
    for raw in texts:
        if not raw:
            continue
        norm = normalize_for_match(raw)
        if not norm:
            continue
        for cat, pat in _COMPILED:
            for m in pat.finditer(norm):
                key = (cat, m.group(0).strip())
                if key in seen:
                    continue
                seen.add(key)
                hits.append(CrisisHit(category=cat, phrase=m.group(0).strip(), pattern=pat.pattern))
    return hits


def has_crisis(hits: Iterable[CrisisHit]) -> bool:
    """True if any hit belongs to the crisis-override categories."""
    return any(h.category in CRISIS_CATEGORIES for h in hits)


# ----- hotlines (shown by the UI when crisis is detected) -----------------
HOTLINES: list[dict[str, str]] = [
    {"region": "India", "name": "iCall (TISS)", "number": "9152987821"},
    {"region": "India", "name": "Vandrevala Foundation", "number": "1860-2662-345"},
    {"region": "India", "name": "AASRA", "number": "+91-9820466726"},
    {"region": "US", "name": "988 Suicide & Crisis Lifeline", "number": "988"},
    {"region": "US", "name": "Crisis Text Line", "number": "Text HOME to 741741"},
    {"region": "UK", "name": "Samaritans", "number": "116 123"},
    {"region": "UK", "name": "SHOUT", "number": "Text SHOUT to 85258"},
    {"region": "EU", "name": "European emergency line", "number": "112"},
    {"region": "EU", "name": "Telefonseelsorge (DE)", "number": "0800 111 0 111"},
    {"region": "AU", "name": "Lifeline Australia", "number": "13 11 14"},
    {"region": "CA", "name": "Talk Suicide Canada", "number": "1-833-456-4566"},
    {"region": "Pakistan", "name": "Umang Pakistan", "number": "0311-7786264"},
    {"region": "Pakistan", "name": "Rozan Counselling Helpline", "number": "0304-1118888"},
    {"region": "Pakistan", "name": "Taskeen Health Initiative", "number": "0316-8275336"},
    {"region": "International", "name": "findahelpline.com", "number": "https://findahelpline.com"},
]


# Region picker labels used by the UI sidebar dropdown.
REGION_LABELS: list[tuple[str, str]] = [
    ("auto", "Auto (show all)"),
    ("India", "India"),
    ("Pakistan", "Pakistan"),
    ("US", "United States"),
    ("UK", "United Kingdom"),
    ("EU", "Europe"),
    ("AU", "Australia"),
    ("CA", "Canada"),
    ("International", "International"),
]


def get_hotlines_for_region(region: str | None) -> list[dict[str, str]]:
    """Return the hotline list filtered by region.

    ``region=None`` or ``"auto"`` returns the full list (caller may decide to
    show all). The ``International`` row is always appended as a safety net
    when filtering by a single country so users always have a fallback link.
    """
    if not region or region.lower() == "auto":
        return list(HOTLINES)
    region_lc = region.strip()
    primary = [h for h in HOTLINES if h["region"].lower() == region_lc.lower()]
    fallback = [h for h in HOTLINES if h["region"] == "International"]
    if not primary:
        return list(HOTLINES)
    # Avoid duplicates if `region == "International"` was already requested.
    seen = {(h["region"], h["name"]) for h in primary}
    extras = [h for h in fallback if (h["region"], h["name"]) not in seen]
    return primary + extras


# ----- thin agent wrapper (so it appears alongside the others) -----------
class SafetyAgent:
    """Stateless wrapper exposing the safety lexicon as an agent."""

    def scan(self, *texts: str) -> list[CrisisHit]:
        return find_crisis_signals(*texts)

    def is_crisis(self, *texts: str) -> bool:
        return has_crisis(self.scan(*texts))
