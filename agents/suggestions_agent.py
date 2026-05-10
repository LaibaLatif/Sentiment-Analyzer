"""Suggestions Agent — deterministic, mode-aware wellbeing tips.

Replaces the old (removed) Gemini integration with a grounded, fully offline
ruleset so the demo always shows useful guidance — even with no network and
no API keys.

Design rules
------------
* **Educational, never medical.** Every suggestion is a generic self-care or
  help-seeking prompt. Nothing here is diagnostic or treatment advice.
* **Deterministic.** Same final status → same suggestions. Easy to test, easy
  to audit, no LLM hallucinations.
* **Mode-aware.** The list changes based on the supervisor's final status:
  ``Crisis — seek support`` / ``High Risk`` / ``Likely High Risk`` /
  ``Conflict`` / ``Mixed / monitor`` / ``Stable``.
* **Safety first.** When the safety net fires, the SOS tips come first and we
  *do not* dilute them with generic "talk to a professional" wording — that's
  already implicit and the crisis banner above already shows hotlines.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suggestion:
    icon: str
    title: str
    detail: str


# ------------------------------------------------------------------
# Per-mode suggestion lists
# ------------------------------------------------------------------
_CRISIS: tuple[Suggestion, ...] = (
    Suggestion(
        "🚨",
        "Reach out for help right now",
        "If you're in immediate danger, call your local emergency number or one "
        "of the hotlines listed in the red banner above. You don't have to go "
        "through this alone.",
    ),
    Suggestion(
        "📞",
        "Tell one person you trust",
        "A friend, family member, teacher, or colleague — saying it out loud, "
        "even in a short text, breaks the isolation. Pick the easiest contact, "
        "not the 'best' one.",
    ),
    Suggestion(
        "🌬",
        "Try a 4-7-8 breath cycle",
        "Inhale through the nose for 4 seconds, hold for 7, exhale through the "
        "mouth for 8. Repeat 4 times. It calms the nervous system enough to "
        "make the next call easier.",
    ),
    Suggestion(
        "🛡",
        "Move to a safer place",
        "If anything around you could be used to hurt yourself, step into a "
        "different room or ask someone to keep it for you for a while.",
    ),
)


_HIGH_RISK: tuple[Suggestion, ...] = (
    Suggestion(
        "🧘",
        "Try 5-4-3-2-1 grounding",
        "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can "
        "smell, and 1 you can taste. It pulls you out of rumination and back "
        "into the present moment.",
    ),
    Suggestion(
        "📝",
        "Write one line about how you feel",
        "Putting feelings into words reduces emotional intensity — even one "
        "sentence helps more than mentally re-running the situation.",
    ),
    Suggestion(
        "🚶",
        "Take a 10-minute walk outside",
        "Light exercise plus daylight changes the body's stress chemistry. "
        "No goal — just move and notice what you see.",
    ),
    Suggestion(
        "💧",
        "Drink a glass of water and check your last meal",
        "Dehydration and skipped meals consistently amplify low mood. Small, "
        "easy fix that often gets skipped.",
    ),
)


_CONFLICT: tuple[Suggestion, ...] = (
    Suggestion(
        "🤔",
        "Notice the mismatch you wrote vs. how you look",
        "The text and the facial signal disagree. Ask yourself: which one is "
        "closer to what I actually feel right now?",
    ),
    Suggestion(
        "📓",
        "Free-write for 3 minutes",
        "Open a note and write without stopping or editing. Pure stream of "
        "thought. Often the conflict resolves itself once it's on the page.",
    ),
    Suggestion(
        "☕",
        "Take a short break before deciding anything",
        "Mixed signals are not a great moment for big choices. Step away for "
        "10–15 minutes, then revisit.",
    ),
)


_MIXED: tuple[Suggestion, ...] = (
    Suggestion(
        "👀",
        "Re-read what you wrote",
        "Your text leans negative even though your face looks neutral. "
        "Is the writing reflecting what you genuinely feel, or a rougher "
        "moment from earlier today?",
    ),
    Suggestion(
        "🎵",
        "Switch your environment for 5 minutes",
        "Music, a window, a different room — small environment shifts often "
        "make 'low-grade' moods easier to interpret.",
    ),
    Suggestion(
        "🧑‍🤝‍🧑",
        "Reach out to someone for a quick chat",
        "A 2-minute message to a friend can clarify whether you're 'fine' or "
        "actually need a longer talk later.",
    ),
)


_STABLE: tuple[Suggestion, ...] = (
    Suggestion(
        "✅",
        "Keep doing what's working",
        "You're in a steady place — the small habits that got you here "
        "(sleep, movement, social contact) are the ones worth protecting.",
    ),
    Suggestion(
        "🙏",
        "Note one thing you're grateful for",
        "A 10-second mental note is enough. Regular gratitude is one of the "
        "few habits with consistent evidence behind it.",
    ),
    Suggestion(
        "📅",
        "Plan something to look forward to",
        "It doesn't need to be big — coffee with a friend on the weekend "
        "counts. Anticipation is a real mood booster.",
    ),
)


_PROFESSIONAL = Suggestion(
    "🤝",
    "Consider talking to a professional",
    "A counsellor, GP, or therapist can offer assessment and support beyond "
    "what this demo can. Many regions also have free anonymous chat services.",
)


# Map normalized status keyword → suggestion list
_BY_MODE: dict[str, tuple[Suggestion, ...]] = {
    "crisis": _CRISIS,
    "high": _HIGH_RISK,
    "conflict": _CONFLICT,
    "mixed": _MIXED,
    "stable": _STABLE,
}


def _mode_key(status: str | None) -> str:
    """Map a supervisor final-status string → an internal mode key."""
    s = (status or "").strip().lower()
    if "crisis" in s:
        return "crisis"
    if "high" in s:
        return "high"
    if "conflict" in s:
        return "conflict"
    if "mixed" in s:
        return "mixed"
    if "stable" in s:
        return "stable"
    return "mixed"


def suggestions_for(status: str | None, *, hopelessness: bool = False) -> list[Suggestion]:
    """Return mode-appropriate suggestions.

    Parameters
    ----------
    status : str | None
        The supervisor's ``final_status`` — e.g. ``"Stable"``, ``"High Risk"``,
        ``"Crisis — seek support"``.
    hopelessness : bool
        Optional flag. When ``True`` and we're not already in crisis, we add
        the "talk to a professional" tip *first* so it appears at the top of
        the list rather than as a footer.

    Notes
    -----
    * Always returns at least one suggestion.
    * Crisis mode never adds the generic "professional" footer — the crisis
      banner above the suggestions already directs the user to hotlines.
    """
    key = _mode_key(status)
    base = list(_BY_MODE[key])

    if key == "crisis":
        return base

    if hopelessness:
        return [_PROFESSIONAL, *base]

    return [*base, _PROFESSIONAL]


def mode_label(status: str | None) -> str:
    """Human-friendly mode tag (used as a small subtitle in the UI)."""
    return {
        "crisis": "Crisis support",
        "high": "Active support",
        "conflict": "Reflection",
        "mixed": "Light self-care",
        "stable": "Maintain",
    }[_mode_key(status)]
