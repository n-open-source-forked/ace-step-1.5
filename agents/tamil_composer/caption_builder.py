"""Build an ACE-Step caption string from structured music parameters."""

from typing import Any, Dict, List, Optional


def _detect_duet(sections: List[Dict[str, Any]]) -> bool:
    """Return True if any section has an explicit ``voice`` field."""
    voices = {s.get("voice", "").lower() for s in sections if s.get("voice")}
    return len(voices) > 1 or "both" in voices


def build_caption(
    music: Dict[str, Any],
    language: str = "tamil",
    sections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble a caption for ACE-Step from the ``music`` section of a song spec.

    Ordering:
        mood + language + genre  ->  vocal style (duet-aware)  ->  instruments  ->  reference style

    For duet songs (sections with ``voice`` fields), the vocal style is
    augmented with "male and female duet" if not already present.

    Parameters
    ----------
    music:
        The ``music`` dict from a :class:`SongSpec`.
    language:
        Language label inserted into the caption (default ``"tamil"``).
    sections:
        Optional lyrics sections list; used to auto-detect duet songs.

    Returns
    -------
    str
        A single-line caption string suitable for ACE-Step's ``caption`` parameter.
    """
    parts: List[str] = []

    # 1. mood + language + genre
    mood: List[str] = music.get("mood", [])
    genre: str = music.get("genre", "")
    if mood:
        parts.append(f"{' and '.join(mood)} {language} {genre}".strip())
    elif genre:
        parts.append(f"{language} {genre}".strip())

    # 2. vocal style (duet-aware)
    vocal_style: str = music.get("vocal_style", "")
    is_duet = sections and _detect_duet(sections)
    if is_duet and "duet" not in vocal_style.lower():
        vocal_style = f"male and female duet, {vocal_style}" if vocal_style else "male and female duet"
    if vocal_style:
        parts.append(f"{vocal_style} vocals singing in {language} language")

    # 3. instruments (prefixed with orchestration hint if present)
    instruments: List[str] = music.get("instruments", [])
    orchestration: str = music.get("orchestration", "")
    if instruments:
        inst_str = ", ".join(instruments)
        if orchestration:
            inst_str = f"{orchestration} {inst_str}"
        parts.append(inst_str)

    # 4. reference style
    reference_style: str = music.get("reference_style", "")
    if reference_style:
        parts.append(reference_style)

    return ", ".join(parts)
