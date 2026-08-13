"""Process lyrics from the song spec for ACE-Step consumption.

Marker conventions defined in the song spec:

* ``~`` (tilde) — binds words together (sung as one phrase).
  Stripped to a space for ACE-Step.
* ``|`` (pipe) — syllable break hint.
  Stripped entirely for ACE-Step.
* ``use_tamil_script`` — selects ``lines_ta`` or ``lines_en`` to send to
  ACE-Step, falling back to the other if the chosen one is empty.
"""

from typing import Any, Dict, List, Tuple


def _select_lines(section: Dict[str, Any], use_tamil: bool) -> List[str]:
    """Pick the appropriate line list from a section, with fallback."""
    if use_tamil:
        lines = section.get("lines_ta") or section.get("lines_en") or []
    else:
        lines = section.get("lines_en") or section.get("lines_ta") or []
    return lines


def _strip_markers(line: str) -> str:
    """Strip performance markers for ACE-Step: ``~`` → space, ``|`` → nothing."""
    return line.replace("~", " ").replace("|", "")


def build_acestep_lyrics(lyrics_spec: Dict[str, Any]) -> str:
    """Build lyrics text formatted for ACE-Step.

    Each section is wrapped with a ``[Tag]`` header line followed by the
    cleaned lyric lines (one per line).  Sections are separated by a blank
    line.

    Parameters
    ----------
    lyrics_spec:
        The ``lyrics`` dict from a :class:`SongSpec`.

    Returns
    -------
    str
        Multi-line lyrics string ready for ACE-Step's ``lyrics`` parameter.
    """
    use_tamil: bool = lyrics_spec.get("use_tamil_script", False)
    sections: List[Dict[str, Any]] = lyrics_spec.get("sections", [])

    blocks: List[str] = []
    for section in sections:
        tag = section.get("tag", "Verse")
        lines = _select_lines(section, use_tamil)
        cleaned = [_strip_markers(l) for l in lines]
        block = f"[{tag}]\n" + "\n".join(cleaned)
        blocks.append(block)

    return "\n\n".join(blocks)


def build_annotated_lyrics(
    lyrics_spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return sections with markers preserved for future LRC post-processing.

    Each element is a dict::

        {"tag": "Verse 1", "lines": ["Kai|yil midha|kkum kanavaa~nee", ...]}

    Parameters
    ----------
    lyrics_spec:
        The ``lyrics`` dict from a :class:`SongSpec`.

    Returns
    -------
    list[dict]
        List of annotated section dicts.
    """
    use_tamil: bool = lyrics_spec.get("use_tamil_script", False)
    sections: List[Dict[str, Any]] = lyrics_spec.get("sections", [])

    result: List[Dict[str, Any]] = []
    for section in sections:
        tag = section.get("tag", "Verse")
        voice = section.get("voice")
        lines = _select_lines(section, use_tamil)
        entry: Dict[str, Any] = {"tag": tag, "lines": list(lines)}
        if voice:
            entry["voice"] = voice
        result.append(entry)

    return result
