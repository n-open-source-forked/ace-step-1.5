"""Load and validate song specification JSON, applying sensible defaults."""

import json
from pathlib import Path
from typing import Any, Dict


# Defaults applied when the spec omits generation parameters.
_GENERATION_DEFAULTS: Dict[str, Any] = {
    "duration": 120,
    "batch_size": 1,
    "inference_steps": 8,
    "guidance_scale": 7.0,
    "audio_format": "mp3",
    "mp3_bitrate": "192k",
    "seed": -1,
    "fade_out_duration": 2.0,
}

_OUTPUT_DEFAULTS: Dict[str, Any] = {
    "song_id": "song-00001",
    "output_dir": "output/tamil_songs",
}


class SongSpec:
    """Validated song specification backed by a plain dict."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    # -- Top-level sections --------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.data["metadata"]

    @property
    def music(self) -> Dict[str, Any]:
        return self.data.get("music", {})

    @property
    def lyrics(self) -> Dict[str, Any]:
        return self.data["lyrics"]

    @property
    def generation(self) -> Dict[str, Any]:
        return self.data.get("generation", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.data.get("output", {})

    # -- Convenience accessors -----------------------------------------------
    @property
    def title(self) -> str:
        m = self.metadata
        return m.get("title_en") or m.get("title_ta", "untitled")

    @property
    def song_id(self) -> str:
        return self.output.get("song_id", _OUTPUT_DEFAULTS["song_id"])

    @property
    def output_dir(self) -> str:
        return self.output.get("output_dir", _OUTPUT_DEFAULTS["output_dir"])


def load_song_spec(path: str) -> SongSpec:
    """Load a song spec JSON file, validate required fields, apply defaults.

    Required fields:
    - ``metadata`` (dict) with at least ``title_en`` or ``title_ta``
    - ``lyrics.sections`` (non-empty list)

    Returns a :class:`SongSpec` instance.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data: Dict[str, Any] = json.loads(raw)

    # --- Validate required fields ---
    if "metadata" not in data or not isinstance(data["metadata"], dict):
        raise ValueError("Song spec must contain a 'metadata' object")

    md = data["metadata"]
    if not md.get("title_en") and not md.get("title_ta"):
        raise ValueError("metadata must contain at least 'title_en' or 'title_ta'")

    if "lyrics" not in data or not isinstance(data["lyrics"], dict):
        raise ValueError("Song spec must contain a 'lyrics' object")

    sections = data["lyrics"].get("sections")
    if not sections or not isinstance(sections, list):
        raise ValueError("lyrics.sections must be a non-empty list")

    # --- Apply generation defaults ---
    gen = data.setdefault("generation", {})
    for key, default in _GENERATION_DEFAULTS.items():
        gen.setdefault(key, default)

    # --- Apply output defaults ---
    out = data.setdefault("output", {})
    for key, default in _OUTPUT_DEFAULTS.items():
        out.setdefault(key, default)

    return SongSpec(data)
