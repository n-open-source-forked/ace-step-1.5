"""Core orchestrator: load spec -> init handlers -> generate -> LRC -> package."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def _build_word_lrc(token_timestamps: List[Any]) -> str:
    """Build word-level LRC from per-token timestamps.

    Groups consecutive non-whitespace tokens into words and emits one
    ``[MM:SS.xx]word`` line per word.  Structural tags (tokens starting with
    ``[``) are emitted on their own line.
    """
    lines: List[str] = []
    word_tokens: List[Any] = []

    def _flush_word() -> None:
        if not word_tokens:
            return
        start = word_tokens[0].start
        text = "".join(t.text for t in word_tokens).strip()
        if not text:
            return
        mins = int(start // 60)
        secs = start % 60
        lines.append(f"[{mins:02d}:{secs:05.2f}]{text}")

    for tok in token_timestamps:
        text = tok.text
        # Structural tag like [Verse 1] — flush and emit
        if text.startswith("["):
            _flush_word()
            word_tokens.clear()
            mins = int(tok.start // 60)
            secs = tok.start % 60
            lines.append(f"[{mins:02d}:{secs:05.2f}]{text}")
            continue
        # Whitespace or newline — word boundary
        if text.strip() == "" or "\n" in text:
            _flush_word()
            word_tokens.clear()
            continue
        word_tokens.append(tok)

    _flush_word()
    return "\n".join(lines)


def compose(spec_path: str, project_root: Optional[str] = None) -> Dict[str, Any]:
    """Run the full Tamil Composer pipeline.

    Steps:
        1. Load and validate the song spec.
        2. Initialise the ACE-Step DiT handler and (optionally) the LLM handler.
        3. Build caption and lyrics from the spec.
        4. Call ``generate_music()`` with appropriate params.
        5. Extract LRC timestamps from ``extra_outputs`` tensors.
        6. Package output into ``{output_dir}/{song_id}/``.

    Parameters
    ----------
    spec_path:
        Path to a song spec JSON file.
    project_root:
        ACE-Step project root (defaults to two levels above this file,
        i.e. the ``ace-step-1.5/`` directory).

    Returns
    -------
    dict
        Summary with keys ``success``, ``song_id``, ``output_path``, ``files``,
        ``generation_time``, and ``error`` (if any).
    """
    wall_start = time.time()

    # -- Resolve project root ------------------------------------------------
    if project_root is None:
        # agents/tamil_composer/engine.py -> agents/tamil_composer -> agents -> ace-step-1.5
        project_root = str(Path(__file__).resolve().parent.parent.parent)
    project_root_path = Path(project_root)

    # -- 1. Load spec --------------------------------------------------------
    from agents.tamil_composer.schema import load_song_spec

    logger.info("Loading song spec: {}", spec_path)
    spec = load_song_spec(spec_path)
    logger.info("Song: {} (id={})", spec.title, spec.song_id)

    # -- 2. Init handlers (mirrors test_tamil.py) ----------------------------
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    logger.info("Initialising DiT handler ...")
    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(project_root_path),
        config_path="acestep-v15-turbo",
        device="cuda",
        offload_to_cpu=True,
    )
    logger.info("DiT: {}", status)
    if not ok:
        return _error_result(spec, f"DiT init failed: {status}", wall_start)

    logger.info("Initialising LLM handler ...")
    llm_handler = LLMHandler()
    llm_status, llm_ok = llm_handler.initialize(
        checkpoint_dir=str(project_root_path / "checkpoints"),
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="vllm",
        device="cuda",
    )
    if not llm_ok:
        logger.warning("LLM init warning: {}. Continuing without LLM.", llm_status)
        llm_handler = None
    else:
        logger.info("LLM: {}", llm_status)

    # -- 3. Build caption + lyrics -------------------------------------------
    from agents.tamil_composer.caption_builder import build_caption
    from agents.tamil_composer.lyrics_processor import (
        build_acestep_lyrics,
        build_annotated_lyrics,
    )

    caption = build_caption(spec.music, sections=spec.lyrics.get("sections", []))
    lyrics_text = build_acestep_lyrics(spec.lyrics)
    annotated_lyrics = build_annotated_lyrics(spec.lyrics)
    logger.info("Caption: {}", caption)
    logger.debug("Lyrics:\n{}", lyrics_text)

    # -- 4. Generate music ---------------------------------------------------
    gen = spec.generation
    music = spec.music

    params = GenerationParams(
        caption=caption,
        lyrics=lyrics_text,
        duration=gen.get("duration", 120),
        vocal_language=spec.metadata.get("language", "ta"),
        inference_steps=gen.get("inference_steps", 8),
        guidance_scale=gen.get("guidance_scale", 7.0),
        thinking=True,
        seed=gen.get("seed", -1),
        fade_out_duration=gen.get("fade_out_duration", 2.0),
        bpm=music.get("bpm"),
        keyscale=music.get("key", ""),
        timesignature=str(music.get("time_signature", "")),
    )

    config = GenerationConfig(
        batch_size=gen.get("batch_size", 1),
        audio_format=gen.get("audio_format", "mp3"),
        mp3_bitrate=gen.get("mp3_bitrate", "192k"),
    )

    # Use a temp save dir; we'll rename/copy into the final layout.
    temp_save_dir = str(project_root_path / "output" / "_tamil_composer_tmp")
    os.makedirs(temp_save_dir, exist_ok=True)

    logger.info("Generating music (duration={}s) ...", params.duration)
    gen_start = time.time()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=temp_save_dir,
    )
    gen_elapsed = time.time() - gen_start
    logger.info("Generation took {:.1f}s", gen_elapsed)

    if not result.success:
        return _error_result(spec, f"Generation failed: {result.error}", wall_start)

    if not result.audios:
        return _error_result(spec, "No audio produced", wall_start)

    audio_info = result.audios[0]
    src_audio_path = audio_info.get("path", "")
    sample_rate = audio_info.get("sample_rate", 48000)
    audio_tensor = audio_info.get("tensor")
    actual_seed = audio_info.get("params", {}).get("seed", -1)

    # -- 5. Extract LRC timestamps -------------------------------------------
    lrc_text = ""
    word_lrc_text = ""
    extra = result.extra_outputs or {}
    pred_latents = extra.get("pred_latents")
    enc_hidden = extra.get("encoder_hidden_states")
    enc_mask = extra.get("encoder_attention_mask")
    ctx_latents = extra.get("context_latents")
    lyric_ids = extra.get("lyric_token_idss")

    if all(t is not None for t in [pred_latents, enc_hidden, enc_mask, ctx_latents, lyric_ids]):
        logger.info("Extracting LRC timestamps ...")
        if audio_tensor is not None:
            total_dur = audio_tensor.shape[-1] / sample_rate
        else:
            total_dur = params.duration

        try:
            ts_result = dit_handler.get_lyric_timestamp(
                pred_latent=pred_latents,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                context_latents=ctx_latents,
                lyric_token_ids=lyric_ids,
                total_duration_seconds=total_dur,
                vocal_language=params.vocal_language,
                inference_steps=params.inference_steps,
                seed=actual_seed if actual_seed >= 0 else 42,
            )
            if ts_result.get("success"):
                lrc_text = ts_result.get("lrc_text", "")
                logger.info("LRC (sentence-level) extracted ({} chars)", len(lrc_text))

                # Build word-level LRC from token timestamps for video sync.
                token_timestamps = ts_result.get("token_timestamps", [])
                if token_timestamps:
                    word_lrc_text = _build_word_lrc(token_timestamps)
                    logger.info("LRC (word-level) extracted ({} chars)", len(word_lrc_text))
            else:
                logger.warning("LRC extraction failed: {}", ts_result.get("error"))
        except Exception as exc:
            logger.warning("LRC extraction error (non-fatal): {}", exc)
    else:
        logger.warning(
            "Skipping LRC extraction: missing tensors in extra_outputs "
            "(save_memory_mode may be active)"
        )

    # -- 6. Package output ---------------------------------------------------
    output_base = Path(spec.output_dir)
    if not output_base.is_absolute():
        output_base = project_root_path / output_base
    song_dir = output_base / spec.song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    files_written = []

    # audio.mp3 (or whatever format)
    audio_ext = gen.get("audio_format", "mp3")
    dest_audio = song_dir / f"audio.{audio_ext}"
    if src_audio_path and os.path.isfile(src_audio_path):
        shutil.copy2(src_audio_path, dest_audio)
        files_written.append(str(dest_audio))
        logger.info("Audio: {}", dest_audio)
    else:
        logger.warning("No audio file to copy (src={})", src_audio_path)

    # lyrics.lrc (sentence-level)
    dest_lrc = song_dir / "lyrics.lrc"
    if lrc_text:
        dest_lrc.write_text(lrc_text, encoding="utf-8")
        files_written.append(str(dest_lrc))
        logger.info("LRC (sentence): {}", dest_lrc)
    else:
        logger.info("No LRC content to write")

    # lyrics_words.lrc (word-level, for video sync)
    dest_word_lrc = song_dir / "lyrics_words.lrc"
    if word_lrc_text:
        dest_word_lrc.write_text(word_lrc_text, encoding="utf-8")
        files_written.append(str(dest_word_lrc))
        logger.info("LRC (word): {}", dest_word_lrc)

    # metadata.json (tamil-music-factory compatible)
    metadata_out = {
        "title": spec.metadata.get("title_en", ""),
        "title_ta": spec.metadata.get("title_ta", ""),
        "artist": spec.metadata.get("artist", "AI Composer"),
        "language": spec.metadata.get("language", "ta"),
        "genre": music.get("genre", ""),
        "mood": music.get("mood", []),
        "bpm": music.get("bpm"),
        "key": music.get("key", ""),
        "duration_seconds": (
            audio_tensor.shape[-1] / sample_rate if audio_tensor is not None else params.duration
        ),
        "disclosure": spec.metadata.get("disclosure", "AI-generated music using ACE-Step 1.5"),
        "has_lrc": bool(lrc_text),
        "has_word_lrc": bool(word_lrc_text),
        "audio_file": f"audio.{audio_ext}",
        "lrc_file": "lyrics.lrc" if lrc_text else None,
        "word_lrc_file": "lyrics_words.lrc" if word_lrc_text else None,
    }
    dest_meta = song_dir / "metadata.json"
    dest_meta.write_text(json.dumps(metadata_out, indent=2, ensure_ascii=False), encoding="utf-8")
    files_written.append(str(dest_meta))
    logger.info("Metadata: {}", dest_meta)

    # generation_report.json (reproducibility)
    report = {
        "song_id": spec.song_id,
        "spec_path": str(Path(spec_path).resolve()),
        "caption": caption,
        "lyrics_text": lyrics_text,
        "annotated_lyrics": annotated_lyrics,
        "generation_params": params.to_dict(),
        "generation_config": config.to_dict(),
        "actual_seed": actual_seed,
        "generation_time_seconds": round(gen_elapsed, 2),
        "total_time_seconds": round(time.time() - wall_start, 2),
        "lrc_extracted": bool(lrc_text),
        "output_files": [os.path.basename(f) for f in files_written],
    }
    dest_report = song_dir / "generation_report.json"
    dest_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    files_written.append(str(dest_report))
    logger.info("Report: {}", dest_report)

    # Clean up temp dir
    try:
        shutil.rmtree(temp_save_dir, ignore_errors=True)
    except Exception:
        pass

    total_elapsed = time.time() - wall_start
    logger.info("Done in {:.1f}s. Output: {}", total_elapsed, song_dir)

    return {
        "success": True,
        "song_id": spec.song_id,
        "title": spec.title,
        "output_path": str(song_dir),
        "files": files_written,
        "generation_time": round(gen_elapsed, 2),
        "total_time": round(total_elapsed, 2),
        "error": None,
    }


def _error_result(
    spec: Any, error: str, wall_start: float
) -> Dict[str, Any]:
    """Build a failure result dict."""
    logger.error("Compose failed: {}", error)
    return {
        "success": False,
        "song_id": getattr(spec, "song_id", "unknown"),
        "title": getattr(spec, "title", "unknown"),
        "output_path": None,
        "files": [],
        "generation_time": 0,
        "total_time": round(time.time() - wall_start, 2),
        "error": error,
    }
