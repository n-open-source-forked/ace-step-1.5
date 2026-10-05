"""Candidate worker: one request JSON -> N seeded takes, each with audio + LRC.

Called as a subprocess by tamil-music-factory's music producer agent::

    python -m agents.tamil_composer.worker request.json --out ROUND_DIR

The request is a plain JSON contract (``version: 1``) so the caller never has
to import ACE-Step.  Models are initialised once and every seed is generated
sequentially (batch size 1) to stay within 12 GB of VRAM.

Output layout::

    ROUND_DIR/
        worker-result.json           # summary of all candidates
        seed-<seed>/
            audio.<fmt>
            lyrics.lrc               # sentence-level LRC from DiT cross-attention
            timestamps.json          # sentences with confidence + word timings
            candidate.json           # seed, LM metadata, timings, errors

The worker is resumable: a seed whose ``candidate.json`` reports success is
skipped on re-run.
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

REQUEST_VERSION = 1

# Request keys passed straight through to GenerationParams.
_PARAM_KEYS = (
    "caption",
    "lyrics",
    "vocal_language",
    "duration",
    "bpm",
    "keyscale",
    "timesignature",
    "inference_steps",
    "guidance_scale",
    "shift",
    "thinking",
    "lm_temperature",
    "fade_in_duration",
    "fade_out_duration",
    "task_type",
    "src_audio",
    "repainting_start",
    "repainting_end",
    "repaint_mode",
    "repaint_strength",
    "repaint_wav_crossfade_sec",
    "reference_audio",
    "audio_cover_strength",
)

_DEFAULTS: Dict[str, Any] = {
    "vocal_language": "ta",
    "duration": 180,
    "bpm": None,
    "keyscale": "",
    "timesignature": "",
    "inference_steps": 8,
    "guidance_scale": 7.0,
    "shift": 3.0,
    "thinking": True,
    "lm_temperature": 0.85,
    "fade_in_duration": 0.0,
    "fade_out_duration": 2.0,
    "seeds": [-1],
    "audio_format": "mp3",
    "mp3_bitrate": "320k",
    "dit_model": "acestep-v15-turbo",
    "lm_model": "acestep-5Hz-lm-1.7B",
    "lm_backend": "vllm",
    "offload_to_cpu": True,
    "extract_lrc": True,
    # Repaint (regenerate one time range of an existing take, 3-90 s).
    "task_type": "text2music",
    "src_audio": None,
    "repainting_start": 0.0,
    "repainting_end": -1,
    "repaint_mode": "balanced",
    "repaint_strength": 0.5,
    "repaint_wav_crossfade_sec": 0.0,
    # Cover (re-sing src_audio's melody with new lyrics/caption) and reference
    # audio (borrow timbre/performance style; works with any task).
    "reference_audio": None,
    "audio_cover_strength": 0.5,
    # Music only: ACE-Step gets "[Instrumental]" as lyrics (its `instrumental`
    # flag is not consumed by the DiT); request lyrics stay for the caller.
    "instrumental": False,
}

_SRC_AUDIO_TASKS = ("repaint", "cover")


def load_request(path: Path) -> Dict[str, Any]:
    """Load a request JSON, check its version and fill defaults."""
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if version != REQUEST_VERSION:
        raise ValueError(f"Unsupported request version {version!r}; expected {REQUEST_VERSION}")
    if not data.get("caption"):
        raise ValueError("request.caption is required")
    if not data.get("lyrics"):
        raise ValueError("request.lyrics is required (use '[Instrumental]' for no vocals)")
    for key, default in _DEFAULTS.items():
        data.setdefault(key, default)
    if not isinstance(data["seeds"], list) or not data["seeds"]:
        raise ValueError("request.seeds must be a non-empty list of integers")
    task = data["task_type"]
    if task in _SRC_AUDIO_TASKS:
        src = data.get("src_audio")
        if not src or not Path(src).is_file():
            raise ValueError(f"{task} needs an existing src_audio, got {src!r}")
        # ACE-Step skips the LM for these tasks; don't spend VRAM loading it.
        data["thinking"] = False
    elif task != "text2music":
        raise ValueError(f"Unsupported task_type {task!r}")
    ref = data.get("reference_audio")
    if ref and not Path(ref).is_file():
        raise ValueError(f"reference_audio not found: {ref!r}")
    return data


def _format_ts(seconds: float) -> str:
    mins = int(seconds // 60)
    return f"{mins:02d}:{seconds % 60:05.2f}"


def _group_words(tokens: List[Any], tokenizer: Any) -> List[Dict[str, Any]]:
    """Group sub-word tokens into whitespace-delimited words.

    Tamil characters often span several byte-level tokens, so word text is
    decoded from the grouped token ids rather than by joining token strings.
    """
    words: List[Dict[str, Any]] = []
    group: List[Any] = []

    def flush() -> None:
        if not group:
            return
        ids = [t.token_id for t in group]
        try:
            text = tokenizer.decode(ids).strip()
        except Exception:
            text = "".join(t.text for t in group).strip()
        if text:
            words.append({
                "text": text,
                "start": round(group[0].start, 3),
                "end": round(group[-1].end, 3),
                "probability": round(sum(t.probability for t in group) / len(group), 4),
            })
        group.clear()

    for tok in tokens:
        text = tok.text or ""
        if text.startswith(" ") and text.strip():
            # Leading-space token begins a new word.
            flush()
            group.append(tok)
        elif not text.strip() or "\n" in text:
            flush()
        else:
            group.append(tok)
    flush()
    return words


def _timestamps_payload(ts_result: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    sentences = []
    for s in ts_result.get("sentence_timestamps") or []:
        sentences.append({
            "text": s.text.strip(),
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "confidence": round(float(s.confidence), 4),
            "words": _group_words(s.tokens or [], tokenizer),
        })
    return {"version": 1, "sentences": sentences}


class CandidateWorker:
    """Holds initialised ACE-Step handlers across candidates."""

    def __init__(self, project_root: Path, request: Dict[str, Any]) -> None:
        self.project_root = project_root
        self.request = request
        self.dit_handler: Any = None
        self.llm_handler: Any = None
        self.llm_status = ""

    def initialise(self) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        req = self.request
        logger.info("Initialising DiT ({}) ...", req["dit_model"])
        self.dit_handler = AceStepHandler()
        status, ok = self.dit_handler.initialize_service(
            project_root=str(self.project_root),
            config_path=req["dit_model"],
            device="cuda",
            offload_to_cpu=req["offload_to_cpu"],
        )
        if not ok:
            raise RuntimeError(f"DiT init failed: {status}")

        if req["thinking"] and req.get("lm_model"):
            logger.info("Initialising LM ({}, backend={}) ...", req["lm_model"], req["lm_backend"])
            llm = LLMHandler()
            status, ok = llm.initialize(
                checkpoint_dir=str(self.project_root / "checkpoints"),
                lm_model_path=req["lm_model"],
                backend=req["lm_backend"],
                device="cuda",
                offload_to_cpu=req["offload_to_cpu"],
            )
            self.llm_status = status
            if ok:
                self.llm_handler = llm
            else:
                logger.warning("LM init failed ({}); continuing without LM", status)

    def generate(self, seed: int, seed_dir: Path) -> Dict[str, Any]:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        req = self.request
        params = GenerationParams(**{k: req[k] for k in _PARAM_KEYS})
        if req["instrumental"]:
            params.lyrics = "[Instrumental]"
        params.seed = seed
        if self.llm_handler is None:
            params.thinking = False

        config = GenerationConfig(
            batch_size=1,
            use_random_seed=seed < 0,
            seeds=[seed] if seed >= 0 else None,
            audio_format=req["audio_format"],
            mp3_bitrate=req["mp3_bitrate"],
        )

        tmp_dir = seed_dir / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result = generate_music(
            dit_handler=self.dit_handler,
            llm_handler=self.llm_handler,
            params=params,
            config=config,
            save_dir=str(tmp_dir),
        )
        gen_seconds = time.time() - started

        if not result.success or not result.audios:
            raise RuntimeError(result.error or "No audio produced")

        audio = result.audios[0]
        actual_seed = audio.get("params", {}).get("seed", seed)
        src = Path(audio.get("path", ""))
        if not src.is_file():
            raise RuntimeError(f"Generated audio missing: {src}")
        audio_name = f"audio.{req['audio_format']}"
        shutil.move(str(src), seed_dir / audio_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        extra = result.extra_outputs or {}
        pred_latents = extra.get("pred_latents")
        duration = (
            pred_latents.shape[1] / 25.0 if pred_latents is not None else float(req["duration"])
        )
        _write_repaint_sidecar(seed_dir, audio_name, pred_latents, actual_seed)

        lrc_ok, lrc_error = False, None
        if req["extract_lrc"]:
            lrc_ok, lrc_error = self._extract_lrc(extra, actual_seed, duration, seed_dir)

        lm_meta = extra.get("lm_metadata") or {}
        return {
            "seed": actual_seed,
            "audio_file": audio_name,
            "duration_seconds": round(duration, 2),
            "lrc_extracted": lrc_ok,
            "lrc_error": lrc_error,
            "lm_metadata": _jsonable(lm_meta),
            "time_costs": _jsonable(extra.get("time_costs") or {}),
            "generation_seconds": round(gen_seconds, 2),
        }

    def _extract_lrc(
        self, extra: Dict[str, Any], seed: int, duration: float, seed_dir: Path
    ) -> "tuple[bool, Optional[str]]":
        needed = ["pred_latents", "encoder_hidden_states", "encoder_attention_mask",
                  "context_latents", "lyric_token_idss"]
        missing = [k for k in needed if extra.get(k) is None]
        if missing:
            return False, f"missing tensors: {missing}"
        try:
            ts = self.dit_handler.get_lyric_timestamp(
                pred_latent=extra["pred_latents"][0:1],
                encoder_hidden_states=extra["encoder_hidden_states"][0:1],
                encoder_attention_mask=extra["encoder_attention_mask"][0:1],
                context_latents=extra["context_latents"][0:1],
                lyric_token_ids=extra["lyric_token_idss"][0:1],
                total_duration_seconds=float(duration),
                vocal_language=self.request["vocal_language"],
                inference_steps=int(self.request["inference_steps"]),
                seed=seed if seed >= 0 else 42,
            )
        except Exception as exc:  # alignment is best-effort
            return False, f"{type(exc).__name__}: {exc}"
        if not ts.get("success"):
            return False, ts.get("error") or "unknown alignment error"

        (seed_dir / "lyrics.lrc").write_text(ts.get("lrc_text", ""), encoding="utf-8")
        payload = _timestamps_payload(ts, self.dit_handler.text_tokenizer)
        (seed_dir / "timestamps.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True, None


def _write_repaint_sidecar(seed_dir: Path, audio_name: str, pred_latents: Any, seed: int) -> None:
    """Cache final latents beside the audio, in the sidecar format ACE-Step's
    repaint path looks for (``<audio>.json`` + ``.repaint_latents.npy``), so a
    later repaint of this take skips the lossy decode/re-encode cycle."""
    if pred_latents is None:
        return
    import numpy as np

    stem = Path(audio_name).stem
    latent_name = f"{stem}.repaint_latents.npy"
    np.save(seed_dir / latent_name, pred_latents[0].detach().cpu().float().numpy())
    (seed_dir / f"{stem}.json").write_text(
        json.dumps({"seed": seed, "repaint_source_latents_file": latent_name}), encoding="utf-8"
    )


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of LM metadata / timing dicts to JSON types."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def run(request_path: Path, out_dir: Path, project_root: Path) -> Dict[str, Any]:
    request = load_request(request_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    candidates: List[Dict[str, Any]] = []
    for seed in request["seeds"]:
        seed_dir = out_dir / f"seed-{seed}"
        cand_file = seed_dir / "candidate.json"
        if seed >= 0 and cand_file.is_file():
            existing = json.loads(cand_file.read_text(encoding="utf-8"))
            if existing.get("success"):
                logger.info("Seed {} already generated; skipping", seed)
                candidates.append(existing)
                continue
        pending.append(seed)

    worker = CandidateWorker(project_root, request)
    init_error = None
    if pending:
        try:
            worker.initialise()
        except Exception as exc:
            init_error = f"{type(exc).__name__}: {exc}"
            logger.error("Initialisation failed: {}", init_error)

    for index, seed in enumerate(pending if init_error is None else [], 1):
        logger.info("Candidate {}/{} (seed={}) ...", index, len(pending), seed)
        seed_dir = out_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            info = worker.generate(seed, seed_dir)
            if seed < 0:
                # Random seed: rename the folder to the seed actually used.
                final_dir = out_dir / f"seed-{info['seed']}"
                if final_dir != seed_dir:
                    shutil.rmtree(final_dir, ignore_errors=True)
                    seed_dir.rename(final_dir)
                    seed_dir = final_dir
            cand = {"success": True, "dir": seed_dir.name, **info}
        except Exception as exc:
            logger.error("Seed {} failed: {}", seed, exc)
            cand = {
                "success": False,
                "dir": seed_dir.name,
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        (seed_dir / "candidate.json").write_text(
            json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        candidates.append(cand)

    if init_error is not None:
        candidates.extend(
            {"success": False, "dir": f"seed-{seed}", "seed": seed, "error": init_error}
            for seed in pending
        )

    summary = {
        "version": 1,
        "request": str(request_path),
        "init_error": init_error,
        "lm_status": worker.llm_status,
        "succeeded": sum(1 for c in candidates if c.get("success")),
        "failed": sum(1 for c in candidates if not c.get("success")),
        "candidates": candidates,
    }
    (out_dir / "worker-result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded ACE-Step candidates with LRC")
    parser.add_argument("request", type=Path, help="Request JSON (version 1)")
    parser.add_argument("--out", type=Path, required=True, help="Output round directory")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="ACE-Step project root (default: auto-detect)",
    )
    args = parser.parse_args()

    summary = run(args.request, args.out, args.project_root)
    logger.info("Done: {} succeeded, {} failed", summary["succeeded"], summary["failed"])
    sys.exit(0 if summary["succeeded"] else 1)


if __name__ == "__main__":
    main()
