"""CLI entry point: python -m agents.tamil_composer spec.json [--project-root PATH]"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tamil Music Composer Agent — generate MP3 + LRC from a song spec JSON",
    )
    parser.add_argument(
        "spec",
        help="Path to the song specification JSON file",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="ACE-Step project root directory (default: auto-detect)",
    )
    args = parser.parse_args()

    # Defer heavy imports so --help works without the full venv.
    from loguru import logger
    from agents.tamil_composer.engine import compose

    result = compose(args.spec, project_root=args.project_root)

    if result["success"]:
        logger.info("=== Composition complete ===")
        logger.info("Song:   {} ({})", result["title"], result["song_id"])
        logger.info("Output: {}", result["output_path"])
        for f in result["files"]:
            logger.info("  - {}", f)
    else:
        logger.error("=== Composition failed ===")
        logger.error("Error: {}", result["error"])
        sys.exit(1)

    # Also dump the result summary to stdout as JSON for scripting.
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
