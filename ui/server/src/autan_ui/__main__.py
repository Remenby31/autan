import argparse
import logging
from pathlib import Path

from autan_ui.app import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="autan-ui", description="Serve the Autan web UI locally.")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="run directories")
    parser.add_argument(
        "--meshes", type=Path, default=None, help="MeshInfo files (*.meshinfo.json) to plan against"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    serve(args.runs, args.host, args.port, open_browser=not args.no_open, meshes=args.meshes)


if __name__ == "__main__":
    main()
