from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import DEFAULT_DATA_PATH, catalog_stats, load_catalog
from .site import build_site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="All-In Post-Training panorama toolkit")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to panorama JSON data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate the panorama catalog")
    subparsers.add_parser("stats", help="Print catalog statistics")

    build_parser = subparsers.add_parser("build", help="Build the static panorama site")
    build_parser.add_argument("--out", default="site", help="Output directory")

    args = parser.parse_args(argv)
    data_path = Path(args.data)

    if args.command == "validate":
        load_catalog(data_path)
        print(f"ok: {data_path} is valid")
        return 0

    if args.command == "stats":
        stats = catalog_stats(load_catalog(data_path))
        print(
            "tracks={tracks} references={references} nodes={nodes} edges={edges} tags={tags}".format(
                **stats.__dict__
            )
        )
        return 0

    if args.command == "build":
        output = build_site(data_path, Path(args.out))
        print(f"built: {output}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

