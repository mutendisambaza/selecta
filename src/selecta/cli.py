"""Selecta command-line entry point.

Subcommands mirror the two-phase pipeline:
  analyze   extract + cache per-track MIR features
  cues      export Rekordbox phrase cues (rekordbox.xml)
  sequence  build a journey-shaped set plan (markdown + json)

The subcommands are stubs at M0; each milestone wires its behaviour in.
"""
from __future__ import annotations

import argparse
import sys


def cmd_analyze(args) -> int:
    from selecta.pipeline import analyze_folder
    from selecta.store.db import FeatureStore

    store = FeatureStore(args.db)
    try:
        analysed, skipped, failed = analyze_folder(args.folder, store)
    finally:
        store.close()
    print(f"\nanalyze: {analysed} analysed, {skipped} skipped, {failed} failed  (db: {args.db})")
    return 0


def cmd_cues(args) -> int:
    from selecta.export.rekordbox import export_cues
    from selecta.pipeline import analyze_folder
    from selecta.store.db import FeatureStore

    store = FeatureStore(args.db)
    try:
        analysed, skipped, failed = analyze_folder(args.folder, store)
        tracks = store.all()
    finally:
        store.close()
    export_cues(tracks, args.out)
    print(f"\ncues: wrote {len(tracks)} tracks -> {args.out}  "
          f"(analysed {analysed}, skipped {skipped}, failed {failed})")
    if failed:
        print(f"  note: {failed} file(s) failed to analyse and are not in the export")
    return 0


def cmd_sequence(args) -> int:
    print("selecta sequence: not implemented yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="selecta", description=__doc__)
    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("analyze", help="extract + cache per-track MIR features")
    a.add_argument("folder")
    a.add_argument("--db", default="selecta.sqlite")

    c = sub.add_parser("cues", help="export Rekordbox phrase cues")
    c.add_argument("folder")
    c.add_argument("--db", default="selecta.sqlite")
    c.add_argument("--out", default="rekordbox.xml")

    s = sub.add_parser("sequence", help="build a journey-shaped set plan")
    s.add_argument("folder")
    s.add_argument("--db", default="selecta.sqlite")
    s.add_argument("--arc", default="experiential")
    s.add_argument("--out", default="set-plan.md")

    return p


_HANDLERS = {"analyze": cmd_analyze, "cues": cmd_cues, "sequence": cmd_sequence}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
