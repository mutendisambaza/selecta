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


def _not_implemented(name: str) -> int:
    print(f"selecta {name}: not implemented yet")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 0
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
