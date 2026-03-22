"""Compatibility entrypoint for current validation snapshot generation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "tooling/write_validation_snapshot.py")


if __name__ == "__main__":
    raise SystemExit(main())
