"""Compatibility entrypoint for the integrated visualization report."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "tooling/write_visualization_report.py")


if __name__ == "__main__":
    raise SystemExit(main())
