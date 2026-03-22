"""Compatibility entrypoint for notebook smoke validation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "tooling/validate_notebooks.py")


if __name__ == "__main__":
    raise SystemExit(main())
