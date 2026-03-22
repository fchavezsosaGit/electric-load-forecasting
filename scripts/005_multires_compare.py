"""Compatibility entrypoint for stage-6 multiresolution comparison."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/multires_compare.py")


if __name__ == "__main__":
    raise SystemExit(main())
