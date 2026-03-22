"""Compatibility entrypoint for Stage-8 horizon-curve characterization."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/horizon_curve.py")


if __name__ == "__main__":
    raise SystemExit(main())
