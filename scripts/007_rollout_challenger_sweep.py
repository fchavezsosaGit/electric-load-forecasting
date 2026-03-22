"""Compatibility entrypoint for Stage-7 rollout challenger sweeps."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/rollout_challenger_sweep.py")


if __name__ == "__main__":
    raise SystemExit(main())
