"""Compatibility entrypoint for stage-7 recursive rollout evaluation."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/recursive_rollout.py")


if __name__ == "__main__":
    raise SystemExit(main())
