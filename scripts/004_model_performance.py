"""Compatibility entrypoint for stage-5 model performance workflows."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/model_performance.py")


if __name__ == "__main__":
    raise SystemExit(main())
