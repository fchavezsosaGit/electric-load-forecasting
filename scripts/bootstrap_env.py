"""Compatibility entrypoint for environment bootstrapping."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "tooling/bootstrap_env.py")


if __name__ == "__main__":
    raise SystemExit(main())
