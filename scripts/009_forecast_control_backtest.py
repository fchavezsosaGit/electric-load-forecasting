"""Compatibility entrypoint for Stage-10 forecast-control backtesting."""

from __future__ import annotations

from _compat import load_into_globals

load_into_globals(globals(), "modeling/forecast_control_backtest.py")


if __name__ == "__main__":
    raise SystemExit(main())
