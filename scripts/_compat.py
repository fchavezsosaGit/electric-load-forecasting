"""Helpers for loading canonical script implementations into wrapper modules."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


class _ModuleProxy:
    """Minimal sys.modules proxy exposing a wrapper namespace as __dict__."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace

    @property
    def __dict__(self) -> dict[str, Any]:
        return self._namespace


def load_into_globals(module_globals: dict[str, Any], relative_impl_path: str) -> None:
    """Execute an implementation file in the wrapper module namespace.

    This preserves compatibility with tests that monkeypatch module globals on the
    numbered wrapper scripts while allowing the canonical implementation to live in
    a nested package.
    """

    wrapper_path = Path(str(module_globals["__file__"])).resolve()
    impl_path = wrapper_path.parent / Path(relative_impl_path)

    original_name = str(module_globals.get("__name__", "__main__"))
    original_file = str(module_globals.get("__file__", wrapper_path))
    exec_name = original_name if original_name != "__main__" else "__compat_impl__"
    proxy_added = False

    if sys.modules.get(exec_name) is None:
        sys.modules[exec_name] = _ModuleProxy(module_globals)
        proxy_added = True

    module_globals["__name__"] = exec_name
    module_globals["__file__"] = str(impl_path)
    try:
        source = impl_path.read_text(encoding="utf-8")
        exec(compile(source, str(impl_path), "exec"), module_globals)
    finally:
        module_globals["__name__"] = original_name
        module_globals["__file__"] = original_file
        if proxy_added:
            sys.modules.pop(exec_name, None)
