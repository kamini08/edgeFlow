import importlib.util
import os
import sys


def _ensure_local_parser_package_shadowing() -> None:
    """Force local project 'parser' package to shadow stdlib 'parser'.

    Python ships a built-in C-extension module named 'parser'. To ensure our
    local package named 'parser' is imported by tests, we explicitly load it
    from the repository path and inject it into sys.modules.
    """

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)
    pkg_init = os.path.join(root, 'parser', '__init__.py')
    if not os.path.isfile(pkg_init):
        return
    spec = importlib.util.spec_from_file_location('parser', pkg_init)
    if spec and spec.loader:  # type: ignore[truthy-bool]
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        sys.modules['parser'] = mod


_ensure_local_parser_package_shadowing()
