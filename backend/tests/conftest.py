import os
import sys


def _ensure_project_root_on_syspath() -> None:
    # When running `cd backend && pytest`, allow importing `backend.app` by
    # making the project root importable.
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_syspath()
