"""Imports the merian package of the build tree in build/, for `import merian` here."""

import importlib.util
import sys
from pathlib import Path

_PACKAGE = (
    Path(__file__).resolve().parent / "build" / "src" / "merian-python" / "merian"
)
if not (_PACKAGE / "__init__.py").is_file():
    raise ImportError(
        f"no merian package in {_PACKAGE}: run `meson setup build` and "
        "`meson compile -C build`"
    )
_spec = importlib.util.spec_from_file_location(
    __name__, _PACKAGE / "__init__.py", submodule_search_locations=[str(_PACKAGE)]
)
_package = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _package
_spec.loader.exec_module(_package)
