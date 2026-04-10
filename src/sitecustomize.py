"""Runtime compatibility hooks loaded automatically by Python's site module.

This project uses Great Expectations, which still imports the deprecated
``sre_constants`` and ``sre_parse`` modules. Python exposes the supported
implementations through ``re._constants`` and ``re._parser``; we alias those
module names here so Great Expectations keeps working without emitting
deprecation warnings or forcing us to patch third-party code in-place.
"""

from __future__ import annotations

import re
import sys


def _alias_regex_compat_modules() -> None:
    if hasattr(re, "_constants"):
        sys.modules.setdefault("sre_constants", re._constants)
    if hasattr(re, "_parser"):
        sys.modules.setdefault("sre_parse", re._parser)


_alias_regex_compat_modules()
