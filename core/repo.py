# ----------------------------------------------------------------
# repo.py  —  makes the repository's folders importable and locates scripts.
#
# The code is split into folders (core/, search_amplitude/, search_fourier/,
# diagnostics/, studies/, reporting/, tools/), but the modules import each
# other by plain name (e.g. `import common`, `import search_core`). Every
# entry-point script therefore starts with
#
#     sys.path.insert(0, <repo>/core)
#     import repo
#
# and importing this module puts all code folders on sys.path, so any
# module can be imported from anywhere in the repository.
#
# Scripts that launch other scripts (run_all.py, studies/*.py) use
# `repo.script("encoding_search.py")` to get the script's full path instead
# of assuming that everything lives in one folder.
#
# All results are written relative to the CURRENT WORKING DIRECTORY, so
# run every command from the repository root (see README.md).
# ----------------------------------------------------------------
import os
import sys

# repository root = the folder that contains core/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# every folder that contains importable code, in search order
CODE_DIRS = ["core", "search_amplitude", "search_fourier", "diagnostics",
             "studies", "reporting", "tools"]

for _d in reversed(CODE_DIRS):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def script(name):
    """Absolute path of a script (e.g. 'encoding_search.py') in any code folder."""
    for d in CODE_DIRS:
        p = os.path.join(ROOT, d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in any of {CODE_DIRS}")
