"""
Auto-import all submodules under tasks.tracking so that gym envs register via side-effects.
"""

from importlib import import_module
from pkgutil import walk_packages

for _, modname, _ in walk_packages(__path__, prefix=__name__ + "."):
    import_module(modname)
