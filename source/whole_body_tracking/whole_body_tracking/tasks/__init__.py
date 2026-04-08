"""Package containing task implementations for various robotic environments."""

import importlib.util

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]

# T1 task configs depend on optional booster_assets. Keep G1 usable when absent.
if importlib.util.find_spec("booster_assets") is None:
    _BLACKLIST_PKGS.append("t1")

# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
