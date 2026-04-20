# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Optional handle_deprecated_rsl_rl_cfg for older Isaac Lab / isaaclab_rl builds."""

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:

    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version: str):
        return agent_cfg


__all__ = ["handle_deprecated_rsl_rl_cfg"]
