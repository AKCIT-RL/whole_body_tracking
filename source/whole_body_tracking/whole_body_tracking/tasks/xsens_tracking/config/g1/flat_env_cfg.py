from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatLowFreqEnvCfg,
    G1FlatWoStateEstimationEnvCfg,
)


@configclass
class XsensG1FlatEnvCfg(G1FlatEnvCfg):
    """Baseline G1 tracking env, under an XSens-specific Gym ID.

    Notes:
    - We intentionally do NOT set `commands.motion.motion_file` here.
      The standard training entry-point (`scripts/rsl_rl/train.py`) resolves the
      motion from `--registry_name` or `--motion_file` and injects it into
      `env_cfg` before creating the env.
    """

    def __post_init__(self):
        super().__post_init__()


@configclass
class XsensG1FlatWoStateEstimationEnvCfg(G1FlatWoStateEstimationEnvCfg):
    """G1 tracking env without state-estimation observations, XSens Gym ID."""

    def __post_init__(self):
        super().__post_init__()


@configclass
class XsensG1FlatLowFreqEnvCfg(G1FlatLowFreqEnvCfg):
    """G1 tracking env at lower control frequency, XSens Gym ID."""

    def __post_init__(self):
        super().__post_init__()
