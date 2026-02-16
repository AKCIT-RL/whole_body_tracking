import gymnasium as gym

from . import flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="XsensTracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.XsensG1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="XsensTracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.XsensG1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="XsensTracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.XsensG1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": "whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)
