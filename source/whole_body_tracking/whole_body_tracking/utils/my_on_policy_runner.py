# import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb

# Temporarily disabled: ONNX export during W&B saves (re-enable when rsl_rl API is aligned).
# from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
#
#
# def _unwrap_to_isaac_env(vec_env):
#     e = vec_env
#     while hasattr(e, "unwrapped") and e.unwrapped is not e:
#         e = e.unwrapped
#     return e
#
#
# def _export_motion_onnx(runner: OnPolicyRunner, policy_path: str, filename: str) -> None:
#     """Export ONNX for W&B; uses rsl_rl built-in when present, else project motion exporter."""
#     if hasattr(runner, "export_policy_to_onnx"):
#         runner.export_policy_to_onnx(path=policy_path, filename=filename)
#         return
#
#     class _ActorCriticCompat:
#         def __init__(self, module):
#             self.actor = module
#             self.is_recurrent = getattr(module, "is_recurrent", False)
#
#     isaac_env = _unwrap_to_isaac_env(runner.env)
#     actor_module = runner.alg.actor
#     normalizer = getattr(actor_module, "obs_normalizer", None)
#     export_motion_policy_as_onnx(
#         isaac_env,
#         _ActorCriticCompat(actor_module),
#         normalizer=normalizer,
#         path=policy_path,
#         filename=filename,
#     )
#     runner.alg.actor.to(runner.device)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        # if getattr(self, "logger_type", None) == "wandb" and wandb.run is not None:
        #     policy_path = path.split("model")[0]
        #     filename = policy_path.split("/")[-2] + ".onnx"
        #     _export_motion_onnx(self, policy_path, filename)
        #     attach_onnx_metadata(
        #         _unwrap_to_isaac_env(self.env), wandb.run.name, path=policy_path, filename=filename
        #     )
        #     run_dir = os.path.normpath(policy_path)
        #     wandb.save(os.path.join(run_dir, filename), base_path=run_dir)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self, "logger_type", None) == "wandb" and wandb.run is not None:
            # policy_path = path.split("model")[0]
            # filename = policy_path.split("/")[-2] + ".onnx"
            # _export_motion_onnx(self, policy_path, filename)
            # attach_onnx_metadata(
            #     _unwrap_to_isaac_env(self.env), wandb.run.name, path=policy_path, filename=filename
            # )
            # run_dir = os.path.normpath(policy_path)
            # wandb.save(os.path.join(run_dir, filename), base_path=run_dir)

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
