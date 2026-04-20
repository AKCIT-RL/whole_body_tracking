import io
import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _algorithm_actor_module(alg: object):
    """Older rsl_rl: ``alg.actor``; newer: ``alg.policy`` is ActorCritic with ``.actor``."""
    a = getattr(alg, "actor", None)
    if a is not None:
        return a
    pol = getattr(alg, "policy", None)
    if pol is None:
        raise AttributeError("RL algorithm has neither .actor nor .policy")
    a = getattr(pol, "actor", None)
    if a is not None:
        return a
    return pol


def _runner_uses_wandb(runner: OnPolicyRunner) -> bool:
    """rsl_rl recente define ``logger_type`` no próprio runner; versões antigas usavam ``runner.logger``."""
    if getattr(runner, "logger_type", None) == "wandb":
        return True
    lg = getattr(runner, "logger", None)
    return lg is not None and getattr(lg, "logger_type", None) == "wandb"


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _runner_uses_wandb(self):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # rsl_rl versions differ: export_policy_to_onnx may not exist.
            # We always use the repo's motion-aware exporter for consistent behavior.
            buf = io.BytesIO()
            torch.save(_algorithm_actor_module(self.alg), buf)
            buf.seek(0)
            actor_cpu = torch.load(buf, map_location="cpu", weights_only=False)
            actor_cpu.eval()
            actor_compat = type("_AC", (), {"actor": actor_cpu, "is_recurrent": False})()
            normalizer = getattr(actor_cpu, "obs_normalizer", None)
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                actor_compat,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _runner_uses_wandb(self):
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"

            # Motion-aware ONNX export (works for both T1 and G1).
            # Use torch.save/load to get a clean copy of the actor, avoiding deepcopy
            # failures caused by weight_norm non-leaf tensors in the training graph.
            buf = io.BytesIO()
            torch.save(_algorithm_actor_module(self.alg), buf)
            buf.seek(0)
            actor_cpu = torch.load(buf, map_location="cpu", weights_only=False)
            actor_cpu.eval()
            actor_compat = type("_AC", (), {"actor": actor_cpu, "is_recurrent": False})()
            normalizer = getattr(actor_cpu, "obs_normalizer", None)
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                actor_compat,
                normalizer=normalizer,
                path=policy_path,
                filename=filename,
            )

            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
