import io
import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self.logger, "logger_type", None) == "wandb":
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # Use built-in export method for rsl_rl >= 4.0.0
            self.export_policy_to_onnx(path=policy_path, filename=filename)
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
        if getattr(self.logger, "logger_type", None) == "wandb":
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"

            anchor = self.env.unwrapped.command_manager.get_term("motion").cfg.anchor_body_name
            is_t1 = anchor == "Trunk"

            if is_t1:
                # T1 deploy uses TorchScript JIT (exported by play.py) — plain ONNX is fine for logging
                self.export_policy_to_onnx(path=policy_path, filename=filename)
            else:
                # G1: export motion-aware ONNX with time_step input and motion data outputs.
                # Use torch.save/load to get a clean copy of the actor, avoiding deepcopy
                # failures caused by weight_norm non-leaf tensors in the training graph.
                import io
                buf = io.BytesIO()
                torch.save(self.alg.actor, buf)
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
