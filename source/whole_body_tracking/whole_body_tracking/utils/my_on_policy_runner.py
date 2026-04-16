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

    def _get_alg_actor(self):
        """Return the MLP actor used for obs -> actions (not the full ActorCritic wrapper).

        Newer rsl_rl exposes ``get_policy()`` as ``ActorCritic``; its ``forward`` is not ``(obs)``.
        The motion ONNX exporter expects the inner ``MLPModel`` (``ActorCritic.actor`` or ``PPO.actor``).
        """
        alg = self.alg
        direct = getattr(alg, "actor", None)
        if direct is not None:
            return direct

        policy = None
        get_policy = getattr(alg, "get_policy", None)
        if callable(get_policy):
            policy = get_policy()
        if policy is None:
            policy = getattr(alg, "policy", None)
        if policy is None:
            raise AttributeError(
                "Cannot find actor/policy on algorithm: expected .actor, .get_policy(), or .policy on "
                f"{type(alg).__name__}"
            )

        inner = getattr(policy, "actor", None)
        if inner is not None:
            return inner
        return policy

    def _should_run_wandb_onnx_hooks(self) -> bool:
        """rsl_rl versions differ: logger may be missing or logger_type only on internal Logger after init_logging_writer."""
        log = getattr(self, "logger", None) or getattr(self, "_logger", None)
        if log is not None and getattr(log, "logger_type", None) == "wandb":
            return True
        cfg = getattr(self, "cfg", None)
        if isinstance(cfg, dict) and str(cfg.get("logger", "")).lower() == "wandb":
            return True
        return False

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if not self._should_run_wandb_onnx_hooks() or wandb.run is None:
            return
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
            buf = io.BytesIO()
            torch.save(self._get_alg_actor(), buf)
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
