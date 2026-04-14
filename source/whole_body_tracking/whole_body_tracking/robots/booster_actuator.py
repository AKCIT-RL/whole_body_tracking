"""Booster-specific actuator models for whole_body_tracking.

Reflects commit 651b7a5 from BoosterRobotics/booster_train:
  - BoosterDelayedPDActuator: delayed PD with piecewise-linear T-N curve torque clipping
  - BoosterJointCfg / per-motor subclasses: hardware-accurate parameters (effort, velocity,
    knee_point_velocity, armature) with stiffness/damping auto-computed from natural_freq
  - BoosterT1AnkleParaWrapperCfg: parallel-mechanism transformation for T1 ankle joints
"""
from __future__ import annotations
from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class BoosterDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with speed-dependent torque clipping (T-N curve).

    Maximum torque is at effort_limit for |vel| <= knee_point_velocity, then
    decreases linearly to zero at velocity_limit.
    """

    cfg: BoosterDelayedPDActuatorCfg

    def __init__(self, cfg: "BoosterDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.knee_point_velocity = self._parse_joint_parameter(cfg.knee_point_velocity, self.velocity_limit)
        self.knee_point_velocity = torch.clamp(self.knee_point_velocity, min=0.0)
        self.knee_point_velocity = torch.minimum(self.knee_point_velocity, self.velocity_limit)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._denom = (self.velocity_limit - self.knee_point_velocity).clamp(min=1e-6)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        joint_vel_abs = self._joint_vel.abs()
        v_max = self.velocity_limit
        tau_max = self.effort_limit

        non_positive_vmax = v_max <= 0.0
        non_finite_vmax = ~torch.isfinite(v_max)

        tau_linear = tau_max * (v_max - joint_vel_abs) / self._denom
        max_effort = tau_linear.clamp(min=0.0).clamp(max=tau_max)
        max_effort = torch.where(non_finite_vmax, tau_max, max_effort)
        max_effort = torch.where(non_positive_vmax, torch.zeros_like(max_effort), max_effort)

        return torch.clip(effort, min=-max_effort, max=max_effort)


@configclass
class BoosterJointCfg:
    """Per-motor hardware parameters. Stiffness/damping auto-computed if not provided."""

    joint_model_name: str = MISSING
    effort_limit: float = MISSING
    velocity_limit: float = MISSING
    knee_point_velocity: float = MISSING
    armature: float = MISSING

    stiffness: float = None
    damping: float = None
    natural_freq: float = 10.0   # Hz
    damping_ratio: float = 2.0

    def __post_init__(self):
        wn = 2 * 3.1415926535 * self.natural_freq
        if self.stiffness is None:
            self.stiffness = self.armature * wn ** 2
        if self.damping is None:
            self.damping = 2 * self.damping_ratio * self.armature * wn


@configclass
class BoosterDelayedActuatorCfg(DelayedPDActuatorCfg):
    """Base config for Booster delayed actuators. Accepts booster_joint_cfgs for convenience."""

    class_type: type = MISSING

    knee_point_velocity: dict[str, float] | float | None = None
    stiffness: dict[str, float] | float | None = None
    damping: dict[str, float] | float | None = None
    booster_joint_cfgs: dict[str, BoosterJointCfg] | BoosterJointCfg | None = None

    def __post_init__(self):
        if self.booster_joint_cfgs is None:
            return
        if isinstance(self.booster_joint_cfgs, BoosterJointCfg):
            j = self.booster_joint_cfgs
            self.effort_limit_sim = j.effort_limit
            self.velocity_limit_sim = j.velocity_limit
            self.knee_point_velocity = j.knee_point_velocity
            self.armature = j.armature
            if self.stiffness is None:
                self.stiffness = j.stiffness
            if self.damping is None:
                self.damping = j.damping
        elif isinstance(self.booster_joint_cfgs, dict):
            self.effort_limit_sim = {k: v.effort_limit for k, v in self.booster_joint_cfgs.items()}
            self.velocity_limit_sim = {k: v.velocity_limit for k, v in self.booster_joint_cfgs.items()}
            self.armature = {k: v.armature for k, v in self.booster_joint_cfgs.items()}
            self.knee_point_velocity = {k: v.knee_point_velocity for k, v in self.booster_joint_cfgs.items()}
            if self.stiffness is None:
                self.stiffness = {k: v.stiffness for k, v in self.booster_joint_cfgs.items()}
            if self.damping is None:
                self.damping = {k: v.damping for k, v in self.booster_joint_cfgs.items()}


@configclass
class BoosterDelayedPDActuatorCfg(BoosterDelayedActuatorCfg):
    """Config for :class:`BoosterDelayedPDActuator`."""

    class_type: type = BoosterDelayedPDActuator


# ---------------------------------------------------------------------------
# Per-motor hardware configs (from commit 651b7a5)
# ---------------------------------------------------------------------------

@configclass
class BoosterJointE8112(BoosterJointCfg):
    """T1 Hip Pitch — E8112. Kp≈207, Kd≈13.2."""
    joint_model_name: str = "E8112"
    effort_limit: float = 96.0
    velocity_limit: float = 16.76
    knee_point_velocity: float = 7.54
    armature: float = 0.0523908


@configclass
class BoosterJointE6408(BoosterJointCfg):
    """T1 Hip Roll/Yaw & Waist — E6408. Kp≈189, Kd≈12.0."""
    joint_model_name: str = "E6408"
    effort_limit: float = 68.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 1.88
    armature: float = 0.0478125


@configclass
class BoosterJointE8116(BoosterJointCfg):
    """T1 Knee Pitch — E8116. Kp≈251, Kd≈16.0."""
    joint_model_name: str = "E8116"
    effort_limit: float = 130.0
    velocity_limit: float = 14.66
    knee_point_velocity: float = 6.28
    armature: float = 0.0636012


@configclass
class BoosterJointE4315(BoosterJointCfg):
    """T1 Ankle base — E4315 (used with parallel wrapper). Kp≈134, Kd≈8.5."""
    joint_model_name: str = "E4315"
    effort_limit: float = 76.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 2.62
    armature: float = 0.0339552


@configclass
class BoosterJointE4310(BoosterJointCfg):
    """T1 Arms — E4310. Kp≈111, Kd≈7.1."""
    joint_model_name: str = "E4310"
    effort_limit: float = 38.3
    velocity_limit: float = 17.59
    knee_point_velocity: float = 7.85
    armature: float = 0.0282528


@configclass
class BoosterJointDM4310(BoosterJointCfg):
    """T1 Head/Neck — DM4310. Kp≈7.1, Kd≈0.45."""
    joint_model_name: str = "DM4310"
    effort_limit: float = 7.0
    velocity_limit: float = 12.57
    knee_point_velocity: float = 41.89
    armature: float = 0.0018


# ---------------------------------------------------------------------------
# Parallel-mechanism wrappers
# ---------------------------------------------------------------------------

@configclass
class ParallelJointWrapperCfg(BoosterJointCfg):
    """Transforms a base BoosterJointCfg through a parallel-mechanism gear ratio."""

    joint_model_name: str = "ParallelJointWrapper"
    effort_ratio: tuple[float, float] = MISSING
    velocity_ratio: tuple[float, float] = MISSING
    armature_ratio: tuple[float, float] = MISSING
    knee_point_velocity_ratio: tuple[float, float] = (1.0, 1.0)
    base_joint_cfg: BoosterJointCfg = MISSING
    serial_index: int = MISSING   # 0 = pitch, 1 = roll

    def __post_init__(self):
        i = self.serial_index
        b = self.base_joint_cfg
        self.effort_limit = self.effort_ratio[i] * b.effort_limit
        self.velocity_limit = self.velocity_ratio[i] * b.velocity_limit
        self.knee_point_velocity = self.knee_point_velocity_ratio[i] * b.knee_point_velocity
        self.armature = self.armature_ratio[i] * b.armature
        self.joint_model_name = f"{self.joint_model_name}({b.joint_model_name})[{i}]"
        super().__post_init__()


@configclass
class BoosterT1AnkleParaWrapperCfg(ParallelJointWrapperCfg):
    """T1 ankle parallel mechanism (ratio 2× armature on each axis)."""
    joint_model_name: str = "BoosterT1AnkleParaWrapper"
    effort_ratio: tuple[float, float] = (1.0, 1.0)
    velocity_ratio: tuple[float, float] = (1.0, 1.0)
    armature_ratio: tuple[float, float] = (2.0, 2.0)
