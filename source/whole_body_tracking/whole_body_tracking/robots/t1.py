import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

import os as _os
try:
    from booster_assets import BOOSTER_ASSETS_DIR
except ImportError:
    BOOSTER_ASSETS_DIR = _os.environ.get("BOOSTER_ASSETS_DIR", "")
    if not BOOSTER_ASSETS_DIR:
        raise ImportError(
            "booster_assets not installed and BOOSTER_ASSETS_DIR env var not set.\n"
            "Fix:\n"
            "  git clone https://github.com/BoosterRobotics/booster_assets\n"
            "  pip install -e booster_assets\n"
            "Or: export BOOSTER_ASSETS_DIR=/path/to/booster_assets"
        )

from .booster_actuator import (
    BoosterDelayedPDActuatorCfg,
    BoosterJointE8112,
    BoosterJointE6408,
    BoosterJointE8116,
    BoosterJointE4315,
    BoosterJointE4310,
    BoosterJointDM4310,
    BoosterT1AnkleParaWrapperCfg,
)

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            booster_joint_cfgs={
                ".*_Hip_Pitch": BoosterJointE8112(),   # 96 Nm, 16.76 rad/s, Kp≈207, Kd≈13.2
                ".*_Hip_Roll": BoosterJointE6408(),    # 68 Nm, 14.66 rad/s, Kp≈189, Kd≈12.0
                ".*_Hip_Yaw": BoosterJointE6408(),     # 68 Nm, 14.66 rad/s, Kp≈189, Kd≈12.0
                ".*_Knee_Pitch": BoosterJointE8116(),  # 130 Nm, 14.66 rad/s, Kp≈251, Kd≈16.0
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4315(),
                    serial_index=0,
                ),  # 76 Nm, 12.57 rad/s, Kp≈268, Kd≈17.1
                ".*_Ankle_Roll": BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=BoosterJointE4315(),
                    serial_index=1,
                ),  # 76 Nm, 12.57 rad/s, Kp≈268, Kd≈17.1
            },
        ),
        "waist": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=["Waist"],
            booster_joint_cfgs=BoosterJointE6408(),    # 68 Nm, 14.66 rad/s, Kp≈189, Kd≈12.0
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=BoosterJointE4310(),    # 38.3 Nm, 17.59 rad/s, Kp≈111, Kd≈7.1
        ),
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=BoosterJointDM4310(),   # 7 Nm, 12.57 rad/s, Kp≈7.1, Kd≈0.45
        ),
    },
)

T1_ACTION_SCALE = {}
for _a in T1_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for n in _names:
        if n in _e and n in _s and _s[n]:
            T1_ACTION_SCALE[n] = 0.25 * _e[n] / _s[n]
