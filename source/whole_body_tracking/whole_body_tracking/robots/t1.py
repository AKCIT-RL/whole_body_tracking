import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from booster_assets import BOOSTER_ASSETS_DIR

ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012

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
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 45.0,
                ".*_Hip_Roll": 25.0,
                ".*_Hip_Yaw": 25.0,
                ".*_Knee_Pitch": 60.0,
            },
            velocity_limit_sim={
                ".*_Hip_Pitch": 16.76,
                ".*_Hip_Roll": 12.57,
                ".*_Hip_Yaw": 12.57,
                ".*_Knee_Pitch": 12.57,
            },
            stiffness={
                ".*_Hip_Pitch": 200.0,
                ".*_Hip_Roll": 200.0,
                ".*_Hip_Yaw": 200.0,
                ".*_Knee_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Pitch": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Yaw": 5.0,
                ".*_Knee_Pitch": 5.0,
            },
            armature={
                ".*_Hip_Pitch": ARMATURE_8112,
                ".*_Hip_Roll": ARMATURE_6408,
                ".*_Hip_Yaw": ARMATURE_6408,
                ".*_Knee_Pitch": ARMATURE_8116,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim={
                ".*_Ankle_Pitch": 24.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*_Ankle_Pitch": 18.8,
                ".*_Ankle_Roll": 12.4,
            },
            stiffness=50.0,
            damping=1.0,
            armature=2.0 * ARMATURE_4315,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=25.0,
            velocity_limit_sim=12.57,
            stiffness=200.0,
            damping=5.0,
            armature=ARMATURE_6408,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 7.33,
                ".*_Shoulder_Roll": 7.33,
                ".*_Elbow_Pitch": 7.33,
                ".*_Elbow_Yaw": 7.33,
            },
            stiffness={
                ".*_Shoulder_Pitch": 50.0,
                ".*_Shoulder_Roll": 50.0,
                ".*_Elbow_Pitch": 50.0,
                ".*_Elbow_Yaw": 50.0,
            },
            damping={
                ".*_Shoulder_Pitch": 1.0,
                ".*_Shoulder_Roll": 1.0,
                ".*_Elbow_Pitch": 1.0,
                ".*_Elbow_Yaw": 1.0,
            },
            armature={
                ".*_Shoulder_Pitch": ARMATURE_4310,
                ".*_Shoulder_Roll": ARMATURE_4310,
                ".*_Elbow_Pitch": ARMATURE_4310,
                ".*_Elbow_Yaw": ARMATURE_4310,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.0,
            stiffness=10.0,
            damping=1.0,
            armature=0.001,
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
