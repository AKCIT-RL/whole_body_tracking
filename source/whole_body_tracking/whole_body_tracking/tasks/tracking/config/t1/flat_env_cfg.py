from dataclasses import MISSING

from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import T1_ACTION_SCALE, T1_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# T1 URDF links (24 total). Key tracking bodies selected below.
# Full link list: Trunk, H1, H2, AL1, AL2, AL3, left_hand_link,
#   AR1, AR2, AR3, right_hand_link, Waist,
#   Hip_Pitch_Left, Hip_Roll_Left, Hip_Yaw_Left, Shank_Left, Ankle_Cross_Left, left_foot_link,
#   Hip_Pitch_Right, Hip_Roll_Right, Hip_Yaw_Right, Shank_Right, Ankle_Cross_Right, right_foot_link


@configclass
class T1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE

        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.loop = False
        self.commands.motion.adaptive_kernel_size = 3
        # T1-specific friction (booster_train values — less aggressive than G1)
        self.events.physics_material.params["static_friction_range"] = (0.3, 0.6)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.6)
        self.commands.motion.body_names = [
            "Trunk",
            "H2",            # head
            # legs
            "Hip_Roll_Left",
            "Shank_Left",
            "left_foot_link",
            "Hip_Roll_Right",
            "Shank_Right",
            "right_foot_link",
            # left arm
            "AL2",
            "AL3",
            "left_hand_link",
            # right arm
            "AR2",
            "AR3",
            "right_hand_link",
        ]

        # motion_file must be set per-task, e.g.:
        # self.commands.motion.motion_file = "/path/to/motion.npz"

        # undesired contacts: everything except hands and feet
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!left_hand_link$)(?!right_hand_link$)"
            r"(?!left_foot_link$)(?!right_foot_link$).+$"
        ]

        # termination: track hands and feet height
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_hand_link",
            "right_hand_link",
            "left_foot_link",
            "right_foot_link",
        ]

        # CoM randomization on Trunk (equivalent to torso_link on G1)
        self.events.base_com.params["asset_cfg"].body_names = "Trunk"


@configclass
class T1FlatWoStateEstimationEnvCfg(T1FlatEnvCfg):
    """Variant without state estimation — remove obs that require position/lin_vel estimate.

    Use this when the hardware does not have a reliable position/linear-velocity estimator.
    The exported ONNX will automatically reflect the reduced observation space.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
