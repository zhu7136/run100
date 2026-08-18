# Copyright 2026 Zachary Olkin. All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT_ASSETS = "robot_assets/hu_d04"

HU_D04_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/hu_d04_default.usd",
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
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Standing
        pos=(0.0, 0.0, 0.886),

        joint_pos={
            # Standing
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.31,
            "left_shoulder_roll_joint": 0.24,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.31,
            "right_shoulder_roll_joint": -0.24,
            ".*_elbow_joint": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=140.0,
            velocity_limit_sim=5.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=13.6,
            stiffness=20.0,
            damping=2.0,
            armature=0.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=13.6,
            stiffness=100.0,
            damping=5.0,
            armature=0.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=42.0,
            velocity_limit_sim=19.6,
            stiffness={
                ".*_shoulder_pitch_joint": 180.0,
                ".*_shoulder_roll_joint": 180.0,
                ".*_shoulder_yaw_joint": 60.0,
                ".*_elbow_joint": 140.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.5,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 4.0,
                ".*_elbow_joint": 4.0,
            },
            armature=0.0,
        ),
    },
)
"""Configuration for the HU_D04 Humanoid robot."""

HU_D04_ACTION_SCALE = {}
for a in HU_D04_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            HU_D04_ACTION_SCALE[n] = 0.125 * e[n] / s[n]