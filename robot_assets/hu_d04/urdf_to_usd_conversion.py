# Copyright 2026 Zachary Olkin. All rights reserved.

# Isaac Sim app must be launched FIRST
from isaaclab.app import AppLauncher

# Launch with headless mode (no GUI needed for conversion)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# 1. Create the configuration
cfg = UrdfConverterCfg(
    asset_path="/home/xf/robot_rl/robot_assets/hu_d04/hu_d04.urdf",
    usd_dir="/home/xf/robot_rl/robot_assets/hu_d04",
    usd_file_name="hu_d04_default.usd",
    fix_base=False,
    merge_fixed_joints=False,
    make_instanceable=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=400.0,
            damping=40.0,
        ),
    ),
)

# 2. Run the converter
converter = UrdfConverter(cfg)

# 3. Access the output path
print(f"USD saved to: {converter.usd_path}")