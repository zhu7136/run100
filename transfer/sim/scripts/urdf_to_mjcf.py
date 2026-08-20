# Copyright 2026 Zachary Olkin. All rights reserved.

"""Convert a URDF to a MuJoCo MJCF model.

Generates an MJCF robot model (no scene) from a URDF, following the same
conventions as the g1_21j.xml model in transfer/sim/robots/g1:

- The root link becomes a body with a free joint.
- Fixed joints are kept as nested bodies (no joint).
- All visual meshes become group-1 geoms (no collision).
- URDF collision geoms (only foot boxes for hu_d04) become collision geoms.
- One position actuator per revolute joint, named "<joint>_pos", with
  per-joint-group default classes and torque limits from the URDF effort.
- A standing keyframe is generated from the policy default joint angles if a
  policy parameters yaml is provided (angles are reordered from the Isaac
  order to the URDF/MJCF tree order via joint names).

Usage:
    python urdf_to_mjcf.py --urdf <path.urdf> --out <path.xml> \
        [--policy_params <policy_parameters.yaml>] [--base-height 0.79]
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def strip_comments(text: str) -> str:
    """Remove XML comments from the URDF text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def rpy_to_quat(rpy: str) -> str:
    """Convert an rpy string to an MJCF quat (w x y z)."""
    rpy_arr = np.array([float(v) for v in rpy.split()])
    quat_xyzw = Rotation.from_euler("xyz", rpy_arr).as_quat()
    return f"{quat_xyzw[3]:.8g} {quat_xyzw[0]:.8g} {quat_xyzw[1]:.8g} {quat_xyzw[2]:.8g}"


def get_origin(joint: ET.Element) -> tuple[str, str]:
    """Get the xyz and quat strings for a joint origin."""
    origin = joint.find("origin")
    if origin is None:
        return "0 0 0", "1 0 0 0"
    xyz = origin.get("xyz", "0 0 0")
    rpy = origin.get("rpy", "0 0 0")
    return xyz, rpy_to_quat(rpy)


def mesh_asset_name(filename: str) -> str:
    """Get the mesh asset name from a URDF mesh filename."""
    return os.path.splitext(os.path.basename(filename))[0]


def format_float(value: float) -> str:
    """Format a float compactly."""
    return f"{value:.8g}"


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to MJCF")
    parser.add_argument("--urdf", type=str, required=True, help="Path to the URDF file")
    parser.add_argument("--out", type=str, required=True, help="Path to write the MJCF file")
    parser.add_argument("--policy_params", type=str, default=None,
                        help="Path to a policy_parameters.yaml used for the keyframe pose")
    parser.add_argument("--base-height", type=float, default=0.79,
                        help="Base height for the standing keyframe")
    args = parser.parse_args()

    text = strip_comments(open(args.urdf).read())
    urdf = ET.fromstring(text)

    # Build link/joint maps
    links = {l.get("name"): l for l in urdf.findall("link")}
    children: dict[str, list[tuple[ET.Element, str]]] = {}
    all_joints = []
    for j in urdf.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        children.setdefault(parent, []).append((j, child))
        all_joints.append(j)

    # Root link: not a child of any joint
    child_names = {j.find("child").get("link") for j in all_joints}
    root_name = [n for n in links if n not in child_names]
    if len(root_name) != 1:
        raise ValueError(f"Expected exactly one root link, found: {root_name}")
    root_name = root_name[0]

    # Collect all meshes used
    mesh_files = {}
    for m in urdf.findall(".//mesh"):
        fname = m.get("filename")
        mesh_files.setdefault(mesh_asset_name(fname), fname)

    # Default joint angles from policy params (reordered to tree order)
    default_angles = {}
    if args.policy_params:
        with open(args.policy_params) as f:
            params = yaml.safe_load(f)
        isaac_angles = params["default_joint_angles"]
        isaac_names = params["joint_names_isaac"]
        default_angles = dict(zip(isaac_names, isaac_angles))

    # Per-joint-group actuator default classes (mirror the g1 model)
    group_classes = {
        "hip_pitch": "hip_pitch_pos", "hip_roll": "hip_roll_pos", "hip_yaw": "hip_yaw_pos",
        "knee": "knee_pos", "ankle_pitch": "ankle_pitch_pos", "ankle_roll": "ankle_roll_pos",
        "waist": "waist_yaw_pos", "shoulder_pitch": "shoulder_pitch_pos",
        "shoulder_roll": "shoulder_roll_pos", "shoulder_yaw": "shoulder_yaw_pos",
        "elbow": "elbow_pos",
    }
    group_defaults = {
        "hip_pitch_pos": (100, 2), "hip_roll_pos": (100, 2), "hip_yaw_pos": (100, 2),
        "knee_pos": (150, 4), "ankle_pitch_pos": (40, 2), "ankle_roll_pos": (40, 2),
        "waist_yaw_pos": (100, 2), "shoulder_pitch_pos": (100, 2),
        "shoulder_roll_pos": (100, 2), "shoulder_yaw_pos": (50, 2), "elbow_pos": (50, 2),
    }

    def joint_group(joint_name: str) -> str:
        """Map a joint name to its actuator default class."""
        for prefix, cls in group_classes.items():
            if joint_name.startswith(prefix):
                return cls
        return "elbow_pos"

    indent = "  "

    def build_body(name: str, depth: int, joint: ET.Element | None = None) -> list[str]:
        """Build the MJCF <body>...</body> element for a URDF link.

        Emits the complete body element including open/close tags. The joint
        connecting this body to its parent (if any) is emitted inside.

        Args:
            name: The URDF link name.
            depth: XML indentation depth.
            joint: The URDF joint whose child is this body (None for the root).
        """
        pad = indent * depth
        inner = indent * (depth + 1)
        link = links[name]
        inertial = link.find("inertial")
        lines = [f'{pad}<body name="{name}" pos="0 0 0">']
        if depth == 0:
            # Root link: free joint (floating base)
            lines.append(f'{inner}<joint name="floating_base_joint" type="free" limited="false" actuatorfrclimited="false"/>')
        elif joint is not None:
            j_type = joint.get("type")
            if j_type in ("revolute", "continuous"):
                j_name = joint.get("name")
                axis = joint.find("axis").get("xyz")
                limit = joint.find("limit")
                lower, upper = float(limit.get("lower", "-3.14")), float(limit.get("upper", "3.14"))
                effort = float(limit.get("effort", "0"))
                lines.append(f'{inner}<joint name="{j_name}" pos="0 0 0" axis="{axis}" limited="true" '
                             f'range="{format_float(lower)} {format_float(upper)}" '
                             f'actuatorfrcrange="{-effort:.8g} {effort:.8g}" class="g1"/>')
            elif j_type == "floating":
                lines.append(f'{inner}<joint name="{joint.get("name")}" type="free" limited="false"/>')
            # fixed joints: no joint element
        if inertial is not None:
            origin = inertial.find("origin")
            mass = float(inertial.find("mass").get("value"))
            inertia = inertial.find("inertia")
            ixx, iyy, izz = float(inertia.get("ixx")), float(inertia.get("iyy")), float(inertia.get("izz"))
            ixy, ixz, iyz = float(inertia.get("ixy")), float(inertia.get("ixz")), float(inertia.get("iyz"))
            i_origin = origin if origin is not None else None
            i_pos = i_origin.get("xyz", "0 0 0") if i_origin is not None else "0 0 0"
            # Rotate the inertia tensor into the body frame (URDF inertia is
            # expressed in the rotated inertial frame).
            i_rpy = np.array([float(v) for v in i_origin.get("rpy", "0 0 0").split()]) if i_origin is not None else np.zeros(3)
            rot = Rotation.from_euler("xyz", i_rpy)
            tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            tensor_rot = rot.as_matrix() @ tensor @ rot.as_matrix().T
            ixx, iyy, izz = tensor_rot[0, 0], tensor_rot[1, 1], tensor_rot[2, 2]
            ixy, ixz, iyz = tensor_rot[0, 1], tensor_rot[0, 2], tensor_rot[1, 2]
            lines.append(f'{inner}<inertial pos="{i_pos}" mass="{format_float(mass)}" '
                         f'fullinertia="{format_float(ixx)} {format_float(iyy)} {format_float(izz)} '
                         f'{format_float(ixy)} {format_float(ixz)} {format_float(iyz)}"/>')
        # Visual geoms (no collision)
        rgba = "0.75294 0.75294 0.75294 1"
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh = geom.find("mesh")
            if mesh is not None:
                v_origin = visual.find("origin")
                v_pos = v_origin.get("xyz", "0 0 0") if v_origin is not None else "0 0 0"
                v_quat = rpy_to_quat(v_origin.get("rpy", "0 0 0")) if v_origin is not None else "1 0 0 0"
                lines.append(f'{inner}<geom type="mesh" contype="0" conaffinity="0" group="1" '
                             f'pos="{v_pos}" quat="{v_quat}" rgba="{rgba}" mesh="{mesh_asset_name(mesh.get("filename"))}"/>')
        # Collision geoms (URDF collision blocks)
        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            origin = collision.find("origin")
            c_pos = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            c_quat = rpy_to_quat(origin.get("rpy", "0 0 0")) if origin is not None else "1 0 0 0"
            box = geom.find("box")
            if box is not None:
                size = box.get("size")
                lines.append(f'{inner}<geom type="box" size="{size}" pos="{c_pos}" quat="{c_quat}" class="collision"/>')
            sphere = geom.find("sphere")
            if sphere is not None:
                lines.append(f'{inner}<geom type="sphere" size="{sphere.get("radius")}" pos="{c_pos}" quat="{c_quat}" class="collision"/>')

        # Sites for the ankle position sensors
        if name.endswith("ankle_roll_link"):
            side = "left" if name.startswith("left") else "right"
            lines.append(f'{inner}<site name="{side}_ankle_site" size="0.001" pos="0.018 0 -0.0625" rgba="1 0 0 1"/>')

        # Child joints/bodies
        for child_joint, child_name in children.get(name, []):
            o_xyz, o_quat = get_origin(child_joint)
            body_lines = build_body(child_name, depth + 1, child_joint)
            # Override the body pos/quat with the joint origin
            body_lines[0] = f'{inner}<body name="{child_name}" pos="{o_xyz}" quat="{o_quat}">'
            lines.extend(body_lines)
        lines.append(f"{pad}</body>")
        return lines

    lines = []
    lines.append('<mujoco model="hu_d04">')
    lines.append('  <compiler angle="radian" meshdir="meshes"/>')
    lines.append('')
    lines.append('  <option cone="elliptic" impratio="100" integrator="implicitfast" timestep="0.001"/>')
    lines.append('')
    lines.append('  <default>')
    lines.append('    <default class="g1">')
    lines.append('      <joint damping="1" armature="0.01" frictionloss="0.1"/>')
    lines.append('    </default>')
    for cls, (kp, kv) in group_defaults.items():
        lines.append(f'    <default class="{cls}">')
        lines.append(f'      <position kp="{kp}" kv="{kv}" ctrllimited="false"/>')
        lines.append('    </default>')
    lines.append('    <default class="collision">')
    lines.append('      <geom size="0.005" priority="1" condim="6" solref="0.01 1.1" friction="1.0 0.3 0.3"/>')
    lines.append('    </default>')
    lines.append('  </default>')
    lines.append('')
    lines.append('  <asset>')
    for asset_name, fname in sorted(mesh_files.items()):
        lines.append(f'    <mesh name="{asset_name}" file="{os.path.basename(fname)}"/>')
    lines.append('  </asset>')
    lines.append('')
    lines.append('  <worldbody>')
    lines.extend(build_body(root_name, 0))
    lines.append('  </worldbody>')
    lines.append('')
    lines.append('  <actuator>')
    for joint in all_joints:
        if joint.get("type") not in ("revolute", "continuous"):
            continue
        j_name = joint.get("name")
        cls = joint_group(j_name)
        lines.append(f'    <position name="{j_name}_pos" class="{cls}" joint="{j_name}"/>')
    lines.append('  </actuator>')
    lines.append('')

    # Standing keyframe with policy default angles
    key_qpos = ["0 0 " + format_float(args.base_height), "1 0 0 0"]
    for joint in all_joints:
        if joint.get("type") not in ("revolute", "continuous"):
            continue
        j_name = joint.get("name")
        if j_name in default_angles:
            key_qpos.append(format_float(default_angles[j_name]))
        else:
            key_qpos.append("0")
    lines.append('  <keyframe>')
    lines.append('    <key name="standing" qpos="' + "\n      ".join(key_qpos) + '"/>')
    lines.append('  </keyframe>')
    lines.append('</mujoco>')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()