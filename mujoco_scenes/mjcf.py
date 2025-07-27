"""Functions for loading and combining MuJoCo models."""

import itertools
import re
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from etils import epath

from .errors import ModelValidationError, TemplateDirectoryNotFoundError, TemplateNotFoundError


def get_template_dir() -> epath.Path:
    if not (template_dir := epath.Path(__file__).parent / "templates").exists():
        raise TemplateDirectoryNotFoundError()
    return template_dir.resolve()


def get_scene(name: str) -> epath.Path:
    scene_path = get_template_dir() / f"{name}.xml"
    if not scene_path.exists():
        raise TemplateNotFoundError(name)
    return scene_path


def list_scenes() -> list[str]:
    return [p.stem for p in get_template_dir().glob("*.xml")]


def rotate_np(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    if len(vec.shape) != 1:
        raise ValueError("vec must have no batch dimensions.")
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def quat_mul_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def _transform_do(
    parent_pos: np.ndarray,
    parent_quat: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = parent_pos + rotate_np(pos, parent_quat)
    rot = quat_mul_np(parent_quat, quat)
    return pos, rot


def _offset(elem: ET.Element, parent_pos: np.ndarray, parent_quat: np.ndarray) -> None:
    pos = elem.attrib.get("pos", "0 0 0")
    quat = elem.attrib.get("quat", "1 0 0 0")
    pos = np.fromstring(pos, sep=" ")
    quat = np.fromstring(quat, sep=" ")
    fromto = elem.attrib.get("fromto", None)
    if fromto:
        # When "fromto" is present, process the positions separately because it
        # is not compatible with pos/quat.
        from_pos = np.fromstring(" ".join(fromto.split(" ")[0:3]), sep=" ")
        to_pos = np.fromstring(" ".join(fromto.split(" ")[3:6]), sep=" ")
        from_pos, _ = _transform_do(parent_pos, parent_quat, from_pos, quat)
        to_pos, _ = _transform_do(parent_pos, parent_quat, to_pos, quat)
        fromto = " ".join("%f" % i for i in np.concatenate([from_pos, to_pos]))
        elem.attrib["fromto"] = fromto
        return
    pos, quat = _transform_do(parent_pos, parent_quat, pos, quat)
    pos_str = " ".join("%f" % i for i in pos)
    quat_str = " ".join("%f" % i for i in quat)
    elem.attrib["pos"] = pos_str
    elem.attrib["quat"] = quat_str


def _get_meshdir(elem: ET.Element) -> str | None:
    elems = list(elem.iter("compiler"))
    return elems[0].get("meshdir") if elems else None


def _find_assets(
    elem: ET.Element,
    path: epath.Path,
    meshdir: str | None,
) -> dict[str, bytes]:
    assets = {}
    path = path if path.is_dir() else path.parent
    fname = elem.attrib.get("file") or elem.attrib.get("filename")

    if fname is not None:
        if fname.lower().endswith(".xml") or fname.lower().endswith(".mjcf"):
            pass
        else:
            path_local = path / meshdir if meshdir else path
            assets[fname] = (path_local / fname).read_bytes()

    for child in list(elem):
        assets.update(_find_assets(child, path, meshdir))

    return assets


def validate_model(mj: mujoco.MjModel) -> None:
    if mj.opt.integrator != 0:
        raise NotImplementedError("Only euler integration is supported.")
    if mj.opt.cone != 0:
        raise NotImplementedError("Only pyramidal cone friction is supported.")
    if (mj.geom_fluid != 0).any():
        raise NotImplementedError("Ellipsoid fluid model not implemented.")
    if mj.opt.wind.any():
        raise NotImplementedError("option.wind is not implemented.")
    if mj.opt.impratio != 1:
        raise NotImplementedError("Only impratio=1 is supported.")

    # actuators
    if any(i not in [0, 1] for i in mj.actuator_biastype):
        raise NotImplementedError("Only actuator_biastype in [0, 1] are supported.")
    if any(i != 0 for i in mj.actuator_gaintype):
        raise NotImplementedError("Only actuator_gaintype in [0] is supported.")
    if not (mj.actuator_trntype == 0).all():
        raise NotImplementedError("Only joint transmission types are supported for actuators.")

    # solver parameters
    if (mj.geom_solmix[0] != mj.geom_solmix).any():
        raise NotImplementedError("geom_solmix parameter not supported.")
    if (mj.geom_priority[0] != mj.geom_priority).any():
        raise NotImplementedError("geom_priority parameter not supported.")

    # check joints
    q_width = {0: 7, 1: 4, 2: 1, 3: 1}
    non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
    if mj.qpos0[non_free].any():
        raise NotImplementedError("The `ref` attribute on joint types is not supported.")

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]):
        position = np.array([p for _, p in group])
        if not (position == position[0]).all():
            raise ModelValidationError("invalid joint stack: only one joint position allowed")

    # check dofs
    jnt_range = mj.jnt_range.copy()
    jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
    for typ, limit, stiffness in zip(mj.jnt_type, jnt_range, mj.jnt_stiffness):
        if typ == 0:
            if stiffness > 0:
                raise ModelValidationError("brax does not support stiffness for free joints")
        elif typ == 1:
            if np.any(~np.isinf(limit)):
                raise ModelValidationError("brax does not support joint ranges for ball joints")
        elif typ in (2, 3):
            continue
        else:
            raise ModelValidationError(f"invalid joint type: {typ}")

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]):
        typs = [t for _, t in group]
        if len(typs) == 1 and typs[0] == 0:
            continue  # free joint configuration
        elif 0 in typs:
            raise ModelValidationError("invalid joint stack: cannot stack free joints")
        elif 1 in typs:
            raise NotImplementedError("ball joints not supported")

    # check collision geometries
    for i, typ in enumerate(mj.geom_type):
        mask = mj.geom_contype[i] | (mj.geom_conaffinity[i] << 32)
        if typ == 5:  # Cylinder
            _, halflength = mj.geom_size[i, 0:2]
            if halflength > 0.001 and mask > 0:
                raise NotImplementedError("Cylinders of half-length>0.001 are not supported for collision.")


def load_mjmodel(
    path: str | Path | epath.Path,
    scene: str | None = None,
    position_overrides: dict[str, dict[str, float]] | None = None,
    class_overrides: dict[str, dict[str, float]] | None = None,
) -> mujoco.MjModel:
    """
    path             Path to robot XML
    scene            Optional scene name
    position_overrides  { joint_name: { attr: value, … }, … } → overrides on <position> tags
    class_overrides  { class_name: { attr: value, … }, … } → overrides on <default class="…"> children
    """
    path      = epath.Path(path)
    robot_dir = path.parent

    robot_text = path.read_text()
    robot_elem = ET.fromstring(robot_text)

    # apply per-joint overrides on <position> tags ---
    if position_overrides:
        for pos in robot_elem.iter("position"):
            jn = pos.attrib.get("joint")
            if jn in position_overrides:
                for attr, val in position_overrides[jn].items():
                    pos.set(attr, str(val))

    if class_overrides:
        for default in robot_elem.findall(".//default"):
            cls = default.attrib.get("class")
            if not cls:
                # skip parent defaults
                continue

            if cls in class_overrides:
                overrides = class_overrides[cls]
                # for each child-tag you want to override (e.g. 'position', 'joint')
                for tag_name, attrs in overrides.items():
                    # find those child elements under this <default class="…">
                    for child in default.findall(tag_name):
                        for attr, val in attrs.items():
                            child.set(attr, str(val))

    comp = robot_elem.find(".//compiler")
    if comp is not None:
        # 1) pick up the paths (falling back to "assets" if missing)
        mesh_base = (robot_dir / comp.attrib.get("meshdir", "assets")).resolve()
        tex_base  = (robot_dir / comp.attrib.get("texturedir", "assets")).resolve()

        # 2) remove those attributes so they don't linger in the final XML
        comp.attrib.pop("meshdir", None)
        comp.attrib.pop("texturedir", None)
    else:
        mesh_base = (robot_dir / "assets").resolve()
        tex_base  = (robot_dir / "assets").resolve()

    for mesh in robot_elem.iter("mesh"):
        fname = mesh.attrib.get("file", "")
        # only rewrite relative paths
        if fname and not os.path.isabs(fname):
            mesh.set("file", str(mesh_base / fname))

    for tex in robot_elem.iter("texture"):
        fname = tex.attrib.get("file", "")
        if fname and not os.path.isabs(fname):
            tex.set("file", str(tex_base / fname))

    meshdir = _get_meshdir(robot_elem)
    assets = _find_assets(robot_elem, epath.Path(path), meshdir)

    robot_xml = ET.tostring(robot_elem, encoding="unicode")
    if scene is None:
        return mujoco.MjModel.from_xml_string(robot_xml, assets=assets)

    if scene and os.path.isabs(scene):
        scene_path = scene
    else:
        scene_path = get_scene(scene)
    scene_text = scene_path.read_text()
    if (robot_match := re.search(r"<mujoco model=\"(.*)\"", robot_xml)) is None:
        robot_name = "robot"
    else:
        robot_name = robot_match.group(1)
    scene_text = scene_text.format(name=robot_name, path=path)
    scene_elem = ET.fromstring(scene_text)

    # Find the <include> that brings in the robot.xml, drop it
    model_path = Path(path).resolve()
    model_name = model_path.name

    for inc in scene_elem.findall(".//include"):
        file_attr = inc.attrib.get("file", "")
        inc_path = Path(file_attr)

        # decide if this <include> refers to our robot.xml
        if inc_path.is_absolute():
            try:
                match = inc_path.resolve() == model_path
            except FileNotFoundError:
                match = False
        else:
            # relative: just compare the filenames
            match = (inc_path.name == model_name)

        if match:
            # remove it
            parent = None
            # xml.etree doesn’t have getparent(), so find the parent manually:
            for p in scene_elem.iter():
                if inc in list(p):
                    parent = p
                    break

            if parent is None:
                scene_elem.remove(inc)
            else:
                parent.remove(inc)
            break

    # 6) Inline the patched robot XML under <mujoco> in the scene
    #    (insert all children of robot_elem into scene_elem)
    for child in list(robot_elem):
        scene_elem.append(child)

    full_xml = ET.tostring(scene_elem, encoding="unicode")

    # 7) Assets may also include scene‑specific ones
    scene_meshdir = _get_meshdir(scene_elem) or "assets"
    assets.update(_find_assets(scene_elem, scene_path, scene_meshdir))

    # 8) Serialize & return
    full_xml = ET.tostring(scene_elem, encoding="unicode")
    return mujoco.MjModel.from_xml_string(full_xml, assets=assets)

