"""Bake clips into standard glTF channels, using Blender only for the static asset.

The source scene never gets actions or temporary poses. The temporary hierarchy is
the existing joint hierarchy, evaluated by the same kinematics as sample(). All
animation accessors are ordinary FLOAT data; custom easing needs no runtime code.
"""
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from urllib.parse import quote

import bpy
from mathutils import Matrix

from . import joints as J, materials, urdf
from .animation import _Clip, _check_movement, _number

_MARKER = "s3d_animation_node"
_Y_UP = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def _read_glb(path):
    raw = Path(path).read_bytes()
    doc, binary, offset = None, bytearray(), 12
    while offset < len(raw):
        size, kind = struct.unpack_from("<I4s", raw, offset)
        chunk = raw[offset + 8:offset + 8 + size]
        if kind == b"JSON":
            doc = json.loads(chunk)
        elif kind == b"BIN\0":
            binary = bytearray(chunk)
        offset += 8 + size
    return doc, binary


def _float32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        raise ValueError("export: animation values exceed glTF float32 precision") from None
    if not math.isfinite(result):
        raise ValueError("export: animation values must be finite float32 numbers")
    return result


def _before(time):
    bits = struct.unpack("<I", struct.pack("<f", time))[0]
    return struct.unpack("<f", struct.pack("<I", bits - 1))[0] if bits else 0.0


def _times(animation, fps):
    end = animation.duration
    count = math.ceil(end * fps)
    if count > 1_000_000:
        raise ValueError("export: clip exceeds one million samples; lower fps or shorten it")
    times = {0.0, _float32(end)}
    times.update(_float32(i / fps) for i in range(count) if i / fps <= end)
    for track in animation.tracks:
        for _, curve in track.curves:
            converted = [_float32(t) for t, _ in curve.keys]
            if len(set(converted)) != len(converted):
                raise ValueError("export: distinct key times collapse at glTF float32 precision")
            times.update(converted)
            if curve.ease == "step":
                times.update(_before(t) for t in converted[1:] if t > 0)
    return sorted(times)


def _local_matrices(mech, k):
    world = {ob.as_pointer(): k.posed(ob) for ob in mech.members}
    root = mech.root.as_pointer()
    result = {(root, None): Matrix.Scale(urdf._metres(), 4) @ world[root]}
    bodies = {ob.as_pointer() for ob in mech.bodies}
    for ob in mech.members:
        key = ob.as_pointer()
        if key == root:
            continue
        if key in bodies:
            above = world[J.parent_of(ob).as_pointer()]
            for i, frame in enumerate(k.frames(ob)):
                result[key, i] = above.inverted() @ frame
                above = frame
        else:
            parent = ob.parent.as_pointer() if ob.parent is not None else root
            above = world.get(parent, world[root])
        result[key, None] = above.inverted() @ world[key]
    return result


def _trs(matrix):
    matrix = _Y_UP @ matrix @ _Y_UP.inverted()
    location, rotation, scale = matrix.decompose()
    rebuilt = Matrix.LocRotScale(location, rotation, scale)
    if any(abs(matrix[r][c] - rebuilt[r][c]) > 2e-5 * max(1, abs(matrix[r][c]))
           for r in range(4) for c in range(4)):
        raise ValueError("export: an animated transform has shear, which glTF TRS cannot represent")
    return {"translation": tuple(location), "rotation": (rotation.x, rotation.y, rotation.z, rotation.w),
            "scale": tuple(scale)}


def _angular_values(animation, time, k):
    values = [j["value"] for key in sorted(k.data) for j in k.data[key]
              if j["kind"] in ("revolute", "continuous")]
    values += [curve.at(time) for track in animation.tracks for name, curve in track.curves
               if name in ("rx", "ry", "rz")]
    return values


def _bake(mech, animation, fps):
    samples = {}
    key_times = {_float32(t): t for track in animation.tracks for _, curve in track.curves for t, _ in curve.keys}

    def evaluate(t):
        if t not in samples:
            authored_time = key_times.get(t, t)
            k, values, _ = animation.evaluate(authored_time)
            _check_movement(values)
            samples[t] = (_local_matrices(mech, k), _angular_values(animation, authored_time, k))
        return samples[t]

    times = _times(animation, fps)
    step_times = {_float32(t) for track in animation.tracks for _, curve in track.curves
                  if curve.ease == "step" for t, _ in curve.keys[1:]}

    def refine(a, b, depth=0):
        av, bv = evaluate(a)[1], evaluate(b)[1]
        # Do not interpolate an intentional step across its one-ULP boundary.
        if b in step_times and a == _before(b):
            return
        middle = _float32((a + b) / 2)
        if not a < middle < b:
            return
        mv = evaluate(middle)[1]
        travel = max((abs(x - y) for x, y in zip(av, bv)), default=0)
        travel = max(travel, max((abs(x - y) for x, y in zip(av, mv)), default=0),
                     max((abs(x - y) for x, y in zip(mv, bv)), default=0))
        if travel > 90:
            if depth >= 20:
                raise ValueError("export: rotation changes too fast to bake; shorten the clip or reduce its speed")
            refine(a, middle, depth + 1)
            refine(middle, b, depth + 1)
        else:
            # Midpoint checks catch rapid winding; fps remains the easing bake resolution.
            samples.pop(middle, None)

    for a, b in zip(times, times[1:]):
        refine(a, b)
    for t in times:
        evaluate(t)
    return sorted(samples.items())


def _interpolation(animation, key, path):
    pointer, index = key
    if index is not None:
        ob = next(ob for ob in animation.members if ob.as_pointer() == pointer)
        while J.joints_of(ob)[index].get("follows") is not None:
            j = J.joints_of(ob)[index]
            ob, index = J.leader_of(ob, index), j["follows"]
        curves = [curve for track in animation.tracks if track.ob == ob and track.joint == index
                  for _, curve in track.curves]
    else:
        names = {"translation": ("x", "y", "z"), "rotation": ("rx", "ry", "rz"), "scale": ()}[path]
        curves = [curve for track in animation.tracks if track.ob.as_pointer() == pointer and track.joint is None
                  for name, curve in track.curves if name in names]
    return "STEP" if curves and all(curve.ease == "step" for curve in curves) else "LINEAR"


def _accessor(doc, binary, rows, kind, bounds=False):
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[kind]
    binary.extend(b"\0" * (-len(binary) % 4))
    offset = len(binary)
    for row in rows:
        binary.extend(struct.pack(f"<{width}f", *(_float32(v) for v in row)))
    views = doc.setdefault("bufferViews", [])
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset})
    entry = {"bufferView": len(views) - 1, "componentType": 5126, "count": len(rows), "type": kind}
    if bounds:
        entry.update(min=[min(row[i] for row in rows) for i in range(width)],
                     max=[max(row[i] for row in rows) for i in range(width)])
    accessors = doc.setdefault("accessors", [])
    accessors.append(entry)
    return len(accessors) - 1


def _append(doc, binary, mech, animations, fps, node_indices):
    doc["animations"] = []
    for animation in animations:
        samples = _bake(mech, animation, fps)
        input_index = _accessor(doc, binary, [(t,) for t, _ in samples], "SCALAR", bounds=True)
        entry = {"name": animation.name, "samplers": [], "channels": []}
        for key, index in node_indices.items():
            frames = [_trs(matrices[key]) for _, (matrices, _) in samples]
            # Every clip carries its baseline, so switching clips does not inherit
            # the previous clip's unkeyed transforms.
            for path, kind in (("translation", "VEC3"), ("rotation", "VEC4"), ("scale", "VEC3")):
                rows = [frame[path] for frame in frames]
                if path == "rotation":
                    for i in range(1, len(rows)):
                        if sum(a * b for a, b in zip(rows[i - 1], rows[i])) < 0:
                            rows[i] = tuple(-v for v in rows[i])
                output = _accessor(doc, binary, rows, kind)
                entry["channels"].append({"sampler": len(entry["samplers"]),
                                          "target": {"node": index, "path": path}})
                entry["samplers"].append({"input": input_index, "output": output,
                                          "interpolation": _interpolation(animation, key, path)})
            node = doc["nodes"][index]
            if "matrix" in node:
                node.pop("matrix")
                node.update(frames[0])
        doc["animations"].append(entry)


def _save(doc, binary, path):
    path = Path(path)
    doc["buffers"] = [{"byteLength": len(binary)}]
    if path.suffix.lower() == ".gltf":
        bin_path = path.with_suffix(".bin")
        doc["buffers"][0]["uri"] = quote(bin_path.name)
        bin_path.write_bytes(binary)
        path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        raw = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
        raw += b" " * (-len(raw) % 4)
        binary.extend(b"\0" * (-len(binary) % 4))
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(raw) + len(binary))
                         + struct.pack("<I4s", len(raw), b"JSON") + raw
                         + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def write(root, path, animations, fps):
    from .assembly import _lend_meshes, _return_meshes
    fps = _number(fps, "export fps")
    if fps <= 0:
        raise ValueError("export: fps must be positive")
    animations = [animations] if isinstance(animations, _Clip) else list(animations)
    if not animations or any(not isinstance(a, _Clip) for a in animations):
        raise ValueError("export: animations must contain one or more clips")
    if len({a.name for a in animations}) != len(animations):
        raise ValueError("export: clip names must be unique")
    mech = urdf.Mechanism(root)
    mech.check()
    for animation in animations:
        if any(track.ob.as_pointer() not in mech.ids for track in animation.tracks):
            raise ValueError("export: a clip targets a part outside the exported asset")
        if any(ob.as_pointer() not in animation.bases for ob in mech.members):
            raise ValueError("export: model changed after clip creation; rebuild the clip")
    materials.resolve()
    baseline = animations[0].evaluate(0)[0]
    nodes, state, lent, coll, tags = {}, None, None, None, {}
    original_names = {ob: ob.name for ob in mech.members}
    try:
        lent = _lend_meshes()
        coll = bpy.data.collections.new("s3d~animation_export")
        bpy.context.scene.collection.children.link(coll)
        state = urdf.glb_tree(root, coll, kinematics=baseline, nodes=nodes)
        for i, node in enumerate(nodes.values()):
            node[_MARKER] = i
        dressed = materials.dressed()
        tags = {m: m.pop(materials.TAG) for m in bpy.data.materials if materials.TAG in m}
        with tempfile.TemporaryDirectory() as directory:
            temporary = os.path.join(directory, "asset.glb")
            bpy.ops.export_scene.gltf(filepath=temporary, export_format="GLB", collection=coll.name,
                                      export_extras=True, export_animations=False, export_tangents=False,
                                      export_materials="EXPORT" if dressed else "NONE")
            doc, binary = _read_glb(temporary)
        exported = {}
        keys = list(nodes)
        for i, node in enumerate(doc["nodes"]):
            marker = node.get("extras", {}).pop(_MARKER, None)
            if marker is not None:
                exported[keys[marker]] = i
            if not node.get("extras"):
                node.pop("extras", None)
        if set(exported) != set(nodes):
            raise ValueError("export: glTF exporter omitted an animated node")
        _append(doc, binary, mech, animations, fps, exported)
        _save(doc, binary, path)
    finally:
        for material, tag in tags.items():
            material[materials.TAG] = tag
        if state is not None:
            urdf.undo(state, coll)
        elif coll is not None:
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob)
            bpy.data.collections.remove(coll)
            for ob, name in original_names.items():
                ob.name = name
        if lent is not None:
            _return_meshes(lent)
