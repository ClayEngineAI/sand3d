"""Clay's node-extras contract and export-only UVs, read from the actual GLB."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import bpy
import sand3d as s3d
from sand3d import materials
from harness import digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    raw = path.read_bytes()
    size = struct.unpack_from("<I", raw, 12)[0]
    return json.loads(raw[20:20 + size]), raw[28 + size:]


def rows(doc, binary, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    n = {"VEC2": 2, "VEC3": 3}[a["type"]]
    offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [struct.unpack_from(f"<{n}f", binary, offset + i * view.get("byteStride", n * 4))
            for i in range(a["count"])]


class ClayMaterials(unittest.TestCase):
    def setUp(self):
        s3d.empty_scene()

    def test_plain_articulated_and_animated_exports_keep_tags_and_original_meshes(self):
        for articulated in (False, True):
            s3d.empty_scene()
            base = s3d.box("base", 1000, 1000, 1000)
            child = s3d.at(s3d.box("child", 100, 100, 100), z=1000)
            if articulated:
                s3d.joint("hinge", child, s3d.local(base), range=(0, 90))
            root = s3d.assembly("asset", base, child)
            s3d.material(root, "wood", color="#7a4a26", roughness=0.6, repeat=2)
            s3d.material(child, "metal", metallic=0.2, opacity=0.5, emission=3, repeat=(3, 4))
            source = base.data
            base["clayMaterial"] = {"previous": 1}
            before = digest()
            with tempfile.TemporaryDirectory() as tmp:
                for animated in (False, True):
                    kwargs = {}
                    if animated:
                        motion = s3d.keyframes(child, {0: 0, 1: 60}) if articulated else s3d.keyframes(child, z={0: 0, 1: 100})
                        kwargs["animations"] = [s3d.clip("move", motion)]
                    path = Path(tmp) / "asset.glb"
                    s3d.export(root, path, **kwargs)
                    doc, binary = read(path)
                    for mesh in doc["meshes"]:
                        for primitive in mesh["primitives"]:
                            self.assertIn("TEXCOORD_0", primitive["attributes"])
                            self.assertNotIn("TANGENT", primitive["attributes"])
                    tags = {n["name"]: n["extras"]["clayMaterial"] for n in doc["nodes"] if "mesh" in n}
                    self.assertEqual(tags["base"], {"preset": "wood", "parameters": {"color": 8014374, "roughness": 0.6}, "repeat": [2, 2]})
                    self.assertEqual(tags["child"], {"preset": "metal", "parameters": {"metalness": 0.2, "opacity": 0.5,
                                     "transparent": True, "emissive": 0xbdc3c9, "emissiveIntensity": 3}, "repeat": [3, 4]})
                    self.assertIs(base.data, source)
                    self.assertEqual(len(source.uv_layers), 0)
                    self.assertEqual(base["clayMaterial"].to_dict(), {"previous": 1})
                    self.assertNotIn("clayMaterial", child)
                    self.assertEqual(digest(), before)

    def test_box_projection_metres_all_faces_and_units(self):
        for unit, size in (("mm", 1000), ("cm", 100), ("m", 1)):
            s3d.empty_scene()
            s3d.units(unit)
            ob = s3d.box("box", size, size, size)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "box.glb"
                s3d.export(ob, path)
                doc, binary = read(path)
                attrs = doc["meshes"][0]["primitives"][0]["attributes"]
                pos, normal, uv = (rows(doc, binary, attrs[name]) for name in ("POSITION", "NORMAL", "TEXCOORD_0"))
                for p, n, actual in zip(pos, normal, uv):
                    x, y, z = p[0] / size, -p[2] / size, p[1] / size
                    nx, ny, nz = n[0], -n[2], n[1]
                    if abs(nz) > 0.9:
                        wanted = (x, 1 + (y if nz > 0 else -y))
                    elif abs(ny) > 0.9:
                        wanted = (-x if ny > 0 else x, 1 + z)
                    else:
                        wanted = (y if nx > 0 else -y, 1 + z)
                    for a, b in zip(actual, wanted):
                        self.assertAlmostEqual(a, b, places=6)

    def test_repeat_validation_defaults_and_removed_names(self):
        ob = s3d.box("box", 1, 1, 1)
        s3d.material(ob, "brick")
        self.assertEqual(materials.clay_tag(ob[materials.TAG]), {"preset": "brick", "parameters": {}, "repeat": [1, 1]})
        for repeat in (0, -1, True, (1,), (1, 0), (1, float("inf")), "2"):
            with self.subTest(repeat=repeat), self.assertRaisesRegex(ValueError, "repeat"):
                s3d.material(ob, "wood", repeat=repeat)
        for kind in ("oak", "steel", "rubber", "light", "painted_metal"):
            with self.assertRaisesRegex(ValueError, "no preset"):
                s3d.material(ob, kind)
        self.assertEqual(len(materials.PRESETS), 25)

    def test_two_fresh_processes_export_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"prop-{i}.glb" for i in range(2)]
            for i, path in enumerate(paths):
                subprocess.run([sys.executable, str(ROOT / "docs/examples/clay_materials.py"), str(path)],
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": str(i)},
                               check=True, capture_output=True, timeout=60)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
