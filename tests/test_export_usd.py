"""Export as USD: the URDF's tree as text, in UsdGeom and UsdPhysics. The stage is
read back by USD's own library (`usd_read.py`, under a Python with `usd-core`, never
bpy's) and held to the URDF of the same program — its names, pivots, axes and ranges —
then posed by its own joints, as `test_export_joints` poses the URDF."""
import json
import math
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import bpy
from mathutils import Matrix

import sand3d as s3d
from sand3d import _core
from test_export_joints import axis_angle, machine, mat, mul, obj_points, urdf_fk

HERE = Path(__file__).resolve().parent
#: A Python with usd-core: $SAND3D_USD_PYTHON, which must then work, or .cache/venv-usd's.
USD_PYTHON = os.environ.get("SAND3D_USD_PYTHON") or str(HERE.parent / ".cache/venv-usd/bin/python")


def read_back(path):
    if not os.environ.get("SAND3D_USD_PYTHON") and not Path(USD_PYTHON).exists():
        raise unittest.SkipTest("no Python with usd-core: set SAND3D_USD_PYTHON")
    done = subprocess.run([USD_PYTHON, str(HERE / "usd_read.py"), str(path)], capture_output=True, text=True,
                          timeout=300)
    if done.returncode:
        raise AssertionError(f"usd_read.py failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout)


def quat_matrix(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def local(pose):
    return mat(quat_matrix(pose["orientation"]), pose["position"])


def inverse(m):
    r = [[m[c][rr] for c in range(3)] for rr in range(3)]
    return mat(r, [-sum(r[i][k] * m[k][3] for k in range(3)) for i in range(3)])


def unit(axis):
    return [1.0 if a == axis else 0.0 for a in "XYZ"]


def usd_fk(stage, values):
    """{body path: world matrix} for joint `values` (USD units: degrees, metres; by joint
    name), mimic joints followed."""
    world = {p: b["world"] for p, b in stage["bodies"].items()}
    left = dict(stage["joints"])
    while left:
        ready = [p for p, j in left.items() if j["body0"] not in [k["body1"] for k in left.values()]]
        for p in ready:
            j = left.pop(p)
            name = p.rsplit("/", 1)[1]
            q = values.get(name, 0.0)
            if j["mimic"]:
                q = j["mimic"]["coef0"] + j["mimic"]["coef1"] * values.get(j["mimic"]["joint"].rsplit("/", 1)[1], 0.0)
            a = unit(j["axis"])
            if j["type"] == "revolute":
                motion = mat(axis_angle(a, math.radians(q)), [0, 0, 0])
            else:
                motion = mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [q * v for v in a])
            world[j["body1"]] = mul(mul(mul(world[j["body0"]], local(j["local0"])), motion), inverse(local(j["local1"])))
    return world


def close(a, b, tol=1e-6):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)) < tol


class Usd(unittest.TestCase):
    """Read back by USD's own library."""

    @classmethod
    def setUpClass(cls):
        s3d.empty_scene()
        cls.root, cls.parts = machine()
        cls.tmp = Path(tempfile.mkdtemp())
        s3d.pose(cls.parts["door"], 70)                    # both files are the rest, whatever the pose
        s3d.export(cls.root, cls.tmp / "washer.usda")
        s3d.export(cls.root, cls.tmp / "washer.urdf")
        s3d.pose(cls.parts["door"], 0)
        cls.stage = read_back(cls.tmp / "washer.usda")
        cls.robot, cls.links = urdf_fk(cls.tmp / "washer.urdf", {})

    def name(self, path):
        return path.rsplit("/", 1)[1]

    def test_the_stage_is_valid_and_simready_shaped(self):
        st = self.stage
        self.assertEqual(st["errors"], [])
        self.assertEqual((st["defaultPrim"], st["upAxis"], st["metersPerUnit"]), ("washer", "Z", 1.0))
        self.assertNotIn("other", st)                       # no fixed joint, no collider
        (art,) = st["articulations"]
        self.assertEqual(art["path"], "/washer")
        self.assertEqual(art["joints"], sorted(st["joints"]))
        self.assertTrue(all(j["articulated"] for j in st["joints"].values()))
        self.assertTrue(all(p.startswith("/washer/Joints/") for p in st["joints"]))
        self.assertTrue(all(p.startswith("/washer/Geometry/") for p in st["bodies"]))

    def test_the_names_are_the_urdfs(self):
        st = self.stage
        links = {l.get("name") for l in self.robot.findall("link")}
        self.assertEqual({self.name(p) for p in st["bodies"]}, links)
        joints = {j.get("name"): j for j in self.robot.findall("joint")}
        self.assertEqual({self.name(p) for p in st["joints"]}, set(joints) - {"cord"})   # free: no joint
        for p, j in st["joints"].items():
            urdf = joints[self.name(p)]
            self.assertEqual((self.name(j["body0"]), self.name(j["body1"])),
                             (urdf.find("parent").get("link"), urdf.find("child").get("link")))
        visuals = {v.get("name"): l.get("name") for l in self.robot.findall("link") for v in l.findall("visual")}
        self.assertEqual({self.name(p): p.split("/")[-2] for p in st["meshes"]}, visuals)

    def test_bodies_pivots_and_axes_are_the_urdfs(self):
        st = self.stage
        for p, b in st["bodies"].items():
            self.assertTrue(close(b["world"], self.links[self.name(p)]), p)
        joints = {j.get("name"): j for j in self.robot.findall("joint")}
        for p, j in st["joints"].items():
            with self.subTest(joint=p):
                frame0 = mul(st["bodies"][j["body0"]]["world"], local(j["local0"]))
                frame1 = mul(st["bodies"][j["body1"]]["world"], local(j["local1"]))
                self.assertTrue(close(frame0, frame1), "the joint's two frames part at rest")
                child = self.links[self.name(j["body1"])]
                self.assertTrue(close([[r[3]] for r in frame0[:3]], [[r[3]] for r in child[:3]]))
                urdf_axis = [float(v) for v in joints[self.name(p)].find("axis").get("xyz").split()]
                want = [sum(child[r][c] * urdf_axis[c] for c in range(3)) for r in range(3)]
                got = [sum(frame0[r][c] * unit(j["axis"])[c] for c in range(3)) for r in range(3)]
                self.assertTrue(close([want], [got]), (want, got))

    def test_ranges_and_mimic_are_the_urdfs(self):
        joints = {j.get("name"): j for j in self.robot.findall("joint")}
        per_si = {"revolute": 180 / math.pi, "prismatic": 1.0}           # a USD unit per SI unit
        for p, j in self.stage["joints"].items():
            with self.subTest(joint=p):
                urdf = joints[self.name(p)]
                limit = urdf.find("limit")
                if urdf.get("type") == "continuous":
                    self.assertIsNone(limit)
                    self.assertIsNone(j["limit"])
                else:
                    want = [float(limit.get(k)) * per_si[j["type"]] for k in ("lower", "upper")]
                    self.assertTrue(close([want], [j["limit"]], 1e-5), (want, j["limit"]))
                mimic = urdf.find("mimic")
                if mimic is None:
                    self.assertIsNone(j["mimic"])
                    continue
                self.assertEqual(self.name(j["mimic"]["joint"]), mimic.get("joint"))
                lead = self.stage["joints"][j["mimic"]["joint"]]["type"]
                self.assertEqual(j["mimic"]["coef0"], 0)
                self.assertAlmostEqual(j["mimic"]["coef1"] * per_si[lead] / per_si[j["type"]],
                                       float(mimic.get("multiplier")), places=9)
        self.assertEqual(self.stage["joints"]["/washer/Joints/door_hinge"]["limit"], [0, 143])
        self.assertIsNone(self.stage["joints"]["/washer/Joints/knob_turn"]["limit"])     # continuous: ±inf

    def test_the_meshes_are_the_objs(self):
        for p, m in self.stage["meshes"].items():
            with self.subTest(mesh=p):
                obj = Path(self.tmp / "washer_meshes" / f"{self.name(p)}.obj").read_text().splitlines()
                pts = obj_points(self.tmp / "washer_meshes" / f"{self.name(p)}.obj")
                self.assertTrue(close(m["points"], pts, 1e-7))
                faces = [int(v) - 1 for line in obj if line.startswith("f ") for v in line.split()[1:]]
                self.assertEqual(m["indices"], faces)
                self.assertEqual(set(m["counts"]), {3})
                self.assertEqual((len(m["normals"]), m["interpolation"], m["subdivision"]),
                                 (len(faces), "faceVarying", "none"))
                for t in range(0, len(faces), 3):                # each corner's normal on its triangle's side
                    a, b, c = (pts[i] for i in faces[t:t + 3])
                    u, v = [q - p for p, q in zip(a, b)], [q - p for p, q in zip(a, c)]
                    n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
                    for corner in m["normals"][t:t + 3]:
                        self.assertGreater(sum(x * y for x, y in zip(n, corner)), 0)
                        self.assertAlmostEqual(math.hypot(*corner), 1, places=6)

    def test_the_materials(self):
        st = self.stage
        meshes = {self.name(p): m["material"] for p, m in st["meshes"].items()}
        self.assertIsNone(meshes["cab"])
        glass, plastic = st["materials"][meshes["door"]], st["materials"][meshes["drawer"]]
        self.assertEqual(meshes["door_2"], meshes["door"])
        self.assertEqual((glass["id"], glass["diffuseColor"], glass["opacity"]), ("UsdPreviewSurface", [1, 1, 1], 0))
        linear = [(c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4)
                  for c in (0x33, 0x66, 0x99)]
        self.assertTrue(close([plastic["diffuseColor"]], [linear], 1e-7))
        self.assertEqual((plastic["opacity"], plastic["roughness"]), (0.5, 0.5))

    def test_the_usd_posed_by_its_own_joints_is_where_pose_puts_the_parts(self):
        p = self.parts
        by_mesh = {"door_2": p["twin"]}
        for values in ({}, {"door_hinge": 90, "knob_turn": 30}, {"door_hinge": 143, "drawer_slide": 150},
                       {"cap_turn": 450}, {"door_hinge": 60, "knob_turn": -200, "cap_turn": 33}):
            with self.subTest(values=values):
                usd = {}
                for name, v in values.items():
                    if name == "door_hinge":
                        s3d.pose(p["door"], v)
                        s3d.pose(p["twin"], v)
                        usd["door_hinge_2"] = v
                    elif name == "cap_turn":
                        s3d.pose(p["cap"], v, joint=name)
                    else:
                        s3d.pose(p[name.split("_")[0]], v)
                    usd[name] = v / 1000 if name == "drawer_slide" else v
                world = usd_fk(self.stage, usd)
                for path, m in self.stage["meshes"].items():
                    name = self.name(path)
                    ob = by_mesh.get(name) or p[name]
                    body = world[path.rsplit("/", 1)[0]]
                    pts = [[sum(body[r][i] * v[i] for i in range(3)) + body[r][3] for r in range(3)] for v in m["points"]]
                    w = _core.world(ob)
                    want = [w @ v.co / 1000 for v in ob.data.vertices]
                    err = max(math.dist(a, b) for a, b in zip(pts, want))
                    self.assertLess(err, 2e-6, name)
                for name in ("door", "twin", "knob", "drawer"):
                    s3d.pose(p[name], 0)
                s3d.pose(p["cap"], 0, joint="cap_turn")


class Text(unittest.TestCase):
    """What the text says, without USD's library."""

    def setUp(self):
        s3d.empty_scene()
        self.tmp = Path(tempfile.mkdtemp())

    def test_the_usda_is_the_rest_whatever_the_pose(self):
        root, parts = machine()
        s3d.export(root, self.tmp / "a.usda")
        s3d.pose(parts["door"], 90)
        s3d.pose(parts["drawer"], 100)
        s3d.export(root, self.tmp / "b.usda")
        self.assertEqual((self.tmp / "a.usda").read_bytes(), (self.tmp / "b.usda").read_bytes())

    def test_the_scene_is_left_as_it_was(self):
        root, _ = machine()
        before = sorted((o.name, tuple(map(tuple, o.matrix_basis))) for o in bpy.data.objects)
        meshes, mats = sorted(m.name for m in bpy.data.meshes), sorted(m.name for m in bpy.data.materials)
        s3d.export(root, self.tmp / "washer.usda")
        self.assertEqual(sorted((o.name, tuple(map(tuple, o.matrix_basis))) for o in bpy.data.objects), before)
        self.assertEqual(sorted(m.name for m in bpy.data.meshes), meshes)
        self.assertEqual(sorted(m.name for m in bpy.data.materials), mats)

    def test_binary_usd_is_refused(self):
        root, _ = machine()
        for suffix, tool in ((".usdc", "usdcat x.usda -o x.usdc"), (".usd", "usdcat x.usda -o x.usd"),
                             (".USDZ", "usdzip")):
            with self.assertRaisesRegex(ValueError, rf"^export: {suffix.lower()} is written by USD's own library, "
                                                    rf"which sand3d does not take: write \.usda, USD as text, and "
                                                    rf"convert it with USD's tools \({re.escape(tool)}\)$"):
                s3d.export(root, self.tmp / f"washer{suffix}")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_an_unknown_extension_is_refused(self):
        """It used to write a GLB under any other name: washer.stl was a GLB."""
        root, _ = machine()
        for name, said in (("washer.ply", ".ply"), ("washer.FBX", ".fbx"),
                           ("washer", "a path with no extension")):
            with self.assertRaisesRegex(ValueError, rf"^export: {re.escape(said)} is not a format sand3d writes: "
                                                    rf"name the file \.glb, \.gltf, \.obj, \.stl, \.urdf or \.usda$"):
                s3d.export(root, self.tmp / name)
        self.assertEqual(list(self.tmp.iterdir()), [])
        s3d.export(root, self.tmp / "washer.GLB")
        self.assertTrue((self.tmp / "washer.GLB").is_file())

    def test_names_become_identifiers(self):
        base = s3d.box("2 base", 100, 100, 10)
        lids = [s3d.at(s3d.box(n, 50, 50, 5), z=10) for n in ("lid-a", "lid_a")]
        for lid in lids:
            s3d.joint("hinge #1", lid, s3d.bbox(base, "+X+Z"), axis="+Y", range=(0, 90))
        s3d.export(s3d.group("box café", base, *lids), self.tmp / "b.usda")
        text = (self.tmp / "b.usda").read_text()
        self.assertIn('defaultPrim = "box_caf_"', text)
        for prim in ('Xform "box_caf_"', 'Mesh "_2_base"', 'Xform "lid_a"', 'Mesh "lid_a_2"',
                     'RevoluteJoint "hinge__1"', 'RevoluteJoint "hinge__1_2"', '</box_caf_/Geometry/lid_a_2>'):
            self.assertIn(prim, text)
        self.assertTrue(text.isascii())

    def test_an_object_with_no_joint_is_one_rigid_body(self):
        root = s3d.group("thing", s3d.box("b", 10, 10, 10), s3d.at(s3d.cylinder("c", 4, 20), x=20))
        s3d.export(root, self.tmp / "thing.usda")
        text = (self.tmp / "thing.usda").read_text()
        self.assertNotIn("Articulation", text)
        self.assertNotIn("Joints", text)
        self.assertNotIn("Looks", text)
        self.assertEqual(text.count("PhysicsRigidBodyAPI"), 1)
        self.assertEqual(re.findall(r'def Mesh "(\w+)"', text), ["b", "c"])

    def test_the_program_unit_goes_to_metres(self):
        s3d.units("cm")
        cab = s3d.box("cab", 60, 60, 85)
        door = s3d.at(s3d.box("door", 2, 56, 80), x=31, z=2.5)
        s3d.joint("slide", door, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 20))
        s3d.export(s3d.group("w", cab, door), self.tmp / "cm.usda")
        text = (self.tmp / "cm.usda").read_text()
        self.assertIn("point3f physics:localPos0 = (0.3, 0, 0.425)", text)
        self.assertIn("float physics:upperLimit = 0.2", text)
        self.assertIn("float3[] extent = [(-0.3, -0.3, 0), (0.3, 0.3, 0.85)]", text)

    def test_a_chain_of_followers_follows_its_head(self):
        base = s3d.box("base", 300, 100, 10)
        gears = [s3d.at(s3d.cylinder(f"g{i}", 20, 10), x=100 * i - 100, z=10) for i in range(3)]
        s3d.joint("a", gears[0], base.face("top").move(x=-100), range=(0, 360))
        s3d.joint("b", gears[1], base.face("top"), follows="a", ratio=-2)
        s3d.joint("c", gears[2], base.face("top").move(x=100), follows="b", ratio=-0.25)
        s3d.export(s3d.group("train", base, *gears), self.tmp / "t.usda")
        text = (self.tmp / "t.usda").read_text()
        self.assertEqual(re.findall(r"newton:mimicJoint = <([^>]+)>\n.*\n.*newton:mimicCoef1 = (\S+)", text),
                         [("/train/Joints/a", "-2"), ("/train/Joints/a", "0.5")])

    def test_a_joint_hanging_outside_what_is_exported(self):
        _, p = machine()
        front = s3d.group("front", p["door"], p["knob"])
        with self.assertRaisesRegex(ValueError, r"^export: 'door' hangs on 'cab', which is not in 'front': "
                                                r"export a group that holds both$"):
            s3d.export(front, self.tmp / "front.usda")
        self.assertEqual(list(self.tmp.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
