"""Export of an articulated object: the URDF at rest, read back and posed by its own
forward kinematics; the GLB nested by its joints, the joints in its extras."""
import json
import math
import re
import os
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy  # noqa: F401
from mathutils import Matrix, Quaternion, Vector

import sand3d as s3d
from sand3d import _core
from sand3d.joints import joints_of


def joints_pivot(ob, k):
    flat = joints_of(ob)[k]["pivot"]
    return [flat[4 * r:4 * r + 4] for r in range(4)]


def machine():
    """Cabinet, a door and its mirror image, a drawer, a screw cap, a free plug, a knob on
    the door; materials on two of them."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.at(s3d.cylinder("door", 420, 30), on=cab.face("+X").move(z=400))
    s3d.material(door, "glass")
    s3d.joint("door_hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
    knob = s3d.at(s3d.cylinder("knob", 30, 15), on=door.face("top").move(x=100))
    s3d.joint("knob_turn", knob, door.face("top").move(x=100), kind="continuous")
    drawer = s3d.at(s3d.box("drawer", 40, 150, 60), x=290, y=-180, z=780)
    s3d.material(drawer, "plastic", color="#336699", opacity=0.5)
    s3d.joint("drawer_slide", drawer, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
    twin = s3d.transform(s3d.copy(door), mirror="y")
    cap = s3d.at(s3d.cylinder("cap", 40, 20), z=850)
    s3d.joint("cap_turn", cap, cab.face("top"), range=(0, 720))
    s3d.joint("cap_lift", cap, cab.face("top"), kind="prismatic", follows="cap_turn", ratio=3 / 360)
    plug = s3d.at(s3d.box("plug", 20, 20, 20), x=-400)
    s3d.joint("cord", plug, cab.face("-X"), kind="free")
    root = s3d.group("washer", cab, door, knob, drawer, twin, cap, plug)
    return root, dict(cab=cab, door=door, knob=knob, drawer=drawer, twin=twin, cap=cap, plug=plug)


# ---------------------------------------------------------------- reading a URDF back, in double precision

def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def mat(rot, t):
    return [rot[0] + [t[0]], rot[1] + [t[1]], rot[2] + [t[2]], [0, 0, 0, 1]]


def mul(a, b):
    return [[sum(a[r][i] * b[i][c] for i in range(4)) for c in range(4)] for r in range(4)]


def axis_angle(u, a):
    x, y, z = u
    c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def urdf_fk(path, values):
    """{link: world matrix} for joint `values` (URDF units), mimic joints followed."""
    robot = ET.parse(path).getroot()
    joints = {}
    for j in robot.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        axis = j.find("axis")
        mimic = j.find("mimic")
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), origin=mat(rpy_matrix(*rpy), xyz),
                                     axis=[float(v) for v in axis.get("xyz").split()] if axis is not None else None,
                                     mimic=(mimic.get("joint"), float(mimic.get("multiplier"))) if mimic is not None
                                     else None)
    by_child = {j["child"]: (name, j) for name, j in joints.items()}
    out = {}

    def world(link):
        if link in out:
            return out[link]
        if link not in by_child:
            out[link] = mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])
            return out[link]
        name, j = by_child[link]
        q = values.get(name, 0.0)
        if j["mimic"]:
            q = j["mimic"][1] * values.get(j["mimic"][0], 0.0)
        if j["type"] in ("revolute", "continuous"):
            motion = mat(axis_angle(j["axis"], q), [0, 0, 0])
        elif j["type"] == "prismatic":
            motion = mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [q * a for a in j["axis"]])
        else:
            motion = mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])
        out[link] = mul(mul(world(j["parent"]), j["origin"]), motion)
        return out[link]
    for link in robot.findall("link"):
        world(link.get("name"))
    return robot, out


def obj_points(path):
    return [[float(v) for v in line.split()[1:]] for line in Path(path).read_text().splitlines()
            if line.startswith("v ")]


def placed(m, pts):
    return [[sum(m[r][i] * p[i] for i in range(3)) + m[r][3] for r in range(3)] for p in pts]


def glb_json(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length])


class Urdf(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.root, self.parts = machine()
        self.tmp = Path(tempfile.mkdtemp())

    def export(self, name="washer.urdf"):
        s3d.export(self.root, self.tmp / name)
        return self.tmp / name

    def test_links_joints_and_their_words(self):
        robot = ET.parse(self.export()).getroot()
        self.assertEqual(robot.get("name"), "washer")
        links = [l.get("name") for l in robot.findall("link")]
        self.assertEqual(links, ["washer", "cap_cap_turn", "cap", "door", "door_2", "drawer", "plug", "knob"])
        joints = {j.get("name"): j for j in robot.findall("joint")}
        self.assertEqual({n: j.get("type") for n, j in joints.items()},
                         {"cap_turn": "revolute", "cap_lift": "prismatic", "door_hinge": "revolute",
                          "knob_turn": "continuous", "door_hinge_2": "revolute", "drawer_slide": "prismatic",
                          "cord": "floating"})
        self.assertEqual(joints["knob_turn"].find("parent").get("link"), "door")     # a copy of the door: no knob
        limit = joints["door_hinge"].find("limit")
        self.assertEqual((limit.get("lower"), limit.get("upper")), ("0", f"{math.radians(143):.9g}"))
        self.assertEqual((limit.get("effort"), limit.get("velocity")), ("0", "0"))
        self.assertEqual(joints["drawer_slide"].find("limit").get("upper"), "0.2")
        mimic = joints["cap_lift"].find("mimic")
        self.assertEqual(mimic.get("joint"), "cap_turn")
        self.assertAlmostEqual(float(mimic.get("multiplier")), 0.003 / (2 * math.pi), places=12)
        self.assertEqual(joints["cap_lift"].find("limit").get("upper"), "0.006")
        self.assertIsNone(joints["cord"].find("axis"))
        self.assertIsNone(joints["knob_turn"].find("limit"))

    def test_the_colours_material_set(self):
        robot = ET.parse(self.export()).getroot()
        colours = {v.get("name"): v.find("material") for v in robot.iter("visual")}
        self.assertIsNone(colours["cab"])
        self.assertEqual(colours["door"].get("name"), "glass")
        self.assertEqual(colours["door"].find("color").get("rgba"), "1 1 1 1")
        self.assertEqual(colours["drawer"].find("color").get("rgba"), f"0.2 0.4 {0x99 / 255:.9g} 0.5")

    def test_the_urdf_posed_by_its_own_kinematics_is_where_pose_puts_the_parts(self):
        path = self.export()
        p = self.parts
        for values in ({}, {"door_hinge": 90, "knob_turn": 30}, {"door_hinge": 143, "drawer_slide": 150},
                       {"cap_turn": 450}, {"door_hinge": 60, "knob_turn": -200, "cap_turn": 33}):
            with self.subTest(values=values):
                twin_values = {}
                for name, v in values.items():
                    if name == "door_hinge":
                        s3d.pose(p["door"], v)
                        s3d.pose(p["twin"], v)
                        twin_values["door_hinge_2"] = math.radians(v)
                    elif name == "knob_turn":
                        s3d.pose(p["knob"], v)
                    elif name == "drawer_slide":
                        s3d.pose(p["drawer"], v)
                    else:
                        s3d.pose(p["cap"], v, joint=name)
                si = {n: (v / 1000 if n == "drawer_slide" else math.radians(v)) for n, v in values.items()}
                robot, links = urdf_fk(path, {**si, **twin_values})
                for link in robot.findall("link"):
                    for vis in link.findall("visual"):
                        name = vis.get("name")
                        ob = p["twin"] if name == "door_2" else p[name]
                        pts = placed(links[link.get("name")], obj_points(self.tmp / vis.find("geometry/mesh").get(
                            "filename")))
                        w = _core.world(ob)
                        want = [w @ v.co / 1000 for v in ob.data.vertices]
                        err = max(math.dist(a, b) for a, b in zip(pts, want))
                        self.assertLess(err, 1e-6, name)
                for name in ("door", "twin", "knob", "drawer"):
                    s3d.pose(p[name], 0)
                s3d.pose(p["cap"], 0, joint="cap_turn")

    def test_the_urdf_is_the_rest_whatever_the_pose(self):
        first = self.export("a.urdf").read_text()
        s3d.pose(self.parts["door"], 90)
        s3d.pose(self.parts["drawer"], 100)
        self.assertEqual(self.export("b.urdf").read_text().replace("b_meshes", "a_meshes"), first)
        self.assertEqual((self.tmp / "a_meshes/door.obj").read_text(), (self.tmp / "b_meshes/door.obj").read_text())

    def test_a_mirror_image_is_baked_into_its_link(self):
        self.export()
        door = (self.tmp / "washer_meshes/door.obj").read_text().splitlines()
        twin = (self.tmp / "washer_meshes/door_2.obj").read_text().splitlines()
        faces = [l for l in door if l.startswith("f ")]
        flipped = [l for l in twin if l.startswith("f ")]
        self.assertEqual(flipped[0].split()[1:], faces[0].split()[1:][::-1])

    def test_the_program_unit_goes_to_metres(self):
        s3d.empty_scene()
        s3d.units("cm")
        cab = s3d.box("cab", 60, 60, 85)
        door = s3d.at(s3d.box("door", 2, 56, 80), x=31, z=2.5)
        s3d.joint("slide", door, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 20))
        path = self.tmp / "cm.urdf"
        s3d.export(s3d.group("w", cab, door), path)
        joint = ET.parse(path).getroot().find("joint")
        self.assertEqual(joint.find("origin").get("xyz"), "0.3 0 0.425")
        self.assertEqual(joint.find("limit").get("upper"), "0.2")

    def test_the_scene_is_left_as_it_was(self):
        before = sorted((o.name, tuple(map(tuple, o.matrix_basis))) for o in bpy.data.objects)
        meshes = sorted(m.name for m in bpy.data.meshes)
        self.export()
        s3d.export(self.root, self.tmp / "washer.glb")
        self.assertEqual(sorted((o.name, tuple(map(tuple, o.matrix_basis))) for o in bpy.data.objects), before)
        self.assertEqual(sorted(m.name for m in bpy.data.meshes), meshes)
        self.assertEqual([c.name for c in bpy.data.collections], [])

    def test_a_joint_hanging_on_a_deleted_part(self):
        s3d.delete(self.parts["cab"])
        with self.assertRaisesRegex(ValueError, r"^export: 'cap' hangs on a part that was deleted$"):
            self.export()

    def test_a_joint_hanging_outside_what_is_exported(self):
        p = self.parts
        front = s3d.group("front", p["door"], p["knob"])
        before = sorted(o.name for o in bpy.data.objects)
        for name in ("front.urdf", "front.glb"):
            with self.assertRaisesRegex(ValueError, r"^export: 'door' hangs on 'cab', which is not in 'front': "
                                                    r"export a group that holds both$"):
                s3d.export(front, self.tmp / name)
        self.assertEqual(sorted(o.name for o in bpy.data.objects), before)       # the scene as it was
        self.assertEqual([c.name for c in bpy.data.collections], [])
        plate = s3d.at(s3d.box("plate", 100, 100, 10), x=1000)
        tab = s3d.at(s3d.box("tab", 10, 10, 10), x=1000, z=10)
        s3d.joint("tab_turn", tab, plate.face("+Z"), kind="continuous", follows=p["door"])
        with self.assertRaisesRegex(ValueError, r"^export: 'tab_turn' on 'tab' follows a joint of 'door', not in "
                                                r"'dial': export a group that holds both$"):
            s3d.export(s3d.group("dial", plate, tab), self.tmp / "dial.urdf")


class Glb(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.root, self.parts = machine()
        self.path = Path(tempfile.mkdtemp()) / "washer.glb"

    def test_nodes_nest_by_joint_and_carry_it(self):
        s3d.pose(self.parts["door"], 90)
        s3d.export(self.root, self.path)
        gltf = glb_json(self.path)
        nodes = gltf["nodes"]
        by_name = {n["name"]: i for i, n in enumerate(nodes)}
        children = {i: n.get("children", []) for i, n in enumerate(nodes)}
        parent = {c: i for i, cs in children.items() for c in cs}
        self.assertEqual(nodes[parent[by_name["door_hinge"]]]["name"], "cab")
        self.assertEqual(nodes[parent[by_name["door"]]]["name"], "door_hinge")
        self.assertEqual(nodes[parent[by_name["knob_turn"]]]["name"], "door")
        self.assertEqual(nodes[parent[by_name["cap_lift"]]]["name"], "cap_turn")
        hinge = nodes[by_name["door_hinge"]]["extras"]["s3d_joint"]
        self.assertEqual(hinge["type"], "revolute")
        self.assertAlmostEqual(hinge["value"], math.pi / 2, places=6)
        self.assertAlmostEqual(hinge["upper"], math.radians(143), places=6)
        lift = nodes[by_name["cap_lift"]]["extras"]["s3d_joint"]
        self.assertEqual(lift["mimic"]["joint"], "cap_turn")
        for n in nodes:
            self.assertEqual(set(n.get("extras", {})) - {"s3d_joint", "clayMaterial"}, set(), n["name"])
        for m in gltf.get("materials", []):
            self.assertNotIn("extras", m)
        for m in bpy.data.materials:                     # the scene keeps its tags
            self.assertIn("s3d_material", m)
        for v in nodes[by_name["washer"]]["scale"]:
            self.assertAlmostEqual(v, 0.001, places=9)

    def test_a_joint_axis_is_in_its_node_frame(self):
        s3d.export(self.root, self.path)
        nodes = glb_json(self.path)["nodes"]
        parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

        def linear(n):
            q, s = n.get("rotation", [0, 0, 0, 1]), n.get("scale", [1, 1, 1])
            r = Matrix(Quaternion((q[3], q[0], q[1], q[2])).to_matrix())
            return r @ Matrix.Diagonal(s)
        checked = 0
        for i, n in enumerate(nodes):
            j = n.get("extras", {}).get("s3d_joint")
            if not j or "axis" not in j:
                continue
            m, k = Matrix.Identity(3), i
            while k is not None:
                m, k = linear(nodes[k]) @ m, parent.get(k)
            got = (m @ Vector(j["axis"])).normalized()
            wants = []
            for o in bpy.data.objects:
                for k, jj in enumerate(joints_of(o)):
                    if jj["name"] == re.sub(r"_\d+$", "", j["name"]) and jj["axis"] is not None:
                        frame = (_core.world(o) @ Matrix(joints_pivot(o, k))).to_3x3()
                        w = (frame @ Vector(jj["axis"])).normalized()                   # Blender, Z up
                        wants.append(Vector((w.x, w.z, -w.y)))                          # glTF, Y up
            self.assertTrue(any((got - w).length < 1e-5 for w in wants), (j["name"], tuple(got), wants))
            checked += 1
        self.assertEqual(checked, 6)

    def test_the_glb_is_the_current_pose(self):
        s3d.export(self.root, self.path)
        rest = glb_json(self.path)
        s3d.pose(self.parts["drawer"], 200)
        s3d.export(self.root, self.path)
        posed = glb_json(self.path)
        node = lambda g, name: next(n for n in g["nodes"] if n["name"] == name)  # noqa: E731
        self.assertNotEqual(node(rest, "drawer_slide").get("translation"), node(posed, "drawer_slide").get(
            "translation"))
        self.assertEqual(node(posed, "drawer_slide")["extras"]["s3d_joint"]["value"], 0.2)

    def test_an_articulated_glb_keeps_its_materials(self):
        """The joints' extras were made by taking the materials' tags off, and the GLB then asked
        whether the scene was dressed: no, so it went out grey. The URDF's colours and the USD's
        looks, from the same scene, keep them too."""
        s3d.export(self.root, self.path)
        gltf = glb_json(self.path)
        nodes = {n["name"]: n for n in gltf["nodes"]}
        looks = gltf.get("materials", [])
        self.assertTrue(looks, "the GLB has no material")
        for part in ("door", "drawer"):
            prims = gltf["meshes"][nodes[part]["mesh"]]["primitives"]
            self.assertTrue(all("material" in p for p in prims), part)
        drawer = looks[gltf["meshes"][nodes["drawer"]["mesh"]]["primitives"][0]["material"]]
        rgba = drawer["pbrMetallicRoughness"]["baseColorFactor"]
        self.assertAlmostEqual(rgba[3], 0.5, places=3)                      # plastic, #336699 at 0.5
        self.assertLess(rgba[0], rgba[2])
        urdf = self.path.with_suffix(".urdf")
        s3d.export(self.root, urdf)
        self.assertIn("<material", urdf.read_text())
        usda = self.path.with_suffix(".usda")
        s3d.export(self.root, usda)
        self.assertIn("UsdPreviewSurface", usda.read_text())
        for m in bpy.data.materials:                     # the scene keeps its tags
            self.assertIn("s3d_material", m)

    def test_a_scene_without_joints_exports_as_before(self):
        s3d.empty_scene()
        root = s3d.group("thing", s3d.box("b", 10, 10, 10))
        s3d.export(root, self.path)
        nodes = glb_json(self.path)["nodes"]
        self.assertEqual(sorted(n["name"] for n in nodes), ["b", "thing"])
        self.assertTrue(all("s3d_joint" not in n.get("extras", {}) for n in nodes))


if __name__ == "__main__":
    unittest.main()
