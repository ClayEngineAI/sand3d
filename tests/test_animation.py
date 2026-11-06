"""Clip semantics and animation readback through an independent glTF evaluator."""
import bisect
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

import bpy
from mathutils import Matrix, Quaternion, Vector

import sand3d as s3d
from sand3d import _core, joints as J
from sand3d.animation import _EASES, _easing
from harness import digest


def read_asset(path):
    raw = Path(path).read_bytes()
    if path.suffix == ".gltf":
        doc = json.loads(raw)
        return doc, path.with_name(unquote(doc["buffers"][0]["uri"])).read_bytes()
    offset, doc, binary = 12, None, None
    while offset < len(raw):
        size, kind = struct.unpack_from("<I4s", raw, offset)
        data = raw[offset + 8:offset + 8 + size]
        if kind == b"JSON":
            doc = json.loads(data)
        if kind == b"BIN\0":
            binary = data
        offset += 8 + size
    return doc, binary


def accessor(doc, binary, index):
    a = doc["accessors"][index]
    view = doc["bufferViews"][a["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride", 4 * width)
    return [struct.unpack_from(f"<{width}f", binary, start + i * stride) for i in range(a["count"])]


def playback(doc, binary, name, time):
    animation = next(a for a in doc["animations"] if a["name"] == name)
    transforms = {i: {p: node.get(p, default) for p, default in
                     (("translation", (0, 0, 0)), ("rotation", (0, 0, 0, 1)), ("scale", (1, 1, 1)))}
                  for i, node in enumerate(doc["nodes"])}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        times = [row[0] for row in accessor(doc, binary, sampler["input"])]
        values = accessor(doc, binary, sampler["output"])
        right = bisect.bisect_right(times, time)
        path = channel["target"]["path"]
        if right == 0:
            value = values[0]
        elif right == len(times) or sampler.get("interpolation") == "STEP":
            value = values[right - 1]
        else:
            left = right - 1
            u = (time - times[left]) / (times[right] - times[left])
            if path == "rotation":
                a, b = values[left], values[right]
                q = Quaternion((a[3], *a[:3])).slerp(Quaternion((b[3], *b[:3])), u)
                value = (q.x, q.y, q.z, q.w)
            else:
                value = tuple(a + (b - a) * u for a, b in zip(values[left], values[right]))
        transforms[channel["target"]["node"]][path] = value
    parents = {child: i for i, node in enumerate(doc["nodes"]) for child in node.get("children", [])}
    world = {}

    def matrix(i):
        if i not in world:
            trs = transforms[i]
            q = trs["rotation"]
            local = Matrix.LocRotScale(Vector(trs["translation"]), Quaternion((q[3], *q[:3])), Vector(trs["scale"]))
            world[i] = matrix(parents[i]) @ local if i in parents else local
        return world[i]

    return {node["name"]: matrix(i) for i, node in enumerate(doc["nodes"])}


def model():
    base = s3d.box("base", 200, 120, 20)
    door = s3d.at(s3d.box("door", 80, 10, 100), x=50, z=20)
    drawer = s3d.at(s3d.box("drawer", 60, 80, 15), y=10, z=40)
    wheel = s3d.at(s3d.box("wheel", 60, 10, 10), x=-50, z=40)
    screw = s3d.at(s3d.cylinder("screw", 8, 30), x=-50, y=40, z=20)
    s3d.joint("hinge", door, s3d.local(base, 10, 0, 20), axis="+Z", range=(0, 120))
    s3d.joint("slide", drawer, s3d.local(base), kind="prismatic", axis="-Y", range=(0, 100))
    s3d.joint("spin", wheel, s3d.local(base, -50, 0, 40), kind="continuous", axis="+Z")
    s3d.joint("turn", screw, s3d.local(base, -50, 40, 20), range=(0, 1080), axis="+Z")
    s3d.joint("lift", screw, s3d.local(base), kind="prismatic", axis="+Z", follows="turn", ratio=4 / 360)
    asset = s3d.assembly("machine", base, door, drawer, wheel, screw)
    return asset, door, drawer, wheel, screw


class Animation(unittest.TestCase):
    def setUp(self):
        s3d.empty_scene()

    def test_all_easings_and_custom_function(self):
        ball = s3d.sphere("ball", 10)
        for name in _EASES:
            with self.subTest(ease=name):
                c = s3d.clip(name, s3d.keyframes(ball, z={0: 0, 1: 100}, ease=name))
                for t in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
                    s3d.sample(c, t)
                    self.assertTrue(math.isfinite(ball.location.z))
                self.assertEqual(_easing(name, 0), 0)
                self.assertEqual(_easing(name, 1), 1)
                s3d.sample(c, 0)
        c = s3d.clip("custom", s3d.keyframes(ball, z={0: 0, 1: 100}, ease=lambda t: t ** 3))
        s3d.sample(c, 0.5)
        self.assertAlmostEqual(ball.location.z, 12.5)
        self.assertAlmostEqual(_easing("in_quad", 0.5), 0.25)
        self.assertAlmostEqual(_easing("out_cubic", 0.5), 0.875)
        self.assertAlmostEqual(_easing("in_out_sine", 0.25), (1 - math.cos(math.pi / 4)) / 2)
        self.assertAlmostEqual(_easing("in_out_back", 0.25), -0.09968184375)
        self.assertAlmostEqual(_easing("out_bounce", 0.5), 0.765625)
        self.assertLess(_easing("in_elastic", 0.5), 0)
        self.assertGreater(_easing("out_back", 0.75), 1)

    def test_named_joints_followers_and_arbitrary_seeking(self):
        asset, door, drawer, wheel, screw = model()
        before = digest()
        c = s3d.clip("move", s3d.keyframes(asset, {0: 0, 1: 90}, joint="hinge"),
                     s3d.keyframes(drawer, {0: 0, 1: 100}),
                     s3d.keyframes(screw, {0: 0, 1: 720}, joint="turn"))
        self.assertEqual(digest(), before)
        self.assertEqual(c.duration, 1)
        s3d.sample(c, 0.5)
        half = digest()
        self.assertEqual(J.joints_of(door)[0]["value"], 45)
        self.assertEqual(J.joints_of(screw)[1]["value"], 4)
        for t in (1, 0.1, -1, 5, 0.5):
            s3d.sample(c, t)
        self.assertEqual(digest(), half)

    def test_offsets_keep_authored_placement_and_unkeyed_channels(self):
        ball = s3d.at(s3d.box("ball", 10, 20, 30), x=80, z=20)
        c = s3d.clip("bounce", s3d.keyframes(ball, z={0: 0, 1: 100}),
                     s3d.keyframes(ball, rz={0: 0, 1: 180}))
        s3d.sample(c, 0.5)
        self.assertAlmostEqual(ball.location.x, 80)
        self.assertAlmostEqual(ball.location.z, 70)
        self.assertAlmostEqual(ball.rotation_euler.z, math.pi / 2, places=6)
        s3d.sample(c, 0.5)
        self.assertAlmostEqual(ball.location.z, 70)

    def test_step_holds_and_sparse_tracks(self):
        ball = s3d.sphere("ball", 10)
        c = s3d.clip("steps", s3d.keyframes(ball, z={0.2: 10, 0.7: 20}, ease="step"))
        for t, expected in ((0, 10), (0.699, 10), (0.7, 20), (2, 20)):
            s3d.sample(c, t)
            self.assertEqual(ball.location.z, expected)

    def test_validation_and_atomic_failure(self):
        asset, door, drawer, wheel, screw = model()
        with self.assertRaisesRegex(ValueError, "outside its range"):
            s3d.keyframes(door, {0: 0, 1: 200})
        with self.assertRaisesRegex(ValueError, "leader"):
            s3d.keyframes(screw, {0: 0, 1: 3}, joint="lift")
        with self.assertRaisesRegex(ValueError, "same channel"):
            s3d.clip("bad", s3d.keyframes(door, {0: 0}), s3d.keyframes(door, {0: 10}))
        for keys in ({}, {-1: 0}, {float("nan"): 0}, {0: float("inf")}, {False: 0}):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                s3d.keyframes(door, keys)
        with self.assertRaisesRegex(ValueError, "unknown easing"):
            s3d.keyframes(door, {0: 0}, ease="typo")
        for ease in ("out_back", lambda t: float("nan")):
            c = s3d.clip("invalid", s3d.keyframes(drawer, {0: 0, 1: 50}),
                         s3d.keyframes(door, {0: 0, 1: 120}, ease=ease))
            before = digest()
            with self.assertRaises(ValueError):
                s3d.sample(c, 0.8)
            self.assertEqual(digest(), before)

    def test_deleted_targets(self):
        ball = s3d.sphere("ball", 10)
        c = s3d.clip("bounce", s3d.keyframes(ball, z={0: 0, 1: 10}))
        s3d.delete(ball)
        with self.assertRaisesRegex(ValueError, "deleted"):
            s3d.sample(c, 0.5)

    def test_followers_in_another_scene_tree(self):
        left = s3d.box("left", 20, 20, 10)
        right = s3d.at(s3d.box("right", 20, 20, 10), x=100)
        leader = s3d.at(s3d.box("leader", 10, 2, 10), z=10)
        follower = s3d.at(s3d.box("follower", 10, 2, 10), x=100, z=10)
        s3d.joint("leader_turn", leader, s3d.local(left), kind="continuous")
        s3d.joint("follower_turn", follower, s3d.local(right), kind="continuous", follows=leader, ratio=-2)
        s3d.assembly("left_tree", left, leader)
        s3d.assembly("right_tree", right, follower)
        c = s3d.clip("turn", s3d.keyframes(leader, {0: 0, 1: 90}))
        s3d.sample(c, 0.5)
        self.assertEqual(J.joints_of(follower)[0]["value"], -90)
        self.assertNotEqual(tuple(follower.rotation_euler), (0, 0, 0))

    def test_unanimated_joint_does_not_trigger_a_false_attachment_error(self):
        asset, door, drawer, _, _ = model()
        badge = s3d.at(s3d.box("badge", 3, 3, 3), on=s3d.local(door))
        badge.parent = asset.root
        c = s3d.clip("slide", s3d.keyframes(drawer, {0: 0, 1: 50}))
        s3d.sample(c, 0.5)
        self.assertEqual(J.joints_of(drawer)[0]["value"], 25)


class AnimationExport(unittest.TestCase):
    def setUp(self):
        s3d.empty_scene()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def assertPlayback(self, doc, binary, clip, parts, times, tolerance=2e-5):
        conversion = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
        unit = Matrix.Scale(_core.MM / 1000, 4)
        for time in times:
            read = playback(doc, binary, clip.name, time)
            s3d.sample(clip, time)
            for part in parts:
                expected = conversion @ unit @ _core.world(part) @ conversion.inverted()
                with self.subTest(time=time, part=part.name):
                    self.assertLess(max(abs(read[part.name][r][c] - expected[r][c])
                                        for r in range(4) for c in range(4)), tolerance)

    def test_gltf_and_glb_contain_playable_named_clips_and_preserve_scene(self):
        asset, door, drawer, wheel, screw = model()
        s3d.material(door, 'metal', color="#d95530", metallic=0)
        opening = s3d.clip("open", s3d.keyframes(door, {0: 0, 1: 90}, ease="in_out_sine"),
                           s3d.keyframes(drawer, {0: 0, 0.5: 0, 1: 80}),
                           s3d.keyframes(screw, {0: 0, 1: 720}, joint="turn"))
        spin = s3d.clip("spin", s3d.keyframes(wheel, {0: 0, 1: 1080}))
        s3d.sample(opening, 0.3)
        for extension in (".gltf", ".glb"):
            path = Path(self.temp.name) / ("animated" + extension)
            before = digest(), len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.actions)
            s3d.export(asset, path, animations=[opening, spin])
            self.assertEqual((digest(), len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.actions)), before)
            doc, binary = read_asset(path)
            self.assertEqual([a["name"] for a in doc["animations"]], ["open", "spin"])
            self.assertTrue(doc.get("materials"))
            self.assertTrue(any("s3d_joint" in n.get("extras", {}) for n in doc["nodes"]))
            for a in doc["animations"]:
                for sampler in a["samplers"]:
                    times = [v[0] for v in accessor(doc, binary, sampler["input"])]
                    self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
                    self.assertEqual(len(times), len(accessor(doc, binary, sampler["output"])))
            self.assertPlayback(doc, binary, opening, [door, drawer, screw], [0, 0.25, 0.5, 0.75, 1, 0.371])
            self.assertPlayback(doc, binary, spin, [door, drawer, wheel, screw], [0, 0.25, 0.5, 0.75, 1])

    def test_fast_multiple_revolutions_survive_low_fps(self):
        asset, _, _, wheel, _ = model()
        c = s3d.clip("fast", s3d.keyframes(wheel, {0: 0, 0.1: 1080}))
        path = Path(self.temp.name) / "fast.gltf"
        s3d.export(asset, path, animations=[c], fps=1)
        doc, binary = read_asset(path)
        a = doc["animations"][0]
        node = next(i for i, n in enumerate(doc["nodes"]) if n["name"] == "spin")
        channel = next(c for c in a["channels"] if c["target"] == {"node": node, "path": "rotation"})
        rotations = accessor(doc, binary, a["samplers"][channel["sampler"]]["output"])
        travel = sum(2 * math.acos(min(1, max(-1, sum(x * y for x, y in zip(a, b)))))
                     for a, b in zip(rotations, rotations[1:]))
        self.assertAlmostEqual(math.degrees(travel), 1080, places=2)
        self.assertPlayback(doc, binary, c, [wheel], [0.0125, 0.025, 0.0375, 0.05, 0.1])

    def test_transform_custom_easing_and_unit_conversion(self):
        for unit, length in (("mm", 100), ("cm", 10), ("m", 0.1)):
            s3d.empty_scene()
            s3d.units(unit)
            ball = s3d.at(s3d.box("ball", length, length / 2, length / 3), x=length)
            asset = s3d.assembly("asset", ball)
            c = s3d.clip("custom", s3d.keyframes(ball, z={0: 0, 1: length}, ease=lambda t: t ** 3),
                         s3d.keyframes(asset, rz={0: 0, 1: 360}))
            path = Path(self.temp.name) / (unit + ".glb")
            s3d.export(asset, path, animations=c)
            doc, binary = read_asset(path)
            self.assertPlayback(doc, binary, c, [ball, asset.root], [0, 0.25, 0.5, 1])

    def test_step_and_mixed_tracks(self):
        ball = s3d.sphere("ball", 10)
        c = s3d.clip("step", s3d.keyframes(ball, z={0: 0, 0.7: 100}, ease="step"),
                     s3d.keyframes(ball, rz={0: 0, 1: 90}))
        path = Path(self.temp.name) / "step.gltf"
        s3d.export(ball, path, animations=[c])
        doc, binary = read_asset(path)
        self.assertIn("STEP", [s["interpolation"] for s in doc["animations"][0]["samplers"]])
        self.assertPlayback(doc, binary, c, [ball], [0, 0.5, 0.699, 0.700001, 1])

    def test_failed_export_restores_scene_and_writes_no_asset(self):
        asset, door, _, _, _ = model()
        c = s3d.clip("overshoot", s3d.keyframes(door, {0: 0, 1: 120}, ease="out_back"))
        before = digest(), len(bpy.data.objects), len(bpy.data.meshes)
        path = Path(self.temp.name) / "bad.gltf"
        with self.assertRaisesRegex(ValueError, "outside its range"):
            s3d.export(asset, path, animations=[c])
        self.assertFalse(path.exists())
        self.assertEqual((digest(), len(bpy.data.objects), len(bpy.data.meshes)), before)
        with self.assertRaisesRegex(ValueError, "require .glb or .gltf"):
            s3d.export(asset, path.with_suffix(".urdf"), animations=[c])
        with self.assertRaisesRegex(ValueError, "positive"):
            s3d.export(asset, path, animations=[c], fps=0)

    def test_static_gltf_export(self):
        ball = s3d.sphere("ball", 10)
        path = Path(self.temp.name) / "still.gltf"
        s3d.export(ball, path)
        doc, binary = read_asset(path)
        self.assertTrue(binary)
        self.assertNotIn("animations", doc)

    def test_every_easing_is_baked_into_the_file(self):
        ball = s3d.box("ball", 10, 10, 10)
        clips = [s3d.clip(name, s3d.keyframes(ball, z={0: 0, 1: 100}, ease=name)) for name in _EASES]
        path = Path(self.temp.name) / "easing.gltf"
        s3d.export(ball, path, animations=clips)
        doc, binary = read_asset(path)
        self.assertEqual(len(doc["animations"]), len(_EASES))
        for c in clips:
            self.assertPlayback(doc, binary, c, [ball], [0, 0.25, 0.5, 0.75, 1])

    def test_nested_joints_and_mirrored_copy(self):
        asset, door, _, _, _ = model()
        twin = s3d.transform(s3d.copy(door), mirror="x")
        knob = s3d.at(s3d.box("knob", 12, 3, 3), on=s3d.local(door, 20, 0, 60))
        s3d.joint("knob_turn", knob, s3d.local(door, 20, 0, 60), kind="continuous", axis="+Y")
        whole = s3d.assembly("whole", asset.root, twin, knob)
        c = s3d.clip("nested", s3d.keyframes(door, {0: 0, 1: 90}),
                     s3d.keyframes(twin, {0: 0, 1: 90}), s3d.keyframes(knob, {0: 0, 1: 720}),
                     s3d.keyframes(whole, rz={0: 0, 1: 90}))
        path = Path(self.temp.name) / "nested.glb"
        s3d.export(whole, path, animations=[c])
        doc, binary = read_asset(path)
        self.assertPlayback(doc, binary, c, [door, twin, knob], [0, 0.25, 0.5, 0.75, 1])

