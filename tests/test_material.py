"""material: a named preset, its values changed where given, inherited down a group."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import bpy

import sand3d as s3d
from sand3d import materials


def worn(ob):
    """The names of the materials `ob`'s mesh wears."""
    return [m.name for m in ob.data.materials]


def bsdf(ob):
    return ob.data.materials[0].node_tree.nodes["Principled BSDF"].inputs


def gltf(path):
    """The JSON chunk of a GLB."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<I", raw[12:16])[0]
    return json.loads(raw[20:20 + n])


def export(root):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thing.glb"
        s3d.export(root, path)
        return gltf(path), path.stat().st_size


class Presets(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_part_wears_its_preset(self):
        plate = s3d.box("plate", 40, 20, 4)
        self.assertIs(s3d.material(plate, 'metal', color='#c9a24a', roughness=0.25), plate)
        self.assertEqual(worn(plate), ["metal #c9a24a roughness=0.25"])
        inputs = bsdf(plate)
        self.assertAlmostEqual(inputs["Metallic"].default_value, 1.0)
        self.assertAlmostEqual(inputs["Roughness"].default_value, 0.25)
        # '#c9a24a' in sRGB, written linear
        self.assertAlmostEqual(inputs["Base Color"].default_value[0], 0.584078 * 0.7681427, delta=0.003)

    def test_values_changed_where_given(self):
        a, b, c = (s3d.at(s3d.box(n, 10, 10, 10), x=20 * i) for i, n in enumerate("abc"))
        s3d.material(a, "plastic", color="#C0392B")
        s3d.material(b, "plastic", color=(192 / 255, 57 / 255, 43 / 255))
        s3d.material(c, "plastic", color="#c0392b", roughness=0.2, opacity=0.5, emission=3)
        self.assertEqual(worn(a), ["plastic #c0392b"])
        self.assertIs(a.data.materials[0], b.data.materials[0])      # one colour, one spelling, one material
        self.assertEqual(worn(c), ["plastic #c0392b emission=3 opacity=0.5 roughness=0.2"])
        inputs = bsdf(c)
        self.assertAlmostEqual(inputs["Alpha"].default_value, 0.5)
        self.assertAlmostEqual(inputs["Emission Strength"].default_value, 3.0)
        self.assertEqual(c.data.materials[0].surface_render_method, "BLENDED")

    def test_the_reference_lists_every_preset(self):
        """docs/api.md shows a docstring's first paragraph: the one place a model learns the names."""
        first = s3d.material.__doc__.split("\n\n")[0]
        listed = first.split(":", 1)[1].split(";")[0].split()
        self.assertEqual(listed, list(materials.PRESETS))

    def test_every_preset_builds(self):
        for i, kind in enumerate(materials.PRESETS):
            s3d.material(s3d.at(s3d.box(kind, 5, 5, 5), x=10 * i), kind)
        self.assertEqual(len(bpy.data.materials), len(materials.PRESETS))


class Inheritance(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def table(self, order):
        s3d.empty_scene()
        top = s3d.box("top", 100, 60, 4)
        legs = s3d.at(s3d.cylinder("leg", 5, 68), on=s3d.grid(2, 2, 90, 50))
        table = s3d.assembly("table", s3d.at(top, z=68), legs)
        calls = [lambda: s3d.material(table, 'wood', color='#b48a5a'), lambda: s3d.material(legs, 'metal', color='#a8abb0', roughness=0.3)]
        for i in order:
            calls[i]()
        return top, legs, table

    def test_the_nearest_call_wins_in_any_order(self):
        for order in ((0, 1), (1, 0)):
            top, legs, _ = self.table(order)
            self.assertEqual(worn(top), ["wood #b48a5a"])
            self.assertEqual([worn(leg) for leg in legs], [["metal #a8abb0 roughness=0.3"]] * 4)

    def test_a_detail_attached_later_inherits(self):
        top, legs, table = self.table((0,))
        lamp = s3d.attach(table, "lamp", s3d.at(s3d.dome("shade", 20, 10), z=72))
        self.assertEqual(worn(lamp.children[0]), ["wood #b48a5a"])

    def test_a_list_and_a_group(self):
        a, b = s3d.box("a", 5, 5, 5), s3d.at(s3d.box("b", 5, 5, 5), x=10)
        s3d.material([a, s3d.group("g", b)], 'plastic', color='#202020', roughness=0.9)
        self.assertEqual((worn(a), worn(b)), (["plastic #202020 roughness=0.9"], ["plastic #202020 roughness=0.9"]))


class ThroughEdits(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_edits_that_rebuild_the_mesh_keep_it(self):
        plate = s3d.material(s3d.box("plate", 40, 20, 4), 'wood', color='#b48a5a')
        s3d.hole(plate, 4, on=(5, 0))
        s3d.cut(plate, s3d.material(s3d.at(s3d.box("notch", 6, 30, 2), x=-15, z=3), 'metal', color='#a8abb0', roughness=0.3))
        s3d.fillet(plate, 0.5, edges="top")
        s3d.shell(plate, 1, open="top")
        self.assertEqual(worn(plate), ["wood #b48a5a"])
        self.assertEqual({p.material_index for p in plate.data.polygons}, {0})
        self.assertNotIn("metal #a8abb0 roughness=0.3", [m.name for m in bpy.data.materials if m.users])

    def test_join_keeps_the_first_objects(self):
        plate = s3d.material(s3d.box("plate", 40, 20, 4), 'wood', color='#d9b27c')
        s3d.join(plate, s3d.material(s3d.at(s3d.cylinder("peg", 6, 10), z=4), 'metal', color='#c9a24a', roughness=0.25))
        self.assertEqual(worn(plate), ["wood #d9b27c"])

    def test_join_says_which_materials_it_loses(self):
        """The review's red and blue boxes; nothing said for one colour, or for none."""
        import warnings
        red = s3d.material(s3d.box("red", 100, 100, 100), "plastic", color="#ff0000")
        blue = s3d.material(s3d.at(s3d.box("blue", 50, 50, 50), z=100), "plastic", color="#0000ff")
        with self.assertWarnsRegex(UserWarning, r"join \(line \d+\): 'red' keeps its 'plastic #ff0000', and the "
                                                r"parts joined into it lose theirs: 'plastic #0000ff' on 'blue' — .*"
                                                r"s3d\.group"):
            s3d.join(red, blue)
        self.assertEqual(worn(red), ["plastic #ff0000"])
        plain = s3d.box("plain", 10, 10, 10)
        with self.assertWarnsRegex(UserWarning, r"'plain' wears no material, and the parts joined into it lose "
                                                r"theirs: 'metal #c9a24a roughness=0.25' on 'peg' —"):
            s3d.join(plain, s3d.material(s3d.at(s3d.cylinder("peg", 6, 10), z=10), 'metal', color='#c9a24a', roughness=0.25))
        pegs = [s3d.material(s3d.at(s3d.cylinder("peg", 6, 10), x=x, z=10), 'metal', color='#c9a24a', roughness=0.25) for x in (-2, 2)]
        with self.assertWarnsRegex(UserWarning, r"'metal #c9a24a roughness=0.25' on 'peg' — "):     # a name said once
            s3d.join(s3d.box("rail", 10, 10, 10), pegs)
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            same = s3d.material(s3d.box("a", 10, 10, 10), "plastic", color="#ff0000")
            s3d.join(same, s3d.material(s3d.at(s3d.box("b", 10, 10, 10), z=10), "plastic", color="#FF0000"))
            s3d.join(s3d.box("c", 10, 10, 10), s3d.at(s3d.box("d", 10, 10, 10), z=10))
            painted = s3d.material(s3d.box("e", 10, 10, 10), 'wood', color='#b48a5a')
            s3d.join(painted, s3d.at(s3d.box("f", 10, 10, 10), z=10))     # it had none: it takes the wood
            kit = s3d.group("kit", s3d.box("g", 10, 10, 10), s3d.at(s3d.box("h", 10, 10, 10), z=10))
            s3d.material(kit, 'metal', color='#a8abb0', roughness=0.3)
            s3d.join(*kit.children)                                       # both wear the group's metal
        self.assertEqual([str(w.message) for w in said if str(w.message).startswith("join")], [])

    def test_copies_keep_it(self):
        peg = s3d.material(s3d.cylinder("peg", 6, 10), 'metal', color='#c9a24a', roughness=0.25)
        pegs = s3d.at(s3d.copy(peg), on=s3d.grid(3, 1, 10))
        self.assertEqual([worn(p) for p in pegs], [["metal #c9a24a roughness=0.25"]] * 3)


class Export(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_without_material_no_material_is_written(self):
        doc, _ = export(s3d.group("thing", s3d.box("b", 10, 10, 10)))
        self.assertNotIn("materials", doc)

    def test_flat_values_reach_the_glb(self):
        parts = [s3d.material(s3d.box("wood", 10, 10, 10), 'wood', color='#5e3b25'),
                 s3d.material(s3d.at(s3d.box("see", 10, 10, 10), x=20), "plastic", opacity=0.4),
                 s3d.material(s3d.at(s3d.box("pane", 10, 10, 10), x=40), "glass"),
                 s3d.material(s3d.at(s3d.box("bulb", 10, 10, 10), x=60), 'plastic', color='#fff1dc', emission=12)]
        doc, _ = export(s3d.group("thing", *parts))
        got = {m["name"]: m for m in doc["materials"]}
        self.assertEqual(set(got), {"wood #5e3b25", "plastic opacity=0.4", "glass", "plastic #fff1dc emission=12"})
        self.assertAlmostEqual(got["wood #5e3b25"]["pbrMetallicRoughness"]["roughnessFactor"], 0.6293181, places=5)
        self.assertEqual(got["plastic opacity=0.4"]["alphaMode"], "BLEND")
        self.assertAlmostEqual(got["plastic opacity=0.4"]["pbrMetallicRoughness"]["baseColorFactor"][3], 0.4, places=5)
        self.assertEqual(got["glass"]["extensions"]["KHR_materials_transmission"]["transmissionFactor"], 1)
        self.assertEqual(got["plastic #fff1dc emission=12"]["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"], 12)


class Mistakes(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()
        self.box = s3d.box("box", 10, 10, 10)

    def message(self, *args, **kwargs):
        with self.assertRaises((ValueError, TypeError)) as caught:
            s3d.material(*args, **kwargs)
        return str(caught.exception)

    def test_an_unknown_preset_says_the_nearest(self):
        self.assertTrue(self.message(self.box, "woood").startswith(
            "material: no preset named 'woood' — did you mean 'wood'?; the presets: brick, ceramic"))

    def test_values_out_of_range(self):
        self.assertEqual(self.message(self.box, "plastic", opacity=1.5),
                         "material: opacity=1.5 is not a number between 0 and 1 — 1 is opaque, 0 invisible")
        self.assertEqual(self.message(self.box, "plastic", emission=-1),
                         "material: emission=-1 is not a number of 0 or more")
        self.assertEqual(self.message(self.box, "plastic", color="red"),
                         "material: color='red' is not a colour — write '#rrggbb' or (r, g, b) in 0..1")
        self.assertEqual(self.message(self.box, "plastic", color=(200, 0, 0)),
                         "material: color=(200, 0, 0) is not a colour — write '#rrggbb' or (r, g, b) in 0..1")

    def test_a_profile_is_not_a_part(self):
        self.assertIn("is not a part, a group or an assembly", self.message(s3d.rect(10, 10), "wood"))

    def test_nothing_is_tagged_when_the_call_fails(self):
        self.message(self.box, "plastic", opacity=2)
        self.assertNotIn(materials.TAG, self.box)


if __name__ == "__main__":
    unittest.main()
