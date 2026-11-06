import unittest

import bpy

import sand3d as s3d
from harness import digest
from test_solids import closed_volume


ERROR = "chamfer: unequal distances require every selected edge to adjoin a top, bottom, start, or end face"


class BoundedEdits(unittest.TestCase):
    def setUp(self):
        s3d.empty_scene()

    def test_countersinks_preserve_raised_bracket_material(self):
        for rounded in (False, True):
            for countersink in (False, True):
                with self.subTest(rounded=rounded, countersink=countersink):
                    s3d.empty_scene()
                    part = s3d.extrude("bracket", s3d.poly([(0, 0), (80, 0), (80, 8), (8, 8), (8, 60), (0, 60)]), 40)
                    if rounded:
                        s3d.fillet(part, 1)
                    for z in (12, 28):
                        s3d.hole(part, 6, on=s3d.Frame((55, 8, z), (0, 1, 0)), csk=12 if countersink else None)
                    self.assertEqual(max(v.co.y for v in part.data.vertices), 60)
                    self.assertTrue(closed_volume(part)[0])

    def test_countersink_boundaries_and_counterbore_precedence(self):
        for depth, plain, bounded, bored in ((None, 968.9417153228114, 964.8006106991864, 950.3067445164993),
                                            (5, 984.4708576614057, 980.3297530377807, 965.8358868550936)):
            for options, expected in (({}, plain), ({"csk": 1}, plain), ({"csk": 2}, plain),
                                      ({"csk": 4}, bounded), ({"csk": (4, 180)}, plain),
                                      ({"cbore": (4, 2)}, bored), ({"cbore": (4, 2), "csk": 6}, bored)):
                with self.subTest(depth=depth, options=options):
                    s3d.empty_scene()
                    part = s3d.hole(s3d.box("part", 10, 10, 10), 2, depth=depth, **options)
                    closed, volume = closed_volume(part)
                    self.assertTrue(closed)
                    self.assertAlmostEqual(volume, expected, delta=0.001)

    def test_unequal_side_chamfers_reject_before_mutation(self):
        for edges in ("side", "all", "top side", "part.side", ["top", "side"]):
            for options in ({"d2": 6}, {"d2": 1}, {"angle": 60}):
                with self.subTest(edges=edges, options=options):
                    s3d.empty_scene()
                    part = s3d.box("part", 20, 20, 20)
                    before = digest(part)
                    with self.assertRaises(ValueError) as error:
                        s3d.chamfer(part, 2, edges=edges, **options)
                    self.assertEqual(str(error.exception), ERROR)
                    self.assertEqual(digest(part), before)

    def test_supported_cap_and_equal_side_chamfers_remain_available(self):
        for edges in ("top", "bottom", "part.top"):
            s3d.empty_scene()
            self.assertAlmostEqual(closed_volume(s3d.chamfer(s3d.box("part", 20, 20, 20), 2, edges, d2=6))[1], 7552, delta=0.001)
        for edges in ("start", "end"):
            s3d.empty_scene()
            part = s3d.loft("part", s3d.rect(20, 20), s3d.at(s3d.rect(20, 20), z=20))
            self.assertTrue(closed_volume(s3d.chamfer(part, 2, edges, d2=6))[0])
        for distance, options, volume in ((2, {"d2": 2}, 7840), (2, {"angle": 45}, 7840),
                                           (0, {"d2": 6}, 8000), (2, {"d2": 0}, 8000)):
            s3d.empty_scene()
            part = s3d.chamfer(s3d.box("part", 20, 20, 20), distance, "side", **options)
            self.assertAlmostEqual(closed_volume(part)[1], volume, delta=0.001)
