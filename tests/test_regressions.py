"""What real programs found: each a bug fixed in the library, pinned here."""
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

import bpy  # noqa: F401 — before bmesh, which bpy provides
import bmesh

import sand3d as s3d
from test_solids import closed_volume, edges


def manifold(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    ok = all(len(e.link_faces) == 2 for e in bm.edges)
    bm.free()
    return ok


GRAZES = (r"a tool grazes the part \(tangent to a face, or flush along a line\), or a solid crosses itself — move "
          r"the tool clear of the part or into it, and fuse overlapping solids with s3d\.join first")


def shells(ob):
    """How many pieces the mesh falls into, faces linked through their edges."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    seen, count = set(), 0
    for f in bm.faces:
        if f.index in seen:
            continue
        count += 1
        stack = [f]
        while stack:
            g = stack.pop()
            if g.index in seen:
                continue
            seen.add(g.index)
            stack += [h for e in g.edges for h in e.link_faces if h.index not in seen]
    bm.free()
    return count


def names_by_place(ob):
    """Every named edge and face, keyed by where it is: {("edge" | "face", rounded
    position): names}. Two meshes of the same shape compare whatever their order."""
    me = ob.data
    out = {}
    for prefix, kind, elems in (("s3d_edge:", "edge", me.edges), ("s3d_face:", "face", me.polygons)):
        for a in me.attributes:
            if not a.name.startswith(prefix):
                continue
            for el, d in zip(elems, a.data):
                if d.value:
                    where = tuple(sorted(tuple(round(c, 4) for c in me.vertices[i].co) for i in el.vertices))
                    out.setdefault((kind, where), set()).add(a.name[len(prefix):])
    return out


def turn_edges_round(ob, every=1):
    """Swap the two vertices of every `every`-th edge: the same mesh, its edges run the other way."""
    for i, e in enumerate(ob.data.edges):
        if i % every == 0:
            a, b = e.vertices
            e.vertices = (b, a)
    ob.data.update()


def rim(ob, z, tag):
    """Of the outer rim edges at height `z`, how many carry `tag`, and how many there are."""
    me = ob.data
    att = me.attributes.get("s3d_edge:" + tag)
    ring = [e.index for e in me.edges
            if all(abs(me.vertices[i].co.z - z) < 1e-4 and math.hypot(*me.vertices[i].co.xy) > 25 for i in e.vertices)]
    return (0 if att is None else sum(att.data[i].value for i in ring)), len(ring)


class FoundByPrograms(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_flange_on_a_side_face_merges_flat(self):
        """A quarter turn stored as Euler angles drifts a face by microns; the
        boolean saw two planes and left a membrane."""
        block = s3d.box("block", 100, 60, 30)
        s3d.join(block, s3d.extrude("flange", s3d.rect(15, 60), 5, on=block.face("+X").move(y=15)))
        faces_on_seam = [p for p in block.data.polygons if abs(p.center.x - 50) < 1e-4]
        self.assertAlmostEqual(sum(p.area for p in faces_on_seam), 2 * 675 + 2 * 225, places=3)

    def test_a_face_frame_keeps_its_axes_after_booleans(self):
        """A +Z face read as (0, 1e-8, 1) turned the frame's X a quarter turn."""
        hub = s3d.extrude("hub", s3d.circle(80), 60)
        s3d.cut(hub, s3d.cone("bore", 120, 19, 30))
        s3d.hole(hub, 6, on=s3d.ring(8, 28, on=hub.face("top")))
        top = hub.face("top")
        self.assertEqual(tuple(top.x), (1.0, 0.0, 0.0))

    def test_boolean_cracks_are_welded(self):
        """A hexagon corner lying on a wing's face came back as three vertices."""
        shape = s3d.poly([(0, 0), (0, 30), s3d.arc(through=(-12, 15), to=(-12, 0))])
        left = s3d.at(s3d.fillet(s3d.extrude("wing", shape, 8), 2, edges="side"), x=-3.547)
        right = s3d.transform(s3d.copy(left), x=2 * 3.547, mirror="y")
        core = s3d.extrude("core", s3d.polygon(6, 11.547), 8)
        s3d.join(core, left, right)
        self.assertTrue(manifold(core))
        positions = [tuple(round(c, 6) for c in v.co) for v in core.data.vertices]
        self.assertEqual(len(positions), len(set(positions)))

    def test_fillet_is_a_true_radius_on_any_bend(self):
        """The bevel's offset is a set-back, r·tan(φ/2) for a radius r."""
        disc = s3d.extrude("disc", s3d.circle(28.5), 6)
        s3d.chamfer(disc, 9, edges="disc.top", d2=3)
        s3d.fillet(disc, 4, edges="chamfer")
        top = [v.co for v in disc.data.vertices if abs(v.co.z - 6) < 1e-4]
        self.assertAlmostEqual(max(math.hypot(v.x, v.y) for v in top), 5.25 - 4 * math.tan(math.atan(3 / 9) / 2),
                               delta=0.02)

    def test_a_chamfer_wider_than_a_face(self):
        """9 mm across a top whose side is 6 mm high — cut equal first, then each rail."""
        disc = s3d.chamfer(s3d.extrude("disc", s3d.circle(28.5), 6), 9, edges="disc.top", d2=3)
        rim = sorted({(round(math.hypot(v.co.x, v.co.y), 1), round(v.co.z, 2)) for v in disc.data.vertices})
        self.assertIn((5.3, 6.0), rim)       # 5.25 off the true circle; the rim is cut on its facets
        self.assertIn((14.2, 3.0), rim)

    def test_plain_names_say_where_qualified_say_who(self):
        """A hole's mouth on the top is `top`, not `plate.top`."""
        plate = s3d.box("plate", 40, 40, 10)
        s3d.hole(plate, 10, on=(0, 0))
        self.assertEqual(edges(plate, "plate.top"), 4)
        self.assertGreater(edges(plate, "top"), 4)

    def test_loft_matches_sections_as_cad_does(self):
        """OpenCascade pairs each section with the last by the closest shift and
        direction; a rectangle turned 60° further is paired mirror-wise."""
        a = s3d.at(s3d.rect(36, 100), rz=120)
        b = s3d.at(s3d.rect(30, 100), z=80, rz=180)
        lofted = s3d.loft("l", a, b)
        self.assertTrue(manifold(lofted))
        bm = bmesh.new()
        bm.from_mesh(lofted.data)
        corner = min(bm.verts, key=lambda v: (v.co - __import__("mathutils").Vector((34.30, 40.59, 0))).length)
        ends = [e.other_vert(corner).co for e in corner.link_edges if abs(e.other_vert(corner).co.z - 80) < 1e-6]
        bm.free()
        self.assertEqual(len(ends), 1)
        self.assertAlmostEqual(ends[0].x, 15, places=3)    # the mirrored corner, 21.5 mm away …
        self.assertAlmostEqual(ends[0].y, 50, places=3)    # … not (-15, 50), 50.2 mm away

    def test_a_countersink_narrower_than_its_hole_is_a_plain_hole(self):
        """CskHole(6, 4, 45)."""
        a, b = s3d.box("a", 20, 20, 8), s3d.box("b", 20, 20, 8)
        s3d.hole(a, 6, on=(0, 0), csk=(4, 45))
        s3d.hole(b, 6, on=(0, 0))
        self.assertAlmostEqual(closed_volume(a)[1], closed_volume(b)[1], places=6)


class FoundByIssues(unittest.TestCase):

    def setUp(self):
        s3d.empty_scene()

    def test_a_process_that_imports_sand3d_exits(self):
        """Bpy 4.5 never exits while `Object.face` sits on the RNA type."""
        src = str(Path(s3d.__file__).resolve().parents[1])
        run = subprocess.run([sys.executable, "-c", "import sand3d"], env={**os.environ, "PYTHONPATH": src},
                             capture_output=True, timeout=120)
        self.assertEqual(run.returncode, 0)

    def test_a_tool_that_only_touches_cuts_nothing(self):
        """A groove cut away from the top it stands on printed its footprint on
        it, in slivers that clamped the top's chamfer to nothing. Since 1.0 such a cut
        raises, and leaves the block as it was."""
        block = s3d.box("block", 80, 60, 12)
        top = block.face("top")
        s3d.fillet(block, 3, edges="side")
        before = len(block.data.vertices)
        with self.assertRaisesRegex(ValueError, r"^cut: no tool cuts into 'block' — they miss it, or only touch it"):
            s3d.cut(block, s3d.extrude("groove", s3d.rect(4, 60), 9, on=top.move(20, 0)))
        self.assertEqual(len(block.data.vertices), before)
        s3d.chamfer(block, 0.8, edges="top")
        top = [v.co for v in block.data.vertices if abs(v.co.z - 12) < 1e-4]
        self.assertAlmostEqual(max(abs(v.y) for v in top), 30 - 0.8, places=4)

    def test_a_tool_that_only_touches_a_lower_step_cuts_nothing(self):
        """A tool standing on a step inside the target's bounding box touches it, no more."""
        block = s3d.box("block", 80, 60, 20)
        s3d.cut(block, s3d.at(s3d.box("step", 40, 70, 20), x=20, z=10))
        before = len(block.data.vertices)
        with self.assertRaisesRegex(ValueError, r"^cut: no tool cuts into 'block'"):
            s3d.cut(block, s3d.extrude("pocket", s3d.circle(8), 6, on=s3d.Frame((20, 0, 10))))
        self.assertEqual(len(block.data.vertices), before)

    def test_a_cut_drops_only_the_tools_that_touch(self):
        """In one cut, a tool that carves stays and is named; one that touches goes."""
        block = s3d.box("block", 80, 60, 12)
        top = block.face("top")
        s3d.cut(block, s3d.extrude("pocket", s3d.circle(8), -4, on=top.move(-20, 0)),
                s3d.extrude("boss", s3d.circle(8), 4, on=top.move(20, 0)))
        self.assertFalse(any(abs(v.co.x - 20) < 8.5 and abs(v.co.z - 6) < 1e-4 and abs(v.co.y) < 8.5
                             for v in block.data.vertices))
        s3d.chamfer(block, 0.5, edges="pocket")

    def test_a_frenet_sweep_ends_as_it_runs(self):
        """An open path's end has no curvature; its ring turned by float noise."""
        from sand3d.solids import _frames_along
        _, _, rs, _ = _frames_along(s3d.helix(20, 5, 30), frenet=True)
        step = rs[1].angle(rs[2])
        self.assertLess(rs[0].angle(rs[1]), step)
        self.assertLess(rs[-1].angle(rs[-2]), step)

    def test_a_boolean_through_a_warped_fillet_stays_closed(self):
        """A part at a scale of 1.0000001: the fillet bevelled a group of setbacks at a time
        and left warped faces the solver split its own way on each side of a seam;
        the holes left 31 open edges. A warped face can still crack."""
        s = 1.0000001
        hub = s3d.extrude("hub", s3d.circle(80 * s), 60 * s)
        s3d.cut(hub, s3d.cone("bore", 120 * s, 19 * s, 30 * s))
        s3d.fillet(hub, 2 * s, edges="bore bore.rim")
        s3d.hole(hub, 6 * s, on=s3d.ring(8, 28 * s, on=hub.face("top")))
        self.assertTrue(manifold(hub))

    def test_a_fillet_next_to_a_hole(self):
        """A hole split the top in two; the chamfer's edges bent 18.50° here and
        18.53° there, were bevelled a group at a time, and each group clamped the next."""
        disc = s3d.extrude("disc", s3d.circle(28.5), 6)
        s3d.hole(disc, 6, on=(0, 0))
        s3d.chamfer(disc, 9, edges="disc.top", d2=3)
        s3d.fillet(disc, 4, edges="chamfer")
        top = [v.co for v in disc.data.vertices if abs(v.co.z - 6) < 1e-4]
        self.assertAlmostEqual(max(math.hypot(v.x, v.y) for v in top), 5.25 - 4 * math.tan(math.atan(3 / 9) / 2),
                               delta=0.02)

    def test_a_chamfer_after_a_hole_names_its_rims_as_without(self):
        """Blender gives a bevel face the names of one neighbour, picked by which way
        the edge runs; after the hole's boolean the chamfer took `top` or `bottom`, and
        so did its lower rim, the edge it shares with the side."""
        for holed in (False, True):
            disc = s3d.cylinder("gear", 60, 10)
            if holed:
                s3d.hole(disc, 15, on=disc.face("top"))
            s3d.chamfer(disc, 0.5)
            with self.subTest(holed=holed):
                self.assertEqual(rim(disc, 9.5, "top"), (0, 55))
                self.assertEqual(rim(disc, 0.5, "bottom"), (0, 55))
                self.assertEqual(rim(disc, 9.5, "chamfer"), (55, 55))
                self.assertEqual(rim(disc, 10, "top"), (55, 55))
                self.assertEqual(rim(disc, 0, "bottom"), (55, 55))
            s3d.empty_scene()

    def test_bevel_names_do_not_follow_the_way_edges_run(self):
        """The same disc with its edges turned round, all or one in two, chamfered or
        filleted, names every edge and face as before."""
        for verb in (s3d.chamfer, s3d.fillet):
            seen = []
            for every in (None, 1, 2):
                s3d.empty_scene()
                disc = s3d.cylinder("gear", 60, 10)
                s3d.hole(disc, 15, on=disc.face("top"))
                if every:
                    turn_edges_round(disc, every)
                verb(disc, 0.5)
                seen.append(names_by_place(disc))
            with self.subTest(verb=verb.__name__):
                self.assertEqual(seen[1], seen[0])
                self.assertEqual(seen[2], seen[0])

    def test_a_bevel_face_takes_the_bevel_name_alone(self):
        """A chamfer or fillet face is part of neither face it cuts: it carries
        `chamfer` or `fillet`, never `top` nor `side`."""
        for verb, kind in ((s3d.chamfer, "chamfer"), (s3d.fillet, "fillet")):
            s3d.empty_scene()
            disc = s3d.cylinder("gear", 60, 10)
            s3d.hole(disc, 15, on=disc.face("top"))
            verb(disc, 0.5)
            names = [n for n in names_by_place(disc).items() if n[0][0] == "face" and kind in n[1]]
            with self.subTest(verb=kind):
                self.assertTrue(names)
                for _, tags in names:
                    self.assertEqual(tags, {kind, f"gear.{kind}"})

    def test_a_named_face_stays_flat_after_a_fillet(self):
        """Once the fillet's facets carried the names of the faces they cut,
        `face("side")` found them not flat and `shell(open="side")` opened the round."""
        b = s3d.box("b", 40, 40, 20)
        s3d.split(b, s3d.Frame((0, 0, 16), (0, 0.4, 1)), keep="below")
        s3d.fillet(b, 3)
        b.face("side")
        b.face("cut")
        side = b.data.attributes["s3d_face:side"]
        self.assertEqual(sum(d.value for d in side.data), 4)

    def test_posts_standing_flush_joined_in_one_call(self):
        """Six posts standing flush on a plate, joined in one call.
        Centred, as cylinders were before 1.0, and placed at z = T + H/2 in single
        precision, they sank 1e-7 mm into it; the union linked the holes they print in its
        top by zero-width bridges, welding them left loose edges, and the next hole kept
        only the drill. Far from the origin the noise is larger: the same part, moved
        300 mm along X and 2 m up."""
        T, H = 1.5875, 6.35
        centres = [(44.323, -15.113), (10.287, 2.413), (-9.398, 17.272), (-9.398, -17.272),
                   (-46.482, 17.272), (-46.482, -17.272)]
        for x0, z0 in ((0, 0), (300, 2000)):
            volumes = {}
            for how in ("one at a time", "one call", "merged first"):
                s3d.empty_scene()
                plate = s3d.at(s3d.extrude("plate", s3d.rect(114.046, 50.546), T), x=x0, z=z0)
                posts = [s3d.at(s3d.extrude("post", s3d.circle(5.588), H, both=True), x=x0 + x, y=y, z=z0 + T + H / 2)
                         for x, y in centres]
                if how == "one call":
                    s3d.join(plate, *posts)
                elif how == "merged first":
                    s3d.join(plate, s3d.merge(*posts))
                else:
                    for post in posts:
                        s3d.join(plate, post)
                with self.subTest("join", how=how, z=z0):
                    self.assertTrue(closed_volume(plate)[0])
                    self.assertEqual(shells(plate), 1)
                for x, y in centres:
                    s3d.hole(plate, 2.032, on=(x0 + x, y))
                closed, volumes[how] = closed_volume(plate)
                with self.subTest("holes", how=how, z=z0):
                    self.assertTrue(closed)
                    self.assertEqual(shells(plate), 1)
                    self.assertAlmostEqual(volumes[how], 9922.32, delta=10)

    def test_a_tool_is_not_settled_on_float_dust(self):
        """A tool's coordinate goes onto the target's only where the target has one
        value there. A revolve's quarter points lie at 3e-15 and -9e-15 where cos 90° is
        0: a split plane put on one of them cut nothing."""
        ring = s3d.revolve("ring", s3d.at(s3d.rect(5, 30), x=47.5, y=15))
        whole = closed_volume(ring)[1]
        s3d.split(ring, s3d.Frame((0, 0, 0), (1, 0, 0)), keep="above")
        closed, half = closed_volume(ring)
        self.assertTrue(closed)
        self.assertAlmostEqual(half, whole / 2, delta=whole * 1e-3)

    def test_a_face_frame_lies_on_the_face(self):
        """The frame was read off Blender's single-precision centre of the n-gon, and
        the top of a 12 mm disc came out at z = 12.0000095."""
        base = s3d.extrude("base", s3d.circle(80), 12)
        top = base.face("top")
        self.assertAlmostEqual(top.origin.z, 12, delta=1e-9)
        self.assertEqual(tuple(top.normal), (0.0, 0.0, 1.0))
        s3d.at(base, x=7.3, z=4.75, rz=30)
        self.assertAlmostEqual(base.face("top").origin.z, 16.75, delta=1e-6)
        self.assertAlmostEqual(base.face("-Z").origin.z, 4.75, delta=1e-6)
        s3d.transform(base, mirror="z")     # its image across z = 0: the top underneath, facing down
        top = base.face("top")
        self.assertEqual((tuple(round(c, 5) + 0.0 for c in top.origin), tuple(top.normal)),
                         ((0.0, 0.0, -16.75), (0.0, 0.0, -1.0)))

    def test_a_part_stood_on_a_face_with_anchor_fuses(self):
        """A pad stood on the top with `anchor="bottom"` floated
        9.5e-6 mm above it, and the join left two shells."""
        base = s3d.extrude("base", s3d.circle(80), 12)
        pad = s3d.at(s3d.extrude("pad", s3d.circle(70), 5), on=base.face("top"), anchor="bottom")
        self.assertEqual(min((pad.matrix_basis @ v.co).z for v in pad.data.vertices), 12.0)
        s3d.join(base, pad)
        self.assertTrue(closed_volume(base)[0])
        self.assertEqual(shells(base), 1)

    def _clamp(self, fuse):
        band = s3d.at(s3d.extrude("band", s3d.poly([s3d.arc(center=(0, 0), r=10, start=-11.5, end=348.5),
                                                     s3d.arc(center=(0, 0), r=11.2, start=348.5, end=-11.5)]), 10),
                      z=-10)
        ear = s3d.at(s3d.extrude("ear", s3d.rect(1.6, 20), 20), x=-10.8, y=-5, z=-10)
        return fuse(band, ear)

    def test_a_hole_in_a_merged_part_fuses_its_shells_first(self):
        """A hole in a band and an ear merged, not joined, came back
        with 205 open edges and the drill reaching 128 mm out."""
        volumes = []
        for fuse in (s3d.join, s3d.merge):
            s3d.empty_scene()
            clamp = self._clamp(fuse)
            s3d.hole(clamp, 3.5, on=s3d.Frame(origin=(-10.8, 0, 0), normal=(1, 0, 0)))
            closed, volume = closed_volume(clamp)
            with self.subTest(fuse=fuse.__name__):
                self.assertTrue(closed)
                self.assertEqual(shells(clamp), 1)
            volumes.append(volume)
        self.assertAlmostEqual(volumes[1], volumes[0], places=6)

    def test_a_merged_tool_repeated_on_frames_is_fused_too(self):
        """The copies `at=` makes of a merged tool keep its mark."""
        volumes = []
        for fuse in (s3d.join, s3d.merge):
            s3d.empty_scene()
            block = s3d.box("block", 40, 40, 6)
            s3d.cut(block, s3d.at(self._clamp(fuse), on=[s3d.Frame((0, 0, 4))]))
            volumes.append(closed_volume(block)[1])
        self.assertAlmostEqual(volumes[1], volumes[0], places=6)

    def test_a_boolean_that_comes_back_broken_raises(self):
        """Overlapping shells that no verb marked (a raw bpy join, say) break the
        boolean; it raises instead of handing back something else than the part."""
        clamp = self._clamp(s3d.merge)
        del clamp["s3d~merged"]
        before = len(clamp.data.vertices)
        with self.assertRaises(ValueError) as caught:
            s3d.hole(clamp, 3.5, on=s3d.Frame(origin=(-10.8, 0, 0), normal=(1, 0, 0)))
        self.assertRegex(str(caught.exception),
                         r"^hole: the result on 'band' reaches outside the target and its tools \([xyz] = [-+\d.e]+, "
                         r"beyond [-+\d.e]+ \.\. [-+\d.e]+\): " + GRAZES + "$")
        self.assertEqual(len(clamp.data.vertices), before, "the target is left as it was")
        self.assertEqual([o.get("s3d_name") for o in bpy.data.objects], ["band"], "the tool is used up")

    def test_a_counterbore_grazing_a_window_is_the_part(self):
        """A Ø10 counterbore whose rim is tangent to a window's wall. The
        solver drops a facet and the part comes back open, but it is the part: it stays in
        its box, and is returned."""
        plate = s3d.extrude("plate", s3d.rect(80, 50), 8)
        s3d.cut(plate, s3d.extrude("window", s3d.rect(48, 20), 8))
        s3d.hole(plate, 8, on=s3d.Frame((0, -15, 0)))
        s3d.hole(plate, 8, on=s3d.Frame((0, -15, 8)), cbore=(10, 4))
        co = [v.co for v in plate.data.vertices]
        self.assertEqual([(round(min(c[k] for c in co), 4), round(max(c[k] for c in co), 4)) for k in range(3)],
                         [(-40.0, 40.0), (-25.0, 25.0), (0.0, 8.0)])
        self.assertEqual(shells(plate), 1)
        volume = 80 * 50 * 8 - 48 * 20 * 8 - math.pi * 4 ** 2 * 8 - math.pi * (5 ** 2 - 4 ** 2) * 4
        self.assertAlmostEqual(closed_volume(plate)[1], volume, delta=volume * 0.003)   # facets, and the lost one

    def test_tools_lying_on_each_other_join_as_one(self):
        """Eight ribs turned by 45°, each symmetric, so the one at 180°
        lies exactly on the one at 0°. Two coincident solids in one union broke it (97 open
        edges, 5 shells), and the holes after it left a drill's cap above the part."""
        def rect(l, w, c):
            x, y = l / 2, w / 2
            return s3d.poly([(-x + c, -y), (x - c, -y), (x, -y + c), (x, y - c), (x - c, y), (-x + c, y), (-x, y - c),
                             (-x, -y + c)])
        for x0, y0 in ((0, 0), (80, 60)):
            volumes = []
            for angles in (range(0, 360, 45), range(0, 180, 45)):
                s3d.empty_scene()
                plate = s3d.at(s3d.extrude("plate", rect(100, 80, 0.5), 6), x=x0, y=y0)
                ribs = [s3d.at(s3d.extrude("rib", rect(30, 4, 0.5), 3), x=x0, y=y0, z=6, rz=a) for a in angles]
                s3d.join(plate, s3d.at(s3d.cylinder("boss", 24, 3), x=x0, y=y0, z=6), ribs)
                s3d.cut(plate, [s3d.at(s3d.cylinder("hole", 8, 10.2), x=x0 + x, y=y0, z=-0.6) for x in (-30, 0, 30)])
                closed, volume = closed_volume(plate)
                with self.subTest(ribs=len(angles), at=(x0, y0)):
                    self.assertTrue(closed)
                    self.assertEqual(shells(plate), 1)
                    self.assertEqual(len(bpy.data.objects), 1, "the ribs left out are used up all the same")
                volumes.append(volume)
            self.assertAlmostEqual(volumes[0], volumes[1], places=5)

    def test_a_tool_left_out_names_its_twins_edges(self):
        """Names are how edges are reached: a tool lying on another under another name is
        left out of the solver, and its name goes on the edges its twin makes."""
        from test_solids import edges
        plate = s3d.box("plate", 40, 40, 4)
        s3d.join(plate, s3d.at(s3d.box("a", 6, 6, 4), z=4), s3d.at(s3d.box("b", 6, 6, 4), z=4))
        self.assertTrue(closed_volume(plate)[0])
        self.assertEqual(edges(plate, "b"), edges(plate, "a"))
        self.assertGreater(edges(plate, "b"), 0)

    def test_a_hole_after_a_chamfer_of_a_tiny_angle(self):
        """Bpy's angle_limit, in radians, passed as the chamfer's
        face angle; the L bulged from z = -1 to 17.2 and the holes after it
        left spikes to ±300 mm. The chamfer refuses radians; at 1°, the smallest face angle
        it takes, the holes after the chamfer are clean."""
        W, H, T, B = 80.0, 60.0, 8.0, 30.0
        plate = s3d.extrude("Bracket", s3d.poly([(0, 0), (W, 0), (W, H - B), (W - B, H), (W - B, H - B), (0, H - B)]), T)
        s3d.join(plate, s3d.at(s3d.extrude("Reinforcement", s3d.poly([(W - B, H - B), (W - B + 20, H - B / 2),
                                                                     (W - B, H)]), T), z=T))
        s3d.join(plate, s3d.at(s3d.extrude("Rib", s3d.rect(10, 40), 4), x=W / 2, y=H / 2, z=T))
        with self.assertRaisesRegex(ValueError, r"^chamfer: angle is the face angle in degrees"):
            s3d.chamfer(plate, 2.0, angle=math.radians(1))
        with self.assertRaisesRegex(ValueError, r"^chamfer: unequal distances require"):
            s3d.chamfer(plate, 2.0, angle=1)
        s3d.chamfer(plate, 2.0, edges="top", angle=1)
        s3d.hole(plate, 8.0, on=(W / 2, H / 2), depth=6.0)
        for x, y in ((10, 10), (W - 10, 10)):
            s3d.hole(plate, 5.0, on=(x, y))
        for x, y in ((10, H - 10), (W - 10, H - 10)):            # off the L: they raise
            with self.assertRaisesRegex(ValueError, r"^hole: no tool cuts into 'Bracket'"):
                s3d.hole(plate, 5.0, on=(x, y))
        self.assertTrue(closed_volume(plate)[0])
        co = [v.co for v in plate.data.vertices]
        for k, size in enumerate((W, H, 2 * T)):             # within the L; the 1° chamfer trims 0.02 off a tip
            self.assertGreaterEqual(min(c[k] for c in co), -1e-6)
            self.assertLessEqual(max(c[k] for c in co), size + 1e-6)
            self.assertGreater(max(c[k] for c in co), size - 0.05)

    def test_dent_presses_a_flat_face(self):
        """A flat face meshed as one n-gon has no vertex near the point to press."""
        pad = s3d.extrude("pad", s3d.rect(60, 60, r=10), 12)
        s3d.dent(pad, [(x, y, 12) for x in (-12, 12) for y in (-12, 12)], 9, 3)
        self.assertTrue(manifold(pad))
        self.assertAlmostEqual(min(v.co.z for v in pad.data.vertices if v.co.z > 1), 9, places=4)

    def test_a_fillet_welds_in_a_fixed_order(self):
        """A torus on z = 0: bevel leaves the edges it makes in an
        order that changes with the run, and welding the zero-width strips it leaves, in
        that order, moved one vertex by an ulp one run in two — and with it the golden
        digest. The same part with its elements in another order fillets to the same mesh
        (an edge may run the other way: Blender's own order, which no digest saw move)."""
        import random

        def mesh(ob):
            me = ob.data
            return ([tuple(v.co) for v in me.vertices], [tuple(p.vertices) for p in me.polygons],
                    [tuple(sorted(e.vertices)) for e in me.edges],
                    {a.name: [d.value for d in a.data] for a in me.attributes if a.data_type == "BOOLEAN"})

        def elbow(seed):
            s3d.empty_scene()
            part = s3d.torus("elbow", 30, 10)
            s3d.cut(part, s3d.at(s3d.box("cut_x", 200, 200, 200), x=-100, z=-100),
                    s3d.at(s3d.box("cut_y", 200, 200, 200), y=-100, z=-100))
            s3d.chamfer(part, 1, edges="cut_x")
            inlet = part.face("-X")
            s3d.hole(part, 4, on=[inlet.move(-15, 5), inlet.move(15, 5)])
            if seed is not None:                  # as another run's bevel would hand them over
                bm = bmesh.new()
                bm.from_mesh(part.data)
                shuffle = random.Random(seed)
                for seq in (bm.verts, bm.edges, bm.faces):
                    keys = {x: shuffle.random() for x in seq}
                    seq.sort(key=keys.__getitem__)
                bm.to_mesh(part.data)
                bm.free()
            s3d.fillet(part, 0.5)
            return mesh(part)
        once = elbow(None)
        for seed in range(4):
            with self.subTest(seed=seed):
                self.assertEqual(elbow(seed), once)


    def glasses_rim(self):
        """A lens seated in a rim joined to a brow, the cut crossing the junction."""
        R, C = 83.0, (15.3, 76.8, 31.9)
        eye = [(-28, 40), (-18, 46), (-4, 46), (18, 44), (27, 38), (24, 22), (18, 10), (4, 1.5), (-16, 3),
               (-26, 10), (-28, 25)]
        on_blank = lambda x, z: (x, C[1] - math.sqrt((R - 1) ** 2 - (x - C[0]) ** 2 - (z - C[2]) ** 2), z)  # noqa: E731
        rim = s3d.wire("rim", s3d.spline([on_blank(x, z) for x, z in eye]), d=3.5, closed=True)
        brow = s3d.wire("brow", s3d.spline([(-32, 14, 40), (-18, 1, 46), (-4, -4, 46), (12, -5, 45), (29, -6, 38),
                                            (35, -7, 38)]), d=5)
        s3d.join(rim, brow)
        blank = s3d.at(s3d.shell(s3d.sphere("lens", 2 * R), 2), x=C[0], y=C[1], z=C[2] - R)
        lens = s3d.intersect(blank, s3d.extrude("edge", s3d.spline(eye, closed=True), R + 10,
                                                on=s3d.Frame((0, C[1], 0), normal=(0, -1, 0))))
        s3d.cut(rim, s3d.intersect(s3d.copy(rim), s3d.copy(lens)))
        return rim

    def test_a_boolean_leaves_no_duplicate_faces(self):
        """A cut across two joined wires left 36 faces twice, back to back."""
        rim = self.glasses_rim()
        keys = [tuple(sorted(p.vertices)) for p in rim.data.polygons]
        self.assertEqual(len(keys), len(set(keys)))

    def test_export_leaves_the_scene_as_it_found_it(self):
        """The glTF exporter validates the meshes it exports; under bpy 4.5 that wrote
        into the arrays a copy shared, and the copy turned into triangles across the rim."""
        import tempfile
        rim = self.glasses_rim()
        twin = s3d.transform(s3d.copy(rim), x=100)
        raw = s3d.box("raw", 10, 10, 10)          # a mesh validate() would change: one face twice
        me = raw.data
        faces = [tuple(p.vertices) for p in me.polygons]
        raw.data = bpy.data.meshes.new("raw")
        raw.data.from_pydata([tuple(v.co) for v in me.vertices], [], faces + [faces[0][::-1]])
        raw_twin = s3d.transform(s3d.copy(raw), x=50)
        root = s3d.group("pair", rim, twin, raw, raw_twin)

        def state():
            out = []
            for ob in (rim, twin, raw, raw_twin):
                bm = bmesh.new()
                bm.from_mesh(ob.data)
                out.append((len(bm.faces), round(sum(f.calc_area() for f in bm.faces), 6)))
                bm.free()
            return out
        before = state()
        s3d.export(root, os.path.join(tempfile.mkdtemp(), "pair.glb"))
        self.assertEqual(state(), before)

    @staticmethod
    def rolled_tub(n=4):
        """A bathtub: a station loft trimmed by a side profile and hollowed, and the loop the
        trim left."""
        body = s3d.loft("tub", s3d.station(0, 510, 1110, y=-20, n=n), s3d.station(602, 686, 1520, y=-20, n=n))
        s3d.chamfer(body, 50, edges="start")
        slipper = s3d.poly([(-1520, 440), (0, 440), (650, 586), (1520, 586), (1520, 1204), (-1520, 1204)],
                           fillet=[0, 1500, 1500, 0, 0, 0])
        s3d.cut(body, s3d.at(s3d.extrude("above_rim", slipper, h=686, both=True), rx=90, rz=90))
        s3d.cut(body, s3d.loft("hollow", s3d.station(97, 480, 870, y=-35, n=n), s3d.station(602, 606, 1440, y=-20, n=n)))
        return body, body.edge("above_rim")

    def test_a_union_the_exact_solver_gives_up_on_is_taken_again(self):
        """A round bead swept along the rim the trim left, joined into the tub, came back as
        an empty object — no face, no error. The exact solver gives up on the bead, which folds
        on itself by a hair round the loop's tight turns; the self-intersecting solver takes it."""
        body, rim = self.rolled_tub()
        top = max(v.co.z for v in body.data.vertices)
        tub = s3d.join(body, s3d.wire("bead", rim, d=70))
        self.assertGreater(len(tub.data.polygons), 1000)
        self.assertAlmostEqual(max(v.co.z for v in tub.data.vertices), top + 35, delta=1, msg="the bead rides the rim")

    def test_a_union_that_leaves_nothing_raises(self):
        """Two solids fused never leave nothing; if the solver cannot read them, join says so."""
        def nothing(name):
            ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
            bpy.context.scene.collection.objects.link(ob)
            ob["s3d_name"] = name
            return ob
        with self.assertRaisesRegex(ValueError, r"join: fusing 'sheet' into 'void' gives nothing"):
            s3d.join(nothing("void"), nothing("sheet"))


if __name__ == "__main__":
    unittest.main()
