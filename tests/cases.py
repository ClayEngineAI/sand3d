"""One case per behaviour worth pinning; together they exercise every public verb.

Each case builds in an empty scene and may return a non-geometric value
(points, frames, a digest), which the harness hashes with the scene.
"""
import hashlib
import os
import tempfile
from pathlib import Path

import bpy

import sand3d as s3d
from harness import case


@case("keyframes", "clip", "sample")
def animated_parts():
    base = s3d.box("base", 40, 40, 10)
    leaf = s3d.at(s3d.box("leaf", 20, 4, 20), z=10)
    s3d.joint("hinge", leaf, s3d.local(base), axis="+Z", range=(0, 90))
    root = s3d.assembly("animated", base, leaf)
    motion = s3d.clip("open", s3d.keyframes(leaf, {0: 0, 1: 90}, ease="in_out_cubic"),
                      s3d.keyframes(root, z={0: 0, 1: 20}, ease=lambda t: t * t))
    s3d.sample(motion, 0.75)
    s3d.sample(motion, 0.5)
    return motion.duration


@case("rect", "circle", "ellipse", "slot", "polygon", "superellipse")
def profile_shapes():
    return [p.mesh() for p in (s3d.rect(40, 20, r=4), s3d.rect(40, 20, chamfer=3), s3d.circle(20),
                               s3d.ellipse(40, 20), s3d.slot(50, 10), s3d.polygon(6, 20),
                               s3d.superellipse(40, 20, 4, taper=0.2))]


@case("station", "loft")
def stations():
    """A loft written as rows of stations, as a program writes it row by row."""
    return s3d.loft("leg", *[s3d.station(z=z, w=a, d=b, x=x, y=y, n=n)
                             for z, a, b, x, y, n in [(0, 34, 38, 10, 34, 2), (16, 48, 55, 11, 35, 2.5),
                                                      (60, 40, 44, 14, 38, 4)]])


@case("poly", "arc", "tangent_arc", "spline")
def profile_outlines():
    shapes = [s3d.poly([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)], fillet=[0, 0, 0, 3, 0, 0]),
              s3d.poly([(0, 0), (20, 0), s3d.arc(through=(30, 10), to=(20, 20)), s3d.tangent_arc((0, 40))]),
              s3d.poly([(0, 0), (30, 0), s3d.spline([(30, 0), (35, 10), (30, 20)]), (0, 20)], chamfer=2),
              s3d.spline([(0, 0), (20, 5), (30, 20), (10, 30), (-5, 15)], closed=True)]
    return [p.mesh() for p in shapes] + [s3d.arc((0, 0, 5), (20, 10), 30, 300).points,
                                         s3d.spline([(0, 0, 0), (10, 5, 3), (20, 0, 6)]).points]


@case("spline")
def spline_tangents():
    """An open spline given the directions it leaves and reaches its ends."""
    path = s3d.spline([(0, 0, 0), (10, 5, 3), (20, 0, 6)], tangents=((0, 1, 0), (1, 0, 1)))
    outline = s3d.poly([(0, 0), (30, 0), s3d.spline([(30, 0), (35, 10), (30, 20)], tangents=((1, 0), (-1, 0))), (0, 20)])
    return [path.points, outline.mesh()]


@case("offset", "mirror", "text")
def profile_operations():
    shapes = [s3d.offset(s3d.circle(30) - s3d.circle(10), 2), s3d.offset(s3d.rect(40, 20) + s3d.rect(10, 50), -1.5),
              s3d.mirror(s3d.poly([(0, 0), (20, 0), (20, 5), (5, 20), (0, 20)])), s3d.text("sand3d", 12)]
    return [p.mesh() for p in shapes]


@case("extrude")
def extrusions():
    s3d.extrude("plate", s3d.rect(80, 40, r=4) - s3d.circle(10), 6)
    s3d.extrude("taper", s3d.rect(40, 40), 20, taper=10)
    s3d.extrude("twist", s3d.rect(40, 20), 50, twist=90)
    s3d.extrude("both", s3d.polygon(6, 20), 10, both=True)
    s3d.extrude("h2", s3d.slot(40, 10), 10, h2=5, on=s3d.Frame((0, 0, 30), (1, 0, 0)))


@case("extrude")
def extrusions_up_to():
    s3d.extrude("a", s3d.rect(10, 10), until=s3d.Frame((0, 0, 20), (0, 0.5, 1)))
    pipe = s3d.at(s3d.cylinder("pipe", 40, 100), x=-50, z=50, ry=90)
    s3d.extrude("rib", s3d.rect(10, 20), until=pipe)


@case("gear", "knurl", "thread")
def parts():
    s3d.gear("g", 20, 2, 10, bore=8)
    s3d.thread("t", 10, 20, 1.5)
    s3d.knurl("k", 20, 10, 1)
    s3d.knurl("s", 20, 10, 1, straight=True)


@case("revolve")
def revolutions():
    s3d.revolve("cup", [(0, 0), (20, 0), (24, 40), (22, 40), (18, 3), (0, 3)])
    s3d.revolve("half", s3d.at(s3d.rect(10, 20), x=10), 180)


@case("sweep", "wire", "helix")
def sweeps():
    s3d.sweep("bar", s3d.rect(10, 4), [(0, 0, 0), (0, 0, 50), (50, 0, 50)])
    s3d.sweep("ring", s3d.rect(4, 4), s3d.circle(60))
    s3d.sweep("spring", s3d.circle(2), s3d.helix(20, 5, 30), frenet=True)
    s3d.sweep("horn", s3d.circle(10), s3d.arc((0, 0, 0), 40, 0, 90), scale=0.3, twist=30)
    s3d.wire("hoop", s3d.arc((0, 0, 0), 30, 0, 270), 4)


@case("loft", "superellipse")
def lofts():
    s3d.loft("vase", s3d.circle(40), s3d.at(s3d.circle(60), z=50), s3d.at(s3d.circle(30), z=100))
    s3d.loft("frustum", s3d.rect(40, 40), s3d.at(s3d.rect(20, 20), z=30))
    s3d.loft("band", *[s3d.at(s3d.superellipse(10, 4), x=x, y=y, ry=-90)
                       for x, y in s3d.superellipse(100, 60).loops[0].pts[::4]], follow=True, loop=True)


@case("box", "cylinder", "tube", "cone", "sphere", "wedge", "torus", "dome")
def primitives():
    s3d.box("box", 40, 30, 20)
    s3d.cylinder("cyl", 20, 30)
    s3d.tube("tube", 30, 20, 10)
    s3d.cone("cone", 30, 40)
    s3d.cone("frustum", 30, 40, 10)
    s3d.sphere("sphere", 30)
    s3d.wedge("wedge", 40, 20, 30, 10)
    s3d.torus("torus", 40, 6)
    s3d.dome("dome", 15, 3)


@case("cut", "join", "intersect")
def booleans():
    plate = s3d.box("plate", 60, 40, 8)
    s3d.cut(plate, s3d.at(s3d.cylinder("pin", 10, 20), x=10, z=-6), s3d.at(s3d.box("notch", 10, 50, 20), x=-30, z=-2))
    s3d.join(plate, s3d.at(s3d.cylinder("boss", 16, 10), x=-10, z=7))
    s3d.intersect(s3d.box("cube", 30, 30, 30), s3d.at(s3d.sphere("ball", 40), z=-5))


@case("cut", "join")
def pattern_booleans():
    """A tool and others repeated on frames: slots round a disc, ribs on a grid."""
    disc = s3d.cylinder("disc", 80, 6)
    s3d.cut(disc, s3d.at(s3d.at(s3d.box("slot", 12, 3, 10), x=25, z=-2), on=s3d.ring(12, 0)))
    plate = s3d.box("plate", 60, 40, 4)
    s3d.join(plate, s3d.at(s3d.at(s3d.box("rib", 2, 30, 6), z=4), on=s3d.grid(4, 1, 12, 0)))


@case("split", "delete", "merge", "scale")
def body_edits():
    s3d.split(s3d.sphere("ball", 40), s3d.Frame((0, 0, 25), (0, 1, 1)), keep="both")
    s3d.split(s3d.box("b", 20, 20, 20), s3d.Frame((0, 0, 0), (1, 0, 0)), keep="below")
    s3d.delete(s3d.box("gone", 5, 5, 5))
    s3d.merge(s3d.box("m", 10, 10, 10), s3d.at(s3d.box("n", 10, 10, 10), x=10), s3d.at(s3d.sphere("o", 8), z=21))
    s3d.scale(s3d.cylinder("s", 10, 10), (2, 1, -1))


@case("copy")
def copies():
    """A solid, a group and an assembly copied, the copies moved; the originals stay."""
    plate = s3d.at(s3d.extrude("plate", s3d.rect(40, 20, r=3), 5), x=10, rz=30)
    s3d.hole(plate, 4, on=(5, 0))
    s3d.at(s3d.copy(plate), y=40)
    pair = s3d.group("pair", s3d.at(s3d.box("a", 10, 10, 10), x=-10), s3d.at(s3d.sphere("b", 8), x=10))
    s3d.at(s3d.copy(pair), z=30, rz=45)
    table = s3d.assembly("table", s3d.at(s3d.box("top", 60, 40, 4), z=38), top=42, seat=s3d.Frame((0, 0, 42)))
    twin = s3d.copy(table)
    s3d.at(twin.root, x=100)
    return [twin.top, twin.seat.matrix]


@case("hole", "pocket")
def holes_and_pockets():
    plate = s3d.box("plate", 80, 50, 10)
    s3d.hole(plate, 6.6, on=[(-30, -15), (30, -15), (-30, 15), (30, 15)])
    s3d.hole(plate, 5, on=(0, 15), cbore=(9, 4))
    s3d.hole(plate, 5, on=(0, -15), csk=(10, 90))
    s3d.hole(plate, 4, on=s3d.Frame((40, 0, 5), (1, 0, 0)), depth=12)
    s3d.pocket(plate, s3d.rect(20, 10, r=2), depth=3)
    s3d.pocket(plate, s3d.slot(20, 6), on=s3d.Frame((0, -25, 5), (0, -1, 0)))


@case("fillet", "chamfer", "soften")
def edge_treatments():
    s3d.fillet(s3d.box("a", 40, 30, 20), 3, edges="top")
    plate = s3d.box("p", 60, 40, 8)
    s3d.hole(plate, 10, on=(0, 0))
    s3d.chamfer(plate, 1, edges="hole")
    s3d.chamfer(s3d.box("b", 40, 30, 20), 2, edges="top", d2=4)
    s3d.chamfer(s3d.cylinder("c", 30, 20), 2, edges="top", angle=30)
    s3d.soften(s3d.box("s", 40, 30, 20), 2)


@case("fillet", "chamfer")
def fillet_on_a_chamfer():
    """Bevel orders the faces it makes differently from run to run, on a round rim as here."""
    disc = s3d.extrude("disc", s3d.circle(28.5), 6)
    s3d.chamfer(disc, 9, edges="disc.top", d2=3)
    s3d.fillet(disc, 4, edges="chamfer")


@case("shell")
def shells():
    s3d.shell(s3d.box("o", 40, 30, 20), 2, open="top")
    s3d.shell(s3d.sphere("ball", 30), 1.5)
    s3d.shell(s3d.fillet(s3d.box("r", 40, 30, 20), 4, edges="bottom side"), 1.5, open="top")


@case("dent", "snap")
def dented_cushion():
    pad = s3d.loft("pad", s3d.superellipse(100, 100, 4), s3d.at(s3d.superellipse(90, 90, 4), z=30))
    spots = s3d.snap(pad, [(x, 0, 40) for x in (-25, 0, 25)])
    s3d.dent(pad, spots, 15, 5)
    return [f.matrix for f in spots]


@case("snap", "along")
def snap_along():
    body = s3d.box("body", 100, 50, 50)
    hits = s3d.snap(body, s3d.along([(-40, -100, 10), (40, -100, 10)], 20), direction=(0, 1, 0))
    ends = s3d.along(s3d.arc((0, 0, 0), 30, 0, 360), count=7)
    return [f.matrix for f in hits + ends]


@case("grid", "ring")
def distributors():
    plate = s3d.box("plate", 100, 60, 10)
    s3d.hole(plate, 5, on=s3d.grid(3, 2, 30, 30, on=plate.face("top")))
    s3d.hole(plate, 3, on=s3d.ring(6, 12, start=15, on=plate.face("top")))
    return [f.matrix for f in s3d.grid(4, 3, 10, 8, stagger=True) + s3d.ring(5, 20, angle=90)]


@case("at")
def placements():
    s3d.at(s3d.box("a", 10, 20, 30), x=5, y=-3, z=2, rx=10, ry=20, rz=30)
    s3d.transform(s3d.at(s3d.box("b", 10, 20, 30), z=4, anchor="bottom-front-left"), mirror="x")
    plate = s3d.box("plate", 60, 40, 8)
    s3d.at(s3d.cylinder("c", 10, 40), on=plate.face("+X").move(y=4))
    return [s3d.transform(s3d.at(s3d.rect(10, 5), x=3, rz=30), mirror="y").matrix,
            s3d.at(s3d.helix(10, 2, 6), z=5, rx=90).points]


@case("at", "pocket", "join", "cut", "merge", "group", "assembly", "attach")
def patterns():
    """`at(on=[frames])` on a solid, a group, a handle and a profile; lists of objects in
    every *objects parameter; `at` on a list; `pocket(on=[frames])`."""
    brick = s3d.box("brick", 64, 32, 10)
    s3d.join(brick, s3d.at(s3d.cylinder("stud", 9.6, 2), on=s3d.grid(4, 2, 16, 16), z=10))
    s3d.cut(brick, s3d.at(s3d.box("slot", 3, 40, 4), on=s3d.grid(3, 1, 20)))
    s3d.pocket(brick, s3d.circle(4), depth=2, on=s3d.grid(2, 1, 40, on=brick.face("-Z")))
    tabs = s3d.at(s3d.box("tab", 8, 4, 6), on=s3d.ring(4, 17))
    s3d.at(tabs, z=4)
    s3d.merge(tabs)
    pair = s3d.group("pair", s3d.box("a", 4, 4, 4), s3d.at(s3d.box("b", 4, 4, 4), x=6))
    s3d.group("pairs", s3d.at(pair, on=s3d.grid(1, 3, 0, 20), rz=15))
    lamp = s3d.assembly("lamp", s3d.cylinder("stem", 2, 20), top=20)
    lamps = s3d.at(lamp, on=s3d.ring(3, 50))
    s3d.attach(lamps[0], "shades", s3d.at(s3d.dome("shade", 8, 3), on=s3d.ring(2, 3, on=s3d.Frame((0, 0, 20)))))
    holes = s3d.at(s3d.circle(6), on=s3d.grid(3, 1, 20))
    return [h.matrix for h in holes]


@case("transform")
def transforms():
    """A door opened on its hinge, an arm and its image, a list, a handle, a profile, a path."""
    door = s3d.at(s3d.box("door", 30, 2, 60), x=15, y=-1)
    s3d.transform(door, x=1, rz=-30)
    arm = s3d.at(s3d.extrude("arm", s3d.poly([(10, 0), (40, 0), (40, 20), (30, 20), (10, 6)]), 6), x=6, z=2)
    s3d.transform(s3d.copy(arm), mirror="x")
    s3d.transform(s3d.at(s3d.box("tab", 8, 4, 6), on=s3d.ring(4, 17)), z=7, rx=10)
    lamp = s3d.assembly("lamp", s3d.cylinder("stem", 2, 20), top=20)
    s3d.transform(lamp, y=40, rz=15, mirror="y")
    return [s3d.transform(s3d.at(s3d.rect(10, 4), x=5), rz=90).matrix,
            s3d.transform(s3d.helix(10, 2, 6), z=5, mirror="xz").points]


@case("Frame")
def frame_of_faces():
    plate = s3d.extrude("plate", s3d.rect(60, 40), 8)
    s3d.join(plate, s3d.extrude("boss", s3d.circle(20), 10, on=plate.face("top").move(10, 5)))
    return [plate.face(w).matrix for w in ("top", "boss.top", "+Z", "-Y", "bottom", "plate.side")] + [
        s3d.Frame((10, 0, 5), (0, 1, 1)).offset(3).turn(rx=20, rz=45).move(2, -1).matrix]


@case("bbox", "at")
def frames_on_boxes():
    """Sides, edges and corners of a part's box and of a turned group's, and parts anchored
    by directions on them."""
    cab = s3d.box("cab", 600, 600, 850)
    shelf = s3d.at(s3d.box("shelf", 200, 40, 20), x=400, z=300, rz=30)
    frames = [s3d.bbox(cab, w) for w in ("+X", "+X-Y", "-Y+X", "+X-Y+Z", "-Z", "Y")]
    s3d.at(s3d.box("knob", 20, 20, 20), on=frames[1].move(y=50), anchor="-Z")
    s3d.at(s3d.box("foot", 40, 40, 10), on=s3d.bbox(cab, "-Z").turn(rx=180), anchor="+X-Y-Z")
    unit = s3d.transform(s3d.group("unit", cab, shelf), rz=20, x=5)
    return [f.matrix for f in frames] + [s3d.bbox(unit, "+X+Y").matrix, s3d.bbox(shelf, "-X").matrix]


@case("joint")
def joints_declared():
    """A washing machine's four joints, a mirrored twin of its door, a screw cap: two
    joints on one child, one following the other."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.at(s3d.cylinder("door", 420, 30), on=cab.face("+X").move(z=400))
    s3d.joint("door_hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
    drum = s3d.at(s3d.cylinder("drum", 400, 450), on=cab.face("+X").move(z=420).offset(-480))
    s3d.joint("drum_spin", drum, cab.face("+X").move(z=420), kind="continuous")
    drawer = s3d.at(s3d.box("drawer", 40, 150, 60), x=290, y=-180, z=780)
    s3d.joint("drawer_slide", drawer, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
    twin = s3d.transform(s3d.copy(door), mirror="y")
    cap = s3d.at(s3d.cylinder("cap", 40, 20), z=850)
    s3d.joint("cap_turn", cap, cab.face("top"), range=(0, 720))
    s3d.joint("cap_lift", cap, cab.face("top"), kind="prismatic", follows="cap_turn", ratio=3 / 360)
    return s3d.group("washer", cab, door, drum, drawer, twin, cap)


@case("local", "at", "joint")
def frames_in_own_space():
    """An arm read as a URDF gives it: each link placed on, and hung on, a frame at a
    point of its parent's own space, xyz and rpy as written; the lower link turned at rest."""
    base = s3d.cylinder("base", 160, 20)
    shoulder = s3d.local(base, z=20, ry=-30)
    lower = s3d.at(s3d.group("lower", s3d.at(s3d.box("beam", 300, 30, 30), x=150)), on=shoulder)
    s3d.joint("shoulder", lower, shoulder, axis="+Y", range=(-45, 90))
    elbow = s3d.local(lower, x=300, rx=90, rz=15)
    upper = s3d.at(s3d.group("upper", s3d.at(s3d.box("rod", 200, 20, 20), x=100)), on=elbow)
    s3d.joint("elbow", upper, elbow, axis="normal", range=(-90, 90))
    head = s3d.local(s3d.assembly("arm", base, lower, upper), x=-60, y=10.5, z=7.25, rz=-40)
    return [shoulder.matrix, elbow.matrix, head.matrix]


@case("pose")
def mechanism_posed():
    """A gate whose leaves are mirror images, a bolt sliding on one, a three-joint lamp
    and a screw cap, posed: forward kinematics down the chains, followers, mirrors."""
    w, h, post = 750, 1200, 100
    l_post = s3d.at(s3d.box("post", post, post, h + 300), x=-w - post / 2)
    r_post = s3d.transform(s3d.copy(l_post), mirror="x")
    board = s3d.at(s3d.box("board", w, 40, h), x=-w / 2, z=150)
    handle = s3d.at(s3d.cylinder("handle", 20, 60), on=s3d.bbox(board, "+Y").move(x=60 - w / 2))
    leaf = s3d.group("leaf", board, handle)
    s3d.joint("hinge", leaf, s3d.bbox(l_post, "+X"), axis="+Z", range=(0, 90))
    twin = s3d.transform(s3d.copy(leaf), mirror="x")
    bolt = s3d.at(s3d.cylinder("bolt", 16, 300), x=-60, y=-30, z=100)
    s3d.joint("drop_bolt", bolt, s3d.bbox(board, "-Y"), kind="prismatic", axis="-Z", range=(0, 170))
    gate = s3d.assembly("gate", l_post, r_post, leaf, twin, bolt)
    s3d.pose(leaf, 60)
    s3d.pose(twin, 90)
    s3d.pose(bolt, 120)
    base = s3d.at(s3d.cylinder("base", 100, 20), x=2000)
    lower = s3d.at(s3d.box("lower", 10, 10, 200), x=2000, z=20)
    upper = s3d.at(s3d.box("upper", 10, 10, 150), x=2000, z=220)
    s3d.joint("shoulder", lower, base.face("top"), axis="+Y", range=(-90, 90))
    s3d.joint("elbow", upper, lower.face("top"), axis="+Y", range=(-120, 120))
    lamp = s3d.assembly("lamp", base, lower, upper)
    s3d.pose(lamp, 35, joint="elbow")
    s3d.pose(lamp, -20, joint="shoulder")
    bottle = s3d.at(s3d.cylinder("bottle", 60, 200), x=-2000)
    cap = s3d.at(s3d.cylinder("cap", 40, 20), x=-2000, z=200)
    s3d.joint("cap_turn", cap, bottle.face("top"), range=(0, 720))
    s3d.joint("cap_lift", cap, bottle.face("top"), kind="prismatic", follows="cap_turn", ratio=3 / 360)
    s3d.pose(cap, 450, joint="cap_turn")
    return s3d.group("yard", gate.root, lamp.root, bottle, cap)


@case("seat")
def parts_seated():
    """A knob seated in a console, pins in a plate, and a drawer seated in its cabinet at
    rest while it is slid out."""
    console = s3d.box("console", 100, 100, 100)
    s3d.seat(console, s3d.at(s3d.cylinder("knob", 40, 30), z=80))
    plate = s3d.at(s3d.box("plate", 100, 100, 10), x=300)
    s3d.seat(plate, s3d.group("pins", *s3d.at(s3d.cylinder("pin", 10, 30), on=s3d.grid(2, 1, 50).move(300), z=-10)))
    cab = s3d.at(s3d.box("cab", 600, 600, 850), y=1000)
    drawer = s3d.at(s3d.box("drawer", 100, 150, 60), x=260, y=1000, z=700)
    s3d.joint("slide", drawer, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
    s3d.pose(drawer, 150)
    return s3d.seat(cab, drawer)


@case("group", "assembly", "attach", "Handle")
def assembly_layer():
    table = s3d.assembly("table", s3d.at(s3d.box("top", 100, 60, 4), z=68),
                         *[s3d.at(s3d.cylinder("leg", 5, 68), x=x, y=y)
                           for x in (-45, 45) for y in (-25, 25)], top=72)
    s3d.attach(table, "lamp", s3d.at(s3d.dome("shade", 20, 10), z=table.top))
    s3d.at(table.root, x=100)
    return s3d.group("room", table.root)


@case("export", "empty_scene")
def export_glb():
    s3d.empty_scene()
    root = s3d.group("thing", s3d.box("b", 10, 10, 10))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thing.glb"
        s3d.export(root, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()


@case("export")
def export_articulated():
    """A door, a drawer and a screw cap on a cabinet, posed: the URDF at rest and its
    meshes, and the GLB at the pose, nested by the joints."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.at(s3d.cylinder("door", 420, 30), on=cab.face("+X").move(z=400))
    s3d.joint("door_hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
    drawer = s3d.at(s3d.box("drawer", 40, 150, 60), x=290, y=-180, z=780)
    s3d.joint("drawer_slide", drawer, s3d.bbox(cab, "+X"), kind="prismatic", axis="+X", range=(0, 200))
    twin = s3d.transform(s3d.copy(door), mirror="y")
    cap = s3d.at(s3d.cylinder("cap", 40, 20), z=850)
    s3d.joint("cap_turn", cap, cab.face("top"), range=(0, 720))
    s3d.joint("cap_lift", cap, cab.face("top"), kind="prismatic", follows="cap_turn", ratio=3 / 360)
    root = s3d.group("washer", cab, door, drawer, twin, cap)
    s3d.pose(door, 70)
    s3d.pose(cap, 200, joint="cap_turn")
    with tempfile.TemporaryDirectory() as tmp:
        s3d.export(root, Path(tmp) / "washer.urdf")
        s3d.export(root, Path(tmp) / "washer.glb")
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(tmp).rglob("*.*"))]


@case("export")
def export_usd():
    """A glass door, its mirror image, a knob and a screw cap on a cabinet, posed: the USD
    layer at rest — bodies, meshes and their normals, joints, the mimic, the looks."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.material(s3d.at(s3d.cylinder("door", 420, 30), on=cab.face("+X").move(z=400)), "glass")
    s3d.joint("door_hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
    knob = s3d.material(s3d.at(s3d.cylinder("knob", 30, 15), on=door.face("top").move(x=100)), "brass")
    s3d.joint("knob_turn", knob, door.face("top").move(x=100), kind="continuous")
    twin = s3d.transform(s3d.copy(door), mirror="y")
    cap = s3d.at(s3d.cylinder("cap", 40, 20), z=850)
    s3d.joint("cap_turn", cap, cab.face("top"), range=(0, 720))
    s3d.joint("cap_lift", cap, cab.face("top"), kind="prismatic", follows="cap_turn", ratio=3 / 360)
    root = s3d.group("washer", cab, door, knob, twin, cap)
    s3d.pose(door, 70)
    with tempfile.TemporaryDirectory() as tmp:
        s3d.export(root, Path(tmp) / "washer.usda")
        return hashlib.sha256((Path(tmp) / "washer.usda").read_bytes()).hexdigest()


@case("export")
def export_meshes():
    """A cabinet, a door posed open and its mirror image: the OBJ and the STL at the pose."""
    cab = s3d.box("cab", 600, 600, 850)
    door = s3d.at(s3d.cylinder("door", 420, 30), on=cab.face("+X").move(z=400))
    s3d.joint("door_hinge", door, s3d.bbox(cab, "+X-Y"), axis="-Z", range=(0, 143))
    twin = s3d.transform(s3d.copy(door), mirror="y")
    root = s3d.group("washer", cab, door, twin)
    s3d.pose(door, 70)
    with tempfile.TemporaryDirectory() as tmp:
        s3d.export(root, Path(tmp) / "washer.obj")
        s3d.export(root, Path(tmp) / "washer.stl")
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(tmp).iterdir())]


@case("material")
def materials():
    """A table dressed from its assembly down, edited after: what every part wears.
    (Its GLB is `test_material`'s: a GLB's hash is no piece the unit test can compare.)"""
    top = s3d.box("top", 100, 60, 4)
    legs = s3d.at(s3d.cylinder("leg", 5, 68), on=s3d.grid(2, 2, 90, 50))
    table = s3d.assembly("table", s3d.at(top, z=68), legs)
    s3d.material(legs, "painted_metal")
    s3d.material(table, "oak")
    s3d.hole(top, 10, on=(30, 0))
    s3d.join(top, s3d.material(s3d.at(s3d.box("rail", 80, 4, 6), z=64), "brass"))
    s3d.attach(table, "lamp", s3d.material(s3d.at(s3d.dome("shade", 20, 10), z=72), "glass", color="#3fae6a"),
               s3d.material(s3d.at(s3d.sphere("bulb", 6), z=75), "light", color=(1, 0.85, 0.6), emission=20))
    s3d.material(s3d.copy(top), "plastic", color="#C0392B", roughness=0.2, metallic=0, opacity=0.5)
    return sorted((ob.get("s3d_name", ob.name), [m.name for m in ob.data.materials])
                  for ob in bpy.data.objects if ob.type == "MESH")


@case("units")
def metre_bracket():
    s3d.units("m")
    plate = s3d.extrude("plate", s3d.rect(0.08, 0.04, r=0.004), 0.006)
    s3d.hole(plate, 0.0066, on=s3d.grid(2, 1, 0.05, 0))
    s3d.fillet(plate, 0.001, edges="top")
    return s3d.units()


@case("fingerprint")
def library_fingerprint():
    return s3d.fingerprint()


@case("spline", "wire")
def spline_loop():
    """A smooth closed path in space: a rim of 3D points."""
    rim = s3d.spline([(0, 0, 40), (60, 0, 50), (60, 40, 60), (0, 40, 50)], loop=True)
    return [rim.points, s3d.wire("rim", rim, d=6)]

@case("shell")
def shell_open_by_direction():
    """A crate open toward +Y, a hull lofted along Y open on top."""
    crate = s3d.shell(s3d.box("crate", 60, 40, 30), 3, open="+Y")
    hull = s3d.loft("hull", s3d.at(s3d.rect(40, 20), rx=90), s3d.at(s3d.rect(40, 20), y=100, rx=90))
    return [crate, s3d.at(s3d.shell(hull, 2, open="+Z start"), x=80)]

@case("station", "loft")
def egg_stations():
    """A cabin lofted through egg sections."""
    return s3d.loft("cabin", *[s3d.station(z, w, d, y=y, n=2.5, taper=t)
                               for z, w, d, y, t in [(0, 300, 900, 50, 0.2), (40, 700, 1500, 0, 0.35), (90, 500, 1200, 20, 0)]])

@case("wire")
def bent_wire():
    """A stool's leg bent from a polyline, and a hoop with its own radius at one corner."""
    leg = s3d.wire("leg", [(0, 0, 0), (0, 0, 650), (-250, 0, 650)], 33, fillet=60)
    hoop = s3d.wire("hoop", [(0, 0, 0), (200, 0, 0), (200, 100, 0), (0, 100, 0)], 10, closed=True,
                    fillet=[20, 20, 40, 0])
    return [leg, s3d.at(hoop, x=100)]

@case("stroke", "extrude")
def strokes():
    """A mudguard round its wheel, a bent bracket, a hook with a tangent run."""
    fender = s3d.extrude("fender", s3d.stroke(s3d.arc((0, 0), 480, 180, 40), 18), 300, both=True)
    bracket = s3d.extrude("bracket", s3d.at(s3d.stroke([(0, 0), (50, 0), (50, 50)], 4), x=600), 20)
    hook = s3d.stroke([(0, 0), (40, 0), s3d.tangent_arc((60, 20)), (60, 60)], 6)
    return [fender, bracket, hook.mesh()]

@case("wrap", "offset", "extrude")
def wrapped_wheels():
    """A track's belt round its wheels, and a link round two bosses and a pin."""
    wheels = [s3d.at(s3d.circle(d), x=x, y=y) for x, y, d in [(0, 0, 160), (420, 0, 160), (210, 90, 60)]]
    outline = s3d.wrap(wheels)
    belt = s3d.extrude("belt", s3d.offset(outline, 12) - outline, 80)
    link = s3d.wrap(s3d.circle(20), s3d.at(s3d.circle(12), x=60), (30, 25))
    return [belt, link.mesh()]

@case("wire", "sweep", "along", "cut")
def edges_as_paths():
    """A bead along the loop a cut left, a moulding round a top, rivets along it."""
    cyl = s3d.cylinder("cyl", 40, 60)
    s3d.cut(cyl, s3d.at(s3d.box("notch", 20, 20, 20), x=20, z=20))
    plate = s3d.at(s3d.box("plate", 80, 40, 6), x=100)
    top = plate.edge("top")
    return [cyl.edge("notch").points, s3d.wire("bead", cyl.edge("notch"), 3),
            s3d.sweep("trim", s3d.rect(2, 3), top), [f.origin for f in s3d.along(top, pitch=20)]]

@case("body", "spline")
def bodies():
    """A fuselage from three views and a squareness along it, a hull of straight views."""
    fuselage = s3d.body("fuselage", s3d.spline([(-3500, 0), (-2500, 1000), (0, 1400), (3000, 900), (4500, 400)]),
                        s3d.spline([(-3500, 900), (-2500, 1400), (0, 1600), (3000, 1300), (4500, 1100)]),
                        s3d.spline([(-3500, 900), (-2500, 500), (0, 300), (3000, 600), (4500, 850)]),
                        n=[(-3500, 2), (0, 3), (4500, 2)])
    hull = s3d.body("hull", [(0, 0), (600, 900), (2400, 1000), (3000, 840)], 800, [(0, 800), (600, 120), (3000, 200)],
                    n=4)
    return [fuselage, s3d.at(hull, x=3000)]

if os.environ.get("SAND3D_HARNESS_SELFTEST"):
    @case()
    def selftest_unseeded():
        """Not a verb: a box sized by the OS's entropy, which the harness must catch."""
        s3d.box("noise", 10 + int.from_bytes(os.urandom(2), "little") / 1e4, 10, 10)


def environment():
    """What the run depended on — printed beside the digests, not hashed."""
    import bpy
    return {"bpy": bpy.app.version_string, "pid": os.getpid()}
