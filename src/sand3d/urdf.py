"""An articulated object written out: as a URDF at rest, or as a GLB nested by its joints.

`export(root, "x.urdf")` writes a link per body — the root's, and one per jointed part
— a joint per joint (`fixed` never: what has no joint welds into its body), the meshes
beside it in `<stem>_meshes/`, one OBJ per part, in its link's frame, in metres,
triangulated as Blender triangulates them. A child
with several joints gets a link between each two, with no geometry. A `free` joint is
URDF's `floating`. The root link is free: placement belongs to the consuming scene.
Limits only: dynamics are out of sand3d (`effort` and `velocity` are written 0, as URDF
wants them written). Names are made unique by a suffix, in a fixed order.

`export(root, "x.glb")` of an object with joints nests its nodes kinematically: a node
per joint, at its pivot, under the part it hangs on, its child under it; each joint
node carries the joint in its extras — URDF's words, metres and radians, and its value
now. The GLB is the current pose; the URDF the rest, whatever the pose.

What is exported is the root's tree, and every jointed part that hangs, through its
joints, on a part of it. `usd.py` writes the same tree, `Mechanism.rest`, as OpenUSD.
"""
import math
import os
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Vector

from . import _core, joints as J, materials

#: URDF's word for each kind.
TYPES = {"revolute": "revolute", "continuous": "continuous", "prismatic": "prismatic", "free": "floating"}


def _order(obs):
    return sorted(obs, key=lambda o: (_core.given_name(o), o.name))


class Mechanism:
    """The objects `export(root, …)` writes, its bodies, and the names they go by."""

    def __init__(self, root, clean=None):
        self.root = root
        members, stack = [], [root]
        seen = set()
        while stack:                                     # the root's tree, and what hangs on it
            ob = stack.pop()
            if ob.as_pointer() in seen:
                continue
            seen.add(ob.as_pointer())
            members.append(ob)
            stack.extend(ob.children)
            if not stack:
                stack.extend(o for o in J.everything() if o.as_pointer() not in seen
                             and J.parent_of(o) is not None and J.parent_of(o).as_pointer() in seen)
        self.members = members
        self.ids = seen
        self.k = J.Kinematics()
        self.bodies = [ob for ob in members if ob is not root and J.jointed(ob)]
        self.order = self._kinematic_order()
        self.names = _Names(clean)
        self.link = {None: self.names.link(_core.given_name(root))}
        self.joint_names = {}
        for c in self.order:
            n = len(J.joints_of(c))
            for k, j in enumerate(J.joints_of(c)):
                self.joint_names[(c.as_pointer(), k)] = self.names.joint(j["name"])
                if k < n - 1:
                    self.link[(c.as_pointer(), k)] = self.names.link(f"{_core.given_name(c)}_{j['name']}")
            self.link[c.as_pointer()] = self.names.link(_core.given_name(c))

    def body(self, ob):
        """The body `ob` moves with: a jointed part of the mechanism, or None (the root's)."""
        while ob is not None and ob is not self.root:
            if J.jointed(ob) and ob.as_pointer() in self.ids:
                return ob
            ob = ob.parent
        return None

    def parent_body(self, c):
        return self.body(J.parent_of(c))

    def _kinematic_order(self):
        """Bodies parents first, siblings by name."""
        out, done, left = [], set(), _order(self.bodies)
        while left:
            ready = [c for c in left if self.parent_body(c) is None or self.parent_body(c).as_pointer() in done]
            if not ready:
                raise ValueError(f"export: the joints of {_core.given_name(left[0])!r} make a cycle")
            out += ready
            done |= {c.as_pointer() for c in ready}
            left = [c for c in left if c.as_pointer() not in done]
        return out

    def rest(self):
        """The mechanism at rest: its links — (key, body, frame), the root's first, a link
        between two joints of one child with no body — each frame a rotation in the world;
        and its joints, parents first: name, URDF's type, kind, the parent and child links'
        keys, the axis in the child link's frame (None: free), the range in the joint's own
        unit (None: no bound), and what it follows — (joint name, ratio, that joint's
        kind) — or None."""
        k = self.k
        frame = {None: _proper(_d(k.rest_world(self.root)))}
        links, joints = [(None, None)], []
        for c in self.order:
            js = J.joints_of(c)
            parent = self.parent_body(c)
            above = parent.as_pointer() if parent is not None else None
            for idx, (F, j) in enumerate(zip(k.frames(c, rest=True), js)):
                F = _proper(_d(F))
                key = (c.as_pointer(), idx) if idx < len(js) - 1 else c.as_pointer()
                frame[key] = F
                if idx < len(js) - 1:
                    links.append((key, None))
                axis = None
                if j["axis"] is not None:
                    _, u = J.axis_in_world(c, idx, frame=k.rest_world(c))
                    axis = [sum(F[r][i] * u[r] for r in range(3)) for i in range(3)]     # Fᵀ u
                follows = None
                if j.get("follows") is not None:
                    leader = J.leader_of(c, idx)
                    follows = (self.joint_names[(leader.as_pointer(), j["follows"])], j["ratio"],
                               J.joints_of(leader)[j["follows"]]["kind"])
                joints.append({"name": self.joint_names[(c.as_pointer(), idx)], "type": _effective_type(c, idx, j),
                               "kind": j["kind"], "parent": above, "child": key, "axis": axis,
                               "span": _span(c, idx, j), "follows": follows})
                above = key
            links.append((c.as_pointer(), c))
        return [(key, body, frame[key]) for key, body in links], joints

    def meshes(self, b):
        """The mesh objects of body `b` (None: the root's), in a fixed order."""
        return _order(ob for ob in self.members if ob.type == "MESH" and self.body(ob) == b)

    def check(self):
        top = _core.given_name(self.root)
        for c in _order(self.bodies):
            p = J.parent_of(c)
            if p is None:
                raise ValueError(f"export: {_core.given_name(c)!r} hangs on a part that was deleted")
            if p.as_pointer() not in self.ids:
                raise ValueError(f"export: {_core.given_name(c)!r} hangs on {_core.given_name(p)!r}, which is not in "
                                 f"{top!r}: export a group that holds both")
            for k, j in enumerate(J.joints_of(c)):
                lead = J.leader_of(c, k) if j.get("follows") is not None else None
                if j.get("follows") is not None and (lead is None or lead.as_pointer() not in self.ids):
                    what = "a part that was deleted" if lead is None else f"{_core.given_name(lead)!r}, not in {top!r}"
                    raise ValueError(f"export: {j['name']!r} on {_core.given_name(c)!r} follows a joint of {what}: "
                                     f"export a group that holds both")


class _Names:
    """Unique names, a suffix on the second and later: hinge, hinge_2, hinge_3 — each
    first made what the format takes by `clean`, if given."""

    def __init__(self, clean=None):
        self.used = {}
        self.clean = clean

    def unique(self, kind, name):
        if self.clean is not None:
            name = self.clean(name)
        used = self.used.setdefault(kind, set())
        got, n = name, 1
        while got in used:
            n += 1
            got = f"{name}_{n}"
        used.add(got)
        return got

    def link(self, name):
        return self.unique("link", name)

    def joint(self, name):
        return self.unique("joint", name)

    def visual(self, name):
        return self.unique("visual", name)


# ---------------------------------------------------------------- numbers

def _metres():
    return _core.MM / 1000.0


# Frames are composed here in double precision: mathutils is single, and a pivot at
# 850 mm would come out at 0.850000024 m.

def _d(m):
    """A 4×4 as rows of Python floats."""
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def _mul(a, b):
    return [[sum(a[r][i] * b[i][c] for i in range(4)) for c in range(4)] for r in range(4)]


def _inv(m):
    """The inverse of a placement — a rotation (or a mirror image) and a translation."""
    r = [[m[c][rr] for c in range(3)] for rr in range(3)]          # the transpose
    t = [-sum(r[i][k] * m[k][3] for k in range(3)) for i in range(3)]
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _orthonormal(m):
    """`m` with its axes made exactly orthonormal again (Gram-Schmidt, in double), its
    translation kept: the frames come from single precision."""
    cols = [[m[r][c] for r in range(3)] for c in range(3)]

    def unit(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]
    x = unit(cols[0])
    y = [a - sum(p * q for p, q in zip(cols[1], x)) * b for a, b in zip(cols[1], x)]
    y = unit(y)
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    if sum(a * b for a, b in zip(z, cols[2])) < 0:
        z = [-v for v in z]
    for v in (x, y, z):
        for i in range(3):
            if abs(v[i]) < 1e-9:
                v[i] = 0.0
            elif abs(abs(v[i]) - 1) < 1e-9:
                v[i] = math.copysign(1.0, v[i])
    return [[x[r], y[r], z[r], m[r][3]] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _proper(F):
    """`F` (rows of floats), orthonormal, its Z turned over if it is a mirror image: URDF
    frames are rotations."""
    F = _orthonormal(F)
    if _det3(F) < 0:
        F = [[row[0], row[1], -row[2], row[3]] for row in F[:3]] + [F[3]]
    return F


def _clean(v):
    return 0.0 if abs(v) < 1e-12 else v


def _num(v):
    return f"{_clean(v):.9g}"


def _vec(vs):
    return " ".join(_num(v) for v in vs)


def _rpy(r):
    """Roll, pitch, yaw (URDF: about fixed X, then Y, then Z) of a rotation (rows)."""
    sy = -r[2][0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 1 - 1e-12:
        roll = math.atan2(r[2][1], r[2][2])
        yaw = math.atan2(r[1][0], r[0][0])
    else:                                              # gimbal lock: the roll goes to the yaw
        roll = 0.0
        yaw = math.atan2(-r[0][1], r[1][1])
    return roll, pitch, yaw


def _span(c, k, j):
    """(lower, upper) of the k-th joint of `c` in its own unit, a follower's from what it
    follows; None when it has no bound."""
    if j.get("follows") is None:
        return j["range"]
    leader = J.leader_of(c, k)
    lead = J.joints_of(leader)[j["follows"]]
    span = _span(leader, j["follows"], lead)
    if span is None:
        return None
    return sorted(v * j["ratio"] for v in span)


def _si(kind):
    """A joint's unit in SI, per unit of the program: metres or radians."""
    return _metres() if kind == "prismatic" else math.radians(1.0)


def _effective_type(c, k, j):
    """URDF's type: a revolute follower of an endless joint turns without end too."""
    if j["kind"] == "revolute" and _span(c, k, j) is None:
        return "continuous"
    return TYPES[j["kind"]]


# ---------------------------------------------------------------- URDF

def _colour(ob):
    tag = materials._nearest(ob)
    if tag is None:
        return None
    v = materials.look(tag)
    h = v["color"]
    rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return materials._name(tag), (*rgb, v["opacity"])


def _write_obj(path, ob, m, unit):
    """`ob`'s mesh placed by `m` (rows: its link's frame from its own), × `unit`, as an OBJ."""
    me = ob.data
    flip = _det3(m) < 0
    rows = [[m[r][c] * unit for c in range(4)] for r in range(3)]
    co = [0.0] * (3 * len(me.vertices))
    me.vertices.foreach_get("co", co)
    lines = [f"# {_core.given_name(ob)} — written by sand3d", f"o {_core.given_name(ob)}"]
    for i in range(0, len(co), 3):
        x, y, z = co[i], co[i + 1], co[i + 2]
        lines.append("v " + _vec(r[0] * x + r[1] * y + r[2] * z + r[3] for r in rows))
    me.calc_loop_triangles()                  # Blender's triangles: a concave face fanned by a reader is wrong
    for t in me.loop_triangles:
        vs = [i + 1 for i in t.vertices]
        if flip:
            vs.reverse()
        lines.append("f " + " ".join(map(str, vs)))
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_urdf(root, path):
    mech = Mechanism(root)
    mech.check()
    k = mech.k
    path = os.fspath(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = f"{stem}_meshes"
    meshdir = os.path.join(os.path.dirname(os.path.abspath(path)), folder)
    os.makedirs(meshdir, exist_ok=True)
    unit = _metres()

    links, joints = mech.rest()
    frame = {key: F for key, _, F in links}
    robot = ET.Element("robot", name=mech.link[None])
    for key, body, F in links:
        link = ET.SubElement(robot, "link", name=mech.link[key])
        if key is not None and body is None:
            continue                                       # between two joints of one child: no geometry
        for ob in mech.meshes(body):
            vis_name = mech.names.visual(_core.given_name(ob))
            rel = f"{folder}/{vis_name}.obj"
            _write_obj(os.path.join(meshdir, f"{vis_name}.obj"), ob, _mul(_inv(F), _d(k.rest_world(ob))), unit)
            vis = ET.SubElement(link, "visual", name=vis_name)
            ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=rel)
            colour = _colour(ob)
            if colour is not None:
                mat = ET.SubElement(vis, "material", name=colour[0])
                ET.SubElement(mat, "color", rgba=_vec(colour[1]))

    for j in joints:
        node = ET.SubElement(robot, "joint", name=j["name"], type=j["type"])
        o = _mul(_inv(frame[j["parent"]]), frame[j["child"]])
        ET.SubElement(node, "origin", xyz=_vec(o[r][3] * unit for r in range(3)), rpy=_vec(_rpy(o)))
        ET.SubElement(node, "parent", link=mech.link[j["parent"]])
        ET.SubElement(node, "child", link=mech.link[j["child"]])
        if j["axis"] is not None:
            ET.SubElement(node, "axis", xyz=_vec(j["axis"]))
        if j["type"] in ("revolute", "prismatic"):
            f = _si(j["kind"])
            lo, hi = j["span"]
            ET.SubElement(node, "limit", lower=_num(lo * f), upper=_num(hi * f), effort="0", velocity="0")
        if j["follows"] is not None:
            leader, ratio, kind = j["follows"]
            ET.SubElement(node, "mimic", joint=leader, multiplier=_num(ratio * _si(j["kind"]) / _si(kind)), offset="0")
    ET.indent(robot, "  ")
    with open(path, "w", newline="\n") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f"<!-- written by sand3d: {mech.link[None]} at rest, metres and radians -->\n")
        f.write(ET.tostring(robot, encoding="unicode") + "\n")
    return path


# ---------------------------------------------------------------- GLB, nested by the joints

def articulated(root):
    """Whether `export(root, …)` writes something with joints."""
    if not J.everything():
        return False
    return bool(Mechanism(root).bodies)


def glb_tree(root, coll, kinematics=None, nodes=None):
    """Temporary objects in `coll` that the glTF exporter writes for `root`: its parts at
    their current pose, nested by the joints, a node per joint carrying it in its extras.
    The program's objects are renamed meanwhile, so the nodes keep their names; returns
    what `undo` needs."""
    mech = Mechanism(root)
    mech.check()
    k = kinematics if kinematics is not None else mech.k
    renamed = []
    for i, ob in enumerate(mech.members):
        renamed.append((ob, ob.name))
    for i, (ob, _) in enumerate(renamed):
        ob.name = f"s3d~export~{i}"
    made = {}
    for ob, name in renamed:
        twin = bpy.data.objects.new(name, ob.data if ob.type == "MESH" else None)
        coll.objects.link(twin)
        made[ob.as_pointer()] = twin
        if ob.type == "MESH" and "clayMaterial" in ob:
            twin["clayMaterial"] = ob["clayMaterial"].to_dict()
        if nodes is not None:
            nodes[(ob.as_pointer(), None)] = twin
    world = {ob.as_pointer(): k.posed(ob) if kinematics is not None else _core.world(ob) for ob, _ in renamed}

    def hang(twin, parent_twin, parent_world, own_world):
        twin.parent = parent_twin
        twin.matrix_parent_inverse = Matrix.Identity(4)
        twin.matrix_basis = parent_world.inverted() @ own_world if parent_twin is not None else own_world

    top = made[root.as_pointer()]
    top.matrix_basis = world[root.as_pointer()]
    top.scale = (_metres(),) * 3
    if kinematics is not None:
        top.matrix_basis = Matrix.Scale(_metres(), 4) @ world[root.as_pointer()]
    jointed = {c.as_pointer() for c in mech.bodies}
    for ob, _ in renamed:
        if ob is root:
            continue
        twin = made[ob.as_pointer()]
        if ob.as_pointer() in jointed:
            parent = J.parent_of(ob)
            above_twin, above_world = made[parent.as_pointer()], world[parent.as_pointer()]
            frames = k.frames(ob)
            for idx, (F, j) in enumerate(zip(frames, J.joints_of(ob))):
                node = bpy.data.objects.new(mech.joint_names[(ob.as_pointer(), idx)], None)
                coll.objects.link(node)
                if nodes is not None:
                    nodes[(ob.as_pointer(), idx)] = node
                hang(node, above_twin, above_world, F)
                span = _span(ob, idx, j)
                f = _si(j["kind"])
                value = k.data[ob.as_pointer()][idx].get("value", 0.0)
                extras = {"name": node.name, "type": _effective_type(ob, idx, j), "value": value * f}
                if j["axis"] is not None:                # the exporter turns every node frame Y-up: so is the axis
                    x, y, z = j["axis"]
                    extras["axis"] = [_clean(x), _clean(z), _clean(-y)]
                if span is not None:
                    extras["lower"], extras["upper"] = span[0] * f, span[1] * f
                if j.get("follows") is not None:
                    leader = J.leader_of(ob, idx)
                    lead = J.joints_of(leader)[j["follows"]]
                    extras["mimic"] = {"joint": mech.joint_names[(leader.as_pointer(), j["follows"])],
                                       "multiplier": j["ratio"] * f / _si(lead["kind"])}
                node["s3d_joint"] = extras
                above_twin, above_world = node, F
            hang(twin, above_twin, above_world, world[ob.as_pointer()])
        elif ob.parent is not None and ob.parent.as_pointer() in made:
            hang(twin, made[ob.parent.as_pointer()], world[ob.parent.as_pointer()], world[ob.as_pointer()])
        else:
            hang(twin, top, world[root.as_pointer()], world[ob.as_pointer()])
    return renamed, list(coll.objects)


def undo(state, coll):
    renamed, temps = state
    for t in temps:
        bpy.data.objects.remove(t)
    bpy.data.collections.remove(coll)
    for ob, name in renamed:
        ob.name = name
