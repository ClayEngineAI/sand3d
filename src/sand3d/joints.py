"""Articulations: how a part moves on another.

A joint is written on its child, the part that moves — an object or a group — and
hangs it on the part its frame `on` was read on, the parent. It does not move the
child and it does not reparent it: the kinematic tree (child -> parent) is kept apart
from the scene tree, which still says what welds to what. A part with no joint moves
with its group; a jointed part moves by its joint, whatever group it is in. The bodies
of a mechanism are its jointed parts and, for everything else, the root.

What a child carries, as custom properties (so `copy` carries it too):

- `s3d~joints`: its joints in declaration order, JSON — name, kind, the pivot frame in
  the child's own space at rest (`pivot`, which so moves with the child: `transform`
  moves a jointed part with its pivot, and the image of a mirrored child moves as the
  image of the motion), the axis in that frame, the range, what it follows, the ratio,
  the value it is posed at;
- `s3d~parent`: the part its joints hang on;
- `s3d~follows.<k>`: the child whose joint the k-th follows;
- `s3d~rest`: its placement in its scene parent at rest.

Several joints on one child compose in declaration order, the first nearest the
parent: a cylindrical joint is a slide and a turn, a screw a slide and a turn that
follows it.
"""
import json
import math
import re

import bpy
from mathutils import Matrix, Vector

from . import _core
from .frames import Frame, _clean

#: The four kinds, and what each takes.
KINDS = ("revolute", "continuous", "prismatic", "free")
JOINTS, PARENT, REST, ON, ON_AT = "s3d~joints", "s3d~parent", "s3d~rest", "s3d~on", "s3d~on_at"
FOLLOWS = "s3d~follows."
_AXIS = re.compile(r"^([+\-<>]?)([xyzXYZ])$")


# ---------------------------------------------------------------- reading what a part carries

def joints_of(ob):
    """The joints `ob` carries, in declaration order (a list of dicts; [] for none)."""
    raw = ob.get(JOINTS) if isinstance(ob, bpy.types.Object) else None
    return json.loads(raw) if raw else []


def _write(ob, joints):
    ob[JOINTS] = json.dumps(joints, separators=(",", ":"))


def jointed(ob):
    return isinstance(ob, bpy.types.Object) and JOINTS in ob


def parent_of(ob):
    """The part `ob`'s joints hang on."""
    return ob.get(PARENT)


def leader_of(ob, k):
    """The child whose joint the k-th joint of `ob` follows, or None."""
    return ob.get(FOLLOWS + str(k))


def body(ob, extra=None):
    """The jointed part `ob` moves with — its nearest jointed ancestor-or-self — or None
    for the root body. `extra` counts as jointed too (a child being declared)."""
    while ob is not None:
        if jointed(ob) or (extra is not None and ob == extra):
            return ob
        ob = ob.parent
    return None


def _under(ob, top):
    """Whether `ob` is `top` or somewhere under it in the scene."""
    while ob is not None:
        if ob == top:
            return True
        ob = ob.parent
    return False


def _flat16(m):
    return [float(m[r][c]) for r in range(4) for c in range(4)]


def _matrix(flat):
    return Matrix([flat[4 * r:4 * r + 4] for r in range(4)])


def _components(ob):
    """`ob`'s placement as Blender holds it — the components and the matrix they make —
    so that the rest comes back to the bit: a matrix set is decomposed again."""
    return {"matrix": _flat16(ob.matrix_basis), "mode": ob.rotation_mode, "location": list(ob.location),
            "euler": list(ob.rotation_euler), "quaternion": list(ob.rotation_quaternion),
            "axis_angle": list(ob.rotation_axis_angle), "scale": list(ob.scale)}


def _rest(ob):
    return json.loads(ob[REST])


def rest_basis(ob):
    """`ob`'s placement in its scene parent at rest: what pose leaves alone at 0."""
    kept = ob.get(REST) if isinstance(ob, bpy.types.Object) else None
    return _matrix(json.loads(kept)["matrix"]) if kept is not None else Matrix(ob.matrix_basis)


def _restore(ob):
    """`ob` back at its rest placement, component by component."""
    kept = _rest(ob)
    ob.rotation_mode = kept["mode"]
    ob.location = kept["location"]
    ob.rotation_euler = kept["euler"]
    ob.rotation_quaternion = kept["quaternion"]
    ob.rotation_axis_angle = kept["axis_angle"]
    ob.scale = kept["scale"]


def everything():
    """Every jointed object of the scene."""
    return [ob for ob in bpy.context.scene.objects if JOINTS in ob]


def _name(ob):
    return _core.given_name(ob)


# ---------------------------------------------------------------- declaring a joint

def _axis(axis, verb):
    """A direction ("+Z", "-X", "Y") as a vector, or "normal"."""
    if axis == "normal":
        return "normal"
    d = _AXIS.match(axis) if isinstance(axis, str) else None
    if not d:
        raise ValueError(f"{verb}: axis {axis!r} is none of \"+X\" \"-X\" \"+Y\" \"-Y\" \"+Z\" \"-Z\" — a direction in "
                         f"the space the parent lives in — or \"normal\", the Z of the frame `on`")
    v = Vector({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[d.group(2).lower()])
    return -v if d.group(1) in ("-", "<") else v


def _range(range, kind, verb, follows):
    if follows is not None:
        if range is not None:
            raise ValueError(f"{verb}: a joint that follows another takes no range: its range is the other's × ratio")
        return None
    if kind in ("continuous", "free"):
        if range is not None:
            what = "turns without end" if kind == "continuous" else "moves freely"
            raise ValueError(f"{verb}: a {kind} joint {what}: it takes no range")
        return None
    if range is None:
        unit = "degrees" if kind == "revolute" else "the program's unit"
        hint = " — kind=\"continuous\" turns without end" if kind == "revolute" else ""
        raise ValueError(f"{verb}: a {kind} joint needs range=(low, high), in {unit}{hint}")
    try:
        lo, hi = (float(v) for v in range)
    except (TypeError, ValueError):
        raise TypeError(f"{verb}: range is (low, high), two numbers, not {range!r}") from None
    if not lo < hi:
        raise ValueError(f"{verb}: range ({lo:g}, {hi:g}) is empty: low comes before high")
    if not lo <= 0 <= hi:
        raise ValueError(f"{verb}: range ({lo:g}, {hi:g}) does not contain 0: everything is built at rest, every "
                         f"joint at 0 — build the part where it rests, and give the range from there")
    return [lo, hi]


def _resolve_follows(follows, child, verb):
    """(leader child, joint index) `follows` names: a jointed child (its only joint), or a
    joint name — on `child` first, then anywhere in the scene."""
    from .assembly import Handle
    if isinstance(follows, Handle):
        follows = follows.root
    if isinstance(follows, bpy.types.Object):
        if follows == child and not jointed(child):
            raise ValueError(f"{verb}: follows its own child, which has no joint yet: a joint cannot follow itself "
                             f"(a cycle)")
        found = joints_of(follows)
        if not found:
            raise ValueError(f"{verb}: follows {_name(follows)!r}, which has no joint")
        if len(found) > 1:
            names = ", ".join(repr(j["name"]) for j in found)
            raise ValueError(f"{verb}: follows {_name(follows)!r}, which has {len(found)} joints ({names}): name the "
                             f"one, follows=\"{found[0]['name']}\"")
        return follows, 0
    if not isinstance(follows, str):
        raise TypeError(f"{verb}: follows is a jointed part or a joint's name, not {type(follows).__name__}")
    own = [k for k, j in enumerate(joints_of(child)) if j["name"] == follows]
    if len(own) == 1:
        return child, own[0]
    found = [(ob, k) for ob in everything() for k, j in enumerate(joints_of(ob)) if j["name"] == follows]
    if not found:
        raise ValueError(f"{verb}: follows {follows!r}, and no joint has that name")
    if len(found) > 1:
        where = ", ".join(sorted({repr(_name(ob)) for ob, _ in found}))
        raise ValueError(f"{verb}: follows {follows!r}, which names {len(found)} joints (on {where}: copies carry "
                         f"their joints): give the part it moves, follows=part")
    return found[0]


def _check_follow_chain(leader, k, child, verb):
    """Refuse a leader that is free, or whose own chain comes back to `child`'s new joint."""
    seen = set()
    while leader is not None:
        j = joints_of(leader)[k]
        if j["kind"] == "free":
            raise ValueError(f"{verb}: follows {j['name']!r}, a free joint: it has no value to follow")
        key = (leader.as_pointer(), k)
        if key in seen:
            raise ValueError(f"{verb}: follows {j['name']!r}, whose own chain of follows is a cycle")
        seen.add(key)
        if j.get("follows") is None:
            return
        nxt = leader_of(leader, k)
        k = j["follows"]
        leader = nxt


def _endless(leader, k):
    """Whether the k-th joint of `leader` turns without end: continuous, or following one."""
    j = joints_of(leader)[k]
    if j.get("follows") is None:
        return j["kind"] == "continuous"
    return j["kind"] != "prismatic" and _endless(leader_of(leader, k), j["follows"])


def _posed_above(ob):
    """The first joint not at 0 that moves `ob` — on its body's chain, or on the chain of
    a group it is in — as (child, joint), or None."""
    seen = set()
    while ob is not None:
        b = body(ob)
        while b is not None and b.as_pointer() not in seen:
            seen.add(b.as_pointer())
            for j in joints_of(b):
                if j.get("value", 0.0) != 0.0:
                    return b, j
            p = parent_of(b)
            b = body(p) if p is not None else None
        ob = ob.parent
    return None


def at_rest(verb, *obs):
    """Refuse when a joint that moves one of `obs` is not at 0: the mechanism is not at
    rest, and what is read off it now is no rest."""
    for ob in obs:
        found = _posed_above(ob)
        if found is not None:
            b, j = found
            raise ValueError(f"{verb}: the mechanism is not at rest — {j['name']!r} on {_name(b)!r} is posed at "
                             f"{j['value']:g}: declare joints at rest, every joint at 0 (pose them back to 0, or "
                             f"declare the joints first)")


def joint(name, child, on, kind="revolute", axis="normal", range=None, follows=None, ratio=1):
    """How child moves on the part on was read on: revolute (range in degrees), continuous
    (no range), prismatic (range in the unit), free (no axis, no range); follows: a child
    or a joint name, and this joint's value is ratio × its (degrees per unit, or units per
    degree), no range; returns child.

    A joint `name` that lets `child` — a part, a group or an assembly — move on the part
    the frame `on` was read on (`ob.face`, `bbox`, an assembly's anchor): about `axis`
    through `on`'s origin (revolute, continuous) or along it (prismatic), or freely (free).
    `axis` is a direction ("+Z", "-X"…) in the space the parent lives in, as in `face`,
    or "normal", `on`'s Z; positive values slide along it, or turn by the right hand about
    it. The range contains 0: everything is built at rest, every joint at 0. `joint`
    does not move the child: place it first. A joint that `follows` another — a child
    with one joint, or a joint's name — is posed with it, at `ratio` × its value
    (degrees per unit, or units per degree, across kinds), and takes no range. Two
    joints on one child compose in order."""
    from . import _guard
    from .assembly import Handle
    verb = f"joint {name!r}" if isinstance(name, str) else "joint"
    if not isinstance(name, str) or not name:
        raise TypeError(f"joint: name is the joint's name, a string, not {name!r}")
    ob = child.root if isinstance(child, Handle) else child
    if not isinstance(ob, bpy.types.Object):
        raise TypeError(f"{verb}: child is the part that moves — a solid, a group or an assembly — not "
                        f"{type(child).__name__}")
    if kind not in KINDS:
        raise ValueError(f"{verb}: kind {kind!r} is none of {', '.join(KINDS)}")
    if kind == "free":
        if "axis" in _guard.given("joint", ["axis"], inside=[] if axis == "normal" else ["axis"]):
            raise ValueError(f"{verb}: a free joint moves every way: it takes no axis")
        direction = None
    else:
        direction = _axis(axis, verb)
    if follows is not None and kind == "free":
        raise ValueError(f"{verb}: a free joint follows nothing: it has no value")
    span = _range(range, kind, verb, follows)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio == 0:
        raise ValueError(f"{verb}: ratio is a number other than 0 — the follower's value per unit of the followed "
                         f"joint's — not {ratio!r}")
    if not isinstance(on, Frame):
        raise TypeError(f"{verb}: on is a frame read on the parent part — ob.face(...), s3d.bbox(ob, ...), an "
                        f"assembly's anchor — not {type(on).__name__}")
    parent = on.part
    if parent is None:
        raise ValueError(f"{verb}: on is a frame built by hand, read on no part: read it on the part the child "
                         f"moves on — ob.face(...), s3d.bbox(ob, \"+X-Y\"), an assembly's anchor — so that the "
                         f"joint knows its parent")
    if _guard.removed(parent):
        raise ValueError(f"{verb}: on was read on a part that is gone from the scene")
    if _under(parent, ob):
        where = "the child itself" if parent == ob else f"{_name(parent)!r}, which is inside the child"
        raise ValueError(f"{verb}: on is read on {where}: read it on the part {_name(ob)!r} moves on")
    _still_on(ob, parent, verb)
    known = joints_of(ob)
    if known and parent_of(ob) != parent:
        raise ValueError(f"{verb}: {_name(ob)!r} already hangs on {_name(parent_of(ob))!r} by "
                         f"{known[0]['name']!r}: every joint of one child is read on the same parent")
    b = body(parent, extra=ob)
    seen = set()
    while b is not None:
        if b == ob:
            raise ValueError(f"{verb}: {_name(parent)!r} already hangs on {_name(ob)!r}, through the joints "
                             f"declared: hanging {_name(ob)!r} on it makes a cycle")
        if b.as_pointer() in seen:
            break
        seen.add(b.as_pointer())
        b = body(parent_of(b), extra=ob) if parent_of(b) is not None else None
    leader = None
    if follows is not None:
        leader, k = _resolve_follows(follows, ob, verb)
        if kind == "prismatic" and _endless(leader, k):
            lead = joints_of(leader)[k]["name"]
            raise ValueError(f"{verb}: a slide that follows {lead!r}, which turns without end, would slide without "
                             f"end: give {lead!r} a range")
        _check_follow_chain(leader, k, ob, verb)
    at_rest(verb, ob, parent, *([leader] if leader is not None else []))

    W_child = _core.world(ob)
    W_on = _core.snapped(_core.world(parent) @ on._in_part())
    pivot = _core.snapped(W_child.inverted() @ W_on)
    if direction == "normal":
        a = Vector((0, 0, 1))
    elif direction is None:
        a = None
    else:
        space = _core.world(parent.parent).to_3x3() if parent.parent is not None else Matrix.Identity(3)
        a = _clean(W_on.to_3x3().inverted() @ (space @ direction))
    entry = {"name": name, "kind": kind, "pivot": _flat16(pivot), "axis": None if a is None else list(a),
             "range": span, "follows": None, "ratio": float(ratio), "value": 0.0}
    if leader is not None:
        entry["follows"] = k
        ob[FOLLOWS + str(len(known))] = leader
    if not known:
        ob[REST] = json.dumps(_components(ob))
        ob[PARENT] = parent
    _write(ob, known + [entry])
    return child


def axis_in_world(ob, k, frame=None):
    """(pivot origin, axis) of the k-th joint of `ob`, in the world, the child placed by
    `frame` (its world placement now by default). The axis is the one the motion turns
    about by the right hand, or slides along: a mirrored child's revolute axis is −M·u."""
    j = joints_of(ob)[k]
    F = (_core.world(ob) if frame is None else frame) @ _matrix(j["pivot"])
    if j["axis"] is None:
        return F.to_translation(), None
    a = (F.to_3x3() @ Vector(j["axis"])).normalized()
    if j["kind"] in ("revolute", "continuous") and F.to_3x3().determinant() < 0:
        a = -a
    return F.to_translation(), a


# ---------------------------------------------------------------- posing

def _descendants(ob):
    out, stack = [], [ob]
    while stack:
        o = stack.pop()
        out.append(o)
        stack.extend(sorted(o.children, key=lambda c: c.name))
    return out


def target(ob, joint, verb="pose"):
    """(child, joint index) `pose(ob, …, joint=)` moves: `ob`'s own joint, or the joint
    named `joint` on `ob` or under it."""
    from .assembly import Handle
    root = ob.root if isinstance(ob, Handle) else ob
    if not isinstance(root, bpy.types.Object):
        raise TypeError(f"{verb}: ob is a jointed part, or a group above it with joint=, not {type(ob).__name__}")
    if joint is None:
        own = joints_of(root)
        if len(own) == 1:
            return root, 0
        if own:
            names = ", ".join(repr(j["name"]) for j in own)
            raise ValueError(f"{verb}: {_name(root)!r} has {len(own)} joints ({names}): say which, "
                             f"joint=\"{own[0]['name']}\"")
        under = sorted({j["name"] for o in _descendants(root) for j in joints_of(o)})
        if under:
            raise ValueError(f"{verb}: {_name(root)!r} has no joint of its own: name one of those under it, "
                             f"joint=\"{under[0]}\" ({', '.join(map(repr, under))})")
        raise ValueError(f"{verb}: {_name(root)!r} has no joint: s3d.joint(...) says how a part moves")
    if not isinstance(joint, str):
        raise TypeError(f"{verb}: joint is a joint's name, not {joint!r}")
    found = [(o, k) for o in _descendants(root) for k, j in enumerate(joints_of(o)) if j["name"] == joint]
    if not found:
        under = sorted({j["name"] for o in _descendants(root) for j in joints_of(o)})
        known = f": the joints under it are {', '.join(map(repr, under))}" if under else ": none is under it"
        raise ValueError(f"{verb}: no joint named {joint!r} on {_name(root)!r} or under it{known}")
    if len(found) > 1:
        where = ", ".join(sorted({repr(_name(o)) for o, _ in found}))
        raise ValueError(f"{verb}: {joint!r} names {len(found)} joints under {_name(root)!r} (on {where}: copies "
                         f"carry their joints): pose the part itself, s3d.pose(part, value)")
    return found[0]


def pose(ob, value, joint=None):
    """Move the mechanism: that joint to value (degrees or the unit), an error outside its
    range; what it carries follows; joint= when child has two.

    The mechanism's state, not its definition: `ob`'s joint — or the joint named `joint`
    on `ob` or under it, `ob` a group or an assembly — at `value`, degrees for a turn,
    the program's unit for a slide, within its range (outside it raises); the joints that follow it at
    their ratio of it, and everything they carry with them. 0 is the rest. Returns `ob`."""
    child, k = target(ob, joint)
    j = joints_of(child)[k]
    verb = f"pose {j['name']!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{verb}: value is a number, not {value!r}")
    if j["kind"] == "free":
        raise ValueError(f"{verb}: a free joint has no value to pose: it moves every way")
    if j.get("follows") is not None:
        lead = joints_of(leader_of(child, k))[j["follows"]]["name"]
        raise ValueError(f"{verb}: it follows {lead!r} at {j['ratio']:g} × its value: pose {lead!r}")
    if j["range"] is not None:
        lo, hi = j["range"]
        slack = 1e-9 * max(1.0, abs(lo), abs(hi))
        if not lo - slack <= value <= hi + slack:
            unit = "°" if j["kind"] == "revolute" else ""
            raise ValueError(f"{verb}: {value:g}{unit} is outside its range ({lo:g}, {hi:g})")
    values = _values({(child.as_pointer(), k): float(value)})
    _left_behind(child, values)
    for ob_, joints in _scene_joints():
        changed = False
        for i, jj in enumerate(joints):
            v = values.get((ob_.as_pointer(), i))
            if v is not None and v != jj.get("value", 0.0):
                jj["value"] = v
                changed = True
        if changed:
            _write(ob_, joints)
    place()
    return ob


def _scene_joints():
    return [(ob, joints_of(ob)) for ob in everything()]


def _values(set_now):
    """{(child pointer, index): value} for every joint the new values `set_now` change,
    followers included, down their chains."""
    values = dict(set_now)
    scene = _scene_joints()
    for _ in range(len(scene) + 1):
        grew = False
        for ob, joints in scene:
            for i, j in enumerate(joints):
                if j.get("follows") is None:
                    continue
                lead = leader_of(ob, i)
                if lead is None:
                    continue
                key = (lead.as_pointer(), j["follows"])
                if key in values and values.get((ob.as_pointer(), i)) != j["ratio"] * values[key]:
                    values[(ob.as_pointer(), i)] = j["ratio"] * values[key]
                    grew = True
        if not grew:
            break
    return values


def _left_behind(child, values):
    """Refuse a part placed on a frame read on a part this pose moves, if it does not move
    with it: it would stay behind."""
    movers = {ob.as_pointer() for ob in everything() for i in range(len(joints_of(ob)))
              if (ob.as_pointer(), i) in values}

    def moves(ob):
        b = body(ob)
        seen = set()
        while b is not None and b.as_pointer() not in seen:
            if b.as_pointer() in movers:
                return True
            seen.add(b.as_pointer())
            p = parent_of(b)
            b = body(p) if p is not None else None
        return False
    for ob in bpy.context.scene.objects:
        part = ob.get(ON)
        if part is not None and moves(part) and not moves(ob):
            raise ValueError(f"pose: {_name(ob)!r} was placed on a frame read on {_name(part)!r}, which this pose "
                             f"moves, and it is in no group that moves with it: it would stay behind — group it "
                             f"with {_name(part)!r}, or give it a joint of its own")


def _motion(j):
    """The k-th joint's motion at its value, in the child's own space at rest."""
    v = j.get("value", 0.0)
    if j["kind"] == "free" or v == 0.0:
        return None
    K = _matrix(j["pivot"])
    a = Vector(j["axis"])
    if j["kind"] == "prismatic":
        local = Matrix.Translation(a * v)
    else:
        local = Matrix.Rotation(math.radians(v), 4, a)
    return K @ local @ K.inverted()


class Kinematics:
    """The scene's placements by its joints: at rest (`rest_world`, every jointed part
    on its rest placement) and posed (`posed`, forward kinematics from the values), for
    any object, computed from what the objects hold — not from the order they were
    posed in."""

    def __init__(self, values=None, bases=None):
        self.data = {ob.as_pointer(): joints_of(ob) for ob in everything()}
        for (key, index), value in (values or {}).items():
            self.data[key][index]["value"] = value
        self.bases = bases or {}
        self._rest, self._posed = {}, {}

    def rest_world(self, ob):
        key = ob.as_pointer()
        if key not in self._rest:
            basis = self.bases.get(key)
            if basis is None:
                basis = rest_basis(ob) if key in self.data else ob.matrix_basis
            local = ob.matrix_parent_inverse @ basis
            self._rest[key] = _core.snapped(self.rest_world(ob.parent) @ local) if ob.parent else Matrix(local)
        return self._rest[key]

    def still(self, ob):
        """Whether `ob`'s body and every body above it are at rest."""
        b, seen = body(ob), set()
        while b is not None and b.as_pointer() not in seen:
            if any(j.get("value", 0.0) != 0.0 for j in self.data[b.as_pointer()]):
                return False
            seen.add(b.as_pointer())
            p = parent_of(b)
            b = body(p) if p is not None else None
        return True

    def hung(self, c):
        """Where jointed `c` sits by its parent's pose, before its own joints move it."""
        p = parent_of(c)
        return self.rest_world(c) if p is None else self.posed(p) @ self.rest_world(p).inverted() @ self.rest_world(c)

    def posed_body(self, c):
        key = c.as_pointer()
        if key not in self._posed:
            self._posed[key] = None              # a cycle stops here (declaring refuses them)
            m = self.hung(c)
            for j in self.data[key]:
                d = _motion(j)
                if d is not None:
                    m = m @ d
            self._posed[key] = m
        return self._posed[key] if self._posed[key] is not None else self.rest_world(c)

    def posed(self, ob):
        b = body(ob)
        if b is None:
            return self.rest_world(ob)
        if b == ob:
            return self.posed_body(ob)
        return self.posed_body(b) @ self.rest_world(b).inverted() @ self.rest_world(ob)

    def frames(self, c, rest=False):
        """The world frame of each joint of `c`, after its own motion (none at rest): the
        k-th rides on the motions of those before it."""
        m = self.rest_world(c) if rest else self.hung(c)
        out = []
        for j in self.data[c.as_pointer()]:
            K = _matrix(j["pivot"])
            d = None if rest else _motion(j)
            out.append(m @ K @ (K.inverted() @ d @ K if d is not None else Matrix.Identity(4)))
            if d is not None:
                m = m @ d
        return out


def place():
    """Every jointed part of the scene where its joints' values put it (forward kinematics).
    A part whose chain is all at 0 gets its rest placement back, exactly."""
    k = Kinematics()
    for ob in everything():
        if k.still(ob) and (ob.parent is None or k.still(ob.parent)):
            _restore(ob)
            continue
        above = k.posed(ob.parent) if ob.parent is not None else Matrix.Identity(4)
        ob.matrix_basis = (above @ ob.matrix_parent_inverse).inverted() @ k.posed_body(ob)


# ---------------------------------------------------------------- what other verbs must respect

def refuse_at(ob, verb="at"):
    """`at` on a jointed part: it would tear it off its pivot."""
    found = joints_of(ob)
    if found:
        raise ValueError(f"{verb}: {_name(ob)!r} moves by its joint {found[0]['name']!r}: placing it would tear it "
                         f"off its pivot — place it before the joint, or s3d.transform it, which moves it with its "
                         f"pivot")


def refuse_using_up(tools, verb):
    """`join`, `cut`… using up a jointed part, or one a joint hangs on."""
    held = {}
    for ob in everything():
        p = parent_of(ob)
        if p is not None:
            held.setdefault(p.as_pointer(), (ob, joints_of(ob)[0]["name"]))
    for tool in tools:
        if not isinstance(tool, bpy.types.Object) or _guard_removed(tool):
            continue
        found = joints_of(tool)
        if found:
            raise ValueError(f"{verb}: {_name(tool)!r} moves by its joint {found[0]['name']!r}: {verb} would use it "
                             f"up, and the joint with it (a copy carries its joints too) — make the tool from a part "
                             f"that does not move; s3d.seat(host, part) cuts the cavity and leaves the part as it is")
        if tool.as_pointer() in held:
            ob, jname = held[tool.as_pointer()]
            raise ValueError(f"{verb}: {_name(ob)!r} hangs on {_name(tool)!r} by {jname!r}: {verb} would use "
                             f"{_name(tool)!r} up, and leave the joint on nothing")


def _guard_removed(ob):
    from . import _guard
    return _guard.removed(ob)


def moved(ob, m):
    """`transform` moved `ob` by `m` in its parent's space: its rest moves with it."""
    if jointed(ob):
        scratch = bpy.data.objects.new("s3d~scratch", None)      # Blender's own decomposition of the new rest
        try:
            scratch.rotation_mode = _rest(ob)["mode"]
            scratch.matrix_basis = m @ rest_basis(ob)
            ob[REST] = json.dumps(_components(scratch))
        finally:
            bpy.data.objects.remove(scratch)


def remap(twins):
    """After a copy of several objects ({original: twin}): a joint whose parent, leader or
    placing part was copied with it hangs on, follows, or sits on the twin."""
    for twin in twins.values():
        for key in [k for k in twin.keys() if k == PARENT or k == ON or k.startswith(FOLLOWS)]:
            target = twin[key]
            if target is not None and target in twins:
                twin[key] = twins[target]


def _on_part(part, ob):
    """Where `ob` sits on `part`: its world placement in the part's."""
    return _core.snapped(_core.world(part).inverted() @ _core.world(ob))


def placed_on(ob, on):
    """Record the part the frame `ob` was placed on was read on, and where the frame put
    `ob` on it (or forget both: a frame built by hand, or read on a part a verb has used
    up since)."""
    if not isinstance(ob, bpy.types.Object):
        return
    part = getattr(on, "part", None)
    if part is not None and not _guard_removed(part):
        ob[ON] = part
        seen = on._seen if on._seen is not None else _core.world(part)
        ob[ON_AT] = json.dumps(_flat16(_core.snapped(seen.inverted() @ _core.world(ob))))
    else:
        for key in (ON, ON_AT):
            if key in ob:
                del ob[key]


def placed_again(ob):
    """`at` or `transform` moved `ob`: everything under it that was placed on a part
    outside it now sits there on purpose."""
    from .assembly import Handle
    if isinstance(ob, (list, tuple)):
        for o in ob:
            placed_again(o)
        return
    ob = ob.root if isinstance(ob, Handle) else ob
    if not isinstance(ob, bpy.types.Object):
        return
    for o in [ob, *ob.children_recursive]:
        part = o.get(ON)
        if part is not None and ON_AT in o and not _guard_removed(part):
            o[ON_AT] = json.dumps(_flat16(_on_part(part, o)))


def _still_on(ob, parent, verb):
    """Refuse a child placed on a frame read on its parent that no longer sits where that
    frame put it: the parent moved after the frame was read, or after the child was
    placed. Its pivot would follow the parent, the child would not."""
    part = ob.get(ON)
    if part is None or ON_AT not in ob or _guard_removed(part) or not _under(part, parent):
        return
    kept, now = _matrix(json.loads(ob[ON_AT])), _on_part(part, ob)
    tol = _core.mm(1e-3)
    if any(abs(kept[r][c] - now[r][c]) > (tol if c == 3 else 1e-6) for r in range(3) for c in range(4)):
        raise ValueError(f"{verb}: {_name(ob)!r} is no longer where its frame put it on {_name(part)!r} — "
                         f"{_name(part)!r} moved after the frame was read, or after {_name(ob)!r} was placed: "
                         f"its pivot would follow {_name(part)!r} and {_name(ob)!r} would not. Group them so "
                         f"they move together, or read the frame on {_name(part)!r} where it is now")


def digest_text(ob):
    """What the golden digest hashes of a jointed object: its joints, and the given names
    of its parent and leaders."""
    if JOINTS not in ob:
        return None
    p = parent_of(ob)
    leaders = sorted((k, _name(v)) for k, v in ob.items() if k.startswith(FOLLOWS) and v is not None)
    return f"{ob[JOINTS]}|parent:{_name(p) if p is not None else None}|follows:{leaders}|rest:{ob[REST]}"
