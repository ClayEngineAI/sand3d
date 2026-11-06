"""Edits: verbs that change a solid in place and return it.

Booleans are Blender's EXACT solver, one evaluation for any number of tools, run
on temporary copies in the target's own mesh space — the result never depends on
the depsgraph's state, the active object or the selection. Tools are consumed.

Names after a boolean: the edges where the operands meet are `cut`, plus the
name each tool was given (`hole`, `pocket`, `boss`…), plus `<target>.cut`, plus
the plain cap names of the faces they bound (a hole's mouth on the top is `top`
too). Plain names say where an edge is; qualified ones (`plate.top`) who made it. The
faces a cutting tool leaves behind are `cut` too; the tool's own plain names
(`top`, `side`) are dropped so they cannot be mistaken for the target's, and its
qualified ones (`slot.side`) are kept. An edge left lying flat between two
coplanar faces loses its names.
"""
import bisect
import math
import warnings
from copy import deepcopy

import bpy
import bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import _core, _guard
from .frames import Frame, frames_of
from .joints import refuse_using_up
from .profile import Path, Placed, Profile, poly
from .solids import _drawing, extrude, revolve

OPERAND = "s3d~operand"
#: Custom property marking an object `merge` made: its shells may overlap.
MERGED = "s3d~merged"
FLAT = math.cos(math.radians(0.5))
#: Points of an EXACT result that are distinct in exact arithmetic but round to (nearly)
#: the same float leave invisible cracks; closer than this (mm) they are one point.
WELD = 1e-5


# ---------------------------------------------------------------- the boolean engine

def _tool_mesh(tool, rel, strip):
    """A copy of `tool`'s mesh in the target's space, its plain names dropped if `strip`."""
    me = tool.data.copy()
    me.transform(rel)
    if rel.to_3x3().determinant() < 0:
        me.flip_normals()
    if strip:
        names = [a.name for a in me.attributes]  # by name: removing one invalidates the others (bpy 4.5)
        for name in names:
            for prefix in (_core.FACE_TAG, _core.EDGE_TAG):
                if name.startswith(prefix) and "." not in name[len(prefix):]:
                    me.attributes.remove(me.attributes[name])
                    break
    return me


def _evaluate(base_mesh, tool_meshes, operation, self_intersect=False):
    """Run one EXACT boolean on temporary objects; the evaluated mesh."""
    coll = bpy.data.collections.new("_s3d_tools")
    bpy.context.scene.collection.children.link(coll)
    base = _core.link(bpy.data.objects.new("_s3d_base", base_mesh))
    tools = []
    for me in tool_meshes:
        t = bpy.data.objects.new("_s3d_tool", me)
        coll.objects.link(t)
        tools.append(t)
    mod = base.modifiers.new("boolean", "BOOLEAN")
    mod.operation, mod.solver = operation, "EXACT"
    mod.use_self = self_intersect
    if len(tools) == 1:
        mod.operand_type, mod.object = "OBJECT", tools[0]
    else:
        mod.operand_type, mod.collection = "COLLECTION", coll
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = bpy.data.meshes.new_from_object(base.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                             depsgraph=depsgraph)
    for t in tools:
        bpy.data.objects.remove(t)
    bpy.data.objects.remove(base)
    bpy.data.collections.remove(coll)
    for me in [base_mesh, *tool_meshes]:
        if me.users == 0:
            bpy.data.meshes.remove(me)
    return result


def _solve(base_mesh, tool_meshes, operation, self_intersect):
    """The EXACT result as a bmesh, in a fixed order, its cracks welded."""
    result = _evaluate(base_mesh, tool_meshes, operation, self_intersect)
    bm = bmesh.new()
    bm.from_mesh(result)
    bpy.data.meshes.remove(result)
    _sort_mesh(bm)                       # the modifier's edges come in an order set by the thread count
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=_core.mm(WELD))
    bmesh.ops.dissolve_degenerate(bm, dist=_core.mm(WELD), edges=bm.edges)
    _drop_duplicates(bm)
    return bm


def _drop_duplicates(bm):
    """Keep one of the faces on the same points: the solver leaves some where two
    joined solids cross, back to back. The first is kept and takes the others' names.
    `Mesh.validate()` — which the glTF exporter runs — would drop them anyway, and under
    bpy 4.5 it writes into the arrays a mesh shares with its copies."""
    bm.verts.index_update()
    seen, extra = {}, []
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen:
            extra.append((seen[key], f))
        else:
            seen[key] = f
    if not extra:
        return
    tags = [lay for lay in bm.faces.layers.bool.values() if lay.name.startswith(_core.FACE_TAG)]
    for kept, f in extra:
        for lay in tags:
            if f[lay]:
                kept[lay] = True
    bmesh.ops.delete(bm, geom=[f for _, f in extra], context="FACES_ONLY")


def _operands(ob, tools, strip):
    """`ob`'s mesh and the tools' meshes in `ob`'s frame, each face marked with its operand."""
    to_local = _core.world(ob).inverted()
    base_mesh = ob.data.copy()
    base_mesh.attributes.new(OPERAND, "INT", "FACE")
    tool_meshes = []
    for k, tool in enumerate(tools, 1):
        rel = to_local @ _core.world(tool)
        me = _tool_mesh(tool, rel, strip)
        att = me.attributes.new(OPERAND, "INT", "FACE")
        att.data.foreach_set("value", [k] * len(me.polygons))
        reach = max(abs(c) for m in (_core.world(ob), _core.world(tool)) for c in m.translation)
        _settle(base_mesh, me, reach)
        tool_meshes.append(me)
    return base_mesh, tool_meshes


def _noise(x):
    """Four steps of single precision at the size of `x`."""
    return math.ldexp(1.0, math.frexp(x)[1] - 22) if x else 0.0


def _settle(base_mesh, me, reach):
    """Put each coordinate of the tool mesh `me` on the target's, where the target has
    exactly one within float noise of it: `WELD`, or four steps of single precision at
    the size of the coordinate, or of the placements (`reach`) it went through.

    Meshes and placements are single precision. A post standing on a plate lands a step
    or two off the plate's top in its space, and the solver sees that sliver: the holes
    the posts print in the top come back linked by zero-width bridges, which the weld
    breaks. On the target's coordinate, the contact is exactly flush. Where the
    target has several values that close (a revolve's 3e-15 where cos 90° is 0), the
    tool is left alone."""
    co = [0.0] * (3 * len(base_mesh.vertices))
    base_mesh.vertices.foreach_get("co", co)
    planes = [sorted(set(co[axis::3])) for axis in range(3)]
    floor = max(_core.mm(WELD), _noise(reach))
    co = [0.0] * (3 * len(me.vertices))
    me.vertices.foreach_get("co", co)
    moved = False
    for i, c in enumerate(co):
        values = planes[i % 3]
        near = max(floor, _noise(c))
        lo, hi = bisect.bisect_left(values, c - near), bisect.bisect_right(values, c + near)
        if hi - lo == 1 and values[lo] != c:
            co[i] = values[lo]
            moved = True
    if moved:
        me.vertices.foreach_set("co", co)
        me.update()


def _extent(meshes):
    """(lo, hi) corners of the box round every vertex of `meshes`."""
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for me in meshes:
        co = [0.0] * (3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
        for axis in range(3):
            if co:
                lo[axis] = min(lo[axis], min(co[axis::3]))
                hi[axis] = max(hi[axis], max(co[axis::3]))
    return lo, hi


class _Operands:
    """The meshes of one boolean, and the box its result must stay in: the target's (the
    target's and the tools' for a union). A tool that is exactly an earlier one of the
    same call — a symmetric rib turned by 180° in a ring of eight — adds nothing, and two
    coincident solids break the solver: it is left out (`same`: its number -> the earlier
    one's), and its name goes on the earlier one's edges."""

    def __init__(self, ob, tools, operation, strip):
        self.base, meshes, self.operation = *_operands(ob, tools, strip), operation
        self.box = _extent([self.base] + (meshes if operation == "UNION" else []))
        self.tools_box = _extent(meshes) if operation == "DIFFERENCE" else None
        self.tools, self.same, kept = [], {}, {}
        for k, (tool, me) in enumerate(zip(tools, meshes), 1):
            shape = _Shape(me)
            first = next((j for j, other in kept.get(shape.key, ()) if other.same(shape)), None)
            if first is not None:
                self.same[k] = first
                bpy.data.meshes.remove(me)
            else:
                kept.setdefault(shape.key, []).append((k, shape))
                self.tools.append(me)

    def check(self, bm, verb, name):
        """Raise if `bm` reaches outside the box: a boolean must not hand back something
        else than the part, as the drill instead of the plate. A result left open by
        a sliver the solver could not close is the part, nearly whole: it is returned."""
        move = "clear of the part or into it" if self.operation == "DIFFERENCE" else "into the part"
        hint = (f"a tool grazes the part (tangent to a face, or flush along a line), or a solid crosses itself — move "
                f"the tool {move}, and fuse overlapping solids with s3d.join first")
        lo, hi = self.box
        pad = _core.mm(1e-3) + 1e-6 * max(hi[k] - lo[k] for k in range(3))
        got = [(min(v.co[k] for v in bm.verts), max(v.co[k] for v in bm.verts)) for k in range(3)] if bm.verts else []
        for axis, (low, high) in enumerate(got):
            for value in (low, high):
                if not lo[axis] - pad <= value <= hi[axis] + pad:
                    where = f"{'xyz'[axis]} = {value:.6g}, beyond {lo[axis]:.6g} .. {hi[axis]:.6g}"
                    raise ValueError(f"{verb}: the result on {name!r} reaches outside the target and its tools "
                                     f"({where}): {hint}")


class _Shape:
    """A tool mesh as a solid to compare: two are the same if they have as many points and
    faces, and each point and face of one is on the other's, within `WELD` or the
    single-precision noise of their size."""

    def __init__(self, me):
        self.me = me
        co = [0.0] * (3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
        n = max(len(me.vertices), 1)
        self.centre = Vector([sum(co[i::3]) / n for i in range(3)])
        self.key = (len(me.vertices), len(me.polygons))
        self.near = max(_core.mm(WELD), _noise(max((abs(c) for c in co), default=0.0)))
        self.tree = None

    def same(self, other):
        if (self.centre - other.centre).length > self.near:
            return False
        if self.tree is None:
            self.tree = KDTree(len(self.me.vertices))
            for v in self.me.vertices:
                self.tree.insert(v.co, v.index)
            self.tree.balance()
        to = []
        for v in other.me.vertices:
            _, i, d = self.tree.find(v.co)
            if d > max(self.near, other.near):
                return False
            to.append(i)
        faces = {frozenset(p.vertices) for p in self.me.polygons}
        return all(frozenset(to[i] for i in p.vertices) in faces for p in other.me.polygons)


class NothingCut(ValueError):
    """A cut, a hole or a pocket whose tools all miss the part, or only touch it."""


def _boxes(ob, operands):
    """The part's box and the tools', in its parent's frame — where the program placed them."""
    m = _core.world(ob)
    out = []
    for lo, hi in (operands.box, operands.tools_box):
        corners = [m @ Vector((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        out.append("(" + ", ".join(f"{min(c[k] for c in corners):.6g}" for k in range(3)) + ") .. ("
                   + ", ".join(f"{max(c[k] for c in corners):.6g}" for k in range(3)) + ")")
    return out


def _boolean(ob, tools, operation, strip, self_intersect=False, verb="boolean", aimed=False):
    """`ob` combined with `tools` (consumed) by `operation`; names written as the module says.
    An operand `merge` made may hold overlapping shells: the boolean fuses them first. A
    result that is no solid the operands can give raises; the tools are used up anyway."""
    name = _core.given_name(ob)
    consumed = list(tools)
    self_intersect = self_intersect or any(o.get(MERGED) for o in (ob, *tools))
    operands = _Operands(ob, tools, operation, strip)
    bm = _solve(operands.base, operands.tools, operation, self_intersect)
    if operation == "UNION" and not bm.faces and not self_intersect:
        # a union of solids is never empty: the exact solver gave up on an operand that
        # crosses itself (a wire swept round a corner tighter than its radius), which
        # the self-intersecting solver takes
        bm.free()
        self_intersect = True
        operands = _Operands(ob, tools, operation, strip)
        bm = _solve(operands.base, operands.tools, operation, self_intersect)
    if operation == "UNION" and not bm.faces:
        bm.free()
        names = ", ".join(repr(_core.given_name(t)) for t in tools)
        for tool in consumed:
            delete(tool)
        raise ValueError(f"{verb}: fusing {names} into {name!r} gives nothing — the solver could not read them as "
                         f"solids: a solid that crosses itself, or one that is not closed")
    if operation == "DIFFERENCE" and not bm.faces:
        bm.free()
        for tool in consumed:
            delete(tool)
        raise ValueError(f"{verb}: the tools take all of {name!r} away — nothing would be left")
    if operation == "DIFFERENCE":
        # a tool that leaves no face took no volume out: it only touched, and printed its
        # footprint on the face it stands on, in slivers a later bevel clamps to nothing
        layer = bm.faces.layers.int[OPERAND]
        used = {f[layer] for f in bm.faces}
        used |= {k for k, first in operands.same.items() if first in used}
        removing = [t for k, t in enumerate(tools, 1) if k in used]
        if len(removing) < len(tools):
            bm.free()
            tools = removing
            if not tools:
                for tool in consumed:
                    delete(tool)
                if aimed:   # a tool in the wrong place is the commonest silent mistake: say so
                    part, tools_at = _boxes(ob, operands)
                    raise NothingCut(f"{verb}: no tool cuts into {name!r} — they miss it, or only touch it (its box "
                                     f"{part}; the tools' {tools_at})")
                return ob
            operands = _Operands(ob, tools, operation, strip)
            bm = _solve(operands.base, operands.tools, operation, self_intersect)
    try:
        operands.check(bm, verb, name)
    except ValueError:
        bm.free()
        for tool in consumed:
            delete(tool)
        raise
    tool_names = [{_core.given_name(tool)} for tool in tools]
    for k, first in operands.same.items():          # a tool left out names its twin's edges
        tool_names[first - 1].add(_core.given_name(tools[k - 1]))
    layer = bm.faces.layers.int[OPERAND]
    bm.faces.index_update()
    bm.edges.index_update()
    operand = [f[layer] for f in bm.faces]
    face_add = {}
    if strip:
        face_add = {f.index: {"cut", f"{name}.cut"} for f in bm.faces if operand[f.index] > 0}
    caps = [lay for lay in bm.faces.layers.bool.values()
            if lay.name.startswith(_core.FACE_TAG) and _is_cap(lay.name[len(_core.FACE_TAG):])]
    edge_add = {}
    for e in bm.edges:
        ids = {operand[f.index] for f in e.link_faces}
        if len(ids) > 1:
            # plain cap names say what an edge bounds; qualified ones keep saying who made it
            bounds = {lay.name[len(_core.FACE_TAG):] for f in e.link_faces for lay in caps
                      if f[lay] and "." not in lay.name[len(_core.FACE_TAG):]}
            edge_add[e.index] = {"cut", f"{name}.cut"} | set().union(*(tool_names[i - 1] for i in ids if i > 0)) \
                | bounds
    bm.faces.layers.int.remove(layer)
    _core.add_tags(bm, face_add, edge_add)
    _flatten_names(bm)
    _replace_mesh(ob, bm)
    if MERGED in ob:
        del ob[MERGED]
    for tool in consumed:
        delete(tool)
    return ob


def _flatten_names(bm):
    """Untag the named edges that now lie flat between two coplanar faces."""
    layers = [lay for lay in bm.edges.layers.bool.values() if lay.name.startswith(_core.EDGE_TAG)]
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.link_faces[0].normal.dot(e.link_faces[1].normal) > FLAT:
            if any(e[lay] for lay in layers):
                for lay in layers:
                    e[lay] = False
                e.smooth = True


def _replace_mesh(ob, bm):
    """`bm` as `ob`'s mesh. Normals are not recomputed: a boolean's are right, and a
    void inside a shell must keep facing in."""
    old = ob.data
    mesh = bpy.data.meshes.new(old.name)
    bm.to_mesh(mesh)
    bm.free()
    ob.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    mesh.name = _core.given_name(ob)


# ---------------------------------------------------------------- booleans

def cut(ob, *tools):
    """Subtract tools.

    `ob` with every tool taken out of it, in one boolean; a list of tools is worth its
    tools. To repeat a tool, place copies with `at(tool, on=[frames])`. The tools are
    used up."""
    tools = _flat(tools)
    _solids("cut", ob, *tools)
    refuse_using_up(tools, "cut")
    _guard.using_up(tools, ob)
    return _boolean(ob, tools, "DIFFERENCE", strip=True, verb="cut", aimed=True) if tools else ob


def join(ob, *others):
    """Fuse others in; ob's material wins: group keeps each its own.

    `ob` and `others` fused into one solid, `ob`'s name kept; a list is worth its
    solids. The others are used up, and so are their materials: the solid wears `ob`'s.
    Parts of different materials that touch are a `group`, each keeping its own."""
    others = _flat(others)
    _solids("join", ob, *others)
    refuse_using_up(others, "join")
    _materials_lost(ob, others)
    _guard.using_up(others, ob)
    return _boolean(ob, others, "UNION", strip=False, verb="join") if others else ob


def _materials_lost(ob, others):
    """Warn when `join` uses up a solid wearing another material than `ob`: the
    result wears one material, `ob`'s, and the others' are gone from the export."""
    from .materials import _name, _nearest
    base = _nearest(ob)
    lost = {}
    for other in others:
        tag = _nearest(other)
        if tag is not None and tag != base:
            parts = lost.setdefault(_name(tag), [])
            if _core.given_name(other) not in parts:
                parts.append(_core.given_name(other))
    if not lost:
        return
    line = f" (line {_guard.calls[0][1]})" if _guard.calls else ""
    keeps = f"keeps its {_name(base)!r}" if base is not None else "wears no material"
    which = "; ".join(f"{name!r} on {', '.join(map(repr, parts))}" for name, parts in sorted(lost.items()))
    warnings.warn(f"join{line}: {_core.given_name(ob)!r} {keeps}, and the parts joined into it lose theirs: "
                  f"{which} — to keep each part its material, s3d.group(name, ...) them instead of joining",
                  stacklevel=4)


def intersect(ob, tool):
    """Keep the common part.

    What `ob` and `tool` share. The tool is used up."""
    _solids("intersect", ob, tool)
    refuse_using_up([tool], "intersect")
    _guard.using_up([tool], ob)
    return _boolean(ob, [tool], "INTERSECT", strip=False, verb="intersect")


def seat(host, part):
    """Take out of host the volume part fills, as it sits at rest; part untouched.

    `host` with the volume it shares with `part` taken out — what a lens leaves in its
    rim, a knob in its console — as they sit at rest, every joint at 0, whatever the
    pose: `part` (a solid, a group or an assembly) is left as it is, and may move (a
    knob turning in its seat, a drawer sliding along its runners). Raises if they do not
    share any volume. Returns `host`."""
    from .assembly import Handle
    from .joints import Kinematics
    root = part.root if isinstance(part, Handle) else part
    _solids("seat", host)
    if not isinstance(root, bpy.types.Object):
        raise TypeError(f"seat: the part is a solid, a group or an assembly, not {type(part).__name__}")
    if root == host:
        raise ValueError(f"seat: {_core.given_name(host)!r} is both the host and the part")
    meshes = [o for o in [root, *root.children_recursive] if o.type == "MESH"]
    if host in meshes:
        raise ValueError(f"seat: {_core.given_name(host)!r} is inside the part {_core.given_name(root)!r}")
    if not meshes:
        raise ValueError(f"seat: {_core.given_name(root)!r} holds no solid")
    k = Kinematics()
    to_host = _core.world(host) @ k.rest_world(host).inverted()     # the part where it sits at rest, by the host
    tools = []
    for ob in meshes:
        tool = _core.link(bpy.data.objects.new(_core.given_name(ob), ob.data.copy()))
        tool[_core.NAME] = _core.given_name(ob)
        tool.matrix_basis = _core.snapped(to_host @ k.rest_world(ob))
        tools.append(tool)
    try:
        return _boolean(host, tools, "DIFFERENCE", strip=True, verb="seat", aimed=True)
    except NothingCut:
        raise ValueError(f"seat: {_core.given_name(root)!r} shares no volume with {_core.given_name(host)!r} at "
                         f"rest — they miss each other, or only touch: there is nothing to seat") from None


def _half_space(ob, frame, below):
    """A box filling the side of `frame` (in `ob`'s parent's frame) away from what is kept."""
    corners = [ob.matrix_basis @ Vector(c) for c in _bbox(ob)]
    reach = 4 * (max((a - b).length for a in corners for b in corners) + (frame.origin - corners[0]).length
                 + _core.mm(1))
    tool = box_tool(reach, below)
    tool.matrix_basis = frame.matrix
    tool.parent = ob.parent
    return tool


def box_tool(reach, below):
    """A cube `reach` wide on one side of its XY plane: above if not `below`."""
    b = _core.Builder("split")
    v = [b.vert((x * reach, y * reach, (z - 1 if below else z) * reach))
         for z in (0, 1) for y in (-1, 1) for x in (-1, 1)]
    for f in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6)):
        b.face([v[i] for i in f], 0, ("side",))
    return b.finish("side")


def _bbox(ob):
    vs = [v.co for v in ob.data.vertices]
    lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
    hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
    return [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]


def split(ob, frame, keep="above"):
    """By a frame's plane.

    `ob` cut by the plane of `frame`: keep the part `"above"` it (on its normal's
    side) or `"below"`; `"both"` returns the two, the second a new solid."""
    _solids("split", ob)
    if keep == "both":
        other = _twin(ob)
        return split(ob, frame, "above"), split(other, frame, "below")
    if keep not in ("above", "below"):
        raise ValueError(f"split: keep is 'above', 'below' or 'both', not {keep!r}")
    return _boolean(ob, [_half_space(ob, frame, below=keep == "above")], "DIFFERENCE", strip=True, verb="split")


def _solids(verb, *obs):
    """Refuse what is not a solid, saying what it is and what to do with it."""
    for ob in obs:
        if isinstance(ob, bpy.types.Object) and ob.type == "MESH":
            continue
        if isinstance(ob, bpy.types.Object) and ob.type == "EMPTY":
            raise TypeError(f"{verb}: {_core.given_name(ob)!r} is a group, not a solid: give its parts, or join them "
                            f"into one first")
        if isinstance(ob, bpy.types.Object):
            raise TypeError(f"{verb}: {_core.given_name(ob)!r} is a {ob.type.lower()} object, not a solid")
        if isinstance(ob, (Profile, Placed)):
            raise TypeError(f"{verb}: a profile is not a solid: extrude it first, s3d.extrude(name, profile, h)")
        if isinstance(ob, Path):
            raise TypeError(f"{verb}: a path is not a solid: s3d.wire(name, path, d) or s3d.sweep(name, profile, path)")
        raise TypeError(f"{verb}: a solid is expected, not {type(ob).__name__}")


def _flat(items):
    """Objects from objects and lists of them, in order: a list of objects is worth its
    elements; a handle stands for its root."""
    from .assembly import Handle
    out = []
    for item in items:
        if isinstance(item, (list, tuple)):
            out += _flat(item)
        else:
            out.append(item.root if isinstance(item, Handle) else item)
    return out


def _delete_tree(ob):
    """Remove `ob` and everything under it."""
    for child in list(ob.children):
        _delete_tree(child)
    delete(ob)


def delete(ob):
    """Remove it.

    Remove `ob` from the scene; an object a verb used up is gone already."""
    if _guard.removed(ob):
        return
    _guard.deleting(ob)
    mesh = ob.data if ob.type == "MESH" else None
    bpy.data.objects.remove(ob)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


# ---------------------------------------------------------------- holes and pockets

def _frames(ob, on):
    """Frames from a frame, a point or a list of either — a point (x, y) facing up from
    `ob`'s top, (x, y, z) facing up where it is — each set where its axis enters `ob`
    from the side the normal points to (`_entry`)."""
    if on is None:
        on = (0, 0)
    if isinstance(on, Frame):
        on = [on]
    elif on and isinstance(on[0], (int, float)):
        on = [on]
    top = max((ob.matrix_basis @ v.co).z for v in ob.data.vertices) if any(
        not isinstance(a, Frame) and len(a) == 2 for a in on) else 0.0
    frames = [a if isinstance(a, Frame) else Frame((a[0], a[1], top) if len(a) == 2 else tuple(a)) for a in on]
    tree = _core.mm_tree(ob, ob.matrix_basis)
    return [_entry(ob, tree, f) for f in frames]


def _entry(ob, tree, frame):
    """`frame` moved along its normal to where its axis enters `ob` from the side the
    normal points to. On a face facing that way, it stays as it is. Inside the part, it
    goes up to the face the axis leaves by; in the air, down to the first face it meets.
    The meeting point is worked out on that face's plane, in double precision. An axis
    that misses the part leaves the frame where it is."""
    from .frames import _plane
    size = _core.in_mm(1.0)
    o, n = frame.origin, frame.z
    near = tree.find_nearest(o * size)
    tol = max(WELD, _noise(max(abs(c) for c in o * size)))
    if near[0] is not None and near[3] <= tol and near[1].dot(n) > 0:
        return frame                                        # on a face looking its way
    up = tree.ray_cast(o * size + n * (2 * tol), n)         # from just past a face looking away, if on one
    hit = up if up[0] is not None and up[1].dot(n) > 0 else None   # inside: the face the axis leaves by
    if hit is None:
        down = tree.ray_cast(o * size, -n)
        hit = down if down[0] is not None and down[1].dot(n) > 0 else None
    if hit is None:
        return frame
    loc, _, index, _ = hit
    normal, point = _plane(ob.data, [ob.data.polygons[index]], ob.matrix_basis)
    along = sum(a * b for a, b in zip(normal, n)) if normal is not None else 0.0
    if abs(along) < 1e-9:                                   # a sliver: the hit itself will do
        return Frame(tuple(c / size for c in loc), n, frame.y)
    t = sum(a * (p - q) for a, p, q in zip(normal, point, o)) / along   # o + n·t lies on the face's plane
    return Frame(tuple(oc + nc * t for oc, nc in zip(o, n)), n, frame.y)


def _reach(ob, frame):
    corners = [ob.matrix_basis @ c for c in _bbox(ob)]
    span = max((a - b).length for a in corners for b in corners) + (frame.origin - corners[0]).length
    return 2 * span + _core.mm(1)


def _place_tool(tool, ob, frame):
    tool.matrix_basis = frame.matrix
    tool.parent = ob.parent
    return tool


def hole(ob, d, on=(0, 0), depth=None, cbore=None, csk=None):
    """On frames or points: through, or `depth` deep.

    Round holes of diameter `d` into `ob`, one per frame or point of `on`: each starts
    where its axis enters the part, from the side the frame's normal points to (a point
    (x, y) faces down onto the part), and runs right through, or `depth` deep.
    `cbore=(D, depth)` adds a counterbore, `csk=(D, angle)` a countersink. Mouth edges
    are named `hole`. Above the entry plane, a countersink is bounded by a cylinder
    of its mouth diameter; its cone extends inward only."""
    _solids("hole", ob)
    if cbore and not (hasattr(cbore, "__len__") and len(cbore) == 2):
        raise TypeError(f"hole: cbore is (diameter, depth), not {cbore!r}")
    if csk and not isinstance(csk, (int, float)) and not (hasattr(csk, "__len__") and len(csk) == 2):
        raise TypeError(f"hole: csk is a diameter, or (diameter, angle), not {csk!r}")
    frames = _frames(ob, on)
    tools = []
    for frame in frames:
        reach = _reach(ob, frame)
        over = max(_core.mm(0.01), 0.01 * d) if depth is not None else reach   # a through hole: both ways
        bottom = -(depth if depth is not None else reach)
        pts = [(0, over)]
        big, angle = ((csk, 90) if isinstance(csk, (int, float)) else csk) if csk else (0, 90)
        if cbore and cbore[0] > d:
            big, cdepth = cbore
            pts += [(big / 2, over), (big / 2, -cdepth), (d / 2, -cdepth)]
        elif csk and big > d:  # a countersink no wider than the hole leaves a plain hole
            fall = (big - d) / 2 / math.tan(math.radians(angle) / 2)
            pts += [(big / 2, over), (big / 2, 0), (d / 2, -fall)]
        else:
            pts += [(d / 2, over)]
        pts += [(d / 2, bottom), (0, bottom)]
        tools.append(_place_tool(revolve("hole", poly(pts)), ob, frame))
    return _boolean(ob, tools, "DIFFERENCE", strip=True, verb="hole", aimed=True)


def pocket(ob, profile, depth=None, on=None):
    """Profile cut in from a face.

    `profile`, drawn in the face's plane (moved within it only), cut into `ob` from the
    frame `on` (by default the surface over `ob`'s
    origin, facing up, where a hole on (0, 0) would start),
    `depth` deep against its normal, or right through; with a list of frames, once on
    each. Floor and walls are named `pocket.bottom`, `pocket.side`."""
    _solids("pocket", ob)
    if isinstance(profile, (list, tuple)) and any(isinstance(p, (Profile, Placed)) for p in profile):
        raise TypeError("pocket: one profile per pocket; to repeat it, give on=[frames]")
    if isinstance(on, (list, tuple)):
        missed = []
        for f in frames_of(on, "pocket"):     # one after the other, as the loop it replaces
            try:
                pocket(ob, profile, depth, f)
            except NothingCut as error:       # one frame off an irregular part: dropped, as a hole's
                missed.append(error)
        if len(missed) == len(on):
            raise missed[0]
        return ob
    profile = _drawing(profile, "pocket: the profile is placed out of the face's plane: draw it in its plane — "
                                "x, y and rz only — and give the face or frames with on=: s3d.pocket(ob, profile, "
                                "depth, on=…)")
    frame = on if on is not None else _frames(ob, (0, 0))[0]
    if isinstance(frame, Placed):
        raise ValueError("pocket: place the profile with `on`, not `at`")
    over = _core.mm(0.01)
    reach = depth if depth is not None else _reach(ob, frame)
    tool = extrude("pocket", profile, -reach, h2=-over)
    return _boolean(ob, [_place_tool(tool, ob, frame)], "DIFFERENCE", strip=True, verb="pocket", aimed=True)


# ---------------------------------------------------------------- edges: fillet, chamfer, soften

def _names(edges):
    if edges is None or edges == "all":
        return None
    return edges.split() if isinstance(edges, str) else list(edges)


def _named_edges(ob, bm, edges):
    """The edges of `bm` carrying any of the names in `edges` ("all" or None: every named edge)."""
    names = _names(edges)
    layers = {lay.name[len(_core.EDGE_TAG):]: lay for lay in bm.edges.layers.bool.values()
              if lay.name.startswith(_core.EDGE_TAG)}
    if names is None:
        chosen = list(layers.values())
    else:
        missing = [n for n in names if n not in layers]
        if missing:
            raise ValueError(_core.unknown(_core.given_name(ob), "edge", missing, sorted(layers)))
        chosen = [layers[n] for n in names]
    found = [e for e in bm.edges if any(e[lay] for lay in chosen)]
    if not found:
        raise ValueError(f"{_core.given_name(ob)}: no edge carries {edges!r} any more")
    return found


def _is_cap(tag):
    return tag.split(".")[-1] in _core.CAPS


def _retag(bm, new_faces, kind, name):
    """After a bevel: the new faces lose what they inherited and take `kind` alone — a
    bevel face is part of neither face it cuts; every edge touching them loses what it
    inherited and, if it is still a real edge, takes `kind` and the cap names of its
    faces. Real: where a new face meets an old one at more than 10°, or two chamfer
    faces meet at more than 40° (a mitre, not a facet of a rounded rim)."""
    bm.faces.index_update()
    bm.edges.index_update()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    new = {f.index for f in new_faces if f.is_valid}
    face_layers = {lay.name[len(_core.FACE_TAG):]: lay for lay in bm.faces.layers.bool.values()
                   if lay.name.startswith(_core.FACE_TAG)}
    edge_layers = [lay for lay in bm.edges.layers.bool.values() if lay.name.startswith(_core.EDGE_TAG)]
    caps = {t: lay for t, lay in face_layers.items() if _is_cap(t)}
    for i in new:  # Blender copied one neighbour's names, picked by which way the edge runs
        for lay in face_layers.values():
            bm.faces[i][lay] = False
    face_add = {i: {kind, f"{name}.{kind}"} for i in new}
    edge_add = {}
    for e in bm.edges:
        faces = list(e.link_faces)
        if not any(f.index in new for f in faces):
            continue
        for lay in edge_layers:
            e[lay] = False
        e.smooth = True
        inside = all(f.index in new for f in faces)  # between two new faces: a facet, or a mitre
        limit = math.cos(math.radians(40 if kind == "chamfer" else 180)) if inside else math.cos(math.radians(10))
        if len(faces) == 2 and faces[0].normal.dot(faces[1].normal) < limit:
            edge_add[e.index] = {kind, f"{name}.{kind}"} | {t for t, lay in caps.items()
                                                           if any(f[lay] for f in faces)}
    _core.add_tags(bm, face_add, edge_add)
    return sorted(new)


def _bevel(ob, edges, offset, segments, kind, radius=False, verb=None, cap_edges=False):
    """Bevel the chosen edges by `offset` along each face; with `radius`, `offset` is a
    fillet radius and each edge is set back r·tan(φ/2) for its own bend φ — a true
    radius on a shallow edge as on a square one (edges are bevelled in groups of
    equal bend).

    A loop whose bend changes along it — the cap of a loft of superellipses — falls in
    many groups, and bevelled a group at a time its pieces overlap: the solid comes back
    open, or closed by slivers the export's single precision folds open. Then every
    chosen edge is also bevelled in one operation, each at its own setback
    (`_bevel_weighted`), and that is kept if it is cleaner — fewer open edges, or as many
    and fewer slivers. A bevel that still leaves the solid more open than it was raises."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.normal_update()
    chosen = _named_edges(ob, bm, edges) if not callable(edges) else edges(bm)
    if cap_edges:
        caps = [lay for lay in bm.faces.layers.bool.values()
                if lay.name.startswith(_core.FACE_TAG) and _is_cap(lay.name[len(_core.FACE_TAG):])]
        if any(not any(face[lay] for face in edge.link_faces for lay in caps) for edge in chosen):
            bm.free()
            raise ValueError("chamfer: unequal distances require every selected edge to adjoin a top, bottom, start, or end face")
    groups, picks = {}, []
    for e in chosen:
        if radius and len(e.link_faces) == 2:
            bend = e.link_faces[0].normal.angle(e.link_faces[1].normal, math.pi / 2)
            setback = round(_core.in_mm(offset * math.tan(min(bend, math.radians(179)) / 2)), 6) / _core.MM
        else:
            setback = offset
        groups.setdefault(setback, []).append(e)
        picks.append((e.index, setback))
    before = (_open_edges(bm), _slivers(bm))
    result = _bevel_groups(ob, bm, _close_setbacks(groups), segments, kind)
    first = (_open_edges(result[0]), _slivers(result[0]))
    if first[0] <= before[0] and first[1] <= before[1]:
        return result
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    again = _bevel_weighted(ob, bm, picks, segments, kind)
    if (_open_edges(again[0]), _slivers(again[0])) < first:
        result[0].free()
        result = again
    else:
        again[0].free()
    if _open_edges(result[0]) > before[0]:
        result[0].free()
        raise ValueError(f"{verb or kind}: {offset:g} on the edges {edges if not callable(edges) else 'sharp'!r} of "
                         f"{_core.given_name(ob)!r} would leave the solid open — it does not fit them: give less, or "
                         f"round the edges in separate calls")
    return result


def _open_edges(bm):
    """How many edges of `bm` do not have exactly two faces: 0 for a closed solid."""
    return sum(1 for e in bm.edges if len(e.link_faces) != 2)


#: An edge shorter than this (mm) is a sliver: a metre-scale export in single precision
#: puts its two ends on one point, and the solid opens there.
SLIVER = 1e-3


def _slivers(bm):
    """How many edges of `bm` are slivers (`SLIVER`)."""
    return sum(1 for e in bm.edges if e.calc_length() < _core.mm(SLIVER))


def _bevel_groups(ob, bm, groups, segments, kind):
    """Bevel each group of edges of `bm` at its setback, the largest first, and weld and
    name what that made: (bm, the new faces)."""
    faces = []
    size = _core.in_mm(1.0)   # Blender's bevel has tolerances of its own, in its units: run it in millimetres
    if size != 1.0:
        bmesh.ops.scale(bm, vec=(size, size, size), verts=bm.verts)
    for setback in sorted(groups, reverse=True):
        geom = [e for e in groups[setback] if e.is_valid]
        if not geom or setback <= 0:
            continue
        out = bmesh.ops.bevel(bm, geom=geom, offset=setback * size, offset_type="OFFSET", segments=segments,
                              profile=0.5, affect="EDGES", clamp_overlap=True, loop_slide=True)
        faces += out["faces"]
    if size != 1.0:
        bmesh.ops.scale(bm, vec=(1 / size,) * 3, verts=bm.verts)
    return _settle_bevel(ob, bm, faces, kind)


def _bevel_weighted(ob, bm, picks, segments, kind):
    """Bevel every picked edge [(index, setback)] of `bm` in one operation, each at
    its own setback, and weld and name what that made: (bm, the new faces). Frees `bm`.

    `bmesh.ops.bevel` takes one offset; Blender's bevel modifier scales its width by an edge
    weight, so every setback is the largest times its share. Its new faces are told apart by
    the material index it gives them; a solid under construction has none of its own
    (`materials.resolve` sets them at export)."""
    if not picks or max(s for _, s in picks) <= 0:
        return _settle_bevel(ob, bm, [], kind)
    size = _core.in_mm(1.0)
    top = max(s for _, s in picks)
    me = bpy.data.meshes.new("_s3d_bevel")
    bm.to_mesh(me)
    bm.free()
    if size != 1.0:
        me.transform(Matrix.Scale(size, 4))
    weight = me.attributes.new("s3d~weight", "FLOAT", "EDGE")
    for i, s in picks:
        weight.data[i].value = s / top
    for p in me.polygons:
        p.material_index = 0
    me.materials.append(None)
    me.materials.append(None)
    temp = _core.link(bpy.data.objects.new("_s3d_bevel", me))
    mod = temp.modifiers.new("bevel", "BEVEL")
    mod.limit_method, mod.edge_weight, mod.affect = "WEIGHT", "s3d~weight", "EDGES"
    mod.width, mod.offset_type, mod.segments, mod.profile = top * size, "OFFSET", segments, 0.5
    mod.use_clamp_overlap, mod.loop_slide, mod.material = True, True, 1
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph), preserve_all_data_layers=True,
                                             depsgraph=depsgraph)
    bpy.data.objects.remove(temp)
    bpy.data.meshes.remove(me)
    out = bmesh.new()
    out.from_mesh(result)
    bpy.data.meshes.remove(result)
    if size != 1.0:
        bmesh.ops.scale(out, vec=(1 / size,) * 3, verts=out.verts)
    lay = out.edges.layers.float.get("s3d~weight")
    if lay is not None:
        out.edges.layers.float.remove(lay)
    out.faces.ensure_lookup_table()
    faces = [f for f in out.faces if f.material_index == 1]
    for f in out.faces:
        f.material_index = 0
    return _settle_bevel(ob, out, faces, kind)


def _settle_bevel(ob, bm, faces, kind):
    """Weld what a bevel made on `bm` — `faces`, the new ones — and name it: (bm, the new faces)."""
    bm.faces.index_update()
    made = {f.index for f in faces if f.is_valid}
    mark = bm.faces.layers.int.new("s3d~bevel")   # a new layer drops every Python reference: by index
    bm.faces.ensure_lookup_table()
    for i in made:
        bm.faces[i][mark] = 1
    _sort_mesh(bm)   # the welds below collapse in the order they are given, which bevel leaves to the run
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=_core.mm(WELD))   # a bevel clamped to nothing leaves
    bmesh.ops.dissolve_degenerate(bm, dist=_core.mm(WELD), edges=bm.edges)  # zero-width strips: weld them away
    bm.faces.index_update()
    made = [f.index for f in bm.faces if f[mark]]
    bm.faces.layers.int.remove(mark)              # and so does removing it: by index again
    bm.faces.ensure_lookup_table()
    new = _retag(bm, [bm.faces[i] for i in made], kind, _core.given_name(ob))
    bm.faces.ensure_lookup_table()
    return bm, [bm.faces[i] for i in new]


def _close_setbacks(groups):
    """Groups of edges whose setbacks are within 1 % of each other as one, at their mean
    setback: a loop whose faces a boolean split bends by 18.50° here and 18.53° there,
    and bevelled a piece at a time, each piece clamps the next to nothing."""
    runs = []
    for setback in sorted(groups, reverse=True):
        if runs and runs[-1][0] - setback <= 0.01 * runs[-1][0]:
            runs[-1].append(setback)
        else:
            runs.append([setback])
    merged = {}
    for run in runs:
        edges = [e for s in run for e in groups[s]]
        merged[run[0] if len(run) == 1 else sum(s * len(groups[s]) for s in run) / len(edges)] = edges
    return merged


def _sort_mesh(bm):
    """Vertices by position, edges and faces by their vertices: bevel and the boolean modifier
    make them in an order that changes with the run or the machine's thread count. Only
    where nothing holds an element: sorting moves them."""
    def by(seq, key):
        rank = {x: i for i, x in enumerate(sorted(seq, key=key))}
        seq.sort(key=rank.__getitem__)
        seq.index_update()
    by(bm.verts, lambda v: tuple(v.co))
    by(bm.edges, lambda e: sorted(v.index for v in e.verts))
    by(bm.faces, lambda f: [v.index for v in f.verts])


def fillet(ob, r, edges="all"):
    """Round named edges.

    Round the edges named `edges` ("top", "top side", ["hole", "cut"]; "all": every
    named edge) to radius `r`. Where it does not fit, the radius is clamped; a radius
    that would leave the solid open raises."""
    _solids("fillet", ob)
    bm, _ = _bevel(ob, edges, r, max(2, _core.segments(r, 90)), "fillet", radius=True, verb="fillet")
    _sort_mesh(bm)
    _replace_mesh(ob, bm)
    return ob


def chamfer(ob, d, edges="all", d2=None, angle=None):
    """Bevel named edges.

    Cut the edges named `edges` by `d` on both faces; or `d` on the face the edge is
    named after (`top` for a top edge) and `d2` on the other — or the other at `angle`
    degrees from it. Unequal positive distances require every selected edge to adjoin
    a top, bottom, start, or end face; unsupported side edges raise before editing."""
    _solids("chamfer", ob)
    if angle is not None:
        if not 1 <= angle < 90:
            raise ValueError(f"chamfer: angle is the face angle in degrees, from 1 up to 90, not {angle:.4g}"
                             + (" — radians? give degrees" if 0 < angle < 1 else ""))
        d2 = d * math.tan(math.radians(angle))
    base = d if d2 is None else min(d, d2)   # cut equal first, as far as both faces allow
    unequal = d2 is not None and base > 0 and abs(d2 - d) > _core.mm(1e-9)
    bm, faces = _bevel(ob, edges, base, 1, "chamfer", verb="chamfer", cap_edges=unequal)
    if d2 is not None and (abs(d2 - base) > _core.mm(1e-9) or abs(d - base) > _core.mm(1e-9)):
        _stretch(bm, faces, base, d, d2)
    _sort_mesh(bm)
    _replace_mesh(ob, bm)
    return ob


def _stretch(bm, strip, base, d, d2):
    """Move the rails of an equal chamfer cut `base` deep: the one on the face the edge
    is named after to `d` off the edge it replaced, the other to `d2` — each within its
    own face, square to the rail, away from the edge."""
    caps = [lay for lay in bm.faces.layers.bool.values()
            if lay.name.startswith(_core.FACE_TAG) and _is_cap(lay.name[len(_core.FACE_TAG):])]
    strip = [f for f in strip if f.is_valid and len(f.verts) == 4]
    chamfer_faces = set(strip)
    bm.normal_update()
    moves = {}
    for f in strip:
        side = {v: any(g[lay] for g in v.link_faces if g not in chamfer_faces for lay in caps) for v in f.verts}
        on_cap = [v for v in f.verts if side[v]]
        off_cap = [v for v in f.verts if not side[v]]
        if len(on_cap) != 2 or len(off_cap) != 2:
            continue
        for rail, other, target in ((off_cap, on_cap, d2), (on_cap, off_cap, d)):
            if abs(target - base) < _core.mm(1e-12):
                continue
            along = rail[1].co - rail[0].co
            if along.length < _core.mm(1e-12):
                continue
            t = along.normalized()
            edge = next((e for e in rail[0].link_edges if e.other_vert(rail[0]) is rail[1]), None)
            faces = [g for g in edge.link_faces if g is not f] if edge else []
            if not faces:
                continue
            n = faces[0].normal
            for v in rail:
                partner = min(other, key=lambda o: (o.co - v.co).length)
                u = v.co - partner.co
                u = u - n * u.dot(n)
                u = u - t * u.dot(t)
                if u.length > _core.mm(1e-12):
                    moves.setdefault(v, []).append(v.co + u.normalized() * (target - base))
    for v, targets in moves.items():
        v.co = sum(targets, Vector()) / len(targets)


def soften(ob, r, angle=30):
    """Round all sharp edges.

    Round every edge whose faces meet at more than `angle` degrees to radius `r` — a
    whole object eased, the way a moulded or worn part is."""
    _solids("soften", ob)
    limit = math.radians(angle)

    def sharp(bm):
        bm.normal_update()
        return [e for e in bm.edges if len(e.link_faces) == 2
                and e.link_faces[0].normal.angle(e.link_faces[1].normal, 0) > limit]
    bm, _ = _bevel(ob, sharp, r, max(2, _core.segments(r, 90)), "fillet", radius=True, verb="soften")
    _sort_mesh(bm)
    _replace_mesh(ob, bm)
    return ob


# ---------------------------------------------------------------- shell

def _named_faces(ob, names):
    layers = {a.name[len(_core.FACE_TAG):]: a for a in ob.data.attributes if a.name.startswith(_core.FACE_TAG)}
    missing = [n for n in names if n not in layers]
    if missing:
        raise ValueError(_core.unknown(_core.given_name(ob), "face", missing, sorted(layers)))
    return {i for n in names for i, d in enumerate(layers[n].data) if d.value}


#: How a curved side is opened, said by `shell` when it is asked to open one by direction.
CUT_FLAT = ("a curved side is opened by cutting it flat first: s3d.split(ob, frame) or s3d.cut(ob, tool), then "
            "shell(ob, t, open=\"cut\") (or the tool's name)")
#: A crease, in degrees: two faces meeting at more (`_retag`'s rule for a real edge).
CREASE = 10
#: Where the faces a direction picks run on smoothly into others, they are a face only if they
#: are this many times the area of the faces they run on into: a flat top blending into a fillet
#: is; the one or two facets at the top of a superellipse (96 of them round) are not.
FACE_SPAN = 4


def _facets(ob, picked):
    """Whether the faces `picked` of `ob` (indices, flat, looking one way) are facets of a curved
    side rather than a face: they run on into the faces round them without a crease
    (`CREASE`), and are not much larger than those (`FACE_SPAN`)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    limit = math.radians(CREASE)
    smooth = set()
    for i in picked:
        f = bm.faces[i]
        for e in f.edges:
            for g in e.link_faces:
                if g.index not in picked and f.normal.angle(g.normal, math.pi) < limit:
                    smooth.add(g.index)
    area = sum(bm.faces[i].calc_area() for i in picked)
    around = sum(bm.faces[i].calc_area() for i in smooth)
    bm.free()
    return bool(smooth) and area < FACE_SPAN * around


def shell(ob, t, open=None):
    """Hollow; open named faces, or the flat faces looking "+Y", "-X +X".

    Hollow `ob`, leaving walls `t` thick inside its surface; the faces `open` gives are
    taken away: by name ("top", ["top", "start"]), or by the way they look ("+Y",
    "-X +X"), as `ob.face` reads a direction — the flat face furthest that way. Names
    and directions mix ("top -Y"). A curved side has no such face, even where its facets
    look that way: it is refused, and opened by cutting it flat first. Without `open`, a
    closed void."""
    from .frames import _DIRECTION, _facing
    _solids("shell", ob)
    terms = _names(open) or []
    ways = [w for w in terms if _DIRECTION.match(w)]
    names = [w for w in terms if not _DIRECTION.match(w)]
    opened = _named_faces(ob, names) if names else set()
    for way in ways:
        found = _facing(ob, way)
        if not found:
            raise ValueError(f"shell: {_core.given_name(ob)!r} has no flat face looking {way} to open — {CUT_FLAT}")
        if _facets(ob, {p.index for p in found}):
            raise ValueError(f"shell: {_core.given_name(ob)!r} has no flat face looking {way}: the faces furthest "
                             f"that way are facets of a curved side, a strip and not a face — a face its maker "
                             f"named opens by its name (open=\"top\"); {CUT_FLAT}")
        opened |= {p.index for p in found}
    me = ob.data
    normals = [p.normal.copy() for p in me.polygons]
    planes = {v.index: [] for v in me.vertices}
    for p in me.polygons:
        for v in p.vertices:
            planes[v].append((normals[p.index], 1.0 if p.index in opened else -1.0))
    inner = bmesh.new()
    bverts = []
    for v in me.vertices:
        bverts.append(inner.verts.new(v.co + _offset(planes[v.index], t)))
    for p in me.polygons:
        inner.faces.new([bverts[i] for i in p.vertices])
    tool_mesh = bpy.data.meshes.new("shell")
    inner.to_mesh(tool_mesh)
    inner.free()
    tool = _core.link(bpy.data.objects.new("shell", tool_mesh))
    tool[_core.NAME] = "shell"
    tool.matrix_basis, tool.parent = ob.matrix_basis, ob.parent
    return _boolean(ob, [tool], "DIFFERENCE", strip=True, self_intersect=True, verb="shell")


def _offset(planes, t):
    """The move that puts a vertex `t` inside every distinct face plane it touches (or
    `t` outside those marked +1): least squares over the planes, even thickness.

    Solved in double precision — `Matrix` is float32, and the planes of a smooth surface
    leave the system ill-conditioned (two planes 3.6° apart: 1e6), which float32 turns
    to noise. A direction the planes barely pin down (under `_WEAK` of the strongest:
    the edge two planes share, the slack of near-parallel faces) is left alone: the
    vertex stays put along it, instead of sliding sideways to satisfy a hair of
    disagreement between the normals."""
    distinct = []
    for n, s in planes:
        if all(n.dot(m) < 0.9999 or s != r for m, r in distinct):
            distinct.append((n, s))
    if len(distinct) == 1:
        n, s = distinct[0]
        return n * (s * t)
    a = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for n, s in distinct:
        for i in range(3):
            b[i] += n[i] * (s * t)
            for j in range(3):
                a[i][j] += n[i] * n[j]
    w, v = _eigen3(a)
    x = [0.0] * 3
    for k in range(3):
        if w[k] >= _WEAK * max(w):
            c = sum(v[i][k] * b[i] for i in range(3)) / w[k]
            for i in range(3):
                x[i] += v[i][k] * c
    return Vector(x)


_WEAK = 0.03


def _eigen3(a):
    """Eigenvalues and eigenvectors (the columns of `v`) of the symmetric 3x3 `a`, by
    Jacobi rotations in double precision: deterministic, and exact for equal values
    (a corner of three square planes has three)."""
    a = [row[:] for row in a]
    v = [[float(i == j) for j in range(3)] for i in range(3)]
    for _ in range(16):
        if abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]) <= 1e-15 * (a[0][0] + a[1][1] + a[2][2]):
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if a[p][q] == 0.0:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            tan = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(tan * tan + 1.0)
            s = tan * c
            for k in range(3):
                a[k][p], a[k][q] = c * a[k][p] - s * a[k][q], s * a[k][p] + c * a[k][q]
            for k in range(3):
                a[p][k], a[q][k] = c * a[p][k] - s * a[q][k], s * a[p][k] + c * a[q][k]
            for k in range(3):
                v[k][p], v[k][q] = c * v[k][p] - s * v[k][q], s * v[k][p] + c * v[k][q]
    return [a[0][0], a[1][1], a[2][2]], v


# ---------------------------------------------------------------- whole objects

def copy(ob):
    """An independent copy, its joints with it.

    An independent copy of `ob`: the same mesh, edge and face names, name and
    placement — a group or an assembly with its children, a handle with its anchors;
    a jointed part with its joints, hanging on the same parent (on the copy of its
    parent, when the copy holds both).
    `ob` is untouched; to keep a tool a verb uses up, give it the copy."""
    return _copy(ob)


def _copy(ob, children=None):
    """`copy`, given the scene's {parent: children} if it is known already."""
    from .assembly import Handle
    if not isinstance(ob, Handle):
        return _copy_tree(ob, children)[ob]
    twins = _copy_tree(ob.root, children)
    memo = {id(o): twins.get(o, o) for o in _ids(vars(ob))}   # an object anchor becomes its twin
    handle = object.__new__(type(ob))
    handle.__dict__.update(deepcopy(vars(ob), memo))
    return handle


def _twin(ob):
    """A new object like `ob` — placement, custom properties, its own copy of the mesh —
    in the scene, under the same parent."""
    twin = ob.copy()                   # Blender shares the mesh: copy it
    if ob.data is not None:
        twin.data = ob.data.copy()
    twin[_core.NAME] = _core.given_name(ob)   # a group has none; the copy's own name is suffixed
    return _core.link(twin)


def _children():
    """{parent: [children]} over the scene, in one pass."""
    out = {}
    for o in bpy.data.objects:
        if o.parent is not None:
            out.setdefault(o.parent, []).append(o)
    return out


def _copy_tree(root, children=None):
    """`root` and everything under it as new objects: {original: copy}."""
    if children is None:
        children = _children() if root.children else {}
    twins = {}

    def walk(ob, parent):
        twins[ob] = twin = _twin(ob)
        twin.parent = parent
        for child in children.get(ob, ()):
            walk(child, twin)
    walk(root, root.parent)
    from .joints import remap
    remap(twins)                         # a joint inside the copy hangs on the copy of its parent
    return twins


def _ids(value, seen=None):
    """The Blender objects reachable from an anchor, through containers and attributes."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, bpy.types.ID):
        return [value]
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        items = list(vars(value).values())
    else:
        return []
    return [i for item in items for i in _ids(item, seen)]


def merge(*obs):
    """No boolean.

    One object holding the meshes of `obs`, untouched — no boolean: parts that
    touch stay separate shells. The first object's name is kept; the others are used up.
    A boolean on it later fuses its shells first."""
    obs = _flat(obs)
    first, others = obs[0], obs[1:]
    refuse_using_up(others, "merge")
    _guard.using_up(others, first)
    to_local = _core.world(first).inverted()
    meshes = [(first.data, Matrix.Identity(4))] + [(o.data, to_local @ _core.world(o)) for o in others]
    bm = _concat(meshes)
    _replace_mesh(first, bm)
    if others:
        first[MERGED] = True
    for o in others:
        delete(o)
    return first


def _concat(meshes):
    """The meshes, each moved by its matrix, as one bmesh; tag attributes kept."""
    face_tags, edge_tags = [], []
    verts, faces, edge_keys = [], [], []
    for me, m in meshes:
        flip = m.to_3x3().determinant() < 0
        base = len(verts)
        verts += [m @ v.co for v in me.vertices]
        fl = {a.name: [d.value for d in a.data] for a in me.attributes
              if a.name.startswith(_core.FACE_TAG)}
        el = {a.name: [d.value for d in a.data] for a in me.attributes
              if a.name.startswith(_core.EDGE_TAG)}
        for i, p in enumerate(me.polygons):
            vs = [base + v for v in p.vertices]
            faces.append(vs[::-1] if flip else vs)
            face_tags.append({n[len(_core.FACE_TAG):] for n, vals in fl.items() if vals[i]})
        for i, e in enumerate(me.edges):
            tags = {n[len(_core.EDGE_TAG):] for n, vals in el.items() if vals[i]}
            if tags:
                edge_keys.append((frozenset((base + e.vertices[0], base + e.vertices[1])), tags))
    bm = bmesh.new()
    bv = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([bv[i] for i in f])
    bm.verts.index_update()
    bm.faces.index_update()
    bm.edges.index_update()
    lookup = {frozenset((e.verts[0].index, e.verts[1].index)): e.index for e in bm.edges}
    _core.add_tags(bm, dict(enumerate(face_tags)), {lookup[k]: t for k, t in edge_keys if k in lookup})
    return bm


def scale(ob, s):
    """About its origin.

    `ob`'s mesh scaled about its own origin: one factor, or (sx, sy, sz)."""
    _solids("scale", ob)
    sx, sy, sz = (s, s, s) if isinstance(s, (int, float)) else s
    ob.data.transform(Matrix.Diagonal((sx, sy, sz, 1)))
    if sx * sy * sz < 0:
        ob.data.flip_normals()
    return ob


# ---------------------------------------------------------------- upholstery

def dent(ob, points, r, depth):
    """Press hollows at points.

    Press the surface in at each of `points`: a round hollow of radius `r`, `depth` deep
    at its centre — how upholstery is buttoned."""
    _solids("dent", ob)
    centres = [Vector(p.origin if isinstance(p, Frame) else p) for p in points]
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for c in centres:
        _densify(bm, c, r)
    bm.normal_update()
    tree = KDTree(len(bm.verts))
    for v in bm.verts:
        tree.insert(v.co, v.index)
    tree.balance()
    moves = {}
    for c in centres:
        for co, index, dist in tree.find_range(c, r):
            moves[index] = moves.get(index, 0) + depth * (1 - (dist / r) ** 2) ** 2
    bm.verts.ensure_lookup_table()
    for index, push in moves.items():
        v = bm.verts[index]
        v.co -= v.normal * push
    bm.to_mesh(ob.data)
    bm.free()
    return ob


def _densify(bm, c, r, cuts=4):
    """Give the surface round `c` vertices to press: where none lies within r/`cuts` of
    it (a flat face meshed as one n-gon has none), cut the faces within `r` of it by
    planes r/`cuts` apart, both ways across the nearest face's normal."""
    if any((v.co - c).length < r / cuts for v in bm.verts):
        return
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    near = tree.find_nearest_range(c, r)
    if not near:
        return
    n = min(near, key=lambda hit: (hit[3], hit[2]))[1]
    u = n.orthogonal().normalized()
    for axis in (u, n.cross(u).normalized()):
        for k in range(-cuts + 1, cuts):
            faces = {bm.faces[hit[2]] for hit in near}
            geom = list(faces) + list({e for f in faces for e in f.edges}) + list({v for f in faces for v in f.verts})
            bmesh.ops.bisect_plane(bm, geom=geom, dist=_core.mm(WELD), plane_co=c + axis * (k * r / cuts), plane_no=axis)
            bm.faces.ensure_lookup_table()
            near = BVHTree.FromBMesh(bm).find_nearest_range(c, r)
