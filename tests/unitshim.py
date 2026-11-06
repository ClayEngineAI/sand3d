"""Run a program written in millimetres as if it had been written in another unit.

`install(k)` wraps every public callable of `sand3d` so that the lengths a program
passes in are multiplied by `k` (0.001: millimetres become metres) and the few
lengths it reads back are divided by it. The program's own arithmetic stays in
millimetres; the library only ever sees the scaled values — exactly what the same
program rewritten with every length × `k` would hand it. The invariance test
builds the golden corpus once plain and once through this shim under the matching
`s3d.units`, and compares.

`LENGTHS` says, for every parameter of the public surface, whether it is a length:
`L` a length (a number, or None), `P` numbers nested in points or lists (objects
inside pass through), `csk` a countersink — a diameter, or (diameter, angle) —, `sv` a number,
or (position, value) points whose positions alone are lengths (`body`'s `n`), `-`
anything else. `test_units` holds it complete against the surface `apicompat` reads,
so a new verb cannot slip past the invariance test unclassified.

Lengths read back: `Frame.point`, `Frame.origin`, `Frame.matrix` and a placed profile's `matrix` (their
translation), `Path.points`, a loop's `pts`, `Profile.bounds` and `Profile.mesh` (its
points). `+` and `-` on profiles run as library calls. Library objects — solids, profiles, placed profiles,
paths, frames — stay in the library's unit; only their reads are converted. A
call's own work runs with the shim off, so the library's internal use of `Frame`,
`Path.points` or `bounds` is untouched.
"""
import functools
import inspect

import bpy  # noqa: F401 — before mathutils, which the bpy module provides
from mathutils import Vector

LENGTHS = {
    # profiles
    "rect": {"w": "L", "d": "L", "r": "L", "chamfer": "L"},
    "circle": {"d": "L"},
    "ellipse": {"w": "L", "d": "L"},
    "slot": {"l": "L", "d": "L"},
    "polygon": {"n": "-", "d": "L"},
    "poly": {"pts": "P", "fillet": "P", "chamfer": "P"},
    "arc": {"center": "P", "r": "P", "start": "-", "end": "-", "through": "P", "to": "P"},
    "tangent_arc": {"to": "P"},
    "spline": {"pts": "P", "closed": "-", "tangents": "-", "loop": "-"},
    "superellipse": {"w": "L", "d": "L", "n": "-", "taper": "-"},
    "station": {"z": "L", "w": "L", "d": "L", "x": "L", "y": "L", "n": "-", "taper": "-"},
    "text": {"s": "-", "size": "L", "font": "-"},
    "offset": {"p": "-", "t": "L"},
    "stroke": {"path": "P", "t": "L"},
    "wrap": {"profiles": "P"},
    "mirror": {"p": "-", "flip": "-"},
    # solids, paths
    "extrude": {"name": "-", "profile": "P", "h": "L", "taper": "-", "twist": "-", "both": "-", "h2": "L",
                "on": "-", "until": "-"},
    "revolve": {"name": "-", "profile": "P", "angle": "-"},
    "sweep": {"name": "-", "profile": "P", "path": "P", "scale": "-", "twist": "-", "frenet": "-"},
    "loft": {"name": "-", "sections": "P", "follow": "-", "loop": "-"},
    "body": {"name": "-", "plan": "P", "top": "P", "bottom": "P", "n": "sv"},
    "wire": {"name": "-", "path": "P", "d": "L", "closed": "-", "fillet": "P"},
    "helix": {"d": "L", "pitch": "L", "h": "L", "left": "-"},
    "box": {"name": "-", "w": "L", "d": "L", "h": "L"},
    "cylinder": {"name": "-", "d": "L", "h": "L"},
    "tube": {"name": "-", "d": "L", "d_in": "L", "h": "L"},
    "cone": {"name": "-", "d": "L", "h": "L", "d2": "L"},
    "sphere": {"name": "-", "d": "L"},
    "wedge": {"name": "-", "w": "L", "d": "L", "h": "L", "top": "L"},
    "torus": {"name": "-", "d": "L", "d_wire": "L"},
    "dome": {"name": "-", "d": "L", "h": "L"},
    # edits
    "cut": {"ob": "-", "tools": "-"},
    "join": {"ob": "-", "others": "-"},
    "intersect": {"ob": "-", "tool": "-"},
    "split": {"ob": "-", "frame": "-", "keep": "-"},
    "delete": {"ob": "-"},
    "copy": {"ob": "-"},
    "hole": {"ob": "-", "d": "L", "on": "P", "depth": "L", "cbore": "P", "csk": "csk"},
    "pocket": {"ob": "-", "profile": "P", "depth": "L", "on": "-"},
    "merge": {"obs": "-"},
    "scale": {"ob": "-", "s": "-"},
    "fillet": {"ob": "-", "r": "L", "edges": "-"},
    "chamfer": {"ob": "-", "d": "L", "edges": "-", "d2": "L", "angle": "-"},
    "soften": {"ob": "-", "r": "L", "angle": "-"},
    "shell": {"ob": "-", "t": "L", "open": "-"},
    "dent": {"ob": "-", "points": "P", "r": "L", "depth": "L"},
    # frames, placement
    "Frame": {"origin": "P", "normal": "-", "up": "-"},
    "Frame.offset": {"d": "L"},
    "Frame.turn": {"rx": "-", "ry": "-", "rz": "-"},
    "bbox": {"ob": "-", "which": "-"},
    "local": {"ob": "-", "x": "L", "y": "L", "z": "L", "rx": "-", "ry": "-", "rz": "-"},
    "Frame.move": {"x": "L", "y": "L", "z": "L"},
    "Frame.point": {"x": "L", "y": "L", "z": "L"},
    "Frame.normal": {},
    "Frame.matrix": {},
    "at": {"ob": "-", "x": "L", "y": "L", "z": "L", "rx": "-", "ry": "-", "rz": "-", "on": "-", "anchor": "-"},
    "transform": {"ob": "-", "x": "L", "y": "L", "z": "L", "rx": "-", "ry": "-", "rz": "-", "mirror": "-"},
    "grid": {"nx": "-", "ny": "-", "px": "L", "py": "L", "stagger": "-", "on": "-"},
    "ring": {"n": "-", "r": "L", "start": "-", "angle": "-", "on": "-"},
    "along": {"path": "P", "pitch": "L", "count": "-"},
    "snap": {"targets": "-", "points": "P", "direction": "-"},
    "ob.face": {"which": "-"},
    "ob.edge": {"which": "-"},
    # parts
    "gear": {"name": "-", "teeth": "-", "module": "L", "width": "L", "bore": "L", "angle": "-"},
    "knurl": {"name": "-", "d": "L", "h": "L", "pitch": "L", "straight": "-"},
    "thread": {"name": "-", "d": "L", "length": "L", "pitch": "L", "left": "-"},
    # articulation: a range or a value is a length on a prismatic joint only (`_joint`, `_pose`)
    "joint": {"name": "-", "child": "-", "on": "-", "kind": "-", "axis": "-", "range": "-", "follows": "-",
              "ratio": "-"},
    "pose": {"ob": "-", "value": "-", "joint": "-"},
    "seat": {"host": "-", "part": "-"},
    "keyframes": {"ob": "-", "values": "-", "joint": "-", "x": "-", "y": "-", "z": "-",
                  "rx": "-", "ry": "-", "rz": "-", "ease": "-"},
    "clip": {"name": "-", "tracks": "-"},
    "sample": {"animation": "-", "time": "-"},
    # assembly
    "material": {"ob": "-", "kind": "-", "color": "-", "roughness": "-", "metallic": "-", "opacity": "-",
                 "emission": "-", "repeat": "-"},
    "group": {"name": "-", "children": "-"},
    "assembly": {"name": "-", "children": "-", "anchors": "-"},
    "attach": {"handle": "-", "name": "-", "children": "-"},
    "Handle": {"root": "-", "anchors": "-"},
    "export": {"root": "-", "path": "-", "animations": "-", "fps": "-"},
    "empty_scene": {},
    "fingerprint": {},
    "units": {"name": "-"},
}

#: Calls of the library under way: the shim converts only at depth 0, the program's side.
_depth = 0


def scaled(value, k):
    """Every number in `value` × k, through tuples, lists and vectors; anything else as is."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value * k
    if isinstance(value, Vector):
        return Vector([c * k for c in value])
    if isinstance(value, (tuple, list)):
        return type(value)(scaled(v, k) for v in value)
    return value


def _convert(kind, value, k):
    if kind == "L":
        return None if value is None else value * k
    if kind == "P":
        return scaled(value, k)
    if kind == "sv":          # a number, or (position, value) points: the positions alone are lengths
        if isinstance(value, (int, float)) or value is None:
            return value
        return type(value)((q[0] * k, *q[1:]) for q in value) if isinstance(value, (list, tuple)) else value
    if kind == "csk":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value * k
        return value if not value else (value[0] * k, value[1])
    return value


def _kind_of(follows, child):
    """The kind of the joint `follows` names, as `joint` resolves it."""
    import sand3d
    from sand3d.joints import _resolve_follows, joints_of
    from sand3d.assembly import Handle
    ob = child.root if isinstance(child, Handle) else child
    leader, index = _resolve_follows(follows, ob, "joint")
    return joints_of(leader)[index]["kind"]


def _joint(arguments, k):
    """`joint`'s range is a length on a prismatic joint; its ratio is per unit of the
    followed joint, so a length per degree, a degree per length, or neither."""
    kind = arguments.get("kind", "revolute")
    if kind == "prismatic" and arguments.get("range") is not None:
        arguments["range"] = scaled(tuple(arguments["range"]), k)
    if arguments.get("follows") is not None:
        lead = _kind_of(arguments["follows"], arguments["child"])
        mine, theirs = kind == "prismatic", lead == "prismatic"
        if mine != theirs:
            arguments["ratio"] = arguments.get("ratio", 1) * (k if mine else 1 / k)


def _pose(arguments, k):
    """`pose`'s value is a length on a prismatic joint."""
    from sand3d.joints import joints_of, target
    child, index = target(arguments["ob"], arguments.get("joint"))
    if joints_of(child)[index]["kind"] == "prismatic":
        arguments["value"] = arguments["value"] * k


#: Verbs whose lengths depend on their other arguments.
def _keyframes(arguments, k):
    from sand3d.joints import joints_of, target
    for name in ("x", "y", "z"):
        if arguments.get(name) is not None:
            arguments[name] = {t: value * k for t, value in arguments[name].items()}
    if arguments.get("values") is not None:
        child, index = target(arguments["ob"], arguments.get("joint"))
        if joints_of(child)[index]["kind"] == "prismatic":
            arguments["values"] = {t: value * k for t, value in arguments["values"].items()}


FIXES = {"joint": _joint, "pose": _pose, "keyframes": _keyframes}


def _wrap(fn, spec, k, out=None, fix=None):
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        global _depth
        if _depth == 0:
            bound = signature.bind(*args, **kwargs)
            for name, value in list(bound.arguments.items()):
                kind = spec.get(name, "-")
                param = signature.parameters[name]
                if param.kind is param.VAR_POSITIONAL:
                    bound.arguments[name] = tuple(_convert(kind, v, k) for v in value)
                elif param.kind is not param.VAR_KEYWORD:
                    bound.arguments[name] = _convert(kind, value, k)
            if fix is not None:
                fix(bound.arguments, k)
            args, kwargs = bound.args, bound.kwargs
        _depth += 1
        try:
            result = fn(*args, **kwargs)
        finally:
            _depth -= 1
        return out(result) if out is not None and _depth == 0 else result
    return call


def install(k):
    """Wrap `sand3d`'s public surface in place so a millimetre program runs × `k`."""
    import bpy
    import sand3d
    from sand3d.frames import Frame
    from sand3d.profile import Loop, Path, Placed, Profile

    for name in sand3d.__all__:
        obj = getattr(sand3d, name)
        if name in LENGTHS and inspect.isfunction(obj):
            setattr(sand3d, name, _wrap(obj, LENGTHS[name], k, fix=FIXES.get(name)))
    Frame.__init__ = _wrap(Frame.__init__, LENGTHS["Frame"], k)
    for method in ("offset", "turn", "move"):
        setattr(Frame, method, _wrap(getattr(Frame, method), LENGTHS[f"Frame.{method}"], k))
    Frame.point = _wrap(Frame.point, LENGTHS["Frame.point"], k, out=lambda v: scaled(v, 1 / k))
    bpy.types.Object.face = _wrap(bpy.types.Object.face, LENGTHS["ob.face"], k)
    Profile.bounds = _wrap(Profile.bounds, {}, k, out=lambda b: scaled(b, 1 / k))
    Profile.mesh = _wrap(Profile.mesh, {}, k, out=lambda m: (scaled(m[0], 1 / k), *m[1:]))
    for cls in (Profile, Placed):
        for op in ("__add__", "__sub__"):
            setattr(cls, op, _wrap(getattr(cls, op), {}, k))
    Path.points = _read_back("points", lambda v: scaled(v, 1 / k))
    slot = Loop.pts
    Loop.pts = property(lambda self: slot.__get__(self) if _depth else scaled(slot.__get__(self), 1 / k),
                        slot.__set__)
    Placed.matrix = _read_back("matrix", lambda m: _moved(m, 1 / k))
    Frame.origin = _read_back("origin", lambda v: scaled(v, 1 / k))
    Frame.matrix = property(_wrap(Frame.matrix.fget, {}, k, out=lambda m: _moved(m, 1 / k)))


def _read_back(attr, convert):
    """A property over a plain attribute: as stored inside the library, converted outside."""
    def get(self):
        return self.__dict__[attr] if _depth else convert(self.__dict__[attr])

    def put(self, value):
        self.__dict__[attr] = value
    return property(get, put)


def _moved(m, f):
    """A copy of the 4×4 `m`, its translation × f."""
    m = m.copy()
    for row in range(3):
        m[row][3] *= f
    return m
