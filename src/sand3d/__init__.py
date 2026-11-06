"""sand3d — the vocabulary an LLM writes 3D assets in.

    import sand3d as s3d

One flat surface: every verb is `s3d.<verb>`. Millimetres, unless the program
sets another unit with `s3d.units`; `export` scales to metres. A program records
`s3d.__version__`, `s3d.fingerprint()` and `s3d.units()` with what it built.
"""
import functools
import hashlib
import pathlib
import sys

from . import _core, _guard

from .assembly import Handle, assembly, attach, empty_scene, export, group, units
from .edit import (chamfer, copy, cut, delete, dent, fillet, hole, intersect, join, merge, pocket, scale, seat, shell,
                   soften, split)
from .parts import gear, knurl, thread
from .frames import Frame, along, at, bbox, grid, local, ring, snap, transform
from .materials import material, resolve as _resolve
from .joints import joint, placed_again as _placed_again, placed_on as _placed_on, pose
from .animation import clip, keyframes, sample
from .profile import Path as _Path, Placed as _Placed, arc, circle, ellipse, mirror, offset, poly, polygon, rect, slot, spline, station, stroke, superellipse, tangent_arc, text, wrap
from .solids import body, box, cone, cylinder, dome, extrude, helix, loft, revolve, sphere, sweep, torus, tube, wedge, wire

# x-release-please-start-version
__version__ = "1.6.2"
# x-release-please-end

__all__ = [
    # profiles
    "rect", "circle", "ellipse", "slot", "polygon", "poly", "arc", "tangent_arc", "spline", "superellipse",
    "station", "text", "offset", "stroke", "wrap", "mirror",
    # solids, paths
    "extrude", "revolve", "sweep", "loft", "body", "wire", "helix",
    "box", "cylinder", "tube", "cone", "sphere", "wedge", "torus", "dome",
    # edits
    "cut", "join", "intersect", "split", "delete", "copy", "hole", "pocket", "merge", "scale",
    "fillet", "chamfer", "soften", "shell", "dent",
    # frames, placement
    "Frame", "bbox", "local", "at", "transform", "grid", "ring", "along", "snap",
    # parts
    "gear", "knurl", "thread",
    # materials
    "material",
    # articulation
    "joint", "pose", "seat",
    "clip", "keyframes", "sample",
    # assembly
    "group", "assembly", "attach", "Handle", "export", "empty_scene", "units", "fingerprint",
]


def _verb(fn):
    """`fn`, fixing the program's unit the moment it is called — even if it raises
    before measuring anything: the unit is set before the first verb — and checking what
    the program hands it (`_guard`)."""
    name = fn.__name__
    positional = fn.__code__.co_varnames[:fn.__code__.co_argcount]

    @functools.wraps(fn)
    def verb(*args, **kwargs):
        _core.settled = True
        line = sys._getframe(1).f_lineno
        _guard.calls.append((name, line, set(positional[:len(args)]) | set(kwargs)))
        try:
            _guard.begin(name, args, kwargs)
            result = fn(*args, **kwargs)
        finally:
            _guard.calls.pop()
        if _core.dressed and not _guard.calls:
            _resolve()                   # every mesh wears its nearest material, whatever the verb rebuilt
        if name in ("at", "transform", "station"):
            for placed in result if isinstance(result, list) else [result]:
                if isinstance(placed, (_Placed, _Path)):
                    _guard.watch(placed, line)
        if name in ("at", "extrude"):
            on = kwargs["on"] if "on" in kwargs else (args[positional.index("on")]
                                                     if len(args) > positional.index("on") else None)
            _placed(result, on)
            if name == "at" and on is None:
                _placed_again(result)
        if name == "transform":
            _placed_again(result)
        return result
    return verb


def _placed(result, on):
    """Remember, on what `at` or `extrude` placed on a frame read on a part, that part:
    `pose` refuses to leave it behind."""
    if on is None:
        return
    if isinstance(result, Handle):
        result = result.root
    if isinstance(result, list):
        if isinstance(on, (list, tuple)) and len(on) == len(result):
            for r, f in zip(result, on):
                _placed(r, f)
        return
    _placed_on(result, on)


for _name in __all__:
    if _name not in ("units", "empty_scene", "fingerprint") and not isinstance(globals()[_name], type):
        globals()[_name] = _verb(globals()[_name])
del _name


def fingerprint():
    """Digest of the source.

    A digest of the library's source: what a program run records beside `__version__`."""
    root = pathlib.Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]
