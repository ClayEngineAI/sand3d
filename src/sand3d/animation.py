"""Named, seekable motion. Times are seconds; values keep pose's units.

Tracks and clips are data, not Blender actions. Sampling evaluates from a captured
baseline; exporters use the same evaluator without changing the source scene.
"""
import bisect
import math
from collections.abc import Mapping
from dataclasses import dataclass

import bpy
from mathutils import Euler, Matrix, Vector

from . import _guard, joints as J


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return float(value)


def _bounce(t):
    if t < 1 / 2.75:
        return 7.5625 * t * t
    if t < 2 / 2.75:
        return 7.5625 * (t - 1.5 / 2.75) ** 2 + 0.75
    if t < 2.5 / 2.75:
        return 7.5625 * (t - 2.25 / 2.75) ** 2 + 0.9375
    return 7.5625 * (t - 2.625 / 2.75) ** 2 + 0.984375


def _ease_in(family, t):
    if family in ("quad", "cubic", "quart", "quint"):
        return t ** {"quad": 2, "cubic": 3, "quart": 4, "quint": 5}[family]
    if family == "sine":
        return 1 - math.cos(t * math.pi / 2)
    if family == "expo":
        return 2 ** (10 * t - 10)
    if family == "circ":
        return 1 - math.sqrt(max(0, 1 - t * t))
    if family == "back":
        return 2.70158 * t ** 3 - 1.70158 * t * t
    if family == "elastic":
        return -(2 ** (10 * t - 10)) * math.sin((10 * t - 10.75) * 2 * math.pi / 3)
    return 1 - _bounce(1 - t)


_FAMILIES = ("quad", "cubic", "quart", "quint", "sine", "expo", "circ", "back", "elastic", "bounce")
_EASES = ("linear", "step", "in", "out", "in_out") + tuple(
    f"{mode}_{family}" for family in _FAMILIES for mode in ("in", "out", "in_out"))


def _easing(ease, t):
    # Endpoints are exact, even for exponential/elastic and user functions.
    if t <= 0 or t >= 1:
        return float(t >= 1)
    if callable(ease):
        return _number(ease(t), "custom easing result")
    if ease == "linear":
        return t
    if ease == "step":
        return 0.0
    name = ease + "_quad" if ease in ("in", "out", "in_out") else ease
    mode, family = name.rsplit("_", 1)
    if mode == "in":
        return _ease_in(family, t)
    if mode == "out":
        return 1 - _ease_in(family, 1 - t)
    if family == "back":
        c = 1.70158 * 1.525
        u = 2 * t if t < 0.5 else 2 * t - 2
        return (u * u * ((c + 1) * u - c) / 2 if t < 0.5
                else (u * u * ((c + 1) * u + c) + 2) / 2)
    if family == "elastic":
        wave = math.sin((20 * t - 11.125) * 2 * math.pi / 4.5)
        return -(2 ** (20 * t - 10)) * wave / 2 if t < 0.5 else 2 ** (-20 * t + 10) * wave / 2 + 1
    if t < 0.5:
        return _ease_in(family, 2 * t) / 2
    return 1 - _ease_in(family, 2 - 2 * t) / 2


@dataclass(frozen=True)
class _Curve:
    keys: tuple
    ease: object

    def at(self, time):
        times = tuple(t for t, _ in self.keys)
        i = bisect.bisect_right(times, time)
        if i == 0:
            return self.keys[0][1]
        if i == len(times):
            return self.keys[-1][1]
        a, av = self.keys[i - 1]
        b, bv = self.keys[i]
        return av + (bv - av) * _easing(self.ease, (time - a) / (b - a))


@dataclass(frozen=True)
class _Track:
    ob: object
    joint: object
    curves: tuple


def _curve(values, ease):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("keyframes: keys are a nonempty mapping {seconds: value}")
    keys = tuple(sorted((_number(t, "keyframe time"), _number(v, "keyframe value"))
                        for t, v in values.items()))
    if keys[0][0] < 0:
        raise ValueError("keyframes: times must be nonnegative seconds")
    return _Curve(keys, ease)


def _joint_value(ob, index, value):
    j = J.joints_of(ob)[index]
    _number(value, f"animation {j['name']!r}")
    if j["kind"] == "free" or j.get("follows") is not None:
        raise ValueError(f"keyframes: {j['name']!r} is free or follows another joint: animate its leader")
    if j["range"] is not None:
        lo, hi = j["range"]
        if not lo - 1e-9 <= value <= hi + 1e-9:
            raise ValueError(f"animation: {j['name']!r} value {value:g} is outside its range ({lo:g}, {hi:g})")


def keyframes(ob, values=None, *, joint=None, x=None, y=None, z=None, rx=None, ry=None, rz=None, ease="linear"):
    """Values over seconds, for a joint or placement offsets.

    `keyframes(door, {0: 0, 1: 90})` uses pose's joint and units; `joint=` names
    one under an assembly. `keyframes(ball, z={0: 0, 1: 100})` offsets the captured
    placement in its parent frame. Rotation is XYZ degrees about its own origin.
    Easing: linear, step, in/out/in_out (quad), or in/out/in_out plus _quad, _cubic,
    _quart, _quint, _sine, _expo, _circ, _back, _elastic, _bounce; a callable takes
    progress in [0, 1] and returns eased progress. Endpoints are always exact.
    Back, elastic and custom easing may overshoot; joint limits still apply.
    Custom functions must be deterministic and have no side effects."""
    from .assembly import Handle
    ob = ob.root if isinstance(ob, Handle) else ob
    if not isinstance(ob, bpy.types.Object) or _guard.removed(ob):
        raise ValueError("keyframes: expected a live part or assembly")
    if not callable(ease) and (not isinstance(ease, str) or ease not in _EASES):
        raise ValueError(f"keyframes: unknown easing {ease!r}; expected one of {_EASES} or a callable")
    channels = {k: v for k, v in zip(("x", "y", "z", "rx", "ry", "rz"), (x, y, z, rx, ry, rz))
                if v is not None}
    if values is not None:
        if channels:
            raise ValueError("keyframes: use joint values or placement channels, not both")
        ob, index = J.target(ob, joint, verb="keyframes")
        curve = _curve(values, ease)
        for _, value in curve.keys:
            _joint_value(ob, index, value)
        return _Track(ob, index, (("value", curve),))
    if joint is not None or not channels:
        raise ValueError("keyframes: provide joint values or x/y/z/rx/ry/rz keyframes")
    if J.jointed(ob):
        raise ValueError("keyframes: a jointed part moves by its joint; animate its value")
    return _Track(ob, None, tuple((name, _curve(v, ease)) for name, v in channels.items()))


class _Clip:
    def __init__(self, name, tracks, members):
        self.name, self.tracks = name, tuple(tracks)
        self.members = tuple(members)
        self.bases = {ob.as_pointer(): Matrix(J.rest_basis(ob)) for ob in members}
        self.values = {(ob.as_pointer(), i): j.get("value", 0.0)
                       for ob in members for i, j in enumerate(J.joints_of(ob))}
        self.duration = max(t for track in tracks for _, curve in track.curves for t, _ in curve.keys)

    def evaluate(self, time):
        time = _number(time, "sample time")
        if any(_guard.removed(ob) for ob in self.members):
            raise ValueError("animation: a part in the clip was deleted; rebuild the clip")
        values, bases = dict(self.values), dict(self.bases)
        offsets = {}
        for track in self.tracks:
            key = track.ob.as_pointer()
            if track.joint is not None:
                value = track.curves[0][1].at(time)
                _joint_value(track.ob, track.joint, value)
                values[key, track.joint] = value
            else:
                offsets.setdefault(key, {}).update({name: curve.at(time) for name, curve in track.curves})
        for key, channels in offsets.items():
            translation = Vector(tuple(channels.get(c, 0) for c in ("x", "y", "z")))
            rotation = Euler(tuple(math.radians(channels.get(c, 0)) for c in ("rx", "ry", "rz")), "XYZ")
            base = bases[key]
            pivot = base.translation
            bases[key] = (Matrix.Translation(pivot + translation) @ rotation.to_matrix().to_4x4()
                          @ Matrix.Translation(-pivot) @ base)
        values = J._values(values)
        for value in values.values():
            _number(value, "animation joint value")
        return J.Kinematics(values=values, bases=bases), values, bases


def clip(name, *tracks):
    """Named coordinated motion, with a baseline captured now; does not move the scene.

    Tracks come from keyframes. Their first/last values hold outside their interval;
    duration is the latest key time. Each object channel occurs once. Finish building
    the model before making clips; rebuild clips after changing its structure."""
    from .urdf import Mechanism
    if not isinstance(name, str) or not name:
        raise ValueError("clip: name must be a nonempty string")
    if not tracks or any(not isinstance(t, _Track) for t in tracks):
        raise ValueError("clip: expected one or more keyframes tracks")
    seen, members, pending = set(), {}, [t.ob for t in tracks]
    for track in tracks:
        if _guard.removed(track.ob):
            raise ValueError("clip: a track's part was deleted")
        for channel, _ in track.curves:
            key = (track.ob.as_pointer(), track.joint, channel)
            if key in seen:
                raise ValueError("clip: two tracks control the same channel")
            seen.add(key)
    # Include the scene trees and kinematic dependencies of the animated parts.
    while pending:
        root = pending.pop()
        while root.parent is not None:
            root = root.parent
        if root.as_pointer() in members:
            continue
        for ob in Mechanism(root).members:
            members[ob.as_pointer()] = ob
            parent = J.parent_of(ob)
            if parent is not None and parent.as_pointer() not in members:
                pending.append(parent)
            for i, j in enumerate(J.joints_of(ob)):
                if j.get("follows") is not None:
                    leader = J.leader_of(ob, i)
                    if leader is not None and leader.as_pointer() not in members:
                        pending.append(leader)
        for ob in J.everything():
            if ob.as_pointer() not in members and any(
                    J.leader_of(ob, i) is not None and J.leader_of(ob, i).as_pointer() in members
                    for i, j in enumerate(J.joints_of(ob)) if j.get("follows") is not None):
                pending.append(ob)
    return _Clip(name, tracks, members.values())


def _check_movement(values):
    changed = {(ob.as_pointer(), i): values[ob.as_pointer(), i]
               for ob, js in J._scene_joints() for i, j in enumerate(js)
               if (ob.as_pointer(), i) in values and values[ob.as_pointer(), i] != j.get("value", 0.0)}
    J._left_behind(None, changed)


def sample(animation, time):
    """Show a clip at time in seconds, independent of all previous sample calls.

    Returns the clip. All channels are evaluated and validated before any placement
    changes. Sampling holds endpoints; it does not loop or advance a global clock."""
    if not isinstance(animation, _Clip):
        raise TypeError("sample: expected a clip")
    k, values, bases = animation.evaluate(time)
    _check_movement(values)
    placements = {}
    for ob in animation.members:
        key = ob.as_pointer()
        if J.jointed(ob):
            above = k.posed(ob.parent) if ob.parent is not None else Matrix.Identity(4)
            placements[key] = (above @ ob.matrix_parent_inverse).inverted() @ k.posed(ob)
        else:
            placements[key] = bases[key]
    for ob in animation.members:
        js = J.joints_of(ob)
        if js:
            for i, j in enumerate(js):
                j["value"] = values[ob.as_pointer(), i]
            J._write(ob, js)
        ob.matrix_basis = placements[ob.as_pointer()]
    return animation
