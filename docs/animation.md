# Animation

`joint` defines how a part moves. `pose` places it now. `keyframes` gives its values
over time; `clip` groups tracks into named motion. All times are seconds, angles
are degrees, and lengths use the program's unit.

```python
opening = s3d.clip(
    "open",
    s3d.keyframes(door, {0: 0, 1: 110}, ease="in_out_sine"),
    s3d.keyframes(drawer, {0: 0, 0.5: 0, 1.5: 180}),
)
s3d.sample(opening, 0.75)
s3d.export(machine, "machine.gltf", animations=[opening])
s3d.export(machine, "machine.glb", animations=[opening])
```

`.gltf` writes JSON and a companion `.bin` in the same directory. Keep both files
together. `.glb` packs everything into one file. Both contain standard glTF named
animations that viewers and engines can select and play. Playback controls,
including loop and speed, belong to the viewer. Export does not play a clip or
change the source model's pose.

The [complete runnable example](examples/animation.py) exports four clips: `open`,
`spin`, `bounce`, and `custom`.

The [washing-machine example](examples/washing_machine_animation.py) exports an
eight-second `washer_cycle`: the door and drawer open and close, then the drum
and dial turn. It exports both GLB and separate glTF. In Clay, load the asset,
clone its scene, and play its clip through `AnimationMixer`; Clay advances the
mixer automatically. Joint metadata alone does not create playback.

## Joints and transforms

Joint tracks follow `pose`'s addressing rules. Use the part directly when it has
one joint, or supply `joint=` to select a joint on a part or under an assembly:

```python
s3d.keyframes(machine, {0: 0, 1: 110}, joint="door_hinge")
s3d.keyframes(screw, {0: 0, 2: 720}, joint="screw_turn")
```

Followers move automatically, including a screw's coupled translation. Keyframe
the leader; a follower and a free joint do not accept scalar tracks. Joint limits
are checked on keys and on every evaluated value, including easing overshoot.

For unjointed parts and assembly roots, use `x`, `y`, `z`, `rx`, `ry`, and `rz`:

```python
bounce = s3d.clip("bounce", s3d.keyframes(ball, z={0: 0, 0.4: 100, 0.8: 0}))
turn = s3d.clip("turn", s3d.keyframes(machine, rz={0: 0, 4: 360}))
```

Translation channels are offsets from the placement captured by `clip`, along the
parent frame's axes. Rotations are XYZ Euler offsets in degrees, about the object's
captured origin, in the parent frame; they precede its original orientation.
Jointed parts move through their joint channels instead. Separate tracks may
control separate channels of one object; two tracks may not control the same channel.

## Easing

`ease="linear"` is the default. `ease="step"` holds each key until the next one.
Each of these ten families supports `in_`, `out_`, and `in_out_`:

| Family | Names |
|---|---|
| Quadratic | `in_quad`, `out_quad`, `in_out_quad` |
| Cubic | `in_cubic`, `out_cubic`, `in_out_cubic` |
| Quartic | `in_quart`, `out_quart`, `in_out_quart` |
| Quintic | `in_quint`, `out_quint`, `in_out_quint` |
| Sine | `in_sine`, `out_sine`, `in_out_sine` |
| Exponential | `in_expo`, `out_expo`, `in_out_expo` |
| Circular | `in_circ`, `out_circ`, `in_out_circ` |
| Back | `in_back`, `out_back`, `in_out_back` |
| Elastic | `in_elastic`, `out_elastic`, `in_out_elastic` |
| Bounce | `in_bounce`, `out_bounce`, `in_out_bounce` |

`in`, `out`, and `in_out` are short aliases for the quadratic variants. Easing
applies separately to each interval in the track. Back and elastic can overshoot;
leave room inside a joint's limits when using them.

A custom function takes normalized interval progress and returns eased progress:

```python
s3d.keyframes(drawer, {0: 0, 1: 100}, ease=lambda t: t * t * (3 - 2 * t))

def slow_start(t):
    return t ** 4

s3d.keyframes(door, {0: 0, 2: 90}, ease=slow_start)
```

The function must return a finite number and be deterministic, with no side
effects. It may overshoot. Exact endpoints are enforced by the library: progress
0 gives the first key, and progress 1 gives the next key. Python code is evaluated
while sampling/baking; exported files contain values, not executable functions.

## Seeking, baseline, and export precision

Finish building the model before creating its clips. A clip captures the baseline
of the affected scene trees and kinematic dependencies, including unkeyed channels.
Creating tracks or clips does not move anything. `sample(clip, seconds)` applies an
absolute state, so seeking backward or sampling the same time twice does not
accumulate motion. Times outside a track's range hold its first or last value.
`clip.duration` is its latest key time. Rebuild clips after changing model structure.

```python
s3d.export(machine, "machine.gltf", animations=[opening, spinning], fps=120)
```

Export bakes at 60 samples per second by default, includes key times, and adds
angular subdivisions so full revolutions survive quaternion export. Increasing
`fps` improves the approximation of easing, especially fast custom oscillations.
Arbitrary custom functions cannot be reconstructed exactly from finite samples.
Pure step channels use glTF `STEP`. When translation or rotation components mix
step and continuous easing, discontinuities are bracketed by adjacent float32
timestamps and the vector channel uses `LINEAR`.

Each clip exports all baseline transforms, so switching between clips does not
inherit unkeyed values from the previous clip. Clip names must be unique in an
export, and all animated targets must belong to the exported asset. Source meshes,
poses, names and Blender actions are preserved. Animated transforms must be
representable as translation, rotation and scale; sheared transforms are rejected.

Without `animations=`, export retains its existing static behavior. Animations
are supported for `.gltf` and `.glb`; requesting them for other formats raises an
error. This API animates rigid parts and transforms, not skinning or morph targets.
