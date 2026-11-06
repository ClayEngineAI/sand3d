# Writing an assembly program

Eight rules for any program that builds a multi-part object in sand3d, by hand or by model. They were
drawn from a review of generated asset and articulated programs, and have not been tested as a prompt.
Every example is a real program, run under 1.4.0, when sand3d was still named sky3d. A fault is followed by how it should have been
written: `# ←` comments are added to the original, and rewrites keep the original's shape.

1. **The head is the table of contents.** First function, named after the object, no geometry.
2. **One function per distinct part.** Built about its own origin; returns one object.
3. **Name each dimension once, derive the rest.** Target 15 numbers per part, ceiling 40.
4. **Place each part once, in its parent's frame.**
5. **Repeat with frames, mirror with a copy.**
6. **Say each shape with the verb that names it.** No `bpy`, no `bmesh`.
7. **Dress the parts when appearance matters.** Optional on shape-only data.
8. **One frame per joint.**

## What a program looks like

A short head, then one function per part, each with its dimensions in its signature.

```python
import sand3d as s3d


def floor_lamp():
    base = s3d.at(round_base(), z=-801)  # ← better: stand the stem on the base, derived from it
    stem = s3d.at(slender_stem(), z=-774)  # ← better: 27 above the base, not a second measured height
    return s3d.assembly("floor_lamp", base, stem)


def round_base(d=202, h=30, edge_r=6, collar_d=68, collar_h=9):
    base = s3d.cylinder("base", d=d, h=h)
    base = s3d.fillet(base, r=edge_r)  # ← accidental relation: the taper has nothing to do with the fillet
    collar = s3d.at(s3d.cone("collar", d=collar_d, d2=collar_d-edge_r*2, h=collar_h), z=h-1)
    base = s3d.join(base, collar)
    s3d.material(base, "metal", metallic=0, color="#606060")
    return s3d.group("round_base", base)


def slender_stem(d=29, h=1419, shade_z=1300):
    stem = s3d.cylinder("stem", d=d, h=h)
    s3d.material(stem, "metal", metallic=0, color="#595959")
    shade = s3d.at(open_shade(), z=shade_z)
    return s3d.group("slender_stem", stem, shade)


def open_shade(d=330.3, top_d=320, h=273, bottom_wall=6, top_wall=13):
    wall = s3d.revolve("shade_wall", s3d.poly([(d/2,0),(top_d/2,h),(top_d/2-top_wall,h),(d/2-bottom_wall,0)]))
    s3d.material(wall, "fabric", color="#cfc5b6")
    return s3d.group("open_shade", wall)  # ← one solid at its origin: return wall would do
```

## 1. The head is the table of contents

The first function is named after the object. It builds no geometry: it calls the parts, places them
and returns `s3d.assembly`. Runners call the first function, so the head comes first. Nothing at module
level but the import. A detail that should stay deletable can hang on the assembly with `attach`, as in
[examples/chair.py](examples/chair.py); none of the 1 089 programs needed it.

**Right**: a picnic table.

```python
def picnic_table():
    top = s3d.at(tabletop(), z=788)
    frame = s3d.at(end_frame(), x=659, z=3)
    twin = s3d.transform(s3d.copy(frame), mirror="x")
    seats = s3d.at(benches(), z=447)
    return s3d.assembly("picnic_table", top, frame, twin, seats)
```

**Wrong**: a turnstile.

```python
def industrial_turnstile_gate():
    base = s3d.extrude("floor_plate", s3d.rect(2700, 2700), 60, both=True)
    frame = s3d.group("frame", s3d.at(base, z=30))
    rotor = s3d.group(
        "rotor", s3d.at(s3d.extrude("spindle_shaft", s3d.circle(96), 760, both=True), z=350)
    )
    f = s3d.local(frame, z=800)
    rotor = s3d.at(rotor, on=f)
    s3d.joint("frame_to_rotor", rotor, f, kind="revolute", axis="+Z", range=(-120, 120))
```

The head is building a floor plate and a shaft. A reader must go through solids to find the plan, and no part can be reused or removed.

**How it should read**: one line per part in the head. Same shape (F 1.000 against the original).

```python
def industrial_turnstile_gate():
    frame = frame_part()
    f = s3d.local(frame, z=800)
    rotor = s3d.at(rotor_part(), on=f)
    s3d.joint("frame_to_rotor", rotor, f, kind="revolute", axis="+Z", range=(-120, 120))
    p = s3d.local(frame, x=535, y=-355, z=1345)
    pawl = s3d.at(pawl_part(), on=p)
    s3d.joint("frame_to_lockout", pawl, p, kind="revolute", axis="normal", range=(0, 46))
    return s3d.assembly("industrial_turnstile_gate", frame, rotor, pawl)
...
def frame_part(w=2700, t=60):
    base = s3d.extrude("floor_plate", s3d.rect(w, w), t, both=True)
    return s3d.group("frame", s3d.at(base, z=t / 2))
...
def rotor_part(d=96, h=760, z=350):
    shaft = s3d.extrude("spindle_shaft", s3d.circle(d), h, both=True)
    return s3d.group("rotor", s3d.at(shaft, z=z))
...
def pawl_part(length=305, w=20, h=40, x=-182.5):
    bar = s3d.extrude("pawl_bar", s3d.rect(length, w), h, both=True)
    return s3d.group("lockout_pawl", s3d.at(bar, x=x))
```

## 2. One function per distinct part

A part builds about its own origin, takes what differs as arguments, and returns one object. Which one
depends on how its pieces hold together:

| the pieces | the verb | what you get |
|---|---|---|
| fused for good | `join` | one solid, one material: the first object's |
| separate, moving together | `group` | each piece keeps its mesh and its material |
| moving against each other | `joint` | parts hung on frames, in the head (rule 8) |

`assembly` is the top-level `group`, plus the anchors it publishes. One solid needs no group: return the
solid. The trap: a lone solid placed inside its function loses that placement at the next `at` (`z=50`
then `z=100` lands at 100, and in a group at 150). Build a lone solid at its origin.

**Right**: one function, three levers.

```python
    lever1 = lever_part(150, "lever1", "#cc3333")
    lever2 = lever_part(120, "lever2", "#33cc33")
    lever3 = lever_part(100, "lever3", "#3333cc")
```

**Wrong**: a seesaw.

```python
    end = s3d.at(end_crossrail(), y=-1143, z=134)
    other = s3d.at(opposite_crossrail(), y=1143, z=134)
...
def end_crossrail(width=325, d=31, height=182, stem_d=29):
    solid = s3d.wire('crossrail', path=[(-width/2+d/2,0,height-d/2),(width/2-d/2,0,height-d/2)], d=d)
    stem = s3d.cylinder('upright', d=stem_d, h=height-d/2)
    s3d.join(solid, stem)
    s3d.material(solid, 'metal', metallic=0, color='#4c71ae')
    return s3d.group('end_crossrail', solid)
...
def opposite_crossrail(width=325, d=31, height=182, stem_d=29):  # ← the same six lines as end_crossrail
    solid = s3d.wire('crossrail', path=[(-width/2+d/2,0,height-d/2),(width/2-d/2,0,height-d/2)], d=d)
    stem = s3d.cylinder('upright', d=stem_d, h=height-d/2)
    s3d.join(solid, stem)
    s3d.material(solid, 'metal', metallic=0, color='#4c71ae')
    return s3d.group('opposite_crossrail', solid)
```

Nothing here crashes. These are one function under two names: a change has to be made twice, and a reader has to compare them line by line to learn they are the same.

**How it should read**: one function, the other side mirrored (rule 5). Same shape (F 1.000 against the original).

```python
    end = s3d.at(end_crossrail(), y=-1143, z=134)
    other = s3d.transform(s3d.copy(end), mirror="y")
...
def end_crossrail(width=325, d=31, height=182, stem_d=29):
    solid = s3d.wire('crossrail', path=[(-width/2+d/2,0,height-d/2),(width/2-d/2,0,height-d/2)], d=d)
    stem = s3d.cylinder('upright', d=stem_d, h=height-d/2)
    s3d.join(solid, stem)
    s3d.material(solid, 'metal', metallic=0, color='#4c71ae')
    return s3d.group('end_crossrail', solid)
```

## 3. Name each dimension once, derive the rest

Main dimensions are keyword arguments in millimetres. A value measured on the target appears once. A
value that follows from another is written as an expression of it.

- **Count the numbers.** Target 15 per part, ceiling 40. A number is any numeric literal, keyword defaults
  included, except 0, 0.5, 1, 2, 45, 90, 180, 270 and 360. Clean programs have a median of 15, programs
  that fought the library 53.
- **Read every argument.** 31.0 % of the programs declare a dimension they never read.
- **No `if x == 28`, no unexplained fraction.** They hold only while the numbers around them stay put.
- **Expect to lose a little fidelity.** Fewer numbers means a coarser match to a target. It is a trade.
- **Copying a value from a source is allowed, once.** A joint origin copied from a URDF is a measurement.

**Right**: each point an expression of the arguments.

```python
    profile = s3d.poly([(d/2,0), (d/2,shoulder), (mouth_d/2,h), (mouth_d/2-wall,h), (d/2-wall,shoulder), (d/2-wall,0)])
    neck = s3d.revolve("neck", profile)
    s3d.material(neck, kind="metal", roughness=0.3, color="#888888")
```

**Wrong**: a freezer cabinet.

```python
def cabinet_part(width=1500, depth=800, height=940):
    shell = s3d.extrude("shell", s3d.rect(width, depth), height, both=True)
    shell = s3d.at(shell, z=height / 2)
    inner = s3d.at(
        s3d.extrude("inner_tool", s3d.rect(width - 100, depth - 100), height - 90, both=True), z=475
    )
    s3d.cut(shell, inner)
    tool1 = s3d.at(s3d.extrude("slot_tool1", s3d.rect(1440, 720), 20, both=True), z=910)
    s3d.cut(shell, tool1)
    tool2 = s3d.at(s3d.extrude("slot_tool2", s3d.rect(1440, 740), 20, both=True), z=930)
    s3d.cut(shell, tool2)
```

It runs and matches its target. But 475, 1440, 720, 740, 910 and 930 all follow from `width`, `depth` and `height`: change the height and the cabinet breaks.

**How it should read**: three named dimensions, the rest derived. Same shape (F 1.000 against the original).

```python
def cabinet_part(width=1500, depth=800, height=940, wall=50, lip=30, slot=20):
    shell = s3d.extrude("shell", s3d.rect(width, depth), height, both=True)
    shell = s3d.at(shell, z=height / 2)
    well = height - 2 * slot - wall
    inner = s3d.at(
        s3d.extrude("inner_tool", s3d.rect(width - 2 * wall, depth - 2 * wall), well, both=True),
        z=wall + well / 2,
    )
    s3d.cut(shell, inner)
    open_w, open_d = width - 2 * lip, depth - 2 * lip
    tool1 = s3d.at(
        s3d.extrude("slot_tool1", s3d.rect(open_w, open_d - slot), slot, both=True),
        z=height - 1.5 * slot,
    )
    s3d.cut(shell, tool1)
    tool2 = s3d.at(
        s3d.extrude("slot_tool2", s3d.rect(open_w, open_d), slot, both=True),
        z=height - slot / 2,
    )
    s3d.cut(shell, tool2)
```

## 4. Place each part once, in its parent's frame

The head places the top-level parts with `at`, in the object's frame. Each part places its own children.
A placement that follows from a dimension is written from it. The fault is a part built at its world
position, then shifted back.

- Prefer a frame when the relation is the point: `on=ob.face("+Z")`, `bbox`, `local`.
- Every solid starts at z = 0, centred in x and y; `extrude(both=True)` centres it on its plane.
- `at` sets the values it is given and keeps the others. `transform` moves by.

**Right**: the bowl sits at the stem’s height.

```python
def stem(h=70, waist=6.5, overlap=0.5):
    solid = s3d.revolve("stem", s3d.poly([(0,0),(14,0),(7.5,14),(waist,34),(9,57),(16,h+overlap),(0,h+overlap)]))
    s3d.material(solid, "plastic", color="#737373")
    bowl_placed = s3d.at(bowl(), z=h)
    return s3d.group("stem_group", solid, bowl_placed)
```

**Wrong**: glasses.

```python
    arm = s3d.at(temple_arm(), x=64.5, y=-14.6, z=37.5)
...
def temple_arm(w=3, length=119.2, base=37.5, origin_y=-14.6, origin_x=64.5):
    path = s3d.spline([(62-origin_x,-72-origin_y,55-base),(63-origin_x,-45-origin_y,54-base),(66-origin_x,0-origin_y,54-base),(66-origin_x,20-origin_y,53-base),(65-origin_x,32-origin_y,46-base),(65-origin_x,44-origin_y,40-base)])
```

The head places the arm at (64.5, -14.6, 37.5), and the part takes those same numbers off every point. Each coordinate is written in the world, then undone.

**How it should read**: the spline in the arm’s own frame. Same shape (F 1.000 against the original).

```python
    arm = s3d.at(temple_arm(), x=64.5, y=-14.6, z=37.5)
...
def temple_arm(w=3, d=5):
    path = s3d.spline([(-2.5,-57.4,17.5),(-1.5,-30.4,16.5),(1.5,14.6,16.5),(1.5,34.6,15.5),(0.5,46.6,8.5),(0.5,58.6,2.5)])
```

## 5. Repeat with frames, mirror with a copy

Copies go through `on=s3d.grid(…)`, `ring(…)` or `along(…)`. The other side of a pair is
`s3d.transform(s3d.copy(placed), mirror="x")`. `at` is for a place, `transform` for a mirror image. Never
cos and sin in a loop.

**Right**: a ring of rods.

```python
    rods = s3d.at(s3d.cylinder("rod", d=rod_d, h=h), on=s3d.ring(n=count, r=radius))
```

**Right**: two mirrored pairs.

```python
    end = s3d.at(end_panel(), x=end_x, z=0)
    twin = s3d.transform(s3d.copy(end), mirror="x")
...
    feet = s3d.at(corner_feet(), x=foot_x, z=foot_z)
    other_feet = s3d.transform(s3d.copy(feet), mirror="x")
```

**Wrong**: a bunk frame.

```python
    for y in (-233, -140, -47):
        solids.append(s3d.at(s3d.box("return_post", w=rail, d=rail, h=h-rail), x=w/2-rail/2, y=y))
```

Three posts at an even pitch of 93. A list of three positions hides that they are evenly spaced.

**How it should read**: the pitch named, the grid says the rest. Same shape (F 1.000 against the original).

```python
    solids += s3d.at(s3d.box("return_post", w=rail, d=rail, h=h-rail), on=s3d.grid(nx=1, ny=3, py=post_pitch), x=w/2-rail/2, y=-140)
```

## 6. Say each shape with the verb that names it

A shape sand3d says in one verb is not written as a table of points. `bpy` and `bmesh` are forbidden: if
sand3d cannot say a shape, open an issue for the missing verb. Free forms still resist: until sand3d has
their verbs, a table may match a target better.

| the shape | the verb |
|---|---|
| round about an axis | `revolve(name, profile)` |
| hollow, open on one side | `shell(ob, t=…, open="top")` |
| a bore, a bolt circle | `hole(ob, d=…, on=…)` |
| a bent tube | `wire(name, path, d, fillet=…)` |
| a section that changes along an axis | `loft(name, station(z, w, d), …)` |
| a long body from its views | `body(name, plan, top, bottom)` |
| a band along a path | `stroke(path, t)` |

`station` is the workhorse. It is a section at height `z`: width, depth, an offset `x`, `y`, and `n` from
ellipse (2) to box (4). A few stations in a `loft` say a taper, a bulge or a drift.

**Right**: a nail, three stations and every value named.

```python
def shaft(length=97.6, base_w=5.8, tip_w=4.0, base_h=6.5, tip_h=2.6, mid_w=4.7, mid_h=4.4, drift=-1.2, rise=0.4):
    solid = s3d.loft("shaft", s3d.station(z=0, w=base_w, d=base_h, y=-base_h/2, n=4), s3d.station(z=length/2, w=mid_w, d=mid_h, x=drift, y=-mid_h/2, n=4), s3d.station(z=length, w=tip_w, d=tip_h, y=-tip_h/2-rise, n=4))
```

**Right**: a lamp shade, a revolve of four points.

```python
def open_shade(d=330.3, top_d=320, h=273, bottom_wall=6, top_wall=13):
    wall = s3d.revolve("shade_wall", s3d.poly([(d/2,0),(top_d/2,h),(top_d/2-top_wall,h),(d/2-bottom_wall,0)]))
```

**Right**: a tray in one verb.

```python
    tray = s3d.shell(s3d.box("tray", w=w, d=d, h=h), t=t, open="top")
```

**Wrong**: a soup bowl.

```python
def bowl_body(r=60, h=66.18, t=1):
    profile = s3d.poly([(0, 0), s3d.spline([(0, 0), (21, 4), (42, 19), (56, 43), (r, h)], tangents=((1, 0), (0, 1))), (r-t, h), s3d.spline([(r-t, h), (54, 41), (41, 19), (21, 5), (0, t)], tangents=((0, -1), (-1, 0)))])
```

The inner wall is typed by hand, point by point, beside the outer one. `shell` says it in one word, and keeps the wall even.

**How it should read**: the outside only, then shell. Same shape (F 1.000 against the original).

```python
def bowl_body(r=60, h=66.18, t=1):
    profile = s3d.poly([(0, 0), s3d.spline([(0, 0), (21, 4), (42, 19), (56, 43), (r, h)], tangents=((1, 0), (0, 1))), (0, h)])
    body = s3d.shell(s3d.revolve("curved_bowl", profile), t=t, open="+Z")
```

## 7. Dress the parts when appearance matters

When the target has colours and materials, every part gets `s3d.material(ob, kind, color="#rrggbb")`,
once, before it is repeated. On shape-only data, such as CAD pieces, it is optional.

- On a group, `material` reaches every part with no material of its own. The nearest call wins.
- `join` keeps the first object's material, `group` keeps each part's. Join only what is one colour.

**Right**: dressed before the grid.

```python
    plank = s3d.box("top_plank", w=w, d=plank_d, h=h)
    s3d.material(plank, "wood", color="#5d4533")
    planks = s3d.at(plank, on=s3d.grid(nx=1, ny=5, py=pitch))
```

## 8. One frame per joint

In the head, `f = s3d.local(parent, x, y, z)` gives the pivot. The same `f` goes to `s3d.at(child, on=f)`
and to `s3d.joint(name, child, f, …)`.

- **Every joint sits in the head**, frame, part and joint side by side, so a part comes off with its own lines.
- **`local` reads the parent's own space**, however deep the chain. No coordinate is composed into the world.
- **`axis` is a direction** in the space the parent lives in: `"+X"`. A source's axis is turned by its `rpy` first.
- **The sign of the axis decides which way it opens.** A positive value turns by the right hand about the
  axis. Of 23 drafts where a small edit made the joints right, 8 changed that sign and nothing else.
- **What does not move has no joint.** Place it and put it in its parent's group before the joint is
  declared, or it stays behind when the parent moves.
- **Build at rest.** Every joint at 0, no `s3d.pose` in the program.

**Right**: two slides and a hatch hinged on a sliding lid.

```python
    left_f = s3d.local(cabinet, -350, 0, 900)
    left = s3d.at(lid_left_part(), on=left_f)
    s3d.joint("slide_left", left, left_f, kind="prismatic", axis="+X", range=(0, 650))
    right_f = s3d.local(cabinet, 350, 0, 920)
    right = s3d.at(lid_right_part(), on=right_f)
    s3d.joint("slide_right", right, right_f, kind="prismatic", axis="-X", range=(0, 650))
    hatch_f = s3d.local(right, 0, 170, 18)
    hatch = s3d.at(service_hatch_part(), on=hatch_f)
    s3d.joint("hatch_hinge", hatch, hatch_f, kind="revolute", axis="-X", range=(0, 114.59))
```

**Wrong**: a rotary shaft, the part on one frame and the joint on another, 0 of 1 joints right.

```python
    f = s3d.local(body, 1, 0, 5)
    spindle = s3d.at(spindle, on=s3d.local(body))
    s3d.joint("body_to_spindle", spindle, f, kind="continuous", axis="+X")
```

**Fixed by the model in its next draft**: one frame for both, 1 of 1.

```python
    f = s3d.local(body, 0, 0, 0)
    spindle = s3d.at(spindle, on=f)
    s3d.joint("body_to_spindle", spindle, f, kind="continuous", axis="+X")
```

**Wrong**: a wall cabinet, the door swings into the cabinet, 0 of 1 joints right.

```python
    s3d.joint("door_hinge", door, f, kind="revolute", axis="+Z", range=(0, 114.59))
```

**Fixed by the model in its next draft**: one sign changed, 1 of 1.

```python
    s3d.joint("door_hinge", door, f, kind="revolute", axis="-Z", range=(0, 114.59))
```
