"""What a part is made of: a named preset, its values changed where a program says so.

`material` tags an object — a part, or the root of a group or an assembly — and
never touches a mesh. Every mesh then takes the material of its nearest tagged
ancestor-or-self: a call on an assembly reaches every part under it that names
none of its own, in whatever order the calls come, and a detail attached later
inherits. The tag lives on the object, so it goes through every edit that rebuilds
the mesh (`cut`, `hole`, `fillet`…) and with `copy`; a tool's tag is used up with
the tool, and `join` keeps its first object's.

`resolve` turns the tags into Blender materials: one Principled BSDF of flat values
per distinct look — what a GLB carries (base colour, metallic, roughness, alpha,
transmission, clearcoat, sheen, emission). It runs after every public verb once a
program has set a material, so the scene is right whoever reads it; a program that
sets none never runs it, and builds and exports exactly as before.
"""
import difflib
import json
import math

import bpy

from . import _core

#: Custom property holding an object's tag, and a material's look.
TAG = "s3d_material"

#: Clay starter presets. Flat colours are the mean linear base-colour image times
#: its linear tint, encoded as sRGB. Roughness is the mean green channel.
#: Derived from ClayEngineAI/clayengine c6f702b4f5 starter textures; no images are embedded.
PRESETS = {
    'brick': {'color': '#9a4932', 'roughness': 0.5919535, 'tint': '#ad5941', 'mean': (0.77396067, 0.67031806, 0.60408652)},
    'ceramic': {'color': '#e9e7e0', 'roughness': 0.2879309, 'tint': '#f9f7f0', 'mean': (0.85952095, 0.85948095, 0.85952988)},
    'cobbles': {'color': '#8a857c', 'roughness': 0.86520384, 'tint': '#a39c92', 'mean': (0.69700559, 0.70007523, 0.7017679)},
    'concrete': {'color': '#9e9c96', 'roughness': 0.71054219, 'tint': '#b4b2ab', 'mean': (0.74694049, 0.74703363, 0.7456364)},
    'fabric': {'color': '#897c6c', 'roughness': 0.88659644, 'tint': '#a1927f', 'mean': (0.70712543, 0.70712543, 0.70712543)},
    'glass': {'color': '#ffffff', 'roughness': 0.05, 'transmission': 1.0, 'ior': 1.5, 'thickness': 0.1},
    'granite': {'color': '#8c8884', 'roughness': 0.15949004, 'tint': '#acabab', 'mean': (0.63490552, 0.60405333, 0.56429499)},
    'grass': {'color': '#4f7d32', 'roughness': 0.90322895, 'tint': '#6aa848', 'mean': (0.54455225, 0.5235333, 0.49094803)},
    'gravel': {'color': '#8a8378', 'roughness': 0.83456566, 'tint': '#aea69a', 'mean': (0.59782038, 0.59488523, 0.57962152)},
    'ice': {'color': '#b8dcef', 'roughness': 0.39332123, 'tint': '#c8ebfd', 'mean': (0.82836466, 0.85843566, 0.88219017)},
    'marble': {'color': '#b5b7c1', 'roughness': 0.43305796, 'tint': '#ffffff', 'mean': (0.46071108, 0.4755572, 0.5362856)},
    'metal': {'color': '#a8adb3', 'roughness': 0.17721718, 'tint': '#bdc3c9', 'mean': (0.7681427, 0.7681427, 0.7681427), 'metallic': 1.0},
    'mud': {'color': '#4a3a2a', 'roughness': 0.53579429, 'tint': '#65503b', 'mean': (0.52983494, 0.52425485, 0.52620791)},
    'pavers': {'color': '#a19584', 'roughness': 0.75917889, 'tint': '#bdaf9c', 'mean': (0.69617215, 0.69735645, 0.69866195)},
    'planks': {'color': '#9b7149', 'roughness': 0.65444815, 'tint': '#b6875a', 'mean': (0.7025811, 0.68162213, 0.64691189)},
    'plastic': {'color': '#a3a2a5', 'roughness': 0.4},
    'rock': {'color': '#76726c', 'roughness': 0.69511942, 'tint': '#918c86', 'mean': (0.64393567, 0.64242022, 0.63172003)},
    'rust': {'color': '#cdafa1', 'roughness': 0.60035967, 'tint': '#ffffff', 'mean': (0.6081077, 0.42886648, 0.35617403), 'metallic': 1.0},
    'sand': {'color': '#cdb586', 'roughness': 0.75636274, 'tint': '#dac696', 'mean': (0.87218789, 0.81483556, 0.78049079)},
    'sandstone': {'color': '#715241', 'roughness': 0.7797827, 'tint': '#ffffff', 'mean': (0.16424743, 0.08461205, 0.05333959)},
    'shingles': {'color': '#5a5550', 'roughness': 0.77772072, 'tint': '#78716f', 'mean': (0.54438165, 0.54510224, 0.50426526)},
    'snow': {'color': '#eff0f0', 'roughness': 0.4900932, 'tint': '#ffffff', 'mean': (0.86428835, 0.87207652, 0.87444021)},
    'soil': {'color': '#6e5640', 'roughness': 0.90590431, 'tint': '#ad896b', 'mean': (0.37484442, 0.37213305, 0.34809781)},
    'tarmac': {'color': '#3c3c3e', 'roughness': 0.86557038, 'tint': '#454547', 'mean': (0.75400036, 0.76152059, 0.77520232)},
    'wood': {'color': '#9f703f', 'roughness': 0.6293181, 'tint': '#be8d51', 'mean': (0.67653082, 0.60850232, 0.60064688)},
}


def material(ob, kind, color=None, roughness=None, metallic=None, opacity=None, emission=None, repeat=None):
    """Kind: brick ceramic cobbles concrete fabric glass granite grass gravel ice marble
    metal mud pavers planks plastic rock rust sand sandstone shingles snow soil tarmac wood;
    color "#rrggbb".

    `ob` — a part, a group, an assembly or a list — made of `kind`, one of the presets,
    its colour (sRGB: "#rrggbb" or (r, g, b) in 0..1), roughness, metallic, opacity (1
    opaque, 0 invisible) or emission (the light it gives, in its own colour) changed where
    given. On a group or an assembly, every part under it that names none of its own: the
    nearest call wins, in whatever order the calls come. `repeat` is a positive number
    or (x, y), texture tiles per metre, default (1, 1). GLB/glTF writes the Clay preset
    tag and local box UVs; other viewers use the flat fallback. Returns `ob`."""
    from .assembly import Handle
    from .edit import _flat
    if kind not in PRESETS:
        near = difflib.get_close_matches(str(kind), PRESETS, n=2)
        raise ValueError(f"material: no preset named {kind!r}" + (f" — did you mean {' or '.join(map(repr, near))}?"
                                                                  if near else "")
                         + f"; the presets: {', '.join(PRESETS)}")
    given = {"color": None if color is None else _hex(color)}
    for name, value, top in (("roughness", roughness, 1), ("metallic", metallic, 1), ("opacity", opacity, 1),
                             ("emission", emission, None)):
        if value is not None:
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (top and value > top)):
                raise ValueError(f"material: {name}={value!r} is not a number "
                                 + ("between 0 and 1" if top else "of 0 or more")
                                 + (" — 1 is opaque, 0 invisible" if name == "opacity" else ""))
        given[name] = None if value is None else float(value)
    if repeat is not None:
        tiles = (repeat, repeat) if isinstance(repeat, (int, float)) else repeat
        if (not isinstance(tiles, (tuple, list)) or len(tiles) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v <= 0 for v in tiles)):
            raise ValueError("material: repeat is a positive finite number or (x, y), tiles per metre")
        given["repeat"] = [float(v) for v in tiles]
    tag = json.dumps({"kind": kind, **{k: v for k, v in given.items() if v is not None}}, sort_keys=True)
    targets = [t.root if isinstance(t, Handle) else t for t in _flat([ob])]
    for target in targets:                  # all checked before any is tagged
        if not isinstance(target, bpy.types.Object):
            raise TypeError(f"material: {target!r} is not a part, a group or an assembly — a profile or a path "
                            "becomes one through extrude, revolve, sweep or loft")
    for target in targets:
        target[TAG] = tag
    _core.dressed = True
    return ob


def _hex(color):
    """`color` as '#rrggbb', lower case: one spelling per colour, so one material."""
    if isinstance(color, str):
        h = color[1:] if color.startswith("#") else ""
        if len(h) == 6 and all(c in "0123456789abcdefABCDEF" for c in h):
            return "#" + h.lower()
    elif (isinstance(color, (tuple, list)) and len(color) == 3
          and all(isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1 for c in color)):
        return "#" + "".join(f"{round(c * 255):02x}" for c in color)
    raise ValueError(f"material: color={color!r} is not a colour — write '#rrggbb' or (r, g, b) in 0..1")


def _linear(hex_color):
    """'#rrggbb' (sRGB) → linear RGBA, what Blender's colour sockets hold."""
    srgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return (*(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb), 1.0)


def look(tag):
    """The values a tag stands for: its preset, changed where the call said."""
    spec = json.loads(tag)
    values = dict(metallic=0.0, opacity=1.0, transmission=0.0, coat=0.0, sheen=0.0, emission=0.0, ior=1.5)
    preset = PRESETS[spec.pop("kind")]
    values.update(preset)
    values.update(spec)
    values["emissive"] = spec.get("color", preset.get("tint", preset["color"]))
    if "color" in spec and "mean" in preset:
        linear = [c * mean for c, mean in zip(_linear(spec["color"]), preset["mean"])]
        values["color"] = _hex(tuple(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
                                     for c in linear))
    return values


def _name(tag):
    """A material's name: its preset, and what the program changed in it."""
    spec = json.loads(tag)
    kind = spec.pop("kind")
    return " ".join([kind, *(v if k == "color" else f"repeat={v[0]:g},{v[1]:g}" if k == "repeat"
                            else f"{k}={v:g}" for k, v in sorted(spec.items()))])


def _material(tag, made):
    """The Blender material of a tag: one per distinct tag, made once."""
    if tag in made:
        return made[tag]
    v = look(tag)
    mat = bpy.data.materials.new(_name(tag))
    mat[TAG] = tag
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    rgba = _linear(v["color"])
    for socket, value in (("Base Color", rgba), ("Roughness", v["roughness"]), ("Metallic", v["metallic"]),
                          ("Alpha", v["opacity"]), ("Transmission Weight", v["transmission"]),
                          ("Coat Weight", v["coat"]), ("Sheen Weight", v["sheen"]),
                          ("IOR", v["ior"]), ("Emission Color", _linear(v["emissive"])),
                          ("Emission Strength", v["emission"])):
        bsdf.inputs[socket].default_value = value
    if v["coat"]:
        bsdf.inputs["Coat Roughness"].default_value = 0.05
    if v["opacity"] < 1:
        mat.surface_render_method = "BLENDED"
    mat.diffuse_color = (*rgba[:3], v["opacity"])       # what Workbench and the viewport show
    mat.metallic, mat.roughness = v["metallic"], v["roughness"]
    made[tag] = mat
    return mat


def _nearest(ob):
    while ob is not None:
        if TAG in ob:
            return ob[TAG]
        ob = ob.parent
    return None


def clay_tag(tag):
    """The public node-extras contract; defaults stay with the Clay preset."""
    spec = json.loads(tag)
    kind = spec["kind"]
    parameters = {}
    for name in ("color", "roughness", "metallic", "opacity", "emission"):
        if name not in spec:
            continue
        value = spec[name]
        if name == "color":
            parameters["color"] = int(value[1:], 16)
        elif name == "metallic":
            parameters["metalness"] = value
        elif name == "emission":
            preset = PRESETS[kind]
            color = spec.get("color", preset.get("tint", preset["color"]))
            parameters.update(emissive=int(color[1:], 16), emissiveIntensity=value)
        else:
            parameters[name] = value
            if name == "opacity" and value < 1:
                parameters["transparent"] = True
    return {"preset": kind, "parameters": parameters, "repeat": spec.get("repeat", [1.0, 1.0])}


def resolve():
    """Give every mesh the material of its nearest tagged ancestor-or-self; take ours off
    a mesh that has none any more. Nothing to do in a program that sets no material."""
    if not _core.dressed:
        return
    made = {m[TAG]: m for m in bpy.data.materials if TAG in m}
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        if ob.type != "MESH":
            continue
        tag = _nearest(ob)
        have = list(ob.data.materials)
        if tag is None:
            if any(m is not None and TAG in m for m in have):
                ob.data.materials.clear()
            continue
        want = _material(tag, made)
        if have != [want]:
            ob.data.materials.clear()
            ob.data.materials.append(want)
        if any(p.material_index for p in ob.data.polygons):
            ob.data.polygons.foreach_set("material_index", [0] * len(ob.data.polygons))


def dressed():
    """Whether any mesh of the scene wears one of our materials."""
    return _core.dressed and any(ob.type == "MESH" and any(m is not None and TAG in m for m in ob.data.materials)
                                 for ob in bpy.data.objects)
