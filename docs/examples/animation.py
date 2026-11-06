"""Four named clips, exported as both glTF and GLB.

    PYTHONPATH=src .venv/bin/python docs/examples/animation.py [output-directory]
"""
from pathlib import Path
import sys

import sand3d as s3d


def animated_machine():
    base = s3d.box("base", 240, 150, 15)
    door = s3d.at(s3d.box("door", 80, 8, 110), x=50, y=40, z=15)
    drawer = s3d.at(s3d.box("drawer", 75, 70, 20), x=50, y=-20, z=15)
    rotor = s3d.at(s3d.box("rotor", 70, 12, 12), x=-60, z=60)
    screw = s3d.at(s3d.cylinder("screw", 10, 45), x=-60, y=45, z=15)
    s3d.material(base, "painted_metal", color="#364254")
    s3d.material(door, "painted_metal", color="#ef8043")
    s3d.material(drawer, "painted_metal", color="#42b5bd")
    s3d.material(rotor, "painted_metal", color="#edc354")
    s3d.material(screw, "steel")
    s3d.joint("door_hinge", door, s3d.local(base, 10, 40, 15), axis="+Z", range=(0, 120))
    s3d.joint("drawer_slide", drawer, s3d.local(base), kind="prismatic", axis="-Y", range=(0, 90))
    s3d.joint("rotor_spin", rotor, s3d.local(base, -60, 0, 60), kind="continuous", axis="+Z")
    s3d.joint("screw_turn", screw, s3d.local(base, -60, 45, 15), axis="+Z", range=(0, 1080))
    s3d.joint("screw_lift", screw, s3d.local(base), kind="prismatic", axis="+Z",
              follows="screw_turn", ratio=8 / 360)
    machine = s3d.assembly("animated_machine", base, door, drawer, rotor, screw)
    opening = s3d.clip("open", s3d.keyframes(door, {0: 0, 1: 110}, ease="in_out_sine"),
                       s3d.keyframes(drawer, {0: 0, 0.5: 0, 1.5: 80}, ease="out_cubic"))
    spinning = s3d.clip("spin", s3d.keyframes(rotor, {0: 0, 2: 1080}),
                        s3d.keyframes(screw, {0: 0, 2: 720}, joint="screw_turn"))
    bouncing = s3d.clip("bounce", s3d.keyframes(machine, z={0: 100, 1.5: 0}, ease="out_bounce"))
    custom = s3d.clip("custom", s3d.keyframes(drawer, {0: 0, 1.5: 80}, ease=lambda t: t * t * (3 - 2 * t)))
    return machine, (opening, spinning, bouncing, custom)


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/animation_demo")
    output.mkdir(parents=True, exist_ok=True)
    s3d.empty_scene()
    machine, animations = animated_machine()
    for suffix in (".gltf", ".glb"):
        path = output / ("machine" + suffix)
        s3d.export(machine, path, animations=animations)
        print(path)
