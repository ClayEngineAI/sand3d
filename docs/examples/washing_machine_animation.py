"""Export a washing machine with an eight-second looping demonstration.

    PYTHONPATH=src python docs/examples/washing_machine_animation.py out/washer
"""
from pathlib import Path
import sys

import sand3d as s3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "programs"))
from washing_machine import washer


def animated_washer():
    machine = washer()
    s3d.material(machine, "plastic", color="#f2f2f0")
    drum = next(part for part in machine.root.children if part.name == "drum")
    back = machine.front.move(0, 400).offset(-400)
    lifter = s3d.at(s3d.box("drum_lifter", 55, 20, 300), on=back.move(0, -175).offset(15))
    s3d.join(drum, lifter)
    cycle = s3d.clip("washer_cycle",
        s3d.keyframes(machine, {0: 0, 1.5: 105, 2.5: 105, 4: 0, 8: 0},
                      joint="door_hinge", ease="in_out_sine"),
        s3d.keyframes(machine, {0: 0, 1.5: 140, 2.5: 140, 4: 0, 8: 0},
                      joint="drawer_slide", ease="in_out_cubic"),
        s3d.keyframes(machine, {0: 0, 4: 0, 8: 1080}, joint="drum_spin"),
        s3d.keyframes(machine, {0: 0, 4: 0, 4.5: 90, 7.5: 90, 8: 360},
                      joint="dial_turn", ease="in_out_sine"))
    return machine, cycle


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/washing-machine-animated")
    output.mkdir(parents=True, exist_ok=True)
    s3d.empty_scene()
    machine, cycle = animated_washer()
    for suffix in (".glb", ".gltf"):
        s3d.export(machine, output / ("washing-machine" + suffix), animations=[cycle])
