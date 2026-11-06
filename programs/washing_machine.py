"""A front-loading washing machine, articulated: written with the joint vocabulary in one piece.

Millimetres; the front looks +X, Z up. Four joints, each on a frame read on the cabinet:
the door centred on the porthole, hinged at its outer -Y edge so that it clears the
cabinet (it opens outward about "-Z", 0–143°), the drum endless about the front's
normal, the detergent drawer sliding out 0–200 mm, the dial endless. The drawer's bay and the dial's seat are cut by `seat`, as they sit at rest;
the door's glass is seated in a groove of its ring. Built at rest; `washer_open()` poses it.
"""
import sand3d as s3d


def washer(w=600, d=600, h=850, porthole=440, door_d=480, drum_d=420):
    cab = s3d.box("cabinet", w, d, h)
    s3d.fillet(cab, 12, edges="side")
    front = cab.face("+X")                                             # X is +Y, Y is up
    s3d.pocket(cab, s3d.circle(porthole), d - 100, on=front.move(0, 400))
    s3d.material(cab, "plastic", color="#f2f2f0")

    back = front.move(0, 400).offset(-(d - 200))                     # a solid's height runs along its frame's normal
    drum = s3d.at(s3d.cylinder("drum", drum_d, d - 220), on=back)
    s3d.cut(drum, s3d.at(s3d.cylinder("bore", drum_d - 20, d - 229), on=back.offset(10)))
    s3d.join(drum, s3d.at(s3d.cylinder("axle", 40, 60), on=back.offset(-60)))
    s3d.material(drum, 'metal', color='#b4b7bb', roughness=0.35)

    ring = s3d.at(s3d.tube("ring", door_d, 300, 45), on=front.move(0, 400))  # centred on the porthole
    glass = s3d.at(s3d.cylinder("glass", 320, 16), on=front.move(0, 400).offset(14))
    hinge = front.move(-door_d / 2, 400).offset(45)                    # the ring's outer -Y edge: it clears the cabinet
    s3d.seat(ring, glass)
    s3d.material(ring, 'metal', color="#3a3d42", metallic=0)
    s3d.material(glass, "glass", color="#cfe3e6", opacity=0.4)
    handle = s3d.at(s3d.box("handle", 24, 110, 30), on=s3d.bbox(ring, "+X").move(x=door_d / 2 - 40))
    door = s3d.group("door", ring, glass, handle)                      # the handle and the glass move with it

    top_band = front.move(0, h - 75)
    drawer = s3d.at(s3d.box("drawer", 200, 100, 231), on=top_band.move(150).offset(-230))
    s3d.join(drawer, s3d.at(s3d.box("drawer_front", 220, 120, 15), on=top_band.move(150)))
    s3d.seat(cab, drawer)
    dial = s3d.at(s3d.cylinder("dial", 60, 30), on=top_band.move(-150).offset(-10))
    s3d.seat(cab, dial)
    s3d.material([drawer, dial], "plastic", color="#e6e6e3")

    s3d.joint("door_hinge", door, hinge, axis="-Z", range=(0, 143))
    s3d.joint("drum_spin", drum, front.move(0, 400), kind="continuous")
    s3d.joint("drawer_slide", drawer, top_band.move(150), kind="prismatic", axis="+X", range=(0, 200))
    s3d.joint("dial_turn", dial, top_band.move(-150), kind="continuous")
    return s3d.assembly("washing_machine", cab, drum, door, drawer, dial, front=front)


def washer_open():
    machine = washer()
    s3d.pose(machine, 92, joint="door_hinge")
    s3d.pose(machine, 120, joint="drawer_slide")
    s3d.pose(machine, 30, joint="dial_turn")
    return machine
