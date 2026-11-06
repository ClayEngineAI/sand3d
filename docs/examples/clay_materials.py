"""A tagged box-and-cylinder prop for Clay's material-preset verification.

    PYTHONPATH=src python docs/examples/clay_materials.py prop.glb
"""
import sys

import sand3d as s3d


def prop():
    base = s3d.material(s3d.box("wood_base", 1600, 600, 80), "wood", repeat=(2, 2))
    shelf = s3d.material(s3d.at(s3d.box("wood_shelf", 1600, 600, 80), z=800), "wood", repeat=2)
    posts = s3d.at(s3d.cylinder("metal_post", 80, 720), on=s3d.grid(2, 2, 1450, 450), z=80)
    s3d.material(posts, "metal", color="#bdc3c9", roughness=0.35)
    wall = s3d.material(s3d.at(s3d.box("brick_wall", 600, 200, 650), x=400, z=80), "brick", repeat=(2, 2))
    mirrored = s3d.transform(s3d.copy(wall), mirror="x")
    markers = s3d.at(s3d.box("unpassed", 120, 120, 120), on=s3d.grid(2, 1, 500), z=880)
    s3d.material(markers, "plastic", color="#b55bd8")
    return s3d.assembly("clay_material_prop", base, shelf, posts, wall, mirrored, markers)


if __name__ == "__main__":
    s3d.empty_scene()
    s3d.export(prop(), sys.argv[1])
