#!/usr/bin/env python3
# Copyright 2023 Aberrant Behavior LLC
"""Writes the character wading scene (scenes/wading.json): a figure walking through water up to its thighs, the P1 exit test for deforming obstacles at a
character's scale. Only Python's standard library, so it runs on coeus.

The figure is eight closed meshes, each a tube skinned along a chain of bones the way a character's skin follows its skeleton: two legs (hip, knee,
ankle), two feet (ankle, toe), the torso (pelvis, chest, neck), the head and two arms (shoulder, elbow, wrist). Each tube's rings sit along its bones,
turned halfway between neighbouring bones at a joint, so it bends without seams, and its ends are rounded. A bend much sharper than the knee's folds the
inside of a tube through itself, so the feet are their own tubes. Each part is its own deforming obstacle: flip2 takes the nearest obstacle at every
point, so parts that overlap (a thigh into the hips, a foot into the shin) make an exact union, which one mesh of overlapping shells wouldn't.

--folded writes scenes/wading-folded.json instead, whose legs run on through the ankle to the toe as one tube each, folding through themselves at the
ankle: the mesh wraps the fold twice, which flip2's winding numbers count as inside (parity and the nearest surface's normal both read it as outside, and
water seeded there was sealed in the foot).

The walk is a plain sinusoidal gait at 0.7 m/s: hips swinging 25 degrees either way, knees bending in the swing, ankles following, arms swinging against
the legs. Feet slide a little in the stance; it's a test of the water, not of the walk. Samples of every vertex at 24 fps, as (n, 3) float32 .npy files.

  wading.py [--scenes DIR] [--seconds S] [--voxel V] [--folded]     (default: the repo's scenes/, 3 s, 0.02 m)
"""
import argparse
import json
import math
from pathlib import Path

from deforming import write_npy

REPO = Path(__file__).resolve().parent.parent
FPS = 24
SPEED = 0.7             # m/s along +x
STRIDE = 1.1            # m per full gait cycle (two steps)
DEPTH = 0.55            # water depth, m
AROUND = 16             # vertices around each ring
CAP = 4                 # rings in each rounded end


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s):
    return tuple(x * s for x in a)


def normalize(a):
    length = math.sqrt(sum(x * x for x in a))
    return tuple(x / length for x in a)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def direction(angle):
    """a unit vector in the sagittal (x, y) plane, angle measured from straight down, positive forwards"""
    return (math.sin(angle), -math.cos(angle), 0.0)


def skeleton(time, folded=False):
    """the figure's joint chains at time: (name, joints, radii), all bones in planes of constant z; folded, the legs and feet as one chain each"""
    phase = 2 * math.pi * time * SPEED / STRIDE
    x = 0.45 + SPEED * time
    pelvis = (x, 0.93 + 0.015 * math.cos(2 * phase), 0.6)
    chains = []
    for side, offset in (("left", 0.0), ("right", math.pi)):
        swing = math.radians(25) * math.sin(phase + offset)                         # hip: forwards positive
        bend = math.radians(55) * max(0.0, math.sin(phase + offset + 1.2)) ** 1.5    # knee flexes in the swing
        lift = math.radians(10) * math.sin(phase + offset + 2.0)                     # ankle
        z = 0.6 + (0.1 if side == "left" else -0.1)
        hip = (pelvis[0], pelvis[1] - 0.04, z)
        knee = add(hip, scale(direction(swing), 0.44))
        ankle = add(knee, scale(direction(swing - bend), 0.42))
        toe = add(ankle, scale(direction(swing - bend + math.radians(80) + lift), 0.16))
        if folded:
            chains.append(("leg-" + side, [hip, knee, ankle, toe], [0.085, 0.062, 0.045, 0.035]))
            continue
        chains.append(("leg-" + side, [hip, knee, ankle], [0.085, 0.062, 0.045]))
        chains.append(("foot-" + side, [ankle, toe], [0.045, 0.035]))  # its own tube: bent 80 degrees at the ankle, one tube would fold through itself
    lean = math.radians(6)
    chest = add(pelvis, scale(direction(math.pi - lean), 0.38))
    neck = add(chest, scale(direction(math.pi - lean), 0.17))
    chains.append(("torso", [pelvis, chest, neck], [0.14, 0.16, 0.065]))
    crown = add(neck, scale(direction(math.pi - lean), 0.24))
    chains.append(("head", [add(neck, scale(direction(math.pi - lean), 0.06)), crown], [0.095, 0.09]))
    for side, offset in (("left", math.pi), ("right", 0.0)):   # against the leg on its side
        swing = math.radians(20) * math.sin(phase + offset)
        z = 0.6 + (0.21 if side == "left" else -0.21)
        shoulder = (chest[0] + 0.01, chest[1] + 0.12, z)
        elbow = add(shoulder, scale(direction(swing), 0.29))
        wrist = add(elbow, scale(direction(swing + math.radians(25)), 0.26))
        chains.append(("arm-" + side, [shoulder, elbow, wrist], [0.05, 0.04, 0.033]))
    return chains


def tube(joints, radii, rings_per_bone=5):
    """a closed tube along joints: its vertices, ring by ring and then the two poles. The same count and order for any pose, so its triangles hold"""
    centres, tangents, sizes = [], [], []
    bones = [normalize(sub(b, a)) for a, b in zip(joints, joints[1:])]
    for index, (a, b) in enumerate(zip(joints, joints[1:])):
        for k in range(rings_per_bone):
            t = k / rings_per_bone
            centres.append(add(a, scale(sub(b, a), t)))
            if k == 0 and index > 0:    # at a joint: halfway between the two bones' directions
                tangents.append(normalize(add(bones[index - 1], bones[index])))
            else:
                tangents.append(bones[index])
            sizes.append(radii[index] + (radii[index + 1] - radii[index]) * t)
    centres.append(joints[-1])
    tangents.append(bones[-1])
    sizes.append(radii[-1])
    side = (0.0, 0.0, 1.0)  # every bone lies in a plane of constant z, so z is across all of them
    vertices = []

    def ring(centre, tangent, radius):
        across = normalize(cross(tangent, side))
        return [add(centre, add(scale(across, radius * math.cos(2 * math.pi * j / AROUND)), scale(side, radius * math.sin(2 * math.pi * j / AROUND))))
                for j in range(AROUND)]

    for k in range(1, CAP):     # the rounded start, from near its pole to the first ring
        angle = math.pi / 2 * (1 - k / CAP)
        vertices += ring(sub(centres[0], scale(tangents[0], sizes[0] * math.sin(angle))), tangents[0], sizes[0] * math.cos(angle))
    for centre, tangent, radius in zip(centres, tangents, sizes):
        vertices += ring(centre, tangent, radius)
    for k in range(1, CAP):     # and the rounded end
        angle = math.pi / 2 * k / CAP
        vertices += ring(add(centres[-1], scale(tangents[-1], sizes[-1] * math.sin(angle))), tangents[-1], sizes[-1] * math.cos(angle))
    vertices.append(sub(centres[0], scale(tangents[0], sizes[0])))      # the poles
    vertices.append(add(centres[-1], scale(tangents[-1], sizes[-1])))
    return vertices


def triangles(rings):
    """a tube's triangles, wound outwards, for rings rings of AROUND vertices then the two poles"""
    out = []
    for r in range(rings - 1):
        for j in range(AROUND):
            a, b = r * AROUND + j, r * AROUND + (j + 1) % AROUND
            c, d = a + AROUND, b + AROUND
            out += [(a, c, b), (b, c, d)]
    first, last, top = rings * AROUND, rings * AROUND + 1, (rings - 1) * AROUND
    for j in range(AROUND):     # each shared edge the other way round from the quads' beside it
        out.append((first, j, (j + 1) % AROUND))
        out.append((last, top + (j + 1) % AROUND, top + j))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=str(REPO / "scenes"))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--folded", action="store_true", help="legs and feet as one tube each, folded through itself at the ankle")
    args = parser.parse_args()
    scenes = Path(args.scenes)
    name = "wading-folded" if args.folded else "wading"
    geo = scenes / "geo" / name
    geo.mkdir(parents=True, exist_ok=True)
    frames = int(round(args.seconds * FPS))
    obstacles = []
    for part, (part_name, joints, radii) in enumerate(skeleton(0.0, args.folded)):
        rest = tube(joints, radii)
        rings = (len(rest) - 2) // AROUND
        faces = triangles(rings)
        # wound outwards: the signed volume they enclose is positive
        volume = sum(sum(p * q for p, q in zip(rest[x], cross(rest[y], rest[z]))) for x, y, z in faces) / 6
        if volume < 0:
            faces = [(x, z, y) for x, y, z in faces]
        with open(geo / f"{part_name}.obj", "w") as obj:
            obj.write(f"# {part_name} of the wading figure, {len(faces)} triangles (tools/wading.py)\n")
            obj.writelines("v %.6f %.6f %.6f\n" % v for v in rest)
            obj.writelines("f %d %d %d\n" % (x + 1, y + 1, z + 1) for x, y, z in faces)
        (geo / part_name).mkdir(exist_ok=True)
        samples = []
        for frame in range(frames + 1):
            time = frame / FPS
            joints_now, radii_now = skeleton(time, args.folded)[part][1:]
            write_npy(geo / part_name / f"{frame:03d}.npy", tube(joints_now, radii_now))
            samples.append({"time": time, "vertices": f"geo/{name}/{part_name}/{frame:03d}.npy"})
        obstacles.append({"mesh": f"geo/{name}/{part_name}.obj", "deforming": samples})
    voxel = args.voxel
    scene = {"schema": "flip2.scene/1", "fps": FPS, "frames": frames,
             "domain": {"min": [0, 0, 0], "max": [3.0, 1.2, 1.2], "voxelSize": voxel},
             "fluids": [{"shape": "box", "min": [0, 0, 0], "max": [3.04, DEPTH, 1.2]}],
             "obstacles": obstacles,
             "output": {"dir": name, "diagnostics": "diagnostics.jsonl"}}
    with open(scenes / f"{name}.json", "w") as out:
        json.dump(scene, out, indent=1)
    print(f"wrote {scenes / (name + '.json')}: {len(obstacles)} deforming parts, {frames + 1} samples each")


if __name__ == "__main__":
    main()
