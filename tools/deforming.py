#!/usr/bin/env python3
# Copyright 2023 Aberrant Behavior LLC
"""Writes the deforming-obstacle test scenes: a coarse icosphere (320 triangles) and samples of its vertices over time, as (n, 3) float32 .npy files,
one per frame at 24 fps, in a quarter-metre tank half full of water (scenes/tank-*.json). Only Python's standard library, so it runs on coeus.

  tank-still-deforming    the sphere held still by two identical samples: the tank should stay as still as tank-meshsphere's
  tank-sweep              the sphere swept along x by its samples, 5 cm either way once a second
  tank-sweep-rigid        the same sweep as keyframed transforms of the rigid mesh, to compare with
  tank-pulse              the sphere breathing in place, its radius 5 cm +-20% twice a second

  deforming.py [--scenes DIR]     (default: the repo's scenes/)
"""
import argparse
import json
import math
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FPS = 24
SECONDS = 1


def icosphere(subdivisions):
    """a unit icosphere: its vertices, and its triangles wound outwards"""
    t = (1 + math.sqrt(5)) / 2
    vertices = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    vertices = [tuple(c / math.sqrt(1 + t * t) for c in v) for v in vertices]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        middles = {}

        def middle(a, b):
            key = (min(a, b), max(a, b))
            if key not in middles:
                m = [(p + q) / 2 for p, q in zip(vertices[a], vertices[b])]
                length = math.sqrt(sum(c * c for c in m))
                vertices.append(tuple(c / length for c in m))
                middles[key] = len(vertices) - 1
            return middles[key]

        faces = [f for a, b, c in faces for f in ((a, middle(a, b), middle(c, a)), (b, middle(b, c), middle(a, b)), (c, middle(c, a), middle(b, c)),
                                                  (middle(a, b), middle(b, c), middle(c, a)))]
    return vertices, faces


def write_npy(path, rows):
    """an (n, 3) little-endian float32 array"""
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, 3), }" % len(rows)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    with open(path, "wb") as out:
        out.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1"))
        out.write(b"".join(struct.pack("<3f", *row) for row in rows))


def tank(name, obstacle):
    """the quarter-metre tank, half full, around one obstacle"""
    return {"schema": "flip2.scene/1", "fps": FPS, "frames": FPS * SECONDS,
            "domain": {"min": [0, 0, 0], "max": [0.25, 0.25, 0.25], "voxelSize": 0.0078125},
            "fluids": [{"shape": "box", "min": [0, 0, 0], "max": [0.25, 0.125, 0.25]}],
            "obstacles": [obstacle],
            "output": {"dir": name, "diagnostics": "diagnostics.jsonl", "positions": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default=str(REPO / "scenes"))
    args = parser.parse_args()
    scenes = Path(args.scenes)
    geo = scenes / "geo"
    unit, faces = icosphere(2)
    geo.mkdir(parents=True, exist_ok=True)
    with open(geo / "sphere320.obj", "w") as obj:
        obj.write(f"# unit icosphere, {len(faces)} triangles (tools/deforming.py)\n")
        obj.writelines("v %.9f %.9f %.9f\n" % v for v in unit)
        obj.writelines("f %d %d %d\n" % (a + 1, b + 1, c + 1) for a, b, c in faces)
    times = [frame / FPS for frame in range(FPS * SECONDS + 1)]

    def sequence(name, centre_and_radius, sample_times):
        """writes the samples of a sphere placed by centre_and_radius(t), and returns the obstacle's deforming list"""
        (geo / name).mkdir(exist_ok=True)
        samples = []
        for index, time in enumerate(sample_times):
            centre, radius = centre_and_radius(time)
            write_npy(geo / name / f"{index:02d}.npy", [tuple(c + radius * u for c, u in zip(centre, v)) for v in unit])
            samples.append({"time": time, "vertices": f"geo/{name}/{index:02d}.npy"})
        return samples

    def sweep(time):
        return (0.125 + 0.05 * math.sin(2 * math.pi * time), 0.06, 0.125), 0.05

    def pulse(time):
        return (0.125, 0.062, 0.125), 0.05 * (1 + 0.2 * math.sin(2 * math.pi * 2 * time))

    written = {
        "tank-still-deforming": {"mesh": "geo/sphere320.obj", "deforming": sequence("still", lambda time: ((0.125, 0.06, 0.125), 0.05), [0, SECONDS])},
        "tank-sweep": {"mesh": "geo/sphere320.obj", "deforming": sequence("sweep", sweep, times)},
        "tank-sweep-rigid": {"mesh": "geo/sphere320.obj", "keyframes": [
            {"time": time, "transform": [0.05, 0, 0, sweep(time)[0][0], 0, 0.05, 0, 0.06, 0, 0, 0.05, 0.125, 0, 0, 0, 1]} for time in times]},
        "tank-pulse": {"mesh": "geo/sphere320.obj", "deforming": sequence("pulse", pulse, times)},
    }
    for name, obstacle in written.items():
        with open(scenes / f"{name}.json", "w") as out:
            json.dump(tank(name, obstacle), out, indent=1)
        print(f"wrote {scenes / name}.json")


if __name__ == "__main__":
    main()
