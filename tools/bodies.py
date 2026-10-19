#!/usr/bin/env python3
# Copyright 2023 Aberrant Behavior LLC
"""Test scenes for two-way coupled bodies (engine/bodies.cu), and what their bakes' diagnostics say about them. Only Python's standard library.

  bodies.py scene KIND [key=value ...] > scene.json     write a scene: a tank of water, half full, with a body in it
  bodies.py report diagnostics.jsonl [key=value ...]    the body's height, speed and turn by frame, and what the kind should come to

Kinds:
  float     a ball at its equilibrium already: density 500 (half the water's), its centre on the waterline. It should stay there
  drop      the same ball let go from above the water: it falls in, bobs and settles with its centre on the waterline
  rise      a light ball (density 50) let go from under the water: it shoots up, and settles riding high
  sink      a heavy ball (density 2000) let go in the water: it sinks (to the floor, once there are contacts)
  capsize   a tall box (density 500), twice as tall as it is wide, let go upright and tipped 5 degrees: it can't float upright, and rolls onto its side
  block     a box (density 500) half as tall as it is wide, tipped 5 degrees: it floats upright, and rights itself
  spin      no gravity, a ball deep in the water spun at 5 rad/s about z: with nothing but pressure between them, the water can't turn a ball, so it
            should keep its spin, and the water stay still (shape=box for a cube, which does stir the water)

Keys: voxel (m, default 1/128), box (the tank's side, m, 0.5), depth (the water's, m, 0.25), radius (m, 0.08), size (the box's width, m, 0.12),
density (kg/m^3), height (where its centre starts, m), tilt (degrees), frames (96), fps (24), solver (multigrid), freeSurface (footprint or sharp),
air (0: no; 1: air simulated), spin (rad/s), shape (box: a cube in place of the ball), friction (the water's hold on it, 0 to 1), pool (0: no water),
bounce, contactFriction, velocity (x,y,z m/s), count (that many, stacked a voxel apart, each turned 7 degrees more about y), substeps (collision
substeps a frame)
"""
import json
import math
import sys


def keys(args):
    out = {}
    for arg in args:
        key, _, value = arg.partition("=")
        out[key] = value
    return out


def scene(kind, settings):
    voxel = float(settings.get("voxel", 1/128))
    box = float(settings.get("box", 0.5))
    depth = float(settings.get("depth", 0.25))
    radius = float(settings.get("radius", 0.08))
    size = float(settings.get("size", 0.12))
    tilt = math.radians(float(settings.get("tilt", 5 if kind in ("capsize", "block") else 0)))
    middle = box/2
    if kind == "spin":
        depth = float(settings.get("depth", 0.4))
    defaults = {"float": (500, depth), "drop": (500, depth + 2.5*radius), "rise": (50, depth/2), "sink": (2000, depth/2), "capsize": (500, depth), "block": (500, depth),
                "spin": (1000, depth/2)}
    density, height = defaults[kind]
    density = float(settings.get("density", density))
    height = float(settings.get("height", height))
    c, s = math.cos(tilt), math.sin(tilt)
    transform = [c, -s, 0, middle, s, c, 0, height, 0, 0, 1, middle, 0, 0, 0, 1]   #tipped about z
    if kind in ("capsize", "block") or settings.get("shape") == "box":
        tall = 2*size if kind == "capsize" else size/2 if kind == "block" else size
        obstacle = {"shape": "box", "min": [-size/2, -tall/2, -size/2], "max": [size/2, tall/2, size/2]}
    else:
        obstacle = {"shape": "sphere", "centre": [0, 0, 0], "radius": radius}
    obstacle["transform"] = transform
    if "friction" in settings:
        obstacle["friction"] = float(settings["friction"])
    obstacle["dynamic"] = {"density": density}
    if kind == "spin":
        obstacle["dynamic"]["angularVelocity"] = [0, 0, float(settings.get("spin", 5))]
    out = {
        "schema": "flip2.scene/1",
        "fps": int(settings.get("fps", 24)),
        "frames": int(settings.get("frames", 96)),
        "domain": {"min": [0, 0, 0], "max": [box, box, box], "voxelSize": voxel},
        "solver": {"pressureSolver": settings.get("solver", "multigrid")},
        "fluids": [{"shape": "box", "min": [0, 0, 0], "max": [box, depth, box]}],
        "obstacles": [obstacle],
        "output": {"dir": settings.get("dir", kind), "cache": False, "positions": False, "diagnostics": "diagnostics.jsonl"},
    }
    if settings.get("pool", "1") == "0":    #no water: the body alone, against the walls
        out["fluids"] = [{"shape": "box", "min": [0, 0, 0], "max": [voxel, voxel, voxel]}]
    if "bounce" in settings:
        obstacle["dynamic"]["bounce"] = float(settings["bounce"])
    if "contactFriction" in settings:
        obstacle["dynamic"]["contactFriction"] = float(settings["contactFriction"])
    if "velocity" in settings:
        obstacle["dynamic"]["velocity"] = [float(x) for x in settings["velocity"].split(",")]
    if "count" in settings:     #a stack: count of them, one over another, a voxel apart, each turned a little more
        bodies = []
        for i in range(int(settings["count"])):
            copy = json.loads(json.dumps(obstacle))
            a = math.radians(7*i)
            c2, s2 = math.cos(a), math.sin(a)
            step = (2*size if kind == "capsize" else size if settings.get("shape") == "box" else 2*radius) + voxel
            copy["transform"] = [c2, 0, s2, middle, 0, 1, 0, height + i*step, -s2, 0, c2, middle, 0, 0, 0, 1]
            bodies.append(copy)
        out["obstacles"] = bodies
    if kind == "spin":
        out["gravity"] = [0, 0, 0]
    if settings.get("freeSurface"):
        out["solver"]["freeSurface"] = settings["freeSurface"]
    if "substeps" in settings:
        out["solver"]["collisionSubsteps"] = int(settings["substeps"])
    if settings.get("air", "0") != "0":
        out["air"] = {"density": 1.2, "band": 8}
    return out


def report(path, settings):
    header, frames = None, []
    for line in open(path):
        record = json.loads(line)
        if "flip2" in record:
            header = record
        elif "frame" in record:
            frames.append(record)
    if not frames or "bodies" not in frames[0]:
        sys.exit(path + ": no bodies in it")
    voxel = header["voxelSize"]
    print("frame   time      y (m)     vy (m/s)   |v| (m/s)  |w| (1/s)  tilt (deg)  substeps")
    heights = []
    for record in frames:
        body = record["bodies"][0]
        p, v, w, q = body["position"], body["velocity"], body["angularVelocity"], body["orientation"]
        up = 1 - 2*(q[1]*q[1] + q[3]*q[3])     #the body's own y axis against the world's: R[1][1]
        tiltDegrees = math.degrees(math.acos(max(-1.0, min(1.0, up))))
        heights.append((record["time"], p[1]))
        print("%5d  %7.4f  %9.5f  %9.5f  %9.5f  %9.5f  %9.3f  %8s" % (record["frame"], record["time"], p[1], v[1], math.sqrt(sum(x*x for x in v)),
                                                                    math.sqrt(sum(x*x for x in w)), tiltDegrees, record.get("substeps", "")))
    # heave: the times the height crosses its mean over the last half going up, and the period between them
    late = heights[len(heights)//2:]
    mean = sum(y for _, y in late) / len(late)
    crossings = [t0 + (t1 - t0)*(mean - y0)/(y1 - y0) for (t0, y0), (t1, y1) in zip(heights, heights[1:]) if y0 < mean <= y1]
    periods = [b - a for a, b in zip(crossings, crossings[1:])]
    spread = max(y for _, y in late) - min(y for _, y in late)
    print("last half: mean height %.5f m, spread %.5f m (%.2f voxels)" % (mean, spread, spread/voxel))
    if periods:
        print("heave periods (s):", " ".join("%.3f" % period for period in periods[:8]))
    if "expect" in settings:
        expect = float(settings["expect"])
        print("expected %.5f m: off by %.5f m, %.2f voxels" % (expect, mean - expect, (mean - expect)/voxel))


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("scene", "report"):
        sys.exit(__doc__)
    if sys.argv[1] == "scene":
        print(json.dumps(scene(sys.argv[2], keys(sys.argv[3:])), indent=1))
    else:
        report(sys.argv[2], keys(sys.argv[3:]))


if __name__ == "__main__":
    main()
