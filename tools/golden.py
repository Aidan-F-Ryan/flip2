#!/usr/bin/env python3
# Copyright 2023 Aberrant Behavior LLC
"""Regression runs for flip2. Runs a few scenes with FLIP2_DIAGNOSTICS set, which makes main write a line of JSON per frame (engine/diagnostics.hu):
the particles' energy, momentum and extent, how they fill the voxels, and a hash of every particle's exact state. Then compares runs frame by frame.

  golden.py record [options]          run the scenes and keep their diagnostics as the golden ones (tests/golden/<scene>.jsonl)
  golden.py check [options]           run the scenes and compare them with the golden ones
  golden.py split [options]           run each scene whole and split into --partitions partitions, and compare the two
  golden.py diff a.jsonl b.jsonl      compare two diagnostics files

A run on the same build of the same code on the same GPU architecture repeats exactly, and so does a run split between partitions, so the hash, the
counts, the histograms and the timesteps have to match exactly. The double sums (energy, momenta, centroid) are added in a different order when the domain
is split, so a split run only has to match them to 1e-9 relative. When check finds the golden files came from another GPU, build or CUDA version, it says
so and compares the physical numbers alone, with looser tolerances. Only Python's standard library: coeus has no numpy.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCENES = {   #main's arguments after the frame count (and any environment it runs with), or a scene file for flip2 bake (engine/scene.hpp)
    "dam16": ["0.95", "0.1", "16"],                 #the 1 m dam break on 16^3 nodes: 164K particles
    "dam": ["0.95", "0.1"],                         #the 1 m dam break on 32^3 nodes: 1.3M particles
    "tank": ["0.95", "0.1", "tank", "16"],          #a still tank, half full: nothing should move
    "swirl": ["0.95", "0.1", "tank", "32", "0.5"],  #a full tank with a 0.5 m/s vortex in its xy cross section
    "swirl-apic": (["0", "0.1", "tank", "32", "0.5"], {"FLIP2_TRANSFER": "apic"}),        #the same with pure APIC transfers
    "swirl-apicflip": (["0.95", "0.1", "tank", "32", "0.5"], {"FLIP2_TRANSFER": "apic"}), #and with APIC, blending FLIP in at 0.95
    "nozzle": "scenes/nozzle.json",                 #an emitter pouring into an empty box: inflow at exactly A*v
    "drain": "scenes/drain.json",                   #a tank draining through a sink and an open face
    "forces": "scenes/forces.json",                 #a drop in zero gravity, pulled, spun and stirred by force fields
    "tank-box": "scenes/tank-box.json",             #a still tank around a submerged, rotated box: it should stay still
    "paddle": "scenes/paddle.json",                 #a box keyframed to spin once a second, stirring a tank
    "tank-meshsphere": "scenes/tank-meshsphere.json",   #a still tank around an OBJ sphere, voxelized on the GPU
    "tank-pulse": "scenes/tank-pulse.json",         #a deforming mesh sphere breathing in a tank, re-voxelized every substep (tools/deforming.py)
    "sealed-pulse": "scenes/sealed-pulse.json",     #the breathing sphere in a box full of water: no air anywhere, so its growth has to be balanced (engine/pockets.cu)
    "sealed-box": "scenes/sealed-box.json",         #water sealed in a hollow box (geo/hollow.obj) above a pool: a pocket with air elsewhere
    "mesh-sources": "scenes/mesh-sources.json",     #a mesh fluid half inside a box fluid, a mesh emitter and a keyframed mesh sink, seeded and tested on the GPU
    "drop-meshsphere": "scenes/drop-meshsphere.json",   #a ball of water from a mesh, dropped into a tank: against drop-sphere's analytic one
    "nozzle-mesh": "scenes/nozzle-mesh.json",       #the nozzle with a cube mesh for its emitter: the same flow, particle for particle
    "drain-mesh": "scenes/drain-mesh.json",         #the drain with a mesh sphere for its sink
    "tank-vdbspin": "scenes/tank-vdbspin.json",     #a VDB level set sphere (needs OpenVDB) spun in place by its "vel" grid, stirring a tank
    "tank-vdbsphere": "scenes/tank-vdbsphere.json", #a still tank around a VDB level set sphere: against tank-meshsphere and tank-sphere
    "drop-vdbsphere": "scenes/drop-vdbsphere.json", #a ball of water from a VDB level set: against drop-sphere
    "tank-meshspin": "scenes/tank-meshspin.json",   #tank-vdbspin's spin as keyframes of the rigid mesh sphere
    "tension-drop": "scenes/tension-drop.json",     #a stretched drop in zero gravity, pulled back and past round by surface tension (engine/levelset.cu)
    "tension-sessile": "scenes/tension-sessile.json",   #half a drop on the floor in zero gravity, spreading to meet it at a 60 degree contact angle
    "viscous-dam": "scenes/viscous-dam.json",       #a dam break of syrup around a box: viscosity, sticking to the floor, the walls and the box (engine/viscosity.cu)
    "dam16-sharp": (["0.95", "0.1", "16"], {"FLIP2_FREE_SURFACE": "sharp"}),           #dam16 with the sharp free surface (engine/freesurface.cu)
    "tank-sharp": (["0.95", "0.1", "tank", "16"], {"FLIP2_FREE_SURFACE": "sharp"}),    #the still tank with it: nothing much should move
    "tension-drop-sharp": "scenes/tension-drop-sharp.json",     #tension-drop with it: surface tension as the surface's pressure, not a force
    "tank-box-sharp": "scenes/tank-box-sharp.json", #tank-box with it: an obstacle's cut faces and the surface's together
    "viscous-dam-sharp": "scenes/viscous-dam-sharp.json",       #viscous-dam with it: the viscous step between two solves with the sharp surface
    "force-volume": "scenes/force-volume.json",     #a drop in zero gravity pushed by a VDB of accelerations (needs OpenVDB): 0.3 m/s^2 along x everywhere
    "velocity-volume": "scenes/velocity-volume.json",   #and drawn to a VDB of velocities: towards 0.2 m/s along x, at 4 a second
    "linear-volume": "scenes/linear-volume.json",   #accelerations that change across the drop, each component by another axis: where the volume is, and which way round
    "patch-volume": "scenes/patch-volume.json",     #a drop crossing the part of a velocity VDB that isn't active, which does nothing, into the part that is
    "strain-volume": "scenes/strain-volume.json",   #accelerations from a staggered VDB, each component changing along its own axis, where its faces are
}
DEFAULT_SCENES = ["dam16", "tank", "swirl", "swirl-apic", "nozzle", "drain", "forces", "tank-box", "paddle", "tank-meshsphere", "tank-pulse", "mesh-sources", "tank-vdbspin", "sealed-pulse", "sealed-box", "tension-drop", "tension-sessile", "viscous-dam", "dam16-sharp", "tank-sharp", "tension-drop-sharp", "tank-box-sharp", "viscous-dam-sharp", "force-volume", "velocity-volume", "linear-volume", "patch-volume", "strain-volume"]

EXACT = ["particles", "hash", "occupiedVoxels", "perVoxel", "core", "substeps", "time", "dtMin", "dtMax", "lowest", "highest", "fastest"]
SUMS = ["kineticEnergy", "momentum", "angularMomentum", "centroid"]


def load(path):
    """the run's header and its frames, by frame number"""
    header, frames = None, {}
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            if "flip2" in record:
                header = record
            else:
                frames[record["frame"]] = record
    return header, frames


def run(binary, scene, frames, workdir, partitions, timeout, keep_frames):
    """runs a scene in workdir and returns its diagnostics"""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    spec, extra = SCENES[scene] if isinstance(SCENES[scene], tuple) else (SCENES[scene], {})
    env = dict(os.environ, FLIP2_DIAGNOSTICS="diagnostics.jsonl", FLIP2_PARTITIONS=str(partitions), **extra)
    if isinstance(spec, str):   #a scene file: flip2 bake, next to main, writes its diagnostics where the scene says, in the output directory
        command = [str(binary.parent / "flip2"), "bake", str(REPO / spec), "--out", ".", "--frames", str(frames)]
    else:
        command = [str(binary), str(frames)] + spec
    start = time.time()
    with open(workdir / "log.txt", "w") as log:
        result = subprocess.run(command, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode != 0:
        sys.exit(f"{scene}: {' '.join(command)} exited with {result.returncode}; see {workdir / 'log.txt'}")
    if not keep_frames:     #N.bin frames, and the cache flip2 bake writes
        for frame in workdir.glob("*.bin"):
            frame.unlink()
        if (workdir / "frames").exists():
            shutil.rmtree(workdir / "frames")
    print(f"  {scene} ({partitions} partition{'s' if partitions > 1 else ''}): {frames} frames in {time.time() - start:.1f} s")
    return load(workdir / "diagnostics.jsonl")


def flatten(value):
    return value if isinstance(value, list) else [value]


def close(a, b, relative):
    scale = max(abs(a), abs(b))
    return a == b or abs(a - b) <= relative*scale + 1e-300


def compare(a, b, exact, sums, relative):
    """the first difference between two runs' frames, as text, or None"""
    _, framesA = a
    _, framesB = b
    if sorted(framesA) != sorted(framesB):
        return f"different frames: {min(framesA)}..{max(framesA)} against {min(framesB)}..{max(framesB)}"
    for frame in sorted(framesA):
        recordA, recordB = framesA[frame], framesB[frame]
        for key in exact:
            if recordA[key] != recordB[key]:
                return f"frame {frame}: {key} differs: {recordA[key]} against {recordB[key]}"
        for key in sums:
            for valueA, valueB in zip(flatten(recordA[key]), flatten(recordB[key])):
                if not close(valueA, valueB, relative):
                    return f"frame {frame}: {key} differs beyond {relative:g}: {recordA[key]} against {recordB[key]}"
    return None


def percentile(histogram, fraction):
    """the particles per voxel below which fraction of the voxels the histogram counts lie"""
    total = sum(histogram)
    if total == 0:
        return 0
    running = 0
    for count, voxels in enumerate(histogram):
        running += voxels
        if running >= fraction*total:
            return count
    return len(histogram) - 1


def summary(frames):
    first, last = frames[min(frames)], frames[max(frames)]
    volume = last["occupiedVoxels"] / first["occupiedVoxels"] if first["occupiedVoxels"] else 0.0
    core = sum(last["core"])
    coreMean = sum(count*voxels for count, voxels in enumerate(last["core"])) / core if core else 0.0
    wall = sum(record["wallSeconds"] for record in frames.values())
    return (f"{last['particles']} particles, hash {last['hash']}, volume x{volume:.4f} of frame {min(frames)}, per voxel p50 {percentile(last['perVoxel'], 0.5)} "
            f"p99 {percentile(last['perVoxel'], 0.99)}, core mean {coreMean:.2f}, KE {last['kineticEnergy']:.6g}, {sum(r['substeps'] for r in frames.values())} substeps, {wall:.2f} s")


def same_platform(headerA, headerB):
    return all(headerA.get(key) == headerB.get(key) for key in ("gpu", "sm", "cudaRuntime"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["record", "check", "split", "diff"])
    parser.add_argument("files", nargs="*", help="diff: the two diagnostics files")
    parser.add_argument("--bin", default=str(REPO / "build" / "engine" / "main"), help="the main to run (default build/engine/main); flip2 is taken from beside it")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES), help=f"comma-separated, from {', '.join(SCENES)} (default {','.join(DEFAULT_SCENES)})")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--partitions", type=int, default=3, help="split: how many partitions to split into (default 3)")
    parser.add_argument("--out", default="golden-runs", help="where the runs go (default ./golden-runs)")
    parser.add_argument("--golden", default=str(REPO / "tests" / "golden"), help="the golden diagnostics (default tests/golden)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    parser.add_argument("--keep-frames", action="store_true", help="keep the runs' frames: .bin files and caches")
    args = parser.parse_args()

    if args.command == "diff":
        if len(args.files) != 2:
            sys.exit("diff takes two diagnostics files")
        a, b = load(args.files[0]), load(args.files[1])
        exact = same_platform(a[0], b[0])
        difference = compare(a, b, EXACT if exact else ["particles"], SUMS, 1e-9 if exact else 1e-3)   #the sums regroup when a run is split
        print(difference or "same")
        sys.exit(1 if difference else 0)

    scenes = args.scenes.split(",")
    for scene in scenes:
        if scene not in SCENES:
            sys.exit(f"no scene {scene}: there are {', '.join(SCENES)}")
    out, golden, binary = Path(args.out).resolve(), Path(args.golden).resolve(), Path(args.bin).resolve()
    failures = 0
    for scene in scenes:
        if args.command == "record":
            result = run(binary, scene, args.frames, out / scene, 1, args.timeout, args.keep_frames)
            golden.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / scene / "diagnostics.jsonl", golden / f"{scene}.jsonl")
            print(f"  recorded {golden / (scene + '.jsonl')}: {summary(result[1])}")
        elif args.command == "check":
            path = golden / f"{scene}.jsonl"
            if not path.exists():
                sys.exit(f"no golden diagnostics for {scene} at {path}: record them first")
            expected = load(path)
            frames = max(expected[1])
            result = run(binary, scene, frames, out / scene, 1, args.timeout, args.keep_frames)
            if same_platform(expected[0], result[0]):
                difference = compare(expected, result, EXACT, SUMS, 0.0)
            else:
                print(f"  {scene}: the golden run came from {expected[0].get('gpu')} sm {expected[0].get('sm')} CUDA {expected[0].get('cudaRuntime')}, this one from "
                      f"{result[0].get('gpu')} sm {result[0].get('sm')} CUDA {result[0].get('cudaRuntime')}: comparing the physics loosely, not the hash")
                difference = compare(expected, result, ["particles"], SUMS, 1e-3)
            failures += difference is not None
            print(f"  {scene}: {'FAIL: ' + difference if difference else 'same as golden'}")
            print(f"    {summary(result[1])}")
        else:
            whole = run(binary, scene, args.frames, out / scene, 1, args.timeout, args.keep_frames)
            split = run(binary, scene, args.frames, out / f"{scene}-{args.partitions}", args.partitions, args.timeout, args.keep_frames)
            difference = compare(whole, split, EXACT, SUMS, 1e-9)
            failures += difference is not None
            print(f"  {scene}: {'FAIL: ' + difference if difference else f'same whole and in {args.partitions} partitions'}")
            print(f"    {summary(whole[1])}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
