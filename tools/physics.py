#!/usr/bin/env python3
# Copyright 2023 Aberrant Behavior LLC
"""Physics checks for flip2. golden.py says a build still gives what it gave when its goldens were recorded; it can't say that was right. Each check here
bakes a small scene whose answer is known beforehand, from a conservation law, an exact solution or a published result, and holds the bake to it.

  physics.py [options]        bake every check's scene and compare
  physics.py --list           what each check holds the engine to, where its tolerance comes from, and which are known to fail

A tolerance is the error the method is known to have at the check's resolution, and each check says where its own comes from: an integrator's bound, or
the accuracy measured when the feature was validated. A check marked known fails for a reason that's understood and written beside it: it's reported,
and doesn't fail the run until the day it passes, when its mark has to come off. Only Python's standard library: coeus has no numpy.
"""
import argparse
import json
import math
import operator
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import flip2cache   # noqa: E402

G = 9.8     #the scenes' gravity, m/s^2, down y (scene.hpp's default)
CHECKS = []


class Run:
    """a baked scene: its diagnostics by frame, and its cache if it wrote one"""
    def __init__(self, directory, scene, seconds):
        self.directory, self.scene, self.seconds = directory, scene, seconds
        self.rows = {}
        with open(directory / "diagnostics.jsonl") as lines:
            for line in lines:
                record = json.loads(line)
                if "frame" in record:
                    self.rows[record["frame"]] = record
        self.frames = sorted(self.rows)

    def time(self, frame):
        return self.rows[frame]["time"]

    def velocity(self, frame):
        """the particles' mean velocity: the momentum per unit mass over their count"""
        row = self.rows[frame]
        return [m / row["particles"] for m in row["momentum"]]

    def longest_substep(self, frame):
        return max(self.rows[f]["dtMax"] for f in self.frames if 0 < f <= frame)

    def positions(self, frame):
        """the particles' x, y and z at a frame, from the cache"""
        values, _ = flip2cache.read_frame(str(self.directory), frame, ("P",))
        return values["P"]


class Result:
    """one number of a check against what it should be: within allowed of expected, as a fraction of it (relative), outright (absolute), or just no
    more than allowed (most)"""
    def __init__(self, what, measured, expected, allowed, unit="", how="relative"):
        self.what, self.measured, self.expected, self.allowed, self.unit, self.how = what, measured, expected, allowed, unit, how

    def passed(self):
        if self.how == "most":
            return self.measured <= self.allowed
        off = abs(self.measured - self.expected)
        return off <= (self.allowed * abs(self.expected) if self.how == "relative" else self.allowed)

    def text(self):
        if self.how == "most":
            return "%-44s %11.5g %s, at most %.5g allowed" % (self.what, self.measured, self.unit, self.allowed)
        if self.how == "relative":
            off = 100.0 * (self.measured / self.expected - 1.0) if self.expected else float("inf")
            return "%-44s %11.5g %s against %.5g: %+.2f%%, %.3g%% allowed" % (self.what, self.measured, self.unit, self.expected, off, 100.0 * self.allowed)
        return "%-44s %11.5g %s against %.5g: off by %.3g, %.3g allowed" % (self.what, self.measured, self.unit, self.expected, abs(self.measured - self.expected), self.allowed)


def check(name, law, tolerance, known=None):
    """registers a check: what it holds the engine to, where its tolerances come from, and if it's known to fail, why"""
    def register(function):
        CHECKS.append({"name": name, "law": law, "tolerance": tolerance, "known": known, "run": function})
        return function
    return register


def scene(low, high, voxel, frames, fluids, fps=24, **more):
    out = {"schema": "flip2.scene/1", "fps": fps, "frames": frames, "domain": {"min": low, "max": high, "voxelSize": voxel}, "fluids": fluids}
    out.update(more)
    return out


def box(low, high, velocity=None):
    out = {"shape": "box", "min": low, "max": high}
    if velocity:
        out["velocity"] = velocity
    return out


def crossings(series):
    """where a series of (time, value) changes sign, by a straight line between the samples either side"""
    found = []
    for (t0, s0), (t1, s1) in zip(series, series[1:]):
        if (s0 > 0.0) != (s1 > 0.0):
            found.append(t0 + (t1 - t0) * s0 / (s0 - s1))
    return found


def period(series):
    """the period of a series that swings about 0: twice the mean time between its sign changes"""
    found = crossings(series)
    if len(found) < 3:
        return float("nan"), found
    return 2.0 * (found[-1] - found[0]) / (len(found) - 1), found


# ---- a body touching nothing ----

@check("free-fall",
       "A body of liquid touching nothing falls at g.",
       "Gravity is added to the grid and comes back to the particles whole, so its speed is held to half a percent. Positions move by the velocity "
       "at each substep's end, which puts it up to g dt t / 2 further down than the exact fall: half a substep ahead.")
def free_fall(bake):
    run = bake("free-fall", scene([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5], 0.025, 6, [box([-0.15, 0.5, -0.15], [0.15, 0.8, 0.15])]))
    last = run.frames[-1]
    t = run.time(last)
    velocity = run.velocity(last)
    fallen = run.rows[0]["centroid"][1] - run.rows[last]["centroid"][1]
    return [Result("fall speed after %.2f s" % t, -velocity[1], G * t, 0.005, "m/s"),
            Result("distance fallen", fallen, 0.5 * G * t * t, 0.5 * G * run.longest_substep(last) * t + 0.002, "m", "absolute")]


@check("momentum",
       "With no gravity and nothing to push it, a ball of liquid keeps the velocity it has, all of it together, and its shape.",
       "Nothing acts on it, so there is nothing to allow for: the limits are a thousandth of its speed, and a voxel on its width.",
       known="Its mean velocity changes by about 0.13% of its speed every substep, and its particles' velocities spread by as much: with FLIP, PIC or "
             "APIC transfers alike, with the density correction off, and at half the voxel size. The same on the 4 October build. Not traced further yet.")
def momentum(bake):
    start = [0.5, 0.3, 0.2]
    speed = math.sqrt(sum(v * v for v in start))
    run = bake("momentum", scene([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 0.025, 6, [{"shape": "sphere", "center": [-0.1, -0.05, -0.05], "radius": 0.1, "velocity": start}],
                              gravity=[0.0, 0.0, 0.0]))
    last = run.frames[-1]
    velocity = run.velocity(last)
    off = math.sqrt(sum((v - s) ** 2 for v, s in zip(velocity, start)))
    row = run.rows[last]
    spread = math.sqrt(max(2.0 * row["kineticEnergy"] / row["particles"] - sum(v * v for v in velocity), 0.0))    #how far the particles' velocities are from their mean, rms
    widths = [row["highest"][axis] - row["lowest"][axis] for axis in range(3)]
    return [Result("change in its velocity after %.2f s" % run.time(last), off, 0.0, 1.0e-3 * speed, "m/s", "most"),
            Result("spread of its particles' velocities", spread, 0.0, 1.0e-3 * speed, "m/s", "most"),
            Result("widest it gets", max(widths), 0.2, 0.025, "m", "absolute")]


# ---- at rest, and energy ----

@check("rest",
       "A tank of liquid at rest stays at rest.",
       "The pressure holds gravity exactly on the faces; what moves is the particles' jitter settling under the density correction. The limits "
       "are a hundredth of sqrt(g dx), the speed gravity gives over a voxel, and a hundredth of a voxel on the centre of mass.")
def rest(bake):
    voxel = 0.025
    run = bake("rest", scene([-0.5, 0.0, -0.25], [0.5, 0.6, 0.25], voxel, 24, [box([-0.5, 0.0, -0.25], [0.5, 0.3, 0.25])]))
    start = run.rows[0]["centroid"]
    moved = max(math.sqrt(sum((run.rows[f]["centroid"][axis] - start[axis]) ** 2 for axis in range(3))) for f in run.frames)
    rms = max(math.sqrt(2.0 * run.rows[f]["kineticEnergy"] / run.rows[f]["particles"]) for f in run.frames if f >= 12)
    return [Result("rms speed in its second half second", rms, 0.0, 0.01 * math.sqrt(G * voxel), "m/s", "most"),
            Result("furthest its centre of mass moves", moved, 0.0, 0.01 * voxel, "m", "most")]


@check("energy",
       "Nothing makes energy or liquid: a dam break's kinetic and potential energy together never exceed what it started with, and it ends "
       "with the particles it began with.",
       "One-sided, so there's no error to allow for but the stepping's: a thousandth of the energy it starts with.")
def energy(bake):
    run = bake("energy", scene([0.0, 0.0, 0.0], [1.0, 1.0, 0.5], 0.025, 48, [box([0.0, 0.0, 0.0], [0.3, 0.6, 0.5])]))
    def total(frame):
        row = run.rows[frame]
        return row["kineticEnergy"] / row["particles"] + G * row["centroid"][1]
    start = total(0)
    most = max(total(f) for f in run.frames)
    least = min(total(f) for f in run.frames)
    return [Result("most energy it ever has, over the start's", most / start, 1.0, 1.001, "", "most"),
            Result("particles at the end, over the start's", run.rows[run.frames[-1]]["particles"] / run.rows[0]["particles"], 1.0, 0.0, "", "absolute"),
            Result("energy left after 2 s, over the start's", least / start, 0.0, 1.0, "", "most")]


@check("emitter",
       "An emitter of area A pouring at speed v adds A v of liquid a second.",
       "The emitter's conveyor makes exactly that; a hundredth is for the frames' ends not falling on substeps of it.")
def emitter(bake):
    voxel, per = 0.0078125, 8
    side, speed = 0.0625, 1.0
    run = bake("emitter", scene([-0.25, 0.0, -0.25], [0.25, 0.5, 0.25], voxel, 24, [],
                                emitters=[{"shape": "box", "min": [-0.5 * side, 0.40625, -0.5 * side], "max": [0.5 * side, 0.4375, 0.5 * side], "velocity": [0.0, -speed, 0.0]}]))
    first, last = 6, run.frames[-1]
    gained = (run.rows[last]["particles"] - run.rows[first]["particles"]) / (run.time(last) - run.time(first))
    return [Result("particles a second", gained, side * side * speed * per / voxel ** 3, 0.01, "")]


# ---- waves ----

def slosh_run(bake, name, solver):
    length, depth, push = 1.0, 0.3, 0.3
    made = scene([-0.5, 0.0, -0.25], [0.5, 0.6, 0.25], 0.025, 96, [box([-0.5, 0.0, -0.25], [0.5, depth, 0.25], [push, 0.0, 0.0])])
    if solver:
        made["solver"] = solver
    run = bake(name, made)
    start = run.rows[0]["centroid"][0]
    series = [(run.time(f), run.rows[f]["centroid"][0] - start) for f in run.frames if f > 0]
    measured, found = period(series)
    peaks = [max(abs(value) for t, value in series if a <= t <= b) for a, b in zip(found, found[1:])]     #the furthest it gets in each half swing
    kept = (peaks[-1] / peaks[0]) ** (1.0 / (len(peaks) - 1)) if len(peaks) > 1 else float("nan")
    k = math.pi / length
    return measured, 2.0 * math.pi / math.sqrt(G * k * math.tanh(k * depth)), len(found), kept


@check("slosh",
       "A tank's water, given a push, swings end to end at the period of its first standing wave, omega^2 = g k tanh(k h) with k = pi / L, and "
       "with nothing to slow it keeps swinging.",
       "The footprint free surface is first order in the voxel: the pressure is 0 a voxel or so outside the particles, and the surface's height "
       "is only known to a voxel. Measured 8 October at 12 voxels' depth: the period 8% long, 17% of the swing lost each half period (3.5% and "
       "3.5% at 24 voxels), and a wave under a voxel high dies within two swings and leaves the surface tilted. 12% and 20% allowed.")
def slosh(bake):
    measured, expected, count, kept = slosh_run(bake, "slosh", None)
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.12, "s"),
            Result("swing kept each half period", kept, 1.0, 0.20, "", "absolute")]


@check("slosh-sharp",
       "The same tank with the sharp free surface.",
       "Measured 8 October at 12 voxels' depth: the period within 1%, 1.5% of the swing lost each half period. 2% and 3% allowed.")
def slosh_sharp(bake):
    measured, expected, count, kept = slosh_run(bake, "slosh-sharp", {"freeSurface": "sharp"})
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.02, "s"),
            Result("swing kept each half period", kept, 1.0, 0.03, "", "absolute")]


# ---- surface tension ----

def swing(run, every=1):
    """a drop's swing between long and flat along y, from its particles' second moments (an ellipsoid's semi-axis is sqrt(5) times the rms distance
    from its centre along it): (time, how much longer the y axis is than the mean of the other two, over the radius), and the radius"""
    series, radius = [], None
    for frame in run.frames[::every]:
        axes = []
        for values in run.positions(frame):
            n = len(values)
            mean = sum(values) / n
            axes.append(math.sqrt(5.0 * max(sum(map(operator.mul, values, values)) / n - mean * mean, 0.0)))
        if radius is None:
            radius = (axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0)
        series.append((run.time(frame), (axes[1] - 0.5 * (axes[0] + axes[2])) / radius))
    return series, radius


def drop(bake, name, solver):
    tension, density = 10.0, 1000.0
    stretched = [0.095, 0, 0, 0, 0, 0.11, 0, 0, 0, 0, 0.095, 0, 0, 0, 0, 1]
    made = scene([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2], 0.00625, 54, [{"mesh": str(REPO / "scenes" / "geo" / "sphere.obj"), "transform": stretched}], fps=48,
                 gravity=[0.0, 0.0, 0.0], liquid={"density": density, "surfaceTension": tension})
    if solver:
        made["solver"] = solver
    run = bake(name, made, cache=True)
    series, radius = swing(run)
    measured, found = period(series)
    return measured, 2.0 * math.pi / math.sqrt(8.0 * tension / (density * radius ** 3)), len(found)


@check("drop",
       "A drop pulled out of round, in no gravity, swings at Rayleigh's frequency: omega^2 = 8 sigma / (rho R^3).",
       "Measured when surface tension was built (2 October): its frequency 9% low at 16 voxels' radius, so its period 10% long, first order in "
       "the voxel: the footprint free surface sits outside the liquid and adds to the mass that swings. 12% allowed.")
def drop_footprint(bake):
    measured, expected, count = drop(bake, "drop", None)
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.12, "s")]


@check("drop-sharp",
       "The same drop with the sharp free surface.",
       "Measured when the sharp surface was built (3 October): its frequency 2% low at 16 voxels' radius. 4% allowed.")
def drop_sharp(bake):
    measured, expected, count = drop(bake, "drop-sharp", {"freeSurface": "sharp"})
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.04, "s")]


# ---- viscosity ----

@check("viscous-front",
       "A dam break of thick liquid spreads as Huppert's viscous current (1982): its front at x = 1.411 (g q^3 / 3 nu)^(1/5) t^(1/5), with q the "
       "liquid's volume per unit width.",
       "Measured when viscosity was built (2 October): within 2% at 1, 2 and 4 s; on 8 October 3% at 4 s. 4% allowed. The law holds once the "
       "current is long and thin, so not before 1 s.")
def viscous_front(bake):
    viscosity, density = 10.0, 1000.0
    run = bake("viscous-front", scene([0.0, 0.0, 0.0], [0.8, 0.1, 0.2], 0.0025, 96, [box([0.0, 0.0, 0.0], [0.1, 0.08, 0.2])], liquid={"density": density, "viscosity": viscosity}))
    k = 1.411 * (G * (0.1 * 0.08) ** 3 / (3.0 * viscosity / density)) ** 0.2
    return [Result("front at %d s" % (frame // 24), run.rows[frame]["highest"][0], k * (frame / 24.0) ** 0.2, 0.04, "m") for frame in (24, 48, 96)]


# ---- walls ----

@check("near-ceiling",
       "Liquid that isn't touching a wall doesn't feel it: a slab at rest two voxels under the ceiling falls at g.",
       "As free-fall's.",
       known="The pressure solve's liquid is the particles' footprint, a voxel or two wider than the particles, and every wall holds what touches it, "
             "pulling as well as pushing: so a wall holds back liquid up to two voxels off it. Waiting on walls that can let go.")
def near_ceiling(bake):
    run = bake("near-ceiling", scene([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5], 0.025, 4, [box([-0.25, 0.75, -0.25], [0.25, 0.95, 0.25])]))
    last = run.frames[-1]
    t = run.time(last)
    return [Result("fall speed after %.2f s" % t, -run.velocity(last)[1], G * t, 0.005, "m/s")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", default=str(REPO / "build" / "engine" / "flip2"), help="the flip2 to run (default build/engine/flip2)")
    parser.add_argument("--only", default="", help="comma-separated checks to run (default all)")
    parser.add_argument("--out", default="physics-runs", help="where the bakes go (default ./physics-runs)")
    parser.add_argument("--keep", action="store_true", help="keep the bakes: by default each is deleted once it's been measured")
    parser.add_argument("--list", action="store_true", help="describe the checks and run nothing")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per bake")
    args = parser.parse_args()
    names = [name for name in args.only.split(",") if name]
    for name in names:
        if name not in [c["name"] for c in CHECKS]:
            sys.exit("no check %s: there are %s" % (name, ", ".join(c["name"] for c in CHECKS)))
    chosen = [c for c in CHECKS if not names or c["name"] in names]
    if args.list:
        for c in chosen:
            print("%s\n    %s\n    Tolerance: %s%s" % (c["name"], c["law"], c["tolerance"], "\n    Known to fail: " + c["known"] if c["known"] else ""))
        return
    binary, out = Path(args.bin).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def bake(name, made, cache=False):
        directory = out / name
        if directory.exists():
            shutil.rmtree(directory)
        made = dict(made)
        made["output"] = {"dir": str(directory), "cache": cache, "compression": "none", "checkpoints": {"every": 0}, "positions": False, "diagnostics": "diagnostics.jsonl"}
        path = out / (name + ".json")
        with open(path, "w") as file:
            json.dump(made, file)
        start = time.time()
        with open(out / (name + ".log"), "w") as log:
            result = subprocess.run([str(binary), "bake", str(path), "--overwrite"], stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        if result.returncode != 0:
            raise RuntimeError("%s bake %s exited with %d: see %s" % (binary, path, result.returncode, out / (name + ".log")))
        return Run(directory, made, time.time() - start)

    failed, known, mended = [], [], []
    began = time.time()
    for c in chosen:
        start = time.time()
        try:
            results = c["run"](bake)
            broken = None
        except Exception as error:      #a bake that died, or numbers that aren't there: the check fails, and the rest still run
            results, broken = [], str(error)
        passed = broken is None and all(r.passed() for r in results)
        verdict = "ok" if passed and not c["known"] else "KNOWN TO FAIL" if not passed and c["known"] else "PASSES NOW: take its known mark off" if c["known"] else "FAILED"
        print("%-14s %s (%.1f s)" % (c["name"], verdict, time.time() - start))
        for r in results:
            print("    %s  %s" % ("ok  " if r.passed() else "FAIL", r.text()))
        if broken:
            print("    " + broken)
        if not passed and c["known"]:
            print("    known: " + c["known"])
            known.append(c["name"])
        elif not passed:
            failed.append(c["name"])
        elif c["known"]:
            mended.append(c["name"])
        if not args.keep:
            for path in out.glob(c["name"] + "*"):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    print("%d checks in %.0f s: %d ok, %d failed%s%s%s" % (len(chosen), time.time() - began, len(chosen) - len(failed) - len(known) - len(mended), len(failed),
          ": " + ", ".join(failed) if failed else "", ", %d known to fail: %s" % (len(known), ", ".join(known)) if known else "",
          ", %d marked known now pass: %s" % (len(mended), ", ".join(mended)) if mended else ""))
    if not args.keep and not any(out.iterdir()):
        out.rmdir()
    sys.exit(1 if failed or mended else 0)


if __name__ == "__main__":
    main()
