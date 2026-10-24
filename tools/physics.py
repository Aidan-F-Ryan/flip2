#!/usr/bin/env python3
# Copyright 2026 Taimen LLC
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
import os
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
       "With no gravity and nothing to push it, a ball of liquid keeps the velocity it has and its shape, and moving changes nothing of what it does: "
       "each of its particles' velocities strays from their mean as the same particle's does in the same ball at rest.",
       "Nothing acts on it but the density correction, which evens out the seeding's jitter in the ball at rest alike (0.0010 m/s of spread by "
       "a quarter of a second, settling as the rest check's tank does), so there is nothing to allow for: the limits are a thousandth of its "
       "speed, and a voxel on its width.",
       known="Traced and mended 8 October, but for the density correction. P2G gave the faces at the edge of its reach wrong velocities (its sums "
             "rounded each particle's share of the weight and of the momentum on their own, and divided by the weight plus 1e-7), and the "
             "extrapolation after it left a handful of unreached faces at rest each substep; the pressure solve took both for divergence, and the "
             "ball lost a thousandth of its speed a substep. The sums are exact now and the extrapolation reaches every face: with the density "
             "correction off the ball keeps its velocity to 2e-8 of its speed over 12 substeps, whatever its size, and moving changes nothing. "
             "What's left is the correction. Until 24 October it measured each voxel's count of particles, which rose and fell by a particle as "
             "the ball's particles crossed the voxels in step: 0.4% of the ball's speed in spread. It measures their B-spline density now, which "
             "slides smoothly with them, and the check holds the moving ball to the still one: moving changes its particles' velocities by "
             "0.0005 to 0.0010 m/s (0.09 to 0.16% of its speed). The B-spline density of jittered particles changes as they slide across the "
             "voxel centres, so the moving ball's correction isn't the still one's: most at frame 3, half way through the 6 frames in which the "
             "ball moves a whole number of voxels along each axis and comes back to its starting place on the grid. Next for it: a correction "
             "that restores each particle's seeded volume rather than one density everywhere.")
def momentum(bake):
    start = [0.5, 0.3, 0.2]
    speed = math.sqrt(sum(v * v for v in start))

    def ball(name, velocity):
        return bake(name, scene([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 0.025, 6, [{"shape": "sphere", "center": [-0.1, -0.05, -0.05], "radius": 0.1, "velocity": velocity}],
                                gravity=[0.0, 0.0, 0.0]), cache=True)

    def strays(run, frame):
        """each particle's velocity less the ball's mean, by its id"""
        values, _ = flip2cache.read_frame(str(run.directory), frame, ("v", "id"))
        v, ids = values["v"], values["id"]
        vx, vy, vz, numbers = (v[:, 0], v[:, 1], v[:, 2], ids[:, 0]) if hasattr(v, "shape") else (v[0], v[1], v[2], ids[0])
        n = len(vx)
        mean = (sum(vx) / n, sum(vy) / n, sum(vz) / n)
        return {int(numbers[i]): (vx[i] - mean[0], vy[i] - mean[1], vz[i] - mean[2]) for i in range(n)}

    run, still = ball("momentum", start), ball("momentum-still", [0.0, 0.0, 0.0])
    last = run.frames[-1]
    velocity = run.velocity(last)
    off = math.sqrt(sum((v - s) ** 2 for v, s in zip(velocity, start)))
    changed, worst = 0.0, last     #the most, over the frames, that moving changes the particles' velocities: rms over them, each against itself at rest
    for frame in run.frames[1:]:
        moving, resting = strays(run, frame), strays(still, frame)
        rms = math.sqrt(sum(sum((a - b) ** 2 for a, b in zip(moving[k], resting[k])) for k in moving) / len(moving))
        if rms > changed:
            changed, worst = rms, frame
    row = run.rows[last]
    widths = [row["highest"][axis] - row["lowest"][axis] for axis in range(3)]
    return [Result("change in its velocity after %.2f s" % run.time(last), off, 0.0, 1.0e-3 * speed, "m/s", "most"),
            Result("most moving changes its particles' velocities (frame %d)" % worst, changed, 0.0, 1.0e-3 * speed, "m/s", "most"),
            Result("widest it gets", max(widths), 0.2, 0.025, "m", "absolute")]


# ---- at rest, and energy ----

@check("rest",
       "A tank of liquid at rest stays at rest.",
       "The pressure holds gravity exactly on the faces; what moves is the particles' jitter settling under the density correction. The limits "
       "are a hundredth of sqrt(g dx), the speed gravity gives over a voxel, and two hundredths of a voxel on the centre of mass: with ghost "
       "fluid's surface, the default since 19 October, the settling moves it up to 0.015 of a voxel in the first second, a different way for "
       "each seed (5 seeds measured, 19 October), and lifts it about 0.005 as the liquid evens out. Marker cells held it within 0.01.")
def rest(bake):
    voxel = 0.025
    run = bake("rest", scene([-0.5, 0.0, -0.25], [0.5, 0.6, 0.25], voxel, 24, [box([-0.5, 0.0, -0.25], [0.5, 0.3, 0.25])]))
    start = run.rows[0]["centroid"]
    moved = max(math.sqrt(sum((run.rows[f]["centroid"][axis] - start[axis]) ** 2 for axis in range(3))) for f in run.frames)
    rms = max(math.sqrt(2.0 * run.rows[f]["kineticEnergy"] / run.rows[f]["particles"]) for f in run.frames if f >= 12)
    return [Result("rms speed in its second half second", rms, 0.0, 0.01 * math.sqrt(G * voxel), "m/s", "most"),
            Result("furthest its centre of mass moves", moved, 0.0, 0.02 * voxel, "m", "most")]


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
       "Measured 8 October with the level set free surface (then opt-in, the default since 19 October) at 12 voxels' depth: the period within 1%, "
       "1.5% of the swing lost each half period. 2% and 3% allowed.")
def slosh(bake):
    measured, expected, count, kept = slosh_run(bake, "slosh", None)
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.02, "s"),
            Result("swing kept each half period", kept, 1.0, 0.03, "", "absolute")]


@check("slosh-voxel",
       "The same tank with the voxel free surface, the opt-in.",
       "The voxel free surface is first order in the voxel: the pressure is 0 a voxel or so outside the particles, and the surface's height is only known to "
       "a voxel. Measured 8 October at 12 voxels' depth: the period 8% long, 17% of the swing lost each half period (3.5% and 3.5% at 24 voxels), "
       "and a wave under a voxel high dies within two swings and leaves the surface tilted. 12% and 20% allowed.")
def slosh_voxel(bake):
    measured, expected, count, kept = slosh_run(bake, "slosh-voxel", {"freeSurface": "voxel"})
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.12, "s"),
            Result("swing kept each half period", kept, 1.0, 0.20, "", "absolute")]


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
       "Measured when the level set free surface was built (3 October): its frequency 2% low at 16 voxels' radius. 4% allowed.")
def drop_ghost(bake):
    measured, expected, count = drop(bake, "drop", None)
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.04, "s")]


@check("drop-voxel",
       "The same drop with the voxel free surface.",
       "Measured when surface tension was built (2 October): its frequency 9% low at 16 voxels' radius, so its period 10% long, first order in "
       "the voxel: the voxel free surface sits outside the liquid and adds to the mass that swings. 12% allowed.")
def drop_voxel(bake):
    measured, expected, count = drop(bake, "drop-voxel", {"freeSurface": "voxel"})
    return [Result("period, over %d half swings" % max(count - 1, 0), measured, expected, 0.12, "s")]


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

def slab(bake, name, low, high, frames, walls=None, solver=None):
    """a slab of water at rest in a 1 m box, 2.5 cm voxels, between two heights and out to +-high[0] across: its fall by frame"""
    made = scene([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5], 0.025, frames, [box([-high[0], low, -high[0]], [high[0], high[1], high[0]])])
    if walls:
        made["domain"]["walls"] = walls
    if solver:
        made["solver"] = solver
    return bake(name, made)


def behind(run, frame):
    """how far its mean speed is behind free fall's at a frame, m/s"""
    return G * run.time(frame) + run.velocity(frame)[1]


@check("near-ceiling",
       "Liquid that isn't touching a wall doesn't feel it: a slab at rest two voxels under the ceiling falls at g.",
       "As free-fall's.")
def near_ceiling(bake):
    run = slab(bake, "near-ceiling", 0.75, (0.25, 0.95), 4)
    last = run.frames[-1]
    t = run.time(last)
    return [Result("fall speed after %.2f s" % t, -run.velocity(last)[1], G * t, 0.005, "m/s")]


@check("near-ceiling-voxel",
       "The same slab with the voxel free surface.",
       "As free-fall's.",
       known="The voxel free surface's liquid is the particles' footprint, a voxel or two wider than the particles, and a wall that holds pulls on whatever "
             "touches it as well as pushing: so it holds back liquid up to two voxels off it. A wall that lets go doesn't (near-ceiling-lets-go), "
             "and neither does the level set's surface, which is the particles' own (near-ceiling).")
def near_ceiling_voxel(bake):
    run = slab(bake, "near-ceiling-voxel", 0.75, (0.25, 0.95), 4, solver={"freeSurface": "voxel"})
    last = run.frames[-1]
    t = run.time(last)
    return [Result("fall speed after %.2f s" % t, -run.velocity(last)[1], G * t, 0.005, "m/s")]


@check("near-ceiling-lets-go",
       "The same slab under a ceiling that lets go.",
       "As free-fall's. Measured 8 October: 0.4% slow after a third of a second, all of it lost in the first substep (free-fall's own is 0.2%).")
def near_ceiling_lets_go(bake):
    run = slab(bake, "near-ceiling-lets-go", 0.75, (0.25, 0.95), 8, {"+y": {"hold": False}})
    last = run.frames[-1]
    t = run.time(last)
    return [Result("fall speed after %.2f s" % t, -run.velocity(last)[1], G * t, 0.005, "m/s")]


@check("lets-go",
       "Liquid on a wall that lets go comes away from it where air can get in behind it: a slab on the ceiling, its edges free, falls.",
       "The pressure lets it go at once, and the wall's faces where it has let go carry the liquid's own velocity on for the particles that read "
       "them, so its top layer leaves with the rest. Measured 8 October: 0.002 m/s behind free fall after a third of a second, 0.01 allowed, and "
       "its top no higher than a voxel over where free fall has it. (0.27 m/s behind while the particles still read the wall's faces as at rest; "
       "on a ceiling that holds, 0.79 m/s behind and still on it.)")
def lets_go(bake):
    run = slab(bake, "lets-go", 0.8, (0.25, 1.0), 8, {"+y": {"hold": False}})
    last = run.frames[-1]
    t = run.time(last)
    return [Result("behind free fall after %.2f s" % t, behind(run, last), 0.0, 0.01, "m/s", "most"),
            Result("its top, as a height", run.rows[last]["highest"][1], 0.0, 1.0 - 0.5 * G * t * t + 0.025, "m", "most")]


@check("sealed",
       "A wall lets go only where air can get in. A slab filling the top of a box, under a ceiling that lets go, stays up while the walls round "
       "it hold, as water does in an upturned glass: nothing can get in behind it. Where the walls let go too, air comes up them, and it falls.",
       "Held, nothing moves but the particles' settling, as rest's: two hundredths of a voxel allowed in a quarter of a second (0.010 measured "
       "with the level set surface, 19 October). (Its flat underside is unstable, and gives way after "
       "about a second: no check of that.) Let go, it's free fall as lets-go's is: 0.007 m/s behind measured, 0.02 allowed; and the half substep "
       "free-fall's distance is ahead by.")
def sealed(bake):
    held = slab(bake, "sealed", 0.8, (0.5, 1.0), 6, {"+y": {"hold": False}})
    freed = slab(bake, "sealed-let-go", 0.8, (0.5, 1.0), 6, {"hold": False})
    last = held.frames[-1]
    t = held.time(last)
    fallen = lambda run: run.rows[0]["centroid"][1] - run.rows[last]["centroid"][1]
    return [Result("walls holding, fallen after %.2f s" % t, fallen(held), 0.0, 0.02 * 0.025, "m", "most"),
            Result("walls letting go, behind free fall", behind(freed, last), 0.0, 0.02, "m/s", "most"),
            Result("walls letting go, short of free fall's distance", 0.5 * G * t * t - fallen(freed), 0.0, 0.03, "m", "most")]


@check("rest-lets-go",
       "Walls that let go still hold up what rests on them: rest's tank, with every wall letting go, stays at rest.",
       "As rest's: nothing pulls on a wall there, so nothing is let go.")
def rest_lets_go(bake):
    voxel = 0.025
    made = scene([-0.5, 0.0, -0.25], [0.5, 0.6, 0.25], voxel, 24, [box([-0.5, 0.0, -0.25], [0.5, 0.3, 0.25])])
    made["domain"]["walls"] = {"hold": False}
    run = bake("rest-lets-go", made)
    start = run.rows[0]["centroid"]
    moved = max(math.sqrt(sum((run.rows[f]["centroid"][axis] - start[axis]) ** 2 for axis in range(3))) for f in run.frames)
    rms = max(math.sqrt(2.0 * run.rows[f]["kineticEnergy"] / run.rows[f]["particles"]) for f in run.frames if f >= 12)
    return [Result("rms speed in its second half second", rms, 0.0, 0.01 * math.sqrt(G * voxel), "m/s", "most"),
            Result("furthest its centre of mass moves", moved, 0.0, 0.02 * voxel, "m", "most")]


@check("collider-lets-go",
       "A collider that lets go is left as a wall is: a slab at rest against its underside falls at g, whether that's flat on the voxels' planes, "
       "flat and off them, tilted, or round. One that holds keeps it.",
       "Measured 8 October, after a sixth of a second (free fall is 1.63 m/s): 0.008 m/s behind on the planes, 0.003 off them, 0.014 tilted 30 "
       "degrees and 0.036 under a ball; 0.03 allowed flat or tilted and 0.05 round, where the faces the surface cuts still blend the liquid's "
       "velocity with the collider's. Off the planes with the slab's edges on the nodes' planes is as off them anywhere: 0.02 between the two. "
       "(Before the faces of a collider that has let go carried the liquid's velocity: 0.25 behind on the planes; and off them it never let go, "
       "the air's way in past the liquid's edge counting as solid.) Held, its top stays within half a voxel of the face.")
def collider_lets_go(bake):
    voxel, frames = 0.025, 4
    def fall(name, obstacle, fluid, hold=False):
        made = scene([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5], voxel, frames, [fluid], obstacles=[dict(obstacle, hold=hold)])
        return bake("collider-" + name, made)
    def flat(name, face, across, hold=False):
        return fall(name, box([-0.6, face, -0.6], [0.6, 1.1, 0.6]), box([-across, face - 0.2, -across], [across, face, across]), hold)
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    tilted = dict(box([-1.0, 0.0, -1.0], [1.0, 0.6, 1.0]), transform=[c, -s, 0, 0, s, c, 0, 0.7, 0, 0, 1, 0, 0, 0, 0, 1])
    runs = [("flat, on the voxels' planes", flat("on", 0.8, 0.25), 0.03),
            ("flat, off them", flat("off", 0.81, 0.25), 0.03),
            ("flat, off them, its edges on the nodes' planes", flat("off-nodes", 0.81, 0.2), 0.03),
            ("tilted 30 degrees", fall("tilted", tilted, box([-0.2, 0.35, -0.2], [0.2, 0.95, 0.2])), 0.03),
            ("round", fall("round", {"shape": "sphere", "center": [0.0, 0.9, 0.0], "radius": 0.3}, box([-0.2, 0.4, -0.2], [0.2, 0.75, 0.2])), 0.05)]
    held = flat("held", 0.81, 0.25, True)
    t = held.time(frames)
    results = [Result("%s: behind free fall after %.2f s" % (what, t), abs(behind(run, frames)), 0.0, allowed, "m/s", "most") for what, run, allowed in runs]
    results.append(Result("off the planes, edges on the nodes' or not", abs(behind(runs[1][1], frames) - behind(runs[2][1], frames)), 0.0, 0.02, "m/s", "most"))
    results.append(Result("held, its top under the face", 0.81 - held.rows[frames]["highest"][1], 0.0, 0.5 * voxel, "m", "most"))
    return results


def differing(one, other):
    """how many of two bakes' frames differ in their state's hash"""
    return sum(1 for frame in one.frames if frame not in other.rows or one.rows[frame]["hash"] != other.rows[frame]["hash"]) + abs(len(one.frames) - len(other.frames))


@check("wall-settings",
       "A wall's own friction and contact angle are that wall's alone. A drop on a floor given 120 degrees, under a liquid whose angle is 60, "
       "is the drop of a liquid whose angle is 120; and friction on walls the liquid never comes near changes nothing.",
       "Exact: the bakes are the same to the bit.")
def wall_settings(bake):
    def drop(name, liquid_angle, walls):
        made = scene([-0.2, 0.0, -0.2], [0.2, 0.2, 0.2], 0.00625, 6, [{"shape": "sphere", "center": [0.0, 0.0, 0.0], "radius": 0.1}], gravity=[0.0, 0.0, 0.0],
                     liquid={"density": 1000.0, "surfaceTension": 10.0, "contactAngle": liquid_angle})
        if walls:
            made["domain"]["walls"] = walls
        return bake(name, made)
    def slide(name, walls):
        made = scene([-2.0, 0.0, -0.1], [2.0, 0.4, 0.1], 0.025, 6, [box([-1.8, 0.0, -0.1], [-1.3, 0.1, 0.1], [2.0, 0.0, 0.0])])
        if walls:
            made["domain"]["walls"] = walls
        return bake(name, made)
    beads = drop("wall-settings-120", 120.0, None)
    floor = drop("wall-settings-floor", 60.0, {"-y": {"contactAngle": 120.0}, "+x": {"contactAngle": 20.0}})
    wets = drop("wall-settings-60", 60.0, None)
    slips = slide("wall-settings-slip", None)
    far = slide("wall-settings-far", {"+y": {"friction": 1.0}, "+x": {"friction": 1.0}})
    rough = slide("wall-settings-rough", {"-y": {"friction": 1.0}})
    return [Result("frames that differ, the floor's own angle against the liquid's", differing(floor, beads), 0, 0, "", "most"),
            Result("frames the same at 60 and 120 degrees (so the angle is read)", len(wets.frames) - differing(wets, beads), 0, 1, "", "most"),
            Result("frames that differ with friction on walls it doesn't touch", differing(far, slips), 0, 0, "", "most"),
            Result("speed kept on a floor with friction, over one without", rough.velocity(rough.frames[-1])[0] / slips.velocity(slips.frames[-1])[0], 0.0, 0.99, "", "most")]


# ---- dynamic bodies (two-way coupled: bodies.cu, contacts.cu) ----

TANK, DEEP, RADIUS = 0.5, 0.25, 0.08    #bodies.py's tank, 0.5 m across with water a quarter of a metre deep, and its ball
LIQUID = 1000.0                         #the scenes' liquid density, kg/m^3 (scene.hpp's default)


def tank(frames, voxel=1.0 / 128, top=TANK, depth=DEEP, fps=24, **more):
    """the tank, top high, with water depth deep (0: dry, but for a voxel of it in a corner, as bodies.py's pool=0)"""
    water = box([0.0, 0.0, 0.0], [TANK, depth, TANK]) if depth > 0.0 else box([0.0, 0.0, 0.0], [voxel, voxel, voxel])
    return scene([0.0, 0.0, 0.0], [TANK, top, TANK], voxel, frames, [water], fps=fps, **more)


def body(shape, place, density, turn=0.0, **dynamic):
    """a dynamic body: an obstacle's shape, in its own space, put at place and turned turn degrees about y"""
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    return dict(shape, transform=[c, 0, s, place[0], 0, 1, 0, place[1], -s, 0, c, place[2], 0, 0, 0, 1], dynamic=dict(dynamic, density=density))


def ball(height, density, **dynamic):
    """the ball, in the middle of the tank with its centre height high"""
    return body({"shape": "sphere", "centre": [0, 0, 0], "radius": RADIUS}, [0.5 * TANK, height, 0.5 * TANK], density, **dynamic)


def stack(size, voxel):
    """three boxes of side size over the middle of the tank, a voxel apart, the lowest's centre 0.125 m up, each turned 7 degrees more than the one under it"""
    return [body(box([-0.5 * size] * 3, [0.5 * size] * 3), [0.5 * TANK, 0.125 + i * (size + voxel), 0.5 * TANK], 2000.0, turn=7.0 * i) for i in range(3)]


def track(run, what="position", axis=1, which=0, start=0):
    """one number of a body by frame, from frame start: (time, value); axis None for its length"""
    def value(f):
        v = run.rows[f]["bodies"][which][what]
        return math.sqrt(sum(x * x for x in v)) if axis is None else v[axis]
    return [(run.time(f), value(f)) for f in run.frames if f >= start]


def parabola(points):
    """least squares y = a + b t + c t^2 through (t, y) points: the fit, and its slope, as functions of t"""
    t0 = points[0][0]
    rows = [[0.0] * 4 for _ in range(3)]    #the normal equations, with their right hand side
    for t, y in points:
        powers = [1.0, t - t0, (t - t0) ** 2]
        for i in range(3):
            for j in range(3):
                rows[i][j] += powers[i] * powers[j]
            rows[i][3] += powers[i] * y
    for i in range(3):      #Gauss-Jordan
        rows[i] = [value / rows[i][i] for value in rows[i]]
        for k in range(3):
            if k != i:
                rows[k] = [value - rows[k][i] * mine for value, mine in zip(rows[k], rows[i])]
    a, b, c = (rows[i][3] for i in range(3))
    return (lambda t: a + b * (t - t0) + c * (t - t0) ** 2), (lambda t: b + 2.0 * c * (t - t0))


def first_bounce(run, rest, step):
    """A body dropped from rest onto a floor, its first hit, which the frames miss: parabolas through its free flight before and after it meet where
    it turned round. The bodies move by semi-implicit Euler, each collision step (step long) adding its gravity and then moving, so a parabola
    through the positions has the steps' mean velocities, half a step's gravity off what each step moves at: the contact solve saw the body closing
    at the fit's speed in plus that, and left it at the fit's speed out less it. Those two speeds, and how high over rest its centre turned round"""
    heights, speeds = track(run), track(run, "velocity")
    ups = [i for i in range(1, len(speeds)) if speeds[i][1] > 0.0 > speeds[i - 1][1]]     #the frames it's going up again after coming down
    hit, again = ups[0], ups[1] if len(ups) > 1 else len(heights)
    before, falling = parabola(heights[:hit])
    after, rising = parabola(heights[hit:again])
    turn = heights[hit][0]
    for _ in range(50):     #where the two meet
        turn -= (before(turn) - after(turn)) / (falling(turn) - rising(turn))
    half = 0.5 * G * step
    return -falling(turn) + half, rising(turn) - half, before(turn) - rest


def cap(fraction):
    """how deep the ball sits holding fraction of its volume under the water: the cap height h with pi h^2 (3 r - h) / 3 = fraction 4/3 pi r^3"""
    low, high = 0.0, 2.0 * RADIUS
    for _ in range(60):
        middle = 0.5 * (low + high)
        if math.pi * middle * middle * (3.0 * RADIUS - middle) / 3.0 < fraction * 4.0 / 3.0 * math.pi * RADIUS ** 3:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def most_work(start, level, density):
    """the most the water's push, less its weight, can give a ball of density rising from start: as if the water's level stayed at level, which
    it only falls from as the ball leaves it, so more than it really gives; the most where push and weight balance"""
    volume = 4.0 / 3.0 * math.pi * RADIUS ** 3
    def under(y):
        h = min(max(level - (y - RADIUS), 0.0), 2.0 * RADIUS)
        return math.pi * h * h * (3.0 * RADIUS - h) / 3.0
    work, most, y, dy = 0.0, 0.0, start, 1.0e-4
    while y < level + RADIUS:
        work += (LIQUID * under(y + 0.5 * dy) - density * volume) * G * dy
        most = max(most, work)
        y += dy
    return most


def bodies_differing(one, other):
    """how many of two bakes' frames differ in their bodies' records"""
    return sum(1 for frame in one.frames if frame not in other.rows or one.rows[frame].get("bodies") != other.rows[frame].get("bodies"))


@check("body-float",
       "A ball of half the liquid's density, let go with its centre on the waterline, floats there (Archimedes) and stays still.",
       "The plan's test for the coupling: its centre within half a voxel of the waterline. Measured 24 October: 0.0002 of a voxel off, and 0.38 of a"
       " voxel between its highest and its lowest over its last 2 s, the particles' settling rocking it (0.09 on 18 October: why it has grown isn't "
       "known). Half a voxel allowed for both.")
def body_float(bake):
    voxel = 1.0 / 128
    run = bake("body-float", tank(96, obstacles=[ball(DEEP, 0.5 * LIQUID)]))
    late = [y for t, y in track(run, start=48)]
    return [Result("its centre over the last 2 s, as a height", sum(late) / len(late), DEEP, 0.5 * voxel, "m", "absolute"),
            Result("its highest less its lowest in them", max(late) - min(late), 0.0, 0.5 * voxel, "m", "most")]


@check("body-heave",
       "A ball of half the liquid's density dropped onto the water bobs at its heave period, T = 2 pi sqrt((m + a) / (rho g pi r^2)): its mass "
       "and the water's that moves with it (added mass, a, half of what it displaces) on the stiffness of its waterline.",
       "The plan's test: within 10%. The added mass changes with the frequency (Hulme 1982: 0.83 of what a floating hemisphere displaces bobbing "
       "slowly, 0.5 bobbing fast, which is 0.63 s and 0.57 s here), and the 0.5 m tank's second sloshing mode, at 0.57 s, keeps it bobbing. Measured"
       " 24 October: 0.606 s over 9 half swings, 6.8% long (0.58 to 0.60 s on 18 October).")
def body_heave(bake):
    run = bake("body-heave", tank(96, obstacles=[ball(DEEP + 2.5 * RADIUS, 0.5 * LIQUID)]))
    late = [y for t, y in track(run, start=48)]
    middle = sum(late) / len(late)
    measured, found = period([(t, y - middle) for t, y in track(run, start=24)])
    mass = 0.5 * LIQUID * 4.0 / 3.0 * math.pi * RADIUS ** 3
    expected = 2.0 * math.pi * math.sqrt(1.5 * mass / (LIQUID * G * math.pi * RADIUS ** 2))
    return [Result("period, over %d half swings" % max(len(found) - 1, 0), measured, expected, 0.10, "s")]


@check("body-light",
       "A ball a twentieth of the liquid's density let go under water rises, leaves it, falls back and floats, and nothing blows up: it's never "
       "faster than all the work the water's push could do on it would make it, it hardly turns (pressure pushes through a ball's centre, so it "
       "can't turn it), and it ends floating where Archimedes has it, the water lower by the volume it no longer displaces.",
       "Its speed's bound is energy's: all the work the water's push less its weight could do on it rising from where it starts, the water's level "
       "held where it starts and none of the water moving with it, 6.7 m/s; measured 24 October, 2.0 m/s as it leaves the water. Its turning is "
       "what's left of the coupling's torque leak, 0.022 rad/s at most measured 24 October (0.014 to 0.038 on 19 October); 0.1 allowed. A ceiling at"
       " 0.5 m, which it would hit, rolls it (0.2 to 0.8 rad/s), so the tank's a metre high. Over its last second, after falling back in and bobbing"
       " for 3 s, its centre was 0.06 of a voxel under Archimedes' height (24 October); two voxels allowed.")
def body_light(bake):
    voxel, density, start = 1.0 / 128, 0.05 * LIQUID, 0.5 * DEEP
    run = bake("body-light", tank(96, top=1.0, obstacles=[ball(start, density)]))  #a metre high: it leaves the water at 2 m/s, and a ceiling at 0.5 m rolls it
    volume = 4.0 / 3.0 * math.pi * RADIUS ** 3
    level = DEEP - volume * (1.0 - density / LIQUID) / TANK ** 2    #the water seeded round the ball, which then displaces only part of its volume
    floats = level + RADIUS - cap(density / LIQUID)
    fastest = math.sqrt(2.0 * most_work(start, DEEP, density) / (density * volume))
    last = [y for t, y in track(run, start=72)]
    return [Result("fastest it goes", max(v for t, v in track(run, "velocity", None)), 0.0, fastest, "m/s", "most"),
            Result("fastest it turns", max(w for t, w in track(run, "angularVelocity", None)), 0.0, 0.1, "rad/s", "most"),
            Result("its centre over the last second, as a height", sum(last) / len(last), floats, 2.0 * voxel, "m", "absolute")]


@check("body-bounce",
       "A body dropped on a floor leaves it at the bounce Bullet gives, the body's times the floor's, which are RBD Bullet Solver's unless given: "
       "a piece's 0.25 times a floor's 0.5, on the domain's floor and on a collider's top alike. It turns round on the floor or over it, not in it.",
       "The contact solve gives the product in floats: 0.125 to 5 digits on both floors, measured 24 October; 0.2% allowed. It bounces at the start "
       "of the collision step in which it would reach the floor, so it turns round up to a step's move over it (1.5 mm, 0.94 of a step's move, from "
       "this drop; 0.06 to 1.6 mm over a frame's worth of drops, 24 October), and it may sink 0.05 of a voxel, the push out's slop, before it's "
       "pushed out: those are the limits. (Read from the frames' speeds instead, gravity has taken a different amount since the hit on each floor, "
       "and the two once looked 4% apart.)")
def body_bounce(bake):
    voxel, fps, substeps, drop = 1.0 / 128, 240, 5, 0.2
    step = 1.0 / (fps * substeps)       #fast enough that the half voxel floor on a step's move adds none
    results = []
    for name, where, top, floors in (("wall", "the domain's floor", 0.0, []), ("collider", "a collider", 0.02, [box([0.1, 0.0, 0.1], [0.4, 0.02, 0.4])])):
        run = bake("body-bounce-" + name, tank(90, depth=0.0, fps=fps, obstacles=[ball(top + RADIUS + drop, 2.0 * LIQUID, bounce=0.25)] + floors,
                                                solver={"collisionSubsteps": substeps}))
        closing, leaving, turned = first_bounce(run, top + RADIUS, step)
        results += [Result("%s: the bounce its contact gave" % where, leaving / closing, 0.125, 0.002),
                    Result("%s: how far it had sunk in at the turn" % where, max(-turned, 0.0), 0.0, 0.05 * voxel, "m", "most"),
                    Result("%s: how far over it it turned, in steps' moves" % where, turned / (closing * step), 0.0, 1.0, "", "most")]
    return results


@check("body-stack",
       "Boxes dropped one over another onto a dry floor come to rest in a stack and stay there.",
       "Measured 24 October, from 2 s on: its top box moves 0.53 mm, and the boxes are still (0.04 mm/s at most) but for one jolt at 3.2 s, 7.8 mm/s"
       " at most for three frames, the contacts' doing and not looked into. On 18 October's engine they were still settling at 1.5 s (15.6 mm/s). 1 "
       "mm and 1 cm/s allowed.")
def body_stack(bake):
    run = bake("body-stack", tank(96, depth=0.0, obstacles=stack(0.12, 1.0 / 128)))
    top = lambda frame: run.rows[frame]["bodies"][2]["position"]
    last = run.frames[-1]
    moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(top(48), top(last))))
    fastest = max(v for which in range(3) for t, v in track(run, "velocity", None, which, 48))
    return [Result("how far its top box moves in its last 2 s", moved, 0.0, 0.001, "m", "most"),
            Result("fastest any box goes in them", fastest, 0.0, 0.01, "m/s", "most")]


@check("body-resume",
       "A bake carried on from a checkpoint is the bake that never stopped, to the bit, with bodies resting on each other and on the floor under "
       "water, whose contacts the checkpoint carries over: in as many partitions as it stopped in, or in more.",
       "Exact. (Until 24 October a resumed bake's first contact steps were dealt from keys the islands' sort never sorted: restoring the "
       "contacts made room for them but not the sort's scratch, so where they filled every slot the first substep needed, it never got any.)")
def body_resume(bake):
    made = tank(48, voxel=1.0 / 64, obstacles=stack(0.12, 1.0 / 64))
    straight = bake("body-resume", made, cache=True)
    results = []
    for partitions in (1, 3):
        name = "body-resume-%d" % partitions
        bake(name, made, cache=True, until=24)
        resumed = bake(name, None, until=48, partitions=partitions)
        into = "in %d partition%s" % (partitions, "s" if partitions > 1 else "")
        results += [Result("resumed at frame 24 %s: frames that differ" % into, differing(straight, resumed), 0, 0, "", "most"),
                    Result("and frames whose bodies differ", bodies_differing(straight, resumed), 0, 0, "", "most")]
    return results


@check("body-spin",
       "With no gravity, a ball spun in still water keeps its spin and leaves the water still: pressure pushes square to the surface, through a "
       "ball's centre, so it can't turn it, and the water slips past it freely.",
       "Against the same bake with the ball still, whose water settles as this one's does: the spun ball's water gains 4.5e-4 of the ball's angular "
       "momentum in 2 s and the ball keeps all but 0.013% of its spin, measured 24 October (4.2e-4 and 0.014% on 18 October's engine, when free slip"
       " was built): the coupling's torque leak, which hasn't been traced. A thousandth and 0.1% allowed.")
def body_spin(bake):
    voxel, depth, spin = 1.0 / 128, 0.4, 5.0
    def spun(name, rate):
        return bake(name, tank(48, depth=depth, gravity=[0.0, 0.0, 0.0], obstacles=[ball(0.5 * depth, LIQUID, angularVelocity=[0.0, 0.0, rate])]))
    run, still = spun("body-spin", spin), spun("body-spin-still", 0.0)
    last = run.frames[-1]
    own = 0.4 * LIQUID * 4.0 / 3.0 * math.pi * RADIUS ** 5 * spin     #2/5 m r^2 w
    particle = LIQUID * voxel ** 3 / 8.0    #the diagnostics' angular momentum is the particles' r x v, 8 of them a voxel
    water = max(abs(run.rows[f]["angularMomentum"][2] - still.rows[f]["angularMomentum"][2]) * particle for f in run.frames)
    return [Result("its spin after %.0f s" % run.time(last), run.rows[last]["bodies"][0]["angularVelocity"][2], spin, 0.001, "rad/s"),
            Result("angular momentum it gives the water, over its own", water / own, 0.0, 0.001, "", "most")]


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

    def bake(name, made, cache=False, until=None, partitions=None):
        """bakes a scene into out/name, stopping after frame until if given; with made None, carries name's bake on from its last checkpoint (a
        cache's, which it writes at its end) to frame until instead (flip2 resume); partitions splits the domain between that many"""
        directory = out / name
        path = out / (name + ".json")
        if made is None:
            made = json.loads(path.read_text())
            command = [str(binary), "resume", str(directory), "--frames", str(until)]
        else:
            if directory.exists():
                shutil.rmtree(directory)
            made = dict(made)
            made["output"] = {"dir": str(directory), "cache": cache, "compression": "none", "checkpoints": {"every": 0}, "positions": False, "diagnostics": "diagnostics.jsonl"}
            with open(path, "w") as file:
                json.dump(made, file)
            command = [str(binary), "bake", str(path), "--overwrite"] + (["--frames", str(until)] if until else [])
        start = time.time()
        with open(out / (name + ".log"), "a" if command[1] == "resume" else "w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                    env=dict(os.environ, FLIP2_PARTITIONS=str(partitions)) if partitions else None)
        if result.returncode != 0:
            raise RuntimeError("%s %s %s exited with %d: see %s" % (binary, command[1], command[2], result.returncode, out / (name + ".log")))
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
