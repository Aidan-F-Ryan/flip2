"""flip2 Solver: a flip2 simulation set up from Houdini geometry, baked by the flip2 program, and loaded back, as File Cache loads what it wrote.

Its inputs are geometry, as Houdini's own FLIP Solver takes it:

    1 Fluid        closed surfaces: the fluid at the start frame
    2 Collisions   closed surfaces the fluid flows around (open ones too, given a thickness): still, moving or deforming
    3 Sources      closed surfaces kept full of fluid moving at the emission velocity (or their v attribute's mean)
    4 Sinks        closed surfaces that remove the fluid inside them

Each input is one object, or with a "name" primitive attribute, one per name (flip2 takes up to 16 collisions). Packed geometry is unpacked, every
primitive turned into polygons and the polygons into triangles. An object that moves or deforms over the frame range is sampled every frame, which
flip2 re-voxelizes between, so its point count and triangles mustn't change.

Everything goes in the output directory ($HIP/geo/<scene>.<node> by default, beside File Cache's caches): scene.json and geo/ (what the node writes), bake/ (flip2's cache, which a
running bake commits frames to, and export/houdini/ beside it) and logs/. The bake's frame 0 is the start frame, and the node shows it there: its
surface, its particles or both (Show).

The surface is meshed on the GPU as each frame is committed (flip2 mesh, beside flip2 export), with the Surface tab's settings: closed quads facing
outwards, their points carrying v for motion blur. Mesh Bake meshes a bake's frames again, with other settings or after a bake made without them.

Bake On picks where flip2 runs: this machine, or another over ssh (a GPU box, for a Mac). A remote bake sends scene.json and geo/ to the remote
directory, runs there detached (remote/job.sh), and is mirrored back every couple of seconds (remote/mirror.sh): its logs, its cache.json and its
exported and meshed frames, so it shows here as a local one does. It needs ssh to reach the host without a password (a key), and rsync at both ends.

Particle separation and grid scale are Houdini's: flip2's voxels are twice the particle separation, the 2 x 2 x 2 particles per voxel Houdini's FLIP
seeds at its default grid scale of 2.
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

import hou

from . import importer

ROLES = ("fluid", "collision", "source", "sink")
LABELS = ("Fluid", "Collisions", "Sources", "Sinks")
FACES = ("-x", "+x", "-y", "+y", "-z", "+z")
CORNERS = """//each triangle's points, which the scene's triangles index; anything not a triangle is left out
int corners = primvertexcount(0, @primnum);
i@flip2_a = corners == 3 ? primpoint(0, @primnum, 0) : -1;
i@flip2_b = corners == 3 ? primpoint(0, @primnum, 1) : -1;
i@flip2_c = corners == 3 ? primpoint(0, @primnum, 2) : -1;
"""
_jobs = {}      #per node path, its bake's processes, while this session runs them
GEO_FILE = re.compile(r"^(%s)\d+_(triangles|vertices|f-?\d+)\.npy$" % "|".join(ROLES))     #the files write_scene writes into geo/


def _callback(function):
    return dict(script_callback="__import__('flip2houdini').solver.%s(kwargs['node'])" % function, script_callback_language=hou.scriptLanguage.Python)


def _interface(node):
    """the node's parameters, from the subnet's own on: applied again to an existing node (update), it keeps the values of the parameters it still has"""
    group = node.type().parmTemplateGroup()
    simulation = hou.FolderParmTemplate("simulation", "Simulation", folder_type=hou.folderType.Tabs)
    simulation.addParmTemplate(hou.FloatParmTemplate("particlesep", "Particle Separation", 1, default_value=(0.02,), min=0.0001,
                                                     help="The distance between particles at rest. flip2's voxels are twice this"))
    simulation.addParmTemplate(hou.FloatParmTemplate("domaincenter", "Domain Center", 3, default_value=(0.0, 0.5, 0.0)))
    simulation.addParmTemplate(hou.FloatParmTemplate("domainsize", "Domain Size", 3, default_value=(2.0, 1.0, 2.0)))
    simulation.addParmTemplate(hou.ButtonParmTemplate("fitdomain", "Fit Domain to Inputs", help="The domain around every input's geometry over the frame range",
                                                      **_callback("fit_domain")))
    for index, face in enumerate(FACES):
        simulation.addParmTemplate(hou.ToggleParmTemplate("closed%d" % index, "Closed %s" % face.upper(), default_value=True,
                                                          join_with_next=index % 2 == 0,
                                                          help="A closed side is a wall; an open one removes the fluid that reaches it"))
    simulation.addParmTemplate(hou.IntParmTemplate("startframe", "Start Frame", 1, default_expression=("$FSTART",)))
    simulation.addParmTemplate(hou.IntParmTemplate("endframe", "End Frame", 1, default_expression=("$FEND",)))
    simulation.addParmTemplate(hou.MenuParmTemplate("transfer", "Velocity Transfer", ("flip", "apic"), ("FLIP (Splashy)", "APIC (Swirly)"), default_value=0))
    simulation.addParmTemplate(hou.FloatParmTemplate("flipratio", "FLIP Ratio", 1, default_value=(0.95,), min=0.0, max=1.0,
                                                     help="How much of each particle's own velocity it keeps: 0 is pure PIC (or pure APIC), 1 pure FLIP"))
    simulation.addParmTemplate(hou.FloatParmTemplate("cfl", "CFL Condition", 1, default_value=(4.0,), min=0.1, max=4.0,
                                                     help="How many voxels the fastest particle may move in a substep"))
    simulation.addParmTemplate(hou.FloatParmTemplate("gravity", "Gravity", 3, default_value=(0.0, -9.8, 0.0)))
    simulation.addParmTemplate(hou.FloatParmTemplate("densitytime", "Density Correction Time", 1, default_value=(0.1,), min=0.0,
                                                     help="Seconds over which crowded or sparse fluid is brought back to its rest density; 0 for none"))
    group.append(simulation)
    collisions = hou.FolderParmTemplate("collisions", "Collisions", folder_type=hou.folderType.Tabs)
    collisions.addParmTemplate(hou.FloatParmTemplate("friction", "Friction", 1, default_value=(0.0,), min=0.0, max=1.0,
                                                     help="0: the fluid slips along collisions freely; 1: the fluid touching them moves with them"))
    collisions.addParmTemplate(hou.FloatParmTemplate("thickness", "Thickness", 1, default_value=(0.0,), min=0.0,
                                                     help="0 for closed surfaces; for open ones, like a ground plane, the thickness of the shell around them"))
    group.append(collisions)
    sources = hou.FolderParmTemplate("sources", "Sources", folder_type=hou.folderType.Tabs)
    sources.addParmTemplate(hou.ToggleParmTemplate("usev", "Use v Attribute", default_value=True,
                                                   help="Each source emits at its points' mean v, if it has a v attribute; otherwise at the emission velocity"))
    sources.addParmTemplate(hou.FloatParmTemplate("emitvel", "Emission Velocity", 3, default_value=(0.0, 0.0, 0.0)))
    group.append(sources)
    surface = hou.FolderParmTemplate("surfacefolder", "Surface", folder_type=hou.folderType.Tabs)
    surface.addParmTemplate(hou.ToggleParmTemplate("meshsurface", "Mesh the Surface", default_value=True,
                                                   help="Mesh the liquid's surface on the GPU as the bake commits each frame (flip2 mesh): closed, facing "
                                                        "outwards, with v for motion blur"))
    surface.addParmTemplate(hou.FloatParmTemplate("voxelscale", "Voxel Scale", 1, default_value=(0.5,), min=0.1, max=2.0,
                                                  help="The surface's sample spacing, in particle separations: smaller is finer, slower and larger on disk"))
    surface.addParmTemplate(hou.FloatParmTemplate("influencescale", "Influence Scale", 1, default_value=(2.0,), min=0.5, max=4.0,
                                                  help="How far each particle reaches into the surface, in particle separations: larger is smoother, and "
                                                       "fills gaps between particles further apart. At most 8 times the voxel scale"))
    surface.addParmTemplate(hou.FloatParmTemplate("radiusscale", "Radius Scale", 1, default_value=(0.6,), min=0.1, max=2.0,
                                                  help="Each particle's radius, in particle separations: where the surface sits around them. 0.6 keeps "
                                                       "the liquid's volume. Less than the influence scale"))
    surface.addParmTemplate(hou.IntParmTemplate("smoothing", "Smoothing", 1, default_value=(2,), min=0, max=10,
                                                help="Passes of a smoothing filter over the surface before it's meshed"))
    surface.addParmTemplate(hou.ButtonParmTemplate("meshbake", "Mesh Bake", help="Mesh the bake's frames again with these settings, replacing their "
                                                   "surfaces: after a bake made without them, or to try others", **_callback("mesh_bake")))
    group.append(surface)
    bake = hou.FolderParmTemplate("bakefolder", "Bake", folder_type=hou.folderType.Tabs)
    bake.addParmTemplate(hou.StringParmTemplate("outputdir", "Output Directory", 1, default_value=("$HIP/geo/$HIPNAME.$OS",),
                                                string_type=hou.stringParmType.FileReference, file_type=hou.fileType.Directory))
    bake.addParmTemplate(hou.MenuParmTemplate("bakeon", "Bake On", ("local", "remote"), ("This Machine", "Remote Machine (ssh)"), default_value=0,
                                              help="Where flip2 runs: here, or on another machine over ssh, its frames brought back as they're exported"))
    bake.addParmTemplate(hou.StringParmTemplate("program", "flip2 Program", 1, default_value=("",),
                                                help="The flip2 executable on the machine that bakes: here, empty for $FLIP2 or flip2 on the PATH; on a remote "
                                                     "machine, as its shell reads it (~ is its home), empty for flip2 on its PATH"))
    bake.addParmTemplate(hou.StringParmTemplate("remotehost", "Remote Host", 1, default_value=("",), disable_when="{ bakeon == 0 }",
                                                help="The machine to bake on, as ssh knows it (a host, user@host, or a Host in ~/.ssh/config)"))
    bake.addParmTemplate(hou.StringParmTemplate("remotedir", "Remote Directory", 1, default_value=("~/flip2-bakes",), disable_when="{ bakeon == 0 }",
                                                help="Where the remote machine keeps bakes: each node's goes in its own folder there"))
    bake.addParmTemplate(hou.IntParmTemplate("gpus", "GPUs", 1, default_value=(0,), min=0, help="How many GPUs to use; 0 for all of them"))
    bake.addParmTemplate(hou.IntParmTemplate("partitions", "Partitions", 1, default_value=(1,), min=1,
                                             help="How many pieces to split the domain into, across the GPUs"))
    bake.addParmTemplate(hou.IntParmTemplate("checkpoints", "Checkpoint Every", 1, default_value=(10,), min=0,
                                             help="Frames between checkpoints, which a cancelled bake resumes from; 0 for one only on cancel and at the end"))
    bake.addParmTemplate(hou.MenuParmTemplate("compression", "Compression", ("zstd", "lz4", "none"), ("Zstd", "LZ4", "None"), default_value=0))
    bake.addParmTemplate(hou.ButtonParmTemplate("writescene", "Write Scene", help="Write scene.json and geo/ without baking", **_callback("write_scene")))
    bake.addParmTemplate(hou.ButtonParmTemplate("bake", "Bake", join_with_next=True, help="Write the scene and bake it from the start", **_callback("bake")))
    bake.addParmTemplate(hou.ButtonParmTemplate("resume", "Resume", join_with_next=True, help="Carry a cancelled bake on from its newest checkpoint",
                                                **_callback("resume")))
    bake.addParmTemplate(hou.ButtonParmTemplate("cancel", "Cancel", help="Stop the bake after the frame it's on, checkpointed", **_callback("cancel")))
    bake.addParmTemplate(hou.ButtonParmTemplate("deleteremote", "Delete Remote Copy", disable_when="{ bakeon == 0 }",
                                                help="Delete this node's bake on the remote machine. The frames already here stay, but it can't be resumed after",
                                                **_callback("delete_remote")))
    status = hou.StringParmTemplate("status", "Status", 1)
    status.setDisableWhen("{ cfl >= 0 }")       #always: it's only to read
    bake.addParmTemplate(status)
    bake.addParmTemplate(hou.ToggleParmTemplate("load", "Load from Disk", default_value=True, help="Show the bake's frames, as they're made"))
    bake.addParmTemplate(importer.show_parm(disable_when="{ load == 0 }"))
    group.append(bake)
    node.setParmTemplateGroup(group)


def _network(node):
    """the node's loaders: the bake's frame showing at this time, its frame 0 on the start frame"""
    return importer.build_loaders(node, '`chs("../outputdir")`/bake/export/houdini', 'round(($T - (ch("../startframe") - 1)/$FPS)*$FPS)')


def create_solver(parent, name="flip2_solver"):
    """a flip2 Solver node in the SOP network parent"""
    node = parent.createNode("subnet", name)
    node.setUserData("flip2", "solver")
    _interface(node)
    for index, role in enumerate(ROLES):     #each input as triangles, with each triangle's points
        unpack = node.createNode("unpack", role + "_unpack")
        unpack.setInput(0, node.indirectInputs()[index])
        unpack.parm("limit_iterations").set(False)
        polygons = node.createNode("convert", role + "_polygons")
        polygons.setInput(0, unpack)
        triangles = node.createNode("divide", role + "_triangles")
        triangles.setInput(0, polygons)
        corners = node.createNode("attribwrangle", role + "_corners")
        corners.parm("class").set(1)
        corners.parm("snippet").set(CORNERS)
        corners.setInput(0, triangles)
        out = node.createNode("null", role.upper())
        out.setInput(0, corners)
    loaded = node.createNode("switch", "load")     #input 0 left empty: nothing loaded
    loaded.setInput(1, _network(node))
    loaded.parm("input").setExpression('ch("../load")')
    output = node.createNode("output", "output0")
    output.setInput(0, loaded)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)
    node.layoutChildren()
    return node


# ---- the scene ----

def _frames(node):
    start, end = node.evalParm("startframe"), node.evalParm("endframe")
    if end < start:
        raise hou.NodeError("the end frame is before the start frame")
    return start, end


def _objects(geometry):
    """an input's triangles at a frame, per object: (name, triangles as point numbers), and its points' positions"""
    import numpy
    positions = numpy.frombuffer(geometry.pointFloatAttribValuesAsString("P"), dtype=numpy.float32).reshape(-1, 3)
    if geometry.findPrimAttrib("flip2_a") is None or len(geometry.prims()) == 0:
        return [], positions
    corners = [numpy.frombuffer(geometry.primIntAttribValuesAsString("flip2_" + corner), dtype=numpy.int32) for corner in "abc"]
    triangles = numpy.stack(corners, axis=1)
    keep = triangles[:, 0] >= 0
    names = geometry.findPrimAttrib("name")
    if names is None or names.dataType() != hou.attribData.String:
        return [("", triangles[keep])], positions
    labels = numpy.array(geometry.primStringAttribValues("name"), dtype=object)
    objects = []
    for label in sorted(set(labels[keep])):
        objects.append((label, triangles[keep & (labels == label)]))
    return objects, positions


def _connected(node, index):
    inputs = node.inputs()
    return index < len(inputs) and inputs[index] is not None


def _sample(node, role, frames, progress=None):
    """an input's objects over frames: per object its name, triangles (into its own vertices) and vertex positions per frame; progress() after each"""
    import numpy
    source = node.node(role.upper())
    first, positions = _objects(source.geometryAtFrame(frames[0]))
    if progress is not None:
        progress()
    samples = []
    for name, triangles in first:
        used, local = numpy.unique(triangles, return_inverse=True)
        samples.append({"name": name, "triangles": local.reshape(-1, 3).astype(numpy.int32), "points": used, "global": triangles,
                        "frames": [positions[used].copy()]})
    for frame in frames[1:]:
        objects, positions = _objects(source.geometryAtFrame(frame))
        same = len(objects) == len(samples) and all(name == sample["name"] and numpy.array_equal(triangles, sample["global"])
                                                    for sample, (name, triangles) in zip(samples, objects))
        if not same:
            raise hou.NodeError("%s: its triangles change at frame %d; flip2 can only move or deform a mesh whose points and triangles stay the same"
                                % (LABELS[ROLES.index(role)], frame))
        for sample in samples:
            sample["frames"].append(positions[sample["points"]].copy())
        if progress is not None:
            progress()
    return samples


def _write_object(samples, prefix, geo_dir, frames, fps):
    """an object's files, and its scene entry: still, or deforming with a sample every frame"""
    import numpy
    numpy.save(os.path.join(geo_dir, prefix + "_triangles.npy"), samples["triangles"])
    still = all(numpy.array_equal(positions, samples["frames"][0]) for positions in samples["frames"][1:])
    if still:
        numpy.save(os.path.join(geo_dir, prefix + "_vertices.npy"), samples["frames"][0])
        return {"vertices": "geo/%s_vertices.npy" % prefix, "triangles": "geo/%s_triangles.npy" % prefix}
    deforming = []
    for frame, positions in zip(frames, samples["frames"]):
        name = "%s_f%d.npy" % (prefix, frame)
        numpy.save(os.path.join(geo_dir, name), positions)
        deforming.append({"time": (frame - frames[0]) / fps, "vertices": "geo/" + name})
    return {"vertices": deforming[0]["vertices"], "triangles": "geo/%s_triangles.npy" % prefix, "deforming": deforming}


def _output_directory(node):
    directory = node.evalParm("outputdir").rstrip("/")
    if not directory:
        raise hou.NodeError("no output directory")
    return directory


def write_scene(node):
    """writes the node's scene: scene.json and geo/ in its output directory. Returns scene.json's path"""
    import numpy
    start, end = _frames(node)
    frames = list(range(start, end + 1))
    fps = hou.fps()
    directory = _output_directory(node)
    geo_dir = os.path.join(directory, "geo")
    os.makedirs(geo_dir, exist_ok=True)
    for name in os.listdir(geo_dir):    #what an earlier scene wrote, and nothing else
        if GEO_FILE.match(name):
            os.remove(os.path.join(geo_dir, name))
    separation = node.evalParm("particlesep")
    center, size = node.evalParmTuple("domaincenter"), node.evalParmTuple("domainsize")
    scene = {
        "schema": "flip2.scene/1", "fps": fps, "frames": end - start,
        "domain": {"min": [c - s/2 for c, s in zip(center, size)], "max": [c + s/2 for c, s in zip(center, size)], "voxelSize": 2*separation,
                   "open": [face for index, face in enumerate(FACES) if not node.evalParm("closed%d" % index)]},
        "solver": {"flipRatio": node.evalParm("flipratio"), "cfl": node.evalParm("cfl"), "densityCorrectionTime": node.evalParm("densitytime"),
                   "transfer": ("flip", "apic")[node.evalParm("transfer")]},
        "gravity": list(node.evalParmTuple("gravity")),
        "fluids": [], "emitters": [], "sinks": [], "obstacles": [],
        "partitions": node.evalParm("partitions"), "devices": node.evalParm("gpus"),
        "output": {"dir": "bake", "compression": ("zstd", "lz4", "none")[node.evalParm("compression")],
                   "checkpoints": {"every": node.evalParm("checkpoints"), "keep": 2}},
    }
    roles = [role for role in ROLES if _connected(node, ROLES.index(role))]
    total = sum(1 if role == "fluid" else len(frames) for role in roles)
    sampled = [0]
    with hou.InterruptableOperation("flip2: sampling the inputs", long_operation_name="Writing the flip2 scene", open_interrupt_dialog=True) as operation:
        def progress():     #a progress bar, which Esc interrupts
            sampled[0] += 1
            operation.updateLongProgress(sampled[0] / max(total, 1), "sampled %d of %d frames of the inputs" % (sampled[0], total))
        sampled_roles = [(role, _sample(node, role, frames[:1] if role == "fluid" else frames, progress)) for role in roles]
    for role, objects in sampled_roles:
        for index, samples in enumerate(objects):
            entry = _write_object(samples, "%s%d" % (role, index), geo_dir, frames[:len(samples["frames"])], fps)
            if role == "fluid":
                entry = {key: entry[key] for key in ("vertices", "triangles")}
                scene["fluids"].append(entry)
            elif role == "collision":
                entry["friction"] = node.evalParm("friction")
                entry["thickness"] = node.evalParm("thickness")
                scene["obstacles"].append(entry)
            elif role == "source":
                entry["velocity"] = _emission_velocity(node, samples)
                scene["emitters"].append(entry)
            else:
                scene["sinks"].append(entry)
    if len(scene["obstacles"]) > 16:
        raise hou.NodeError("flip2 takes up to 16 collision objects; this has %d (one per name)" % len(scene["obstacles"]))
    path = os.path.join(directory, "scene.json")
    with open(path, "w") as out:
        json.dump(scene, out, indent=1)
    node.parm("status").set("wrote %s" % path)
    return path


def _emission_velocity(node, samples):
    velocity = list(node.evalParmTuple("emitvel"))
    if node.evalParm("usev"):
        import numpy
        geometry = node.node("SOURCE").geometryAtFrame(node.evalParm("startframe"))
        if geometry.findPointAttrib("v") is not None:
            v = numpy.frombuffer(geometry.pointFloatAttribValuesAsString("v"), dtype=numpy.float32).reshape(-1, 3)[samples["points"]]
            velocity = [float(component) for component in v.mean(axis=0)]
    return velocity


def fit_domain(node):
    """sets the domain to the box around every input's geometry over the frame range, a little larger"""
    start, end = _frames(node)
    low, high = [float("inf")]*3, [float("-inf")]*3
    for role in ROLES:
        source = node.node(role.upper())
        for frame in ([start] if role == "fluid" else range(start, end + 1)):
            geometry = source.geometryAtFrame(frame)
            box = geometry.boundingBox() if geometry is not None else None
            if box is not None and box.isValid():
                low = [min(a, b) for a, b in zip(low, box.minvec())]
                high = [max(a, b) for a, b in zip(high, box.maxvec())]
    if low[0] > high[0]:
        raise hou.NodeError("no input geometry to fit the domain to")
    margin = 4*node.evalParm("particlesep")
    node.parmTuple("domaincenter").set([(a + b)/2 for a, b in zip(low, high)])
    node.parmTuple("domainsize").set([b - a + 2*margin for a, b in zip(low, high)])


# ---- baking ----
#
#Nothing here makes Houdini's own thread, the one that draws its interface, wait on a bake. Only writing the scene runs there, as it reads Houdini's
#geometry, with a progress bar. A remote bake's checks, uploads and start run on a thread of their own, which touches no hou objects (they aren't
#thread-safe), and the bake itself runs in processes of its own. Its progress is looked at twice a second, from the end of its event log, and shown under
#the node and in the status bar, which cook nothing: Status only changes when a bake starts and ends, and the frame on show is reloaded only when its
#file arrives or changes.

def _program(node):
    program = node.evalParm("program") or os.environ.get("FLIP2") or shutil.which("flip2")
    if not program:
        raise hou.NodeError("no flip2 program: set flip2 Program, or $FLIP2, or put flip2 on the PATH")
    return program


REMOTE_DIRECTORY = re.compile(r"^[A-Za-z0-9_./~-]+$")
JOB_FOLDER = re.compile(r"^[A-Za-z0-9_.-]+-[0-9a-f]{8}$")     #<node>-<hash of its output directory>: the only folders delete_remote deletes
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/flip2-%r@%h:%p",
               "-o", "ControlPersist=120"]   #one connection, kept open between the mirror's syncs


def _remote(node):
    """a remote bake's host, its job directory there (one per node and output directory), and flip2 there"""
    host = node.evalParm("remotehost").strip()
    if not host:
        raise hou.NodeError("no remote host")
    base = node.evalParm("remotedir").strip().rstrip("/") or "~/flip2-bakes"
    if not REMOTE_DIRECTORY.match(base):
        raise hou.NodeError("Remote Directory can only hold letters, digits and ~ . _ - /: %r" % base)
    job = "%s-%s" % (node.name(), hashlib.sha1(os.path.abspath(_output_directory(node)).encode()).hexdigest()[:8])
    return host, base + "/" + job, node.parm("program").unexpandedString().strip() or "flip2"     #unexpanded: ~ is the remote machine's home


def _ssh(host, command):
    """runs command on host; RuntimeError, not a hou one, as it runs on a launching thread"""
    result = subprocess.run(["ssh"] + SSH_OPTIONS + [host, command], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError("ssh %s: %s" % (host, (result.stderr or result.stdout).strip() or "exit status %d" % result.returncode))
    return result.stdout


def _rsync(arguments):
    result = subprocess.run(["rsync", "-a", "-e", " ".join(["ssh"] + SSH_OPTIONS)] + arguments, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError("rsync: %s" % (result.stderr.strip() or "exit status %d" % result.returncode))


def _launch_remote(job, host, directory, program, arguments, upload, local, logs):
    """on a thread: checks there's a flip2 on host, sends the scene if upload, starts the bake there detached (remote/job.sh), and the mirror bringing
    it back here (remote/mirror.sh). Says how far it's got in job["state"], and why it stopped in job["failure"]"""
    scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote")
    try:
        job["state"] = "connecting to %s" % host
        _ssh(host, 'program=%s; program="${program/#\\~/$HOME}"; '     #a path names an executable file; a bare name is looked up on the PATH
                   'case "$program" in */*) [ -f "$program" ] && [ -x "$program" ];; *) command -v "$program" > /dev/null;; esac || '
                   '{ echo "no flip2 program at $program on this machine: set flip2 Program" >&2; exit 1; }' % shlex.quote(program))
        _ssh(host, "mkdir -p %s/geo %s/logs && rm -f %s/logs/finished" % (directory, directory, directory))
        if upload:
            job["state"] = "sending the scene to %s" % host
            _rsync(["--delete", os.path.join(local, "geo") + "/", "%s:%s/geo/" % (host, directory)])
            _rsync([os.path.join(local, "scene.json"), "%s:%s/" % (host, directory)])
        _rsync([os.path.join(scripts, "job.sh"), "%s:%s/" % (host, directory)])
        job["state"] = "starting on %s" % host
        _ssh(host, "cd %s && nohup bash job.sh %s > /dev/null 2>&1 < /dev/null &" % (directory, " ".join(shlex.quote(word) for word in [program] + arguments)))
        environment = dict(os.environ, FLIP2_RSH=" ".join(["ssh"] + SSH_OPTIONS))
        job["process"] = subprocess.Popen(["bash", os.path.join(scripts, "mirror.sh"), host, directory, local], env=environment,
                                          stdout=subprocess.DEVNULL, stderr=open(os.path.join(logs, "mirror.log"), "a"))
        job["state"] = "baking on %s" % host
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        job["failure"] = str(error)


def _mesh_options(node):
    """flip2 mesh's options, from the Surface tab, checked as flip2 mesh would check them"""
    voxel, influence, radius = node.evalParm("voxelscale"), node.evalParm("influencescale"), node.evalParm("radiusscale")
    smoothing = node.evalParm("smoothing")
    if min(voxel, influence, radius) <= 0 or smoothing < 0:
        raise hou.NodeError("the surface's scales have to be positive, and its smoothing 0 or more")
    if radius >= influence:
        raise hou.NodeError("the surface's radius scale has to be less than its influence scale")
    if influence > 8*voxel:
        raise hou.NodeError("the surface's influence scale can be at most 8 times its voxel scale")
    return ["--voxel-scale", repr(voxel), "--influence-scale", repr(influence), "--radius-scale", repr(radius), "--smoothing", str(smoothing)]


def _start(node, kind, upload=True):
    """starts a bake ("bake", or "resume" from the newest checkpoint), meshing its frames as they're committed if the Surface tab says to, or meshes
    the bake's frames again ("mesh"); returns at once, and _watch follows it"""
    if node.path() in _jobs:
        raise hou.NodeError("a bake is running already")
    meshing = _mesh_options(node) if kind == "mesh" or node.evalParm("meshsurface") else None
    local = _output_directory(node)
    logs = os.path.join(local, "logs")
    os.makedirs(logs, exist_ok=True)
    name = "mesh" if kind == "mesh" else "bake"     #its logs: events, and what it says on standard error
    for stale in ("finished", name + ".events.jsonl", "bake.pid"):     #an earlier one's
        if os.path.exists(os.path.join(logs, stale)):
            os.remove(os.path.join(logs, stale))
    job = {"process": None, "export": None, "mesh": None, "events": os.path.join(logs, name + ".events.jsonl"), "log": os.path.join(logs, name + ".log"),
           "kind": kind, "meshing": meshing, "state": "starting", "failure": None, "said": None, "shown": {}}
    if node.evalParm("bakeon") == 1:
        host, directory, program = _remote(node)
        job["remote"] = (host, directory)
        arguments = {"bake": ["bake", "scene.json", "--out", "bake", "--overwrite"], "resume": ["resume", "bake"], "mesh": ["mesh"]}[kind]
        if meshing is not None:
            arguments = ["--mesh", " ".join(meshing)] + arguments
        threading.Thread(target=_launch_remote, args=(job, host, directory, program, arguments, upload, local, logs), daemon=True).start()
    else:
        program = _program(node)
        bake = os.path.join(local, "bake")
        job["program"] = program
        if kind == "mesh":
            job["process"] = subprocess.Popen([program, "mesh", bake, "--overwrite"] + meshing, stdout=open(job["events"], "w"),
                                              stderr=open(job["log"], "w"), cwd=local)
        else:
            arguments = {"bake": ["bake", os.path.join(local, "scene.json"), "--out", bake, "--overwrite"], "resume": ["resume", bake]}[kind]
            job["process"] = subprocess.Popen([program] + arguments, stdout=open(job["events"], "w"), stderr=open(job["log"], "w"), cwd=local)
            job["export"] = subprocess.Popen([program, "export", bake, "--follow"], stdout=open(os.path.join(logs, "export.events.jsonl"), "w"),
                                             stderr=open(os.path.join(logs, "export.log"), "w"), cwd=local)
            if meshing is not None:
                job["mesh"] = subprocess.Popen([program, "mesh", bake, "--follow"] + meshing, stdout=open(os.path.join(logs, "mesh.events.jsonl"), "w"),
                                               stderr=open(os.path.join(logs, "mesh.log"), "w"), cwd=local)
        job["state"] = "meshing" if kind == "mesh" else "baking"
    _jobs[node.path()] = job
    node.parm("status").set("meshing" if kind == "mesh" else "baking")
    _show(node, job["state"])
    if hou.isUIAvailable() and _watch not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_watch)


def bake(node):
    """writes the scene and bakes it from the start, over any bake already in the output directory"""
    write_scene(node)
    _start(node, "bake")


def resume(node):
    """carries the bake on from its newest checkpoint"""
    _start(node, "resume", upload=False)


def mesh_bake(node):
    """meshes the bake's frames again with the Surface tab's settings, replacing their surfaces"""
    if node.evalParm("bakeon") == 0 and importer.read_bake(os.path.join(_output_directory(node), "bake")) is None:
        raise hou.NodeError("no bake to mesh in %s" % os.path.join(_output_directory(node), "bake"))
    _start(node, "mesh", upload=False)


def cancel(node):
    """asks the bake to stop after the frame it's on, with a checkpoint to resume from"""
    job = _jobs.get(node.path())
    if job is None or job["process"] is None or job["process"].poll() is not None:
        node.parm("status").set("no bake running")
        return
    if "remote" in job:
        host, directory = job["remote"]
        def signal_remote():
            try:
                _ssh(host, "kill -TERM $(cat %s/logs/bake.pid)" % directory)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                job["state"] = "couldn't cancel: %s" % error
        threading.Thread(target=signal_remote, daemon=True).start()
    else:
        job["process"].send_signal(signal.SIGTERM)
    job["cancelled"] = True
    job["state"] = "cancelling" if job["kind"] == "mesh" else "cancelling after this frame"
    _show(node, job["state"])


def delete_remote(node, confirm=True):
    """deletes this node's bake on the remote machine, after asking: the frames already here stay, but it can't be resumed or extended after"""
    if node.path() in _jobs:
        raise hou.NodeError("a bake is running: cancel it first")
    host, directory, _ = _remote(node)
    if not JOB_FOLDER.match(directory.rsplit("/", 1)[-1]):
        raise hou.NodeError("not deleting %s: it isn't a flip2 job folder" % directory)
    if confirm and hou.isUIAvailable():
        choice = hou.ui.displayMessage("Delete %s:%s?" % (host, directory), buttons=("Delete", "Cancel"), default_choice=1, close_choice=1,
                                       severity=hou.severityType.Warning,
                                       details="The frames already here stay, but the bake can't be resumed or extended once its remote copy is gone.")
        if choice != 0:
            return
    path = node.path()

    def remove():
        try:
            missing = "missing" in _ssh(host, "test -d %s || echo missing; rm -rf -- %s" % (directory, directory))
            _notices.append((path, "no remote copy at %s:%s" % (host, directory) if missing else "deleted the remote copy, %s:%s" % (host, directory)))
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            _notices.append((path, "couldn't delete the remote copy: %s" % error))
        finally:
            _working[0] -= 1
    _working[0] += 1
    threading.Thread(target=remove, daemon=True).start()
    _show(node, "deleting the remote copy")
    if hou.isUIAvailable() and _watch not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_watch)


def _last_event(path):
    """the last event in a bake's log, read from its end"""
    try:
        with open(path, "rb") as events:
            events.seek(0, os.SEEK_END)
            events.seek(max(0, events.tell() - 4096))
            lines = [line for line in events.read().decode("utf-8", "replace").splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def _describe(event):
    if event is None:
        return None
    kind = event.get("event")
    return {"start": "started: %s particles" % event.get("particles"), "frame": "frame %s" % event.get("frame"),
            "committed": "frame %s on disk" % event.get("frame"), "checkpoint": "checkpoint at frame %s" % event.get("frame"),
            "cancelled": "cancelled after frame %s" % event.get("frame"), "done": "done: %s frames in %.0f s" % (event.get("frames"), event.get("seconds", 0)),
            "meshed": "meshed frame %s" % event.get("frame"),
            "meshDone": "meshed %s frames in %.0f s" % (event.get("meshed"), event.get("seconds", 0)) if not event.get("problems")
            else "meshed %s frames; %s failed: see logs/mesh.log" % (event.get("meshed"), event.get("problems")),
            "error": "error: %s" % event.get("message")}.get(kind, kind)


def _show(node, text):
    """a bake's progress where showing it cooks nothing: under the node in the network editor, and in the status bar"""
    node.setComment(text)
    node.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    if hou.isUIAvailable():
        hou.ui.setStatusMessage("%s: %s" % (node.name(), text))


def _refresh(node, job):
    """reloads the frame on show, surface and particles, where its file has arrived or changed since it was loaded"""
    if not node.evalParm("load"):
        return
    for name in (importer.SURFACE, importer.PARTICLES):
        loader = node.node(name)
        if loader is None:
            continue
        path = loader.evalParm("file")
        try:
            stamp = os.stat(path)
            key = (path, stamp.st_mtime_ns, stamp.st_size)
        except OSError:
            key = (path, None, None)
        if key != job["shown"].get(name):
            job["shown"][name] = key
            loader.parm("reload").pressButton()


def _ending(job):
    """why a bake (or meshing) ended, if it didn't finish or cancel: the last thing it said on standard error"""
    logs = os.path.dirname(job["events"])
    if "remote" in job:
        try:
            with open(os.path.join(logs, "finished")) as finished:
                status = int(finished.read().strip() or 0)
        except (OSError, ValueError):
            return "lost touch with %s: see %s" % (job["remote"][0], os.path.join(logs, "mirror.log"))
    else:
        status = job["process"].returncode
    stopped = (0, 3) if job["kind"] != "mesh" else (0, -signal.SIGTERM, 128 + signal.SIGTERM)     #done, or cancelled (a bake after a checkpoint)
    if status in stopped:
        return None
    try:
        with open(job["log"]) as log:
            lines = [line.strip() for line in log.read().splitlines() if line.strip()]
    except OSError:
        lines = []
    return "failed (exit status %d): %s" % (status, lines[-1] if lines else "see " + job["log"])


_next_look = [0.0]
_notices = []       #(node path, text) from threads, which mustn't touch hou objects: _watch shows them
_working = [0]      #threads still at work for nodes (deleting remote copies)


def _watch():
    """from Houdini's event loop, which calls it whenever it's idle: twice a second, each bake's progress, the frame on show if it's arrived, and when
    a bake ends, its result in Status. A local bake that's done is watched until its exporter and mesher have caught up with it and stopped; one that
    stopped short has them stopped, and run once more over the frames it committed"""
    now = time.monotonic()
    if now < _next_look[0]:
        return
    _next_look[0] = now + 0.5
    while _notices:
        path, text = _notices.pop(0)
        node = hou.node(path)
        if node is not None:
            node.parm("status").set(text)
            _show(node, text)
    for path, job in list(_jobs.items()):
        node = hou.node(path)
        if node is None:    #deleted: its bake carries on, unwatched
            del _jobs[path]
            continue
        if job["failure"] is not None:
            node.parm("status").set("failed: %s" % job["failure"])
            _show(node, "failed")
            del _jobs[path]
            continue
        process = job["process"]
        event = _last_event(job["events"]) if process is not None else None
        text = _describe(event) or job["state"]
        if text != job["said"]:
            job["said"] = text
            _show(node, text)
        if process is None:     #still being launched
            continue
        _refresh(node, job)
        if process.poll() is None:
            continue
        if process.returncode == 0 and any(job[follower] is not None and job[follower].poll() is None for follower in ("export", "mesh")):
            continue
        failure = _ending(job)
        final = failure if failure is not None and (event is None or event.get("event") not in ("done", "cancelled", "error", "meshDone")) else text
        if job.get("cancelled") and job["kind"] == "mesh":
            final = "cancelled: %s" % text
        bake = os.path.join(os.path.dirname(os.path.dirname(job["events"])), "bake")
        for follower, command in (("export", ["export", bake]), ("mesh", ["mesh", bake] + (job["meshing"] or []))):
            if job[follower] is not None and job[follower].poll() is None:     #following frames that won't come now: the last ones, once more
                job[follower].terminate()
                subprocess.Popen([job["program"]] + command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        node.parm("status").set(final)
        _show(node, final)
        job["shown"] = {}
        _refresh(node, job)
        del _jobs[path]
    if not _jobs and not _notices and _working[0] == 0 and hou.isUIAvailable() and _watch in hou.ui.eventLoopCallbacks():
        hou.ui.removeEventLoopCallback(_watch)


def shelf_tool(kwargs):
    """flip2 Solver from the Tab menu or the shelf: in the network being edited, inside a new geometry object if that's the object level, wired to the
    selected nodes in order (fluid, collisions, sources, sinks)"""
    pane = kwargs.get("pane")
    network = pane.pwd() if pane is not None else hou.node("/obj")
    position = pane.cursorPosition() if pane is not None else None
    if network.childTypeCategory() == hou.objNodeTypeCategory():
        container = network.createNode("geo", "flip2")
        if position is not None:
            container.setPosition(position)
        network, position = container, None
    elif network.childTypeCategory() != hou.sopNodeTypeCategory():
        raise hou.Error("flip2 Solver goes in a geometry network")
    selected = [node for node in hou.selectedNodes() if node.parent() == network]
    node = create_solver(network)
    for index, source in enumerate(selected[:len(ROLES)]):
        node.setInput(index, source)
    if position is not None:
        node.setPosition(position)
    elif selected:
        node.moveToGoodPosition()
    node.setDisplayFlag(True)
    node.setRenderFlag(True)
    node.setSelected(True, clear_all_selected=True)
    return node


def update_all(root=None):
    """gives every flip2 Solver and flip2 Import node under root (the whole scene by default) the latest parameters and inner nodes, keeping their
    values. Returns them"""
    updated = []
    for node in (root or hou.node("/")).allSubChildren():
        if node.type().name() != "subnet":
            continue
        kind = node.userData("flip2") or ("solver" if node.parm("particlesep") is not None and node.parm("remotehost") is not None
                                          else "import" if node.parm("bakedir") is not None else None)
        if kind == "solver":
            _interface(node)
            _network(node)
        elif kind == "import":
            importer._interface(node)
            importer._network(node)
        else:
            continue
        node.setUserData("flip2", kind)
        updated.append(node)
    return updated


def update_tool(kwargs):
    """the Update flip2 Nodes tool"""
    updated = update_all()
    hou.ui.displayMessage("Updated %d flip2 node%s to this version's parameters." % (len(updated), "" if len(updated) == 1 else "s"),
                          details="\n".join(node.path() for node in updated) or None)
