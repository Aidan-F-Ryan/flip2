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
running bake commits frames to, and export/houdini/ beside it) and logs/. The bake's frame 0 is the start frame, and the node shows it there.

Bake On picks where flip2 runs: this machine, or another over ssh (a GPU box, for a Mac). A remote bake sends scene.json and geo/ to the remote
directory, runs there detached (remote/job.sh), and is mirrored back every couple of seconds (remote/mirror.sh): its logs, its cache.json and its
exported frames, so it shows here as a local one does. It needs ssh to reach the host without a password (a key), and rsync at both ends.

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

import hou

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
    group = node.parmTemplateGroup()
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
    status = hou.StringParmTemplate("status", "Status", 1)
    status.setDisableWhen("{ cfl >= 0 }")       #always: it's only to read
    bake.addParmTemplate(status)
    bake.addParmTemplate(hou.ToggleParmTemplate("load", "Load from Disk", default_value=True, help="Show the bake's particles at each frame"))
    group.append(bake)
    node.setParmTemplateGroup(group)


def create_solver(parent, name="flip2_solver"):
    """a flip2 Solver node in the SOP network parent"""
    node = parent.createNode("subnet", name)
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
    particles = node.createNode("file", "particles")
    particles.parm("file").set('`chs("../outputdir")`/bake/export/houdini/particles.`padzero(4, round(($T - (ch("../startframe") - 1)/$FPS)*$FPS))`.bgeo')
    particles.parm("missingframe").set("empty")
    loaded = node.createNode("switch", "load")     #input 0 left empty: nothing loaded
    loaded.setInput(1, particles)
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


def _sample(node, role, frames):
    """an input's objects over frames: per object its name, triangles (into its own vertices) and vertex positions per frame"""
    import numpy
    source = node.node(role.upper())
    first, positions = _objects(source.geometryAtFrame(frames[0]))
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
    for role in ROLES:
        if not _connected(node, ROLES.index(role)):
            continue
        for index, samples in enumerate(_sample(node, role, frames[:1] if role == "fluid" else frames)):
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

def _program(node):
    program = node.evalParm("program") or os.environ.get("FLIP2") or shutil.which("flip2")
    if not program:
        raise hou.NodeError("no flip2 program: set flip2 Program, or $FLIP2, or put flip2 on the PATH")
    return program


SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/flip2-%r@%h:%p",
               "-o", "ControlPersist=120"]   #one connection, kept open between the mirror's syncs


def _remote(node):
    """a remote bake's host, its job directory there (one per node and output directory), and flip2 there"""
    host = node.evalParm("remotehost").strip()
    if not host:
        raise hou.NodeError("no remote host")
    base = node.evalParm("remotedir").strip().rstrip("/") or "~/flip2-bakes"
    job = "%s-%s" % (node.name(), hashlib.sha1(os.path.abspath(_output_directory(node)).encode()).hexdigest()[:8])
    return host, base + "/" + job, node.parm("program").unexpandedString().strip() or "flip2"     #unexpanded: ~ is the remote machine's home


def _ssh(host, command):
    result = subprocess.run(["ssh"] + SSH_OPTIONS + [host, command], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise hou.NodeError("ssh %s: %s" % (host, (result.stderr or result.stdout).strip() or "exit status %d" % result.returncode))
    return result.stdout


def _rsync(arguments):
    result = subprocess.run(["rsync", "-a", "-e", " ".join(["ssh"] + SSH_OPTIONS)] + arguments, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise hou.NodeError("rsync: %s" % (result.stderr.strip() or "exit status %d" % result.returncode))


def _run_remote(node, arguments, upload):
    """runs flip2 with arguments on the remote host (sending the scene first if upload), mirrored back here by remote/mirror.sh"""
    host, job, program = _remote(node)
    local = _output_directory(node)
    scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote")
    logs = os.path.join(local, "logs")
    os.makedirs(logs, exist_ok=True)
    for stale in ("finished", "bake.events.jsonl", "bake.pid"):     #from an earlier bake: the mirror replaces them
        if os.path.exists(os.path.join(logs, stale)):
            os.remove(os.path.join(logs, stale))
    _ssh(host, 'program=%s; program="${program/#\\~/$HOME}"; '      #a path names an executable file; a bare name is looked up on the PATH
               'case "$program" in */*) [ -f "$program" ] && [ -x "$program" ];; *) command -v "$program" > /dev/null;; esac || '
               '{ echo "no flip2 program at $program on this machine: set flip2 Program" >&2; exit 1; }' % shlex.quote(program))
    _ssh(host, "mkdir -p %s/geo %s/logs && rm -f %s/logs/finished" % (job, job, job))
    if upload:
        _rsync(["--delete", os.path.join(local, "geo") + "/", "%s:%s/geo/" % (host, job)])
        _rsync([os.path.join(local, "scene.json"), "%s:%s/" % (host, job)])
    _rsync([os.path.join(scripts, "job.sh"), "%s:%s/" % (host, job)])
    _ssh(host, "cd %s && nohup bash job.sh %s > /dev/null 2>&1 < /dev/null &" % (job, " ".join(shlex.quote(word) for word in [program] + arguments)))
    environment = dict(os.environ, FLIP2_RSH=" ".join(["ssh"] + SSH_OPTIONS))
    mirror = subprocess.Popen(["bash", os.path.join(scripts, "mirror.sh"), host, job, local], env=environment, stdout=subprocess.DEVNULL,
                              stderr=open(os.path.join(logs, "mirror.log"), "a"))
    _jobs[node.path()] = {"bake": mirror, "export": None, "events": os.path.join(logs, "bake.events.jsonl"), "remote": (host, job)}
    node.parm("status").set("baking on %s" % host)
    if hou.isUIAvailable() and _watch not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_watch)
    return mirror


def _run(node, arguments, upload=True):
    """starts flip2 with arguments, and flip2 export following the bake beside it; their events and messages go to logs/. On a remote machine if the
    node says so"""
    if node.path() in _jobs and _jobs[node.path()]["bake"].poll() is None:
        raise hou.NodeError("a bake is running already")
    if node.evalParm("bakeon") == 1:
        remote = {"bake": ["bake", "scene.json", "--out", "bake", "--overwrite"], "resume": ["resume", "bake"]}[arguments[0]]
        return _run_remote(node, remote, upload)
    program = _program(node)
    directory = _output_directory(node)
    logs = os.path.join(directory, "logs")
    os.makedirs(logs, exist_ok=True)
    events = os.path.join(logs, "bake.events.jsonl")
    bake = subprocess.Popen([program] + arguments, stdout=open(events, "w"), stderr=open(os.path.join(logs, "bake.log"), "w"), cwd=directory)
    exporter = subprocess.Popen([program, "export", os.path.join(directory, "bake"), "--follow"], stdout=open(os.path.join(logs, "export.events.jsonl"), "w"),
                                stderr=open(os.path.join(logs, "export.log"), "w"), cwd=directory)
    _jobs[node.path()] = {"bake": bake, "export": exporter, "events": events}
    node.parm("status").set("baking")
    if hou.isUIAvailable() and _watch not in hou.ui.eventLoopCallbacks():
        hou.ui.addEventLoopCallback(_watch)
    return bake


def bake(node):
    """writes the scene and bakes it from the start, over any bake already in the output directory"""
    scene = write_scene(node)
    return _run(node, ["bake", scene, "--out", os.path.join(_output_directory(node), "bake"), "--overwrite"])


def resume(node):
    """carries the bake on from its newest checkpoint"""
    return _run(node, ["resume", os.path.join(_output_directory(node), "bake")], upload=False)


def cancel(node):
    """asks the bake to stop after the frame it's on, with a checkpoint to resume from"""
    job = _jobs.get(node.path())
    if job is None or job["bake"].poll() is not None:
        node.parm("status").set("no bake running")
        return
    if "remote" in job:
        host, directory = job["remote"]
        _ssh(host, "kill -TERM $(cat %s/logs/bake.pid)" % directory)
    else:
        job["bake"].send_signal(signal.SIGTERM)
    node.parm("status").set("cancelling after this frame")


def _last_event(path):
    try:
        with open(path) as events:
            lines = [line for line in events.read().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def _ending(job):
    """why a bake ended, if it didn't finish or cancel: the last thing it said on standard error"""
    logs = os.path.dirname(job["events"])
    if "remote" in job:
        try:
            with open(os.path.join(logs, "finished")) as finished:
                status = int(finished.read().strip() or 0)
        except (OSError, ValueError):
            return "lost touch with %s: see %s" % (job["remote"][0], os.path.join(logs, "mirror.log"))
    else:
        status = job["bake"].returncode
    if status in (0, 3):    #done, or cancelled after a checkpoint
        return None
    try:
        with open(os.path.join(logs, "bake.log")) as log:
            lines = [line.strip() for line in log.read().splitlines() if line.strip()]
    except OSError:
        lines = []
    return "failed (exit status %d): %s" % (status, lines[-1] if lines else "see " + os.path.join(logs, "bake.log"))


def _watch():
    """from Houdini's event loop: each bake's progress into its node's status; when one ends, its exporter catches up and stops"""
    for path, job in list(_jobs.items()):
        node = hou.node(path)
        event = _last_event(job["events"])
        ended = job["bake"].poll()
        if node is not None and event is not None:
            kind = event.get("event")
            text = {"start": "started: %s particles" % event.get("particles"), "frame": "frame %s" % event.get("frame"),
                    "committed": "frame %s on disk" % event.get("frame"), "checkpoint": "checkpoint at frame %s" % event.get("frame"),
                    "cancelled": "cancelled after frame %s" % event.get("frame"), "done": "done: %s frames in %.0f s" % (event.get("frames"), event.get("seconds", 0)),
                    "error": "error: %s" % event.get("message")}.get(kind, kind)
            if node.evalParm("status") != text:
                node.parm("status").set(text)
                particles = node.node("particles")
                if kind == "committed" and particles is not None:
                    particles.parm("reload").pressButton()
        if ended is not None:
            failure = _ending(job)
            if node is not None and failure is not None and (event is None or event.get("event") not in ("done", "cancelled", "error")):
                node.parm("status").set(failure)
            if job["export"] is not None and job["export"].poll() is None:
                job["export"].terminate()
                subprocess.Popen([_program(node) if node else "flip2", "export", os.path.join(os.path.dirname(os.path.dirname(job["events"])), "bake")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)     #the frames it committed last
            if node is not None and node.node("particles") is not None:
                node.node("particles").parm("reload").pressButton()
            del _jobs[path]
    if not _jobs and hou.isUIAvailable() and _watch in hou.ui.eventLoopCallbacks():
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
