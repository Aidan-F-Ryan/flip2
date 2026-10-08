"""flip2 Import: a bake's surface and particles, at the current time, from the files `flip2 mesh` and `flip2 export` write
(BAKE/export/houdini/surface.NNNN.bgeo.sc and particles.NNNN.bgeo.sc, or .bgeo uncompressed: Export Format, which it finds), on its first output. Its second has the bake's fluid fields if `flip2 mesh --fields`
wrote them (fields.NNNN.vdb: surface and vel), its third the domain the bake filled, and its fourth the collision geometry wired to its input, the last
two as Houdini's own FLIP nodes make them: what Whitewater Source takes as its Liquid Simulation, Container and Collisions (whitewater.py sets it up).
Its fifth has the bake's own whitewater, if it was baked with any (whitewater.NNNN.bgeo.sc): points with v, id, age, life, radius, kind (0 spray, 1
foam, 2 a bubble) and the pscale to draw them at.

A flip2 frame is a time: frame N is N/fps seconds after the bake's start. Houdini's frame 1 is time 0, so loading by time puts the bake's frame 0 on
Houdini's frame 1, and keeps it there whatever frame rate either uses. A frame the bake hasn't made (yet) loads as no geometry.
"""
import glob
import json
import os

import hou

PARTICLES = "particles"
SURFACE = "surface"
FIELDS = "fields"
WHITEWATER = "whitewater"
SHOW = (("surface", "Surface"), ("particles", "Particles"), ("both", "Surface and Particles"))
FORMATS = (("bgeo.sc", "Compressed (.bgeo.sc)"), ("bgeo", "Uncompressed (.bgeo)"))


def read_bake(directory):
    """a bake's cache.json, or None if there isn't one (yet)"""
    try:
        with open(os.path.join(hou.text.expandString(directory), "cache.json")) as record:
            return json.load(record)
    except (OSError, ValueError):
        return None


def show_parm(**kwargs):
    """the Show menu: what of each frame to load"""
    return hou.MenuParmTemplate("show", "Show", [item for item, _ in SHOW], [label for _, label in SHOW], default_value=0,
                                help="What of each frame to load: the liquid's surface (flip2 mesh), its particles, or both", **kwargs)


def format_parm(**kwargs):
    """the Export Format menu: whether a bake's frames are .bgeo.sc files or .bgeo"""
    return hou.MenuParmTemplate("exportformat", "Export Format", [item for item, _ in FORMATS], [label for _, label in FORMATS], default_value=0,
                                help="How the bake's frames are written for Houdini: compressed as Houdini's own .bgeo.sc are (Blosc), 20-45% smaller and a "
                                     "little slower to load, or not", **kwargs)


def detect_format(folder):
    """the format a bake's export/houdini directory holds frames in, or None if it holds none"""
    folder = hou.text.expandString(folder)
    for format_, _ in FORMATS:
        for name in (PARTICLES, SURFACE):
            if glob.glob(os.path.join(folder, "%s.[0-9]*.%s" % (name, format_))):
                return format_
    return None


def build_loaders(node, folder, frame):
    """inside node, File nodes loading a frame's surface and particles from folder (the bake's export/houdini directory) at frame (the bake's frame
    number), both expressions, and a switch between them by the node's Show menu, which it returns. Makes only what's missing, so an older node, which
    loaded only particles, gains the rest, and whatever took its particles takes the switch"""
    loaders = {}
    for name in (SURFACE, PARTICLES):
        loader = node.node(name)
        if loader is None:
            loader = node.createNode("file", name)
            loader.parm("missingframe").set("empty")
        loader.parm("file").set('%s/%s.`padzero(4, %s)`.`chs("../exportformat")`' % (folder, name, frame))
        loaders[name] = loader
    switch = node.node("show")
    if switch is None:
        takers = [(taker, taker.inputs().index(loaders[PARTICLES])) for taker in loaders[PARTICLES].outputs()]
        both = node.createNode("merge", "both")
        both.setInput(0, loaders[SURFACE])
        both.setInput(1, loaders[PARTICLES])
        switch = node.createNode("switch", "show")
        for index, source in enumerate((loaders[SURFACE], loaders[PARTICLES], both)):     #in the Show menu's order
            switch.setInput(index, source)
        switch.parm("input").setExpression('ch("../show")')
        for taker, index in takers:
            taker.setInput(index, switch)
        node.layoutChildren()
    return switch


def build_fields(node, folder, frame):
    """inside node, a File node loading a frame's fluid fields from folder at frame (as build_loaders), on the node's second output. Makes only what's
    missing"""
    loader = node.node(FIELDS)
    if loader is None:
        loader = node.createNode("file", FIELDS)
        loader.parm("missingframe").set("empty")
    loader.parm("file").set('%s/fields.`padzero(4, %s)`.vdb' % (folder, frame))
    _output(node, 1, loader)
    return loader


def build_whitewater(node, folder, frame):
    """inside node, a File node loading a frame's whitewater from folder at frame (as build_loaders), on the node's fifth output. Makes only what's
    missing"""
    loader = node.node(WHITEWATER)
    if loader is None:
        loader = node.createNode("file", WHITEWATER)
        loader.parm("missingframe").set("empty")
    loader.parm("file").set('%s/whitewater.`padzero(4, %s)`.`chs("../exportformat")`' % (folder, frame))
    _output(node, 4, loader)
    return loader


def has_whitewater(folder):
    """whether a bake's export/houdini directory holds any whitewater frames"""
    return bool(glob.glob(os.path.join(hou.text.expandString(folder), WHITEWATER + ".[0-9]*.bgeo*")))


def build_container(node, collision, wired):
    """inside node, the domain and the collisions as Houdini's FLIP has them, for its whitewater: a box for the domain the bake filled, which the caller
    sizes, through FLIP Container (the Container stream, on the node's third output), and the node's collision geometry (collision: an inner node, or
    one of its inputs) through FLIP Collide (the Collisions stream, on its fourth: a level set of the colliders and their velocity); or while nothing's
    wired to the node's input number wired, the container's own empty collisions. Both are at the node's particle separation. Returns the box. Makes
    only what's missing, so an older node gains the rest"""
    box = node.node("container")
    if box is None:
        box = node.createNode("box", "container")
    container = node.node("container_stream")
    if container is None:
        container = node.createNode("flipcontainer", "container_stream")
        container.setInput(0, box)
        container.parm("particlesep").setExpression('ch("../particlesep")')
    collide = node.node("collision_stream")
    if collide is None:
        collide = node.createNode("flipcollide", "collision_stream")
        for index in range(3):
            collide.setInput(index, container, index)
    collide.setInput(3, collision)
    collisions = node.node("collisions")
    if collisions is None:
        collisions = node.createNode("switch", "collisions")
        collisions.setInput(0, container, 2)
        collisions.setInput(1, collide, 2)
        collisions.parm("input").setExpression('strlen(opinputpath("..", %d)) > 0' % wired)
    _output(node, 2, container, 1)
    _output(node, 3, collisions)
    return box


def _output(node, index, source, source_output=0):
    """the node's output number index, showing source's output"""
    output = node.node("output%d" % index)
    if output is None:
        output = node.createNode("output", "output%d" % index)
        output.parm("outputidx").set(index)
        node.layoutChildren()
    output.setInput(0, source, source_output)
    return output


def has_surface(folder):
    """whether a bake's export/houdini directory holds any surface frames"""
    return bool(glob.glob(os.path.join(hou.text.expandString(folder), SURFACE + ".[0-9]*.bgeo*")))


def _interface(node):
    """the node's parameters, from the subnet's own on, so applying them again (an update) keeps its values"""
    group = node.type().parmTemplateGroup()
    group.append(hou.StringParmTemplate("bakedir", "Bake", 1, string_type=hou.stringParmType.FileReference, file_type=hou.fileType.Directory,
                                        help="The bake's directory: the one with cache.json in it",
                                        script_callback="__import__('flip2houdini').importer.reload(kwargs['node'])",
                                        script_callback_language=hou.scriptLanguage.Python))
    group.append(hou.FloatParmTemplate("fps", "Bake Frame Rate", 1, default_value=(24.0,), min=1.0,
                                       help="Frames per second of the bake, read from its cache.json"))
    group.append(hou.FloatParmTemplate("timeoffset", "Time Offset", 1, default_value=(0.0,),
                                       help="Seconds to delay the bake by: its start shows at this time"))
    group.append(hou.FloatParmTemplate("particlesep", "Particle Separation", 1, default_value=(0.02,), min=0.0001,
                                       help="The bake's particle separation, read from its cache.json: the size of the container and collisions on the third "
                                            "and fourth outputs, and of whitewater set up on this bake"))
    group.append(show_parm())
    group.append(format_parm())
    group.append(hou.ButtonParmTemplate("reload", "Reload", script_callback="__import__('flip2houdini').importer.reload(kwargs['node'])",
                                        script_callback_language=hou.scriptLanguage.Python,
                                        help="Read the bake's cache.json again, and the frame on show from disk"))
    status = hou.StringParmTemplate("status", "Status", 1, help="What the bake holds")
    status.setDisableWhen("{ fps >= 0 }")     #always: it's only to read
    group.append(status)
    node.setParmTemplateGroup(group)


def _network(node):
    """the node's network: its loaders, by the frame showing at this time (a bake's frame N is at N/fps), and its domain. Returns what the first output
    shows"""
    folder, frame = '`chs("../bakedir")`/export/houdini', 'round(($T - ch("../timeoffset"))*ch("../fps"))'
    shown = build_loaders(node, folder, frame)
    build_fields(node, folder, frame)
    build_container(node, node.indirectInputs()[0], 0)
    build_whitewater(node, folder, frame)
    if node.parm("label1") is not None:
        node.parm("label1").set("Collisions (for Whitewater)")
    return shown


def create_import(parent, bake_dir="", name="flip2_import"):
    """a flip2 Import node in the SOP network parent, loading bake_dir's frames: its surface if it has one, otherwise its particles"""
    node = parent.createNode("subnet", name)
    node.setUserData("flip2", "import")
    _interface(node)
    output = node.createNode("output", "output0")
    output.setInput(0, _network(node))
    output.setDisplayFlag(True)
    output.setRenderFlag(True)
    node.layoutChildren()
    node.parm("bakedir").set(bake_dir)
    if bake_dir and not has_surface(os.path.join(bake_dir, "export", "houdini")):
        node.parm("show").set(PARTICLES)
    _match_format(node)
    reload(node)
    return node


def _match_format(node):
    """the Export Format of the frames the bake has, if it has any"""
    found = detect_format(os.path.join(node.evalParm("bakedir"), "export", "houdini")) if node.evalParm("bakedir") else None
    if found is not None and found != node.parm("exportformat").evalAsString():
        node.parm("exportformat").set(found)


def reload(node):
    """reads the bake's cache.json into the node's frame rate and status, and its frame on show from disk again"""
    _match_format(node)
    bake = read_bake(node.evalParm("bakedir"))
    if bake is None:
        node.parm("status").set("no bake there (no cache.json)" if node.evalParm("bakedir") else "")
    else:
        node.parm("fps").set(float(bake.get("fps", 24.0)))
        if float(bake.get("voxelSize", 0.0)) > 0.0 and node.parm("particlesep") is not None:
            node.parm("particlesep").set(float(bake["voxelSize"])/2)
        box = node.node("container")
        try:
            low = [float(value) for value in bake["domainMin"]]
            size = [int(nodes)*float(bake["nodeSize"]) for nodes in bake["nodes"]]
        except (KeyError, TypeError, ValueError):
            box = None
        if box is not None:
            box.parmTuple("size").set(size)
            box.parmTuple("t").set([a + s/2 for a, s in zip(low, size)])
        committed, frames = bake.get("committed", -1), bake.get("frames", -1)
        node.parm("status").set("%d of frames 0-%d committed, at %g fps" % (committed + 1, frames, bake.get("fps", 24.0)) if committed < frames
                                else "frames 0-%d, at %g fps" % (frames, bake.get("fps", 24.0)))
    for name in (SURFACE, PARTICLES, FIELDS, WHITEWATER):
        loader = node.node(name)
        if loader is not None:
            loader.parm("reload").pressButton()


def shelf_tool(kwargs):
    """flip2 Import from the Tab menu or the shelf: asks for a bake and makes the node in the network being edited, inside a new geometry object if
    that's the object level"""
    pane = kwargs.get("pane")
    network = pane.pwd() if pane is not None else hou.node("/obj")
    position = pane.cursorPosition() if pane is not None else None
    if network.childTypeCategory() == hou.objNodeTypeCategory():
        container = network.createNode("geo", "flip2")
        if position is not None:
            container.setPosition(position)
        network, position = container, None
    elif network.childTypeCategory() != hou.sopNodeTypeCategory():
        raise hou.Error("flip2 Import goes in a geometry network")
    bake = hou.ui.selectFile(title="A flip2 bake: its directory, with cache.json in it", file_type=hou.fileType.Directory)
    node = create_import(network, bake.rstrip("/"))
    if position is not None:
        node.setPosition(position)
    node.setDisplayFlag(True)
    node.setRenderFlag(True)
    node.setSelected(True, clear_all_selected=True)
    return node
