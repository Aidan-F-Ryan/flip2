"""flip2 Import: a bake's surface and particles, at the current time, from the .bgeo files `flip2 mesh` and `flip2 export` write
(BAKE/export/houdini/surface.NNNN.bgeo and particles.NNNN.bgeo).

A flip2 frame is a time: frame N is N/fps seconds after the bake's start. Houdini's frame 1 is time 0, so loading by time puts the bake's frame 0 on
Houdini's frame 1, and keeps it there whatever frame rate either uses. A frame the bake hasn't made (yet) loads as no geometry.
"""
import glob
import json
import os

import hou

PARTICLES = "particles"
SURFACE = "surface"
SHOW = (("surface", "Surface"), ("particles", "Particles"), ("both", "Surface and Particles"))


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
        loader.parm("file").set('%s/%s.`padzero(4, %s)`.bgeo' % (folder, name, frame))
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


def has_surface(folder):
    """whether a bake's export/houdini directory holds any surface frames"""
    return bool(glob.glob(os.path.join(hou.text.expandString(folder), SURFACE + ".[0-9]*.bgeo")))


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
    group.append(show_parm())
    group.append(hou.ButtonParmTemplate("reload", "Reload", script_callback="__import__('flip2houdini').importer.reload(kwargs['node'])",
                                        script_callback_language=hou.scriptLanguage.Python,
                                        help="Read the bake's cache.json again, and the frame on show from disk"))
    status = hou.StringParmTemplate("status", "Status", 1, help="What the bake holds")
    status.setDisableWhen("{ fps >= 0 }")     #always: it's only to read
    group.append(status)
    node.setParmTemplateGroup(group)


def _network(node):
    """the node's network: its loaders, by the frame showing at this time (a bake's frame N is at N/fps)"""
    return build_loaders(node, '`chs("../bakedir")`/export/houdini', 'round(($T - ch("../timeoffset"))*ch("../fps"))')


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
    reload(node)
    return node


def reload(node):
    """reads the bake's cache.json into the node's frame rate and status, and its frame on show from disk again"""
    bake = read_bake(node.evalParm("bakedir"))
    if bake is None:
        node.parm("status").set("no bake there (no cache.json)" if node.evalParm("bakedir") else "")
    else:
        node.parm("fps").set(float(bake.get("fps", 24.0)))
        committed, frames = bake.get("committed", -1), bake.get("frames", -1)
        node.parm("status").set("%d of frames 0-%d committed, at %g fps" % (committed + 1, frames, bake.get("fps", 24.0)) if committed < frames
                                else "frames 0-%d, at %g fps" % (frames, bake.get("fps", 24.0)))
    for name in (SURFACE, PARTICLES):
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
