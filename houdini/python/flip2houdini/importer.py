"""flip2 Import: a bake's particles, at the current time, from the .bgeo files `flip2 export` writes (BAKE/export/houdini/particles.NNNN.bgeo).

A flip2 frame is a time: frame N is N/fps seconds after the bake's start. Houdini's frame 1 is time 0, so loading by time puts the bake's frame 0 on
Houdini's frame 1, and keeps it there whatever frame rate either uses. A frame the bake hasn't made (yet) loads as no geometry.
"""
import json
import os

import hou

PARTICLES = "particles"


def read_bake(directory):
    """a bake's cache.json, or None if there isn't one (yet)"""
    try:
        with open(os.path.join(hou.text.expandString(directory), "cache.json")) as record:
            return json.load(record)
    except (OSError, ValueError):
        return None


def _interface(node):
    group = node.parmTemplateGroup()
    group.append(hou.StringParmTemplate("bakedir", "Bake", 1, string_type=hou.stringParmType.FileReference, file_type=hou.fileType.Directory,
                                        help="The bake's directory: the one with cache.json in it",
                                        script_callback="__import__('flip2houdini').importer.reload(kwargs['node'])",
                                        script_callback_language=hou.scriptLanguage.Python))
    group.append(hou.FloatParmTemplate("fps", "Bake Frame Rate", 1, default_value=(24.0,), min=1.0,
                                       help="Frames per second of the bake, read from its cache.json"))
    group.append(hou.FloatParmTemplate("timeoffset", "Time Offset", 1, default_value=(0.0,),
                                       help="Seconds to delay the bake by: its start shows at this time"))
    group.append(hou.ButtonParmTemplate("reload", "Reload", script_callback="__import__('flip2houdini').importer.reload(kwargs['node'])",
                                        script_callback_language=hou.scriptLanguage.Python,
                                        help="Read the bake's cache.json again, and the frame on show from disk"))
    status = hou.StringParmTemplate("status", "Status", 1, help="What the bake holds")
    status.setDisableWhen("{ fps >= 0 }")     #always: it's only to read
    group.append(status)
    node.setParmTemplateGroup(group)


def create_import(parent, bake_dir="", name="flip2_import"):
    """a flip2 Import node in the SOP network parent, loading bake_dir's particles"""
    node = parent.createNode("subnet", name)
    _interface(node)
    particles = node.createNode("file", PARTICLES)
    #the frame showing at this time: a bake's frame N is at N/fps
    particles.parm("file").set('`chs("../bakedir")`/export/houdini/particles.`padzero(4, round(($T - ch("../timeoffset"))*ch("../fps")))`.bgeo')
    particles.parm("missingframe").set("empty")
    output = node.createNode("output", "output0")
    output.setInput(0, particles)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)
    node.layoutChildren()
    node.parm("bakedir").set(bake_dir)
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
    particles = node.node(PARTICLES)
    if particles is not None:
        particles.parm("reload").pressButton()


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
