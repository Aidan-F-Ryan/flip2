"""flip2 Whitewater: Houdini's own whitewater (foam, spray and bubbles), set up on a flip2 bake.

Houdini's Whitewater Source and Whitewater Solver take a liquid simulation as its fluid fields, its container and its collisions, which a flip2 Solver or
flip2 Import node has on its second, third and fourth outputs. This makes the two nodes, wired to those, and sizes them for the bake: Houdini's defaults
are for a scene tens of metres across (whitewater 10 cm apart, depths of a metre), so in a tank they make a handful of points. Here the whitewater's
separation is the bake's particle separation, and every length that goes with it (the solver's voxels, the depths either side of the foam layer) follows
it in proportion, as expressions: change Whitewater Scale on the solver, and they all follow.

What isn't set: what emits. Whitewater Source's Speed Range starts at 2 m/s, and nothing slower than that makes whitewater; a slow scene wants it lower.
"""
import glob
import os

import hou

#lengths that go with the whitewater's scale, set in proportion to it: in the solver, and in the source
SOLVER_LENGTHS = ("voxelsize", "depthrange", "projectionrange", "pbfdepthrange", "erosionrange", "depthcontrolrange", "wind_shadowmaxdistance", "depth_range")
SOURCE_LENGTHS = ("depthrange",)


def _follow(node, scale, names):
    """makes node's lengths follow its scale parameter: each keeps the ratio to it that Houdini's defaults have"""
    reference = node.parm(scale).parmTemplate().defaultValue()[0]
    for name in names:
        lengths = node.parmTuple(name)
        if lengths is None:     #not in this version of Houdini
            continue
        for length, default in zip(lengths, lengths.parmTemplate().defaultValue()):
            if default != 0:
                length.setExpression('ch("%s")*%g' % (scale, default/reference))


def create_whitewater(node):
    """Whitewater Source and Whitewater Solver for a flip2 Solver or flip2 Import node's bake, beside it. Returns them, and anything the artist has to
    do first (or None)"""
    kind = node.userData("flip2")
    if kind not in ("solver", "import"):
        raise hou.Error("flip2 Whitewater goes on a flip2 Solver or flip2 Import node: select one first")
    if len(node.outputConnectors()) < 4 or node.node("collisions") is None:
        raise hou.Error("%s is from an older flip2: run Update flip2 Nodes first" % node.path())
    parent = node.parent()
    source = parent.createNode("whitewatersource", node.name() + "_whitewater_source")
    solver = parent.createNode("whitewatersolver", node.name() + "_whitewater")
    for index in range(3):      #the liquid's fields, its container, its collisions
        source.setInput(index, node, index + 1)
        solver.setInput(index, source, index)
    solver.parm("scale").setExpression('ch("%s/particlesep")' % solver.relativePathTo(node))
    source.parm("wwscale").setExpression('ch("%s/scale")' % source.relativePathTo(solver))
    _follow(solver, "scale", SOLVER_LENGTHS)
    _follow(source, "wwscale", SOURCE_LENGTHS)
    for whitewater in (source, solver):     #the frame the bake starts on
        path = whitewater.relativePathTo(node)
        whitewater.parm("startframe").setExpression('ch("%s/startframe")' % path if kind == "solver" else 'ch("%s/timeoffset")*$FPS + 1' % path)
    source.setPosition(node.position() + hou.Vector2(2.5, -1.5))
    solver.setPosition(node.position() + hou.Vector2(2.5, -3.0))
    source.setComment("No whitewater? Lower Speed Range below the liquid's speed")
    source.setGenericFlag(hou.nodeFlag.DisplayComment, True)
    solver.setDisplayFlag(True)
    todo = None
    folder = os.path.join(node.evalParm("outputdir").rstrip("/"), "bake") if kind == "solver" else node.evalParm("bakedir").rstrip("/")
    if not glob.glob(os.path.join(hou.text.expandString(folder), "export", "houdini", "fields.[0-9]*.vdb")):
        if kind == "solver":
            node.parm("outputfields").set(True)
            todo = "This bake has no fluid fields yet. Output Fluid Fields is now on: press Mesh Bake on %s's Surface tab to make them, or bake again." % node.name()
        else:
            todo = "This bake has no fluid fields. Make them where it was baked: flip2 mesh BAKE --fields, or Output Fluid Fields on its flip2 Solver."
    return source, solver, todo


def shelf_tool(kwargs):
    """flip2 Whitewater from the Tab menu or the shelf: for the selected flip2 Solver or flip2 Import node"""
    selected = [node for node in hou.selectedNodes() if node.userData("flip2") in ("solver", "import")]
    if not selected:
        raise hou.Error("Select a flip2 Solver or flip2 Import node first: flip2 Whitewater sets Houdini's whitewater up on its bake")
    source, solver, todo = create_whitewater(selected[0])
    solver.setSelected(True, clear_all_selected=True)
    if todo is not None and hou.isUIAvailable():
        hou.ui.displayMessage(todo)
    return solver
