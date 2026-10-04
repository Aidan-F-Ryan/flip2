"""flip2 Karma Setup: a flip2 bake, ready to render in Karma.

Builds in /stage what a first render needs: the liquid's surface with a water material, the collision geometry, Houdini's whitewater if flip2 Whitewater
set it up, a ground under the domain, a sky, a camera that takes the domain in, Karma's render settings and a USD Render node to render with.

One of those settings is the reason to have this at all. flip2's surface carries the liquid's velocity as v, which Solaris imports as USD's velocities,
and Karma blurs a moving mesh by them, but only when its Velocity Blur says to, and it starts at No Velocity Blur: a flip2 surface renders with no
motion blur at all until that's changed. Here it's on.

Whitewater is rendered as points as wide as they are apart (the solver's pscale makes each twice that, which renders as beads), and the sun is on the
camera's side: where Karma Physical Sky starts it, behind the scene, whitewater and everything else facing the camera is in shade, and reads as grey.
"""
import math

import hou

LOOKS = {     #MaterialX standard surfaces
    "water": dict(base=0.0, specular=1.0, specular_roughness=0.02, specular_IOR=1.33, transmission=1.0, transmission_depth=0.6,
                  transmission_colorr=0.55, transmission_colorg=0.8, transmission_colorb=0.9),
    "whitewater": dict(base=1.0, base_colorr=0.95, base_colorg=0.97, base_colorb=1.0, specular=0.2, specular_roughness=0.6),
    "solid": dict(base=1.0, base_colorr=0.35, base_colorg=0.33, base_colorb=0.3, specular_roughness=0.5),
    "ground": dict(base=1.0, base_colorr=0.45, base_colorg=0.45, base_colorb=0.45, specular_roughness=0.7),
}
WHITEWATER_WIDTH = 0.5      #a whitewater point's radius, as a fraction of the whitewater's separation: so each is as wide as they are apart
FOCAL_LENGTH, APERTURE, ASPECT = 30.0, 20.955, 16.0/9.0


def _domain(node):
    """the centre and size of the box the bake filled"""
    if node.userData("flip2") == "solver":
        return node.evalParmTuple("domaincenter"), node.evalParmTuple("domainsize")
    box = node.node("container")
    return box.evalParmTuple("t"), box.evalParmTuple("size")


def create_render(node):
    """a Karma setup in /stage for a flip2 Solver or flip2 Import node's bake. Returns its USD Render node"""
    kind = node.userData("flip2")
    if kind not in ("solver", "import"):
        raise hou.Error("flip2 Karma Setup goes on a flip2 Solver or flip2 Import node: select one first")
    if node.node("surface") is None:
        raise hou.Error("%s is from an older flip2: run Update flip2 Nodes first" % node.path())
    stage = hou.node("/stage") or hou.node("/").createNode("lopnet", "stage")
    prefix = node.name()
    made, looks = [], {}

    def imported(what, sop, look):
        lop = stage.createNode("sopimport", "%s_%s" % (prefix, what))
        lop.parm("soppath").set(sop.path())
        if made:
            lop.setInput(0, made[-1])
        made.append(lop)
        looks["/" + lop.name()] = look

    imported("liquid", node.node("surface"), "water")     #the surface itself, whatever the node's Show says
    whitewater = node.parent().node(prefix + "_whitewater")
    if whitewater is not None:      #sized to render: the solver's pscale is the points' whole separation
        sized = node.parent().node(prefix + "_whitewater_render")
        if sized is None:
            sized = node.parent().createNode("attribwrangle", prefix + "_whitewater_render")
            sized.setInput(0, whitewater)
            sized.parm("snippet").set("@pscale *= %g;     // the fraction of the whitewater's separation each point is drawn as" % WHITEWATER_WIDTH)
            sized.setPosition(whitewater.position() + hou.Vector2(0.0, -1.5))
        imported("whitewater", sized, "whitewater")
    wired = node.inputs()
    collision = (node.node("COLLISION") if len(wired) > 1 and wired[1] is not None else None) if kind == "solver" else (wired[0] if wired else None)
    if collision is not None:
        imported("collisions", collision, "solid")
    centre, size = _domain(node)
    ground = stage.createNode("cube", prefix + "_ground")       #a slab under the domain, three times as wide
    ground.setInput(0, made[-1])
    ground.parm("size").set(1.0)
    ground.parmTuple("s").set((3.0*size[0], 0.01*size[1], 3.0*max(size[2], size[0])))
    ground.parmTuple("t").set((centre[0], centre[1] - 0.5*size[1] - 0.006*size[1], centre[2]))
    made.append(ground)
    looks[ground.evalParm("primpath")] = "ground"
    library = stage.createNode("materiallibrary", prefix + "_materials")
    library.setInput(0, made[-1])
    library.parm("materials").set(len(looks))
    shaders = {}
    for index, (prim, look) in enumerate(looks.items(), 1):
        if look not in shaders:
            shaders[look] = library.createNode("mtlxstandard_surface", look)
            for parm, value in LOOKS[look].items():
                shaders[look].parm(parm).set(value)
        library.parm("matnode%d" % index).set(look)
        library.parm("matpath%d" % index).set("/materials/" + look)
        library.parm("assign%d" % index).set(1)
        library.parm("geopath%d" % index).set(prim)
    library.layoutChildren()
    #a camera in front of the domain (+z), far enough to take its width and height in, a little above its middle and looking down at it
    half = math.atan(APERTURE / (2.0*FOCAL_LENGTH))
    distance = max(0.55*size[0] / math.tan(half), 0.55*size[1] / math.tan(math.atan(math.tan(half) / ASPECT))) + 0.5*size[2]
    camera = stage.createNode("camera", prefix + "_camera")
    camera.setInput(0, library)
    camera.parm("focalLength").set(FOCAL_LENGTH)
    camera.parmTuple("t").set((centre[0], centre[1] + 0.45*size[1], centre[2] + distance))
    camera.parmTuple("r").set((-math.degrees(math.atan(0.5*size[1] / distance)), 0.0, 0.0))
    sky = stage.createNode("karmaphysicalsky", prefix + "_sky")
    sky.setInput(0, camera)
    sky.parm("solar_altitude").set(50.0)
    sky.parm("solar_azimuth").set(150.0)    #from the camera's side and a little to one: at its 0 the sun is behind the scene, which leaves everything facing the camera dark
    settings = stage.createNode("karmarenderproperties", prefix + "_karma")
    settings.setInput(0, sky)
    settings.parm("camera").set(camera.evalParm("primpath"))
    settings.parm("vblur").set("Velocity Blur")     #or the surface's v blurs nothing
    render = stage.createNode("usdrender_rop", prefix + "_render")
    render.setInput(0, settings)
    stage.layoutChildren()
    settings.setDisplayFlag(True)
    return render


def shelf_tool(kwargs):
    """flip2 Karma Setup from the Tab menu or the shelf: for the selected flip2 Solver or flip2 Import node"""
    selected = [node for node in hou.selectedNodes() if node.userData("flip2") in ("solver", "import")]
    if not selected:
        raise hou.Error("Select a flip2 Solver or flip2 Import node first: flip2 Karma Setup builds a stage to render its bake")
    render = create_render(selected[0])
    if hou.isUIAvailable():
        hou.ui.displayMessage("Built a Karma setup in /stage for %s." % selected[0].name(),
                              details="Open /stage to see it. %s renders the frame range; its picture is set on %s. Velocity Blur is on, so the "
                                      "surface's motion blurs." % (render.name(), render.inputs()[0].name()))
    return render
