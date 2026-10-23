"""flip2 Solver: a flip2 simulation set up from Houdini geometry, baked by the flip2 program, and loaded back, as File Cache loads what it wrote.

Its inputs are geometry, as Houdini's own FLIP Solver takes it:

    1 Fluid        closed surfaces: the fluid at the start frame
    2 Collisions   closed surfaces the fluid flows around (open ones too, given a thickness): still, moving or deforming
    3 Sources      closed surfaces kept full of fluid moving at the emission velocity (or their v attribute's mean)
    4 Sinks        closed surfaces that remove the fluid inside them

Each input is one object, or with a "name" primitive attribute, one per name. A collision object's geometry can carry its own settings, as primitive
attributes with one value per object: flip2_friction, flip2_hold (0 or 1) and flip2_thickness; without them it gets the Collisions tab's. Packed
geometry is unpacked, every primitive turned into polygons and the polygons into triangles. An object is sampled every frame of the range, and goes to flip2 as what it does: still, its vertices once; moving rigidly (no
vertex further than a hundredth of a voxel from where a turn and a shift put it), its vertices once and a transform per frame, which flip2 turns it
through exactly; or deforming, its vertices every frame, which flip2 re-voxelizes between, so its points and triangles mustn't change. flip2 takes
any number of collisions, each named object on its own.

VDBs in the Collisions input stay volumes: a level set that's still (with a velocity VDB beside it, if there is one: named after it with "vel" on the
end, or "vel" or "v") or moved rigidly goes to flip2 as it is. One that changes shape over time flip2 can't take yet.

RBD pieces come from a SOP named on the RBD tab (a subnet has only the four inputs), as RBD Bullet Solver's Geometry input takes them: packed
pieces with a name each, and its point attributes meaning what they mean to it (active, density or mass, friction, bounce, v, w; its defaults without
them). flip2 simulates the active ones with the liquid, two-way: each piece moves the liquid as the liquid moves it, and they collide with each other,
the walls and the collisions, Collision Substeps times a frame at least. The others collide, as the Collisions input's objects do. flip2 simulates
any number of pieces, as many as fit in the GPU's memory.

Forces are set on the Forces tab, 16 in all. Force Fields are made there: a point that pulls the fluid in or pushes it out, a vortex that swirls it
around an axis, turbulence that stirs it, or wind that drags its surface along. Force Volumes come from a SOP too, but one named on the tab, as a
subnet has only the four inputs: every vector VDB it holds is a force on the fluid wherever it has active voxels, and nothing elsewhere. Its vectors
are accelerations, which push the fluid, or velocities, which the fluid inside it is drawn to at Drag a second: Volumes Are says which, or leaves it
to each volume's name. flip2 takes both as they are at the start frame; it doesn't yet take forces that change over time.

Everything goes in the output directory ($HIP/geo/<scene>.<node> by default, beside File Cache's caches): scene.json and geo/ (what the node writes), bake/ (flip2's cache, which a
running bake commits frames to, and export/houdini/ beside it) and logs/. The bake's frame 0 is the start frame, and the node shows it there: its
surface, its particles or both (Show).

The surface is meshed on the GPU as each frame is committed (flip2 mesh, beside flip2 export), with the Surface tab's settings: closed quads facing
outwards, their points carrying v for motion blur, flat against the domain's walls, and carried on into the collisions and RBD pieces where the water
touches them (or cut at their surfaces: Water at Collisions). Mesh Bake meshes a bake's frames again, with other settings or after a bake made without
them.

Its outputs: 1, the bake's surface, particles or both (Show); 2, its fluid fields, with Output Fluid Fields (surface, a level set, and vel, as Houdini's
FLIP outputs them); 3, the domain, and 4, the collisions, both as Houdini's own FLIP nodes make them from the domain, the Collisions input and the RBD
pieces where they've been moved. Outputs 2 to 4 are what Whitewater Source takes as its Liquid Simulation, Container and Collisions: flip2 Whitewater
(whitewater.py) sets Houdini's up on them. 5, the bake's own whitewater, and with air the air's own bubbles as whitewater bubbles (kind 2): the air
mixed into the liquid finer than its voxels. 6 and 7 are as RBD Bullet Solver's Geometry and Simulation Points outputs: the
RBD pieces moved where flip2 moved them (the simulated ones by Transform Pieces from the start frame; the others as the geometry has them), and the
points that moved them, one per piece with name, P, orient, pivot, scale, v and w.

Bake On picks where flip2 runs: this machine, or another over ssh (a GPU box, for a Mac). A remote bake sends scene.json and geo/ to the remote
directory, runs there detached (remote/job.sh), and is mirrored back every couple of seconds (remote/mirror.sh): its logs, its cache.json and its
exported and meshed frames, so it shows here as a local one does. It needs ssh to reach the host without a password (a key), and rsync at both ends.
Find GPUs asks that machine for its GPUs (flip2 info) and gives each a box to tick: the bake and its meshing see only the ticked ones, by their
UUIDs in CUDA_VISIBLE_DEVICES. With every box ticked, or before Find GPUs, they see all of them.

Particle separation and grid scale are Houdini's: flip2's voxels are twice the particle separation, the 2 x 2 x 2 particles per voxel Houdini's FLIP
seeds at its default grid scale of 2.

The Simulation tab's Phases picks what is simulated: the liquid alone (fast; the domain's walls and the collisions each hold the liquid or let it go,
Walls and Collisions tabs), or the liquid and the air around it (2-8x the time; the Air tab).
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
SOLIDS = 'if(primintrinsic(0, "typename", @primnum) == "VDB") removeprim(0, @primnum, 1);'      #the Collisions input without its VDBs, and only them
VOLUMES = 'if(primintrinsic(0, "typename", @primnum) != "VDB") removeprim(0, @primnum, 1);'
VECTORS = 'if(!match("vec3*", primintrinsic(0, "vdb_value_type", @primnum))) removeprim(0, @primnum, 1);'     #of VDBs, only those of vectors
VOLUME_FILE = "collision_volumes.vdb"
FORCE_FILE = "force_volumes.vdb"
MESH_VERTICES = "meshes_vertices.npy"       #every object's vertices where it starts, one after another (_Packer)
MESH_TRIANGLES = "meshes_triangles.npy"     #and their triangles, each numbering its own vertices from 0
DEFORMING_FILE = "deforming_f%d.npy"        #every deforming object's vertices at a frame
SETTINGS = (("friction", "flip2_friction", float), ("hold", "flip2_hold", int), ("thickness", "flip2_thickness", float))  #a collision object's own, as primitive attributes
FORCE_LIMIT = 16    #and the forces
FIELD_TYPES = ("point", "vortex", "turbulence", "wind")     #the Forces tab's force fields, as the scene names them
FIELD_SETTINGS = {  #each one's scene keys, and the parameters of a Force Fields instance they come from
    "point": (("position", "fieldpos"), ("strength", "fieldstrength"), ("radius", "fieldradius"), ("falloff", "fieldfalloff")),
    "vortex": (("position", "fieldpos"), ("axis", "fieldaxis"), ("strength", "fieldstrength"), ("radius", "fieldradius"), ("falloff", "fieldfalloff")),
    "turbulence": (("strength", "fieldstrength"), ("scale", "fieldscale"), ("speed", "fieldspeed"), ("seed", "fieldseed")),
    "wind": (("velocity", "fieldvel"), ("drag", "fielddrag"), ("depth", "fieldwinddepth")),     #depth in metres here, in voxels in the scene
}
SURFACE_DEFAULTS = {"influencescale": (3.0,), "radiusscale": (0.8,), "smoothing": (4,)}     #flip2 mesh's own
OLD_SURFACE_DEFAULTS = {"influencescale": 2.0, "radiusscale": 0.6, "smoothing": 2}         #what they were, which left the surface dimpled
_jobs = {}      #per node path, its bake's processes, while this session runs them
#the files write_scene writes into geo/, and the ones an object each it wrote before it packed them
GEO_FILE = re.compile(r"^((%s)\d+_(triangles|vertices|f-?\d+)\.npy|meshes_(vertices|triangles)\.npy|deforming_f-?\d+\.npy|(collision|force)_volumes\.vdb)$"
                      % "|".join(ROLES + ("rbd",)))
RBD_DEFAULTS = {"density": 1000.0, "friction": 1.0, "bounce": 0.25}     #RBD Bullet Solver's, for a piece without the attribute
PIECE_TAG = '''//a packed piece's name, as a point attribute or its primitive's, which unpacking hands its geometry
string piece = point(0, "name", @ptnum);
if(!haspointattrib(0, "name")){
    int prims[] = pointprims(0, @ptnum);
    piece = len(prims) > 0 ? prim(0, "name", prims[0]) : "";
}
s@flip2_piece = piece;
'''


def _callback(function):
    return dict(script_callback="__import__('flip2houdini').solver.%s(kwargs['node'])" % function, script_callback_language=hou.scriptLanguage.Python)


def _hidden(template, condition):
    """template, hidden when condition holds"""
    template.setConditional(hou.parmCondType.HideWhen, condition)
    return template


def _force_fields_interface():
    """the Forces tab's Force Fields: a block of settings per field, those that its Type doesn't use hidden"""
    fields = hou.FolderParmTemplate("forcefields", "Force Fields", folder_type=hou.folderType.MultiparmBlock)
    fields.setHelp("Forces on the fluid besides gravity, made here: each a point, a vortex, turbulence or wind. Up to 16, with the Force Volumes")
    fields.addParmTemplate(hou.ToggleParmTemplate("fieldenable#", "Enable", default_value=True))
    fields.addParmTemplate(hou.MenuParmTemplate("fieldtype#", "Type", FIELD_TYPES, ("Point", "Vortex", "Turbulence", "Wind"), default_value=0,
                                                help="Point pulls the fluid towards a point, or pushes it away. Vortex swirls it around an axis. "
                                                     "Turbulence stirs it with swirls of a size you set. Wind drags the liquid's surface along, as "
                                                     "wind does water"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldpos#", "Position", 3, default_value=(0.0, 0.5, 0.0),
                                                         help="Point: where it pulls towards. Vortex: a point on its axis"),
                                   "{ fieldtype# == turbulence } { fieldtype# == wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldaxis#", "Axis", 3, default_value=(0.0, 1.0, 0.0),
                                                         help="The line through Position the fluid swirls around, by the right-hand rule: with your "
                                                              "thumb along it, the fluid goes the way your fingers curl. Any length but 0"),
                                   "{ fieldtype# != vortex }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldstrength#", "Strength", 1, default_value=(2.0,), min=-20.0, max=20.0,
                                                         help="Its acceleration in m/s2 (gravity's is 9.8), the same at any distance until Radius fades "
                                                              "it. Point: towards Position, or away from it when negative. Vortex: around the axis, the "
                                                              "other way round when negative. Turbulence: the swirls' typical acceleration"),
                                   "{ fieldtype# == wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldradius#", "Radius", 1, default_value=(0.0,), min=0.0, max=10.0,
                                                         help="How far it reaches, from Position (Point) or from the axis (Vortex): it fades to nothing "
                                                              "there. 0 for no fade, so it reaches everywhere"),
                                   "{ fieldtype# == turbulence } { fieldtype# == wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldfalloff#", "Falloff", 1, default_value=(1.0,), min=0.0, max=4.0,
                                                         disable_when="{ fieldradius# == 0 }",
                                                         help="How it fades out to Radius, as (1 - distance/radius)^falloff: 1 fades it evenly, more keeps "
                                                              "it near the middle, 0 keeps it at full strength right up to Radius"),
                                   "{ fieldtype# == turbulence } { fieldtype# == wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldscale#", "Scale", 1, default_value=(0.1,), min=0.001, max=2.0,
                                                         help="The size of its swirls, in metres"),
                                   "{ fieldtype# != turbulence }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldspeed#", "Speed", 1, default_value=(0.5,), min=0.0, max=5.0,
                                                         help="How fast its pattern of swirls changes, in swirls a second: 0 holds it still"),
                                   "{ fieldtype# != turbulence }"))
    fields.addParmTemplate(_hidden(hou.IntParmTemplate("fieldseed#", "Seed", 1, default_value=(0,), min=0, max=100,
                                                       help="Another pattern of swirls, of the same size and strength"),
                                   "{ fieldtype# != turbulence }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldvel#", "Velocity", 3, default_value=(2.0, 0.0, 0.0),
                                                         help="The wind's velocity, in m/s, which the liquid's surface is drawn towards"),
                                   "{ fieldtype# != wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fielddrag#", "Drag", 1, default_value=(1.0,), min=0.0, max=10.0,
                                                         help="How fast the surface takes up the wind's velocity, per second: at 1 it closes about two "
                                                              "thirds of the gap in a second, at 10 most of it in a tenth of one"),
                                   "{ fieldtype# != wind }"))
    fields.addParmTemplate(_hidden(hou.FloatParmTemplate("fieldwinddepth#", "Depth", 1, default_value=(0.08,), min=0.0, max=1.0,
                                                         help="How far below the liquid's surface the wind reaches, in metres. flip2 counts it in "
                                                              "voxels (twice the Particle Separation each), so it takes the nearest whole number of "
                                                              "them: the same depth at any resolution, to within half a voxel"),
                                   "{ fieldtype# != wind }"))
    return fields


def _interface(node):
    """the node's parameters, from the subnet's own on: applied again to an existing node (update), it keeps the values of the parameters it still has"""
    group = node.type().parmTemplateGroup()
    simulation = hou.FolderParmTemplate("simulation", "Simulation", folder_type=hou.folderType.Tabs)
    simulation.addParmTemplate(hou.MenuParmTemplate("phases", "Phases", ("liquid", "air"), ("Liquid Only (Fast)", "Liquid and Air (Realistic, 2-8x the Time)"), default_value=0,
                                                    help="Liquid Only simulates the liquid with nothing where it isn't: fast, and right for most shots, but "
                                                         "with no air to get in behind it the liquid clings to ceilings and the undersides of collisions "
                                                         "unless they let it go (Walls and Collisions tabs), nothing cushions a splash, and no bubbles are "
                                                         "carried under. Liquid and Air simulates the air around the liquid as well: entrained bubbles, "
                                                         "splash crowns, water that pours through a hole only as air rises past it, and nothing clings. "
                                                         "Measured on the sample scenes: 2 to 8 times the bake time. The air's own settings are on the Air tab"))
    simulation.addParmTemplate(hou.FloatParmTemplate("particlesep", "Particle Separation", 1, default_value=(0.02,), min=0.0001,
                                                     help="The distance between particles at rest. flip2's voxels are twice this"))
    simulation.addParmTemplate(hou.FloatParmTemplate("domaincenter", "Domain Center", 3, default_value=(0.0, 0.5, 0.0)))
    simulation.addParmTemplate(hou.FloatParmTemplate("domainsize", "Domain Size", 3, default_value=(2.0, 1.0, 2.0)))
    simulation.addParmTemplate(hou.ButtonParmTemplate("fitdomain", "Fit Domain to Inputs", help="The domain around every input's geometry over the frame range",
                                                      **_callback("fit_domain")))
    simulation.addParmTemplate(hou.IntParmTemplate("startframe", "Start Frame", 1, default_expression=("$FSTART",)))
    simulation.addParmTemplate(hou.IntParmTemplate("endframe", "End Frame", 1, default_expression=("$FEND",)))
    simulation.addParmTemplate(hou.MenuParmTemplate("transfer", "Velocity Transfer", ("flip", "apic"), ("FLIP (Splashy)", "APIC (Swirly)"), default_value=0))
    simulation.addParmTemplate(hou.MenuParmTemplate("freesurface", "Free Surface", ("voxel", "levelset"), ("Voxel (Fast/Damped)", "Level Set (Accurate)"),
                                                    default_value=1, disable_when="{ phases == 1 }",
                                                    help="Where the pressure solve puts the liquid's surface. Level Set: at the particles' own surface, found as a "
                                                         "level set each substep, with the pressure set exactly there (the ghost fluid method), so drops "
                                                         "oscillate and waves travel at the right speed, the liquid keeps its volume, and floating objects sit "
                                                         "at the right height; calm water stays a little lively. Voxel: every voxel the particles touch is "
                                                         "liquid, and the surface is the edge of those voxels, a voxel or two outside the particles (marker "
                                                         "cells), which settles quickly, damps small waves, and bakes in about half the time; not with dynamic "
                                                         "objects, which would float high. Not with air yet: with Liquid and Air the surface is where the two "
                                                         "fluids meet, and Voxel is used"))
    simulation.addParmTemplate(hou.FloatParmTemplate("flipratio", "FLIP Ratio", 1, default_value=(0.95,), min=0.0, max=1.0,
                                                     help="How much of each particle's own velocity it keeps: 0 is pure PIC (or pure APIC), 1 pure FLIP"))
    simulation.addParmTemplate(hou.FloatParmTemplate("cfl", "CFL Condition", 1, default_value=(4.0,), min=0.1, max=4.0,
                                                     help="How many voxels the fastest particle may move in a substep"))
    simulation.addParmTemplate(hou.FloatParmTemplate("gravity", "Gravity", 3, default_value=(0.0, -9.8, 0.0)))
    simulation.addParmTemplate(hou.FloatParmTemplate("densitytime", "Density Correction Time", 1, default_value=(0.1,), min=0.0,
                                                     help="Seconds over which crowded or sparse fluid is brought back to its rest density; 0 for none"))
    group.append(simulation)
    liquid = hou.FolderParmTemplate("liquid", "Liquid", folder_type=hou.folderType.Tabs)
    liquid.addParmTemplate(hou.FloatParmTemplate("density", "Density", 1, default_value=(1000.0,), min=1.0, max=20000.0,
                                                 help="kg/m3: water 1000, honey 1400. Only viscosity and surface tension use it: what moves the liquid is each "
                                                      "of them over its density"))
    liquid.addParmTemplate(hou.FloatParmTemplate("viscosity", "Viscosity", 1, default_value=(0.0,), min=0.0, max=100.0,
                                                 help="Pa s: water 0.001, olive oil 0.1, honey 2 to 10, molasses 10 to 100; 0 for none. Viscous liquid "
                                                      "sticks to the domain's walls and to collisions. Any value over 0 adds a viscous solve to every "
                                                      "substep (up to twice the bake time with air simulated), and water's 0.001 shows nothing at "
                                                      "voxels over a millimetre or two: leave it 0 unless the liquid is thick"))
    liquid.addParmTemplate(hou.FloatParmTemplate("viscouscfl", "Viscous CFL", 1, default_value=(6.0,), min=0.0, max=20.0,
                                                 disable_when="{ viscosity == 0 }",
                                                 help="How many voxels viscosity may spread across in a substep, as CFL Condition is for how far the fastest "
                                                      "particle moves: the substeps shorten to keep to both. Past about 6, threads of thick liquid fold "
                                                      "from side to side instead of coiling. 0 for no limit, which is fine for liquid that only spreads "
                                                      "and pools. Thicker liquid and smaller voxels need more substeps: honey at 2.5 mm voxels about 2 a "
                                                      "frame, at 1 mm about 9"))
    liquid.addParmTemplate(hou.FloatParmTemplate("surfacetension", "Surface Tension", 1, default_value=(0.0,), min=0.0, max=1.0,
                                                 help="N/m: water 0.073; 0 for none. It only shows on liquid a few centimetres across or less, so at "
                                                      "voxels of a centimetre it costs (a surface pass every substep) and shows nothing. It shortens the "
                                                      "timestep: to sqrt(density x voxel^3 / (2 pi x surface tension)), 6 ms at 2.5 mm voxels for water "
                                                      "and 0.5 ms at 0.5 mm"))
    liquid.addParmTemplate(hou.FloatParmTemplate("contactangle", "Contact Angle", 1, default_value=(60.0,), min=0.0, max=180.0,
                                                 disable_when="{ surfacetension == 0 }",
                                                 help="Degrees between the liquid's surface and the domain's walls where they meet, measured through the "
                                                      "liquid. Under 90 the liquid wets them: it spreads along them, and a splash's crater in a shallow pool "
                                                      "closes again; water on glass is about 30, on most other things nearer 60. Over 90 it beads up on "
                                                      "them, and a pool shallower than a few millimetres pulls back from a dry patch. Collisions don't have "
                                                      "one yet: liquid beads on them a little"))
    group.append(liquid)
    air = hou.FolderParmTemplate("airfolder", "Air", folder_type=hou.folderType.Tabs)    #the second fluid, when the Simulation tab's Phases asks for it
    no_air = "{ phases == 0 }"
    air.addParmTemplate(hou.IntParmTemplate("airband", "Air Band", 1, default_value=(8,), min=1, max=64, disable_when=no_air,
                                            help="Voxels of air simulated around the liquid, in whole blocks of 4; past the band the pressure is the open "
                                                 "air's, and the air flows out and in there freely. Air shut in, as under a lid or a plate with holes, "
                                                 "isn't simulated from this node yet (that needs the air everywhere, which flip2 takes only with its "
                                                 "analytic liquid shapes, not geometry). The timestep follows the liquid: where the liquid lands, the "
                                                 "air it squeezes out goes many times faster, and is let go a bounded distance a substep rather than "
                                                 "shortening every substep to it"))
    air.addParmTemplate(hou.FloatParmTemplate("airdensity", "Air Density", 1, default_value=(1.2,), min=0.001, max=1000.0, disable_when=no_air,
                                              help="kg/m^3: 1.2 is air at room temperature. Heavier air drags more on spray and slows the bubbles' rise. "
                                                   "It can't be heavier than the liquid"))
    air_advanced = hou.FolderParmTemplate("airadvanced", "Advanced", folder_type=hou.folderType.Collapsible)
    air_advanced.addParmTemplate(hou.ToggleParmTemplate("airescaped", "Droplets and Bubbles Fly", default_value=True, disable_when=no_air,
                                                        help="A particle of liquid that finds itself in the air flies as a droplet, with the air's drag on it, "
                                                             "and one of air in the liquid rises as a bubble, rather than taking the grid's velocity where "
                                                             "it is. Off only for comparing"))
    air_advanced.addParmTemplate(hou.FloatParmTemplate("airdropletradius", "Droplet Radius", 1, default_value=(0.0,), min=0.0, max=0.1, disable_when=no_air,
                                                       help="Metres, for the air's drag on a flying droplet: 0 for a particle's worth of liquid"))
    air_advanced.addParmTemplate(hou.FloatParmTemplate("airbubbleradius", "Bubble Radius", 1, default_value=(0.0,), min=0.0, max=0.1, disable_when=no_air,
                                                       help="Metres: every bubble's radius, for the liquid's drag on it as it rises. 0 sizes each bubble as it escapes, from the air it escaped from and how hard the water churns there"))
    air_advanced.addParmTemplate(hou.FloatParmTemplate("airviscosity", "Air Viscosity", 1, default_value=(1.5e-5,), min=1.0e-7, max=1.0e-2, disable_when=no_air,
                                                       help="The air's kinematic viscosity, m^2/s, for its drag on droplets: 1.5e-5 is air's"))
    air.addParmTemplate(air_advanced)
    group.append(air)
    walls = hou.FolderParmTemplate("walls", "Walls", folder_type=hou.folderType.Tabs)  #the domain's six sides: what each is, and what it does to the liquid on it
    for index, face in enumerate(FACES):
        open_side = "{ closed%d == 0 }" % index
        walls.addParmTemplate(hou.ToggleParmTemplate("closed%d" % index, "%s Closed" % face.upper(), default_value=True, join_with_next=True,
                                                     help="A closed side is a wall. An open one is an outflow: the fluid that reaches it leaves the simulation"))
        walls.addParmTemplate(hou.ToggleParmTemplate("wallhold%d" % index, "Holds Liquid", default_value=True, join_with_next=True,
                                                     disable_when=open_side,
                                                     help="On, the wall pulls on the liquid as well as pushing it, as a sealed tank's does: with no air "
                                                          "simulated nothing can get in behind the liquid, so a splash that reaches the ceiling hangs "
                                                          "there. Off, the wall lets the liquid go wherever air can reach it, as anything standing in "
                                                          "open air does, and still holds it where none can (a lid over a full tank). It costs a second "
                                                          "pressure solve in the substeps where liquid pulls on the wall: a few percent more bake time "
                                                          "for a ceiling, up to half as much again with every wall and collision letting go. With "
                                                          "Liquid and Air simulated the air gets in behind the liquid itself and this does nothing"))
        walls.addParmTemplate(hou.FloatParmTemplate("wallfriction%d" % index, "Friction", 1, default_value=(0.0,), min=0.0, max=1.0, join_with_next=True,
                                                    disable_when=open_side,
                                                    help="As a collision's: 0, the liquid slides along the wall freely; 1, the liquid touching it is at rest"))
        walls.addParmTemplate(hou.FloatParmTemplate("wallangle%d" % index, "Contact Angle", 1, default_value=(60.0,), default_expression=('ch("contactangle")',),
                                                    default_expression_language=(hou.scriptLanguage.Hscript,), min=0.0, max=180.0,
                                                    disable_when="{ closed%d == 0 } { surfacetension == 0 }" % index,
                                                    help="Degrees between the liquid's surface and this wall where they meet. It follows the Liquid tab's "
                                                         "Contact Angle until it's given a value of its own: a floor the liquid beads on under walls it "
                                                         "climbs"))
    group.append(walls)
    collisions = hou.FolderParmTemplate("collisions", "Collisions", folder_type=hou.folderType.Tabs)
    collisions.addParmTemplate(hou.LabelParmTemplate("collisionnote", "Per Object",
                                                     column_labels=("These are every collision object's, unless its geometry carries its own: primitive "
                                                                    "attributes flip2_friction, flip2_hold (0 or 1) and flip2_thickness, one value per object "
                                                                    "(objects are told apart by their name attribute)",)))
    collisions.addParmTemplate(hou.FloatParmTemplate("friction", "Friction", 1, default_value=(0.0,), min=0.0, max=1.0,
                                                     help="0: the fluid slips along collisions freely; 1: the fluid touching them moves with them. An object's "
                                                          "own flip2_friction primitive attribute overrides this"))
    collisions.addParmTemplate(hou.ToggleParmTemplate("collisionhold", "Hold Liquid", default_value=True,
                                                      help="As a wall's Holds Liquid: on, a collision pulls on the liquid as well as pushing it, so liquid "
                                                           "clings to its underside; off, it lets the liquid go wherever air can reach it. An object's own "
                                                           "flip2_hold primitive attribute (0 or 1) overrides this. Does nothing with Liquid and Air simulated"))
    collisions.addParmTemplate(hou.FloatParmTemplate("thickness", "Thickness", 1, default_value=(0.0,), min=0.0,
                                                     help="0 for closed surfaces; for open ones, like a ground plane, the thickness of the shell around them. "
                                                          "An object's own flip2_thickness primitive attribute overrides this"))
    group.append(collisions)
    rbd = hou.FolderParmTemplate("rbdfolder", "RBD", folder_type=hou.folderType.Tabs)
    rbd.addParmTemplate(hou.StringParmTemplate("rbdsop", "RBD Geometry", 1, default_value=("",), string_type=hou.stringParmType.NodeReference,
                                               tags={"opfilter": "!!SOP!!", "oprelative": "."},
                                               help="A SOP of packed pieces, as RBD Bullet Solver's Geometry input takes them (from RBD Material Fracture, "
                                                    "RBD Configure, or Pack with a name per piece): flip2 simulates them with the liquid, each moving the "
                                                    "liquid as the liquid moves it. Their point attributes mean what they mean to RBD Bullet Solver: active "
                                                    "(0: it doesn't move itself, but collides, still or animated as the geometry has it; without the "
                                                    "attribute, simulated), density in kg/m3 (1000 without it) or mass in kg, friction (1), bounce (0.25), "
                                                    "and v and w, its velocity and spin at the start frame, as they are there. The liquid slips along "
                                                    "them and holds to them as the Collisions tab says, and there can be as many as fit in the GPU's "
                                                    "memory. The node's sixth output has the pieces where flip2 moved them, as RBD Bullet Solver's "
                                                    "Geometry output; its seventh the Simulation Points they were moved by"))
    rbd.addParmTemplate(hou.IntParmTemplate("collisionsubsteps", "Collision Substeps", 1, default_value=(5,), min=1, max=100,
                                            help="How many times a frame, at least, the pieces' contacts with each other, the walls and the collisions "
                                                 "are solved, as RBD Bullet Solver's Substeps: 5 at 24 frames a second is 120 a second, enough for a "
                                                 "stack to stand; a tall one needs more. More steps are taken when a piece moves fast, so none moves "
                                                 "more than half a voxel in one"))
    group.append(rbd)
    sources = hou.FolderParmTemplate("sources", "Sources", folder_type=hou.folderType.Tabs)
    sources.addParmTemplate(hou.ToggleParmTemplate("usev", "Use v Attribute", default_value=True,
                                                   help="Each source emits at its points' mean v, if it has a v attribute; otherwise at the emission velocity"))
    sources.addParmTemplate(hou.FloatParmTemplate("emitvel", "Emission Velocity", 3, default_value=(0.0, 0.0, 0.0)))
    group.append(sources)
    forces = hou.FolderParmTemplate("forces", "Forces", folder_type=hou.folderType.Tabs)
    forces.addParmTemplate(_force_fields_interface())
    forces.addParmTemplate(hou.StringParmTemplate("forcesop", "Force Volumes", 1, default_value=("",), string_type=hou.stringParmType.NodeReference,
                                                  tags={"opfilter": "!!SOP!!", "oprelative": "."},
                                                  help="A SOP whose vector VDBs act on the fluid, each a force of its own (up to 16, with the Force "
                                                       "Fields), wherever it has "
                                                       "active voxels and not elsewhere. They're taken as they are at the start frame: flip2 doesn't yet "
                                                       "take volumes that change over time. Anything else the SOP holds is left out"))
    forces.addParmTemplate(hou.MenuParmTemplate("forcemode", "Volumes Are", ("name", "force", "velocity"),
                                                ("Told by Name (v, vel...: Velocities)", "Forces", "Velocities"), default_value=0,
                                                help="What the volumes' vectors are. Forces: accelerations in m/s2, which push the fluid as gravity "
                                                     "does. Velocities: the fluid inside the volume is drawn to them, as wind carries smoke: a current, "
                                                     "a pump, or another simulation's flow to follow. Told by Name: a volume named v, or whose name "
                                                     "starts or ends with vel (vel, velocity, pump_vel), is velocities, and any other is forces"))
    forces.addParmTemplate(hou.FloatParmTemplate("forcestrength", "Strength", 1, default_value=(1.0,), min=0.0, max=10.0,
                                                 help="Multiplies the volumes' vectors"))
    forces.addParmTemplate(hou.FloatParmTemplate("forcedrag", "Drag", 1, default_value=(1.0,), min=0.0, max=100.0, disable_when="{ forcemode == force }",
                                                 help="How fast the fluid takes up a velocity volume's velocities, per second: at 1 it closes about two "
                                                      "thirds of the gap in a second, at 10 in a tenth of one, and from 100 or so it has them almost at "
                                                      "once. However large, it never overshoots them"))
    group.append(forces)
    surface = hou.FolderParmTemplate("surfacefolder", "Surface", folder_type=hou.folderType.Tabs)
    surface.addParmTemplate(hou.ToggleParmTemplate("meshsurface", "Mesh the Surface", default_value=True,
                                                   help="Mesh the liquid's surface on the GPU as the bake commits each frame (flip2 mesh): closed, facing "
                                                        "outwards, with v for motion blur"))
    surface.addParmTemplate(hou.FloatParmTemplate("voxelscale", "Voxel Scale", 1, default_value=(0.5,), min=0.1, max=2.0,
                                                  help="The surface's sample spacing, in particle separations: smaller is finer, slower and larger on disk"))
    surface.addParmTemplate(hou.FloatParmTemplate("influencescale", "Influence Scale", 1, default_value=SURFACE_DEFAULTS["influencescale"], min=0.5, max=4.0,
                                                  help="How far each particle reaches into the surface, in particle separations: larger is smoother, and "
                                                       "fills gaps between particles further apart; at 2 the surface dimples over each particle. At "
                                                       "most 8 times the voxel scale"))
    surface.addParmTemplate(hou.FloatParmTemplate("radiusscale", "Radius Scale", 1, default_value=SURFACE_DEFAULTS["radiusscale"], min=0.1, max=2.0,
                                                  help="Each particle's radius, in particle separations: where the surface sits around them. 0.8 keeps "
                                                       "the liquid's volume at an influence scale of 3 (0.6 at 2). Less than the influence scale"))
    surface.addParmTemplate(hou.IntParmTemplate("smoothing", "Smoothing", 1, default_value=SURFACE_DEFAULTS["smoothing"], min=0, max=10,
                                                help="Passes of a smoothing filter over the surface before it's meshed"))
    surface.addParmTemplate(hou.MenuParmTemplate("surfacesolids", "Water at Collisions", ("overlap", "cut"),
                                                 ("Overlap (Karma Setup Resolves It)", "Cut at Their Surfaces"), default_value=0,
                                                 help="Where the water touches a collision object or an RBD piece. Overlap carries the water's surface a "
                                                      "voxel or so into the object, which hides it; for glass, ice or anything else see-through, flip2 "
                                                      "Karma Setup gives the water a lower dielectric priority than every other object, so Karma "
                                                      "renders the object's own surface against the water. Cut stops the water exactly at the "
                                                      "object's surface: for renderers without dielectric priorities, where the two surfaces in "
                                                      "one place can show specks with see-through objects. Either way the domain's walls cut the "
                                                      "water flat, as a cutaway shows it"))
    surface.addParmTemplate(hou.ToggleParmTemplate("outputfields", "Output Fluid Fields", default_value=False,
                                                   help="Write the liquid's surface level set and velocity field each frame too (surface and vel, as Houdini's FLIP "
                                                        "outputs them): the node's second output, for Whitewater Source"))
    surface.addParmTemplate(hou.FloatParmTemplate("fieldvoxelscale", "Field Voxel Scale", 1, default_value=(2.0,), min=0.5, max=8.0,
                                                  disable_when="{ outputfields == 0 }",
                                                  help="The fields' voxel size, in particle separations: 2 is the simulation's own, as Houdini's FLIP has them"))
    surface.addParmTemplate(hou.ButtonParmTemplate("meshbake", "Mesh Bake", help="Mesh the bake's frames again with these settings, replacing their "
                                                   "surfaces: after a bake made without them, or to try others", **_callback("mesh_bake")))
    group.append(surface)
    whitewater = hou.FolderParmTemplate("whitewaterfolder", "Whitewater", folder_type=hou.folderType.Tabs)
    whitewater.addParmTemplate(hou.ToggleParmTemplate("whitewater", "Whitewater", default_value=False,
                                                      help="Bake spray, foam and bubbles with the liquid: made where its surface breaks up (liquid closing on "
                                                           "liquid or on a wall, the surface folding under or stretching until it tears), each moving as what "
                                                           "it is. On the node's fifth output, and flip2 Karma Setup draws it. It takes nothing from the liquid: "
                                                           "the liquid bakes the same with it or without"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwamount", "Amount", 1, default_value=(1.0,), min=0.0, max=10.0, disable_when="{ whitewater == 0 }",
                                                     help="How much a breaking surface makes, against the default: 2 is twice the whitewater"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwspray", "Spray", 1, default_value=(1.0,), min=0.0, max=10.0, disable_when="{ whitewater == 0 }",
                                                     help="How much of it is spray, against the default: droplets thrown off where liquid collides or its "
                                                          "surface tears. They fly until they land, and are foam where they do"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwbubbles", "Bubbles", 1, default_value=(1.0,), min=0.0, max=10.0, disable_when="{ whitewater == 0 }",
                                                     help="How much of it is bubbles, against the default: air folded in where liquid collides or its surface "
                                                          "is drawn under. They rise, and are foam once they're up"))
    whitewater.addParmTemplate(hou.IntParmTemplate("wwpervoxel", "Particles per Voxel", 1, default_value=(27,), min=1, max=216, disable_when="{ whitewater == 0 }",
                                                   help="How finely it's drawn: the particles a voxel's volume of whitewater becomes. 27 is a third of a voxel "
                                                        "apart, as Houdini's own whitewater is set up; 64 is finer and makes 2.4 times as many. Their pscale "
                                                        "follows, half their spacing"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwfoamlife", "Foam Life", 1, default_value=(0.5,), min=0.05, max=60.0, disable_when="{ whitewater == 0 }",
                                                     help="Seconds a bubble lasts once it has reached the surface, on average: each bursts at the same rate whatever "
                                                          "its age, so a patch thins evenly. Half a second is fresh water's; the sea's foam lasts 3 or 4"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwmaxparticles", "Max Particles (millions)", 1, default_value=(16.0,), min=0.01, max=1000.0,
                                                     disable_when="{ whitewater == 0 }",
                                                     help="The most there can be at once, on each GPU. With less room than the surface would fill, what it "
                                                          "makes is thinned evenly, and the bake's log says so. About 100 MB of GPU memory per million there "
                                                          "are, not per million allowed"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwdropletscale", "Droplet Size", 1, default_value=(1.0,), min=0.01, max=100.0, disable_when="{ whitewater == 0 }",
                                                     help="Droplets' sizes against real ones (a millimetre or so, finer the harder the surface is torn), which "
                                                          "is what sets how they fly: larger keep going, smaller hang as mist. For a miniature that should read "
                                                          "as big water, lower it"))
    whitewater.addParmTemplate(hou.FloatParmTemplate("wwbubblescale", "Bubble Size", 1, default_value=(1.0,), min=0.01, max=100.0, disable_when="{ whitewater == 0 }",
                                                     help="Bubbles' sizes against real ones (most of the air is in bubbles a millimetre or more across), which "
                                                          "sets how fast they rise: larger are up and gone sooner, smaller follow the liquid as a haze"))
    group.append(whitewater)
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
    for template in _gpus_interface(node):
        bake.addParmTemplate(template)
    bake.addParmTemplate(hou.IntParmTemplate("partitions", "Partitions", 1, default_value=(1,), min=1,
                                             help="How many pieces to split the domain into, spread over the GPUs. The results are the same however "
                                                  "many, but a GPU with no piece sits idle, so a bake raises it to at least one per GPU ticked "
                                                  "above; more pieces than GPUs share them"))
    bake.addParmTemplate(hou.IntParmTemplate("checkpoints", "Checkpoint Every", 1, default_value=(10,), min=0,
                                             help="Frames between checkpoints, which a cancelled bake resumes from; 0 for one only on cancel and at the end"))
    bake.addParmTemplate(hou.MenuParmTemplate("compression", "Compression", ("zstd", "lz4", "none"), ("Zstd", "LZ4", "None"), default_value=0,
                                              help="How flip2's own cache of the bake is compressed"))
    bake.addParmTemplate(hou.ToggleParmTemplate("writeid", "Write id", default_value=True, join_with_next=True,
                                                help="Give the particles Houdini's id attribute: a number each keeps from the frame it's emitted to the "
                                                     "frame it's gone, and that no other particle of the bake has, the same however many GPUs bake it and "
                                                     "across a resume. It's what lets Houdini follow a particle from one frame to the next: for trails, "
                                                     "retiming, and anything that blends frames. 8 bytes a particle in flip2's cache before compression"))
    bake.addParmTemplate(hou.ToggleParmTemplate("writeage", "Write age", default_value=True,
                                                help="Give the particles Houdini's age attribute: seconds since each was emitted, or since the start for "
                                                     "the fluid that was there then. 4 bytes a particle in flip2's cache before compression"))
    bake.addParmTemplate(importer.format_parm())
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
    """the node's loaders: the bake's frame showing at this time, its frame 0 on the start frame; and its domain, and its inputs' labels. Returns what
    its first output shows"""
    folder, frame = '`chs("../outputdir")`/bake/export/houdini', 'round(($T - (ch("../startframe") - 1)/$FPS)*$FPS)'
    shown = importer.build_loaders(node, folder, frame)
    importer.build_fields(node, folder, frame)
    importer.build_whitewater(node, folder, frame)
    _split_volumes(node)
    _force_network(node)
    _rbd_network(node)
    moved = importer.build_bodies(node, folder, frame, node.node("rbd_rest"), node.node("RBD_PIECES"))
    box = importer.build_container(node, importer.collisions_with(node, node.node("collision_unpack"), ROLES.index("collision"), moved),
                                   ROLES.index("collision"), 'strlen(chs("../rbdsop")) > 0')
    for axis in "xyz":
        box.parm("size" + axis).setExpression('ch("../domainsize%s")' % axis)
        box.parm("t" + axis).setExpression('ch("../domaincenter%s")' % axis)
    for index, label in enumerate(LABELS):
        if node.parm("label%d" % (index + 1)) is not None:
            node.parm("label%d" % (index + 1)).set(label)
    return shown


def _split_volumes(node):
    """the Collisions input's VDBs apart from the rest of it: the rest goes on to be triangles (COLLISION), and the VDBs stay as they are
    (COLLISION_VOLUMES). Makes only what's missing, so an older node gains it"""
    unpack = node.node("collision_unpack")
    if node.node("collision_solids") is None:
        solids = node.createNode("attribwrangle", "collision_solids")
        solids.parm("class").set(1)
        solids.parm("snippet").set(SOLIDS)
        solids.setInput(0, unpack)
        node.node("collision_polygons").setInput(0, solids)
    if node.node("COLLISION_VOLUMES") is None:
        only = node.createNode("attribwrangle", "collision_volumes")
        only.parm("class").set(1)
        only.parm("snippet").set(VOLUMES)
        only.setInput(0, unpack)
        node.createNode("null", "COLLISION_VOLUMES").setInput(0, only)
        node.layoutChildren()


def _force_network(node):
    """the Force Volumes SOP's geometry brought into this node's space (force_merge), and of it only the vector VDBs (FORCE_VOLUMES). Makes only what's
    missing, so an older node gains it"""
    if node.node("FORCE_VOLUMES") is not None:
        return
    merge = node.createNode("object_merge", "force_merge")
    merge.parm("objpath1").set('`chsop("../forcesop")`')
    merge.parm("xformtype").set("local")    #into this object, as the inputs are
    volumes = node.createNode("attribwrangle", "force_vdbs")
    volumes.parm("class").set(1)
    volumes.parm("snippet").set(VOLUMES)
    volumes.setInput(0, merge)
    vectors = node.createNode("attribwrangle", "force_vectors")
    vectors.parm("class").set(1)
    vectors.parm("snippet").set(VECTORS)
    vectors.setInput(0, volumes)
    node.createNode("null", "FORCE_VOLUMES").setInput(0, vectors)
    node.layoutChildren()


def _rbd_network(node):
    """the RBD tab's pieces brought into this node's space (rbd_merge): as they are now (RBD_PIECES), at the start frame, where the bake starts them
    (rbd_rest), and as triangles by name (RBD), as an input's are. Makes only what's missing, so an older node gains it"""
    if node.node("RBD") is not None:
        return
    merge = node.createNode("object_merge", "rbd_merge")
    merge.parm("objpath1").set('`chsop("../rbdsop")`')
    merge.parm("xformtype").set("local")    #into this object, as the inputs are
    node.createNode("null", "RBD_PIECES").setInput(0, merge)
    rest = node.createNode("timeshift", "rbd_rest")
    rest.setInput(0, merge)
    rest.parm("frame").setExpression('ch("../startframe")')
    tag = node.createNode("attribwrangle", "rbd_tag")     #each piece's geometry named as the piece is: what's inside can be named otherwise
    tag.parm("class").set(2)
    tag.parm("snippet").set(PIECE_TAG)
    tag.setInput(0, merge)
    unpack = node.createNode("unpack", "rbd_unpack")
    unpack.setInput(0, tag)
    unpack.parm("limit_iterations").set(False)
    unpack.parm("transfer_attributes").set("flip2_piece")
    named = node.createNode("attribwrangle", "rbd_named")
    named.parm("class").set(1)
    named.parm("snippet").set('s@name = point(0, "flip2_piece", primpoint(0, @primnum, 0));')
    named.setInput(0, unpack)
    polygons = node.createNode("convert", "rbd_polygons")
    polygons.setInput(0, named)
    triangles = node.createNode("divide", "rbd_triangles")
    triangles.setInput(0, polygons)
    corners = node.createNode("attribwrangle", "rbd_corners")
    corners.parm("class").set(1)
    corners.parm("snippet").set(CORNERS)
    corners.setInput(0, triangles)
    node.createNode("null", "RBD").setInput(0, corners)
    node.layoutChildren()


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


def _settings(geometry, chosen, label):
    """a collision object's own settings, from the SETTINGS primitive attributes its geometry carries, over the primitives chosen: one value each"""
    import numpy
    settings = {}
    for key, name, cast in SETTINGS:
        attrib = geometry.findPrimAttrib(name)
        if attrib is None:
            continue
        if attrib.dataType() == hou.attribData.Float:
            values = numpy.array(geometry.primFloatAttribValues(name))
        elif attrib.dataType() == hou.attribData.Int:
            values = numpy.array(geometry.primIntAttribValues(name))
        else:
            raise hou.NodeError("the primitive attribute %s should be a number, not %s" % (name, attrib.dataType().name().lower()))
        values = values[chosen]
        if len(values) and (values != values[0]).any():
            raise hou.NodeError("%s differs between the primitives of the object %r: flip2 takes one value per object" % (name, label))
        if len(values):
            settings[key] = cast(values[0])
    return settings


def _objects(geometry):
    """an input's triangles at a frame, per object: (name, triangles as point numbers, its own settings), and its points' positions"""
    import numpy
    positions = numpy.frombuffer(geometry.pointFloatAttribValuesAsString("P"), dtype=numpy.float32).reshape(-1, 3)
    if geometry.findPrimAttrib("flip2_a") is None or len(geometry.prims()) == 0:
        return [], positions
    corners = [numpy.frombuffer(geometry.primIntAttribValuesAsString("flip2_" + corner), dtype=numpy.int32) for corner in "abc"]
    triangles = numpy.stack(corners, axis=1)
    keep = triangles[:, 0] >= 0
    names = geometry.findPrimAttrib("name")
    if names is None or names.dataType() != hou.attribData.String:
        return [("", triangles[keep], _settings(geometry, keep, ""))], positions
    labels = numpy.array(geometry.primStringAttribValues("name"), dtype=object)
    objects = []
    for label in sorted(set(labels[keep])):
        chosen = keep & (labels == label)
        objects.append((label, triangles[chosen], _settings(geometry, chosen, label)))
    return objects, positions


def _label(role):
    return LABELS[ROLES.index(role)] if role in ROLES else "RBD Geometry"


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
    for name, triangles, settings in first:
        used, local = numpy.unique(triangles, return_inverse=True)
        samples.append({"name": name, "triangles": local.reshape(-1, 3).astype(numpy.int32), "points": used, "global": triangles,
                        "frames": [positions[used].copy()], "settings": settings})
    for frame in frames[1:]:
        objects, positions = _objects(source.geometryAtFrame(frame))
        same = len(objects) == len(samples) and all(name == sample["name"] and numpy.array_equal(triangles, sample["global"])
                                                    for sample, (name, triangles, _) in zip(samples, objects))
        if not same:
            raise hou.NodeError("%s: its triangles change at frame %d; flip2 can only move or deform a mesh whose points and triangles stay the same"
                                % (_label(role), frame))
        for sample in samples:
            sample["frames"].append(positions[sample["points"]].copy())
        if progress is not None:
            progress()
    return samples


def _rbd_pieces(node, frames, progress=None):
    """the RBD tab's pieces over frames, as _sample has an input's objects, each with its settings under "rbd": whether it's simulated (active, or
    without active), its density or mass, friction, bounce, and v and w, from the point attributes RBD Bullet Solver reads on each packed piece (or
    primitive ones), as they are at the start frame; RBD Bullet Solver's own defaults for those it hasn't. [] with no SOP named"""
    path = node.evalParm("rbdsop").strip() if node.parm("rbdsop") is not None else ""
    if not path:
        return []
    if node.parm("rbdsop").evalAsNode() is None:
        raise hou.NodeError("RBD: RBD Geometry names %r, and there's no such node" % path)
    geometry = node.node("RBD_PIECES").geometryAtFrame(frames[0])
    settings = {}
    for prim in geometry.prims():
        if not isinstance(prim, hou.PackedPrim):
            raise hou.NodeError("RBD Geometry: flip2 takes packed pieces, as RBD Bullet Solver does: pack them first (RBD Material Fracture and RBD "
                                "Configure give packed pieces, or Pack with Name Attribute)")
        point = prim.points()[0]

        def value(name, default):
            attrib = geometry.findPointAttrib(name)
            if attrib is not None:
                return point.attribValue(attrib)
            attrib = geometry.findPrimAttrib(name)
            return prim.attribValue(attrib) if attrib is not None else default
        name = value("name", "")
        if not name:
            raise hou.NodeError("RBD Geometry: its pieces need a name point attribute, as RBD Bullet Solver's do: RBD Material Fracture and RBD Configure "
                                "give them one, and Pack does with Transfer Attributes set to name")
        if name in settings:
            raise hou.NodeError("RBD Geometry: two pieces are both named %r; flip2 tells them apart by name, as RBD Bullet Solver does" % name)
        piece = {"active": int(value("active", 1)) != 0, "density": float(value("density", RBD_DEFAULTS["density"])), "mass": float(value("mass", 0.0)),
                 "friction": float(value("friction", RBD_DEFAULTS["friction"])), "bounce": float(value("bounce", RBD_DEFAULTS["bounce"])),
                 "v": [float(c) for c in value("v", (0.0, 0.0, 0.0))], "w": [float(c) for c in value("w", (0.0, 0.0, 0.0))]}
        if piece["active"] and (piece["density"] <= 0.0 or piece["mass"] < 0.0 or not 0.0 <= piece["bounce"] <= 1.0 or piece["friction"] < 0.0):
            raise hou.NodeError("RBD Geometry: the piece %r has density %g, mass %g, bounce %g and friction %g; flip2 takes a density over 0, a mass of 0 "
                                "or more (0: from the density), a bounce from 0 to 1, and a friction of 0 or more"
                                % (name, piece["density"], piece["mass"], piece["bounce"], piece["friction"]))
        settings[name] = piece
    pieces = _sample(node, "rbd", frames, progress)
    for samples in pieces:
        if samples["name"] not in settings:
            raise hou.NodeError("RBD Geometry: geometry named %r that isn't in any of its packed pieces" % samples["name"])
        samples["rbd"] = settings[samples["name"]]
    return pieces


def _rigid(first, later, tolerance):
    """the turn and shift that take first's points to later's, as flip2 has a transform (16 numbers: a 4x4 matrix by rows, acting on column vectors), or
    None if none does to within tolerance. The best fit there is: Kabsch's"""
    import numpy
    centre, moved = first.mean(axis=0), later.mean(axis=0)
    u, _, vt = numpy.linalg.svd((first - centre).T @ (later - moved))
    turn = vt.T @ numpy.diag([1.0, 1.0, numpy.sign(numpy.linalg.det(vt.T @ u.T))]) @ u.T
    shift = moved - turn @ centre
    if numpy.abs(first @ turn.T + shift - later).max() > tolerance:
        return None
    return [float(value) for row in range(3) for value in (*turn[row], shift[row])] + [0.0, 0.0, 0.0, 1.0]


def _motion(samples, tolerance):
    """how a sampled object moves: still; rigid, with a transform per frame from where it is at the first; or deforming"""
    import numpy
    first = samples["frames"][0]
    if all(numpy.array_equal(positions, first) for positions in samples["frames"][1:]):
        return "still", None
    transforms = []
    for positions in samples["frames"]:
        transform = _rigid(first.astype(numpy.float64), positions.astype(numpy.float64), tolerance)
        if transform is None:
            return "deforming", None
        transforms.append(transform)
    return "rigid", transforms


class _Packer:
    """The scene's meshes, packed: every object's vertices where it starts in one file and its triangles in another, and every deforming object's
    vertices at a frame in a file a frame, which the scene names rows of ({"file", "first", "count"}), rather than files an object: a scene of a
    thousand pieces is a handful of files to write, copy to a farm and read back"""

    def __init__(self):
        self.arrays = {}    #per file, its arrays in the order they go in, and its rows so far
        self.rows = {}

    def add(self, name, values):
        """values (n x 3), at the end of the file name, and the rows they take there"""
        first = self.rows.get(name, 0)
        self.arrays.setdefault(name, []).append(values)
        self.rows[name] = first + len(values)
        return {"file": "geo/" + name, "first": first, "count": len(values)}

    def write(self, geo_dir):
        import numpy
        for name, arrays in self.arrays.items():
            kind = numpy.int32 if name == MESH_TRIANGLES else numpy.float32
            numpy.save(os.path.join(geo_dir, name), numpy.concatenate([numpy.asarray(values, dtype=kind).reshape(-1, 3) for values in arrays]))


def _write_object(samples, motion, packer, frames, fps):
    """an object's scene entry, its geometry packed: still, moving rigidly with a transform every frame, or deforming with its vertices every frame"""
    kind, transforms = motion
    triangles = packer.add(MESH_TRIANGLES, samples["triangles"])
    if kind != "deforming":
        entry = {"vertices": packer.add(MESH_VERTICES, samples["frames"][0]), "triangles": triangles}
        if kind == "rigid":
            entry["keyframes"] = [{"time": (frame - frames[0]) / fps, "transform": transform} for frame, transform in zip(frames, transforms)]
        return entry
    deforming = [{"time": (frame - frames[0]) / fps, "vertices": packer.add(DEFORMING_FILE % frame, positions)} for frame, positions in zip(frames, samples["frames"])]
    return {"vertices": deforming[0]["vertices"], "triangles": triangles, "deforming": deforming}


def _volume_state(volume):
    """a VDB's voxels as far as it's cheap to tell them from another frame's (how many, over what extent, their least, greatest and mean values); where
    it is: its transform from voxels to the world, as Houdini has one (4x4, on row vectors); and its middle"""
    import numpy
    voxels = (volume.intrinsicValue("activevoxelcount"), tuple(volume.intrinsicValue("activevoxeldimensions")), volume.intrinsicValue("vdb_value_type"))
    if "volumeminvalue" in volume.intrinsicNames() and voxels[2] in ("float", "double"):
        voxels += tuple(round(volume.intrinsicValue(name), 6) for name in ("volumeminvalue", "volumemaxvalue", "volumeavgvalue"))
    return voxels, numpy.array(volume.intrinsicValue("transform"), dtype=numpy.float64).reshape(4, 4), numpy.array(volume.boundingBox().center())


def _sample_volumes(node, frames, progress=None):
    """the Collisions input's VDB level sets over frames: per level set its name, its velocity VDB's name (or None), and a transform per frame from where
    it is at the first if it moves (rigidly), or None if it keeps still. One that changes shape is an error: flip2 hasn't those yet. Houdini's own
    transform of a VDB it has turned a few degrees is a little off a turn (its scale along the axis is the angle's cosine), so the nearest turn is taken,
    about the volume's middle, when the scales are within a hundredth of 1"""
    import numpy
    volumes, first = [], {}
    for frame in frames:
        geometry = node.node("COLLISION_VOLUMES").geometryAtFrame(frame)
        states, settings = {}, {}
        for volume in geometry.prims():
            name = volume.attribValue("name") if geometry.findPrimAttrib("name") is not None else ""
            if name in states:
                raise hou.NodeError("Collisions: two volumes are both named %r; flip2 tells them apart by name, so give each its own" % name)
            states[name] = _volume_state(volume)
            settings[name] = _settings(geometry, numpy.array([prim.number() == volume.number() for prim in geometry.prims()]), name)
        if frame == frames[0]:
            first = states
            vectors = [name for name, state in states.items() if state[0][2].startswith("vec3")]
            for name, state in states.items():
                if state[0][2] in ("float", "double"):
                    velocity = next((candidate for candidate in (name + "vel", "vel", "v") if candidate in vectors), None)
                    volumes.append({"name": name, "velocity": velocity, "transforms": [], "still": True, "settings": settings[name]})
        elif set(states) != set(first):
            raise hou.NodeError("Collisions: its volumes change at frame %d; flip2 takes volumes that are there throughout" % frame)
        for volume in volumes:
            for name in [volume["name"]] + ([volume["velocity"]] if volume["velocity"] else []):
                if states[name][0] != first[name][0]:
                    raise hou.NodeError("Collisions: the volume %r changes shape at frame %d. flip2 takes a level set that keeps still or moves rigidly, but "
                                        "not yet one that changes over time: give it the geometry the volume is made from instead" % (name, frame))
            moved = numpy.linalg.inv(first[volume["name"]][1]) @ states[volume["name"]][1]      #from where it was at the first frame, on row vectors
            u, scales, vt = numpy.linalg.svd(moved[:3, :3].T)       #on column vectors: the nearest turn is u vt
            if numpy.abs(scales - 1.0).max() > 0.01:
                raise hou.NodeError("Collisions: the volume %r is scaled or sheared at frame %d, which makes its distances wrong; flip2 takes "
                                    "one that keeps still or moves rigidly" % (volume["name"], frame))
            turn = u @ numpy.diag([1.0, 1.0, numpy.sign(numpy.linalg.det(u @ vt))]) @ vt
            middle = first[volume["name"]][2]
            shift = moved[:3, :3].T @ middle + moved[3, :3] - turn @ middle     #so its middle goes where Houdini puts it
            volume["still"] = volume["still"] and numpy.allclose(moved, numpy.eye(4), atol=1e-6)
            volume["transforms"].append([float(value) for row in range(3) for value in (*turn[row], shift[row])] + [0.0, 0.0, 0.0, 1.0])
            if volume["velocity"] and not numpy.allclose(states[volume["velocity"]][1], first[volume["velocity"]][1], atol=1e-6):
                volume["still"] = False
        if progress is not None:
            progress()
    for volume in volumes:
        if volume["still"]:
            volume["transforms"] = None
        elif volume["velocity"]:    #its velocities are the world's, so they can't turn with it
            volume["velocity"] = None
    return volumes


def _velocities(name, mode):
    """whether a force volume's vectors are velocities the fluid is drawn to, or accelerations that push it: as the Forces tab says, or by its name"""
    if mode != "name":
        return mode == "velocity"
    name = name.lower()
    return name == "v" or name.startswith("vel") or name.endswith("vel")


def _force_fields(node, frame):
    """the Forces tab's force fields as they are at frame: the scene's forces for them, and what to say of them (None with none on)"""
    if node.parm("forcefields") is None:    #a node from before Force Fields has none
        return [], None
    forces, animated = [], False
    for instance in range(1, node.evalParm("forcefields") + 1):
        if not node.parm("fieldenable%d" % instance).evalAtFrame(frame):
            continue
        kind = FIELD_TYPES[node.parm("fieldtype%d" % instance).evalAtFrame(frame)]
        force = {"type": kind}
        for key, name in FIELD_SETTINGS[kind]:
            parms = node.parmTuple("%s%d" % (name, instance))
            values = parms.evalAtFrame(frame)
            force[key] = list(values) if len(values) == 3 else values[0]
            animated = animated or any(parm.isTimeDependent() for parm in parms)
        if kind == "vortex" and not any(force["axis"]):
            raise hou.NodeError("Forces: force field %d is a vortex with an Axis of 0, which has no direction to swirl around" % instance)
        if kind == "wind":  #the nearest whole number of voxels to its depth in metres
            force["depth"] = int(force["depth"] / (2*node.evalParm("particlesep")) + 0.5)
        forces.append(force)
    if not forces:
        return [], None
    kinds = [force["type"] for force in forces]
    said = "force fields: " + ", ".join("%d %s" % (kinds.count(kind), kind) for kind in FIELD_TYPES if kind in kinds)
    depths = sorted(set(force["depth"] for force in forces if force["type"] == "wind"))
    if depths:
        said += " (wind reaching %s voxel%s deep)" % ("/".join(str(depth) for depth in depths), "" if depths == [1] else "s")
    if animated:
        said += " (animated: taken as they are at frame %d)" % frame
    return forces, said


def _force_volumes(node, frame, geo_dir):
    """the Forces tab's volumes as they are at frame: the scene's forces for them, their VDBs written into geo_dir in one file, each named, and what to
    say of them (None with no Force Volumes)"""
    import numpy
    if node.parm("forcesop") is None or not node.evalParm("forcesop").strip():      #a node from before the Forces tab has none
        return [], None
    if node.parm("forcesop").evalAsNode() is None:
        raise hou.NodeError("Forces: Force Volumes names %r, and there's no such node" % node.evalParm("forcesop"))
    source = node.node("FORCE_VOLUMES")
    geometry = source.geometryAtFrame(frame)
    held = len(node.node("force_merge").geometryAtFrame(frame).prims())
    named = geometry.findPrimAttrib("name") is not None
    mode = ("name", "force", "velocity")[node.evalParm("forcemode")]
    forces, names = [], set()
    for volume in geometry.prims():
        name = volume.attribValue("name") if named else ""
        if name in names:
            raise hou.NodeError("Forces: two volumes are both named %r; flip2 tells them apart by name, so give each its own" % name)
        names.add(name)
        kind = volume.intrinsicValue("vdb_value_type")
        if kind != "vec3s":
            raise hou.NodeError("Forces: the volume %r holds %s values, and flip2 takes 32-bit float vectors: convert it (Convert VDB, with VDB "
                                "Precision at 32-bit)" % (name, kind))
        scales = numpy.linalg.svd(numpy.array(volume.intrinsicValue("transform"), dtype=numpy.float64).reshape(4, 4)[:3, :3], compute_uv=False)
        if scales.max() - scales.min() > 1e-4*scales.max():
            raise hou.NodeError("Forces: the volume %r is scaled more along one axis than another, so its voxels aren't cubes; flip2 takes ones that "
                                "are: resample it (VDB Resample) after scaling it" % name)
        forces.append({"type": "volume", "vdb": "geo/" + FORCE_FILE, "grid": name, "mode": "velocity" if _velocities(name, mode) else "force",
                       "strength": node.evalParm("forcestrength"), "drag": node.evalParm("forcedrag")})
    if len(forces) > 1 and "" in names:
        raise hou.NodeError("Forces: with more than one volume, each needs a name for flip2 to tell them apart; one here has none")
    if len(forces) > FORCE_LIMIT:
        raise hou.NodeError("flip2 takes up to %d forces; Force Volumes has %d vector VDBs" % (FORCE_LIMIT, len(forces)))
    if forces:
        geometry.saveToFile(os.path.join(geo_dir, FORCE_FILE))
    kinds = [force["mode"] for force in forces]
    said = "forces: " + (", ".join("%d of %s" % (kinds.count(kind), label) for kind, label in (("force", "forces"), ("velocity", "velocities")) if kind in kinds)
                         or "no vector VDBs")
    if forces and source.isTimeDependent():
        said += " (they change over time: taken as they are at frame %d)" % frame
    if held > len(forces):
        said += ", %d other primitive%s left out" % (held - len(forces), "" if held - len(forces) == 1 else "s")
    return forces, said


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
    _, ticked = _chosen_gpus(node)
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
                   "transfer": ("flip", "apic")[node.evalParm("transfer")],
                   "viscousCfl": node.evalParm("viscouscfl") if node.parm("viscouscfl") is not None else 6.0,
                   "freeSurface": ("voxel", "levelSet")[node.evalParm("freesurface")] if node.parm("freesurface") is not None else "levelSet"},
        "gravity": list(node.evalParmTuple("gravity")),
        "liquid": {key: node.evalParm(name) if node.parm(name) is not None else default       #a node from before the Liquid tab has none of it
                   for key, name, default in (("density", "density", 1000.0), ("viscosity", "viscosity", 0.0), ("surfaceTension", "surfacetension", 0.0),
                                              ("contactAngle", "contactangle", 60.0))},
        "fluids": [], "emitters": [], "sinks": [], "obstacles": [],
        "partitions": node.evalParm("partitions"), "devices": node.evalParm("gpus") if node.parm("gpus") is not None else 0,    #the ticked GPUs are all it sees
        "output": {"dir": "bake", "compression": ("zstd", "lz4", "none")[node.evalParm("compression")],
                   "attributes": [name for name in ("id", "age") if node.parm("write" + name) is None or node.evalParm("write" + name)],
                   "checkpoints": {"every": node.evalParm("checkpoints"), "keep": 2}},
    }
    walls = {}      #what each wall does to the liquid on it, where that isn't the engine's default (a node from before the Walls tab says nothing)
    for index, face in enumerate(FACES):
        if node.parm("wallhold%d" % index) is None:
            continue
        wall = {}
        if not node.evalParm("wallhold%d" % index):
            wall["hold"] = False
        if node.evalParm("wallfriction%d" % index) != 0.0:
            wall["friction"] = node.evalParm("wallfriction%d" % index)
        if node.evalParm("wallangle%d" % index) != scene["liquid"]["contactAngle"]:
            wall["contactAngle"] = node.evalParm("wallangle%d" % index)
        if wall:
            walls[face] = wall
    if walls:
        scene["domain"]["walls"] = walls
    holds = node.parm("collisionhold") is None or bool(node.evalParm("collisionhold"))     #whether collisions hold the liquid on them, as walls do
    if node.parm("whitewater") is not None and node.evalParm("whitewater"):    #a node from before the Whitewater tab bakes none
        scene["whitewater"] = {"amount": node.evalParm("wwamount"), "spray": node.evalParm("wwspray"), "bubbles": node.evalParm("wwbubbles"),
                               "perVoxel": node.evalParm("wwpervoxel"), "foamLife": node.evalParm("wwfoamlife"),
                               "maxParticles": round(node.evalParm("wwmaxparticles")*1.0e6), "dropletScale": node.evalParm("wwdropletscale"),
                               "bubbleScale": node.evalParm("wwbubblescale")}
    roles = [role for role in ROLES if _connected(node, ROLES.index(role))]
    rbd_named = node.parm("rbdsop") is not None and bool(node.evalParm("rbdsop").strip())
    total = sum(1 if role == "fluid" else len(frames) for role in roles) + (len(frames) if "collision" in roles else 0) + (len(frames) if rbd_named else 0)
    sampled = [0]
    with hou.InterruptableOperation("flip2: sampling the inputs", long_operation_name="Writing the flip2 scene", open_interrupt_dialog=True) as operation:
        def progress():     #a progress bar, which Esc interrupts
            sampled[0] += 1
            operation.updateLongProgress(sampled[0] / max(total, 1), "sampled %d of %d frames of the inputs" % (sampled[0], total))
        sampled_roles = {role: _sample(node, role, frames[:1] if role == "fluid" else frames, progress) for role in roles}
        volumes = _sample_volumes(node, frames, progress) if "collision" in roles else []
        rbd = _rbd_pieces(node, frames, progress)
    simulated = [samples for samples in rbd if samples["rbd"]["active"]]    #dynamic bodies; the others collide as the Collisions input's objects do
    if len(simulated) < len(rbd):
        sampled_roles.setdefault("collision", []).extend(samples for samples in rbd if not samples["rbd"]["active"])
        roles = [role for role in ROLES if role in sampled_roles]
    if simulated and scene["solver"]["freeSurface"] == "voxel" and not (node.parm("phases") is not None and node.evalParm("phases") == 1):
        raise hou.NodeError("RBD: a simulated piece floats at the right height only with the Level Set free surface, or with Liquid and Air: the Voxel "
                            "surface is a voxel or two over the liquid, and pieces float that much high. Set Free Surface on the Simulation tab")
    tolerance = 0.01*2*separation      #a hundredth of a voxel: less than the fluid can feel
    made = []
    packer = _Packer()
    for role in roles:
        objects = [(samples, _motion(samples, tolerance)) for samples in sampled_roles[role]]   #each object on its own, still, rigid or deforming
        kinds = [kind for _, (kind, _) in objects] + (["volume"]*len(volumes) if role == "collision" else [])
        made.append("%s: %s" % (LABELS[ROLES.index(role)].lower(), ", ".join("%d %s" % (kinds.count(kind), kind) for kind in ("still", "rigid", "deforming", "volume")
                                                                               if kind in kinds) or "nothing"))
        for index, (samples, motion) in enumerate(objects):
            entry = _write_object(samples, motion, packer, frames[:len(samples["frames"])], fps)
            if role == "fluid":
                entry = {key: entry[key] for key in ("vertices", "triangles")}
                scene["fluids"].append(entry)
            elif role == "collision":     #the node's settings, unless the object's geometry carries its own
                entry["friction"] = samples["settings"].get("friction", node.evalParm("friction"))
                entry["thickness"] = samples["settings"].get("thickness", node.evalParm("thickness"))
                if not samples["settings"].get("hold", holds):
                    entry["hold"] = False
                scene["obstacles"].append(entry)
            elif role == "source":
                entry["velocity"] = _emission_velocity(node, samples)
                scene["emitters"].append(entry)
            else:
                scene["sinks"].append(entry)
    if volumes:     #the level sets as they are at the start, in one file, each named
        node.node("COLLISION_VOLUMES").geometryAtFrame(frames[0]).saveToFile(os.path.join(geo_dir, VOLUME_FILE))
    for volume in volumes:
        entry = {"vdb": "geo/" + VOLUME_FILE, "grid": volume["name"], "friction": volume["settings"].get("friction", node.evalParm("friction"))}
        if not volume["settings"].get("hold", holds):
            entry["hold"] = False
        if volume["velocity"]:
            entry["velocityGrid"] = volume["velocity"]
        if volume["transforms"]:
            entry["keyframes"] = [{"time": (frame - frames[0]) / fps, "transform": transform} for frame, transform in zip(frames, volume["transforms"])]
        scene["obstacles"].append(entry)
    for index, samples in enumerate(simulated):     #the RBD tab's simulated pieces, each a dynamic body starting where the start frame has it
        settings = samples["rbd"]
        entry = _write_object(samples, ("still", None), packer, frames[:1], fps)
        dynamic = {"velocity": settings["v"], "angularVelocity": settings["w"], "bounce": settings["bounce"], "contactFriction": settings["friction"]}
        dynamic.update({"mass": settings["mass"]} if settings["mass"] > 0.0 else {"density": settings["density"]})
        entry.update({"name": samples["name"], "dynamic": dynamic, "friction": samples["settings"].get("friction", node.evalParm("friction"))})
        if not samples["settings"].get("hold", holds):
            entry["hold"] = False
        scene["obstacles"].append(entry)
    if rbd:
        made.append("rbd: %d simulated, %d colliding" % (len(simulated), len(rbd) - len(simulated)))
    packer.write(geo_dir)
    if simulated and node.parm("collisionsubsteps") is not None:
        scene["solver"]["collisionSubsteps"] = node.evalParm("collisionsubsteps")
    forces, said = _force_volumes(node, frames[0], geo_dir)
    fields, fields_said = _force_fields(node, frames[0])
    if len(forces) + len(fields) > FORCE_LIMIT:
        raise hou.NodeError("flip2 takes up to %d forces; the Forces tab has %d force fields on and %d force volumes" % (FORCE_LIMIT, len(fields), len(forces)))
    if forces or fields:
        scene["forces"] = forces + fields
    for told in (fields_said, said):
        if told:
            made.append(told)
    if ticked is not None and scene["partitions"] < ticked:    #a GPU with no piece of the domain would sit idle
        scene["partitions"] = ticked
        made.append("partitions: %d, one per GPU ticked" % ticked)
    if node.parm("phases") is not None and node.evalParm("phases") == 1:   #the air, with the Air tab's settings where they aren't the engine's defaults
        air = {"density": node.evalParm("airdensity"), "band": node.evalParm("airband")}
        if not node.evalParm("airescaped"):
            air["escaped"] = False
        for key, name, default in (("dropletRadius", "airdropletradius", 0.0), ("bubbleRadius", "airbubbleradius", 0.0), ("viscosity", "airviscosity", 1.5e-5)):
            if node.evalParm(name) != default:
                air[key] = node.evalParm(name)
        scene["air"] = air
        scene["solver"]["freeSurface"] = "voxel"        #the level set surface doesn't go with air yet: the tab says so
        made.append("air: a band of %d voxels" % air["band"] if air["band"] else "air: everywhere")
    path = os.path.join(directory, "scene.json")
    with open(path, "w") as out:
        json.dump(scene, out, indent=1)
    node.parm("status").set("wrote %s (%s)" % (path, "; ".join(made)))
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
    for source in [node.node(role.upper()) for role in ROLES] + [node.node("COLLISION_VOLUMES")] + ([node.node("RBD")] if node.node("RBD") is not None else []):
        role = source.name().lower()
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


def _where(node):
    """the machine that bakes, as the GPUs Find GPUs found there are filed under: this one, or the remote host"""
    return "remote:" + node.evalParm("remotehost").strip() if node.evalParm("bakeon") == 1 else "local"


def _gpu_cache(node):
    """what Find GPUs found last, {"where": the machine it asked (_where), "gpus": [flip2 info's, for each GPU]}, or None if it hasn't been pressed"""
    try:
        found = json.loads(node.userData("flip2_gpus") or "null")
    except ValueError:
        return None
    return found if isinstance(found, dict) and isinstance(found.get("gpus"), list) else None


def _machine(where):
    return "this machine" if where == "local" else where[len("remote:"):] or "the remote host"


def _gpus_interface(node):
    """the Bake tab's GPUs: Find GPUs, a line on what it found where, and a box to tick for each GPU there that flip2 runs on, labelled with its number,
    name and memory (and its PCI slot, when two have the same name)"""
    found = _gpu_cache(node)
    gpus = [gpu for gpu in found["gpus"] if gpu.get("runs", True)] if found else []
    if found is None:
        note = "Find GPUs to choose which of the baking machine's GPUs flip2 uses; until then it uses all of them"
    elif not gpus:
        note = "Found no GPU that flip2 runs on, on %s" % _machine(found["where"])
    else:
        note = "On %s: flip2 bakes on the GPUs ticked here" % _machine(found["where"])
    templates = [hou.ButtonParmTemplate("findgpus", "Find GPUs", join_with_next=True, **_callback("find_gpus"),
                                        help="Ask the machine that bakes (this one, or the Remote Host) which GPUs it has, and list them here to tick"),
                 hou.LabelParmTemplate("gpusfound", "GPUs", column_labels=(note,), is_label_hidden=True)]
    names = [gpu["name"] for gpu in gpus]
    for gpu in gpus:
        label = "GPU %d: %s, %d GB" % (gpu["index"], gpu["name"], round(gpu["memoryMB"] / 1024))
        if names.count(gpu["name"]) > 1 and gpu.get("pci"):
            label += " (PCI %s)" % gpu["pci"]
        templates.append(hou.ToggleParmTemplate("gpu%d" % gpu["index"], label, default_value=True,
                                                help="Whether flip2 bakes on this GPU. Ticking more than one splits the domain between them (Partitions)"))
    return templates


def find_gpus(node):
    """asks the machine that bakes which GPUs it has (flip2 info), keeps the answer on the node, and lists them on the Bake tab to tick: as they were
    ticked if it asked the same machine before, otherwise all ticked. Returns them"""
    if node.evalParm("bakeon") == 1:
        host, _, program = _remote(node)
        command = ["ssh"] + SSH_OPTIONS + [host, 'program=%s; "${program/#\\~/$HOME}" info' % shlex.quote(program)]
        machine = host
    else:
        command = [_program(node), "info"]
        machine = "this machine"
    try:    #flip2 info ends in failure when no GPU runs its kernels, and still says what it found
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise hou.NodeError("Find GPUs: couldn't run flip2 on %s: %s" % (machine, error))
    info = next((json.loads(line) for line in result.stdout.splitlines() if line.startswith("{") and '"event":"info"' in line), None)
    if info is None:
        raise hou.NodeError("Find GPUs: flip2 info on %s said nothing about GPUs: %s" % (machine, (result.stderr or result.stdout).strip() or
                                                                                         "exit status %d" % result.returncode))
    gpus = [{key: gpu[key] for key in ("index", "name", "uuid", "pci", "memoryMB", "runs") if key in gpu} for gpu in info.get("gpus", [])]
    earlier = _gpu_cache(node)
    where = _where(node)
    node.setUserData("flip2_gpus", json.dumps({"where": where, "gpus": gpus}))

    def relist():   #the boxes for what was found; another machine's GPUs all start ticked, rather than as the boxes of the same numbers were
        _interface(node)
        if earlier is None or earlier["where"] != where:
            for gpu in gpus:
                box = node.parm("gpu%d" % gpu["index"])
                if box is not None:
                    box.set(True)
    if hou.isUIAvailable():     #after this button's callback returns, as relisting replaces the parameters, the button among them
        import hdefereval
        hdefereval.executeDeferred(relist)
        if not any(gpu.get("runs", True) for gpu in gpus):
            hou.ui.displayMessage("%s has no GPU this flip2 runs on%s" % (machine, ":\n" + "\n".join("%d: %s" % (gpu["index"], gpu["name"]) for gpu in gpus)
                                                                          if gpus else ""), title="flip2: Find GPUs")
    else:
        relist()
    return gpus


def _chosen_gpus(node):
    """the GPUs ticked on the Bake tab: their UUIDs, as CUDA_VISIBLE_DEVICES takes them (their numbers if Find GPUs asked an older flip2), or None
    for every GPU there is, and how many GPUs the bake will have, or None if that isn't known. With every box ticked, or before Find GPUs, that's all
    of them"""
    found = _gpu_cache(node)
    boxes = [(gpu, node.parm("gpu%d" % gpu["index"])) for gpu in (found["gpus"] if found else ()) if gpu.get("runs", True)]
    boxes = [(gpu, box) for gpu, box in boxes if box is not None]
    if not boxes:
        return None, None
    ticked = [gpu for gpu, box in boxes if box.eval()]
    here = found["where"] == _where(node)
    if len(ticked) == len(boxes):   #no need to name them, and one added to the machine since is used as well
        return None, len(boxes) if here else None
    if not here:
        raise hou.NodeError("GPUs: the boxes ticked are %s's GPUs, and this bakes on %s: press Find GPUs to list its own" % (
            _machine(found["where"]), _machine(_where(node))))
    if not ticked:
        raise hou.NodeError("GPUs: no GPU is ticked; tick at least one")
    return [gpu.get("uuid") or str(gpu["index"]) for gpu in ticked], len(ticked)


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


def _launch_remote(job, host, directory, program, arguments, upload, local, logs, visible=None):
    """on a thread: checks there's a flip2 on host, sends the scene if upload, starts the bake there detached (remote/job.sh), seeing only the GPUs
    visible names (CUDA_VISIBLE_DEVICES; None for all), and the mirror bringing it back here (remote/mirror.sh). Says how far it's got in job["state"],
    and why it stopped in job["failure"]"""
    scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote")
    try:
        job["state"] = "connecting to %s" % host
        _ssh(host, 'program=%s; program="${program/#\\~/$HOME}"; '     #a path names an executable file; a bare name is looked up on the PATH
                   'case "$program" in */*) [ -f "$program" ] && [ -x "$program" ];; *) command -v "$program" > /dev/null;; esac || '
                   '{ echo "no flip2 program at $program on this machine: set flip2 Program" >&2; exit 1; }' % shlex.quote(program))
        #a job still running there (one this node lost touch with, say) would have its cache cleared from under it by this one, and both would write it
        running = _ssh(host, 'cd %s 2> /dev/null || exit 0; for pid in logs/job.pid logs/bake.pid; do [ -f "$pid" ] && [ ! -f logs/finished ] && '
                             'kill -0 "$(cat "$pid")" 2> /dev/null && { echo running; break; }; done; exit 0' % directory)
        if "running" in running:
            raise RuntimeError("a bake is still running on %s in %s, started by an earlier Bake: wait for it to finish, or stop it there with "
                               "  kill $(cat %s/logs/job.pid)" % (host, directory, directory))
        _ssh(host, "mkdir -p %s/geo %s/logs && rm -f %s/logs/finished" % (directory, directory, directory))
        if upload:
            job["state"] = "sending the scene to %s" % host
            _rsync(["--delete", os.path.join(local, "geo") + "/", "%s:%s/geo/" % (host, directory)])
            _rsync([os.path.join(local, "scene.json"), "%s:%s/" % (host, directory)])
        _rsync([os.path.join(scripts, "job.sh"), "%s:%s/" % (host, directory)])
        job["state"] = "starting on %s" % host
        #in a subshell: a command line ending in a bare & keeps the ssh session open until the job ends (seen with OpenSSH 10.3 on the Mac), and the node
        #would wait on it until _ssh's timeout and call the bake failed while it ran on; a subshell's background job lets the session close at once
        _ssh(host, "cd %s && (%snohup bash job.sh %s > /dev/null 2>&1 < /dev/null &)" % (directory, "CUDA_VISIBLE_DEVICES=%s " % shlex.quote(visible) if visible else "",
                                                                                        " ".join(shlex.quote(word) for word in [program] + arguments)))
        environment = dict(os.environ, FLIP2_RSH=" ".join(["ssh"] + SSH_OPTIONS))
        #a bake from the start replaces the frames here; one resumed or meshed again only adds to them, as its remote copy may have been deleted since
        mirror = ["bash", os.path.join(scripts, "mirror.sh"), host, directory, local] + (["replace"] if upload else [])
        job["process"] = subprocess.Popen(mirror, env=environment, stdout=subprocess.DEVNULL, stderr=open(os.path.join(logs, "mirror.log"), "a"))
        job["state"] = "baking on %s" % host
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        job["failure"] = str(error)


def _mesh_options(node):
    """flip2 mesh's options, from the Surface tab, checked as flip2 mesh would check them"""
    fields = node.evalParm("outputfields")
    voxel, influence, radius = node.evalParm("voxelscale"), node.evalParm("influencescale"), node.evalParm("radiusscale")
    smoothing = node.evalParm("smoothing")
    if min(voxel, influence, radius) <= 0 or smoothing < 0:
        raise hou.NodeError("the surface's scales have to be positive, and its smoothing 0 or more")
    if radius >= influence:
        raise hou.NodeError("the surface's radius scale has to be less than its influence scale")
    if influence > 8*voxel:
        raise hou.NodeError("the surface's influence scale can be at most 8 times its voxel scale")
    options = ["--format", node.parm("exportformat").evalAsString(), "--voxel-scale", repr(voxel), "--influence-scale", repr(influence),
               "--radius-scale", repr(radius), "--smoothing", str(smoothing)]
    if node.parm("surfacesolids") is not None and node.parm("surfacesolids").evalAsString() == "cut":    #overlap is flip2 mesh's own
        options += ["--solids", "cut"]
    if fields:
        if node.evalParm("fieldvoxelscale") <= 0:
            raise hou.NodeError("the fields' voxel scale has to be positive")
        options += ["--fields", "--field-voxel-scale", repr(node.evalParm("fieldvoxelscale"))]
    if not node.evalParm("meshsurface"):
        if not fields:
            raise hou.NodeError("nothing to mesh: turn on Mesh the Surface or Output Fluid Fields")
        options.append("--no-surface")
    return options


def _start(node, kind, upload=True):
    """starts a bake ("bake", or "resume" from the newest checkpoint), meshing its frames as they're committed if the Surface tab says to, or meshes
    the bake's frames again ("mesh"); returns at once, and _watch follows it"""
    if node.path() in _jobs:
        raise hou.NodeError("a bake is running already")
    meshing = _mesh_options(node) if kind == "mesh" or node.evalParm("meshsurface") or node.evalParm("outputfields") else None
    chosen, _ = _chosen_gpus(node)  #the GPUs its processes may use, and see alone
    visible = ",".join(chosen) if chosen else None
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=visible) if visible else None
    local = _output_directory(node)
    logs = os.path.join(local, "logs")
    os.makedirs(logs, exist_ok=True)
    name = "mesh" if kind == "mesh" else "bake"     #its logs: events, and what it says on standard error
    for stale in ("finished", name + ".events.jsonl", "bake.pid"):     #an earlier one's
        if os.path.exists(os.path.join(logs, stale)):
            os.remove(os.path.join(logs, stale))
    exporting = ["--format", node.parm("exportformat").evalAsString()]
    job = {"process": None, "export": None, "mesh": None, "events": os.path.join(logs, name + ".events.jsonl"), "log": os.path.join(logs, name + ".log"),
           "kind": kind, "meshing": meshing, "exporting": exporting, "state": "starting", "failure": None, "said": None, "shown": {}}
    with open(os.path.join(logs, kind + ".options.txt"), "w") as asked:    #what this job was asked for, as the node read its parameters when it started
        asked.write("%s %s\nbake on: %s\ngpus: %s\nexport: %s\nmesh: %s\n" % (kind, time.strftime("%Y-%m-%d %H:%M:%S"), ("this machine", "remote")[node.evalParm("bakeon")],
                                                                              visible or "all", " ".join(exporting),
                                                                              " ".join(meshing) if meshing is not None else "none (Mesh the Surface and Output Fluid Fields off)"))
    if node.evalParm("bakeon") == 1:
        host, directory, program = _remote(node)
        job["remote"] = (host, directory)
        arguments = {"bake": ["bake", "scene.json", "--out", "bake", "--overwrite"], "resume": ["resume", "bake"], "mesh": ["mesh"]}[kind]
        arguments = ["--export", " ".join(exporting)] + arguments
        if meshing is not None:
            arguments = ["--mesh", " ".join(meshing)] + arguments
        threading.Thread(target=_launch_remote, args=(job, host, directory, program, arguments, upload, local, logs, visible), daemon=True).start()
    else:
        program = _program(node)
        bake = os.path.join(local, "bake")
        job["program"] = program
        if kind == "mesh":
            job["process"] = subprocess.Popen([program, "mesh", bake, "--overwrite"] + meshing, stdout=open(job["events"], "w"),
                                              stderr=open(job["log"], "w"), cwd=local, env=environment)
        else:
            if kind == "bake":      #from the start: an earlier bake's cache and what was made of it go first, or the exporter and mesher following this
                for part in ("frames", "frames.discarded", "checkpoints", "export"):    #one would take them for its own, and stop
                    shutil.rmtree(os.path.join(bake, part), ignore_errors=True)
                if os.path.exists(os.path.join(bake, "cache.json")):
                    os.remove(os.path.join(bake, "cache.json"))
            arguments = {"bake": ["bake", os.path.join(local, "scene.json"), "--out", bake, "--overwrite"], "resume": ["resume", bake]}[kind]
            job["process"] = subprocess.Popen([program] + arguments, stdout=open(job["events"], "w"), stderr=open(job["log"], "w"), cwd=local, env=environment)
            job["export"] = subprocess.Popen([program, "export", bake, "--follow"] + exporting, stdout=open(os.path.join(logs, "export.events.jsonl"), "w"),
                                             stderr=open(os.path.join(logs, "export.log"), "w"), cwd=local, env=environment)
            if meshing is not None:
                job["mesh"] = subprocess.Popen([program, "mesh", bake, "--follow"] + meshing, stdout=open(os.path.join(logs, "mesh.events.jsonl"), "w"),
                                               stderr=open(os.path.join(logs, "mesh.log"), "w"), cwd=local, env=environment)
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
    for name in (importer.SURFACE, importer.PARTICLES, importer.FIELDS, importer.WHITEWATER, importer.BUBBLES, importer.BODIES):
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
        for follower, command in (("export", ["export", bake] + job["exporting"]), ("mesh", ["mesh", bake] + (job["meshing"] or []))):
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
            had = node.parm("exportformat") is not None
            dimpled = all(node.parm(name) is not None and abs(node.evalParm(name) - value) < 1e-6 for name, value in OLD_SURFACE_DEFAULTS.items())
            _interface(node)
            _network(node)
            if dimpled:     #still at the surface settings it was made with: it follows them to today's
                for name, value in SURFACE_DEFAULTS.items():
                    node.parm(name).set(value[0])
            folder = node.evalParm("outputdir").rstrip("/")
            found = importer.detect_format(os.path.join(folder, "bake", "export", "houdini")) if folder else None
            if not had and found is not None:      #older than Export Format: its frames are whatever it wrote
                node.parm("exportformat").set(found)
        elif kind == "import":
            importer._interface(node)
            importer._network(node)
            importer._match_format(node)
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
