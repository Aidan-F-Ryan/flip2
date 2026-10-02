"""flip2 in Houdini.

The nodes are built from Python when Houdini runs: subnets with their own parameters, made by tools in the Tab menu and on the flip2 shelf. Nothing
licence-tagged ships with them, so they work the same in Apprentice, Indie and FX, and saved scenes keep them as ordinary subnets.

    flip2 Solver        sets a simulation up from geometry, bakes it with the flip2 program, and loads it back (solver.py)
    flip2 Import        loads a bake's surface, particles and fluid fields at the current time (importer.py)
    flip2 Whitewater    sets Houdini's whitewater up on a bake, wired to its fields and sized for it (whitewater.py)
"""
from .importer import create_import, read_bake
from .solver import create_solver, write_scene
from .whitewater import create_whitewater
