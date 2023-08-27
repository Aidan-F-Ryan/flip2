bl_info = {
    "name": "FLUID SOLVER",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy

class FluidSolver(bpy.types.Operator):
    """Fluid Solver"""
    bl_idname = "fluid.solver"
    bl_label = "Simulate fluid motion of particles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        # for obj in scene.objects:
            # pass

        return {'FINISHED'}
    
def menu_func(self, context):
    self.layout.operator(FluidSolver.bl_idname)

def register():
    bpy.utils.register_class(FluidSolver)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(FluidSolver)

if __name__ == "__main__":
    register()