from re import L
import bpy
import os
import sys
import numpy as np


'''
@input: 
    <obj_file>
    <outfile> name of output deformed .obj file
@output:
    deformed mesh
    
to run it from cmd line:
/opt/blender/blender --background --python blender_deform.py /home/rana/code/remesh/datasets/raw/cube.obj /home/rana/Downloads/temp.obj
'''

class Process:
    def __init__(self, obj_file, export_name, n=3):
        mesh = self.load_obj(obj_file)
        for _ in range(n):
            self.deform(mesh)
        self.export_obj(mesh, export_name)

    def load_obj(self, obj_file):
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_edges=True,
                                 use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
        ob = bpy.context.selected_objects[0]
        return ob

    def deform(self, mesh):
        # randomly deform in X Y or Z axis
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new(name='SimpleDeform', type='SIMPLE_DEFORM')
        mod.angle = np.random.uniform(low=-0.5, high=0.5)
        mod.factor = np.random.uniform(low=-0.8, high=0.8)
        mod.deform_axis = np.random.choice(['X', 'Y', 'Z'])
        mod.deform_method = np.random.choice(['TWIST', "BEND", "TAPER", "STRETCH"])
        
        # For debugging
        print(f"Deform: {mod.deform_method}\nAxis: {mod.deform_axis}\nAngle: {mod.angle}\nFactor: {mod.factor}")

    def export_obj(self, mesh, export_name):
        print('EXPORTING', export_name)
        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(state=True)
        bpy.ops.export_scene.obj(filepath=export_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                 use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True,
                                 use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True,
                                 use_uvs=False, use_materials=False, use_triangles=False, use_nurbs=False,
                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                 group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO',
                                 axis_forward='-Z', axis_up='Y')

if __name__ == "__main__":  
    datadir = "/Users/guanzhi/Documents/InteractiveSegmentation/MeshCNNFeatures/datasets/meshgen/source_highres"
    outdir = "./datasets/meshgen/polyhedrons_def_highres"
    import shutil 
    for filename in os.listdir(outdir):
        file_path = os.path.join(outdir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    n = 19
    for file in os.listdir(datadir):
        if not file.endswith(".obj"):
            continue 
        meshname = os.path.splitext(file)[0] 
        # TODO: We start with simple shapes first 
        if meshname in ['dodecahedron', 'icosahedron']:
            continue
        for i in range(n):
            n_def = np.random.choice(range(3,10))
            Process(os.path.join(datadir, file), os.path.join(outdir, f"{meshname}{i}.obj"), n=n_def)