# DAWand: generate deformed primitives using Blender 
from re import L
import bpy
import os
import sys
import numpy as np
from pathlib import Path

'''
@input: 
    <obj_file>
    <outfile> name of output deformed .obj file
@output:
    deformed mesh
    
to run it from cmd line:
blender --background --python blender_deform.py -- --datadir ./datasets/meshgen/source_highres --exportdir ./datasets/deformed_primitives
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
    import sys 
    
    # Important: all named arguments should be passed after a double dash following the python script call e.g. '--python script.py -- -arg1 -arg2'
    argv = sys.argv[sys.argv.index('--') + 1:]
    
    import argparse 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', required=True, type=str, help='path to base primitives with ground truth segmentations in a "labels" folder')
    parser.add_argument('--exportdir', required=True, type=str, help='export path')
    parser.add_argument('--n', type=int, default=10, help='number of deformed meshes to generate per primitive')
    parser.add_argument('--ndef_low', type=int, default=3, help='lower bound of deformations to apply')
    parser.add_argument('--ndef_high', type=int, default=10, help='upper bound of deformations to apply')
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args(argv)
    
    n = args.n 
    exportdir = args.exportdir 
    datadir = args.datadir 
    
    if not os.path.exists(exportdir):
        Path(exportdir).mkdir(exist_ok=True, parents=True)
        
    # Copy labels folder over 
    # import shutil 
    # shutil.copytree(os.path.join(datadir,"labels"), os.path.join(exportdir,"labels"))
    
    if args.overwrite:
        for filename in os.listdir(exportdir):
            file_path = os.path.join(exportdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    for file in os.listdir(datadir):
        if not file.endswith(".obj"):
            continue 
        meshname = os.path.splitext(file)[0] 
        for i in range(n):
            n_def = np.random.choice(range(args.ndef_low, args.ndef_high))
            Process(os.path.join(datadir, file), os.path.join(exportdir, f"{meshname}{i}.obj"), n=n_def)