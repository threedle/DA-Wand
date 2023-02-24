from data.intseg_data import IntSegData
import dill as pickle
from models.layers.meshing.analysis import computeFaceAreas, computeDihedrals
from models import create_model
from models.layers.meshing import Mesh
from models.layers.meshing.io import PolygonSoup
from models.layers.meshing.edit import VertexStarCollapse, EdgeCollapse
from models.networks import floodfill_scalar_v1, floodfill_scalar_v2
from util.util import graphcuts 
import numpy as np
import os 
import torch 
import random 
from pathlib import Path
import polyscope as ps
import polyscope.imgui as psim 
import sys 
import argparse 

def run_forward_pass(model, dataset, face_list, return_features=False):
    # TODO: Will likely need to debug this so all the preprocessing works on multiple anchors 
    dataset.update_anchor(face_list)
    input_meta = dataset[0] 
    input_meta = {key: [val] for key,val in input_meta.items()}
    
    # input_meta['mean'] = [np.zeros(input_meta['edge_features'][0].shape[0])]
    # input_meta['std'] = [np.ones(input_meta['edge_features'][0].shape[0])]
    # input_meta['edge_features'] = [np.zeros_like(input_meta['edge_features'][0])]
    
    model.set_input(input_meta)
    
    with torch.no_grad():
        preds, features = model.forward()
        
        if return_features: 
            return preds, features
                
    return preds 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modeldir', required=True, type=str, help='path to saved network')
    parser.add_argument('--meshdir', default="./datasets/meshgen/developable_highres/test", help='path to mesh data')
    parser.add_argument('--modelname', default="best", help='loads \{modelname\}_net.pth')
    parser.add_argument('--meshfile', default="sample81.obj", help='loads obj file from the meshdir')
    parser.add_argument('--optname', default="opt", help='loads \{optname\}.pkl from modeldir')
    parser.add_argument('--normalize', action="store_true", help='normalize mesh to unit sphere and save new base file')
    
    args = parser.parse_args()
    
    with open(os.path.join(args.modeldir, f"{args.optname}.pkl"), 'rb') as f:
        opt = pickle.load(f)
    opt.export_save_path = args.modeldir
    opt.dataroot = args.meshdir 
    opt.test_dir = args.meshdir
    opt.network_load_path = args.modeldir 
    
    # TODO: temporary opt fix 
    opt.arch = "meshcnn"
    
    # Turn off all extraneous settings 
    opt.name = ""
    opt.time = False 
    opt.semantic_loss = False
    opt.is_train = False 
    opt.testaug = False 
    opt.num_aug = 0
    opt.serial_batches = True  # no shuffle
    opt.time = False 
    opt.export_view_freq = 0
    opt.export_preds = False 
    opt.continue_train = False 
    opt.floodfillparam = False 
    opt.test_aug = False
    opt.which_epoch = args.modelname
    opt.ff_epoch = args.modelname
    opt.export_pool = False 
    opt.distortion_loss = None 
    opt.delayed_distortion_epochs = float('inf')
    opt.solo_distortion = False
    opt.shuffle_topo = False 
    opt.num_threads = 0 
    opt.supervised = False 
    opt.gcsupervision = False 
    opt.floodfillparam = False 
    
    # NOTE: hacky way to guarantee only 1 copy of mesh in dataset but it works 
    meshname = args.meshfile.replace(".obj", "")
    opt.subset = [f"{meshname}_0"] 
    opt.max_dataset_size = 1
    opt.max_sample_size = 1
    opt.interactive = True 
        
    opt.overwritecache = True 
    # opt.overwriteopcache = True   
    opt.overwriteanchorcache = True   
    opt.overwritemeanstd = True  
    if not torch.cuda.is_available():
        opt.gpu_ids = []
    else:
        opt.gpu_ids = [0] 
    model = create_model(opt)
    print(f"Model loaded from {model.save_dir}")
    
    dataset = IntSegData(opt)
    soup = PolygonSoup.from_obj(os.path.join(args.meshdir, args.meshfile))
    mesh = Mesh(soup.vertices, soup.indices)
    
    if args.normalize:
        mesh.normalize() 
        mesh.export_obj(args.meshdir, f"{meshname}_norm")
    
    current_index_list = []
    current_anchor_pos = [] 
    previous_preds = None 
    preds = None
    pred_cache = {} 
    mode = "model"
    prev_mode = "model"
    mode_options = ['model']
    
    # Default settings 
    alpha = 1 
    beta = 0.7
    gamma = 0.5 
    dthreshold = 0.3 
    ethreshold = 100 
    method = None 
    postprocess = ["gc", "ff"] 
    ff = True 
    gc = True 
    uvmode = False 
    include_anchor = True 
    patchgrow = False 
    face_index = None 
    
    def callback():
        # If we want to use local variables & assign to them in the UI code below, 
        # we need to mark them as nonlocal. This is because of how Python scoping 
        # rules work, not anything particular about Polyscope or ImGui.
        # Of course, you can also use any other kind of python variable as a controllable 
        # value in the UI, such as a value from a dictionary, or a class member. Just be 
        # sure to assign the result of the ImGui call to the value, as in the examples below.
        # 
        # If these variables are defined at the top level of a Python script file (i.e., not
        # inside any method), you will need to use the `global` keyword instead of `nonlocal`.
        
        global ps_mesh, ps_curve, mesh, model, dataset, current_index_list, previous_preds
        global current_anchor_pos, preds, postprocess
        global uvmode, isometric, method, pred_cache, mode, mode_options, alpha, beta, gamma, dthreshold, ethreshold
        global prev_mode, ff, gc, include_anchor, patchgrow, face_index 
        
        vertices, faces, _ = mesh.export_soup()
        vrange = np.arange(len(vertices))
        frange = np.arange(len(vertices), len(vertices) + len(faces))
        
        # == Settings
        # Use settings like this to change the UI appearance.
        # Note that it is a push/pop pair, with the matching pop() below.
        psim.PushItemWidth(150)

        # == Title window 
        psim.TextUnformatted("DA Wand Interactive Module")
        psim.TextUnformatted(f"Current anchor list: {current_index_list}")
        n_selection = len(np.where(preds >= 0.5)[0]) if preds is not None else 0 
        psim.TextUnformatted(f"# faces in selection: {n_selection}")
        psim.Separator()
        if uvmode: 
            psim.TextUnformatted(f"UV Method: {method}")
            # psim.TextUnformatted(f"UV mean isometric distortion: {isometric:0.4f}")
            psim.Separator()
        # == Get currently selected 
        current_struct = ''
        structure, index = ps.get_selection() 
        
        # Clear current anchor set 
        if(psim.Button("Clear anchors")):
            ps.remove_all_structures()
            previous_preds = None 
            preds = None
            ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices, edge_width=1, enabled=True)
            psim.CloseCurrentPopup()
            dataset.anchor_fs = [] 
            current_index_list = []
            current_anchor_pos = [] 
            pred_cache = {} 
            uvmode = False 
            isometric = 0
            method = None 
            ps.show() 
        
        # changed, include_anchor = psim.Checkbox("Anchor include mode", include_anchor) 
        changed, patchgrow = psim.Checkbox("Patch growing (experimental)", patchgrow) 
        psim.Separator()
        
        # Postprocess options  
        psim.Separator() 
        if mode == "model":
            changed, ff = psim.Checkbox("floodfill", ff) 
            if changed: 
                if ff:
                    # Should never be able to turn on FF when already in postprocess list 
                    assert "ff" not in postprocess, f"Error: postprocess ff should always be unique"
                    postprocess.append("ff")
                    
                    # Check if postprocess preds already cached
                    if preds is not None:
                        predkey = "_".join(['model'] + postprocess)
                        if predkey in pred_cache.keys(): 
                            preds = pred_cache[predkey]
                        else: 
                            preds = floodfill_scalar_v2(mesh, torch.from_numpy(preds).float(), face_index, previous_preds=torch.from_numpy(previous_preds).float() if (previous_preds is not None and patchgrow) else None).detach().numpy()
                            pred_cache[predkey] = preds                     
                else: 
                    # Remove ff from postprocesses and check if new cachekey exists 
                    postprocess = [p for p in postprocess if p != "ff"]
                    predkey = "_".join(['model'] + postprocess)
                    if predkey in pred_cache.keys(): 
                        preds = pred_cache[predkey]
                    else: 
                        # If raw prediction not present, then do nothing
                        if "model" in pred_cache.keys():
                            for process in postprocess: # NOTE: order matters
                                if process == 'gc':
                                    preds = graphcuts(preds, mesh, anchors=current_index_list) 
                            pred_cache[predkey] = preds      
                if preds is not None:               
                    hard_preds = np.round(preds)
                    ps_mesh.add_scalar_quantity("model soft segmentation", preds, defined_on='faces', enabled=True, vminmax=(0,1))
                    ps_mesh.add_scalar_quantity("model hard segmentation", hard_preds, defined_on='faces', enabled=True, vminmax=(0,1))
                    psim.CloseCurrentPopup()
                    ps.show()
            changed, gc = psim.Checkbox("graphcuts", gc) 
            if changed: 
                if gc:
                    # Should never be able to turn on FF when already in postprocess list 
                    assert "gc" not in postprocess, f"Error: postprocess gc should always be unique"
                    postprocess.append("gc")
                    
                    # Check if postprocess preds already cached
                    if preds is not None:
                        predkey = "_".join(['model'] + postprocess)
                        if predkey in pred_cache.keys(): 
                            preds = pred_cache[predkey]
                        else: 
                            preds = graphcuts(preds, mesh, anchors=current_index_list)
                            pred_cache[predkey] = preds                     
                else: 
                    # Remove gc from postprocesses and check if new cachekey exists 
                    postprocess = [p for p in postprocess if p != "gc"]
                    predkey = "_".join(['model'] + postprocess)
                    if predkey in pred_cache.keys(): 
                        preds = pred_cache[predkey]
                    else: 
                        # If raw prediction not present, then do nothing
                        if "model" in pred_cache.keys():
                            for process in postprocess: # NOTE: order matters
                                if process == 'ff':
                                    preds = floodfill_scalar_v2(mesh, torch.from_numpy(preds).float(), face_index, previous_preds=torch.from_numpy(previous_preds).float() if (previous_preds is not None and patchgrow) else None).detach().numpy()
                            pred_cache[predkey] = preds        
                if preds is not None:             
                    hard_preds = np.round(preds)
                    ps_mesh.add_scalar_quantity("model soft segmentation", preds, defined_on='faces', enabled=True, vminmax=(0,1))
                    ps_mesh.add_scalar_quantity("model hard segmentation", hard_preds, defined_on='faces', enabled=True, vminmax=(0,1))
                    psim.CloseCurrentPopup()
                    ps.show()
                
        # Compute and show SLIM UV (texture + embedding)
        changed, uvmode = psim.Checkbox("Show UV", uvmode) 
        if changed:
            if uvmode: 
                # NOTE: UV value in the pred cache will be a DICT
                uvkey = "_".join(['model'] + postprocess + ['uv'])
                if uvkey in pred_cache: 
                    uvdict = pred_cache[uvkey] 
                    uv = uvdict['uv']
                    method = uvdict['method']
                    isometric = uvdict['isometric']
                    subvs = uvdict['subvs']
                    subfs = uvdict['subfs']
                    
                    selection = np.where(preds >= 0.5)[0]
                else: 
                    from util.util import run_slim

                    # Compute UVs using SLIM 
                    selection = np.where(preds >= 0.5)[0]
                    subvs, subfs = mesh.export_submesh(selection)
                    submesh = Mesh(subvs, subfs)
                    slim_uv, slim_energy, did_cut = run_slim(submesh, cut=True)
                    subvs, subfs, _ = submesh.export_soup() 
                    if slim_energy > 100:  
                        print(f"Warning: SLIM bad convergence")
                    
                    uv = slim_uv
                    min_uv = np.min(uv)
                    if min_uv < 0: 
                        uv -= min_uv
                    max_uv = np.max(uv)
                    if max_uv > 1: 
                        uv /= max_uv
                        
                    isometric = 0
                    method = "SLIM"
                    
                    # Set cache values 
                    uvdict = {"uv": uv, "isometric": isometric, "method": method, "subvs": subvs, 'subfs': subfs} 
                    pred_cache[uvkey] = uvdict 
                    subvs, subfs, _ = submesh.export_soup() 
                
                # Turn off current mesh 
                ps_mesh.set_enabled(False)
                    
                # Re-render new UV objects 
                ps_sub = ps.register_surface_mesh("selectmesh", subvs, subfs, edge_width=1)
                ps_sub.add_parameterization_quantity("uv", uv, defined_on='vertices', enabled=True, viz_style="checker")
                
                # Re-render full mesh without the selection region 
                other_selection = np.array(list(set(range(len(faces))).difference(set(selection))))
                othervs, otherfs = mesh.export_submesh(other_selection)
                ps_other = ps.register_surface_mesh("other mesh", othervs, otherfs, edge_width=1)
                
                ps_emb = ps.register_surface_mesh("embedding", uv, subfs, edge_width=1, enabled=False)
                ps_emb.add_parameterization_quantity("uv", uv, defined_on='vertices', enabled=True, viz_style="checker")
                psim.CloseCurrentPopup()
                ps.show()
            else: 
                # Turn back on the original values 
                ps_mesh.set_enabled(True)
                psim.CloseCurrentPopup()
                ps.show()                 

        # == Execute inference if change detected in selection (append anchor to inference) ==
        face_index = index - min(frange)
        new_selection = (face_index not in current_index_list)
        if structure == "mesh" and new_selection and index in frange: 
            print(f"Current anchors list: {current_index_list}, New anchor: {face_index}")
            if patchgrow: 
                current_index_list.append(face_index) 
            else: 
                current_index_list = [face_index]
                current_anchor_pos = [] 
                preds = None 
                previous_preds = None 
            
            # Run inference on new anchor set 
            if mode == "model": 
                preds = run_forward_pass(model, dataset, current_index_list)
                preds = preds.squeeze().detach().cpu().numpy()  
                ps_mesh.add_scalar_quantity("model raw predictions", preds, defined_on='faces', enabled=True, vminmax=(0,1))
                # Reset the pred cache 
                pred_cache = {'model': preds}

                # Run postprocesses 
                predkey = "model"
                for post in postprocess:
                    if post == "gc":
                        preds = graphcuts(preds, mesh, anchors=current_index_list)
                        predkey += "_gc"
                        pred_cache[predkey] = preds
                    if post == "ff":
                        preds = floodfill_scalar_v2(mesh, torch.from_numpy(preds).float(), face_index, previous_preds = torch.from_numpy(previous_preds).float() if (previous_preds is not None and patchgrow) else None).detach().numpy()
                        predkey += "_ff"
                        pred_cache[predkey] = preds 
                            
                previous_preds = preds 
                hard_preds = np.round(preds)
                ps_mesh.add_scalar_quantity("model soft segmentation", preds, defined_on='faces', enabled=True, vminmax=(0,1))
                ps_mesh.add_scalar_quantity("model hard segmentation", hard_preds, defined_on='faces', enabled=True, vminmax=(0,1))
           
            anchor_pos = np.mean([mesh.vertices[v.index] for v in mesh.topology.faces[face_index].adjacentVertices()], axis=0, keepdims=True)
            current_anchor_pos.append(anchor_pos)
            
            # Anchor colors are all fixed except most recent 
            anchor_colors = [[0,0,1] for _ in current_anchor_pos[:-1]] + [[0,1,0]]
            ps_curve = ps.register_curve_network("anchors", np.concatenate(current_anchor_pos, axis=0), np.array([[i,i] for i in range(len(current_anchor_pos))]),
                                                 enabled=True)
            ps_curve.add_color_quantity("acolor", np.array(anchor_colors), enabled=True)
            psim.CloseCurrentPopup()
            ps.show() 
        
        psim.Separator()
        if(psim.Button("Export UV")):
            uvkey = "_".join(['model'] + postprocess + ['uv'])
            if uvkey in pred_cache: 
                uvdict = pred_cache[uvkey] 
                uv = uvdict['uv']
                selection = np.where(preds >= 0.5)[0]
                subvs, subfs = mesh.export_submesh(selection) 
                submesh = Mesh(subvs, subfs) 
                submesh.export_obj(args.meshdir, args.meshfile.replace(".obj", "_seg"), uv, submesh.faces)
            
        psim.Separator()
          
        if(psim.Button("Exit")):
            exit()     
              
        psim.PopItemWidth()
        
    ps.init() 
    ps.set_navigation_style("free")
    ps_mesh = ps.register_surface_mesh("mesh", soup.vertices, soup.indices, edge_width=1)
    isometric = 0 
    ps.set_invoke_user_callback_for_nested_show(True)
    ps.set_user_callback(callback)
    ps.show()