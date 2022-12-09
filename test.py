from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from collections import defaultdict
import numpy as np
import os 
from models.layers.meshing.mesh import Mesh 
from util.util import get_local_tris, get_ss, run_slim
from models.networks import floodfill_scalar_v2
import torch 

def run_test(epoch, dataset, opt):
    from util.losses import compute_pr_auc, mAP, f1
    print('Running Test')
    
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    metricsdict = defaultdict(lambda: defaultdict(list))
    distortdict = defaultdict(lambda: defaultdict(list))
    lossesdict = defaultdict(lambda: defaultdict(float))
    meshcounts = defaultdict(int)
    total_loss = 0.0 
    
    # Isometric, Conformal, ARAP, # faces 
    # Collect these values when running test() function 
    count = 0 
    for i, data in enumerate(dataset):
        success = model.set_input(data)
        if not success:
            continue 
        labels, preds, loss, loss_dict = model.test(epoch) # labels: (M x V x 1); preds: (M x V x 1)
        total_loss += loss
        # Batch by shape
        for j in range(len(data['meshdata'])):
            meshdata = data['meshdata'][j]
            meshname = os.path.splitext(data['file'][j])[0]
            meshno = data['no'][j]
            anchor = data['anchor_fs'][j][0]
            
            if labels[j] is None: 
                # Distortion statistics of the selection 
                # NOTE: This will be WITHOUT floodfill 
                mesh = Mesh(meshdata=meshdata)
                
                # Need to run floodfill preprocess before SLIM will work 
                meshpreds = preds[j][:len(mesh.faces)]
                meshpreds = floodfill_scalar_v2(mesh, torch.from_numpy(meshpreds).squeeze().float(), anchor).detach().numpy()
                select = np.where(meshpreds >= 0.5)[0]
                
                # Edge case: if everything selected, then remove one face
                if len(select) >= len(meshpreds):
                    select = select[:-1]
                
                subvs, subfs = mesh.export_submesh(select)
                if len(subfs) > 0:
                    submesh = Mesh(subvs, subfs) 
                    
                    # Run SLIM 
                    # TODO: Need one more floodfill postprocess after cutting 
                    uvmap, slim_energy, did_cut = run_slim(submesh, cut=True)
                            
                    # SS 
                    ss = get_ss(submesh, uvmap)
                    isometric = np.maximum(ss[:,0], 1/ss[:,1])
                    isometric = (isometric - 1) ** 2
                    conformal = (ss[:,0] - ss[:,1])**2
                    
                    distortdict['Isometric'][meshname].append(np.nanmean(isometric))
                    distortdict['Conformal'][meshname].append(np.nanmean(conformal)) 
                    distortdict['% I < 0.05'][meshname].append(np.sum(isometric <= 0.05)/len(meshdata['faces']))
                    distortdict['% C < 0.05'][meshname].append(np.sum(conformal <= 0.05)/len(meshdata['faces']))
                else:
                    # NOTE: Floodfill fail/null selection 
                    distortdict['Isometric'][meshname].append(np.nan)
                    distortdict['Conformal'][meshname].append(np.nan) 
                    distortdict['% I < 0.05'][meshname].append(0)
                    distortdict['% C < 0.05'][meshname].append(0)
                    
                distortdict['# Faces'][meshname].append(len(subfs))
            else:
                meshpreds = preds[j][:len(meshdata['faces'])]
                meshlabels = labels[j][:len(meshdata['faces'])]
                precision, recall, thresholds, auc_score = compute_pr_auc(meshlabels, meshpreds)
                ap = mAP(meshlabels, meshpreds)
                # F1 score: computed for a given threshold (default to 0.5 for now)
                pred_classes = np.round(meshpreds)
                f1_score = f1(meshlabels, pred_classes)
                # NOTE: meshname will be SAME across anchors
                metricsdict['prauc'][meshname].append(auc_score)
                metricsdict['f1'][meshname].append(f1_score)
                metricsdict['mAP'][meshname].append(ap)

            # Save loss components
            meshcounts[meshname] += 1
            if meshno in loss_dict.keys():
                for loss_key in loss_dict[meshno].keys():
                    # print(f"Saving loss: {mesh.no}, {loss_key}")
                    lossesdict[meshname][loss_key] += loss_dict[meshno][loss_key]
        
        count += len(data['meshdata'])
        # Janky way to set max sample size while randomizing the samples 
        if count >= opt.max_sample_size: 
            print(f"Breaking test epoch {epoch} after {opt.max_sample_size} samples.")
            break

    for meshname, meshdict in lossesdict.items():
        for key in meshdict.keys():
            meshdict[key] /= meshcounts[meshname]
    
    # Edge case: somehow no meshes in test data are valid 
    avg_auc = avg_f1 = avg_map = 0
    avgstats = [] 
    metricnames = [] 
    if len(metricsdict) > 0:
        avg_auc = np.mean([np.mean(aucs) for aucs in metricsdict['prauc'].values()])
        avg_f1 = np.mean([np.mean(f1s) for f1s in metricsdict['f1'].values()])
        avg_map = np.mean([np.mean(maps) for maps in metricsdict['mAP'].values()])
        avgstats = [avg_auc, avg_map, avg_f1]
        metricnames.extend(['prauc', 'mAP', 'f1'])
        writer.print_stats(epoch, avgstats, metricnames)
    # Record distortion values if set 
    if len(distortdict) > 0:
        printstats = [] 
        distortnames = []
        for name in distortdict.keys():
            distortnames.append(name)
            printstats.append(np.nanmean([np.nanmean(val) for val in distortdict[name].values()]))
            # We only care about isometric energy for the sake of the test metrics 
            if "%" in name:
                avgstats.append(np.mean([np.mean(val) for val in distortdict[name].values()]))
                metricnames.append(name)
        writer.print_stats(epoch, printstats, distortnames, distortion=True)
        
    return total_loss/len(dataset), metricnames, avgstats

if __name__ == '__main__':
    import numpy as np
    # Test test
    run_test()
