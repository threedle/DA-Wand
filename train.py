# IntSeg: Training that allows for simultaneous batching of label/unlabelled shapes 
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from util.util import clear_directory
from test import run_test
import numpy as np
import torch
from collections import defaultdict
import os
import copy 
import dill as pickle 

if __name__ == '__main__':    
    opt = TrainOptions().parse()
    opt.phase = 'train' 

    if not opt.overwrite and os.path.exists(os.path.join(opt.export_save_path, opt.name, "test_prauc_batch.png")) and not opt.continue_train:
        print(f"Already done with {opt.name}.")
        exit()
    elif opt.overwrite and not opt.continue_train: 
        if os.path.exists(os.path.join(opt.export_save_path, opt.name)):
            clear_directory(os.path.join(opt.export_save_path, opt.name))
        # Also clear cache directories
        if os.path.exists(os.path.join(opt.dataroot, opt.phase, opt.cachefolder)):
            clear_directory(os.path.join(opt.dataroot, opt.phase, opt.cachefolder))
        if os.path.exists(os.path.join(opt.dataroot, opt.phase, opt.anchorcachefolder)):
            clear_directory(os.path.join(opt.dataroot, opt.phase, opt.anchorcachefolder))
        meanstd_path = os.path.join(opt.dataroot, opt.phase, f'mean_std_{opt.name}_cache.p')
        if os.path.exists(meanstd_path):
            os.remove(meanstd_path)
            
    dataset = DataLoader(opt)
    dataset_size = len(dataset.dataset)
    print('#training meshes = %d' % dataset_size)
    
    if opt.run_test_freq > 0:
        # Generate test cache once 
        testopt = TestOptions().parse()
        testopt.phase = 'test'
        testopt.serial_batches = True  # no shuffle
        testopt.shuffle_topo = False 
        testopt.load_pretrain = False 
        
        if testopt.overwrite:  
            # Also clear cache directories
            if os.path.exists(os.path.join(testopt.dataroot, testopt.phase, testopt.cachefolder)):
                clear_directory(os.path.join(testopt.dataroot, testopt.phase, testopt.cachefolder))
            if os.path.exists(os.path.join(testopt.dataroot, testopt.phase, testopt.anchorcachefolder)):
                clear_directory(os.path.join(testopt.dataroot, testopt.phase, testopt.anchorcachefolder))
            # Clear mean std
            meanstd_path = os.path.join(testopt.dataroot, testopt.phase, f'mean_std_{testopt.name}_cache.p')
            if os.path.exists(meanstd_path):
                os.remove(meanstd_path)
        
        print("Generating test cache...")
        testdata = DataLoader(testopt)
        print(f'# testing meshes = {len(testdata.dataset)}')

    # Pickle options (for easier inference)
    with open(os.path.join(opt.export_save_path, opt.name, "opt.pkl"), 'wb') as f: 
        pickle.dump(opt, f)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    
    # If continuing training, then load the values from cache
    from pathlib import Path
    cachedir = os.path.join(opt.export_save_path, opt.name, "cache")
    Path(cachedir).mkdir(exist_ok=True, parents=True)
    if opt.continue_train == True and os.path.exists(os.path.join(cachedir, "losses.pkl")) and os.path.exists(os.path.join(cachedir, "dicts.pkl")):
        print(f"Loading losses and metrics from cache...") 
        with open(os.path.join(cachedir, "losses.pkl"), 'rb') as f:
            avg_losses, test_losses, test_epochs, best_test_stats  = pickle.load(f)
    else:
        avg_losses = []
        test_losses = []
        test_epochs = []
        
        best_test_stats = 0 
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): 
        if epoch == opt.delayed_distortion_epochs:
            print(f"Epoch {epoch}: introducing distortion supervision")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = 0
            
        for i, data in enumerate(dataset):
            # Edge case: if whole batch is invalid then data is None 
            if data is None: 
                print(f"Warning: batch {i} skipped because no valid samples.")
                
            if opt.time == True: 
                # Memory profiling 
                import psutil
                print(f"===== Epoch {epoch}, batch {i} ======")
                if torch.cuda.is_available():
                    import time
                    # Get GPU memory usage 
                    t = torch.cuda.get_device_properties(0).total_memory
                    r = torch.cuda.memory_reserved(0)
                    a = torch.cuda.memory_allocated(0)
                    m = torch.cuda.max_memory_allocated(0)
                    f = r-a  # free inside reserved
                    print(f"{a/1024**3:0.3f} GB allocated. \nGPU max memory alloc: {m/1024**3:0.3f} GB. \nGPU total memory: {t/1024**3:0.3f} GB.\n")
                    
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            success = model.set_input(data)
            if not success:
                continue 
            else:
                loss, loss_dict = model.optimize_parameters(epoch)
            
            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / len(data['meshdata'])
                
                # Extract individual loss components from loss dict 
                tmp_losses_dict = defaultdict(float)
                for meshname, tmp_dict in loss_dict.items():
                    for key, val in tmp_dict.items():
                        tmp_losses_dict[key] += val 
                
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data, tmp_losses_dict)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
            epoch_loss += loss

            iter_data_time = time.time()            

            # If first epoch, save best
            if epoch == 0: 
                model.save_network("best")
                
            # Janky way to set max sample size while randomizing the samples 
            if epoch_iter >= opt.max_sample_size: 
                print(f"Breaking epoch after {opt.max_sample_size} samples.")
                break
        
        print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
        model.save_network(epoch, wipe=True)
        model.save_network('latest')
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        # Cache losses 
        avg_losses.append(epoch_loss / dataset_size)
        with open(os.path.join(cachedir, "losses.pkl"), 'wb') as f:
            pickle.dump((avg_losses, test_losses, test_epochs, best_test_stats), f)
    
        if opt.lr_policy:
            model.update_learning_rate()
            
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if opt.run_test_freq > 0 and epoch % opt.run_test_freq == 0:
            print("Running test...")
            # NOTE: This means every shape gets equal weight in the mean
            # TODO: keep track of parameterization distortion here 
            testopt.which_epoch = epoch 
            avg_loss, metricnames, avgstats = run_test(epoch, testdata, testopt)
                    
            test_losses.append(avg_loss)
            test_epochs.append(epoch)
            if len(avgstats) > 0:
                # Only track best stats we care about 
                best_stats_candidate = [avgstats[i] for i in range(len(avgstats)) if '% I < 0.05' in metricnames[i]]
                mean_avgstats = np.mean(best_stats_candidate)
                if  best_test_stats < mean_avgstats:
                    model.save_network('best')
                    best_test_stats = mean_avgstats
                    
                    print(f"Saved best network at epoch {epoch}")
            
            with open(os.path.join(cachedir, "losses.pkl"), 'wb') as f:
                pickle.dump((avg_losses, test_losses, test_epochs, best_test_stats), f)
                
    writer.close()

    # Plot loss and test accuracy graphs
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(avg_losses)), avg_losses, label="Train")
    ax.plot(test_epochs, test_losses, label="Test")
    ax.set_title(f"Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(opt.export_save_path, opt.name, f"loss.png"))
    plt.cla()
    plt.close()
