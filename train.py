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
    # meshnames = [] 
    # for i, data in enumerate(dataset):
    #     meshnames.extend(file for file in data['file'])
    # print(f"Meshes: {meshnames}")
    
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
        # testopt.overwriteopcache = True 
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
            avg_losses, test_losses, test_epochs, best_test_stats, \
                s1_distortions, s2_distortions, avg_contrastive_loss, \
                test_s1_distortion, test_s2_distortion, test_contloss = pickle.load(f)
        with open(os.path.join(cachedir, "dicts.pkl"), 'rb') as f:
            metricdict, batchdict, test_losses_dict, train_losses_dict, test_distort_dict = pickle.load(f)
    else:
        avg_losses = []
        test_losses = []
        test_epochs = []
        s1_distortions = [] 
        s2_distortions = [] 
        avg_contrastive_loss = [] 
        test_s1_distortion = [] 
        test_s2_distortion = [] 
        test_contloss = []
        
        metricdict = defaultdict(list)
        batchdict = defaultdict(lambda: defaultdict(list))
        test_losses_dict = defaultdict(lambda: defaultdict(list))
        train_losses_dict = defaultdict(lambda: defaultdict(list))
        test_distort_dict = defaultdict(lambda: defaultdict(list))
        
        best_test_stats = 0 
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): 
        if epoch == opt.delayed_distortion_epochs:
            print(f"Epoch {epoch}: introducing distortion supervision")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = 0
        epoch_acc = 0
        
        epoch_contloss = 0
        epoch_s1_distortion = 0 
        epoch_s2_distortion = 0
            
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
                        
            if opt.profile == True:
                from torch.profiler import profile, record_function, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    with record_function('model_train'):
                        loss, loss_dict = model.optimize_parameters(epoch)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50))
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
            pickle.dump((avg_losses, test_losses, test_epochs, best_test_stats, s1_distortions, s2_distortions,
                         avg_contrastive_loss, test_s1_distortion, test_s2_distortion, test_contloss), f)
        
        # Training batch losses 
        for meshname, tmp_dict in loss_dict.items():
            for key, val in tmp_dict.items():
                train_losses_dict[meshname][key].append(val)
                        
        with open(os.path.join(cachedir, "dicts.pkl"), 'wb') as f:
            pickle.dump((metricdict, batchdict, test_losses_dict, train_losses_dict, test_distort_dict), f)
        
        if opt.lr_policy:
            model.update_learning_rate()
            
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if opt.run_test_freq > 0 and epoch % opt.run_test_freq == 0:
            print("Running test...")
            # NOTE: This means every shape gets equal weight in the mean
            # TODO: keep track of parameterization distortion here 
            testopt.which_epoch = epoch 
            avg_loss, metricnames, avgstats, testdict, lossesdict, distortdict = run_test(epoch, testdata, testopt)
            # Save epoch metrics
            for j in range(len(metricnames)):
                metricname = metricnames[j]
                metricdict[metricname].append(avgstats[j])
                # Keyed by metric -> meshname
                for name, vals in testdict[metricname].items():
                    batchdict[metricname][name].append(np.mean(vals))
            # Save epoch losses
            for meshname, tmp_dict in lossesdict.items():
                for key, val in tmp_dict.items():
                    test_losses_dict[meshname][key].append(val)
                
            for distortname, meshdict in distortdict.items():
                for meshname, val in meshdict.items():
                    test_losses_dict[meshname][distortname].append(val)
                    
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
                pickle.dump((avg_losses, test_losses, test_epochs, best_test_stats, s1_distortions, s2_distortions,
                            avg_contrastive_loss, test_s1_distortion, test_s2_distortion, test_contloss), f)
            
            with open(os.path.join(cachedir, "dicts.pkl"), 'wb') as f:
                pickle.dump((metricdict, batchdict, test_losses_dict, train_losses_dict, test_distort_dict), f)
                
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
    
    # Training loss breakdown  
    for meshname, losshistory in train_losses_dict.items():
        fig, ax = plt.subplots()
        for name, vals in losshistory.items():
            ax.plot(range(len(avg_losses)), vals, label=name)
        ax.set_title(f"Train {meshname}: Loss Breakdown")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(opt.export_save_path, opt.name, f"train_{meshname}_loss.png"))
        plt.cla()
        plt.close()
    
    # Testing stats 
    for key, metrichistory in metricdict.items():
        fig, ax = plt.subplots()
        ax.plot(test_epochs, metrichistory)
        ax.set_title(f"Test {key}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(key)
        fig.tight_layout()
        plt.savefig(os.path.join(opt.export_save_path, opt.name, f"test_{key}.png"))
        plt.cla()
        plt.close()

    # Shape Batches
    for key, batchhistory in batchdict.items():
        fig, ax = plt.subplots()
        for name, vals in batchhistory.items():
            ax.plot(test_epochs, vals, label=name)
        ax.set_title(f"Test {key} by Shape")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(key)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(opt.export_save_path, opt.name, f"test_{key}_batch.png"))
        plt.cla()
        plt.close()

    # Loss breakdown by shape batch
    for meshname, losshistory in test_losses_dict.items():
        fig, ax = plt.subplots()
        for name, vals in losshistory.items():
            ax.plot(test_epochs, vals, label=name)
        ax.set_title(f"Test {meshname}: Loss Breakdown")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(opt.export_save_path, opt.name, f"test_{meshname}_loss.png"))
        plt.cla()
        plt.close()


