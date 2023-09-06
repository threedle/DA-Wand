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
import wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.phase = 'train'

    if not opt.overwrite and os.path.exists(os.path.join(opt.export_save_path, opt.name, "test_prauc_batch.png")) and not opt.continue_train:
        print(f"Already done with {opt.name}.")
        exit()
    if opt.overwrite and not opt.continue_train:
        if os.path.exists(os.path.join(opt.export_save_path, opt.name)):
            clear_directory(os.path.join(opt.export_save_path, opt.name))
    # Also clear cache directories
    if opt.overwritecache:
        if os.path.exists(os.path.join(opt.dataroot, opt.phase, opt.cachefolder)):
            clear_directory(os.path.join(opt.dataroot, opt.phase, opt.cachefolder))
    if opt.overwriteanchorcache:
        if os.path.exists(os.path.join(opt.dataroot, opt.phase, opt.anchorcachefolder)):
            clear_directory(os.path.join(opt.dataroot, opt.phase, opt.anchorcachefolder))
    if opt.overwritemeanstd:
        meanstd_path = os.path.join(opt.dataroot, opt.phase, f'mean_std_{opt.name}_cache.p')
        if os.path.exists(meanstd_path):
            os.remove(meanstd_path)

    ### Wandb
    c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
    c.cleanup(int(1e9))

    id = None
    if opt.continue_train:
        import re
        if os.path.exists(os.path.join(opt.export_save_path, opt.name, 'wandb', 'latest-run')):
            for idfile in os.listdir(os.path.join(opt.export_save_path, opt.name, 'wandb', 'latest-run')):
                if idfile.endswith(".wandb"):
                    result = re.search(r'run-([a-zA-Z0-9]+)', idfile)
                    if result is not None:
                        id = result.group(1)
                        break
        else:
            print(f"Warning: No wandb record found in {os.path.join(opt.export_save_path, opt.name, 'wandb', 'latest-run')}!. Starting log from scratch...")

    wandb.login()
    run = wandb.init(project='dawand', name=opt.name, dir=os.path.join(opt.export_save_path, opt.name),
                     mode= "offline" if opt.debug else "online", id = id)
    wandb.define_metric("epoch")
    wandb.define_metric("batch")

    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

    # Validation
    wandb.define_metric("val_isoperc", step_metric="batch")
    wandb.define_metric("val_isometric", step_metric="batch")
    wandb.define_metric("val_n_faces", step_metric="batch")

    # Define metrics for the losses
    if opt.supervised:
        wandb.define_metric("train_supervision_loss", step_metric="batch")

    if opt.gcsupervision:
        wandb.define_metric("train_gc_loss", step_metric="batch")

    if opt.delayed_distortion_epochs:
        wandb.define_metric("train_s2_arap_distortion", step_metric="batch")
        wandb.define_metric("train_s2_distortion_loss", step_metric="batch")

    if opt.anchor_loss:
        wandb.define_metric("train_anchor_loss", step_metric="batch")

    if opt.gcsmoothness:
        wandb.define_metric("train_gcsmoothness", step_metric="batch")

    ### Dataloaders
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

        if testopt.overwritecache:
            # Also clear cache directories
            if os.path.exists(os.path.join(testopt.dataroot, testopt.phase, testopt.cachefolder)):
                clear_directory(os.path.join(testopt.dataroot, testopt.phase, testopt.cachefolder))
        if testopt.overwriteanchorcache:
            if os.path.exists(os.path.join(testopt.dataroot, testopt.phase, testopt.anchorcachefolder)):
                clear_directory(os.path.join(testopt.dataroot, testopt.phase, testopt.anchorcachefolder))
        if testopt.overwritemeanstd:
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

    # If not continuing training, and need to initialize model
    if not opt.continue_train and opt.initialize:
        dataset.dataset.init_data(opt.initialize)
        # Initialize network to predict just the anchor selection
        print(f"Initializing network with {opt.initialize}...")
        count = 0
        done = False
        while not done:
            done = False
            for i, data in enumerate(dataset):
                # Edge case: if whole batch is invalid then data is None
                if data is None:
                    print(f"Warning: batch {i} skipped because no valid samples.")
                success = model.set_input(data)
                if not success:
                    continue
                else:
                    loss, loss_dict, preds, labels = model.optimize_supervised()

                # Initialization is complete once all the rounded values match the labels
                done = done and torch.all(torch.round(preds) == labels)
            count += 1
        # Reset the labels
        dataset.dataset.precomputed_labels = None
        print(f"Initialization done in {count} epochs.")

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        if epoch == opt.delayed_distortion_epochs:
            print(f"Epoch {epoch}: introducing distortion supervision")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = 0
        epoch_loss_dict = defaultdict(list)

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

            # Average loss values over epochs
            for lossname, val in tmp_losses_dict.items():
                epoch_loss_dict[f"train_{lossname}"].append(val)

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

        # Log losses
        epoch_end_losses = {"train_loss": epoch_loss}
        for key, losslist in epoch_loss_dict.items():
            epoch_end_losses[key] = np.mean(losslist)
        wandb.log(epoch_end_losses)

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

                    wandb.run.summary["best_test_loss"] = avg_loss
                    print(f"Saved best network at epoch {epoch}")

            with open(os.path.join(cachedir, "losses.pkl"), 'wb') as f:
                pickle.dump((avg_losses, test_losses, test_epochs, best_test_stats), f)

            isometric_i = None
            isoperc_i = None
            for i in range(len(metricnames)):
                if metricnames[i] == "Isometric":
                    isometric_i = i
                if metricnames[i] == '% I < 0.05':
                    isoperc_i = i

            val_dict = {'val_loss': avg_loss}
            if isometric_i:
                val_dict['val_isometric'] = avgstats[isometric_i]
            if isoperc_i:
                val_dict['val_isoperc'] = avgstats[isoperc_i]
            wandb.log(val_dict)

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
