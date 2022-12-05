import argparse
import os
from util import util
import torch

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', required=True, help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--ninput_edges', type=int, default=None, help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples per epoch')
        self.parser.add_argument('--max_sample_size', type=int, default=float("inf"), help='Max samples while randomizing over LARGER SUPERSET')
        self.parser.add_argument('--overwritecache', action="store_true", help='whether to overwrite the cached mesh data. MUST DO IF CHANGE THE ANCHOR FACE SPECS')
        self.parser.add_argument('--overwritemeanstd', action="store_true", help='whether to overwrite the mean std cache data')
        self.parser.add_argument('--overwrite', action="store_true", help='whether to overwrite the experiment results')
        self.parser.add_argument('--overwriteanchorcache', action="store_true", help='whether to overwrite anchor cache')
        self.parser.add_argument('--overwriteopcache', action="store_true", help='whether to overwrite operator cache')
        self.parser.add_argument('--shuffle_topo', action="store_true", help='whether to shuffle vertex ordering of input data')
        self.parser.add_argument("--cachefolder", type=str, default="cache", help="name of folder to save/load base mesh cache")
        self.parser.add_argument("--anchorcachefolder", type=str, default="anchorcache", help="name of folder to save/load anchor conditional/extrinsic values")
        self.parser.add_argument("--operatorcachefolder", type=str, default="opcache", help="name of folder to save/load operator values")
        
        # network params
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--subset', nargs="+", default=[], help='list of meshnos to include')
        self.parser.add_argument('--exclude', nargs="+", default=[], help='list of meshnos to exclude')
        self.parser.add_argument("--clamp", type=str, choices=['sigmoid', 'tanh', 'softmax'], default='sigmoid', help="type of sigmoid to apply to segmentation output")
        self.parser.add_argument("--resconv", action="store_true", help="use residual mesh convolutions instead of regular")
        self.parser.add_argument("--leakyrelu", action="store_true", help="use leaky relu")
        self.parser.add_argument("--layernorm", action="store_true", help="use layer norm instead of instance norms")
        self.parser.add_argument('--arch', type=str, default='intseg', help='selects network to use') #todo add choices
        self.parser.add_argument('--weight_init', choices=["gaussian", "binary"], default='gaussian', help='how to initialize weights for conditioning')
        self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        self.parser.add_argument('--ncf', nargs='+', default=[16, 32, 32], type=int, help='conv filters')
        self.parser.add_argument('--emb_dim', type=int, default=128, help="size of output embedding dimension")
        self.parser.add_argument('--edgefeatures', nargs="+", default=[], choices={'dihedrals', 'symmetricoppositeangles', 'edgeratios'}, help="Input edge features")
        self.parser.add_argument('--extrinsic_features', nargs="+", default=[], help="Extrinsic features to concat prior to segmentation")
        self.parser.add_argument('--hks_t', nargs="+", type=float, default=None, help="time values to sample for hks feature")
        self.parser.add_argument('--extrinsic_condition_placement', choices=["pre", "post"], default=None, help='when to introduce extrinsic conditioning -- pre: meshcnn inputs, post: selection module inputs')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument("--drop_relu", action='store_true', help="drop relu layer in final mesh convolution")
        self.parser.add_argument("--selection_module", action='store_true', help="whether to pass meshcnn output through final MLP")
        self.parser.add_argument("--binary_conv", action='store_true', help="whether to add a final binary upconvolution")
        self.parser.add_argument("--softmax", action='store_true', help="whether to softmax channelwise the final output")
        self.parser.add_argument("--transfer_data_off", action='store_false', help="don't transfer prepool data to decoder layers")
        # Selection module parameters 
        self.parser.add_argument('--selectwidth', type=int, default=256, help='width of hidden features')
        self.parser.add_argument('--selectdepth', type=int, default=3, help='# of hidden layers of selection module')
        # Loss function weights 
        self.parser.add_argument("--gcsupervision_weight", type=float, default=1, help="weight on graphcuts loss")
        self.parser.add_argument("--gcsmoothness_weight", type=float, default=0.01, help="weight on graphcuts smoothness loss")
        # L2 loss settings 
        self.parser.add_argument("--loss", type=str, choices={'bce', 'ce'}, help="loss function to use", default="ce")
        self.parser.add_argument("--reweight_loss", action="store_true", help="whether to reweight the loss function for each batch")
        self.parser.add_argument("--supervised", action="store_true", help="whether to supervised with labels")
        self.parser.add_argument("--gcsupervision", action="store_true", help="l2 graphcuts")
        self.parser.add_argument("--gcparamloss", action="store_true", help="param loss for l2 graphcuts")
        # Other loss options
        self.parser.add_argument('--mixedtraining', action="store_true", help='whether training is with both synthetic labels/distortion only')
        self.parser.add_argument("--gcsmoothness", action="store_true", help="graphcuts smoothness loss")
        self.parser.add_argument("--contiguity_loss", action="store_true", help="whether to supervise with contiguity loss")
        self.parser.add_argument("--compactness_loss", action="store_true", help="whether to supervise with compactness loss")
        self.parser.add_argument("--anchor_loss", action="store_true", help="whether to supervise with anchor loss")
        self.parser.add_argument("--distortion_loss", type=str, choices={'count'}, help="distortion loss function to use", default=None) 
        self.parser.add_argument("--cut_param", action="store_true", help="whether to run cutting prior to parameterization (not necessary for LSCM to work)")   
        self.parser.add_argument("--softs2", action="store_true", help="set all weights lower than 0.5 to 0.5 in s2 loss to stabilize training")   
        self.parser.add_argument("--segboundary", type=str, choices={'neighbor'}, help="whether first stage segmentation should include soft boundary", default=None) 
        self.parser.add_argument("--segradius", type=int, default=1, help="# of neighbors to grow out soft boundary") 
        # Distortion loss settings 
        self.parser.add_argument("--distortion_metric", type=str, choices={'arap', 'conformal', 'singular', 'ss_isometric', 'ss_conformal'}, help="distortion metric to compute for entering into distortion loss function", default=None)    
        self.parser.add_argument("--delayed_distortion_epochs", type=int, default=None, help="how many epochs to delay distortion supervision")
        self.parser.add_argument("--solo_distortion", action="store_true", help="whether to only supervise with distortion after delayed entry")
        self.parser.add_argument("--lscmreg", action='store_true')
        self.parser.add_argument("--lscmregweight", type=float, default=1.)
        self.parser.add_argument("--lscmthreshold", type=float, default=0.001, help="counting loss lscm threshold")
        self.parser.add_argument("--arapthreshold", type=float, default=0.01, help="counting loss arap threshold")
        self.parser.add_argument("--step1paramloss", action="store_true", help="parameterization loss on the step 1")
        self.parser.add_argument("--step2paramloss", action="store_true", help="parameterization loss on the step 2")
        self.parser.add_argument("--floodfillparam", action="store_true", help="whether to first floodfill the parameterization weights")
        self.parser.add_argument("--clip_grad", type=float, default=None, help="max gradient norm")
        # data augmentation stuff
        self.parser.add_argument('--num_aug', type=int, default=0, help='# of augmentation files')
        self.parser.add_argument('--flipaug', type=float, default=0, help='percent of edges to randomly flip')
        self.parser.add_argument('--slideaug', type=float, default=0, help='percent of vertices to randomly slide')
        self.parser.add_argument('--vnormaug', action="store_true", help='whether to augment vertices along normal')
        self.parser.add_argument('--vaug', action="store_true", help='whether to augment vertices')
        self.parser.add_argument('--rotaug', action="store_true", help='whether to rotate augmentations')
        self.parser.add_argument('--testaug', action="store_true", help='whether to augment test shapes once')
        # general params
        self.parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--export_save_path', type=str, default="./outputs/training", help='path to save visualizations')
        self.parser.add_argument('--network_load_path', type=str, default=None, help='path to load models')
        self.parser.add_argument('--load_pretrain', action='store_true', help='load pretrained model from network load path')
        # visualization params
        self.parser.add_argument('--time', action="store_true", help='time every operator of network')
        self.parser.add_argument('--export_view_freq', default=0, type=int, help="export views with fresnel every n>0 epochs")
        self.parser.add_argument('--export_preds', action="store_true", help='export predictions only instead of views')
        self.parser.add_argument('--plot_preds', action="store_true", help='export views with fresnel (render on cluster)')
        self.parser.add_argument('--checkpoints_dir', type=str, default=None, help='checkpoints directory')
        self.parser.add_argument('--interactive', action='store_true', help='interactive mode')
        #
        self.initialized = True

    def parse(self, *args):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args(*args)
        self.opt.is_train = self.is_train   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        # Set default for distortion epochs 
        if self.opt.delayed_distortion_epochs is None: 
            self.opt.delayed_distortion_epochs = float('inf')
        
        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.export_save_path, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.export_save_path, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
