from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--train_file', type=str, default="", help='training mesh for unsupervised training')
        self.parser.add_argument('--phase', type=str, default="train")
        self.parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=300, help='frequency of saving the latest results')
        self.parser.add_argument("--savetraindata", action="store_true", help="whether to save training tensors")
        self.parser.add_argument("--clip", action="store_true", help="toggle to use CLIP perceptual distortion during training")
        self.parser.add_argument("--threshold_patience", type=int, help="loss plateau patience before reducing threshold", default=10)
        self.parser.add_argument("--loss_threshold", type=float, help="minimum loss reduction amount for each mesh", default=0.1)
        self.parser.add_argument("--max_grad", type=float, help="clip gradients value", default=None)
        self.parser.add_argument("--threshold_steps", type=int, help="# of even steps to take when reducing threshold", default=5)
        self.parser.add_argument("--threshold_quantile", type=float,  help="minimum quantile of threshold to trigger decrease", default=0.85)
        self.parser.add_argument("--adaptive_threshold", type=float,  help="default quantile to set new threshold to", default=0.5)
        self.parser.add_argument("--final_quantile", type=float,  help="minimum proportion of faces to clear final threshold", default=0.98)
        self.parser.add_argument("--texturedir", type=str, default="./datasets/textures", help="path to texture image files with additional subfolder for clip resolution image file")
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_embed_freq', type=int, default=5, help='frequency of exporting embeddings at the end of epochs')
        self.parser.add_argument('--save_int_freq', type=int, default=5,
                                 help='frequency of exporting intermediate deep features by epoch')
        self.parser.add_argument('--run_test_freq', type=int, default=1,
                                 help='frequency of running test in training script')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--finetune_mlp', action='store_true', help='whether to finetue the pretrained model MLP')
        self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--vectoradam', action='store_true', help='use vector adam optimization')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default=None, help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument(
            "--norm_batch", action='store_true', help="normalize across meshes during batch training (deals with case of super big mesh + small mesh in same batch)")
        # Triplet mining and shape descriptors 
        self.parser.add_argument(
            "--triplet_mine_type", type=str, default="all", choices=['all', 'mesh'], help="which geometric level to triplet mine over")
        self.parser.add_argument(
            "--descrip_type", type=str, default="face", choices=['vertex', 'edge', 'face'], help="which geometric feature to compute descriptors over (vertex, edge, face)")
        self.parser.add_argument(
            "--training_mode", type=str, default="clusters", choices=['clusters', 'distances'], help="triplet mining mode: over 'clusters' or 'distances'")
        self.parser.add_argument(
            "--k_min", type=int, default=2, help="min number of clusters for kmeans-based cluster supervision (only relevant if randomizing k)")
        self.parser.add_argument(
            "--k_max", type=int, default=3, help="max number of clusters for kmeans-based cluster supervision (k defaults to this value if not randomizing)")
        self.parser.add_argument(
            "--randomize_k", type=bool, default=False, help="whether to randomize cluster selection from 2-k every triplet update")
        self.parser.add_argument(
            "--n_seeds", type=int, default=100, help="number of cluster seeds (samples) to evaluate to estimate final cluster probs")
        self.parser.add_argument(
            "--descriptors", type=str, nargs="+", default=['gc', 'agd', 'sdf'], help="list of per-mesh metrics to compute for triplet mining")
        self.parser.add_argument(
            "--normalize_descrip", type=bool, default=True, help="whether to normalize descriptors for training")
        self.parser.add_argument(
            "--randomize_descrip", type=bool, default=False, help="whether to randomize descriptors")
        self.parser.add_argument(
            "--descrip_update_freq", type=int, default=5, help="frequency of descriptor randomization")
        self.parser.add_argument(
            "--descrip_dim", type=int, default=3, help="number of metrics to sample for shape descriptor generation (only relevant if randomizing)")
        self.parser.add_argument(
            "--lscm_rad", type=float, nargs="+", default=[0.2], help="radii of charts to compute LSCM over")
        self.parser.add_argument(
            "--lscm_bdry_samples", type=int, default=0, help="number of boundary samples to average LSCM over")
        self.parser.add_argument(
            "--sdf_samples", type=int, default=30, help="number of samples to estimate shape diameter function")
        self.parser.add_argument(
            "--triplet_update_freq", type=int, default=1, help="frequency of triplet updating")
        # visualization
        self.parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
        self.parser.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')
        self.is_train = True
