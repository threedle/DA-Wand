from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--test_dir', type=str, required=True, help='directory with test meshes')
        self.parser.add_argument('--phase', type=str, default="test")
        self.parser.add_argument('--network_save_path', type=str, default="", help='directory with saved network')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--test_file', type=str, help='test file to run inference over')
        self.is_train = False
