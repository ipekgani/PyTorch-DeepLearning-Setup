from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--max_epochs', type=int, default=20, help='max epoch')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        # training parameters
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--optimizer', default='SGD', type=str, help='Which optimizer to use (SGD | adam | rmsprop)')
        parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
        parser.add_argument('--lr_policy', type=str, default='multi', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--weight_decay', type=float, default=1e-3)
        parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help='Learning rate decay factor')
        parser.add_argument('--lr_scheduler_milestones', default='25,50', type=str, help='Learning rate scheduling, when to multiply learning rate by gamma')
        parser.add_argument('--lr_plateau_factor', default='0.1', type=float, help='Decay learning rate on plateauing of the validation loss (set -1 to disable)')
        parser.add_argument('--lr_plateau_patience', default='10', type=int, help='Decay learning rate on plateauing of the validation loss (set -1 to disable)')

        # Validation
        parser.add_argument('--validation', action='store_true', help='whether to do valdiation')
        parser.add_argument('--validation_datamode', type=str, default='template')
        parser.add_argument('--validation_collate', type=str, default='default')
        parser.add_argument('--valid_root', type=str, default='')

        self.isTrain = True
        return parser
