mport argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='CREStereo', help="name your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

parser.add_argument('--training_data_path', default='E:/yxz/Deep_learning/RAFT/RAFT-Stereo-main/SceneFlow',
                    help='use mixed precision')
parser.add_argument('--log_dir', default='./train_log_cres_loss_b4_i50000', help='use mixed precision')
parser.add_argument('--save_txt', default='FlyingThings3D (TEST).txt', help='use mixed precision')
parser.add_argument('--log_level', default='logging.INFO', help='use mixed precision')
parser.add_argument('--seed', type=int, default=0, help="batch size used during training.")
# parser.add_argument('--loadmodel', default='~', help='use mixed precision')


# Training parameters
# parser.add_argument('--mixed_precision', default='false', help="-")
parser.add_argument('--base_lr', type=float, default=4.0e-4, help="batch size used during training.")
parser.add_argument('--nr_gpus', type=int, default=1, help="batch size used during training.")
parser.add_argument('--batch_size_single', type=int, default=4, help="batch size used during training.")
parser.add_argument('--n_total_epoch', type=int, default=250, help="batch size used during training.")
parser.add_argument('--minibatch_per_epoch', type=int, default=200, help="batch size used during training.")
parser.add_argument('--model_save_freq_epoch', type=int, default=10, help="batch size used during training.")
parser.add_argument('--max_disp', type=int, default=256, help="batch size used during training.")
parser.add_argument('--image_width', type=int, default=512, help="batch size used during training.")
parser.add_argument('--image_height', type=int, default=384, help="batch size used during training.")

# parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512],
                    help="size of the random image crops used during training.")
parser.add_argument('--train_iters', type=int, default=16,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

# Validation parameters
parser.add_argument('--valid_iters', type=int, default=32,
                    help='number of flow-field updates during validation forward pass')

# Architecure choices
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                    help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true',
                    help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
# parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
# parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
#                     help="hidden state and context dimensions")

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                    help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
args = parser.parse_args()