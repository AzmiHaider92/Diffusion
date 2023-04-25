import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--images_dir', help='files path')
    parser.add_argument('--images_dim', type=int, default=256, help='M , image is dxMxM')
    parser.add_argument('--images_channel', type=int, default=192, help='d , image is dxMxM')
    parser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=4, help='size of batch, duh')
    parser.add_argument('--checkpoints_dir', help='checkpoints to save path')
    parser.add_argument('--save_checkpoint_each', type=int, default=10000, help='save checkpoint each N iterations')
    parser.add_argument('--par_refresh', type=int, default=100, help='print loss')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()