import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--images_dir', help='files path')
    parser.add_argument('--images_dim', type=int, default=256, help='M , image is dxMxM')
    parser.add_argument('--images_channel', type=int, default=192, help='d , image is dxMxM')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='size of batch, duh')
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()