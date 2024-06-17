import argparse
from cfgs.utils import str2bool


def parse_pugan_o_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    # training
    parser.add_argument('--seed', default=10, type=float, help='seed')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers number')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=5, type=int, help='model save frequency')
    parser.add_argument('--use_smooth_loss', default=False, type=str2bool, help='whether use smooth L1 loss')
    parser.add_argument('--beta', default=0.01, type=float, help='beta for smooth L1 loss')

    # dataset
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="./data/PU-GAN/train/PUGAN_poisson_256_poisson_1024.h5", type=str, help='the path of train dataset')
    parser.add_argument('--num_points', default=256, type=int, help='the points number of each input patch')
    parser.add_argument('--skip_rate', default=1, type=int, help='used for dataset')
    parser.add_argument('--use_random_input', default=True, type=str2bool, help='whether use random sampling for input generation')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    parser.add_argument('--data_augmentation', default=True, type=str2bool, help='whether use data augmentation')
    
    # encoder
    parser.add_argument('--k', default=16, type=int, help='neighbor number in encoder')
    parser.add_argument('--encoder_dim', default=64, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--out_dim', default=128, type=int, help='input(output) feature dimension in each dense block')
    parser.add_argument('--encoder_bn', default=True, type=str2bool, help='whether use batch normalization in encoder')
    #  kernel points
    # kernel_radius = kernel_point_receptive_radius * 1.5
    # conv_radius  = kernel_point_receptive_radius * 4
    parser.add_argument('--conv_radius', default=0.8, type=float, help='radius of kernel point convolution')
    parser.add_argument('--neighbor_limits', default=30, type=int, help='maximum number of points')
    parser.add_argument('--kernel_radius', default=0.3, type=float, help='radius of kernel point sphere')
    parser.add_argument('--kernel_point_receptive_radius', default=0.2, type=float, help='receptive field of kernel point convolution')
    parser.add_argument('--num_kernel_points', default=15, type=int, help='number of kernel points in kernel point representation')
    parser.add_argument('--in_dim', default=128, type=int, help='input feature dimension of REM & KGM')
    parser.add_argument('--kp_dim', default=128, type=int, help='feature dimension in REM & KGM')
    parser.add_argument('--is_kp_bn', default=True, type=str2bool, help='whether use batch normalization in kpconv')
    parser.add_argument('--is_kp_bias', default=False, type=str2bool, help='whether use bias in kpconv')
    parser.add_argument('--rigid_scale', default=0.625, type=float, help='radius of kernel point sphere')
    parser.add_argument('--query_scale', default=1.0, type=float, help='radius of kernel point sphere')
    parser.add_argument('--up_rate', default=4, type=int, help='number of kernel points in kernel point queries')
    # cross-attention
    parser.add_argument('--head_num', default=4, type=int, help='head number of attention')
    parser.add_argument('--trans_num', default=2, type=int, help='number of attention blocks')
    parser.add_argument('--trans_dim', default=128, type=int, help='dim of attention')
    parser.add_argument('--is_attn_bn', default=False, type=str2bool, help='whether use batch normalization in attention')
    
    
    
    # ouput
    parser.add_argument('--out_path', default='./output/pugan_o', type=str, help='the checkpoint and log save path')
    
    # test
    parser.add_argument('--patch_rate', default=3, type=int, help='used for patch generation')
    parser.add_argument('--r', default=4, type=int, help='upsampling rate')
    parser.add_argument('--o', action='store_true', help='using original model')
    parser.add_argument('--flexible', action='store_true', help='aribitrary scale?')
    parser.add_argument('--input_dir', default='./data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/', type=str, help='path to folder of input point clouds')
    parser.add_argument('--gt_dir', default='./data/PU-GAN/test_pointcloud/input_2048_4X/gt_8192/', type=str, help='path to folder of gt point clouds')
    parser.add_argument('--save_dir', default='./result', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='./pretrain/pugan_best_o.pth', type=str, help='checkpoints')
    
    args = parser.parse_args()

    return args
