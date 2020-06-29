import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_resblocks', type=int, default=20)
    parser.add_argument('--n_resgroups', type=int, default=10)
    parser.add_argument('--reduction',   type=int, default=16)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--rgb_range', type=float, default=1.)
    parser.add_argument('--n_colors', type=int, default=3)
    
    parser.add_argument('--im_crop_H', type=int, default=128)
    parser.add_argument('--im_crop_W', type=int, default=128)
    parser.add_argument('--random_crop', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=500)
    parser.add_argument('--lr_decay', type=int, default=0.618)
    parser.add_argument('--LAMBDA', type=float, default=10)
    parser.add_argument('--d_step', type=int, default=10)
    parser.add_argument('--mask_prop', type=int, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--freq_visual', type=int, default=100)
    parser.add_argument('--epoch_to_load', type=str, default='1000')
    parser.add_argument('--model_name', type=str, default='FlowSRNet')

    return parser