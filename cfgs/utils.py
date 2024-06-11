import argparse


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reset_model_args(args1, args2):
    for arg in vars(args1):
        setattr(args2, arg, getattr(args1, arg))
