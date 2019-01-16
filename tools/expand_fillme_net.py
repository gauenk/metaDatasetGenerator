#!/usr/bin/env python
import _init_paths
from core.config import replaceFillmeNetwork
import argparse,sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--prototxt', dest='prototxt',
                        help='Prototxt File',type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    replaceFillmeNetwork(args.prototxt)
