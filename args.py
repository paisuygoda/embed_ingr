# -*- coding: utf-8 -*-
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='parameters')
    # general
    parser.add_argument('-g', '--gpu', default="0", nargs='+')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=0.0004, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=30, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--updatefreq', default=2, type=int)
    parser.add_argument('--valfreq', default=10, type=int)
    parser.add_argument('--numofingr', default=304, type=int)

    return parser
