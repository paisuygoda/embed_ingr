# -*- coding: utf-8 -*-
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='parameters')
    # general
    parser.add_argument('-g', '--gpu', default=0, nargs='+', type=int)
    parser.add_argument('--seed', default=1234, type=int)

    return parser
