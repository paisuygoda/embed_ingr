# -*- coding: utf-8 -*-
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import pickle
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)

def main():

    # gpus = ','.join(map(str, opts.gpu))
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,469)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load("model/ResNet50_469_best.pth.tar") # FoodLog-finetuned single-class food recognition model
    model.load_state_dict(checkpoint["state_dict"])
    modules = list(model.modules())[:-1]  # we do not use the last fc layer.
    model = nn.Sequential(*modules)
    model = torch.nn.DataParallel(model).cuda()
    print(model)

if __name__ == '__main__':
    main()