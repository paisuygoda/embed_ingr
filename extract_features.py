# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from model import im_ingr_embed
from RakutenData import RakutenData
from args import get_parser
import pickle

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


def main():
    gpus = ','.join(map(str, opts.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = im_ingr_embed(trainmode=False)
    criterion = nn.CosineEmbeddingLoss(0.1).cuda()

    test_loader = torch.utils.data.DataLoader(RakutenData(partition='test'), batch_size=opts.batch_size,
                                              shuffle=True, num_workers=opts.workers)
    checkpoint = torch.load(opts.model_path)
    model.load_state_dict(checkpoint["state_dict"])

    loss_counter = AvgCount()
    model.eval()
    for i, data in enumerate(test_loader):
        if len(data[0]) != opts.batch_size:
            break

        img = torch.autograd.Variable(data[0]).cuda()
        ingr = torch.autograd.Variable(data[1]).cuda()
        ingr_ln = torch.autograd.Variable(data[2]).cuda()
        target = torch.autograd.Variable(data[5].cuda(async=True))
        recipe_id = np.asarray(data[4])

        output = model(img, ingr, ingr_ln)

        # compute loss
        loss = criterion(output[0], output[1], target)
        # measure performance and record loss
        loss_counter.add(loss.data[0])

        if i==0:
            img_feature = output[0].data.cpu().numpy()
            ing_feature = output[1].data.cpu().numpy()
            recipe_id_list = recipe_id
        else:
            img_feature = np.concatenate((img_feature, output[0].data.cpu().numpy()),axis=0)
            ing_feature = np.concatenate((ing_feature, output[1].data.cpu().numpy()),axis=0)
            recipe_id_list = np.concatenate((recipe_id_list, recipe_id), axis=0)

    print("Test loss: ", loss_counter.avg)

    with open('results/img_feature.p', 'wb') as f:
        pickle.dump(img_feature, f)
    with open('results/ing_feature.p', 'wb') as f:
        pickle.dump(ing_feature, f)
    with open('results/recipe_id_list.p', 'wb') as f:
        pickle.dump(recipe_id_list, f)

    print("Saved img & recipe features.")

    for i in range(opts.numofingr):
        input_label = [0.0] * opts.numofingr
        input_label[i] = 1.0
        ingr = torch.autograd.Variable(torch.FloatTensor(input_label)).cuda()
        ingr_ln = torch.autograd.Variable(torch.FloatTensor(1)).cuda()
        emb = model.ingr_model(ingr, ingr_ln)

        if i == 0:
            ind_feature = emb.data.cpu().numpy()
        else:
            ind_feature = np.concatenate((ind_feature, emb.data.cpu().numpy()),axis=0)

    with open('results/individual_ing_feature.p', 'wb') as f:
        pickle.dump(ind_feature, f)
    print("Saved individual ingredient features.")


class AvgCount(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def switch_optim_lr(optimizer, opts):
    if optimizer.param_groups[0]['lr'] is 0.0:
        optimizer.param_groups[0]['lr'] = opts.lr
        optimizer.param_groups[1]['lr'] = 0.0
    else:
        optimizer.param_groups[1]['lr'] = opts.lr
        optimizer.param_groups[0]['lr'] = 0.0


if __name__ == '__main__':
    main()
