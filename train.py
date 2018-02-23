# -*- coding: utf-8 -*-
import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from model import im_ingr_embed
from RakutenData import RakutenData
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)

def main():

    gpus = ','.join(map(str, opts.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = im_ingr_embed()

    best_val = float('inf')
    temp_best_val = float('inf')
    tempfilename = 'snapshots/model_temp.pth.tar'

    criterion = nn.CosineEmbeddingLoss(0.1).cuda()
    optimizer = torch.optim.Adam([
        {'params': model.ingr_model.parameters(), 'lr': opts.lr},
        {'params': model.image_model.parameters(), 'lr': 0.0}])

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(RakutenData(partition='train'), batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.workers)
    val_loader = torch.utils.data.DataLoader(RakutenData(partition='val'), batch_size=opts.batch_size, shuffle=True,
                                             num_workers=opts.workers)

    for epoch in range(opts.epoch):

        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = val(val_loader, model, criterion)

            if val_loss <= temp_best_val:
                temp_best_val = val_loss
                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, tempfilename)
            else:
                switch_optim_lr(optimizer, opts)
                temp_best_val = float('inf')
                print("switched learning model.. ingr lr: {0}, "
                      "img lr: {1}".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
                checkpoint = torch.load(tempfilename)
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            if is_best:
                filename = 'snapshots/model_e{0:03d}_v{1:.3f}.pth.tar'.format(epoch + 1, best_val)
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_val': best_val,
                            'optimizer': optimizer.state_dict()}, filename)

            print('----- Validation: {:.3f} -----'.format(val_loss))
            print('----- Best Score: {:.3f} (best) -----'.format(best_val))


def train(train_loader, model, criterion, optimizer, epoch):
    loss_counter = AvgCount()
    model.train()
    for i, data in enumerate(train_loader):
        if len(data[0]) != opts.batch_size:
            break
        sys.stdout.write("\r%d" % i)

        img = torch.autograd.Variable(data[0]).cuda()
        ingr = torch.autograd.Variable(data[1]).cuda()
        ingr_ln = torch.autograd.Variable(data[2]).cuda()
        target = torch.autograd.Variable(data[5].cuda(async=True))

        output = model(img, ingr, ingr_ln)

        # compute loss

        loss = criterion(output[0], output[1], target)
        # measure performance and record loss
        loss_counter.add(loss.data[0])

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sys.stdout.write("\r")
    print('Epoch: {0}\tLoss {1:.4f}'.format(epoch, loss_counter.avg))


def val(val_loader, model, criterion):
    loss_counter = AvgCount()
    model.eval()
    for data in val_loader:
        if len(data[0]) != opts.batch_size:
            break
            
        img = torch.autograd.Variable(data[0]).cuda()
        ingr = torch.autograd.Variable(data[1]).cuda()
        ingr_ln = torch.autograd.Variable(data[2]).cuda()
        target = torch.autograd.Variable(data[5].cuda(async=True))

        output = model(img, ingr, ingr_ln)

        # compute loss

        loss = criterion(output[0], output[1], target)
        # measure performance and record loss
        loss_counter.add(loss.data[0])

    return loss_counter.avg


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
