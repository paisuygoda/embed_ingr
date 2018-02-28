# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.models as models
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from RakutenData import RakutenData
from args import get_parser
from model import MultilabelModel

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)


def main():

    gpus = ','.join(map(str, opts.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = MultilabelModel()

    best_val = float('inf')

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': opts.lr}])

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(RakutenData(partition='train', mismatch_rate=0.0), batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.workers)
    val_loader = torch.utils.data.DataLoader(RakutenData(partition='val'), batch_size=opts.batch_size, shuffle=True,
                                             num_workers=opts.workers)

    for epoch in range(opts.epoch):

        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = val(val_loader, model, criterion)

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            if is_best:
                filename = 'snapshots/multilabel_model_e{0:03d}_v{1:.3f}.pth.tar'.format(epoch + 1, best_val)
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

        output = model(img)

        # compute loss
        loss = criterion(output, ingr)
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

        output = model(img)

        # compute loss

        loss = criterion(output, ingr)
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