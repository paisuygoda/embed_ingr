# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import pickle
from RakutenData import RakutenData
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
    model = im_ingr_embed()
    model = models.resnet50()
    model.fc = nn.Linear(2048,469)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load("model/ResNet50_469_best.pth.tar") # FoodLog-finetuned single-class food recognition model
    model.load_state_dict(checkpoint["state_dict"])
    modules = list(model.modules())[:-1]  # remove last layer to get image feature
    model = nn.Sequential(*modules)
    model = torch.nn.DataParallel(model).cuda()

    valtrack = opts.updatefreq
    criterion = nn.CosineEmbeddingLoss(0.1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    cudnn.benchmark = True

    updater = RakutenData(partition='train')
    train_loader = torch.utils.data.DataLoader(RakutenData(partition='train'), batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.workers)
    val_loader = torch.utils.data.DataLoader(RakutenData(partition='val'), batch_size=opts.batch_size, shuffle=True,
                                             num_workers=opts.workers)

    for epoch in range(opts.epoch):

        if valtrack == opts.updatefreq:
            valtrack = 0
            update_ingr_embed(updater, model, opts.numofingr)

        train(train_loader, model, criterion, optimizer, epoch)

        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = val(val_loader, model, criterion)

            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0
            if valtrack >= opts.updatefreq:
                opts.lr *= 0.96
                valtrack = 0

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            if is_best:
                filename = 'snapshots/model_e{0:03d}_v{1:.3f}.pth.tar'.format(epoch + 1, best_val)
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_val': best_val,
                            'optimizer': optimizer.state_dict(), 'valtrack': valtrack}, filename)

            print('----- Validation: {.3f} -----'.format(best_val))
            print('----- Best Score: {.3f} (best) -----'.format(best_val))


def update_ingr_embed(dataset, model, numofingr):
    # 食材特徴もdeepで求めたい
    known_list = []
    entry_count = 0
    matrix_not_full = True
    A = np.array([[]])
    b = np.array([])

    while matrix_not_full:
        for i in range(dataset.__len__):
            (img, ingrs, ingr_ln, _, _, _) = dataset[i]
            ingrs = ingrs[:ingr_ln]

            output = model(img).numpy()

            if (i+1 in ingrs) and (sorted(ingrs) not in known_list):
                known_list.append(sorted(ingrs))
                ingr_mat = np.zeros((1,numofingr))
                for ingr in ingrs:
                    ingr_mat[ingr] = 1.0
                ingr_mat[0] = 0.0
                A = np.concatenate((A, ingr_mat))
                b = np.concatenate((b, output), axis=1)
                entry_count += 1

                if entry_count == numofingr:
                    matrix_not_full = False
                    break


def train(train_loader, model, criterion, optimizer, epoch):
    loss = AvgCount()
    model.train()
    for data in train_loader:

        output = model(data)

        # compute loss
        if opts.semantic_reg:
            cos_loss = criterion[0](output[0], output[1], target_var[0])
            img_loss = criterion[1](output[2], target_var[1])
            rec_loss = criterion[1](output[3], target_var[2])
            # combined loss
            loss =  opts.cos_weight * cos_loss +\
                    opts.cls_weight * img_loss +\
                    opts.cls_weight * rec_loss

            # measure performance and record losses
            cos_losses.update(cos_loss.data, input[0].size(0))
            img_losses.update(img_loss.data, input[0].size(0))
            rec_losses.update(rec_loss.data, input[0].size(0))
        else:
            loss = criterion(output[0], output[1], target_var[0])
            # measure performance and record loss
            cos_losses.update(loss.data[0], input[0].size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if opts.semantic_reg:
        print('Epoch: {0}\t'
                  'cos loss {cos_loss.val[0]:.4f} ({cos_loss.avg[0]:.4f})\t'
                  'img Loss {img_loss.val[0]:.4f} ({img_loss.avg[0]:.4f})\t'
                  'rec loss {rec_loss.val[0]:.4f} ({rec_loss.avg[0]:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, cos_loss=cos_losses, img_loss=img_losses,
                   rec_loss=rec_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))
    else:
         print('Epoch: {0}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))


class AvgCount(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def count(self, val):
        self.sum += val * 1
        self.count += 1
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()