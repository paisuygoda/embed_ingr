import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import numpy as np
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class im_embed(nn.Module):
    def __init__(self, trainmode):
        super(im_embed, self).__init__()

        image_model = models.resnet50()
        image_model.fc = nn.Linear(2048, 469)
        image_model = torch.nn.DataParallel(image_model).cuda()

        if trainmode:
            checkpoint = torch.load(
                "model/ResNet50_469_best.pth.tar")  # FoodLog-finetuned single-class food recognition model
            image_model.load_state_dict(checkpoint["state_dict"])
        modules = list(image_model.module.children())[:-1]  # remove last layer to get image feature
        image_model = nn.Sequential(*modules)
        image_model = torch.nn.DataParallel(image_model).cuda()
        self.image_model = image_model

    def forward(self, data):
        return self.image_model(data).view(opts.batch_size, 2048)


class ingr_embed(nn.Module):
    def __init__(self):
        super(ingr_embed, self).__init__()
        ingr_model = nn.Sequential(
            nn.Linear(opts.numofingr, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2047),
        )
        ingr_model = torch.nn.DataParallel(ingr_model).cuda()
        self.ingr_model = ingr_model

    def forward(self, ingr, ingr_ln, is_from_datasetloader=True):
        ingr_ln = ingr_ln.float().cuda().view(len(ingr), 1)
        for i, single_ingr in enumerate(ingr):
            if i == 0:
                emb = norm(self.ingr_model(single_ingr.view(1, opts.numofingr)))
            else:
                emb = torch.cat((emb, norm(self.ingr_model(single_ingr.view(1, opts.numofingr)))))
        final_emb = torch.cat((ingr_ln.float(), emb), dim=1)
        return final_emb


class im_ingr_embed(nn.Module):
    def __init__(self, trainmode=True):
        super(im_ingr_embed, self).__init__()

        self.image_model = im_embed(trainmode)
        self.ingr_model = ingr_embed()

    def forward(self, img, ingr, ingr_ln, is_from_datasetloader=True):  # we need to check how the input is going to be provided to the model

        im_emb = self.image_model(img)
        ingr_emb = self.ingr_model(ingr, ingr_ln, is_from_datasetloader)

        output = [im_emb, ingr_emb]
        return output


class MultilabelModel(nn.Module):
    def __init__(self, trainmode=True):
        super(MultilabelModel, self).__init__()

        image_model = models.resnet50()
        image_model.fc = nn.Linear(2048, 469)
        image_model = torch.nn.DataParallel(image_model).cuda()
        checkpoint = torch.load(
            "model/ResNet50_469_best.pth.tar")  # FoodLog-finetuned single-class food recognition model
        image_model.load_state_dict(checkpoint["state_dict"])

        image_model.module.fc = nn.Linear(2048, opts.numofingr)
        self.image_model = image_model.cuda()

    def forward(self, data):
        return self.image_model(data)
