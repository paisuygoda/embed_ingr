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
    def __init__(self):
        super(im_embed, self).__init__()

        image_model = models.resnet50()
        image_model.fc = nn.Linear(2048, 469)
        image_model = torch.nn.DataParallel(image_model).cuda()

        checkpoint = torch.load(
            "model/ResNet50_469_best.pth.tar")  # FoodLog-finetuned single-class food recognition model
        image_model.load_state_dict(checkpoint["state_dict"])
        modules = list(image_model.modules())[:-1]  # remove last layer to get image feature
        image_model = nn.Sequential(*modules)
        image_model = torch.nn.DataParallel(image_model).cuda()
        self.image_model = image_model

    def forward(self, data):
        return norm(self.image_model(data[0]))


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

    def forward(self, data, is_from_datasetloader=True):
        if is_from_datasetloader:
            ingrs = data[1].numpy().ravel()
            ingr_ln = int(data[2].numpy()[0])
            ingrs = ingrs[:ingr_ln]
        else:
            ingrs = data[1] # temporal setting - might be changed

        final_emb = np.zeros((1, 2047))
        for ingr in ingrs:
            input_label = np.zeros((1, opts.numofingr))
            input_label[0][ingr] = 1.0
            emb = self.ingr_model(input_label)
            final_emb += norm(emb)

        return torch.cat((data[2], final_emb))


class im_ingr_embed(nn.Module):
    def __init__(self, resume=False):
        super(im_ingr_embed, self).__init__()

        self.image_model = im_embed()
        self.ingr_model = ingr_embed()

    def forward(self, data, is_from_datasetloader=True):  # we need to check how the input is going to be provided to the model

        im_emb = self.image_model(data)
        ingr_emb = self.ingr_model(data, is_from_datasetloader)

        output = [im_emb, ingr_emb]
        return output
