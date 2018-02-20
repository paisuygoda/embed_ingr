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
        modules = list(image_model.module.children())[:-1]  # remove last layer to get image feature
        image_model = nn.Sequential(*modules)
        image_model = torch.nn.DataParallel(image_model).cuda()
        self.image_model = image_model

    def forward(self, data):
        return norm(self.image_model(data))


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
        if is_from_datasetloader:
            ingr_list = []
            for i in range(len(ingr)):
                single_ingr = ingr[i].data.cpu().numpy().ravel()
                single_ingr_ln = int(ingr_ln[i].data.cpu().numpy()[0])
                ingr_list.append(single_ingr[:single_ingr_ln])
        else:
            pass  # temporal setting - might be changed
        """
        final_emb = np.zeros((len(ingr), 2048))
        for i, single_ingr in enumerate(ingr_list):
            input_label = np.zeros((1, opts.numofingr))
            for a in single_ingr:
                input_label[0][a] = 1.0
            input_label[0][0] = 0.0
            input_label = torch.autograd.Variable(torch.from_numpy(input_label).float()).cuda()
            emb = self.ingr_model(input_label)
            final_emb[i] = torch.cat((ingr_ln.float(), norm(emb)))
        """
        ingr_ln = ingr_ln.float().view(1,len(ingr))
        for i, single_ingr in enumerate(ingr_list):
            input_label = np.zeros((1, opts.numofingr))
            for a in single_ingr:
                input_label[0][a] = 1.0
            input_label[0][0] = 0.0
            input_label = torch.autograd.Variable(torch.from_numpy(input_label).float()).cuda()
            if i == 0:
                all_input = input_label
            else:
                all_input = torch.cat((all_input, input_label))
        emb = self.ingr_model(all_input)
        final_emb = torch.cat((ingr_ln.float(), norm(emb)))

        return final_emb


class im_ingr_embed(nn.Module):
    def __init__(self, resume=False):
        super(im_ingr_embed, self).__init__()

        self.image_model = im_embed()
        self.ingr_model = ingr_embed()

    def forward(self, img, ingr, ingr_ln, is_from_datasetloader=True):  # we need to check how the input is going to be provided to the model

        im_emb = self.image_model(img)
        ingr_emb = self.ingr_model(ingr, ingr_ln, is_from_datasetloader)

        output = [im_emb, ingr_emb]
        return output
