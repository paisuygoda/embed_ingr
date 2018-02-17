import torch
import torch.nn as nn
import torch.nn.parallel
import torch.legacy as legacy
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


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

    def forward(self, image):
        return self.image_model(image)


class ingr_embed(nn.Module):
    def __init__(self):
        self.ingr_model = nn.Sequential(
            nn.Linear(opts.numofingr, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2048),
        )

    def forward(self, ingrs):


class im_ingr_embed(nn.Module):
    def __init__(self, resume):
        super(im_ingr_embed, self).__init__()

        if resume:
            image_model = models.resnet50()
            image_model.fc = nn.Linear(2048, 469)
            modules = list(image_model.modules())[:-1]  # remove last layer to get image feature
            image_model = nn.Sequential(*modules)
            image_model = torch.nn.DataParallel(image_model).cuda()
            self.image_model = image_model

        else:
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

            self.ingr_model = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.Tanh(),
            )

        self.ingRNN_ = ingRNN()
        self.table = TableModule()

    def forward(self, x, y1, y2, z1, z2):  # we need to check how the input is going to be provided to the model
        # recipe embedding
        recipe_emb = self.table([self.stRNN_(y1, y2), self.ingRNN_(z1, z2)], 1)  # joining on the last dim
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        if opts.semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output
            output = [visual_emb, recipe_emb]
        return output

        # Tweaked im2recipe model for ingredient retrieval


class im2ingr(nn.Module):
    def __init__(self):
        super(im2ingr, self).__init__()
        if opts.preModel == 'resNet50':

            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.Tanh(),
            )

            self.recipe_embedding = nn.Sequential(
                nn.Linear(opts.irnnDim * 2, opts.embDim),
                nn.Tanh(),
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.ingRNN_ = ingRNN()
        self.table = TableModule()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, z1, z2):  # we need to check how the input is going to be provided to the model
        # recipe embedding
        recipe_emb = self.recipe_embedding(self.ingRNN_(z1, z2))
        recipe_emb = norm(recipe_emb)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        if opts.semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output
            output = [visual_emb, recipe_emb]
        return output
