# -*- coding: utf-8 -*-
import torch.utils.data as data
from PIL import Image
import os
import pickle
import torch
import torchvision.transforms as transforms
from func import J2H
import MeCab
import numpy as np
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================


class RakutenData(data.Dataset):
    def __init__(self, img_path='/srv/datasets/Rakuten/', data_path='data/subdata/', partition=None,
                 mode=None, mismatch_rate=0.8):

        with open('data/subdata/ingr_id.p','rb') as f:
            self.ingr_id = pickle.load(f)

        with open('data/subdata/ontrogy_ingrcls.p','rb') as f:
            self.ontrogy = pickle.load(f)

        if data_path==None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition=partition

        with open(os.path.join(data_path,partition+'_images.p'),'rb') as f:
            self.ids = pickle.load(f)

        self.maxInst = 20
        with open(os.path.join(data_path,'dataset_dict.p'),'rb') as f:
            self.dataset_dict = pickle.load(f)
        with open(os.path.join(data_path,'recipe_class.p'),'rb') as f:
            self.recipe_class = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256), # we get only the center of that rescaled
            transforms.RandomCrop(224), # random crop within the center crop
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.imgPath = img_path
        self.mecab = MeCab.Tagger("-Ochasen")
        self.mode = mode
        self.mismatch_rate = mismatch_rate

    def __getitem__(self, index):
        recipe_id = self.ids[index][27:-4]
        validity = True

        # load image
        if self.partition == 'train':
            match = np.random.uniform() > self.mismatch_rate
        else:
            match = True

        if match:
            path = self.imgPath + self.ids[index]
        else:
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index
            path = self.imgPath + self.ids[rndindex]
            validity = False

        try:
            img = Image.open(path).convert('RGB')
            if img.size[0] < 224 or img.size[1] < 224:
                validity = False
        except:
            img = Image.new('RGB', (224, 224), 'white')
            validity = False

        if self.transform is not None:
            img = self.transform(img)

        # ingredients
        ingrs = []
        try:
            l = self.dataset_dict[recipe_id]['ingredients']
        except:
            l = []
        if len(l) is 0:
            l = ['*']
            validity = False
        for item in l:
            item = J2H(self.mecab, item)
            if item in self.ontrogy:
                try:
                    new_item = self.ontrogy[item]
                    ingrs.append(self.ingr_id[new_item])
                except:
                    validity = False
                    ingrs.append(0)
            else:
                ingrs.append(0)
                validity = False
        ingr_ln = len(ingrs)
        ingr_ln_tensor = torch.LongTensor(np.zeros(20))
        ingr_ln_tensor[ingr_ln] = 1

        if self.mode == "use":
            if len(ingrs) < 50:
                ingrs = ingrs + [0] * (50 - len(ingrs))
            ingrs = torch.LongTensor(ingrs)
        else:
            input_label = [0] * opts.numofingr
            for ingr in ingrs:
                input_label[ingr] = 1
            input_label[0] = 0
            ingrs = torch.FloatTensor(input_label)

        try:
            rec_class = self.recipe_class[self.dataset_dict[recipe_id]['dish_class']]
        except:
            rec_class = 0
            validity = False

        # output
        if self.partition == "all_valid":
            return img, ingrs, ingr_ln_tensor, rec_class, self.ids[index], validity
        else:
            if validity:
                target = 1
            else:
                target = -1
            return img, ingrs, ingr_ln_tensor, rec_class, recipe_id, target

    def __len__(self):
        return len(self.ids)
