import torch.utils.data as data
from PIL import Image
import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        # print path
        return Image.new('RGB', (224,224), 'white')

def resize(img):
    w,h = img.size
    if w<h:
        ratio = float(h)/float(w)
        img = img.resize((256,int(256*ratio)))
    else:
        ratio = float(w)/float(h)
        img = img.resize((int(256*ratio),256))

    return img

class RakutenData(data.Dataset):
    def __init__(self, img_path='data/images/', data_path='data', partition=None):

        with open('data/ingr_id.p','rb') as f:
            self.ingr_id = pickle.load(f)

        with open('data/ontrogy_ingrcls.p','rb') as f:
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
        with open(os.path.join(data_path,'ingredients_dict.p'),'rb') as f:
            self.ingr_dic = pickle.load(f)
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

    def __getitem__(self, index):
        recipeId = self.ids[index]
        path = self.imgPath + recipeId + '.jpg'

        # ingredients
        ingrs = []
        try:
            l = self.ingr_dic[recipeId]['ingr']
        except:
            l = []
        if len(l) is 0:
            l = ['*']
            target = -1
        for item in l:
            if item in self.ingr_id:
                try:
                    new_item = self.ontrogy[item]
                    ingrs.append(self.ingr_id[new_item])
                except:
                    ingrs.append(0)
            else:
                ingrs.append(0)
        igr_ln = len(ingrs)
        if len(ingrs) < 50:
            ingrs = ingrs+[0]*(50-len(ingrs))
        # ingrs = torch.LongTensor(ingrs)

        # load image
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        rec_class = self.recipe_class[self.ingr_dic[recipeId]['class']]

        img_id = recipeId

        # output
        return img, ingrs, igr_ln, rec_class, recipeId

    def __len__(self):
        return len(self.ids)
