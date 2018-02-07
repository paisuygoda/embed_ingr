# -*- coding: utf-8 -*-
import pickle
import os
from numpy import random
from PIL import Image
import csv
import sys
import MeCab
import re


def img_sep(path):
    directory = os.listdir(path)
    train_list  =[]
    test_list   =[]
    val_list    =[]

    totalnum = len(directory)
    train_limit = int(totalnum * 0.7)
    test_limit = int((totalnum - train_limit) / 2)
    val_limit = totalnum - train_limit - test_limit

    for pic in directory:
        seed = random.rand()
        if seed < 0.7 and len(train_list) < train_limit:
            train_list.append(pic)
        elif seed < 0.85 and len(test_list) < test_limit:
            test_list.append(pic)
        elif len(val_list) < val_limit:
            val_list.append(pic)
        elif len(train_list) < train_limit:
            train_list.append(pic)
        else:
            test_list.append(pic)

    print("len(train) = ", len(train_list))
    print("len(test)  = ", len(test_list))
    print("len(val)   = ", len(val_list))

    with open("data/subdata/train_images.p", 'wb') as f:
        pickle.dump(train_list,f)
    with open("data/subdata/test_images.p", 'wb') as f:
        pickle.dump(test_list,f)
    with open("data/subdata/val_images.p", 'wb') as f:
        pickle.dump(val_list,f)


def J2H(mecab, text):

    mecab.parse("")
    node = mecab.parseToNode(text)
    yomi = ""
    while node:
        parseresult = node.feature.split(",")
        if node.surface in ["酒", "柿"]:
            yomi += node.surface
        else:
            try:
                if parseresult[7] != "*":
                    yomi += parseresult[7]
            except:
                yomi += node.surface
        node = node.next
    return yomi

def ontrogy():

    tsv = csv.reader(open("data/synonym_edited.tsv", "r", encoding="utf-8"), delimiter='\t')
    dic = {}
    word = ""
    count = 1
    id_dic = {}
    ingr_id2ingr_text = ["*"]
    mecab = MeCab.Tagger("-Ochasen")
    for row in tsv:
        if row[0] == "調理器具":
            break
        if row[1] != word:
            word = row[1]
            id_dic[word] = count
            ingr_id2ingr_text.append(word)
            count += 1
        kana = J2H(mecab, row[2])
        dic[kana] = row[1]

    with open('data/subdata/ontrogy_ingrcls.p', mode='wb') as f:
        pickle.dump(dic, f)
    with open('data/subdata/ingr_id.p', mode='wb') as f:
        pickle.dump(id_dic, f)
    with open('data/subdata/ingr_id2ingr_text.p', mode='wb') as f:
        pickle.dump(ingr_id2ingr_text, f)
    print(count)


def process_outline():
    data = {}
    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split()
        try:
            linedict = {"title": linelist[5], "dish": linelist[9], "dish_class" : linelist[3]}
        except:
            # print(linelist)
            b = 5 # int(input())
            c = 4 # int(input())
            a = 4 # int(input())
            linedict = {"title": linelist[b], "dish": linelist[c], "dish_class": linelist[a]}
            # print("linedict = {'id': ", linelist[0], ", 'title': ", linelist[b], ", 'dish': ", linelist[c], ", 'dish_class': ", linelist[a], "}")
        data[linelist[0]] = linedict

    with open('data/subdata/outline_dict.p', mode='wb') as f:
        pickle.dump(data, f)


def process_ingredients():
    data = {}
    id = 0

    for line in open('data/Rakuten/recipe02_material_20160112.txt', 'r', encoding="utf-8"):
        linelist = line.split()
        if id == 0:
            id = linelist[0]
            ingrlist = []
        elif not linelist[0] == id:
            data[id] = ingrlist
            id = linelist[0]
            ingrlist = []
        text = re.sub('[◎●Ａ　ABＢ■○①②③☆★※＊*▽▼▲△◆◇・()（）]', '', linelist[1])
        ingrlist.append(text)
    data[id] = ingrlist

    with open('data/subdata/ingredients_dict.p', mode='wb') as f:
        pickle.dump(data, f)

def combine_outline_ingredients():
    with open('data/subdata/ingredients_dict.p', mode='rb') as f:
        ingr_dict = pickle.load(f)
    with open('data/subdata/outline_dict.p', mode='rb') as f:
        outlinedict = pickle.load(f)

    for k,v in outlinedict.items():
        try:
            v["ingredients"] = ingr_dict[k]
        except:
            v["ingredients"] = []

    with open('data/subdata/dataset_dict.p', mode='wb') as f:
        pickle.dump(outlinedict, f)

def data_dict():
    data = {}
    recipeid = 0
    count = 0.0
    misscount = 0.0
    dropped_dict = {}
    mecab = MeCab.Tagger("-Ochasen")
    with open('data/subdata/ontrogy_ingrcls.p', mode='rb') as f:
        ontrogy = pickle.load(f)

    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split('\t')
        data[linelist[0]] = {"title": linelist[5], "dish": linelist[9], "class": linelist[3]}

    for line in open('data/Rakuten/recipe02_material_20160112.txt', 'r', encoding="utf-8"):
        count += 1.0
        proceeding = count / 5274990.0 * 100.0
        sys.stdout.write("\r%f%%" % proceeding)
        linelist = line.split()
        if recipeid == 0:
            recipeid = linelist[0]
            ingrlist = []
        elif not linelist[0] == recipeid:
            data[recipeid]['ingr'] = ingrlist
            recipeid = linelist[0]
            ingrlist = []
        text = re.sub('[◎●Ａ　 ABＢ■○①②③☆★※＊*▽▼▲△◆◇・()（）]', '', linelist[1])
        text = J2H(mecab, text)
        if text in ontrogy:
            ingrlist.append(ontrogy[text])
        else:
            if text in dropped_dict:
                dropped_dict[text] += 1
            else:
                dropped_dict[text] = 1
            misscount += 1.0
    data[recipeid]['ingr'] = ingrlist

    with open('data/subdata/ingredients_dict.p', mode='wb') as f:
        pickle.dump(data, f)

    print("\ndropped ingredient: ", misscount / count * 100.0, "%")

    f = open('dropped_list.txt', 'w', encoding="utf-8")
    for k, v in dropped_dict.items():
        f.write(str(k) + "\t" + str(v) + "\n")
    f.close()


def class_id_set():
    recipe_class = {}
    recipe_id = 1
    id2text = ["*"]
    for line in open('data/Rakuten/recipe01_all_20170118.txt', 'r', encoding="utf-8"):
        linelist = line.split('\t')
        dish = linelist[9]
        dish_class = linelist[3]
        if dish_class not in recipe_class:
            recipe_class[dish_class] = recipe_id
            id2text.append(dish_class)
            recipe_id += 1
    print(recipe_id)
    with open('data/subdata/recipe_class.p', mode='wb') as f:
        pickle.dump(recipe_class, f)

    with open('data/subdata/recipe_id2recipe_text.p', mode='wb') as f:
        pickle.dump(id2text, f)

# img_sep("home/goda/im2ingr/data/images/")
# process_outline()
# process_ingredients()
# combine_outline_ingredients()
# ontrogy()
# data_dict()
class_id_set()