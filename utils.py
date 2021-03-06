import pickle
from RakutenData import RakutenData
import sys
import numpy as np

"""
sys.stdin =  open('/dev/stdin',  'r', encoding='UTF-8')
sys.stdout = open('/dev/stdout', 'w', encoding='UTF-8')
sys.stderr = open('/dev/stderr', 'w', encoding='UTF-8')
"""


def look_pickle(path):
    path = "data/subdata/" + path
    with open(path, 'rb') as f:
        file = pickle.load(f)

    if type(file) is list:
        for key, value in enumerate(file):
            print(value)
            if key > 10:
                break
    elif type(file) is dict:
        print("key?")
        s = input()
        if s in file:
            print(file[s])
        else:
            print(file)
    else:
        print(file)
    print(len(file))


def look_pickle_r(path):
    path = "results/" + path
    with open(path, 'rb') as f:
        file = pickle.load(f)

    if type(file) is list:
        for key, value in enumerate(file):
            print(value)
            if key > 10:
                break
    elif type(file) is dict:
        print("key?")
        s = input()
        if s in file:
            print(file[s])
        else:
            print(file)
    else:
        print(file)
    print(len(file))


def datacheck():

    d = RakutenData(img_path="/home/goda/im2ingr/data/images/", partition="train", mode="use")
    with open('data/subdata/recipe_id2recipe_text.p', mode='rb') as f:
        recipe_id2text = pickle.load(f)
    with open('data/subdata/ingr_id2ingr_text.p', mode='rb') as f:
        ingr_id2text = pickle.load(f)
    with open('data/subdata/dataset_dict.p', mode='rb') as f:
        dataset_dict = pickle.load(f)
    for i in range(10):
        c = d[i]
        ingr_id = c[1]
        ingr = []
        for i in range(c[2]):
            ingr.append(ingr_id2text[ingr_id[i]])
        rec_class = recipe_id2text[int(c[3])]
        recipe_id = c[4]
        print(recipe_id)
        print("\nWhat you got...\ningr: ", ingr, "\nclass: ", rec_class)
        actual = dataset_dict[recipe_id]
        print("What actually is ...\ningr: ", actual["ingredients"], "\nclass: ", actual["dish_class"])

print("MODE? (1 = datacheck, 2 = pickle_fromresult, 3 = pickle, 4 = text, 5 = img separation, \n\t6 = recipe_ingr, 7 = ontrogy)")
m = input()
print("PATH?")
path = input()
if m == "3":
    look_pickle(path)
elif m == "1":
    datacheck()
elif m == '2':
    look_pickle_r(path)
else:
    print("Bad input mode")