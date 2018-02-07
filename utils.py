import pickle
from RakutenData import RakutenData
import sys

sys.stdin =  open('/dev/stdin',  'r', encoding='UTF-8')
sys.stdout = open('/dev/stdout', 'w', encoding='UTF-8')
sys.stderr = open('/dev/stderr', 'w', encoding='UTF-8')

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


def datacheck():

    d = RakutenData(img_path="/home/goda/im2ingr/data/images/", partition="train")
    with open('data/subdata/recipe_id2recipe_text.p', mode='rb') as f:
        recipe_id2text = pickle.load(f)
    with open('data/subdata/ingr_id2ingr_text.p', mode='rb') as f:
        ingr_id2text = pickle.load(f)
    with open('data/subdata/ingredients_dict.p', mode='rb') as f:
        actual_ingr = pickle.load(f)
    with open('data/subdata/outline_dict.p', mode='rb') as f:
        actual_class = pickle.load(f)
    for i in range(10):
        c = d[i]
        ingr_id = c[1]
        ingr = []
        for i in range(c[2]):
            ingr.append(ingr_id2text[ingr_id[i]])
        rec_class = recipe_id2text[int(c[3])]
        print("\nWhat you got...\ningr: ", ingr, "\nclass: ", rec_class)
        recipe_id = c[4]
        ingr = actual_ingr[recipe_id]
        rec_class = actual_class[recipe_id]
        print("What actually is ...\ningr: ", ingr, "\nclass: ", rec_class["dish_class"])

print("MODE? (1 = datacheck, 2 = image, 3 = pickle, 4 = text, 5 = img separation, \n\t6 = recipe_ingr, 7 = ontrogy)")
m = input()
print("PATH?")
path = input()
if m == "3":
    look_pickle(path)
elif m == "1":
    datacheck()
else:
    print("Bad input mode")