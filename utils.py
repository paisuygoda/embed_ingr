import pickle
from RakutenData import RakutenData

def look_pickle(path):
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
        print(type(s))
        if s in file:
            print(file[s])
        else:
            print(file)
    else:
        print(file)
    print(len(file))


def datacheck():
    print(RakutenData[0])

print("MODE? (1 = datacheck, 2 = image, 3 = pickle, 4 = text, 5 = img separation, \n\t6 = recipe_ingr, 7 = ontrogy)")
m = input()
print("PATH?")
path = input()
if m == "3":
    if path == 'O':
        path = 'data/ontrogy_ingrcls.p'
    look_pickle(path)
elif m == "1":
    datacheck()
else:
    print("Bad input mode")