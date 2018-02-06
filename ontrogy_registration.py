import csv
import pickle

"""
with open("data/synonym_edited.tsv", 'r', encoding='utf8') as f:
    reader = csv.reader(f, delimiter='\t')
    dic_all = {}
    first_tag = "材料-魚介"
    first_tag_inside = {}
    second_tag = "サケ"
    second_tag_inside = []
    gyaku_dic = {}
    gyaku_cat_dic = {"サケ":"材料-魚介"}
    for r in reader:
        if (second_tag != r[1]) and (len(second_tag_inside) != 0):
            first_tag_inside[second_tag] = second_tag_inside
            second_tag_inside = []
            second_tag = r[1]
            gyaku_cat_dic[r[1]] = r[0]
        if (first_tag != r[0]) and (len(first_tag_inside) != 0):
            dic_all[first_tag] = first_tag_inside
            first_tag_inside = {}
            first_tag = r[0]

        second_tag_inside.append(r[2])
        gyaku_dic[r[2]] = r[1]
"""
try:
    with open('data/numcount.p', mode='rb') as f:
        start = pickle.load(f)
    with open('data/ontrogy_original.p', mode='rb') as f:
        dic_all = pickle.load(f)
    with open('data/ontrogy_gyaku.p', mode='rb') as f:
        gyaku_dic = pickle.load(f)
    with open('data/ontrogy_gyaku_cat.p', mode='rb') as f:
        gyaku_cat_dic = pickle.load(f)
except:
    print("pickle load error")
    start = 0

start = 0
with open("data/dropped_list.csv", 'r') as f:
    reader = csv.reader(f)
    inputmode = True
    for i, r in enumerate(reader):
        if i < start or r[0] in gyaku_dic:
            continue
        print("\n", r[0], " : ", r[1], "\nquery?")
        while True:
            s = input()
            if s is "":
                inputmode = False
                break
            elif s in gyaku_dic:
                print(s, " exists! \nits category: ", gyaku_dic[s], ", its first_category: ", gyaku_cat_dic[gyaku_dic[s]])
                print("Do you want to add this?")
                ss = input()
                if ss in ["", "y", "yes"]:
                    dic_all[gyaku_cat_dic[gyaku_dic[s]]][gyaku_dic[s]].append(r[0])
                    break
                else:
                    print("Try another query.")
            elif s is "C":
                print("Create new category")
                s = input()
                if s is "":
                    print("Create category canceled. Try another query.")
                else:
                    print("New category ", s, "is created.\nWhich first_category it belongs?(leave it blank for new first_category)")
                    ss = input()
                    if ss is "":
                        print("Name of new first_category?")
                        ss = input()
                        ne = {s:[r[0]]}
                        print("New entry: {", ss, " : ", ne)
                        dic_all[ss] = ne
                        gyaku_dic[r[0]] = s
                        gyaku_cat_dic[s] = ss
                        break
                    elif ss in gyaku_cat_dic:
                        print("new category ", s, "belongs to ", gyaku_cat_dic[ss])
                        dic_all[gyaku_cat_dic[ss]][s] = [r[0]]
                        gyaku_dic[r[0]] = s
                        gyaku_cat_dic[s] = gyaku_cat_dic[ss]
                        break
                    else:
                        print("first_category not found.\nTry another query")
            else:
                print("No ", s, " in original dic...\nTry another query.")
        if not inputmode:
            num = i
            break

with open('data/ontrogy_original.p', mode='wb') as f:
    pickle.dump(dic_all, f)
with open('data/ontrogy_gyaku.p', mode='wb') as f:
    pickle.dump(gyaku_dic, f)
with open('data/ontrogy_gyaku_cat.p', mode='wb') as f:
    pickle.dump(gyaku_cat_dic, f)
with open('data/numcount.p', mode='wb') as f:
    pickle.dump(num, f)

with open("data/synonym_edited.tsv", 'w', encoding='utf8') as f:
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    for k,v in dic_all.items():
        for vk, vv in v.items():
            for vvi in vv:
                r = [k, vk, vvi]
                writer.writerow(r)