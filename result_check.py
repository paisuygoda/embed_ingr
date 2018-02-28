import pickle
import json
import numpy as np
import sys


def check_vectorize():
    with open("data/subdata/dataset_dict.p", 'rb') as f:
        dataset_dict = pickle.load(f)
    with open("results/img_feature.p", 'rb') as f:
        img_feature = pickle.load(f)
    with open("results/ing_feature.p", 'rb') as f:
        ing_feature = pickle.load(f)
    with open("results/recipe_id_list.p", 'rb') as f:
        recipe_id_list = pickle.load(f)

    tp = 0
    fp = 0
    fn = 0
    for qid, query_emb in enumerate(img_feature):

        proceeding = float(qid)/10.0
        sys.stdout.write("\r%d%%" % proceeding)
        if proceeding >= 100:
            break
        min = 9999.0
        id = -1
        for i, ing_emb in enumerate(ing_feature):
            dist = np.linalg.norm(query_emb - ing_emb)
            if dist < min and (i is not qid):
                min = dist
                id = i
        if id == -1:
            print("all embeds are too far!")
            exit(1)
        query_id = str(recipe_id_list[qid])
        result_id = str(recipe_id_list[id])

        query = dataset_dict[query_id]['ingredients']
        result = dataset_dict[result_id]['ingredients']

        if 10 < qid < 20:
            print("new_query = ", query, ", new_result = ", result)

        TP = []
        for item in query:
            if item in result:
                TP.append(item)

        tp += len(TP)
        fp += len(result) - len(TP)
        fn += len(query) - len(TP)

    precision = float(tp)/float(tp+fp)
    recall = float(tp)/float(tp+fn)
    f = 2 * precision * recall / (precision + recall)
    print('\nprecision = ', precision, '\nrecall = ', recall, '\nf-measure = ', f)


def check_multilabel():

    with open("data/subdata/dataset_dict.p", 'rb') as f:
        ingr_dic = pickle.load(f)
    with open("results/img_embeds.pkl", 'rb') as f:
        img_embeds = pickle.load(f)
    with open("results/rec_ids.pkl", 'rb') as f:
        rec_ids = pickle.load(f)
    with open('data/subdata/ontrogy_ingrcls.p', mode='rb') as f:
        ontrogy = pickle.load(f)
    with open('data/subdata/ingr_id.p', 'rb') as f:
        ingr_id = pickle.load(f)

    id_ingr = {}
    for k,v in ingr_id.items():
        id_ingr[v] = k

    tp = 0
    fp = 0
    fn = 0
    thres = 0.7

    for count, (out_label, ans_label) in enumerate(zip(img_embeds, rec_embeds)):

        # sys.stdout.write("\r%d" % count)
        # sys.stdout.flush()
        if count >= 1000:
            break

        query = []
        for i, v in enumerate(out_label):
            if v < thres:
                continue
            try:
                query.append(id_ingr[i])
            except:
                pass

        result_id = str(rec_ids[count])
        ori_result = ingr_dic[result_id]['ingr']

        result = []
        for i, v in enumerate(ans_label):
            if v < thres:
                continue
            try:
                result.append(ontrogy[id_ingr[i]])
            except:
                pass

        if 20 < count < 30:
            print("query = ", query, "\nresult = ", result, "\nori_result = ", ori_result)

        TP = []
        for item in query:
            if item in result:
                TP.append(item)

        tp += len(TP)
        fp += len(result) - len(TP)
        fn += len(query) - len(TP)

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f = 2 * precision * recall / (precision + recall)
    print('\nprecision = ', precision, '\nrecall = ', recall, '\nf-measure = ', f)


def getdiff(a,b):
    count = 0
    total = len(a) + len(b)
    for aa in a:
        if aa in b:
            count += 2
    diff = total-count
    return diff


def check_feature():
    with open("data/subdata/dataset_dict.p", 'rb') as f:
        dataset_dict = pickle.load(f)
    with open("results/img_feature.p", 'rb') as f:
        img_feature = pickle.load(f)
    with open("results/ing_feature.p", 'rb') as f:
        ing_feature = pickle.load(f)
    with open("results/recipe_id_list.p", 'rb') as f:
        recipe_id_list = pickle.load(f)
    with open("results/individual_ing_feature.p", 'rb') as f:
        ind_feature = pickle.load(f)
    with open('data/subdata/ingr_id2ingr_text.p', mode='rb') as f:
        ingr_id2ingr_text = pickle.load(f)

    count = 0
    for qid, query_id in enumerate(recipe_id_list):
        query_id = str(query_id)
        query = dataset_dict[query_id]['ingredients']
        for rid, result_id in enumerate(recipe_id_list):
            result_id = str(result_id)
            result = dataset_dict[result_id]['ingredients']
            dist = getdiff(query, result)
            if dist == 0:
                query_emb = img_feature[qid]
                ing_emb = ing_feature[rid]
                print(query_emb - ing_emb)
                print("Should be 0\n--------------------")
            elif dist == 1:
                query_emb = img_feature[qid]
                ing_emb = ing_feature[rid]
                diff = query_emb - ing_emb
                min = 9999.0
                id = -1
                for i, ind_emb in enumerate(ind_feature):
                    dist = np.linalg.norm(diff - ind_emb)
                    if dist < min:
                        min = dist
                        id = i
                    dist = np.linalg.norm(ind_emb - diff)
                    if dist < min:
                        min = dist
                        id = i
                if id == -1:
                    print("all embeds are too far!")
                    exit(1)
                print("actual diff\n" , diff)
                print("closest feature\n", ind_feature[id])
                print("ingredients: \t", query, "\n\t", result)
                print("estimated diff: ", ingr_id2ingr_text[id])
                print("------------------------")
                count += 1
                break
        if count >= 3:
            break


print("MODE? (1 = vectorize, 2 = multilabel, 3 = feature)")
m = input()
if m == "1":
    check_vectorize()
elif m == "2":
    check_multilabel()
elif m == "3":
    check_feature()
else:
    print("Bad input mode")