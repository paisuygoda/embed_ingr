import pickle
import json
import numpy as np
import sys


def check_vectorize():
    with open("data/dataset_dict.p", 'rb') as f:
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
        if proceeding >= 1000:
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

        query = dataset_dict[query_id]['ingr']
        result = dataset_dict[result_id]['ingr']

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

    with open("data/ingredients_dict.p", 'rb') as f:
        ingr_dic = pickle.load(f)
    with open("results/img_embeds.pkl", 'rb') as f:
        img_embeds = pickle.load(f)
    # with open("results/img_ids.pkl", 'rb') as f:
    #     img_ids = pickle.load(f)
    with open("results/rec_embeds.pkl", 'rb') as f:
        rec_embeds = pickle.load(f)
    with open("results/rec_ids.pkl", 'rb') as f:
        rec_ids = pickle.load(f)
    with open('data/ontrogy_ingrcls.p', mode='rb') as f:
        ontrogy = pickle.load(f)
    with open('data/ingr_id.p', 'rb') as f:
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


print("MODE? (1 = vectorize, 2 = multilabel)")
m = input()
if m == "1":
    check_vectorize()
elif m == "2":
    check_multilabel()
else:
    print("Bad input mode")