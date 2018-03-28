import pickle


def count_ingr():
    with open("data/subdata/major_all_valid_images.p", 'rb') as f:
        all_images = pickle.load(f)

    with open('data/subdata/major_dish_dataset_dict.p', 'rb') as f:
        dataset_dict = pickle.load(f)

    dish_map = {}
    for recipe_id in all_images:
        recipe_id = recipe_id[27:-4]
        data = dataset_dict[recipe_id]
        if data['title'] in dish_map:
            for ingr in data['ingr']:
                if ingr in dish_map[data['title']]:
                    dish_map[data['title']][ingr] += 1
                else:
                    dish_map[data['title']][ingr] = 1
        else:
            ingr_dic = {}
            for ingr in data['ingr']:
                ingr_dic[ingr] = 1
            dish_map[data['title']] = ingr_dic

    with open('data/subdata/major_dish_ingr_count.p', mode='wb') as f:
        pickle.dump(dish_map, f)

if __name__ == "__main__":
    count_ingr()
