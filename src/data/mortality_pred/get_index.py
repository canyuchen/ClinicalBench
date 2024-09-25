import csv
import json
import os
import random
import numpy as np

datasets = ["mimic3", "mimic4"]


for dataset in datasets:
    with open(f"{dataset}/mortality_pred_data.json", "r") as f:
        data = json.load(f)

    count_0 = 0
    count_1 = 0

    label0 = []
    label1 = []

    for index in range(len(data)):
        if data[index]["label"] == 0:
            count_0 += 1
            label0.append(data[index]["visit_id"])
        else:
            count_1 += 1
            label1.append(data[index]["visit_id"])



    def shuffle_with_seed(original_list, seed):
        shuffled_list = original_list.copy()
        random.seed(seed)
        random.shuffle(shuffled_list)
        return shuffled_list


    seed_list = [3, 5, 7, 11, 13]

    for i in range(5):
        cur_label0 = shuffle_with_seed(label0, seed=seed_list[i])
        cur_label1 = shuffle_with_seed(label1, seed=seed_list[i])
        
        if dataset == "mimic3":
            train_index = cur_label0[0:2100] + cur_label1[0:2100]
            
            cur_label0 = cur_label0[15911:]
            cur_label1 = cur_label1[2100:]

            val_index = cur_label0[0:2273] + cur_label1[0:300]
            cur_label0 = cur_label0[2273:]
            cur_label1 = cur_label1[300:]

            test_index = cur_label0[0:4546] + cur_label1[0:600]

        elif dataset == "mimic4":
            train_index = cur_label0[0:700] + cur_label1[0:700]
            
            cur_label0 = cur_label0[19487:]
            cur_label1 = cur_label1[700:]

            val_index = cur_label0[0:2784] + cur_label1[0:100]
            cur_label0 = cur_label0[2784:]
            cur_label1 = cur_label1[100:]

            test_index = cur_label0[0:5568] + cur_label1[0:200]     

        with open(f"{dataset}/train_index_{i}.npy", "wb") as f:
            np.save(f, train_index)
        with open(f"{dataset}/val_index_{i}.npy", "wb") as f:
            np.save(f, val_index)
        with open(f"{dataset}/test_index_{i}.npy", "wb") as f:
            np.save(f, test_index)



    if dataset == "mimic3":

        # 500 sample mimic3

        cur_label0 = shuffle_with_seed(label0, seed=19)
        cur_label1 = shuffle_with_seed(label1, seed=19)

        # 1546	204
        # 221	29
        # 442	58

        train_index = cur_label0[0:204] + cur_label1[0:204]
        cur_label0 = cur_label0[1546:]
        cur_label1 = cur_label1[204:]

        val_index = cur_label0[0:221] + cur_label1[0:29]
        cur_label0 = cur_label0[221:]
        cur_label1 = cur_label1[29:]

        test_index = cur_label0[0:442] + cur_label1[0:58]
    




    if dataset == "mimic4":
        # 500 sample mimic4

        cur_label0 = shuffle_with_seed(label0, seed=19)
        cur_label1 = shuffle_with_seed(label1, seed=19)

        # 1689	61
        # 241	9
        # 483	17

        train_index = cur_label0[0:61] + cur_label1[0:61]
        cur_label0 = cur_label0[1689:]
        cur_label1 = cur_label1[61:]

        val_index = cur_label0[0:241] + cur_label1[0:9]
        cur_label0 = cur_label0[241:]
        cur_label1 = cur_label1[9:]

        test_index = cur_label0[0:483] + cur_label1[0:17]


    with open(f"{dataset}/train_index_6.npy", "wb") as f:
        np.save(f, train_index)
    with open(f"{dataset}/val_index_6.npy", "wb") as f:
        np.save(f, val_index)
    with open(f"{dataset}/test_index_6.npy", "wb") as f:
        np.save(f, test_index)

