import csv
import json
import os
import random
import numpy as np

dataset = "mimic4"

with open(f"/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/mortality_pred_data.json", "r") as f:
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
    # 创建原始列表的副本
    shuffled_list = original_list.copy()
    
    # 设置随机种子
    random.seed(seed)
    
    # 对副本进行洗牌
    random.shuffle(shuffled_list)
    
    return shuffled_list


seed_list = [3, 5, 7, 11, 13]

for i in range(5):
    cur_label0 = shuffle_with_seed(label0, seed=seed_list[i])
    cur_label1 = shuffle_with_seed(label1, seed=seed_list[i])
    
    # train_index = cur_label0[0:15911] + cur_label1[0:2100]
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

    with open(f"/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/train_index_{i}.npy", "wb") as f:
        np.save(f, train_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/val_index_{i}.npy", "wb") as f:
        np.save(f, val_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/test_index_{i}.npy", "wb") as f:
        np.save(f, test_index)
