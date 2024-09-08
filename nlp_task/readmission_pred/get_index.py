import csv
import json
import os
import random
import numpy as np

dataset = "mimic4"

with open(f"/home/peter/files/PyHealth/nlp_task/readmission_pred/{dataset}/readmission_pred_data.json", "r") as f:
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
    if dataset == "mimic3":
        # train_index = cur_label0[0:15911] + cur_label1[0:2100]
        train_index = cur_label0[0:277] + cur_label1[0:277]
        
        cur_label0 = cur_label0[3500:]
        cur_label1 = cur_label1[277:]

        val_index = cur_label0[0:500] + cur_label1[0:40]
        cur_label0 = cur_label0[500:]
        cur_label1 = cur_label1[40:]

        test_index = cur_label0[0:1000] + cur_label1[0:79]
    elif dataset == "mimic4":
        train_index = cur_label0[0:2323] + cur_label1[0:2323]
        
        cur_label0 = cur_label0[14000:]
        cur_label1 = cur_label1[2323:]

        val_index = cur_label0[0:2000] + cur_label1[0:332]
        cur_label0 = cur_label0[2000:]
        cur_label1 = cur_label1[332:]

        test_index = cur_label0[0:4000] + cur_label1[0:664]

    with open(f"/home/peter/files/PyHealth/nlp_task/readmission_pred/{dataset}/train_index_{i}.npy", "wb") as f:
        np.save(f, train_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/readmission_pred/{dataset}/val_index_{i}.npy", "wb") as f:
        np.save(f, val_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/readmission_pred/{dataset}/test_index_{i}.npy", "wb") as f:
        np.save(f, test_index)
