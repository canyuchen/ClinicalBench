import csv
import json
import os
import random
import numpy as np

dataset = "mimic4"

with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data.json", "r") as f:
    data = json.load(f)


count_1 = 0
count_2 = 0
count_3 = 0

label1 = []
label2 = []
label3 = []

for index in range(len(data)):
    if data[index]["label"] == 1:
        count_1 += 1   
        label1.append(data[index]["visit_id"])
    elif data[index]["label"] == 2:
        count_2 += 1
        label2.append(data[index]["visit_id"])
    else:
        count_3 += 1
        label3.append(data[index]["visit_id"])


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
    cur_label1 = shuffle_with_seed(label1, seed=seed_list[i])
    cur_label2 = shuffle_with_seed(label2, seed=seed_list[i])
    cur_label3 = shuffle_with_seed(label3, seed=seed_list[i])
    # train_index = cur_label1[0:8400] + cur_label2[0:4175] + cur_label3[0:2980]
    if dataset == "mimic3":
        print("mimic3")
        train_index = cur_label1[0:2980] + cur_label2[0:2980] + cur_label3[0:2980]
        cur_label1 = cur_label1[8400:]
        cur_label2 = cur_label2[4175:]
        cur_label3 = cur_label3[2980:]
        val_index = cur_label1[0:1200] + cur_label2[0:596] + cur_label3[0:426]
        cur_label1 = cur_label1[1200:]
        cur_label2 = cur_label2[596:]
        cur_label3 = cur_label3[426:]
        test_index = cur_label1[0:2400] + cur_label2[0:1193] + cur_label3[0:852]
    elif dataset == "mimic4":
        print("mimic4")
        train_index = cur_label1[0:1292] + cur_label2[0:1292] + cur_label3[0:1292]
        cur_label1 = cur_label1[14000:]
        cur_label2 = cur_label2[2278:]
        cur_label3 = cur_label3[1292:]
        val_index = cur_label1[0:2000] + cur_label2[0:325] + cur_label3[0:185]
        cur_label1 = cur_label1[2000:]
        cur_label2 = cur_label2[325:]
        cur_label3 = cur_label3[185:]
        test_index = cur_label1[0:4000] + cur_label2[0:651] + cur_label3[0:369]
        


    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/train_index_{i}.npy", "wb") as f:
        np.save(f, train_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/val_index_{i}.npy", "wb") as f:
        np.save(f, val_index)
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/test_index_{i}.npy", "wb") as f:
        np.save(f, test_index)
