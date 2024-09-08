import csv
import json
import numpy as np

with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/train_index.npy', 'rb') as f:
    train_index = np.load(f)
train_index = train_index.tolist()
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/val_index.npy', 'rb') as f:
    val_index = np.load(f)
val_index = val_index.tolist()
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/test_index.npy', 'rb') as f:
    test_index = np.load(f)
test_index = test_index.tolist()

val_data = []
train_data = []
test_data = []
count1 = 0
count2 = 0

with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_data_0shot_no_sen_feature_sample.csv', 'r') as f:
    csvreader = csv.DictReader(f)
    for idx, row in enumerate(csvreader):
        if idx not in val_index:
            continue
        input = row['QUESTION']
        input = input[:-64] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': 'Given the patient information, predict the mortality of the patient.\nAnswer 1 if the patient will die, answer 0 otherwise.\nAnswer with only the number', 'input': input, 'output': output}
        # dic = {'instruction': 'Given the patient information, predict the mortality of the patient.', 'input': input, 'output': output}
        val_data.append(dic)

with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_data_0shot_no_sen_feature_sample.csv', 'r') as f:
    csvreader = csv.DictReader(f)
    for idx, row in enumerate(csvreader):
        if idx not in train_index:
            continue
        input = row['QUESTION']
        input = input[:-64] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': 'Given the patient information, predict the mortality of the patient.\nAnswer 1 if the patient will die, answer 0 otherwise.\nAnswer with only the number', 'input': input, 'output': output}
        # dic = {'instruction': 'Given the patient information, predict the mortality of the patient.', 'input': input, 'output': output}
        train_data.append(dic)

with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_data_0shot_no_sen_feature_sample.csv', 'r') as f:
    csvreader = csv.DictReader(f)
    for idx, row in enumerate(csvreader):
        if idx not in test_index:
            continue
        input = row['QUESTION']
        input = input[:-64] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': 'Given the patient information, predict the mortality of the patient.\nAnswer 1 if the patient will die, answer 0 otherwise.\nAnswer with only the number', 'input': input, 'output': output}
        # dic = {'instruction': 'Given the patient information, predict the mortality of the patient.', 'input': input, 'output': output}
        test_data.append(dic)
import random
random.seed(33)
random.shuffle(train_data)

# train_data = val_data + train_data
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/train_data.json', 'w') as f:
    json.dump(train_data, f)
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/test_data.json', 'w') as f:
    json.dump(test_data, f)
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/val_data.json', 'w') as f:
    json.dump(val_data, f)