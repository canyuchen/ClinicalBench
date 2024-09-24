from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import argparse




def calculate_f1(task, dataset, model, random_index, mode_str, temp_str):
    model_name = model.split('/')[-1]
    with open(f'{result_root}/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}{mode_str}{temp_str}.csv', "r") as f:
        csvreader = csv.DictReader(f)
        answer = []
        predict = []
        for row in csvreader:
            answer.append(int(row["ANSWER"]))
            if row["PREDICTION"] not in ['0', '1', '2', '3']:
                row["PREDICTION"] = '4'
            predict.append(int(row["PREDICTION"]))
        if task == "length_pred":
            for i in range(len(predict)):
                if predict[i] not in [1, 2, 3]:
                    predict[i] = 2 if answer[i] == 1 else 1
            f1 = f1_score(answer, predict, average="macro")
        else:
            for i in range(len(predict)):
                if predict[i] not in [0, 1]:
                    predict[i] = 1 - answer[i]
            f1 = f1_score(answer, predict)
        confu = confusion_matrix(answer, predict)
        print(confu)
    return f1

def calculate_auroc(task, dataset, model, random_index, mode_str, temp_str):
    model_name = model.split('/')[-1]
    with open(f'{result_root}/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}{mode_str}{temp_str}.csv', "r") as f:
        csvreader = csv.DictReader(f)
        answer = []
        prob = []
        for row in csvreader:
            if task == "length_pred":
                answer.append(0 if (row["ANSWER"] == '1' or row["ANSWER"] == '2') else 1)
            else:
                answer.append(int(row["ANSWER"]))
            
            prob.append(float(row["PROB"]))
        auroc = roc_auc_score(answer, prob)
    return auroc

if __name__ == "__main__":
    pragma = argparse.ArgumentParser()
    pragma.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model name")
    pragma.add_argument("--task", type=str, default="length_pred", help="Task name")
    pragma.add_argument("--random_index", type=int, default=0, help="Random index")
    pragma.add_argument("--mode", type=str, default="ORI", choices=["ORI", "ICL", "COT", "RP", "SR"], help="Mode")
    pragma.add_argument("--temperature", type=float, default=None, help="Temperature for sampling")
    pragma.add_argument("--result_root", type=str, default="results", help="Result root")
    pragma.add_argument("--dataset", type=str, default="mimic3", help="Dataset name")
    pragma.add_argument("--AUROC", type=bool, default=False)
    
    args = pragma.parse_args()
    model = args.base_model
    task = args.task
    random_index = args.random_index
    mode = args.mode
    result_root = args.result_root
    temperature = args.temperature
    dataset = args.dataset
    cal_auroc = args.AUROC


    mode_to_str = {
        "ORI": "",
        "ICL": "_ICL",
        "COT": "_COT",
        "RP": "_RP",
        "SR": "_SR",
    }
    mode_str = mode_to_str[mode]

    temp_str = f"_temp{temperature}" if temperature else ""
    
    
    
    f1 = calculate_f1(task, dataset, model, random_index, mode_str, temp_str)
    print(f"F1: {round((f1*100),2)}")
    if cal_auroc:
        auroc = calculate_auroc(task, dataset, model, random_index, mode_str, temp_str)
        print(f"AUROC: {round((auroc*100),2)}")