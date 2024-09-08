import csv
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
import json



def calculate_pr_auc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    return pr_auc

def calculate_roc_auc(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return roc_auc

def calculate_f1(y_true, y_pred, threshold=0.5):
    f1 = f1_score(y_true, y_pred)
    return f1

def calculate_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall = tp / (tp + fn)
    return recall

def calculate_precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    precision = tp / (tp + fp)
    return precision

def equal_opportunity(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the equal opportunity, which measures the difference in true positive rate between different values
    of a binary sensitive attribute. The true positive rate is defined as the proportion of positive data points
    that are correctly predicted to be positive.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of binary ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in true positive rate
        between different values of the sensitive attribute.
    """
    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred[y_gt == 1]
    sensitive_attribute = sensitive_attribute[y_gt == 1]

    # Convert predicted probabilities to binary predictions using the threshold.
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    
    # Calculate the difference in true positive rate.
    equality = abs(y_z_1.mean() - y_z_0.mean())
    equality *= 100
    return equality
def demographic_parity(y_pred: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the demographic parity, which measures the difference in positive rate between different
    values of a binary sensitive attribute. The positive rate is defined as the proportion of data points
    that are predicted to be positive.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in positive rate
        between different values of the sensitive attribute.
    """
    # Convert predicted probabilities to binary predictions using the threshold.
    
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate the difference in positive rate.
    parity = abs(y_z_1.mean() - y_z_0.mean())
    parity *= 100
    return parity

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calculate_metrics(y_true, y_pred, sensitive, threshold=0.5):
    pr_auc = calculate_pr_auc(y_true, y_pred)
    roc_auc = calculate_roc_auc(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred, threshold)
    recall = calculate_recall(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    EO = equal_opportunity(y_pred, y_true, sensitive, threshold)
    DP = demographic_parity(y_pred, sensitive, threshold)
    EO2 = equal_opportunity(y_pred, y_true, sensitive2, threshold)
    DP2 = demographic_parity(y_pred, sensitive2, threshold)
    EO3 = equal_opportunity(y_pred, y_true, sensitive3, threshold)
    DP3 = demographic_parity(y_pred, sensitive3, threshold)
    EO4 = equal_opportunity(y_pred, y_true, sensitive4, threshold)
    DP4 = demographic_parity(y_pred, sensitive4, threshold)
    acc = calculate_accuracy(y_true, y_pred)


    return {
        'Accuracy': acc,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1,
        'EO': EO,
        'DP': DP,
        'EO2': EO2,
        'DP2': DP2,
        'EO3': EO3,
        'DP3': DP3,
        'EO4': EO4,
        'DP4': DP4
    }



with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/patient_dic.json', 'r') as f:
    patient_dic = json.load(f)



task = "mortality_pred"
read_list = ["Transformer1", "Transformer2", "Transformer5", "RNN1", "RNN2", "RNN5", "XGBoost", "LogisticRegression", "DecisionTree", "RandomForest", "AdaBoost", "SVM", "NaiveBayes", "KNN", "NeuralNetwork", "llama3_instruct", "llama3_instruct_FT", "llama3_ins_qv", "llama3_ins_lm", "llama3_ins_last", "llama3_instruct_ICL", "llama3", "llama3_FT", "llama3_qv", "llama3_lm", "llama3_last", "llama3_ICL", "Mistral", "Mistral_ICL", "Gemma2", "Gemma2_ICL", "gemma-2-9b-it", "gemma-2-9b-it_ICL", "Llama-2-7b-hf", "Llama-2-7b-hf_ICL", "Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf_ICL", "Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf_ICL", "meditron-7b", "meditron-7b_ICL", "MedLLaMA_13B", "MedLLaMA_13B_ICL", "medllama3-v20", "medllama3-v20_ICL", "vicuna-7b-v1.5", "vicuna-7b-v1.5_ICL", "vicuna-13b-v1.5", "vicuna-13b-v1.5_ICL", "Qwen2-0.5B-Instruct", "Qwen2-1.5B-Instruct", "Qwen2-7B-Instruct", "Yi-1.5-6B-Chat", "Yi-1.5-9B-Chat", "HuatuoGPT2-7B", "HuatuoGPT2-7B_ICL", "Baichuan2-7B-Chat", "Baichuan2-13B-Chat", "XGBoost6", "XGBoost8", "XGBoost10", "DecisionTree6", "DecisionTree8", "DecisionTree10", "Llama3-Med42-8B", "Llama3-Med42-8B_ICL", "XGBoost2000", "LogisticRegression2000", "DecisionTree2000", "RandomForest2000", "AdaBoost2000", "SVM2000", "NaiveBayes2000", "KNN2000", "NeuralNetwork2000"]
mor_list = ["Transformer1", "Transformer2", "Transformer5", "RNN1", "RNN2", "RNN5", "XGBoost", "LogisticRegression", "DecisionTree", "RandomForest", "AdaBoost", "SVM", "NaiveBayes", "KNN", "NeuralNetwork", "llama3_instruct", "llama3_instruct_FT", "llama3_ins_qv", "llama3_ins_lm", "llama3_ins_last", "llama3_instruct_ICL", "llama3", "llama3_FT", "llama3_FT_qv", "llama3_lm", "llama3_last", "llama3_ICL", "Mistral", "Mistral_ICL", "Gemma2", "Gemma2_ICL", "gemma-2-9b-it", "gemma-2-9b-it_ICL", "Llama-2-7b-hf", "Llama-2-7b-hf_ICL", "Llama-2-7b-chat-hf", "Llama-2-7b-chat-hf_ICL", "Llama-2-13b-chat-hf", "Llama-2-13b-chat-hf_ICL", "meditron-7b", "meditron-7b_ICL", "MedLLaMA_13B", "MedLLaMA_13B_ICL", "medllama3-v20", "medllama3-v20_ICL", "vicuna-7b-v1.5", "vicuna-7b-v1.5_ICL", "vicuna-13b-v1.5", "vicuna-13b-v1.5_ICL", "Qwen2-0.5B-Instruct", "Qwen2-1.5B-Instruct", "Qwen2-7B-Instruct", "Yi-1.5-6B-Chat", "Yi-1.5-9B-Chat", "HuatuoGPT2-7B", "HuatuoGPT2-7B_ICL", "Baichuan2-7B-Chat", "Baichuan2-13B-Chat", "XGBoost6", "XGBoost8", "XGBoost10", "DecisionTree6", "DecisionTree8", "DecisionTree10", "Llama3-Med42-8B", "Llama3-Med42-8B_ICL", "XGBoost2000", "LogisticRegression2000", "DecisionTree2000", "RandomForest2000", "AdaBoost2000", "SVM2000", "NaiveBayes2000", "KNN2000", "NeuralNetwork2000"]
# method = "LR"

for method in mor_list:
    y_true = []
    y_pred = []
    sensitive = []
    sensitive2 = []
    sensitive3 = []
    sensitive4 = []
    if method in ["Transformer", "RNN","Transformer1", "RNN1","Transformer2", "RNN2","Transformer5", "RNN5", "RNNtest"]:
        with open(f"/home/peter/files/PyHealth/nlp_task/{task}/{task}_{method}_result_sample.csv", "r") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                y_true.append(int(row["ANSWER"]))
                y_pred.append(int(row["PREDICTION"]))
                sen = "F"
                mode = 1
                sensitive.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "BLACK"
                mode = 2
                sensitive2.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "ASIAN"
                mode = 2
                sensitive3.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "WHITE"
                mode = 2
                sensitive4.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
    else:
        with open(f"/home/peter/files/PyHealth/nlp_task/{task}/{task}_result_data_{method}.csv", "r") as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                y_true.append(int(row["ANSWER"]))
                if row["PREDICTION"] == "1" or row["PREDICTION"] == "0":
                    y_pred.append(int(row["PREDICTION"]))
                else:
                    y_pred.append(1 - int(row['ANSWER']))
                    
                sen = "F"
                mode = 1
                sensitive.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "BLACK"
                mode = 2
                sensitive2.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "ASIAN"
                mode = 2
                sensitive3.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)
                sen = "WHITE"
                mode = 2
                sensitive4.append(1 if patient_dic[row["SUBJECT_ID"]][mode][:5] == sen else 0)


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    sensitive2 = np.array(sensitive2)
    sensitive3 = np.array(sensitive3)
    sensitive4 = np.array(sensitive4)


    metrics = calculate_metrics(y_true, y_pred, sensitive)
    # 保留3位小数
    for key in metrics.keys():
        metrics[key] = round(metrics[key], 3)
    values = [f"{round(value, 3)}" for value in metrics.values()]
    output = " ".join(values)
    # print(method)
    print(output)
    # print(metrics)


