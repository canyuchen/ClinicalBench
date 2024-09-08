from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import csv
import numpy as np

def calculate_f1_aucroc(model, dataset, random_index, task):
    with open(f"/home/peter/files/PyHealth/nlp_task/{task}/{dataset}/{task}_result_data_{model}_{random_index}.csv", "r") as f:
        csvreader = csv.DictReader(f)
        answer = []
        predict = []
        prob = []
        for row in csvreader:
            answer.append(int(row["ANSWER"]))
            predict.append(int(row["PREDICTION"]))
            prob.append(float(row["PROB"]))
        if task == "length_pred":
            for i in range(len(predict)):
                if predict[i] not in [1, 2, 3]:
                    # set wrong
                    predict[i] = 1 if answer[i] == 1 else 2
            f1 = f1_score(answer, predict, average="macro")
            temp_answer = []
            for a in answer:
                if a in [1, 2]:
                    temp_answer.append(0)
                else:
                    temp_answer.append(1)
            aucroc = roc_auc_score(temp_answer, prob)
        else:
            for i in range(len(predict)):
                if predict[i] not in [0, 1]:
                    predict[i] = 1 - answer[i]
            f1 = f1_score(answer, predict)
            aucroc = roc_auc_score(answer, prob)
    return f1, aucroc

def calculate95margin(results):
    results = np.array(results)
    mean = np.mean(results)
    std_dev = np.std(results, ddof=1)
    n = len(results)
    t = 0.2776
    margin_of_error = t * (std_dev / np.sqrt(n))
    mean = mean * 100
    margin_of_error = margin_of_error * 100
    mean = round(mean, 2)
    margin_of_error = round(margin_of_error, 2)
    return mean, margin_of_error


# model_list = ["Transformer", "RNN", "XGBoost", "LogisticRegression", "DecisionTree", "RandomForest", "AdaBoost", "SVM", "NaiveBayes", "KNN", "NeuralNetwork", "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.3","gemma-2-9b-it", "Qwen2-7B-Instruct", "Yi-1.5-9B-Chat", "vicuna-7b-v1.5", "vicuna-13b-v1.5",  "MedLLaMA_13B", "meditron-7b", "medllama3-v20", "BioMistral-7B", "Llama3-Med42-8B", "BioMedGPT-LM-7B", "base-7b-v0.2"]
# model_list = ["meditron-7b", "medllama3-v20", "BioMistral-7B", "Llama3-Med42-8B", "BioMedGPT-LM-7B", "base-7b-v0.2"]
# model_list = ["Qwen2-7B-Instruct", "Yi-1.5-9B-Chat", "vicuna-7b-v1.5", "meditron-7b", "medllama3-v20", "BioMistral-7B", "Llama3-Med42-8B", "BioMedGPT-LM-7B", "base-7b-v0.2"]
# for model in model_list:
#     # task: length_pred
#     task = "mortality_pred"
#     dataset = "mimic3"
#     f1s = []
#     aucrocs = []
#     for random_index in range(5):
#         f1, aucroc = calculate_f1_aucroc(model, dataset, random_index, task)
#         f1s.append(f1)
#         aucrocs.append(aucroc)
#     f1_mean, f1_margin = calculate95margin(f1s)
#     aucroc_mean, aucroc_margin = calculate95margin(aucrocs)
#     print(f1_mean, f1_margin, aucroc_mean, aucroc_margin)




# length
# prediction = []
# answer = []
# prob = []
# for i in range(4000):
#     answer.append(1)
# for i in range(651):
#     answer.append(2)
# for i in range(369):
#     answer.append(3)

# for i in range(4000+651+369):
#     prediction.append(3)

# for i in range(4000+651+369):
#     prob.append(1)

# f1 = f1_score(answer, prediction, average="macro")
# temp_answer = []
# for a in answer:
#     if a in [1, 2]:
#         temp_answer.append(0)
#     else:
#         temp_answer.append(1)
# aucroc = roc_auc_score(temp_answer, prob)

# print(f1, aucroc)

# f1s = [f1 for i in range(5)]
# aucrocs = [aucroc for i in range(5)]

# print(calculate95margin(f1s))
# print(calculate95margin(aucrocs))
        


# mortality / readmission
prediction = []
answer = []
prob = []
for i in range(4000):
    answer.append(0)
for i in range(664):
    answer.append(1)

for i in range(4000+664):
    prediction.append(1)

for i in range(4000+664):
    prob.append(1)

f1 = f1_score(answer, prediction)
aucroc = roc_auc_score(answer, prob)

print(f1, aucroc)

f1s = [f1 for i in range(5)]
aucrocs = [aucroc for i in range(5)]

print(calculate95margin(f1s))
print(calculate95margin(aucrocs))
