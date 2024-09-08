from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import csv

for model in ["Meta-Llama-3-8B-Instruct", "Meta-Llama-3-8B"]:
    with open(f"/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_result_data_{model}.csv", "r") as f:
        csvreader = csv.DictReader(f)
        answer = []
        prediction = []
        breakpoint()
        for row in csvreader:
            answer.append(int(row["ANSWER"]))
            prediction.append(int(row["PREDICTION"]))

        accuracy = accuracy_score(answer, prediction)
        f1 = f1_score(answer, prediction)
        recall = recall_score(answer, prediction)
        precision = precision_score(answer, prediction)
        print(f"Model: {model}")
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
