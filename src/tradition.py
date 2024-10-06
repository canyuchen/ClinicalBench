from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse





def process_data(data):
        for i in range(len(data)):
            data[i] = data[i][0]
            temp = ''
            for j in range(len(data[i])):
                temp += str(data[i][j]) + ' '
            data[i] = temp
      
def adjust_labels(labels):
    return labels - 1

def train_validate_and_evaluate(model, model_name, train_features, train_labels, val_features, val_labels, test_features, test_labels, task):
    print(f"\nTraining and evaluating {model_name}...")

    best_val_f1 = -1
    best_model = None

    for i in range(20): 
        current_model = clone(model)
        current_model.random_state = i 
        current_model.fit(train_features, train_labels)
        val_pred = current_model.predict(val_features)
        if task == 'length_pred':
            val_f1 = f1_score(val_labels, val_pred, average='macro')
        else:
            val_f1 = f1_score(val_labels, val_pred)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = current_model

    print(f'{model_name} Best Validation f1: {best_val_f1:.4f}')

    y_pred = best_model.predict(test_features)
    test_f1 = f1_score(test_labels, y_pred, average='macro')
    print(f'{model_name} Test f1: {test_f1:.4f}')
    
    return best_model, best_val_f1, test_f1

def train_dl_model(model, train_features, train_labels, val_features, val_labels, task, batch_size=32, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(val_features).to(device))
            _, val_preds = torch.max(val_outputs, 1)
            if task == 'length_pred':
                val_f1 = f1_score(val_labels, val_preds.cpu().numpy(), average='macro')
            else:
                val_f1 = f1_score(val_labels, val_preds.cpu().numpy())
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation f1: {val_f1:.4f}')
    
    model.load_state_dict(best_model_state)
    return model, best_val_f1

def evaluate_dl_model(model, test_features, test_labels, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.FloatTensor(test_features).to(device))
        _, test_preds = torch.max(test_outputs, 1)
        test_probs = torch.softmax(test_outputs, dim=1)[:, -1].cpu().numpy()
        if task == 'length_pred':
            test_f1 = f1_score(test_labels, test_preds.cpu().numpy(), average='macro')
        else:
            test_f1 = f1_score(test_labels, test_preds.cpu().numpy())
    return test_f1, test_preds.cpu().numpy(), test_probs

def get_data_and_convert_to_features(task, dataset, random_index, ratio):

    with open(f"data/{task}/{dataset}/{task}_data.json", "r") as f:
        data = json.load(f)

    with open(f"data/{task}/{dataset}/train_index_{random_index}.npy", "rb") as f:
        train_index = np.load(f)
    with open(f"data/{task}/{dataset}/val_index_{random_index}.npy", "rb") as f:
        val_index = np.load(f)
    with open(f"data/{task}/{dataset}/test_index_{random_index}.npy", "rb") as f:
        test_index = np.load(f)

    print(f"Train: {len(train_index)}, Val: {len(val_index)}, Test: {len(test_index)}")
    train_conditions = []
    train_procedures = []
    train_drugs = []
    train_labels = []
    val_conditions = []
    val_procedures = []
    val_drugs = []
    val_labels = []
    test_conditions = []
    test_procedures = []
    test_drugs = []
    test_labels = []


    # length pred: 1, 2, 3. mortality pred: 0, 1. readmission pred: 0, 1
    number0 = 0
    number1 = 0
    number2 = 0
    number3 = 0

    for i in range(len(data)):
        if data[i]["visit_id"] in train_index:
            if task == "length_pred":
                if data[i]["label"] == 1:
                    number1 += 1
                elif data[i]["label"] == 2:
                    number2 += 1
                elif data[i]["label"] == 3:
                    number3 += 1
            else:
                if data[i]["label"] == 0:
                    number0 += 1
                elif data[i]["label"] == 1:
                    number1 += 1

    number0 = number0 * ratio
    number1 = number1 * ratio
    number2 = number2 * ratio
    number3 = number3 * ratio
        

    for i in range(len(data)):
        if data[i]["visit_id"] in train_index:
            if data[i]["label"] == 0 and number0 > 0:
                train_conditions.append(data[i]["conditions"])
                train_procedures.append(data[i]["procedures"])
                train_drugs.append(data[i]["drugs"])
                train_labels.append(data[i]["label"])
                number0 -= 1
            elif data[i]["label"] == 1 and number1 > 0:
                train_conditions.append(data[i]["conditions"])
                train_procedures.append(data[i]["procedures"])
                train_drugs.append(data[i]["drugs"])
                train_labels.append(data[i]["label"])
                number1 -= 1
            elif data[i]["label"] == 2 and number2 > 0:
                train_conditions.append(data[i]["conditions"])
                train_procedures.append(data[i]["procedures"])
                train_drugs.append(data[i]["drugs"])
                train_labels.append(data[i]["label"])
                number2 -= 1
            elif data[i]["label"] == 3 and number3 > 0:
                train_conditions.append(data[i]["conditions"])
                train_procedures.append(data[i]["procedures"])
                train_drugs.append(data[i]["drugs"])
                train_labels.append(data[i]["label"])
                number3 -= 1

        elif data[i]["visit_id"] in val_index:
            val_conditions.append(data[i]["conditions"])
            val_procedures.append(data[i]["procedures"])
            val_drugs.append(data[i]["drugs"])
            val_labels.append(data[i]["label"])
        elif data[i]["visit_id"] in test_index:
            test_conditions.append(data[i]["conditions"])
            test_procedures.append(data[i]["procedures"])
            test_drugs.append(data[i]["drugs"])
            test_labels.append(data[i]["label"])

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)



    process_data(train_conditions)
    process_data(train_procedures)
    process_data(train_drugs)
    process_data(val_conditions)
    process_data(val_procedures)
    process_data(val_drugs)
    process_data(test_conditions)
    process_data(test_procedures)
    process_data(test_drugs)

    # Use CountVectorizer to convert text data to vectors
    vectorizer_conditions = CountVectorizer(max_features=2000)
    vectorizer_procedures = CountVectorizer(max_features=2000)
    vectorizer_drugs = CountVectorizer(max_features=2000)

    # Fit and transform the data
    train_conditions_vec = vectorizer_conditions.fit_transform(train_conditions)
    train_procedures_vec = vectorizer_procedures.fit_transform(train_procedures)
    train_drugs_vec = vectorizer_drugs.fit_transform(train_drugs)
    val_conditions_vec = vectorizer_conditions.transform(val_conditions)
    val_procedures_vec = vectorizer_procedures.transform(val_procedures)
    val_drugs_vec = vectorizer_drugs.transform(val_drugs)
    test_conditions_vec = vectorizer_conditions.transform(test_conditions)
    test_procedures_vec = vectorizer_procedures.transform(test_procedures)
    test_drugs_vec = vectorizer_drugs.transform(test_drugs)

    # concatenate the features
    train_features = np.hstack([train_conditions_vec.toarray(), train_procedures_vec.toarray(), train_drugs_vec.toarray()])
    val_features = np.hstack([val_conditions_vec.toarray(), val_procedures_vec.toarray(), val_drugs_vec.toarray()])
    test_features = np.hstack([test_conditions_vec.toarray(), test_procedures_vec.toarray(), test_drugs_vec.toarray()])

    if task == 'length_pred':
        train_labels = adjust_labels(train_labels)
        val_labels = adjust_labels(val_labels)
        test_labels = adjust_labels(test_labels)

    print(train_features.shape, val_features.shape, test_features.shape)
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out





def main(args):
    task = args.task
    dataset = args.dataset
    random_index = args.random_index
    ratio = args.ratio

    train_features, train_labels, val_features, val_labels, test_features, test_labels = get_data_and_convert_to_features(task, dataset, random_index, ratio)

    # define models
    models = {
        'Transformer': TransformerModel(train_features.shape[1], len(np.unique(train_labels))),
        'RNN': RNNModel(train_features.shape[1], hidden_dim=512, num_layers=2, num_classes=len(np.unique(train_labels))),
        'XGBoost': xgb.XGBClassifier(max_depth=6, objective='multi:softmax', num_class=3) if task == 'length_pred' else xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=3),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=6),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf', probability=True),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        
    }

    # train
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        
        if isinstance(model, (TransformerModel, RNNModel)):
            # deep learning model
            best_model, val_f1 = train_dl_model(model, train_features, train_labels, val_features, val_labels, task)
            test_f1, test_predictions, test_probs = evaluate_dl_model(best_model, test_features, test_labels, task)
        else:
            # machine learning model
            best_model, val_f1, test_f1 = train_validate_and_evaluate(
                model, model_name, train_features, train_labels, 
                val_features, val_labels, test_features, test_labels, task
            )
            test_predictions = best_model.predict(test_features)
            test_probs = best_model.predict_proba(test_features)[:, -1]
            if task == 'length_pred':
                test_f1 = f1_score(test_labels, test_predictions, average='macro')
            else:
                test_f1 = f1_score(test_labels, test_predictions)
        
        results[model_name] = (best_model, val_f1, test_f1)
        with open(f'results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}_{ratio}.csv', 'w') as file:
            filenames = ['ANSWER', 'PREDICTION', 'PROB']
            writer = csv.DictWriter(file, fieldnames=filenames)
            writer.writeheader()
            for true_label, pred_label, prob in zip(test_labels, test_predictions, test_probs):
                if task == 'length_pred':
                    writer.writerow({
                        'ANSWER': int(true_label) + 1,
                        'PREDICTION': int(pred_label) + 1,
                        'PROB': prob
                    })
                else:
                    writer.writerow({
                        'ANSWER': int(true_label),
                        'PREDICTION': int(pred_label),
                        'PROB': prob
                    })


if __name__ == "__main__":
    pragma = argparse.ArgumentParser()
    pragma.add_argument("--task", type=str, default="mortality_pred")
    pragma.add_argument("--dataset", type=str, default="mimic4")
    pragma.add_argument("--random_index", type=int, default=6)
    pragma.add_argument("--ratio", type=float, default=1)

    args = pragma.parse_args()
    main(args)
