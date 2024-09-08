from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import length_of_stay_prediction_mimic3_fn
from sklearn.feature_extraction.text import CountVectorizer
from pyhealth.datasets.splitter import split_by_patient, split_by_sample
from pyhealth.datasets import split_by_patient, get_dataloader
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


dataset = "mimic4"


def process_data(data):
        for i in range(len(data)):
            data[i] = data[i][0]
            temp = ''
            for j in range(len(data[i])):
                temp += str(data[i][j]) + ' '
            data[i] = temp
      
def adjust_labels(labels):
    return labels - 1


# 训练和验证机器学习模型的函数
def train_validate_and_evaluate(model, model_name, train_features, train_labels, val_features, val_labels, test_features, test_labels):
    print(f"\nTraining and evaluating {model_name}...")

    best_val_accuracy = 0
    best_model = None

    for i in range(5):  # 可以调整迭代次数
        current_model = clone(model)
        current_model.random_state = i  # 每次设置不同的随机状态
        
        # 在训练集上拟合模型
        current_model.fit(train_features, train_labels)
        
        # 验证模型
        val_pred = current_model.predict(val_features)
        # val_accuracy = accuracy_score(val_labels, val_pred)
        val_accuracy = f1_score(val_labels, val_pred, average='macro')
        
        # 如果这个模型在验证集上表现更好，保存它
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = current_model

    print(f'{model_name} Best Validation f1: {best_val_accuracy:.4f}')

    # 使用最佳模型（基于验证准确率）在测试集上预测
    y_pred = best_model.predict(test_features)
    test_accuracy = f1_score(test_labels, y_pred, average='macro')
    print(f'{model_name} Test f1: {test_accuracy:.4f}')
    
    return best_model, best_val_accuracy, test_accuracy

# 训练深度学习模型的函数
def train_dl_model(model, train_features, train_labels, val_features, val_labels, batch_size=32, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_val_accuracy = 0
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
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(val_features).to(device))
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = f1_score(val_labels, val_preds.cpu().numpy(), average='macro')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    return model, best_val_accuracy


# 评估深度学习模型的函数
def evaluate_dl_model(model, test_features, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.FloatTensor(test_features).to(device))
        _, test_preds = torch.max(test_outputs, 1)
        test_probs = torch.softmax(test_outputs, dim=1)[:, -1].cpu().numpy()
        test_accuracy = f1_score(test_labels, test_preds.cpu().numpy(), average='macro')
    return test_accuracy, test_preds.cpu().numpy(), test_probs

random_index = 0


for random_index in range(5):
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data.json", "r") as f:
        data = json.load(f)

    # 创建数据加载器
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/train_index_{random_index}.npy", "rb") as f:
        train_index = np.load(f)
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/val_index_{random_index}.npy", "rb") as f:
        val_index = np.load(f)
    with open(f"/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/test_index_{random_index}.npy", "rb") as f:
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


    for i in range(len(data)):
        if data[i]["visit_id"] in train_index:
            train_conditions.append(data[i]["conditions"])
            train_procedures.append(data[i]["procedures"])
            train_drugs.append(data[i]["drugs"])
            train_labels.append(data[i]["label"])
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



    # def extract_features(loader):
    #     all_conditions = []
    #     all_procedures = []
    #     all_drugs = []
    #     all_labels = []
    #     patient_ids = []
        
    #     for batch in loader:
    #         all_conditions.extend([' '.join(c[0]) for c in batch['conditions']])
    #         all_procedures.extend([' '.join([str(p) for p in proc]) for proc in batch['procedures']])
    #         all_drugs.extend([' '.join(d[0]) for d in batch['drugs']])
    #         if isinstance(batch['label'], list):
    #             all_labels.extend(batch['label'])
    #         else:
    #             all_labels.extend(batch['label'].numpy())
    #         patient_ids.extend(batch['patient_id'])
        
    #     return all_conditions, all_procedures, all_drugs, np.array(all_labels), patient_ids

    # # 提取训练、验证和测试数据的特征
    # train_conditions, train_procedures, train_drugs, train_labels, train_ids = extract_features(train_loader)
    # val_conditions, val_procedures, val_drugs, val_labels, val_ids = extract_features(val_loader)
    # test_conditions, test_procedures, test_drugs, test_labels, test_ids = extract_features(test_loader)



    # 使用CountVectorizer将文本特征转换为数值特征
    vectorizer_conditions = CountVectorizer(max_features=2000)
    vectorizer_procedures = CountVectorizer(max_features=2000)
    vectorizer_drugs = CountVectorizer(max_features=2000)

    # 拟合并转换训练数据
    train_conditions_vec = vectorizer_conditions.fit_transform(train_conditions)
    train_procedures_vec = vectorizer_procedures.fit_transform(train_procedures)
    train_drugs_vec = vectorizer_drugs.fit_transform(train_drugs)

    # 转换验证数据
    val_conditions_vec = vectorizer_conditions.transform(val_conditions)
    val_procedures_vec = vectorizer_procedures.transform(val_procedures)
    val_drugs_vec = vectorizer_drugs.transform(val_drugs)

    # 转换测试数据
    test_conditions_vec = vectorizer_conditions.transform(test_conditions)
    test_procedures_vec = vectorizer_procedures.transform(test_procedures)
    test_drugs_vec = vectorizer_drugs.transform(test_drugs)

    # 将特征拼接在一起
    train_features = np.hstack([train_conditions_vec.toarray(), train_procedures_vec.toarray(), train_drugs_vec.toarray()])
    val_features = np.hstack([val_conditions_vec.toarray(), val_procedures_vec.toarray(), val_drugs_vec.toarray()])
    test_features = np.hstack([test_conditions_vec.toarray(), test_procedures_vec.toarray(), test_drugs_vec.toarray()])


    train_labels = adjust_labels(train_labels)
    val_labels = adjust_labels(val_labels)
    test_labels = adjust_labels(test_labels)

    # 定义Transformer模型
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

    # 定义RNN模型
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


    # 定义所有模型（包括机器学习和深度学习模型）
    models = {
        'Transformer': TransformerModel(train_features.shape[1], len(np.unique(train_labels))),
        'RNN': RNNModel(train_features.shape[1], hidden_dim=512, num_layers=2, num_classes=len(np.unique(train_labels))),
        'XGBoost': xgb.XGBClassifier(max_depth=6, objective='multi:softmax', num_class=3),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(max_depth=6),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'SVM': SVC(kernel='rbf', probability=True),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
        
    }

    # 训练、验证和评估每个模型
    results = {}
    for model_name, model in models.items():
        print(f"\nTraining and evaluating {model_name}...")
        
        if isinstance(model, (TransformerModel, RNNModel)):
            # 深度学习模型
            best_model, val_accuracy = train_dl_model(model, train_features, train_labels, val_features, val_labels)
            test_accuracy, test_predictions, test_probs = evaluate_dl_model(best_model, test_features, test_labels)
        else:
            # 机器学习模型
            best_model, val_accuracy, test_accuracy = train_validate_and_evaluate(
                model, model_name, train_features, train_labels, 
                val_features, val_labels, test_features, test_labels,
            )
            test_predictions = best_model.predict(test_features)
            test_probs = best_model.predict_proba(test_features)[:, -1]
        
        results[model_name] = (best_model, val_accuracy, test_accuracy)
    
        
        # 将结果写入CSV文件
        with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_result_data_{model_name}_{random_index}.csv', 'w') as file:
            filenames = ['ANSWER', 'PREDICTION', 'PROB']
            writer = csv.DictWriter(file, fieldnames=filenames)
            writer.writeheader()
            for true_label, pred_label, prob in zip(test_labels, test_predictions, test_probs):
                writer.writerow({
                    'ANSWER': int(true_label) + 1,  # 加1以转换回原始标签
                    'PREDICTION': int(pred_label) + 1,  # 加1以转换回原始标签
                    'PROB': prob
                })
