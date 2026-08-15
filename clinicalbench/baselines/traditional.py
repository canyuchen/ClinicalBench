"""Traditional ML baselines.

Eleven models over bag-of-codes features. Each sklearn model is refit under
several seeds and the checkpoint with the best validation F1 is the one scored
on test; the two torch models train for 20 epochs under the same rule.

Usage::

    python -m clinicalbench.baselines.traditional --task mortality_pred --dataset mimic3 \\
        --random_index 0 [--ratio 0.4] [--models XGBoost SVM]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

from clinicalbench.baselines.features import build_features
from clinicalbench.config import DATASETS, TASK_SPECS, TASKS
from clinicalbench.naming import result_path

#: Deterministic estimators, or ones whose seed does not change the fit here --
#: refitting them under more seeds would just repeat the same model.
SINGLE_FIT_MODELS = frozenset(
    {"XGBoost", "LogisticRegression", "AdaBoost", "SVM", "NaiveBayes", "KNN"}
)
SEED_SWEEP = 20
DL_EPOCHS = 20


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = x + self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc(x.mean(dim=1))


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        return self.fc(out[:, -1, :])


def build_models(input_dim: int, num_classes: int, task: str) -> Dict[str, object]:
    if task == "length_pred":
        booster = xgb.XGBClassifier(max_depth=6, objective="multi:softmax", num_class=num_classes)
    else:
        booster = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=3)
    return {
        "Transformer": TransformerModel(input_dim, num_classes),
        "RNN": RNNModel(input_dim, hidden_dim=512, num_layers=2, num_classes=num_classes),
        "XGBoost": booster,
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=6),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "SVM": SVC(kernel="rbf", probability=True),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    }


def _f1(y_true, y_pred, task: str) -> float:
    return f1_score(y_true, y_pred, average=TASK_SPECS[task].average)


def fit_sklearn(model, name, Xtr, ytr, Xva, yva, task):
    """Refit under several seeds, keep the best model by validation F1."""
    n_seeds = 1 if name in SINGLE_FIT_MODELS else SEED_SWEEP
    best_f1, best_model = -1.0, None
    for seed in range(n_seeds):
        candidate = clone(model)
        # assigned rather than passed to set_params: several of these estimators
        # (KNN, GaussianNB) have no random_state and set_params would raise
        if "random_state" in candidate.get_params():
            candidate.random_state = seed
        candidate.fit(Xtr, ytr)
        val_f1 = _f1(yva, candidate.predict(Xva), task)
        if val_f1 > best_f1:
            best_f1, best_model = val_f1, candidate
    return best_model, best_f1


def fit_torch(model, Xtr, ytr, Xva, yva, task, batch_size=32, epochs=DL_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)),
        batch_size=batch_size, shuffle=True,
    )
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-4)
    best_f1, best_state = -1.0, None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(Xva).to(device)).argmax(dim=1).cpu().numpy()
        val_f1 = _f1(yva, preds, task)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_f1


def predict_torch(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X).to(device))
        return (
            logits.argmax(dim=1).cpu().numpy(),
            torch.softmax(logits, dim=1)[:, -1].cpu().numpy(),
        )


def write_results(path: Path, task: str, y_true, y_pred, probs) -> None:
    offset = 1 if TASK_SPECS[task].is_multiclass else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ANSWER", "PREDICTION", "PROB"])
        writer.writeheader()
        for true, pred, prob in zip(y_true, y_pred, probs):
            writer.writerow({
                "ANSWER": int(true) + offset,
                "PREDICTION": int(pred) + offset,
                "PROB": prob,
            })


def run(args) -> None:
    Xtr, ytr, Xva, yva, Xte, yte = build_features(
        args.task, args.dataset, args.random_index, args.ratio, args.data_root
    )
    print(f"features {Xtr.shape[1]}  train {len(ytr)}  val {len(yva)}  test {len(yte)}")

    models = build_models(Xtr.shape[1], len(np.unique(ytr)), args.task)
    selected = args.models or list(models)

    for name in selected:
        if name not in models:
            raise SystemExit(f"unknown model {name!r}; choose from {list(models)}")
        model = models[name]
        print(f"\n-- {name}")
        if isinstance(model, (TransformerModel, RNNModel)):
            model, val_f1 = fit_torch(model, Xtr, ytr, Xva, yva, args.task)
            preds, probs = predict_torch(model, Xte)
        else:
            model, val_f1 = fit_sklearn(model, name, Xtr, ytr, Xva, yva, args.task)
            preds = model.predict(Xte)
            probs = model.predict_proba(Xte)[:, -1]

        test_f1 = _f1(yte, preds, args.task)
        out = result_path(
            args.result_root, args.task, args.dataset, name,
            args.random_index, ratio=args.ratio,
        )
        write_results(out, args.task, yte, preds, probs)
        print(f"   val F1 {val_f1 * 100:.2f}   test F1 {test_f1 * 100:.2f}   -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--random_index", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="fraction of the training set to use")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of baselines to run (default: all 11)")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
