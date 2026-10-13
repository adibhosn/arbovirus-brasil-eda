"""Verificação de sanidade: baselines simples em subamostra do treino.

Não é o modelo final — serve para (1) validar que os arquivos processados
carregam e treinam, (2) dar uma referência de desempenho para os colegas e
(3) detectar vazamento (acurácia/AUC irreais).
"""
import sys, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from config import DATA_DIR, SEED

cen = sys.argv[1] if len(sys.argv) > 1 else "binario"
D = DATA_DIR / "processed" / cen
tr = pd.read_parquet(D / "train_tree.parquet").sample(300_000, random_state=SEED)
te = pd.read_parquet(D / "test_tree.parquet").sample(100_000, random_state=SEED)
Xtr, ytr, Xte, yte = tr.drop(columns="y"), tr.y, te.drop(columns="y"), te.y
res = {}
for nome, m in {"logreg": LogisticRegression(max_iter=300, class_weight="balanced"),
                "rf": RandomForestClassifier(150, min_samples_leaf=20, n_jobs=-1, class_weight="balanced", random_state=SEED)}.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); pr = m.predict_proba(Xte)
    res[nome] = {"bal_acc": round(balanced_accuracy_score(yte, p), 4), "f1_macro": round(f1_score(yte, p, average="macro"), 4),
                 "auc": round(roc_auc_score(yte, pr[:, 1] if pr.shape[1] == 2 else pr, multi_class="ovr"), 4)}
    if nome == "rf":
        imp = pd.Series(m.feature_importances_, Xtr.columns).sort_values(ascending=False).head(10).round(4)
        res["rf_top10"] = imp.to_dict()
print(cen, json.dumps(res, indent=1))
