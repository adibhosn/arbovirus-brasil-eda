"""Limpeza, engenharia de features, split e escalonamento.

Uso:
    python 02_preprocess.py --cenario binario      # Dengue vs Descartado (padrão)
    python 02_preprocess.py --cenario multiclasse  # Dengue / Chikungunya / Descartado

Saída em data/processed/<cenario>/:
    {train,val,test}_tree.parquet    -> sem escala        (Random Forest)
    {train,val,test}_std.parquet     -> padronizado       (Rede Neural)
    {train,val,test}_minmax.parquet  -> escala [0,1]      (Fuzzy)
    feature_names.json, scalers.joblib, resumo.json
A coluna-alvo se chama `y`. Todas as estatísticas (medianas, médias, min/max)
são calculadas SOMENTE no treino e aplicadas a val/test (sem vazamento).
Decisões documentadas em docs/DECISOES.md.
"""
import argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import RAW_PARQUET, DATA_DIR, TARGET, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--cenario", choices=["binario", "multiclasse"], default="binario")
args = ap.parse_args()
OUT = DATA_DIR / "processed" / args.cenario
OUT.mkdir(parents=True, exist_ok=True)

SINTOMAS = ["FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA", "DOR_COSTAS", "CONJUNTVIT",
            "ARTRITE", "ARTRALGIA", "PETEQUIA_N", "LEUCOPENIA", "LACO", "DOR_RETRO"]
COMORB = ["DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "HIPERTENSA", "ACIDO_PEPT", "AUTO_IMUNE"]
BIN_COLS = SINTOMAS + COMORB
log = {}

df = pd.read_parquet(RAW_PARQUET)
log["linhas_brutas"] = len(df)

# 1. Linhas inválidas: cabeçalhos repetidos (alvo == 'CLASSI_FIN') e anos fora de 2013-2020
df = df[df[TARGET].isin(["Dengue", "Chikungunya", "Discarded/Inconclusive"])]
df = df[df.NU_ANO.between(2013, 2020)]
log["apos_remover_invalidas"] = len(df)

# 2. Duplicatas exatas (antes do split, para não haver cópia entre treino e teste)
df = df.drop_duplicates().reset_index(drop=True)
log["apos_dedup"] = len(df)

# 3. Ficha clínica preenchida? As 21 colunas são todas 9 ou nenhuma é.
sem_ficha = df[BIN_COLS].eq(9).all(axis=1)
if args.cenario == "binario":
    df = df[~sem_ficha & df[TARGET].ne("Chikungunya")].reset_index(drop=True)
log["apos_filtro_cenario"] = len(df)

# 4. Alvo
if args.cenario == "binario":
    y = (df[TARGET] == "Dengue").astype("int8")          # 1 = Dengue, 0 = Descartado
    classes = {"0": "Discarded/Inconclusive", "1": "Dengue"}
else:
    classes = {"0": "Discarded/Inconclusive", "1": "Dengue", "2": "Chikungunya"}
    y = df[TARGET].map({v: int(k) for k, v in classes.items()}).astype("int8")

# 5. Features
X = pd.DataFrame(index=df.index)

# 5a. Binárias: 1=sim -> 1 ; 2=não -> 0 ; 9=ignorado -> 0 (+ flag no cenário multiclasse)
for c in BIN_COLS:
    X[c] = df[c].eq(1).astype("int8")
if args.cenario == "multiclasse":
    X["sem_ficha_clinica"] = df[BIN_COLS].eq(9).all(axis=1).astype("int8")

# 5b. Numéricas
idade = df.NU_IDADE_N.where(df.NU_IDADE_N.between(0, 110))      # 999, 8xx, 5xx -> NaN
X["idade"] = idade
dias = (df.DT_NOTIFIC - df.DT_SIN_PRI).dt.days
X["dias_sintoma_notif"] = dias.where(dias.between(0, 30))       # datas absurdas -> NaN
mes = df.DT_NOTIFIC.dt.month
X["mes_sin"] = np.sin(2 * np.pi * mes / 12)
X["mes_cos"] = np.cos(2 * np.pi * mes / 12)
NUM_COLS = ["idade", "dias_sintoma_notif"]   # as únicas que precisam de imputação/escala
SCALE_COLS = NUM_COLS

# 5c. Categóricas nominais -> one-hot
def onehot(series, prefix, mapping=None):
    s = series.astype("Int64") if mapping is None else series.map(mapping)
    return pd.get_dummies(s.astype(str), prefix=prefix, dtype="int8")

# sexo: 0 e 1 viram colunas próprias; códigos raros (2, 9) são consolidados em "sexo_ign"
sexo = df.CS_SEXO.map({0: "0", 1: "1"}).fillna("ign")
X = X.join(pd.get_dummies(sexo, prefix="sexo", dtype="int8"))
gest = df.CS_GESTANT.map({1: "sim", 2: "sim", 3: "sim", 4: "sim", 5: "nao", 6: "na", 9: "ign"})
X = X.join(pd.get_dummies(gest, prefix="gest", dtype="int8"))
X = X.join(onehot(df.CS_RACA, "raca"))
X = X.join(onehot(df.CS_ESCOL_N, "escol"))
X = X.join(onehot(df.HOSPITALIZ, "hosp"))
regiao = (df.SG_UF_NOT // 10).map({1: "N", 2: "NE", 3: "SE", 4: "S", 5: "CO"})
X = X.join(pd.get_dummies(regiao, prefix="reg", dtype="int8"))

# Em colunas dummies, categorias raras podem não existir; garantimos schema estável abaixo
X = X.loc[:, ~X.columns.duplicated()]

# 6. Split estratificado 70/15/15
idx_tr, idx_tmp = train_test_split(X.index, test_size=0.30, stratify=y, random_state=SEED)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y.loc[idx_tmp], random_state=SEED)
splits = {"train": idx_tr, "val": idx_va, "test": idx_te}

# 7. Imputação (mediana do treino) + indicadores de faltante
med = X.loc[idx_tr, NUM_COLS].median()
# n_imputados: quantos campos numéricos do registro eram inválidos/ausentes (0 a 2)
X["n_imputados"] = X[NUM_COLS].isna().sum(axis=1).astype("int8")
for c in NUM_COLS:
    X[c] = X[c].fillna(med[c])
X = X.astype("float32")
log["imputacao_medianas"] = med.to_dict()
assert not X.isna().any().any()

# 8. Escalonamento (ajustado no treino)
std = StandardScaler().fit(X.loc[idx_tr, SCALE_COLS])
mm = MinMaxScaler().fit(X.loc[idx_tr, SCALE_COLS])
joblib.dump({"standard": std, "minmax": mm, "cols": SCALE_COLS, "medianas": med}, OUT / "scalers.joblib")

def variant(frame, kind):
    f = frame.copy()
    if kind == "std":
        f[SCALE_COLS] = std.transform(f[SCALE_COLS])
    elif kind == "minmax":
        f[SCALE_COLS] = np.clip(mm.transform(f[SCALE_COLS]), 0, 1)
    return f

for name, idx in splits.items():
    for kind in ["tree", "std", "minmax"]:
        f = variant(X.loc[idx], kind)
        f["y"] = y.loc[idx].values
        f.reset_index(drop=True).to_parquet(OUT / f"{name}_{kind}.parquet", index=False)

json.dump(list(X.columns), open(OUT / "feature_names.json", "w"), indent=1)
log["n_features"] = X.shape[1]
log["classes"] = classes
log["tamanhos"] = {k: len(v) for k, v in splits.items()}
log["distribuicao_y"] = {k: y.loc[v].value_counts(normalize=True).round(4).to_dict() for k, v in splits.items()}
json.dump(log, open(OUT / "resumo.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(log, indent=1, default=str))
