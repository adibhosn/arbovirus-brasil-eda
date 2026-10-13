"""Análise exploratória (EDA) sobre o dataset completo.

Saídas em reports/: tabelas CSV + eda_summary.txt + figures/*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import RAW_PARQUET, REPORTS_DIR, FIG_DIR, TARGET, DATE_COLS

REPORTS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
df = pd.read_parquet(RAW_PARQUET)
out = open(REPORTS_DIR / "eda_summary.txt", "w")
def log(*a):
    print(*a); print(*a, file=out)

log(f"Shape: {df.shape}")
log("\n== Alvo ==")
vc = df[TARGET].value_counts(); log(pd.concat([vc, (vc/len(df)).round(4)], axis=1))

log("\n== Atributos constantes ==")
const = [c for c in df.columns if df[c].nunique(dropna=False) == 1]; log(const)

log("\n== Valores nulos (NaN) por coluna ==")
na = df.isna().sum(); log(na[na > 0])

log("\n== Datas inválidas / fora de faixa ==")
for c in DATE_COLS:
    log(c, "min", df[c].min(), "max", df[c].max(), "NaT", int(df[c].isna().sum()))

log("\n== Duplicatas exatas ==")
log(int(df.duplicated().sum()))

log("\n== Registros por ano ==")
yr = pd.crosstab(df["NU_ANO"], df[TARGET]); log(yr)
yr.to_csv(REPORTS_DIR / "classe_por_ano.csv")

log("\n== Idade (NU_IDADE_N) ==")
log(df["NU_IDADE_N"].describe())
log("idade<0:", int((df.NU_IDADE_N < 0).sum()), " idade>110:", int((df.NU_IDADE_N > 110).sum()))
d = (df.DT_NOTIFIC.dt.year - df.DT_NASC.dt.year)
log("|idade calculada - NU_IDADE_N|>2:", int(((d - df.NU_IDADE_N).abs() > 2).sum()))

log("\n== Intervalos entre datas (dias) ==")
for a, b in [("DT_SIN_PRI", "DT_NOTIFIC"), ("DT_NOTIFIC", "DT_INVEST"), ("DT_NOTIFIC", "DT_ENCERRA")]:
    dd = (df[b] - df[a]).dt.days; log(f"{b}-{a}:", dd.describe()[["min", "50%", "max"]].to_dict(), "negativos:", int((dd < 0).sum()))

# Colunas categóricas/binárias: distribuição de códigos (inclui 'ignorado')
cat_cols = [c for c in df.columns if c not in DATE_COLS + [TARGET] and df[c].dtype == "float32"
            and df[c].nunique() <= 12]
log("\n== Distribuição de códigos (colunas com <=12 valores) ==")
rows = []
for c in cat_cols:
    p = df[c].value_counts(normalize=True, dropna=False).round(4).to_dict()
    log(c, p); rows.append({"col": c, **{str(k): v for k, v in p.items()}})
pd.DataFrame(rows).to_csv(REPORTS_DIR / "codigos_por_coluna.csv", index=False)

log("\n== Cardinalidade das colunas geográficas ==")
geo = ["SG_UF_NOT", "ID_MUNICIP", "ID_REGIONA", "ID_UNIDADE", "SG_UF", "ID_MN_RESI", "ID_RG_RESI", "COUFINF", "COMUNINF"]
log({c: int(df[c].nunique()) for c in geo})

log("\n== Taxa de 'ignorado' (9) nos sintomas/comorbidades por classe ==")
sint = ["FEBRE","MIALGIA","CEFALEIA","EXANTEMA","VOMITO","NAUSEA","DOR_COSTAS","CONJUNTVIT","ARTRITE","ARTRALGIA",
        "PETEQUIA_N","LEUCOPENIA","LACO","DOR_RETRO","DIABETES","HEMATOLOG","HEPATOPAT","RENAL","HIPERTENSA","ACIDO_PEPT","AUTO_IMUNE"]
log(df[sint].eq(9).groupby(df[TARGET], observed=True).mean().round(3).T)
log("\nValores de FEBRE:", df.FEBRE.value_counts().to_dict())

log("\n== Resultados laboratoriais (4 = não realizado?) por classe ==")
lab = ["RESUL_SORO","RESUL_NS1","RESUL_VI_N","RESUL_PCR_","HISTOPA_N","IMUNOH_N"]
for c in lab:
    log(c); log(pd.crosstab(df[c], df[TARGET], normalize="columns").round(3))

log("\n== Prevalência de sintomas (==1) por classe ==")
prev = df[sint].eq(1).groupby(df[TARGET], observed=True).mean().round(3).T; log(prev)
prev.to_csv(REPORTS_DIR / "sintomas_por_classe.csv")

# Figuras
fig, ax = plt.subplots(figsize=(6, 4)); vc.plot.bar(ax=ax, color="steelblue"); ax.set_title("Distribuição do alvo"); ax.set_yscale("log")
fig.tight_layout(); fig.savefig(FIG_DIR / "alvo.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4)); yr.plot.bar(stacked=True, ax=ax); ax.set_title("Notificações por ano e classe")
fig.tight_layout(); fig.savefig(FIG_DIR / "classe_por_ano.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for k, g in df[df.NU_IDADE_N.between(0, 110)].groupby(TARGET, observed=True):
    sns.kdeplot(g.NU_IDADE_N.sample(min(len(g), 200_000), random_state=0), ax=ax, label=k, common_norm=False)
ax.legend(); ax.set_title("Idade por classe"); fig.tight_layout(); fig.savefig(FIG_DIR / "idade.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 8)); sns.heatmap(prev, annot=True, fmt=".2f", cmap="viridis", ax=ax)
ax.set_title("Prevalência (==1) por classe"); fig.tight_layout(); fig.savefig(FIG_DIR / "sintomas_heatmap.png", dpi=120); plt.close(fig)

num = df[sint + ["NU_IDADE_N", "HOSPITALIZ"]].sample(500_000, random_state=0)
fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(num.corr(), cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Correlação (amostra 500k)"); fig.tight_layout(); fig.savefig(FIG_DIR / "correlacao.png", dpi=120); plt.close(fig)
out.close()
