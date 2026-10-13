"""Checa (1) cobertura desigual entre colunas e (2) diferença de grandezas.

Saída: reports/escala_cobertura.txt
"""
import numpy as np, pandas as pd
from config import RAW_PARQUET, DATA_DIR, REPORTS_DIR, DATE_COLS
out = open(REPORTS_DIR / "escala_cobertura.txt", "w")
def log(*a):
    print(*a); print(*a, file=out)
pd.set_option("display.width", 200, "display.max_rows", 200)

raw = pd.read_parquet(RAW_PARQUET)
raw = raw[raw.NU_ANO.between(2013, 2020)]
log("=== 1. COBERTURA (bruto, 2013-2020) ===")
log("Linhas:", len(raw))
IGN = {"CS_SEXO": 9, "CS_GESTANT": 9, "CS_RACA": 9, "CS_ESCOL_N": [9, 10], "TPAUTOCTO": 9, "HOSPITALIZ": 9,
       "EVOLUCAO": 9, "FEBRE": 9, "RESUL_SORO": 4, "RESUL_NS1": 4, "RESUL_VI_N": 4, "RESUL_PCR_": 4,
       "HISTOPA_N": 4, "IMUNOH_N": 4}
rows = []
for c in raw.columns:
    r = {"col": c, "NaN_%": 100 * raw[c].isna().mean()}
    if c in IGN:
        r["ignorado/nao_realizado_%"] = 100 * raw[c].isin(np.atleast_1d(IGN[c])).mean()
    rows.append(r)
cov = pd.DataFrame(rows).fillna(0).round(2)
cov["info_util_%"] = (100 - cov["NaN_%"] - cov["ignorado/nao_realizado_%"]).round(2)
log(cov.sort_values("info_util_%").to_string(index=False))

log("\n=== 2. GRANDEZAS (bruto, colunas numéricas) ===")
num = raw.select_dtypes("number")
st = num.agg(["min", "max", "mean", "std"]).T
st["amplitude"] = st["max"] - st["min"]
log(st.sort_values("amplitude", ascending=False).round(2).to_string())
log("\nRazão maior/menor desvio-padrão (bruto):", round(st["std"].max() / st[st["std"] > 0]["std"].min(), 1))

for cen in ["binario", "multiclasse"]:
    log(f"\n=== 3. CENÁRIO {cen}: desvio-padrão e amplitude por variante (treino) ===")
    d = DATA_DIR / "processed" / cen
    if not (d / "train_tree.parquet").exists():
        log(f"(cenário {cen} não gerado — rode 02_preprocess.py --cenario {cen} para incluí-lo)"); continue
    res = {}
    for k in ["tree", "std", "minmax"]:
        t = pd.read_parquet(d / f"train_{k}.parquet").drop(columns="y")
        res[k] = pd.DataFrame({"min": t.min(), "max": t.max(), "std": t.std()})
    t = res["tree"]
    log("[tree - sem escala] colunas com amplitude > 1:")
    log(t[(t["max"] - t["min"]) > 1].round(3).to_string())
    log("razão max(std)/min(std) entre colunas não constantes:")
    for k, r in res.items():
        rr = r[r["std"] > 0]["std"]
        log(f"  {k:7s}: {rr.max()/rr.min():8.1f}   (maior std: {rr.idxmax()}={rr.max():.3f}; menor: {rr.idxmin()}={rr.min():.4f})")
    s = res["std"].loc[["idade", "dias_sintoma_notif"]]
    log("[std] contínuas:"); log(s.round(3).to_string())
    m = res["minmax"].loc[["idade", "dias_sintoma_notif"]]
    log("[minmax] contínuas:"); log(m.round(3).to_string())
out.close()
