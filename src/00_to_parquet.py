"""Converte o CSV bruto (~2 GB) em Parquet com dtypes compactos.

Lê em blocos para não estourar a memória. Nenhuma limpeza é feita aqui:
os valores são preservados (inclusive códigos 'ignorado' como 9) para que a
EDA enxergue os dados exatamente como vieram.
"""
import pandas as pd
from config import RAW_CSV, RAW_PARQUET, DATA_DIR, DATE_COLS

DATA_DIR.mkdir(exist_ok=True)
CHUNK = 500_000
STR_COLS = ["ID_AGRAVO", "CLASSI_FIN"]
CODE_COLS = ["SG_UF_NOT", "ID_MUNICIP", "ID_REGIONA", "ID_UNIDADE", "SG_UF",
             "ID_MN_RESI", "ID_RG_RESI", "ID_PAIS", "COUFINF", "COPAISINF", "COMUNINF",
             "SEM_NOT", "SEM_PRI"]

parts = []
for i, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=CHUNK, low_memory=False)):
    for c in DATE_COLS:
        chunk[c] = pd.to_datetime(chunk[c], errors="coerce")
    for c in STR_COLS:
        chunk[c] = chunk[c].astype("category")
    for c in chunk.columns:
        if c in STR_COLS or c in DATE_COLS or c in CODE_COLS:
            continue
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
    for c in CODE_COLS:  # identificadores geográficos: categóricos, não numéricos
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int32")
    parts.append(chunk)
    print(f"bloco {i}: {len(chunk):,} linhas", flush=True)

df = pd.concat(parts, ignore_index=True)
df["CLASSI_FIN"] = df["CLASSI_FIN"].astype("category")
df.to_parquet(RAW_PARQUET, index=False)
print(df.shape, f"{df.memory_usage(deep=True).sum()/1e9:.2f} GB em memória")
