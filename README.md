# Pré-processamento — Dengue/Chikungunya (SINAN 2013–2020)

## Pré-requisitos (vale para as duas opções)
* Python ≥ 3.11 e ~16 GB de RAM.
* Baixar `data set.csv` do [Mendeley Data](https://doi.org/10.17632/2d3kr8zynf.4) e colocá-lo na **raiz** do repositório (o nome deve ser exatamente esse; ele é ignorado pelo Git).
* Instalar o ambiente:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
```

Ordem de execução: **00 → 01 → 02 → 03 → 04**. O 02 gera um cenário por vez (`binario` ou `multiclasse`); rode-o duas vezes para ter os dois. O 03 e o 04 usam o que existir.

## Opção A — Notebooks (`notebooks/`)
```bash
.venv/bin/jupyter lab        # abrir notebooks/ e executar na ordem
```
No notebook 02, escolha o cenário na variável `CENARIO` (`"binario"` ou `"multiclasse"`) ou via ambiente: `CENARIO=multiclasse jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessamento.ipynb`. Os notebooks já estão no repositório com as saídas e figuras da execução original.

## Opção B — Scripts (`src/`)
```bash
cd src
../.venv/bin/python 00_to_parquet.py
../.venv/bin/python 01_eda.py
../.venv/bin/python 02_preprocess.py --cenario binario        # padrão
../.venv/bin/python 02_preprocess.py --cenario multiclasse
../.venv/bin/python 03_sanity_baseline.py binario
../.venv/bin/python 04_check_escala_cobertura.py
```

| Etapa | Notebook | Script |
|---|---|---|
| CSV 2 GB → `data/raw.parquet` | `00_conversao_parquet.ipynb` | `00_to_parquet.py` |
| Análise exploratória (`reports/`) | `01_eda.ipynb` | `01_eda.py` |
| Limpeza, features, split, escala → `data/processed/<cenario>/` | `02_preprocessamento.ipynb` | `02_preprocess.py` |
| Baselines de sanidade | `03_baseline_sanidade.ipynb` | `03_sanity_baseline.py` |
| Cobertura e grandezas | `04_escala_e_cobertura.ipynb` | `04_check_escala_cobertura.py` |

Os dois caminhos fazem o mesmo processamento e geram os mesmos arquivos. Se alterar a lógica de um, replique no outro.

Para os colegas (modelagem):
```python
import pandas as pd
tr = pd.read_parquet("data/processed/binario/train_std.parquet")   # _tree | _std | _minmax
X, y = tr.drop(columns="y"), tr["y"]
```
| Algoritmo | Arquivo |
|---|---|
| Random Forest | `*_tree.parquet` |
| Rede Neural | `*_std.parquet` |
| Fuzzy | `*_minmax.parquet` |

Decisões e justificativas: [docs/DECISOES.md](docs/DECISOES.md). Resumo numérico: `data/processed/<cenario>/resumo.json`.
