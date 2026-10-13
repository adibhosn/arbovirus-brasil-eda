# Pré-processamento — decisões e justificativas

Dataset: *Arbovirus clinical data, Brazil, 2013–2020* (SINAN; Sampaio et al., 2022, Mendeley Data, doi:10.17632/2d3kr8zynf.4).
Arquivo usado: `data set.csv` (já pré-processado pelos autores). Reprodução: ver `README.md`.
Números abaixo vêm de `reports/eda_summary.txt` e `data/processed/*/resumo.json`.

## 1. O que a EDA revelou

| # | Achado | Evidência |
|---|--------|-----------|
| 1 | O arquivo tem **6.732.548 linhas** (o site cita 7,6 M) e 56 colunas. | `wc -l` / parquet |
| 2 | **6 linhas-lixo**: cabeçalho repetido no meio do arquivo (`CLASSI_FIN == "CLASSI_FIN"`, tudo NaN). | `eda_summary.txt` |
| 3 | **Não há NaN "de verdade"**: o faltante vem codificado (9 = ignorado, 4 = não realizado, 6 = não se aplica), conforme o dicionário SINAN. | distribuição de códigos |
| 4 | **Bloco clínico é "tudo ou nada"**: as 21 colunas de sintomas/comorbidades são todas 9 ou nenhuma é (0 casos intermediários). 52,4 % das linhas estão sem ficha clínica. | contagem de 9 por linha |
| 5 | Sem ficha clínica: **100 % de 2013–2014**, ~96 % de 2015, 40 % de 2016 e **100 % da Chikungunya** (só 150 de 325 k têm sintomas). | cruzamento ano × classe |
| 6 | Classes: Dengue 64,0 %, Descartado 31,2 %, Chikungunya 4,8 %. Chikungunya só existe a partir de 2015. | `classe_por_ano.csv` |
| 7 | Exames laboratoriais: 97,6–99,9 % são "4 = não realizado". | `eda_summary.txt` |
| 8 | Idades absurdas (999, 8xx, 5xx — 8 registros > 110); 286 k registros com `NU_IDADE_N` ≠ idade calculada por `DT_NASC` (>2 anos). `DT_NASC` ausente em 389 k (62 % da Chikungunya). | EDA |
| 9 | Datas corrompidas: `DT_SIN_PRI` de 0201 a 2106; intervalos negativos (ex.: notificação antes do sintoma, 270 casos). | EDA |
| 10 | 8.751 linhas duplicadas exatas (0,13 %). Anos 2012 e 2021 têm <450 linhas (fora do período do dataset). | EDA |
| 11 | A proporção Dengue/Descartado varia muito por UF (de 20 % a 86 % Dengue) e por ano. | `eda_summary.txt` |
| 12 | Sintomas diferenciam pouco Dengue × Descartado (ex.: febre 39,8 % vs 37,8 %; dor retro-orbital 17 % vs 11 %). | `sintomas_por_classe.csv` |

## 2. Decisões

### D1 — Remoção de linhas inválidas
Removidas as 6 linhas de cabeçalho e as linhas com `NU_ANO` fora de 2013–2020. (−445 linhas.)

### D2 — Duplicatas
Removidas as duplicatas exatas **antes** do split (−8.745). Sem identificador de paciente não dá para saber se eram pessoas distintas, mas o custo é 0,13 % e evita cópias idênticas entre treino e teste.

### D3 — Definição do problema (**decisão que precisa ser alinhada com o grupo**)
Gerados dois cenários:

* **`binario` (recomendado/padrão)** — Dengue (1) vs Descartado/Inconclusivo (0), usando só linhas **com ficha clínica preenchida** (3.202.815 linhas; 65,8 % / 34,2 %).
  *Por quê:* nas linhas sem ficha (achado 4–5) os sintomas são inexistentes; imputá-los seria inventar informação, e a Chikungunya não tem dado clínico, então não há como classificá-la com base em sintomas.
* **`multiclasse`** — as 3 classes, todas as linhas (6.723.358).
  Os sintomas "ignorados" viram 0 e é criada a flag `sem_ficha_clinica`. **Atenção:** nesse cenário o faltante revela a classe (Chikungunya ⇒ ficha vazia), e o `hosp_9` e `sem_ficha_clinica` dominam a importância das variáveis. O resultado reflete o *processo de coleta*, não a clínica. Use apenas se o enunciado exigir 3 classes, e reporte essa limitação.

### D4 — Variáveis excluídas
| Variável(is) | Motivo |
|---|---|
| `ID_AGRAVO`, `ID_PAIS`, `COPAISINF` | constantes/quase constantes |
| `CLASSI_FIN` | é o alvo |
| `RESUL_SORO/NS1/VI_N/PCR_`, `HISTOPA_N`, `IMUNOH_N` | **vazamento**: o resultado laboratorial é o que *define* a classificação final; além disso 98–99,9 % são "não realizado" |
| `EVOLUCAO`, `DT_ENCERRA` | só existem após o encerramento do caso (pós-desfecho) |
| `DT_INVEST`, `TPAUTOCTO`, `COUFINF`, `COMUNINF` | preenchidos na investigação/conclusão do caso, não na triagem |
| `ID_MUNICIP`, `ID_UNIDADE`, `ID_REGIONA`, `ID_MN_RESI`, `ID_RG_RESI` | cardinalidade 342–39 k: one-hot inviável, ordinal sem sentido; risco de decorar município/unidade |
| `SG_UF` (residência) | redundante com `SG_UF_NOT` |
| `DT_NASC` | redundante com idade e 6 % ausente |
| `NU_ANO`, `SEM_NOT`, `SEM_PRI`, `DT_*` brutas | o ano não generaliza (modelos aprenderiam o surto de 2013/2016); datas viram features derivadas (D6) |

> Se o grupo quiser testar o efeito do laboratório, basta incluir `RESUL_*` de volta — mas deve-se reportar como cenário com vazamento.

### D5 — Valores faltantes / codificados
* Sintomas e comorbidades (1 = sim, 2 = não): binarizados `1→1, 2→0`. No cenário binário não há "9" depois do filtro de D3. No multiclasse, 9→0 + flag.
* Categóricas (`CS_SEXO`, `CS_RACA`, `CS_ESCOL_N`, `HOSPITALIZ`): "ignorado" mantido como **categoria própria** (one-hot) — informação de preenchimento pode ser relevante e não temos base para imputar raça/escolaridade.
* `CS_GESTANT`: agrupada em `sim` (trimestres 1–4) / `nao` / `na` (não se aplica) / `ign`, porque os trimestres são raros (<0,3 % cada).
* `CS_SEXO`: códigos raros (2 e 9, ~0,1 % juntos) foram **consolidados em `sexo_ign`** em vez de ficarem em colunas quase vazias ou serem removidos. O dataset foi recodificado pelos autores (0 / 1 / 2 / 9; 0 = 55 %, 1 = 44 %) e **não há dicionário confiável** no pacote que baixamos (`attributes.csv` não foi incluído). Tratado como nominal, sem atribuir M/F.
* Numéricas: valores impossíveis viram NaN (idade fora de 0–110; `dias_sintoma_notif` fora de 0–30) e são **imputados pela mediana do treino** (idade 30; dias 2). Cria-se a coluna `n_imputados` (0–2 = quantos campos numéricos do registro foram imputados), que preserva a informação do faltante e funciona como indicador de qualidade do registro (taxa de Dengue 71 % nos registros com imputação vs 66 % nos demais). Proporção afetada é minúscula (<1 %).

### D6 — Engenharia de variáveis
* `idade` (de `NU_IDADE_N`, já em anos).
* `dias_sintoma_notif` = `DT_NOTIFIC − DT_SIN_PRI` (atraso até notificar).
* `mes_sin`, `mes_cos` — sazonalidade codificada de forma cíclica (dez ≈ jan). Foi a variável mais importante no baseline do RF.
* `reg_{N,NE,SE,S,CO}` — macrorregião, derivada do 1º dígito da UF de notificação, em vez de 27 UFs, para manter a dimensionalidade baixa (útil para o fuzzy). Alternativa: reincluir `SG_UF_NOT` em one-hot.

### D7 — Split
Estratificado pelo alvo, 70 / 15 / 15 (treino / validação / teste), `random_state=42`. Proporções idênticas nos três conjuntos. **Limitação:** split aleatório mistura anos, então o desempenho estimado pode ser otimista em relação a prever um surto futuro; um split temporal (treino ≤ 2018, teste 2019–2020) é um bom teste adicional.

### D8 — Escalonamento
Ajustado **apenas no treino**; só as 2 colunas contínuas (`idade`, `dias_sintoma_notif`) precisam de escala — o resto já é 0/1 ou seno/cosseno em [−1, 1]. Três versões do mesmo dado:
| Arquivo | Escala | Para quem |
|---|---|---|
| `*_tree.parquet` | nenhuma | Random Forest (invariante a escala) |
| `*_std.parquet` | StandardScaler (média 0, dp 1) | Rede Neural |
| `*_minmax.parquet` | MinMaxScaler [0,1] (com clip em val/test) | Fuzzy (funções de pertinência) |

### D9 — Desbalanceamento
Não foi aplicado nenhum reamostramento nos arquivos. O binário é 66/34 (leve) e o multiclasse tem Chikungunya com 4,8 %. Recomendação: usar `class_weight="balanced"` (RF, MLP via sample weights) ou reamostrar **somente o treino** e avaliar com *balanced accuracy*, F1-macro e AUC — acurácia pura engana.

### D10 — Tipos e memória
CSV de 2 GB → Parquet (137 MB) com `float32`/`int8`. Identificadores geográficos como `Int32` (não numéricos). A leitura é feita em blocos de 500 k linhas.

### D11 — Colunas com cobertura desigual (verificado em `reports/escala_cobertura.txt`)
Nenhuma coluna tem NaN relevante, mas "preenchida" ≠ "informativa". Fração de valores realmente informados (bruto 2013–2020):

| Coluna | Informação útil | Tratamento |
|---|---|---|
| `RESUL_*` (6 exames) | 0,05 % a 2,4 % | **excluídas** (D4) |
| `CS_ESCOL_N` | 35,5 % (64 % ignorado/não se aplica; no cenário binário 54 % ignorado + 7,6 % N/A) | mantida, "ignorado" como categoria própria; **candidata a remoção no Fuzzy** |
| `HOSPITALIZ` | 56 % (no binário, 29,7 % ignorado; Dengue = 60 % nesse grupo vs 68 % nos demais) | mantida como categoria |
| `CS_RACA` | 66 % (binário: 27,8 % ignorado) | mantida como categoria |
| `CS_GESTANT` | 89 % | agrupada (D5) |
| 21 sintomas/comorbidades | 100 % **dentro do cenário binário** (nos demais cenários, 0 %) | filtro por ficha preenchida (D3) |
| `idade`, `dias_sintoma_notif`, sexo, UF, data | ≈ 100 % (2,5 % de `dias_sintoma_notif` inválidos → imputados + flag) | — |

Consequências: (i) o "ignorado" de `HOSPITALIZ`/`CS_RACA` carrega um sinal fraco sobre a qualidade da notificação — por isso é categoria e não imputação; (ii) `idade_ausente` (2 linhas), `sexo_2` (3.874) e `sexo_9` (26) eram quase constantes; em vez de removê-las, foram **consolidadas**: `sexo_2`+`sexo_9` → `sexo_ign`; `idade_ausente`+`dias_sintoma_notif_ausente` → `n_imputados`. Nenhuma informação foi descartada.

### D12 — Diferença de grandezas / escala (verificado)
**No dataset bruto o problema existe e é grande**: `ID_UNIDADE` vai a 9.999.999, `SEM_NOT` ≈ 202.000, `ID_MUNICIP` ≈ 600.000, `NU_IDADE_N` até 999, contra sintomas em 1–9 e flags 0/1. A razão entre o maior e o menor desvio-padrão é ~1,7×10⁸. Esses casos foram resolvidos assim:

* IDs geográficos (`ID_UNIDADE`, `ID_MUNICIP`, …) e semanas epidemiológicas (`SEM_*`): **removidos** — são códigos, não quantidades; seu valor numérico não tem sentido ordinal (D4).
* `NU_IDADE_N` (999): limpa e escalonada.
* Sintomas com códigos 1/2/9: recodificados para 0/1 (a codificação original 1–9 faria "ignorado=9" parecer "9× mais sintoma").
* Categóricas nominais (sexo, raça, escolaridade, hospitalização, região): one-hot, para que a rede/fuzzy não leiam "raça=4" como maior que "raça=1".

**Nos arquivos processados** só existem 2 colunas contínuas fora de [0,1]: `idade` (0–99, dp 18,5) e `dias_sintoma_notif` (0–30, dp 3,9); `mes_sin/mes_cos` estão em [−1,1]; todo o resto é 0/1. Sem escala (`*_tree`) elas dominariam por magnitude — o que é irrelevante para RF, mas prejudicial para a Rede Neural e o Fuzzy; por isso existem `*_std` (média 0, dp 1) e `*_minmax` ([0,1], dp 0,19 e 0,13, comparáveis aos de uma flag 0/1 com prevalência de ~30 %).

Obs.: a razão max/min de desvio-padrão ainda aparece alta (≈ 1.000 em `std`) **não por grandeza, mas por raridade**: é causada por flags quase constantes (`idade_ausente`, dp 0,0009). Isso não dá peso extra a nada; só inclui colunas inúteis.

> Importante para os colegas: **use o arquivo certo para cada algoritmo** (D8). Se a Rede Neural ler `*_tree`, o problema de grandezas volta.

## 3. Verificação de sanidade (baseline, 300 k linhas de treino, 100 k de teste)

| Cenário | Modelo | Bal. Acc | F1-macro | AUC |
|---|---|---|---|---|
| binario | Reg. logística | 0,636 | 0,621 | 0,688 |
| binario | Random Forest | 0,670 | 0,656 | 0,733 |
| multiclasse | Reg. logística | 0,658 | 0,482 | 0,747 |
| multiclasse | Random Forest | 0,680 | 0,505 | 0,766 |

Interpretação: o problema binário é **intrinsecamente difícil** com esses dados (sintomas pouco discriminativos), e é isso que se espera; resultados próximos de 0,95+ indicariam vazamento. Isso é um piso de referência, não o resultado final.

## 4. Recomendações adicionais (não aplicadas aos arquivos)
* Colunas raras foram consolidadas, não removidas (ver D11). Persiste a decisão sobre `escol_*` no Fuzzy.
* Para o Fuzzy (poucas variáveis fazem sentido): sugerido usar `idade`, `dias_sintoma_notif`, os 14 sintomas, os 7 comorbidades agregados em um índice, `mes_*` e `reg_*`; descartar `escol_*` (54 % ignorado).
* Rede Neural: se for usar embeddings/outra codificação, ajuste só no treino.

## 5. Limitações a mencionar no relatório
1. Os rótulos são do próprio sistema de notificação: "Descartado/Inconclusivo" mistura casos que não eram dengue com casos sem conclusão.
2. 52 % do dataset não tem informação clínica (mudança de ficha ~2016) — viés de seleção no cenário binário (só 2015–2020, predominantemente 2017+).
3. Dependência de UF/ano e possível diferença de critério entre estados.
4. Não foi possível confirmar a codificação de `CS_SEXO` (sem `attributes.csv`).
