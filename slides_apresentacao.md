---
marp: true
theme: default
paginate: true
---

# Inflação e Comunicação do Copom
### Previsão Multivariada do IPCA Regional e Análise de Sentimento com LLM

**Disciplina:** Séries Temporais e Análise Preditiva  
**Período:** Janeiro/2012 – Setembro/2026 (177 meses)

---

## 1. Problema e Pergunta de Pesquisa

> *Qual será o IPCA mensal de cada região metropolitana e de cada grupo de despesa nos próximos 3 meses, e o tom das atas do Copom ajuda a prever a inflação?*

- **Alvo:** variação mensal do IPCA (%) por área geográfica e grupo de despesa.
- **Horizonte:** $h = 1, 2, 3$ meses (alinhado a dois ciclos de reunião do Copom).
- **Benchmark obrigatório:** mediana das expectativas **Focus** (BCB).

---

## 2. Base de Dados e Estrutura de Painel

- **IBGE/SIDRA (tabelas 1419 e 7060):** 16 áreas × 9 grupos de despesa × 177 meses = **25.488 observações**.
- **BCB/SGS:** Selic (`4189`), câmbio USD/BRL (`3697`), IC-Br (`27574`), IBC-Br dessazonalizado (`24364`), desocupação (`24369`) e IPCA Brasil (`433`).
- **BCB/Expectativas:** mediana Focus mensal (`base_calculo = 0`, sem vazamento temporal).
- **BCB/Copom:** **238** atas e comunicados oficiais desde `2012-01-18`.

---

## 3. Análise Exploratória (EDA)

- **Ausentes:** `2.487` valores nulos preservados sem imputação artificial.
- **Outliers:** `1.239` observações sinalizadas por `1,5 × IQR` (mantidas por refletirem choques reais).
- **Decomposição e estacionariedade:** STL (`period=12`), testes conjuntos ADF + KPSS e ACF/PACF nas 144 séries.

![w:700](figuras/ipca_por_grupo.png)

---

## 4. Correlação, Granger e VIF

| Variável | Entra | Melhor lag CCF | Granger min *p* | VIF (nível) | VIF (1ª dif.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `selic_efetiva_aa` | Sim | 2 | **0,0370** | 2,08 | 1,06 |
| `cambio_usd_brl` | Sim | 1 | 0,1627 | 12,44 | 1,62 |
| `ic_br` | Sim | 11 | 0,0595 | 17,42 | 1,46 |
| `ibc_br_sa` | Sim | 4 | 0,1245 | 7,69 | 1,15 |
| `desocupacao` | Sim | 4 | 0,5577 | 6,55 | 1,06 |

- Colinearidade em nível eliminada em **primeira diferença** (VIF ≤ 1,62).
- Lags selecionados: `1, 3, 6, 12` (e primeiras diferenças no VAR).

---

## 5. Modelagem e Validação Walk-Forward ($h = 1, 2, 3$)

- **4 famílias + Focus:** Naïve Sazonal, Focus Mediana, Estatísticos (VAR(1), SARIMAX(1,0,0), RidgeCV), Aprendizado de Máquina (Random Forest residual sobre âncora) e Rede Neural (LSTM PyTorch).
- **Validação:** origem móvel nos últimos 12 meses (`2025-10` a `2026-09`), sem embaralhamento, com preditores sazonais/Focus alinhados ao mês-alvo e defasagens da Etapa 2.

| Modelo | MAE | RMSE | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: |
| `sarimax_exogeno` | 0,1152 | **0,1413** | 49,77 | 0,6112 |
| `var_multivariado` | **0,1143** | 0,1466 | **46,84** | **0,6065** |
| `regressao_ridge` | 0,1267 | 0,1509 | 53,25 | 0,6723 |
| `lstm_multivariado` | 0,1255 | 0,1525 | 57,57 | 0,6659 |
| `focus_mediana` | 0,1351 | 0,1526 | 58,45 | 0,7168 |
| `random_forest` | 0,1305 | 0,1604 | 58,70 | 0,6923 |
| `baseline_naive_sazonal` | 0,1391 | 0,1837 | 53,62 | 0,7382 |

---

## 6. Intervalos de Previsão e Diebold-Mariano

- **Todos os 5 modelos próprios superam o baseline** (`baseline_naive_sazonal`) em MAE, RMSE e MASE no consolidado; 4 superam a `focus_mediana`.
- **Diebold-Mariano:** sem diferença significativa a 5% entre `sarimax_exogeno` e `var_multivariado` ($p > 0,25$ em $h=1,2,3$).
- **Destaque em $h = 1$ mês:** `sarimax_exogeno` (RMSE = `0,1359`) e `var_multivariado` (RMSE = `0,1420`) superam o Focus (`0,1625`) e o naïve sazonal (`0,2007`).

![w:720](figuras/previsoes_intervalos_modelos.png)

---

## 7. Previsão no Painel Completo (`16 áreas × 9 grupos`)

Modelos globais com efeitos fixos (`OneHotEncoder`) e perfil sazonal local superam o baseline nos três horizontes:

| Modelo | Horizonte | N | MAE | RMSE | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `random_forest_painel` | 1 mês | 1.728 | **0,5047** | **0,7166** | **0,6066** |
| `ridge_painel` | 1 mês | 1.728 | 0,5239 | 0,7423 | 0,6296 |
| `baseline_naive_sazonal_painel` | 1 mês | 1.728 | 0,6713 | 1,0602 | 0,8067 |
| `random_forest_painel` | 2 meses | 1.584 | **0,5174** | **0,7396** | **0,6004** |
| `random_forest_painel` | 3 meses | 1.440 | **0,5187** | **0,7449** | **0,6097** |

---

## 8. Sentimento do Copom com LLM e Validação Humana (Etapa 6)

- **Amostra validada (200 documentos distintos):** humano (`108` neutros, `59` contracionistas, `33` expansionistas) vs. LLM (`76`, `76`, `48`).
- **Kappa de Cohen (`validacao_sentimento.csv`):** `manual_vs_llm` = **`0,7357`** (substancial) vs. `manual_vs_lexico` = `0,3078` e `llm_vs_lexico` = `0,3803`.
- **Impacto no melhor SARIMAX walk-forward ($h = 1$ mês, 12 meses de teste):**

| Modelo | MAE | RMSE | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: |
| `sarimax_sem_sentimento` | **0,1073** | **0,1359** | 55,85 | **0,5693** |
| `sarimax_com_sentimento` | 0,1121 | 0,1420 | **44,26** | 0,5951 |

O sentimento defasado melhorou o sMAPE (`44,26%` vs. `55,85%`), mas não reduziu MAE/RMSE/MASE em $h=1$.

---

## 9. Limitações e Próximos Passos

- **Validação humana concluída (1 anotador):** 200 textos rotulados (`kappa = 0,7357`); falta apenas eventual 2º anotador (`rotulo_manual_2`).
- **Cobertura textual:** expandir dos 200 trechos representativos para os 1.094 trechos completos e incluir notícias econômicas.
- **Métricas:** usar MASE/sMAPE em vez de MAPE devido a valores de IPCA próximos de zero.
- **Amostra de teste:** ampliar a janela fora da amostra para maior poder no teste de Diebold-Mariano.
