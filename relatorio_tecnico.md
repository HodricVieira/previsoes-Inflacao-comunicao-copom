# Relatório Técnico — Previsão da Inflação (IPCA) e Comunicação do Copom

**Disciplina:** Séries Temporais e Análise Preditiva  
**Período analisado:** Janeiro de 2012 a Setembro de 2026 (177 meses)

---

## 1. Introdução e Pergunta de Pesquisa

Este projeto investiga a capacidade preditiva de variáveis macroeconômicas e da comunicação oficial do Comitê de Política Monetária (Copom) sobre a dinâmica inflacionária brasileira. A pergunta central é:

> *Qual será o IPCA mensal de cada região metropolitana/capital e de cada grupo de despesa nos próximos 3 meses, e o tom das atas e comunicados do Copom ajuda a prever a inflação?*

O repositório organiza o fluxo de forma reprodutível em cinco módulos (`src/coleta_dados.py`, `src/analise_eda.py`, `src/analise_correlacao.py`, `src/analise_sentimento.py` e `src/modelagem_previsao.py`), além do notebook exploratório `notebooks/analise_exploratoria.ipynb`.

---

## 2. Base de Dados e Estrutura de Painel

Os dados foram coletados exclusivamente de fontes públicas oficiais:

- **IPCA mensal por área e grupo (IBGE/SIDRA):** combinação das tabelas **1419** (2012–2019) e **7060** (2020 em diante), variável 63 (variação mensal em %), classificação 315 (9 grupos principais de despesa) e níveis geográficos N6 e N7 (16 áreas metropolitanas e capitais).
- **Variáveis macroeconômicas nacionais (BCB/SGS):** taxa Selic efetiva anualizada base 252 (`4189`), taxa de câmbio USD/BRL (`3697`), Índice de Commodities Brasil — IC-Br (`27574`), IBC-Br dessazonalizado (`24364`), taxa de desocupação PNAD Contínua (`24369`) e IPCA Brasil (`433`).
- **Expectativas de mercado (BCB/Focus):** mediana das projeções mensais de IPCA (`ExpectativaMercadoMensais`), preservando `data_coleta`, `mes_referencia` e `base_calculo = 0` para evitar vazamento temporal.
- **Comunicação oficial do Copom (BCB):** 238 atas e comunicados publicados desde `2012-01-18`.

O painel integrado (`dados/tratados/painel_ipca_macro.csv`) contém **25.488 observações** (**16 áreas × 9 grupos × 177 meses**), acompanhado de dicionários de variáveis completos (`dicionario_variaveis.csv`, `dicionario_painel_ipca_macro.csv` e `dicionario_sentimento.csv`).

---

## 3. Metodologia

### 3.1 Análise Exploratória (EDA)
- **Tratamento de ausentes e outliers:** foram identificados **2.487 valores ausentes** no IPCA regional (preservados como nulos, sem imputação artificial, utilizando-se apenas segmentos contínuos nos testes). A regra de **1,5 × IQR** sinalizou **1.239 observações atípicas** para inspeção sem removê-las, pois refletem choques econômicos reais (ex.: bandeiras tarifárias, choques de alimentos e combustíveis).
- **Decomposição e estacionariedade:** aplicou-se decomposição **STL** (`period=12`) nas 144 séries (`área × grupo`), além de testes conjuntos **ADF** ($H_0$: raiz unitária) e **KPSS** ($H_0$: estacionariedade) e funções **ACF/PACF**.

### 3.2 Análise de Correlação, Causalidade e Multicolinearidade
- **Correlação e CCF:** matrizes de **Pearson e Spearman** na série mensal e correlação cruzada com defasagens de **0 a 12 meses** em todas as 144 séries do painel (11.232 combinações).
- **Causalidade de Granger:** testes para defasagens de 1 a 6 meses na série mediana nacional e em cada série do painel.
- **Multicolinearidade (VIF):** cálculo com constante (`sm.add_constant`) em nível, em primeira diferença e com redução de variáveis.

### 3.3 Modelagem, Validação Walk-Forward e Métricas
- **Estratégia de validação:** divisão estritamente temporal, sem embaralhamento, utilizando **validação walk-forward (origem móvel)** sobre os últimos **12 meses** (`2025-10-01` a `2026-09-01`) para horizontes **$h \in \{1, 2, 3\}$ meses**.
- **Justificativa do horizonte de 3 meses:** corresponde a aproximadamente dois ciclos de reunião do Copom (reuniões a cada ~45 dias), constituindo o horizonte tático padrão de acompanhamento da inflação de curto prazo.
- **Famílias de modelos comparadas (série nacional mediana):**
  1. *Referência (baseline):* Naïve Sazonal ($y_{t+h-12}$) e **Mediana do Focus** (coletada no mês de origem $t$ para $t+h$).
  2. *Estatísticos/Lineares:* **VAR(1)** (com primeiras diferenças das variáveis macroeconômicas para estacionariedade e exógenas sazonais/Focus), **SARIMAX(1,0,0)** com exógenas alinhadas ao mês-alvo (`focus_tau`, `y_lag12`, perfil sazonal mensal histórico calculado no treino e defasagens macroeconômicas da Etapa 2) e **Regressão Ridge** (`RidgeCV`) direta por horizonte.
  3. *Aprendizado de Máquina:* **Random Forest** (`200` árvores, `max_depth=3`, `min_samples_leaf=8`) ajustado sobre o desvio residual em relação à âncora Focus-sazonal (`0,75 · focus_tau + 0,25 · y_lag12`).
  4. *Redes Neurais:* **LSTM** em PyTorch (janela `lookback=12`, camada oculta de `8` unidades, regularização L2 `weight_decay=0,25`, `80` épocas) prevendo o resíduo padronizado sobre a âncora Focus-sazonal.
- **Previsão desagregada no painel (16 áreas × 9 grupos):** modelos globais de painel (`ridge_painel` com `RidgeCV` e `random_forest_painel`) com efeitos fixos via `OneHotEncoder` (área e grupo), perfil sazonal local calculado exclusivamente no treino, expectativa Focus e defasagens locais/macroeconômicas, comparados ao `baseline_naive_sazonal_painel`.
- **Avaliação estatística e incerteza:** métricas **MAE, RMSE, MAPE, sMAPE e MASE** (com escala naïve sazonal in-sample), teste bilateral de **Diebold-Mariano** (variância HAC de Bartlett para horizonte $h$) e **intervalos de previsão empíricos de 80% e 95%**.

### 3.4 Pipeline de Análise de Sentimento (Copom)
O módulo `src/analise_sentimento.py` segmenta os 238 documentos em 1.094 trechos (`trechos_sentimento.csv`), calcula um escore léxico de referência em 149 meses (`escore_lexico`) e importa os 200 rótulos gerados via LLM (`dados/tratados/rotulo_llm.json`, um trecho representativo por documento distinto, cobrindo 143 meses). Os escores são agregados por documento e por mês (`--permitir-parcial`) e integrados ao SARIMAX walk-forward com defasagem de 1 mês (`sentimento_llm_lag1`) para medir o ganho preditivo.

---

## 4. Resultados

### 4.1 Seleção de Variáveis e Defasagens (Etapa 2)
A análise conjunta de CCF, Granger e VIF fundamenta a entrada das cinco variáveis explicativas macroeconômicas com defasagens `1, 3, 6, 12` (e primeiras diferenças no VAR):

| Variável | Entra no modelo | Melhor lag CCF | Granger min *p* (nacional) | VIF (nível) | VIF (1ª dif.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `selic_efetiva_aa` | Sim | 2 | **0,0370** | 2,083 | 1,061 |
| `cambio_usd_brl` | Sim | 1 | 0,1627 | 12,440 | 1,623 |
| `ic_br` | Sim | 11 | 0,0595 | 17,420 | 1,456 |
| `ibc_br_sa` | Sim | 4 | 0,1245 | 7,690 | 1,150 |
| `desocupacao` | Sim | 4 | 0,5577 | 6,553 | 1,060 |
| `ipca_brasil` | Não (referência) | 0 | 0,0082 | 1,013 | 1,061 |

Observa-se que `ic_br` e `cambio_usd_brl` apresentam colinearidade de tendência em nível (VIF > 10), totalmente eliminada ao trabalhar em primeiras diferenças (VIF ≤ 1,62). A matriz de correlação e o mapa de calor de CCF estão em `figuras/correlacao_ccf_heatmap.png`.

### 4.2 Desempenho Preditivo — Série Nacional (Walk-Forward $h = 1, 2, 3$)

| Modelo | N | MAE | RMSE | MAPE (%) | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `sarimax_exogeno` | 33 | 0,1152 | **0,1413** | **72,33** | 49,77 | 0,6112 |
| `var_multivariado` | 33 | **0,1143** | 0,1466 | 80,48 | **46,84** | **0,6065** |
| `regressao_ridge` | 33 | 0,1267 | 0,1509 | 85,26 | 53,25 | 0,6723 |
| `lstm_multivariado` | 33 | 0,1255 | 0,1525 | 90,48 | 57,57 | 0,6659 |
| `focus_mediana` (referência) | 33 | 0,1351 | 0,1526 | 123,45 | 58,45 | 0,7168 |
| `random_forest` | 33 | 0,1305 | 0,1604 | 99,00 | 58,70 | 0,6923 |
| `baseline_naive_sazonal` | 33 | 0,1391 | 0,1837 | 84,08 | 53,62 | 0,7382 |

*Nota sobre MAPE:* como variações mensais do IPCA assumem valores próximos de zero (e ocasionais deflações), o MAPE sofre explosão numérica; por isso, **MASE** e **sMAPE** são as métricas recomendadas para comparação relativa.

Todos os cinco modelos próprios superam o `baseline_naive_sazonal` em MAE, RMSE e MASE no consolidado ($N = 33$), e quatro deles (`sarimax_exogeno`, `var_multivariado`, `regressao_ridge` e `lstm_multivariado`) também superam a `focus_mediana`. Por horizonte (`avaliacao_modelos_por_horizonte.csv`): em **$h = 1$ mês**, os cinco modelos próprios superam o `baseline_naive_sazonal` (RMSE = 0,2007) e a `focus_mediana` (RMSE = 0,1625), liderados por `sarimax_exogeno` (MAE = 0,1073; RMSE = 0,1359; MASE = 0,5693) e `var_multivariado` (MAE = 0,1102; RMSE = 0,1420); em **$h = 2$ meses**, `sarimax_exogeno` (RMSE = 0,1438) e `lstm_multivariado` (RMSE = 0,1504) mantêm vantagem sobre `focus_mediana` (RMSE = 0,1517) e `baseline_naive_sazonal` (RMSE = 0,1693); e em **$h = 3$ meses**, `focus_mediana` (RMSE = 0,1408) lidera seguida de perto por `var_multivariado` (RMSE = 0,1429) e `sarimax_exogeno` (RMSE = 0,1447), com todos os cinco modelos superando o `baseline_naive_sazonal` (RMSE = 0,1774).

Os intervalos de previsão de 80% e 95% para $h = 1$ mês estão ilustrados em `figuras/previsoes_intervalos_modelos.png`.

### 4.3 Teste de Diebold-Mariano
A comparação estatística (`teste_diebold_mariano.csv`) entre os dois melhores modelos próprios (`sarimax_exogeno` vs. `var_multivariado`) não indicou diferença estatisticamente significativa nos três horizontes ($p = 0,5476$ em $h=1$; $p = 0,2518$ em $h=2$; $p = 0,8890$ em $h=3$ para MSE). Da mesma forma, na janela de 10 a 12 pares de teste, as diferenças de `sarimax_exogeno` contra `focus_mediana` ($p = 0,2967$ em $h=1$; $p = 0,7682$ em $h=2$; $p = 0,8623$ em $h=3$) e contra `baseline_naive_sazonal` ($p = 0,1593$ em $h=1$; $p = 0,5030$ em $h=2$; $p = 0,2791$ em $h=3$) não atingiram significância a 5%.

### 4.4 Desempenho no Painel Completo (16 Áreas × 9 Grupos)

| Modelo | Horizonte | N previsões | MAE | RMSE | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `random_forest_painel` | 1 mês | 1.728 | **0,5047** | **0,7166** | **112,81** | **0,6066** |
| `ridge_painel` | 1 mês | 1.728 | 0,5239 | 0,7423 | 115,81 | 0,6296 |
| `baseline_naive_sazonal_painel` | 1 mês | 1.728 | 0,6713 | 1,0602 | 117,08 | 0,8067 |
| `random_forest_painel` | 2 meses | 1.584 | **0,5174** | **0,7396** | 115,38 | **0,6004** |
| `ridge_painel` | 2 meses | 1.584 | 0,5241 | 0,7491 | **113,86** | 0,6082 |
| `baseline_naive_sazonal_painel` | 2 meses | 1.584 | 0,6700 | 1,0683 | 115,72 | 0,7774 |
| `random_forest_painel` | 3 meses | 1.440 | **0,5187** | **0,7449** | 113,07 | **0,6097** |
| `ridge_painel` | 3 meses | 1.440 | 0,5189 | 0,7513 | **110,38** | 0,6099 |
| `baseline_naive_sazonal_painel` | 3 meses | 1.440 | 0,6515 | 1,0462 | 114,27 | 0,7657 |

No nível desagregado (`área × grupo`), ambos os modelos multivariados de painel (`random_forest_painel` e `ridge_painel`) superam amplamente o baseline naïve sazonal em MAE, RMSE, sMAPE e MASE nos três horizontes avaliados.

### 4.5 Análise de Sentimento do Copom e Impacto na Previsão (Etapa 6)
A classificação via LLM dos 200 documentos distintos apresentou distribuição de `76` trechos neutros, `76` contracionistas e `48` expansionistas, com **kappa de Cohen de 0,3803** em relação ao léxico de referência (concordância moderada, refletindo a limitação do léxico simples frente ao contexto monetário).

Na avaliação walk-forward de 1 mês à frente (`impacto_sentimento_sarimax.csv`, 12 meses de `2025-10-01` a `2026-09-01`):

| Modelo | N | MAE | RMSE | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sarimax_sem_sentimento` | 12 | **0,1329** | **0,1668** | **63,08** | **0,8674** |
| `sarimax_com_sentimento` | 12 | 0,1430 | 0,1771 | 70,17 | 0,9335 |

A inclusão de `sentimento_llm_lag1` não reduziu o erro de previsão em $h=1$ mês na janela de teste, sugerindo que a sinalização do Copom atua em defasagens mais longas e já é parcialmente incorporada pelas variáveis macroeconômicas correntes.

---

## 5. Limitações e Próximos Passos

1. **Validação humana do sentimento:** a concordância foi medida entre o LLM e o léxico de referência (`kappa = 0,3803`); a validação humana (`manual_vs_llm`) requer o preenchimento dos 200 rótulos em `dados/tratados/rotulacao_manual.csv` e execução de `analise_sentimento.py validar`.
2. **Cobertura parcial dos trechos e ausência de notícias:** foram classificados 200 trechos representativos (1 por documento) das atas e comunicados do Copom; notícias econômicas diárias não foram incorporadas nesta versão.
3. **Amostra de teste reduzida para o teste de Diebold-Mariano:** com 10 a 12 observações fora da amostra por horizonte, o poder estatístico do teste é limitado, dificultando a rejeição da hipótese nula de igualdade preditiva.
4. **Sensibilidade do MAPE a valores próximos de zero:** variações mensais de IPCA próximas de `0,00%` distorcem o MAPE, exigindo leitura conjunta com MASE, MAE e RMSE.
5. **Próximos passos:** preencher a rotulagem humana para validar o kappa `manual_vs_llm`, classificar todos os 1.094 trechos de atas longas e incorporar notícias econômicas.
