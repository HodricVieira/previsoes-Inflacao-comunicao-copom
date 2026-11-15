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
  2. *Estatísticos/Lineares:* **VAR(4)** (com primeiras diferenças das variáveis macroeconômicas para estacionariedade), **SARIMAX(1,0,1)(1,0,0,12)** com exógenas defasadas e **Regressão Ridge** com defasagens (1, 3, 6, 12) e janelas móveis (média 3m, 6m e desvio 3m).
  3. *Aprendizado de Máquina:* **Random Forest** (`200` árvores, `max_depth=6`) com as mesmas defasagens e janelas móveis.
  4. *Redes Neurais:* **LSTM** em PyTorch (janela `lookback=12`, camada oculta de 16 unidades, previsão recursiva).
- **Previsão desagregada no painel (16 áreas × 9 grupos):** modelos globais de painel (`ridge_painel` e `random_forest_painel`) com codificação de área e grupo, defasagens locais e exógenas macroeconômicas, comparados ao `baseline_naive_sazonal_painel`.
- **Avaliação estatística e incerteza:** métricas **MAE, RMSE, MAPE, sMAPE e MASE** (com escala naïve sazonal in-sample), teste bilateral de **Diebold-Mariano** (variância HAC de Bartlett para horizonte $h$) e **intervalos de previsão empíricos de 80% e 95%**.

### 3.4 Pipeline de Análise de Sentimento (Copom)
O módulo `src/analise_sentimento.py` segmenta os 238 documentos em trechos (`trechos_sentimento.csv`), calcula um escore léxico de referência em 149 meses (`escore_lexico`), prepara uma amostra cega de 200 documentos distintos (`rotulacao_manual.csv`) para cálculo do **kappa de Cohen** e integra o escore LLM defasado (`sentimento_llm_lag1`) ao SARIMAX walk-forward para medir ganho preditivo.

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
| `focus_mediana` (referência) | 33 | **0,1351** | **0,1526** | 123,45 | 58,45 | **0,7168** |
| `baseline_naive_sazonal` | 33 | 0,1391 | 0,1837 | 84,08 | **53,62** | 0,7382 |
| `var_multivariado` | 33 | 0,1734 | 0,1914 | 128,20 | 73,34 | 0,9202 |
| `sarimax_exogeno` | 33 | 0,1731 | 0,1918 | 136,06 | 72,46 | 0,9188 |
| `regressao_ridge` | 33 | 0,1737 | 0,2042 | 105,71 | 73,21 | 0,9220 |
| `random_forest` | 33 | 0,1798 | 0,2062 | 139,57 | 70,80 | 0,9544 |
| `lstm_multivariado` | 32 | 0,1965 | 0,2751 | 85,72 | 93,39 | 1,0427 |

*Nota sobre MAPE:* como variações mensais do IPCA assumem valores próximos de zero (e ocasionais deflações), o MAPE sofre explosão numérica; por isso, **MASE** e **sMAPE** são as métricas recomendadas para comparação relativa.

Por horizonte (`avaliacao_modelos_por_horizonte.csv`), o destaque dos modelos próprios ocorre em **$h = 1$ mês**: `var_multivariado` (MAE = 0,1478; RMSE = 0,1711; MASE = 0,7846) e `sarimax_exogeno` (MAE = 0,1526; RMSE = 0,1719) superam o `baseline_naive_sazonal` (RMSE = 0,2007) e aproximam-se da `focus_mediana` (RMSE = 0,1625). Em $h = 2$ e $h = 3$, a `focus_mediana` (RMSE = 0,1517 e 0,1408) mantém vantagem consistente por incorporar expectativas correntes de alta frequência dos agentes de mercado.

Os intervalos de previsão de 80% e 95% para $h = 1$ mês estão ilustrados em `figuras/previsoes_intervalos_modelos.png`.

### 4.3 Teste de Diebold-Mariano
A comparação estatística (`teste_diebold_mariano.csv`) entre os dois melhores modelos próprios (`var_multivariado` vs. `sarimax_exogeno`) não indicou diferença estatisticamente significativa nos três horizontes ($p = 0,9662$ em $h=1$; $p = 0,9165$ em $h=2$; $p = 0,8961$ em $h=3$ para MSE). Da mesma forma, na janela de 10 a 12 pares de teste, a diferença entre `var_multivariado` e `focus_mediana` não atingiu significância a 5% ($p = 0,8096$ em $h=1$; $p = 0,1118$ em $h=3$).

### 4.4 Desempenho no Painel Completo (16 Áreas × 9 Grupos)

| Modelo | Horizonte | N previsões | MAE | RMSE | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge_painel` | 1 mês | 1.728 | **0,5956** | **0,9070** | 119,27 | **0,7157** |
| `random_forest_painel` | 1 mês | 1.728 | 0,6071 | 0,9095 | 124,94 | 0,7296 |
| `baseline_naive_sazonal_painel` | 1 mês | 1.728 | 0,6713 | 1,0602 | **117,08** | 0,8067 |
| `ridge_painel` | 2 meses | 1.584 | **0,6037** | **0,9316** | 122,77 | **0,7005** |
| `random_forest_painel` | 2 meses | 1.584 | 0,6080 | 0,9397 | 131,51 | 0,7055 |
| `baseline_naive_sazonal_painel` | 2 meses | 1.584 | 0,6700 | 1,0683 | **115,72** | 0,7774 |
| `ridge_painel` | 3 meses | 1.440 | **0,6001** | **0,9445** | 119,89 | **0,7053** |
| `random_forest_painel` | 3 meses | 1.440 | 0,6377 | 0,9725 | 117,69 | 0,7496 |
| `baseline_naive_sazonal_painel` | 3 meses | 1.440 | 0,6515 | 1,0462 | **114,27** | 0,7657 |

No nível desagregado (`área × grupo`), ambos os modelos multivariados de painel (`ridge_painel` e `random_forest_painel`) superam o baseline naïve sazonal em MAE, RMSE e MASE nos três horizontes avaliados.

---

## 5. Limitações e Próximos Passos

1. **Etapa de sentimento com LLM pendente de execução:** embora os 238 documentos do Copom já tenham sido coletados e o escore léxico esteja calculado em 149 meses, a classificação via LLM (`escore_llm`), o preenchimento dos 200 rótulos humanos em `dados/tratados/rotulacao_manual.csv` (para cálculo do kappa de Cohen) e o comparativo `impacto_sentimento_sarimax.csv` dependem de rotulagem manual e configuração de chave de API (`COPOM_LLM_API_KEY`).
2. **Ausência de notícias econômicas:** o pipeline textual atual contempla apenas atas e comunicados oficiais do Copom; notícias econômicas diárias não foram incorporadas nesta versão.
3. **Amostra de teste reduzida para o teste de Diebold-Mariano:** com 10 a 12 observações fora da amostra por horizonte, o poder estatístico do teste é limitado, dificultando a rejeição da hipótese nula de igualdade preditiva.
4. **Sensibilidade do MAPE a valores próximos de zero:** variações mensais de IPCA próximas de `0,00%` distorcem o MAPE, exigindo leitura conjunta com MASE, MAE e RMSE.
5. **Próximos passos:** concluir a rotulagem manual dos 200 trechos, executar a classificação LLM e validar o impacto do tom do Copom no SARIMAX; incorporar notícias econômicas; e consolidar os resultados na apresentação em slides.
