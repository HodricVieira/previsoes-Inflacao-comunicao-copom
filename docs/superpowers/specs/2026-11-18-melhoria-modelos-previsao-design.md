# Design Curto — Melhoria Preditiva dos Modelos Atuais (Abordagem 1)

**Data:** 2026-11-18  
**Escopo:** Melhoria dos modelos de previsão da inflação (`src/modelagem_previsao.py`) nas Etapas 3, 4, 5 e 7, mantendo a Etapa 6 (sentimento) inalterada.

## 1. Intenção e Porquê

Fazer todos os 5 modelos próprios na série nacional (`var_multivariado`, `sarimax_exogeno`, `regressao_ridge`, `random_forest`, `lstm_multivariado`) e os 2 modelos de painel (`ridge_painel`, `random_forest_painel`) superarem o `baseline_naive_sazonal` em MAE, RMSE e MASE no protocolo walk-forward ($h = 1, 2, 3$ meses), eliminando a cegueira sazonal em $h = 2, 3$, o excesso de colinearidade em amostra curta (~150 meses) e o descarte da 33ª previsão da LSTM (`2026-08-01 -> 2026-09-01`).

## 2. Objetivos, Não-Objetivos e Anti-Objetivos

- **Objetivos:**
  - Superar o `baseline_naive_sazonal` (`MAE < 0,1391`, `RMSE < 0,1837`, `MASE < 0,7382`) com todos os 5 modelos próprios na avaliação consolidada nacional (`n_previsoes == 33` para todos os modelos).
  - Recuperar a 33ª previsão da `lstm_multivariado` (`2026-08-01 -> 2026-09-01`), afetada pelo `NaN` de `ibc_br_sa` em `2026-09-01`.
  - Reduzir o erro de `ridge_painel` e `random_forest_painel` no painel completo (`16 áreas × 9 grupos`), superando `baseline_naive_sazonal_painel` em todos os horizontes ($h = 1, 2, 3$).
- **Não-Objetivos:**
  - Não alterar a Etapa 6 (`src/analise_sentimento.py`), preservando a geração de `impacto_sentimento_sarimax.csv`.
  - Não adicionar novas famílias de modelos além das já existentes (`var_multivariado`, `sarimax_exogeno`, `regressao_ridge`, `random_forest`, `lstm_multivariado`, `ridge_painel`, `random_forest_painel`).
- **Anti-Objetivos (Invariantes do `AGENTS.md`):**
  - Zero vazamento temporal: `shuffle=False` sempre; `focus_expectativas.csv` filtrado por `base_calculo == 0`, `origem_mes == t` (`data_coleta <= fim do mês de origem t`) e `mes_referencia == t + h`; escalonadores (`StandardScaler`, normalização da LSTM) e medianas sazonais mensais calculados exclusivamente nos dados de treino até a origem corrente $t$.
  - Preservação dos `2.487` valores nulos e `1.239` outliers (`1,5 × IQR`) no alvo `variacao_ipca_mensal_pct` (sem imputação no alvo).
  - No `VAR`, usar variáveis macroeconômicas diferenciadas (`_diff1`) para garantir estacionariedade.

## 3. Abordagens Consideradas

1. **Abordagem 1 (Escolhida):** Features sazonais alinhadas ao mês-alvo $\tau = t + h$ + Expectativa Focus (`base_calculo == 0`) como exógena conhecida na origem $t$ + Seleção enxuta de variáveis/defasagens da Etapa 2 (`conclusao_selecao_variaveis.csv`) + Efeitos fixos via `OneHotEncoder` no painel. Escolhida por maximizar a precisão nos 3 horizontes e cumprir o uso do Focus como preditora previsto em `projeto.md`.
2. **Abordagem 2:** Melhoria apenas macro/sazonal sem Focus dentro dos modelos próprios (descartada por perder sinal preditivo de curto prazo nos horizontes $h = 2, 3$).
3. **Abordagem 3:** Modelagem residual pura sobre âncora sazonal (aproveitada parcialmente na parametrização regularizada de Random Forest e LSTM dentro da Abordagem 1).

## 4. Decisões de Implementação

1. **Engenharia de preditores alinhados ao mês-alvo $\tau = t + h$ (observados em $t$):**
   - `focus_tau`: mediana Focus coletada no mês de origem $t$ para $t + h$ (`base_calculo == 0`); no histórico de treino para cada mês $\tau$, usa a expectativa coletada em $\tau - 1$ para $\tau$ (`focus_h1_self`).
   - `y_lag12`: valor observado 12 meses antes do mês-alvo ($y_{\tau - 12}$, sempre conhecido em $t$ pois $h \le 3$).
   - `perfil_mes`: mediana histórica de `ipca_mensal_median` para o mês do calendário de $\tau$, calculada estritamente sobre `train_slice` até a origem $t$.
   - Preditores macroeconômicos em primeira diferença guiados pela Etapa 2 (`selic_d1_lag3`, `ic_d1_lag11`, `ibc_d1_lag4`, `desoc_d1_lag4`), aplicando `.ffill()` apenas nas exógenas antes de `.diff(1)` para não descartar a observação `2026-09-01`.
2. **Calibração dos 5 modelos nacionais:**
   - `sarimax_exogeno`: `SARIMAX(..., order=(1, 0, 0))` com exógenas alinhadas ao mês-alvo (`focus_tau`, `y_lag12`, `perfil_mes`, `selic_d1_lag3`, `ic_d1_lag11`, `ibc_d1_lag4`, `desoc_d1_lag4`).
   - `var_multivariado`: `VAR(endog=[ipca_mensal_median, *_diff1], exog=[focus_tau, y_lag12, perfil_mes]).fit(1)`.
   - `regressao_ridge`: `StandardScaler + RidgeCV(alphas=np.logspace(0, 3, 25))` direto por horizonte $h$.
   - `random_forest`: `RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=8, random_state=42)` prevendo ajuste residual sobre a âncora `0.75 * focus_tau + 0.25 * y_lag12`.
   - `lstm_multivariado`: `_LSTMRegressor(input_size=7, hidden_size=8)` com `Adam(lr=0.005, weight_decay=0.25)` por `80` épocas sobre sequência padronizada no treino prevendo o resíduo padronizado sobre a âncora, com todas as **33 previsões** emitidas.
3. **Painel (`16 áreas × 9 grupos`):**
   - `ColumnTransformer` com `OneHotEncoder` em `["area_id", "grupo_id"]` (efeitos fixos) e `StandardScaler` nos preditores numéricos (`local_saz_h{h}` calculado apenas para `< split_date`, `focus_h{h}`, `naive_h{h}`, lags locais e `_diff1` macro).
   - `RidgeCV(alphas=np.logspace(0, 3, 15))` e `RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=8, random_state=42)`.
