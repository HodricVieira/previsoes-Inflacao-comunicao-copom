# Melhoria Preditiva dos Modelos de Inflação (Abordagem 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer todos os 5 modelos próprios na série nacional (`var_multivariado`, `sarimax_exogeno`, `regressao_ridge`, `random_forest`, `lstm_multivariado`) e os 2 modelos de painel (`ridge_painel`, `random_forest_painel`) superarem com folga o `baseline_naive_sazonal` em validação walk-forward ($h = 1, 2, 3$ meses), preservando 100% das invariantes de zero vazamento temporal do `AGENTS.md`.

**Architecture:** Refatora `src/modelagem_previsao.py` para construir features alinhadas ao mês-alvo $\tau = t + h$ mas estritamente observadas na origem $t$ (`focus_tau` coletado em $t$ para $t+h$ com `base_calculo == 0`, defasagem sazonal $y_{\tau-12}$, perfil sazonal mensal histórico calculado exclusivamente no treino até $t$, e primeiras diferenças macroeconômicas nas defasagens selecionadas pela Etapa 2: `selic_d1_lag3`, `ic_d1_lag11`, `ibc_d1_lag4`, `desoc_d1_lag4`). Substitui a codificação ordinal do painel (`16 áreas × 9 grupos`) por efeitos fixos via `OneHotEncoder` + perfil sazonal local calculado no treino. Sincroniza os CSVs, figura e relatórios/notebooks (ignorando modificações na Etapa 6).

**Tech Stack:** Python 3.14 (`.venv\Scripts\python.exe`), `pandas`, `numpy`, `scikit-learn` (`RidgeCV`, `RandomForestRegressor`, `OneHotEncoder`, `ColumnTransformer`, `StandardScaler`), `statsmodels` (`SARIMAX`, `VAR`), `torch` (`nn.LSTM`), `pytest` / `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-18-melhoria-modelos-previsao-design.md`

## Global Constraints

- **Interpretador obrigatório:** `.venv\Scripts\python.exe`
- **Zero vazamento temporal (`AGENTS.md` Regra 1):** `shuffle=False` sempre; expectativas Focus filtradas por `base_calculo == 0`, `origem_mes == t` (`data_coleta <= fim do mês de origem t`) e `mes_referencia == t + h`; escalonadores (`StandardScaler`, normalização da LSTM) e medianas sazonais mensais ajustados exclusivamente nos dados de treino até a origem corrente $t$.
- **Preservação de ausentes e outliers no alvo (`AGENTS.md` Regra 2):** nunca imputar `variacao_ipca_mensal_pct` (`2.487` nulos no painel); `ffill()` é permitido apenas em variáveis exógenas de entrada até a origem corrente $t$ para evitar que `ibc_br_sa` nulo em `2026-09-01` descarte a 33ª previsão da LSTM (`2026-08-01 -> 2026-09-01`).
- **Consistência de modelagem (`AGENTS.md` Regra 4):** reportar sempre `mae`, `rmse`, `mape_pct`, `smape_pct` e `mase`; no `VAR`, usar variáveis macroeconômicas diferenciadas (`_d1` / `_diff1`).
- **Ignorar Etapa 6:** manter `_evaluate_sentiment_effect` funcionando sem quebrar `impacto_sentimento_sarimax.csv`, sem alterar `src/analise_sentimento.py`.

## Review Focus

1. **Perda da 33ª previsão da LSTM em `2026-09-01`:** `macro_mensal.csv` tem `ibc_br_sa = NaN` em `2026-09-01`; aplicar `.ffill()` nas séries exógenas antes de `.diff(1)` garante que todas as 33 previsões (`12 + 11 + 10`) sejam emitidas por todos os 7 modelos.
2. **Vazamento de expectativas Focus no treino de `SARIMAX`/`VAR`/`LSTM`:** ao montar a série histórica `focus_tau` para um mês de treino $\tau$, usar `focus_h1_self` (expectativa coletada em $\tau - 1$ para $\tau$), e nas etapas de previsão $\tau = t + h$ usar `focus_lookup[(t, t + h)]` (coletada na origem $t$ para $t + h$).
3. **Vazamento no perfil sazonal mensal (`perfil_mes` / `local_saz`):** calcular a mediana mensal de cada mês do calendário (`1..12`) exclusivamente sobre a fatia de treino `< split_date` (no painel) ou `<= origin_idx` (no walk-forward nacional).
4. **Compatibilidade com `notebooks/projeto_completo_reprodutivel.ipynb`:** manter as assinaturas públicas e nomes de arquivos CSV/PNG gerados por `src/modelagem_previsao.py` inalterados.
5. **Reprodutibilidade determinística:** fixar `torch.manual_seed(42)`, `np.random.seed(42)` e `random_state=42` em todos os estimadores.

---

### Task 1: Salvar Spec Aprovada e Criar Suíte de Testes (`tests/test_modelagem_previsao.py`)

**Files:**
- Create: `docs/superpowers/specs/2026-11-18-melhoria-modelos-previsao-design.md`
- Create: `docs/superpowers/plans/2026-11-18-melhoria-modelos-previsao.md`
- Create: `tests/test_modelagem_previsao.py`

**Interfaces:**
- Consumes: `src.modelagem_previsao.gerar_modelos() -> pd.DataFrame` e CSVs em `dados/tratados/`
- Produces: testes automatizados verificando que todos os 5 modelos próprios batem `baseline_naive_sazonal` em MAE, RMSE e MASE, possuem 33 previsões completas e que os modelos de painel superam `baseline_naive_sazonal_painel`.

- [ ] **Step 1: Salvar a especificação aprovada em `docs/superpowers/specs/2026-11-18-melhoria-modelos-previsao-design.md` e este plano em `docs/superpowers/plans/2026-11-18-melhoria-modelos-previsao.md`**
- [ ] **Step 2: Escrever o teste em `tests/test_modelagem_previsao.py` (que deve falhar inicialmente antes da refatoração de `src/modelagem_previsao.py`)**
- [ ] **Step 3: Executar o teste para confirmar a falha inicial no estado atual**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
Expected: FAIL (`var_multivariado` RMSE >= `naive` e `lstm_multivariado` com 32 previsões).

---

### Task 2: Refatorar `src/modelagem_previsao.py` com a Abordagem 1

**Files:**
- Modify: `src/modelagem_previsao.py`
- Test: `tests/test_modelagem_previsao.py`

**Interfaces:**
- Mantém intactas as funções e constantes exportadas: `EXOG_FEATURES`, `HORIZONS`, `TEST_MONTHS`, `LSTM_LOOKBACK`, `_LSTMRegressor`, `_load_base()`, `_load_focus_benchmark()`, `_build_lagged_features(df)`, `_metrics(...)`, `_diebold_mariano(...)`, `_fit_lstm_and_predict(df, split_index, focus=None)`, `_walk_forward_monthly(df, focus)`, `_panel_forecast_evaluation()`, `_evaluate_sentiment_effect(df)`, `gerar_modelos()`.

- [ ] **Step 1: Atualizar `_build_lagged_features(df: pd.DataFrame) -> pd.DataFrame` em `src/modelagem_previsao.py`**
- [ ] **Step 2: Atualizar `_fit_lstm_and_predict(df: pd.DataFrame, split_index: int, focus: pd.DataFrame | None = None) -> dict[tuple[int, int], float]` em `src/modelagem_previsao.py`**
- [ ] **Step 3: Atualizar `_walk_forward_monthly(df: pd.DataFrame, focus: pd.DataFrame)` em `src/modelagem_previsao.py`**
- [ ] **Step 4: Atualizar `_panel_forecast_evaluation()` em `src/modelagem_previsao.py`**
- [ ] **Step 5: Executar `src/modelagem_previsao.py` e rodar a suíte de testes**

---

### Task 3: Sincronizar Relatório Técnico, Slides, Documentação e Notebook

**Files:**
- Modify: `relatorio_tecnico.md`
- Modify: `slides_apresentacao.md`
- Modify: `DOCUMENTACAO.md`
- Modify: `notebooks/projeto_completo_reprodutivel.ipynb`

**Interfaces:**
- Consumes: `dados/tratados/avaliacao_modelos.csv`, `dados/tratados/avaliacao_modelos_por_horizonte.csv`, `dados/tratados/teste_diebold_mariano.csv`, `dados/tratados/avaliacao_painel_area_grupo.csv`.
- Produces: tabelas e interpretações atualizadas nos documentos e saídas reexecutadas no notebook.

- [ ] **Step 1: Atualizar as tabelas e comentários da Seção 3.3 e Seção 4.2–4.4 em `relatorio_tecnico.md`, `slides_apresentacao.md` e `DOCUMENTACAO.md`**
- [ ] **Step 2: Reexecutar as células de modelagem de `notebooks/projeto_completo_reprodutivel.ipynb`**
- [ ] **Step 3: Verificar integridade final com os testes**
