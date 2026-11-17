# AGENTS.md — Orientações para Agentes de IA no Repositório

Este documento define o contexto operacional, a arquitetura e as regras metodológicas obrigatórias para agentes de IA que atuem neste projeto (**Previsão da Inflação Brasileira e Comunicação do Copom**).

---

## 1. Objetivo do Projeto

Investigar e prever a variação mensal do **IPCA** (%) por **região metropolitana/capital** (16 áreas) e **grupo de despesa** (9 grupos) nos horizontes de **$h = 1, 2, 3$ meses**, avaliando o ganho preditivo do tom das atas e comunicados oficiais do **Copom**, com comparação obrigatória contra a mediana das expectativas **Focus**.

Os requisitos acadêmicos completos estão descritos em [projeto.md](projeto.md).

---

## 2. Ambiente e Comandos de Execução

- **Sistema operacional / Shell:** Windows (PowerShell).
- **Interpretador Python obrigatório:** `.venv\Scripts\python.exe` (Python 3.14+).
- **Instalação de dependências:**
  ```powershell
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

### Ordem Canônica do Pipeline

1. **Coleta e montagem do painel (`Etapa 7`):**
   ```powershell
   .venv\Scripts\python.exe src\coleta_dados.py
   ```
2. **Análise exploratória — EDA (`Etapa 1`):**
   ```powershell
   .venv\Scripts\python.exe src\analise_eda.py
   ```
3. **Correlação, CCF, Granger e VIF (`Etapa 2`):**
   ```powershell
   .venv\Scripts\python.exe src\analise_correlacao.py
   ```
4. **Análise de sentimento do Copom (`Etapa 6`):**
   ```powershell
   .venv\Scripts\python.exe src\analise_sentimento.py coletar
   .venv\Scripts\python.exe src\analise_sentimento.py preparar
   # Classificação via API configurada no .env OU importação de JSON pronto:
   .venv\Scripts\python.exe src\analise_sentimento.py classificar --somente-amostra
   .venv\Scripts\python.exe src\analise_sentimento.py importar-json
   .venv\Scripts\python.exe src\analise_sentimento.py agregar --permitir-parcial
   # Após preenchimento humano de dados/tratados/rotulacao_manual.csv:
   .venv\Scripts\python.exe src\analise_sentimento.py validar
   ```
5. **Modelagem multivariada, walk-forward, métricas e Diebold-Mariano (`Etapas 3, 4 e 5`):**
   ```powershell
   .venv\Scripts\python.exe src\modelagem_previsao.py
   ```

---

## 3. Arquitetura dos Módulos (`src/`)

| Arquivo | Responsabilidade Principal | Entradas | Principais Saídas (`dados/tratados/` e `figuras/`) |
| :--- | :--- | :--- | :--- |
| `src/coleta_dados.py` | Coleta IBGE/SIDRA (1419, 7060), BCB/SGS e Focus | APIs públicas | `ipca_painel.csv`, `macro_mensal.csv`, `focus_expectativas.csv`, `painel_ipca_macro.csv`, `dicionario_variaveis.csv` |
| `src/analise_eda.py` | Descritivas, ausentes, IQR, STL, ADF/KPSS, ACF/PACF | `ipca_painel.csv`, `macro_mensal.csv` | `resumo_descritivo_unidade_grupo.csv`, `perfil_ausentes.csv`, `outliers_iqr.csv`, `componentes_stl.csv`, `testes_estacionariedade.csv`, `acf_pacf.csv`, 4 PNGs |
| `src/analise_correlacao.py` | Pearson/Spearman, CCF, Granger, VIF com constante, conclusão | `painel_ipca_macro.csv`, `macro_mensal.csv` | `matriz_correlacao_*.csv`, `correlacoes_defasadas.csv`, `selecao_variaveis_correlacao.csv`, `causalidade_granger.csv`, `vif_variaveis.csv`, `conclusao_selecao_variaveis.csv`, `correlacao_ccf_heatmap.png` |
| `src/analise_sentimento.py` | Coleta BCB/Copom, trechos, léxico, LLM (`.env`/JSON), kappa | API Copom / `rotulo_llm.json` / `rotulacao_manual.csv` | `documentos_copom.csv`, `trechos_sentimento.csv`, `rotulacao_manual.csv`, `sentimento_documentos.csv`, `sentimento_mensal.csv`, `painel_ipca_macro_sentimento.csv`, `dicionario_sentimento.csv` |
| `src/modelagem_previsao.py` | Walk-forward $h=1..3$, Focus, 4 famílias, painel, DM, ICs | `painel_ipca_macro.csv`, `macro_mensal.csv`, `focus_expectativas.csv`, `sentimento_mensal.csv` | `avaliacao_modelos.csv`, `avaliacao_modelos_por_horizonte.csv`, `previsoes_walk_forward.csv`, `teste_diebold_mariano.csv`, `avaliacao_painel_area_grupo.csv`, `previsoes_painel_area_grupo.csv`, `impacto_sentimento_sarimax.csv`, `previsoes_intervalos_modelos.png` |

---

## 4. Invariantes e Regras Metodológicas Estritas

Qualquer modificação feita por agentes neste repositório **deve preservar rigorosamente** as seguintes regras:

1. **Zero vazamento temporal (*no data leakage*):**
   - **Nunca** embaralhar séries temporais (`shuffle=False` sempre).
   - A validação deve ser estritamente temporal (**walk-forward / origem móvel**).
   - Expectativas **Focus** devem ser alinhadas pela origem da coleta (`data_coleta` $\le$ fim do mês de origem $t$, `base_calculo == 0`, `mes_referencia == t + h`).
   - Escores de sentimento do Copom entram sempre defasados (`sentimento_llm_lag1`).
   - Padronizações (`StandardScaler`, normalização da LSTM) devem ser ajustadas apenas sobre os dados de treino até a origem corrente.

2. **Preservação de ausentes e outliers no alvo:**
   - **Não imputar nem interpolar** valores ausentes de `variacao_ipca_mensal_pct` (`2.487` registros nulos preservados no painel); testes estatísticos devem operar apenas sobre segmentos contínuos observados.
   - **Não remover** observações sinalizadas por `1,5 × IQR` (`1.239` registros), pois representam choques econômicos reais.

3. **Proibição de dados ou rótulos fabricados:**
   - **Nunca** preencher automaticamente ou simular rótulos humanos em `dados/tratados/rotulacao_manual.csv`.
   - **Nunca** fabricar escores LLM sintéticos; usar apenas retornos reais de API ou arquivos explícitos fornecidos pelo usuário (como `dados/tratados/rotulo_llm.json`).

4. **Consistência de métricas e modelagem:**
   - Como o IPCA mensal assume valores próximos de zero (ou negativos), **MAPE** sofre explosão numérica; reportar sempre **MAE, RMSE, sMAPE e MASE** em conjunto.
   - Calcular **VIF sempre com constante** (`sm.add_constant`) e reportar também em **primeira diferença** para variáveis em nível com tendência.
   - No **VAR**, usar variáveis macroeconômicas diferenciadas (`_diff1`) para garantir estacionariedade.

5. **Segurança e Git:**
   - Credenciais (`COPOM_LLM_API_KEY`, `COPOM_LLM_BASE_URL`, `COPOM_LLM_MODEL`) devem residir apenas em `.env` (listado no `.gitignore`) ou variáveis de ambiente — **nunca** comitar chaves no Git.
   - No Windows em ambiente sandboxed, comandos `git commit` podem exigir execução fora da sandbox se houver bloqueio de escrita em `.git/index.lock`.
