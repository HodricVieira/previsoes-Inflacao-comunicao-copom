# Inflação e comunicação do Copom

Base reprodutível e análise exploratória do IPCA mensal por área geográfica e
grupo de despesa, com variáveis macroeconômicas brasileiras e expectativas
Focus.

## Conteúdo

- `src/coleta_dados.py`: coleta e valida as fontes oficiais.
- `src/analise_eda.py`: estatísticas, valores ausentes, sinalização IQR,
  decomposição STL, ADF/KPSS, ACF/PACF e gráficos.
- `src/analise_correlacao.py`: matrizes de Pearson/Spearman, CCF por defasagem,
  causalidade de Granger, VIF com constante e conclusão da seleção de variáveis.
- `src/analise_sentimento.py`: coleta atas e comunicados do Copom, prepara
  trechos, classifica via LLM (lendo `.env`) ou importa JSON (`rotulo_llm.json`),
  agrega escores por mês e calcula kappa de Cohen.
- `src/modelagem_previsao.py`: compara 4 famílias de modelos + Focus em
  walk-forward ($h = 1, 2, 3$ meses), prevê no painel completo (`16 áreas × 9 grupos`),
  calcula Diebold-Mariano, intervalos de previsão (80%/95%) e impacto do sentimento no SARIMAX.
- `notebooks/projeto_completo_reprodutivel.ipynb`: notebook completo,
  autocontido, modificável e visualizável cobrindo todas as 7 etapas obrigatórias.
- `notebooks/analise_exploratoria.ipynb`: EDA independente a partir do painel
  tratado `dados/tratados/painel_ipca_macro.csv`.
- `AGENTS.md`: diretrizes operacionais, arquitetura e invariantes metodológicas para agentes de IA.
- `DOCUMENTACAO.md`: documentação técnica completa do pipeline, CLI, catálogo de bases e notebooks.
- `relatorio_tecnico.md`: relatório técnico curto com metodologia, resultados e limitações.
- `slides_apresentacao.md`: slides da apresentação em formato Marp.
- `dados/tratados/ipca_painel.csv`: observações em formato longo
  (mês × área × grupo).
- `dados/tratados/macro_mensal.csv`: séries macroeconômicas mensais nacionais.
- `dados/tratados/focus_expectativas.csv`: expectativas com data de coleta,
  mês de referência e base de cálculo preservados.
- `dados/tratados/painel_ipca_macro.csv`: IPCA e variáveis macroeconômicas
  unidos pelo mês, sem incorporar expectativas com risco de vazamento temporal.
- `dados/tratados/dicionario_painel_ipca_macro.csv`: dicionário das 14 colunas
  do painel integrado, com descrição, fonte, unidade, frequência e tratamento.
- `dados/tratados/dicionario_variaveis.csv`: fonte, identificador, unidade,
  frequência, período e tratamento.
- `dados/tratados/documentos_copom.csv`: textos oficiais e metadados de atas e
  comunicados coletados pela API aberta do BCB.
- `dados/tratados/trechos_sentimento.csv`: trechos, escores LLM e léxico de
  referência.
- `dados/tratados/rotulo_llm.json`: rótulos LLM importados para os 200 trechos
  representativos de documentos distintos.
- `dados/tratados/rotulacao_manual.csv`: amostra reprodutível para validação
  humana; os rótulos devem ser preenchidos pelos anotadores.
- `dados/tratados/sentimento_mensal.csv`: escores LLM agregados por mês para
  integração temporal ao modelo.
- `dados/tratados/painel_ipca_macro_sentimento.csv`: painel IPCA/macro com
  sinal nacional do Copom defasado e repetido por área e grupo.
- `dados/tratados/dicionario_sentimento.csv`: definições das variáveis de
  sentimento e suas regras de agregação.
- `dados/tratados/*.csv`: tabelas de resultados da EDA, correlação/Granger/VIF,
  avaliação walk-forward (`avaliacao_modelos.csv`, `avaliacao_modelos_por_horizonte.csv`,
  `avaliacao_painel_area_grupo.csv`, `teste_diebold_mariano.csv`,
  `previsoes_walk_forward.csv`) e impacto do sentimento (`impacto_sentimento_sarimax.csv`).
- `figuras/*.png`: gráficos gerados pela EDA, correlação/CCF e previsões com intervalos.

## Fontes

- IBGE/SIDRA: tabelas [1419](https://sidra.ibge.gov.br/tabela/1419)
  (2012–2019) e [7060](https://sidra.ibge.gov.br/tabela/7060)
  (2020 em diante), variável 63, classificação 315 e os nove grupos principais.
  A coleta combina os níveis geográficos N6 e N7 disponibilizados pelas
  tabelas; a API determina as áreas e o período final disponível.
- Banco Central do Brasil: séries SGS 4189 (Selic efetiva mensal anualizada
  base 252), 3697 (câmbio),
  27574 (IC-Br), 24364 (IBC-Br dessazonalizado), 24369 (desocupação) e 433
  (IPCA Brasil).
- Banco Central do Brasil: API Expectativas de Mercado, indicador IPCA
  mensal. Para cada mês de coleta, a base preserva as previsões da última data
  disponível nos últimos dez dias do mês, incluindo `base_calculo`.

## Execução

No PowerShell, na pasta do projeto:

```powershell
python -m pip install -r requirements.txt
python src\coleta_dados.py
python src\analise_eda.py
python src\analise_correlacao.py
python src\analise_sentimento.py coletar
python src\analise_sentimento.py preparar
python src\modelagem_previsao.py
```

O módulo de correlação gera matrizes de Pearson/Spearman, CCF por defasagem,
testes de causalidade de Granger, VIF com constante (nível e 1ª diferença) e
conclusão de seleção de variáveis. O script de modelagem compara Focus, baseline
naïve sazonal, VAR, SARIMAX, Ridge, Random Forest e LSTM em validação temporal
walk-forward ($h = 1, 2, 3$ meses), além de modelos de painel (`16 áreas × 9 grupos`),
teste de Diebold-Mariano e intervalos de previsão de 80% e 95%.

### Análise de sentimento

`src/analise_sentimento.py coletar` baixa atas e comunicados desde 2012 da
[API oficial de documentos do Copom](https://dadosabertos.bcb.gov.br/dataset/atas-comunicados-copom).
`preparar` divide os textos em trechos e cria a amostra `rotulacao_manual.csv`.
Ela seleciona um trecho de cada um de 200 documentos distintos. Rotule
`rotulo_manual` usando somente `expansionista`, `neutro` ou
`contracionista`; a coluna opcional `rotulo_manual_2` permite medir concordância
entre anotadores. Aqui, expansionista significa viés por flexibilização/juros
menores; contracionista, viés por aperto/juros maiores; neutro, equilíbrio,
ambiguidade ou ausência de implicação de política monetária. Não confunda tom
com uma avaliação positiva/negativa da economia. A planilha omite escores
automáticos para manter a anotação cega e o script preserva os rótulos se a
preparação for repetida.

Para classificar todos os trechos dos 238 documentos via API OpenAI-compatible,
configure `.env` (ou variáveis de ambiente `COPOM_LLM_API_KEY`,
`COPOM_LLM_MODEL` e `COPOM_LLM_BASE_URL`) e execute a sequência completa (classificar todo o corpus, agregar, validar contra os 200 rótulos humanos de `rotulacao_manual.csv` e modelar):

```powershell
python src\analise_sentimento.py classificar
python src\analise_sentimento.py agregar
python src\analise_sentimento.py validar
python src\modelagem_previsao.py
```

Também é possível usar um endpoint local compatível com a API de chat
completions; nesse caso configure `COPOM_LLM_BASE_URL` para `localhost`.
Tanto `agregar` quanto `validar` exigem que 100% dos trechos de todos os 238
documentos (`1.094` trechos) já tenham sido classificados pelo LLM. `validar`
calcula o kappa de Cohen entre a amostra humana, o LLM e o léxico de
referência e exige no mínimo 200 rótulos humanos válidos de documentos
distintos. O léxico é um comparador simples, não substitui a validação. Os
escores são defasados em um mês na modelagem; meses sem nova publicação mantêm
o último escore conhecido e registram zero documentos novos.
`modelagem_previsao.py` gera `impacto_sentimento_sarimax.csv`, comparando
previsões SARIMAX walk-forward de um mês com e sem sentimento, nos mesmos meses
de teste.

O fluxo implementa a coleta de documentos oficiais do Copom. Notícias não são
incluídas nesta versão. A rotulagem humana e a execução do LLM dependem de
ações/configuração do usuário e não são simuladas nem preenchidas
automaticamente. Os escores do Copom são nacionais e, no painel, são repetidos
por área/grupo; isso não implica que exista sentimento local específico.

Também é possível executar as células de
`notebooks/projeto_completo_reprodutivel.ipynb` (todas as 7 etapas obrigatórias,
com parâmetros modificáveis na Seção 0) ou `notebooks/analise_exploratoria.ipynb`
em sequência. Os notebooks leem os arquivos tratados de `dados/tratados/`; para
atualizar os dados a partir das APIs, execute separadamente os scripts em `src/`.

## Decisões de tratamento

- A variação do IPCA é mantida na unidade publicada (% mensal). O painel não
  transforma a taxa em nível de índice.
- Códigos de ausência publicados pelo SIDRA são convertidos em nulos. Não há
  interpolação nem preenchimento automático; os testes usam apenas o trecho
  observado contínuo de cada série e são omitidos quando há lacunas internas.
- A regra de 1,5 × IQR apenas sinaliza observações para inspeção. Não remove
  valores; movimentos atípicos podem refletir eventos econômicos reais.
- STL usa sazonalidade mensal (`period=12`) e é calculada por série
  (área × grupo) no trecho observado contínuo. ADF (H0: raiz unitária) e
  KPSS (H0: estacionariedade) são apresentados em conjunto, com constante.
- As expectativas Focus são armazenadas separadamente do painel mensal. Uma
  junção para modelagem deve respeitar `data_coleta` e `mes_referencia` para
  evitar vazamento de informação futura.

Os textos, a classificação, a validação humana e o teste de contribuição do
sentimento são etapas separadas e reproduzíveis. Notícias econômicas ainda não
fazem parte da fonte textual implementada.
