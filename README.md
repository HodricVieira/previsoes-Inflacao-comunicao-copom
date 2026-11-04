# Inflação e comunicação do Copom

Base reprodutível e análise exploratória do IPCA mensal por área geográfica e
grupo de despesa, com variáveis macroeconômicas brasileiras e expectativas
Focus.

## Conteúdo

- `src/coleta_dados.py`: coleta e valida as fontes oficiais.
- `src/analise_eda.py`: estatísticas, valores ausentes, sinalização IQR,
  decomposição STL, ADF/KPSS, ACF/PACF e gráficos.
- `notebooks/analise_exploratoria.ipynb`: EDA independente a partir do painel
  tratado `dados/tratados/painel_ipca_macro.csv`.
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
- `dados/tratados/*.csv`: tabelas de resultados da EDA.
- `figuras/*.png`: gráficos gerados pela EDA.

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
```

Também é possível executar as células de
`notebooks/analise_exploratoria.ipynb` em sequência. O notebook usa somente o
CSV integrado incluído em `dados/tratados/`; não consulta APIs nem importa
módulos do projeto. Para atualizar os dados, execute separadamente os scripts
de coleta e análise acima.

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

Esta entrega cobre a obtenção, estruturação e EDA quantitativa. Coleta de
notícias/documentos, classificação de sentimento e validação humana do LLM
pertencem à etapa de texto do projeto e não são simuladas aqui.
