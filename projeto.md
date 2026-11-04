# Projetos da disciplina Séries Temporais e Análise Preditiva

## Entregáveis:
notebook reprodutível (Python ou R) em repositório GitHub ou pasta compartilhada;
base de dados tratada + dicionário de variáveis (fonte, unidade, frequência, período);
slides da apresentação;
relatório técnico curto (até 8 páginas) com metodologia, resultados e limitações.

## Requisitos obrigatórios

O projeto segue 7 etapas; a avaliação considera o cumprimento de todas, não apenas o melhor erro de previsão.

1. Análise exploratória (EDA). Estatísticas descritivas por unidade; gráficos das séries; decomposição STL (tendência, sazonalidade, resíduo); testes de estacionariedade ADF e KPSS; ACF/PACF; tratamento justificado de faltantes e outliers.

2. Análise de correlação. Matriz de correlação (Pearson e Spearman); correlação cruzada com defasagens (CCF) entre o alvo e cada variável exógena; causalidade de Granger; multicolinearidade (VIF). Conclusão obrigatória: quais variáveis e quais defasagens entram no modelo.

3. Predição multivariada com técnicas diversas. No mínimo 4 modelos, de pelo menos 3 famílias:

    referência (baseline): naïve sazonal ou média móvel;

    estatísticos: SARIMAX, VAR/VECM, regressão de Poisson ou Binomial Negativa, modelos de painel com efeitos fixos;

    aprendizado de máquina: Random Forest, XGBoost ou LightGBM com defasagens e janelas móveis;

    redes neurais: LSTM/GRU, N-BEATS/N-HiTS, Temporal Fusion Transformer;

    opcional: modelos fundacionais de séries temporais como comparador.

4. Validação temporal. Nunca embaralhar os dados: treino e teste separados no tempo, com validação walk-forward (origem móvel). Definir e justificar o horizonte de previsão (ex.: 4 semanas, 3 meses).

5. Métricas de avaliação. MAE, RMSE e MAPE em todos os modelos, mais MASE para comparar com o baseline (use sMAPE ou MASE quando a série tiver zeros). Quando houver evento a detectar (surto, onda de calor, colapso hídrico), incluir precisão, recall e F1. Comparar os dois melhores modelos com o teste de Diebold-Mariano e mostrar intervalos de previsão.

6. Análise de sentimentos com LLM (sempre que houver texto). Coletar textos datados (notícias, posts, fóruns, comunicados oficiais), classificar com um LLM (zero-shot ou few-shot) numa escala de -1 a +1 e agregar por unidade e período. Validar com pelo menos 200 textos rotulados manualmente (kappa de Cohen) e comparar com um método de referência (léxico ou BERTimbau). Rodar o melhor modelo com e sem a variável de sentimento para medir se ela melhora a previsão.

7. Estrutura de painel: várias unidades (municípios, estados, estações, alunos, ativos) observadas ao longo do tempo, com pelo menos 100 períodos ou 3 anos de histórico.

## Tema: Inflação e comunicação do Copom

Pergunta: qual será o IPCA mensal de cada região metropolitana (e de cada grupo de despesa) nos próximos 3 meses, e o tom das atas do Copom ajuda a prever?

Painel e alvo: regiões metropolitanas e capitais × 9 grupos × mês, de 2012 em diante (tabelas 1419 e 7060 do SIDRA); alvo = variação mensal do IPCA.

Explicativas: Selic, câmbio, preços de commodities, expectativas do Focus, IBC-Br, desocupação.
Sentimento (LLM): atas e comunicados do Copom classificados numa escala de -1 (expansionista) a +1 (contracionista), mais notícias econômicas.

Técnicas sugeridas: VAR/VECM, SARIMAX, LASSO/Ridge com muitos preditores, XGBoost, LSTM; comparar obrigatoriamente com a mediana do Focus como referência.

Dados: IBGE (SIDRA), BCB (SGS, Expectativas/Focus, documentos do Copom), Ipeadata.

## Fonte de dados

As fontes abaixo são públicas e gratuitas; a maioria tem API ou pacote em Python/R, o que facilita montar o painel de forma reprodutível

https://dadosabertos.bcb.gov.br/
https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado
https://dadosabertos.bcb.gov.br/dataset/atas-comunicados-copom