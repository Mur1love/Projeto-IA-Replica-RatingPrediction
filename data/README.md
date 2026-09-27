# Dados

## Origem

Os dados são derivados do dataset de reviews da Amazon Brasil utilizado no artigo
*Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual
Features* (Marreira, Oliveira, Melo). Réplica baseada no repositório original:
[github.com/emanuellemarreira/rating-prediction-with-textual-features](https://github.com/emanuellemarreira/rating-prediction-with-textual-features).

- **Licença**: não especificada pelo projeto de origem. Os arquivos
  (`amazon_original_dataset`, `train.csv`, `test.csv`) foram obtidos de um
  projeto anterior que não documenta licença de uso, apenas a origem dos dados
  (reviews da Amazon Brasil). **Sem licença explícita, o uso deve ser
  restrito a fins acadêmicos/educacionais deste projeto**; evitar redistribuição
  pública dos CSVs além do necessário para a disciplina.
- **Forma de obtenção**: arquivos baixados diretamente do repositório GitHub do
  projeto original (réplica anterior do mesmo artigo), sem scraping ou coleta
  própria.

## Splits

Os splits `train.csv` e `test.csv` são os splits oficiais utilizados no artigo
replicado:

| Split | Linhas | Proporção |
| --- | --- | --- |
| `train.csv` | 41.172 | 80% |
| `test.csv` | 10.293 | 20% |
| **Total** | 51.465 | 100% |

As classes de `rating` (1 a 5 estrelas) estão balanceadas em ambos os splits —
não refletem a distribuição natural de reviews (que tende a ser majoritariamente
positiva). Isso foi uma decisão já presente no dataset original, mantida na réplica.

> **Nota sobre duplicatas entre splits**: foram encontradas 280 combinações
> idênticas de (`categoria`, `text`, `rating`) presentes tanto em `train.csv`
> quanto em `test.csv` — 41 delas são reviews longas e específicas (não apenas
> textos curtos genéricos como "Ok" ou "10", que naturalmente se repetem entre
> usuários diferentes). Isso é uma limitação herdada do split oficial do
> artigo, mantida aqui sem alteração conforme o escopo desta issue. Pode causar
> otimismo artificial nas métricas de avaliação (o modelo pode ter visto
> exemplos do teste durante o treino). Decisão sobre se e como tratar isso
> deve ser registrada na Issue 11.

## Estrutura das colunas

Cada CSV contém:

- `categoria`: categoria do produto avaliado (`auto`, `baby`, `celular`, `food`,
  `games`, `laptops`, `livros`, `moda`, `pets`, `toys`).
- `text`: texto da review, em português brasileiro.
- `rating`: nota atribuída pelo usuário (1 a 5 estrelas). Variável alvo.
- **58 features textuais** extraídas conforme o artigo, agrupadas em:
  - estruturais (`str_*`): contagem/proporção de caracteres, sentenças,
    palavras, maiúsculas;
  - classe gramatical / POS (`pos_*`): contagem de cada classe gramatical
    (substantivos, adjetivos, verbos, etc.);
  - sintáticas (`synt_1` a `synt_5`);
  - léxicas (`lex_*`): subjetividade, polaridade positiva/negativa, sentenças
    negativas/positivas, uso de exclamação;
  - concretude (`conc_*`): prazer, atenção, sensibilidade, aptidão, polaridade;
  - estilo "twitter" (`twt_*`): palavras elongadas, expressões, negação,
    polaridade de emoji/emoticon;
  - subjetividade (`sbj_*`): contagem de erros/palavras corretas.

> **Nota sobre escopo**: as 58 features acima **não serão utilizadas no escopo
> inicial** do projeto. O baseline inicial utiliza apenas `text` normalizado +
> vetorização TF-IDF. As features serão incorporadas em uma etapa posterior,
> conforme o estudo de ablação previsto no artigo.

## Arquivos não versionados

`train.csv` e `test.csv` não são versionados no Git — já cobertos pelo
`.gitignore` do projeto (ver `data/raw/` e `data/processed/`). Para obter os
arquivos localmente, ver seção "Forma de obtenção" acima.
