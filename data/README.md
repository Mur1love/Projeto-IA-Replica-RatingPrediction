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
| --- | ---: | ---: |
| `train.csv` | 41.172 | 80% |
| `test.csv` | 10.293 | 20% |
| **Total** | **51.465** | **100%** |

As classes de `rating` (1 a 5 estrelas) estão balanceadas em ambos os splits —
não refletem a distribuição natural de reviews (que tende a ser majoritariamente
positiva). Isso foi uma decisão já presente no dataset original, mantida na réplica.

### Distribuição dos ratings

| Rating | Treino | Teste | Total |
| ---: | ---: | ---: | ---: |
| 1 | 8.234 | 2.059 | 10.293 |
| 2 | 8.234 | 2.059 | 10.293 |
| 3 | 8.235 | 2.058 | 10.293 |
| 4 | 8.235 | 2.058 | 10.293 |
| 5 | 8.234 | 2.059 | 10.293 |
| **Total** | **41.172** | **10.293** | **51.465** |

### Categorias de produto

O desbalanceamento **entre categorias** permanece, como no artigo: Livros e Moda
concentram mais avaliações; Computadores e Pets, menos. Os totais por categoria
conferem exatamente com a Tabela 1 do artigo.

| Categoria no CSV | Nome no artigo | Treino | Teste | Total |
| --- | --- | ---: | ---: | ---: |
| `auto` | Automotivo | 3.490 | 875 | 4.365 |
| `baby` | Bebê | 4.230 | 1.055 | 5.285 |
| `celular` | Celulares | 3.469 | 866 | 4.335 |
| `food` | Alimentos | 2.968 | 742 | 3.710 |
| `games` | Jogos | 4.870 | 1.215 | 6.085 |
| `laptops` | Computadores | 740 | 185 | 925 |
| `livros` | Livros | 9.035 | 2.260 | 11.295 |
| `moda` | Moda | 5.770 | 1.445 | 7.215 |
| `pets` | Pets | 1.780 | 445 | 2.225 |
| `toys` | Brinquedos | 4.820 | 1.205 | 6.025 |
| **Total** | | **41.172** | **10.293** | **51.465** |

## Balanceamento e undersampling

O artigo relata *undersampling* aleatório das classes majoritárias (4 e 5
estrelas) para igualá-las às minoritárias (1, 2 e 3). Os CSVs distribuídos já
refletem esse balanceamento: cada rating tem cerca de 8.234–8.235 linhas no
treino e 2.058–2.059 no teste.

**Não foi necessário aplicar undersampling adicional na réplica.** Nenhuma linha
foi removida por balanceamento. A única remoção no baseline TF-IDF é a de
**1 texto vazio por split**; o protocolo das 58 features mantém essas duas
linhas, porque as features já estão calculadas.

## Estrutura das colunas

Cada CSV contém **61 colunas**:

- `categoria`: categoria do produto avaliado (`auto`, `baby`, `celular`, `food`,
  `games`, `laptops`, `livros`, `moda`, `pets`, `toys`).
- `text`: texto da review, em português brasileiro.
- `rating`: nota atribuída pelo usuário (1 a 5 estrelas). Variável alvo.
- **58 features textuais** extraídas conforme o artigo, agrupadas em:
  - estruturais (`str_*`, 7): contagem/proporção de caracteres, sentenças,
    palavras, maiúsculas;
  - classe gramatical / POS (`pos_*`, 19): contagem de cada classe gramatical
    (substantivos, adjetivos, verbos, etc.);
  - sintáticas (`synt_*`, 5): padrões sintáticos;
  - léxicas (`lex_*`, 10): subjetividade, polaridade positiva/negativa,
    sentenças negativas/positivas, pontuação;
  - concretude (`conc_*`, 10): prazer, atenção, sensibilidade, aptidão,
    polaridade;
  - estilo "twitter" (`twt_*`, 5): palavras elongadas, expressões, negação,
    polaridade de emoji/emoticon;
  - miscelânea (`sbj_*`, 2): entidades nomeadas e palavras escritas
    corretamente.

A descrição detalhada de cada uma das 58 features está em
[`docs/features.md`](../docs/features.md).

As features **já vêm calculadas** nos CSVs; a réplica não reextrai nenhuma
delas. No protocolo da RQ1, elas são usadas diretamente. No baseline TF-IDF,
apenas `text` e `rating` são lidas.

## Carregamento e preparação

Há dois caminhos:

| Caminho | Código | Colunas | Tratamento |
| --- | --- | --- | --- |
| Baseline TF-IDF | `src/preprocessing/embedding.py`, `run_baseline.py` | `text`, `rating` | normaliza o texto, remove textos vazios e codifica o rating em 0..4 |
| Protocolo das 58 features | `src/models/article_protocol.py` | todas as 61 | seleciona as 58 features, valida colunas, mantém textos vazios, codifica o rating em 0..4 e cria `categoria_rating` para estratificar |

No protocolo, `categoria`, `text` e `rating` não entram na matriz `X`; a
categoria serve apenas para estratificar os folds e para análises futuras.

## Duplicatas entre treino e teste

Foram encontradas **280 combinações distintas** de (`categoria`, `text`,
`rating`) presentes tanto em `train.csv` quanto em `test.csv`, correspondendo a
**351 ocorrências** no total (um mesmo texto pode aparecer mais de uma vez em
cada split). Cerca de 41 combinações são reviews longas e específicas, não
apenas textos curtos genéricos como "Ok" ou "10".

**Decisão:** manter o split oficial sem alteração. Remover duplicatas mudaria a
base do artigo e invalidaria a comparação direta; além disso, o artigo não discute
o problema. A limitação fica registrada: o modelo pode ter visto no treino
exemplos do teste, o que tende a **inflar** as métricas. O mesmo vale para as
duplicatas dentro de cada split (949 linhas repetidas no treino e 104 no teste,
além da primeira ocorrência), que não foram tratadas.

## Limitações da base

- Licença não especificada e ausência dos dados brutos de coleta.
- Duplicatas entre treino e teste (280 combinações / 351 ocorrências).
- Desbalanceamento entre categorias.
- 1 texto vazio por split.
- 58 features pré-calculadas, sem código de extração no repositório.
- O artigo não disponibiliza os JSONs brutos; a réplica usa apenas os CSVs processados.

## Arquivos não versionados

`train.csv` e `test.csv` não são versionados no Git — já cobertos pelo
`.gitignore` do projeto (ver `data/raw/` e `data/processed/`). Para obter os
arquivos localmente, ver seção "Forma de obtenção" acima.

### Split local para validação (Issue 7)

Para separar 20% do treino oficial para validação, execute na raiz do projeto:

```bash
python scripts/create_validation_split.py \
  --train data/raw/train.csv \
  --out data/raw/validation.csv \
  --validation-size 0.2 \
  --seed 42
```

O script estratifica por `rating`, exclui textos vazios conforme a normalização
do projeto e preserva no CSV de saída todas as colunas/valores originais. Com a
base atual, gera 8.235 registros de validação e deixa 32.936 para treino. O
arquivo `data/raw/validation.csv` é gerado localmente e continua ignorado pelo
Git: contém reviews derivados da mesma base sem licença de redistribuição
especificada. O código, a seed e os resultados das métricas são versionados; o
CSV deve ser recriado a partir do `train.csv` local.
