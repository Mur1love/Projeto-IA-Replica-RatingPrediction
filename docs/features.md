# As 58 features textuais

Este documento descreve as 58 features textuais usadas no artigo *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features* e presentes nos arquivos `data/raw/train.csv` e `data/raw/test.csv`.

As features **já vêm calculadas** nos CSVs do projeto original. A réplica não reextrai nenhuma delas: o protocolo da RQ1 lê as 58 colunas diretamente. A extração original usou spaCy, scikit-learn, SenticNet e Hunspell, além do léxico SentiProdBR; esse pipeline não faz parte deste repositório. As definições abaixo vêm da Tabela 2 do artigo e dos nomes das colunas dos CSVs.

## Resumo por grupo

| Grupo | Nome no artigo | Prefixo no CSV | Quantidade | Features |
| --- | --- | --- | ---: | --- |
| POS | Part-of-speech | `pos_` | 19 | F1–F19 |
| SYNT | Padrões sintáticos | `synt_` | 5 | F20–F24 |
| LEX | Léxico | `lex_` | 10 | F25–F34 |
| CONC | Conceito (SenticNet) | `conc_` | 10 | F35–F44 |
| EST | Estruturais | `str_` | 7 | F45–F51 |
| TWT | Twitter/X | `twt_` | 5 | F52–F56 |
| MCL | Miscelânea | `sbj_` | 2 | F57–F58 |
| **Total** | | | **58** | |

> **Atenção ao prefixo `sbj_`:** ele não significa "subjetividade". As duas colunas `sbj_*` pertencem ao grupo **MCL (Miscelânea)** do artigo. O nome do prefixo é herdado do projeto original.

## Tabela detalhada

### POS — Part-of-speech (19 features)

Contagem de cada categoria gramatical identificada pelo spaCy.

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F1 | `pos_adjectives` | Adjetivos |
| F2 | `pos_adv` | Advérbios |
| F3 | `pos_det` | Artigos |
| F4 | `pos_cconj` | Conjunções coordenativas |
| F5 | `pos_sconj` | Conjunções subordinativas |
| F6 | `pos_intj` | Interjeições |
| F7 | `pos_punct` | Marcadores de pontuação |
| F8 | `pos_num` | Números cardinais |
| F9 | `pos_part` | Partículas |
| F10 | `pos_adp` | Preposições |
| F11 | `pos_pron` | Pronomes |
| F12 | `pos_propn` | Pronomes próprios / nomes próprios |
| F13 | `pos_sym` | Símbolos |
| F14 | `pos_noun` | Substantivos |
| F15 | `pos_sup` | Superlativos |
| F16 | `pos_verb` | Verbos |
| F17 | `pos_aux` | Verbos auxiliares |
| F18 | `pos_comp` | Comparativos |
| F19 | `pos_x` | Tokens não categorizáveis |

### SYNT — Padrões sintáticos (5 features)

Contagem dos padrões sintáticos propostos por Chenlo e Losada (2014).

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F20 | `synt_1` | Adjetivo → Substantivo |
| F21 | `synt_2` | Advérbio → Adjetivo → Não-Substantivo |
| F22 | `synt_3` | Adjetivo → Adjetivo → Não-Substantivo |
| F23 | `synt_4` | Substantivo → Adjetivo → Não-Substantivo |
| F24 | `synt_5` | Advérbio → Verbo, seguido de particípio, gerúndio, passado simples ou imperfeito |

### LEX — Léxico (10 features)

Contagens e proporções calculadas com o léxico SentiProdBR (de Melo, 2021).

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F25 | `lex_subjective` | Quantidade de termos subjetivos |
| F26 | `lex_positive` | Quantidade de palavras positivas |
| F27 | `lex_negative` | Quantidade de palavras negativas |
| F28 | `lex_quotationexclamation` | Quantidade de marcadores de pontuação (`!`, `?`, `...`) |
| F29 | `lex_countpositivesentences` | Quantidade de frases positivas |
| F30 | `lex_countnegativesentences` | Quantidade de frases negativas |
| F31 | `lex_proppositive` | Proporção de palavras positivas em relação ao texto |
| F32 | `lex_propnegative` | Proporção de palavras negativas em relação ao texto |
| F33 | `lex_propquotationexclamation` | Proporção de marcadores de pontuação em relação ao texto |
| F34 | `lex_propsubjective` | Proporção de termos subjetivos em relação ao texto |

### CONC — Conceito (10 features)

Métricas extraídas do SenticNet. As features de soma agregam os valores dos termos do comentário; as de média dividem pelo número de termos.

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F35 | `conc_pleasantness` | Soma de *pleasantness* (agradabilidade) |
| F36 | `conc_avgpleasantness` | Média de *pleasantness* |
| F37 | `conc_sensitivity` | Soma de *sensitivity* (sensibilidade) |
| F38 | `conc_avgsensitivity` | Média de *sensitivity* |
| F39 | `conc_aptitude` | Soma de *aptitude* (aptidão) |
| F40 | `conc_avgaptitude` | Média de *aptitude* |
| F41 | `conc_attention` | Soma de *attention* (atenção) |
| F42 | `conc_avgattention` | Média de *attention* |
| F43 | `conc_polarity` | Soma da polaridade geral do comentário |
| F44 | `conc_avgpolarity` | Média da polaridade geral do comentário |

### EST — Estruturais (7 features)

Características estruturais do texto.

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F45 | `str_qtcharacters` | Número total de caracteres |
| F46 | `str_qtwords` | Número de palavras |
| F47 | `str_qtsentences` | Número de frases |
| F48 | `str_qtcapitalizedwords` | Número de palavras iniciadas por letra maiúscula |
| F49 | `str_qtcapitalizedchars` | Número total de letras maiúsculas |
| F50 | `str_propcapitalizedwords` | Proporção de palavras iniciadas por maiúscula em relação ao total de palavras |
| F51 | `str_propcapitalizedchars` | Proporção de letras maiúsculas em relação ao total de caracteres |

### TWT — Twitter/X (5 features)

Termos típicos de redes sociais. As features de emoji e emoticon usam as pontuações de sentimento propostas por Kimura e Katsurai (2018) e Hogenboom et al. (2015), respectivamente. Os nomes das colunas registram a pontuação de polaridade; a Tabela 2 do artigo descreve F55 e F56 como contagem de emojis e emoticons com essa pontuação.

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F52 | `twt_countelongated` | Número de palavras alongadas (ex.: `muuuito`) |
| F53 | `twt_countexpressions` | Número de gírias específicas do Twitter/X |
| F54 | `twt_negation` | Número de palavras de negação (ex.: `não`, `nunca`) |
| F55 | `twt_emojipolarityscore` | Pontuação de polaridade de emojis |
| F56 | `twt_emoticonpolarityscore` | Pontuação de polaridade de emoticons |

### MCL — Miscelânea (2 features)

| Feature | Coluna no CSV | Descrição |
| --- | --- | --- |
| F57 | `sbj_countne` | Número de Entidades Nomeadas (nomes próprios, marcas, locais), via spaCy |
| F58 | `sbj_countcorrectwords` | Quantidade de palavras escritas corretamente, via dicionário Hunspell |

## Como as features entram nos modelos

No protocolo da RQ1 (`src/models/article_protocol.py`), a matriz `X` é formada apenas pelas colunas que começam com os sete prefixos acima:

```python
FEATURE_PREFIXES = ("str_", "pos_", "synt_", "lex_", "conc_", "twt_", "sbj_")
N_FEATURES = 58
```

Regras de seleção e proteção contra vazamento:

- `categoria`, `text` e `rating` **nunca** entram em `X`. A categoria é usada apenas para estratificar os folds (`categoria_rating`) e o rating é o alvo.
- Qualquer coluna que não seja uma das 58 features nem `categoria`, `text` ou `rating` interrompe a execução. Isso evita que um índice (`Unnamed: 0`) ou um `id` entre no modelo em silêncio.
- As features são convertidas para `float64`; valores ausentes interrompem a execução, porque o artigo não os trata.
- As linhas com `text` vazio **não** são removidas no protocolo das 58 features: as features já estão calculadas e o artigo usa todas as linhas. O baseline TF-IDF, que depende do texto, remove uma linha vazia em cada split.
- As features **não são padronizadas**, como no artigo. As escalas variam muito: `str_qtcharacters` chega a milhares, enquanto proporções ficam entre 0 e 1. Essa decisão deixa o SVM mais lento e pode prejudicar o kernel RBF; padronizar seria uma extensão separada, não uma fidelidade ao artigo.

## Origem e limitações

- **Não há reextração.** As 58 colunas foram produzidas pelo projeto original e são reutilizadas como estão. A réplica não reproduz o pipeline de spaCy, SenticNet, Hunspell e SentiProdBR.
- **Não é possível auditar a extração** a partir deste repositório. Não há código de geração, versões das bibliotecas de NLP nem testes das features originais.
- **Muitos zeros.** Comentários curtos produzem uma matriz esparsa, como o próprio artigo observa. Isso influenciou a escolha de modelos que lidam bem com esparsidade.
- **Recalcular exigiria o texto bruto.** Se a equipe for reextrair as features no futuro, deve usar o texto bruto, nunca o texto normalizado do baseline. A normalização remove ou altera informação usada por várias features: maiúsculas (F48–F51), palavras alongadas (F52), emojis/emoticons (F55–F56), pontuação (F28, F33) e palavras escritas corretamente (F58). Essa recomendação vem da leitura das definições do artigo e não foi testada empiricamente.

## Como conferir nos CSVs

Com o ambiente virtual ativado e os CSVs em `data/raw/`:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
prefixes = ("str_", "pos_", "synt_", "lex_", "conc_", "twt_", "sbj_")
features = [c for c in df.columns if c.startswith(prefixes)]

print("colunas:", len(df.columns))
print("features:", len(features))
print("nulos nas features:", int(df[features].isna().sum().sum()))
print("textos vazios:", int(df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()))
PY
```

Resultado esperado na base atual: 61 colunas, 58 features, nenhum valor ausente nas features e 1 texto vazio em cada split.

## Referências

- MARREIRA, E.; OLIVEIRA, M.; MELO, T. *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features*. Tabela 2.
- Repositório original: [github.com/emanuellemarreira/rating-prediction-with-textual-features](https://github.com/emanuellemarreira/rating-prediction-with-textual-features).
- Definições complementares: `data/README.md`, `docs/documentacao.md` (seção 3) e `results/metrics/README.md`.
