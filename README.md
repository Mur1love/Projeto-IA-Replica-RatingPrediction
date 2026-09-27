# Replica de Rating Prediction com Features Textuais

Projeto da disciplina de Inteligência Artificial dedicado à réplica do artigo *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features*. O trabalho investiga a previsão de avaliações de produtos da Amazon Brasil, escritas em português brasileiro, utilizando características textuais e modelos tradicionais de aprendizado de máquina.

## Equipe

| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| Murilo | [`@Mur1love`](https://github.com/Mur1love) | Liderança e organização do projeto |
| David Carvalho | [`@DaviidCarvallho`](https://github.com/DaviidCarvallho) | A definir |
| Eduardo Araujo | [`@Duuduaraujo`](https://github.com/Duuduaraujo) | A definir |
| Enriko Martins | [`@EnrikoMartins`](https://github.com/EnrikoMartins) | A definir |
| João Pazzin | [`@joaoppazzin1`](https://github.com/joaoppazzin1) | A definir |

## Descrição do projeto

O objetivo deste projeto é reproduzir a metodologia e os experimentos apresentados no artigo de referência. A tarefa consiste em prever a nota, de 1 a 5 estrelas, atribuída por usuários da Amazon a partir apenas do texto das avaliações.

A réplica investigará a extração de características textuais e a aplicação de modelos de classificação, com atenção especial aos seguintes pontos:

- comparação entre SVM, Random Forest, Logistic Regression, Gradient Boosting e XGBoost;
- avaliação de diferentes grupos de características textuais, como características léxicas, sintáticas, estruturais e de classe gramatical;
- estudo de ablação para analisar a contribuição de cada grupo de características;
- seleção de características por meio de Recursive Feature Elimination (RFE);
- comparação do desempenho entre diferentes categorias de produtos.

As métricas principais serão MAE, RMSE e AUC, conforme o protocolo descrito no artigo. As decisões metodológicas, resultados obtidos e diferenças em relação ao trabalho original serão documentados neste repositório ao longo do desenvolvimento.

## Artigo replicado

O artigo utilizado como base está disponível localmente na raiz deste repositório:

- [`RatingPredictionInBrazilianPortugueseReviews.pdf`](RatingPredictionInBrazilianPortugueseReviews.pdf)

## Estrutura do repositório

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── task.md
│   └── pull_request_template.md
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── scripts/
│   └── check_environment.py
├── src/
│   ├── data/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── .gitkeep
│   ├── __init__.py
│   └── test_environment.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
├── RatingPredictionInBrazilianPortugueseReviews.pdf
├── .gitignore
├── README.md
└── requirements.txt
```

### Finalidade dos diretórios

- `data/raw/`: datasets originais, sem alterações. Os arquivos de dados não devem ser versionados por padrão.
- `data/processed/`: dados gerados por limpeza, transformação ou preparação.
- `notebooks/`: análises exploratórias, experimentos e registros de investigação em Jupyter.
- `scripts/`: utilitários executados diretamente, como a verificação do ambiente (`scripts/check_environment.py`).
- `src/data/`: código reutilizável de carregamento e manipulação inicial dos dados.
- `src/preprocessing/`: código reutilizável de pré-processamento e extração de características.
- `src/models/`: implementação e treinamento dos modelos.
- `src/evaluation/`: avaliação dos modelos e cálculo das métricas.
- `src/utils/`: funções auxiliares compartilhadas pelo projeto.
- `tests/`: testes automatizados do código produzido pela equipe.
- `results/figures/`: figuras e gráficos gerados pelos experimentos.
- `results/metrics/`: resultados das métricas gerados pelos experimentos.
- `.github/`: modelos para Issues e Pull Requests.

Os notebooks devem apoiar a exploração e a comunicação dos experimentos. O código reutilizável deve ser mantido em `src/` para evitar concentrar toda a implementação nos notebooks.

## Configuração do ambiente

O projeto requer **Python 3.12 ou mais recente** — piso exigido pelas versões travadas em `requirements.txt` (`numpy` e `scipy` exigem 3.12+). A instalação foi validada em **Linux com Python 3.14**; os comandos para Windows estão documentados abaixo, mas ainda não foram validados em uma máquina Windows.

As dependências têm as versões fixadas em `requirements.txt` para garantir um ambiente reproduzível entre os integrantes.

1. Clone o repositório e acesse sua pasta:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual:

   No Linux ou macOS:

   ```bash
   source .venv/bin/activate
   ```

   No Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Se o PowerShell bloquear a execução do script de ativação (política padrão em algumas máquinas), execute `Set-ExecutionPolicy -Scope Process RemoteSigned` antes de ativar, ou use o prompt do cmd: `.venv\Scripts\activate.bat`.

4. Instale as dependências:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Verifique a instalação:

   ```bash
   python scripts/check_environment.py
   ```

   O comando importa todas as dependências principais e imprime as versões instaladas. Se algo estiver faltando, ele termina com código de saída diferente de zero.

### Decisão sobre dependências de NLP

A normalização de textos (Issue 4) utilizará o **nltk**, incluído em `requirements.txt`: ele é instalável apenas via pip e oferece *stopwords* e o stemmer RSLP em português. Os dados linguísticos necessários são leves e baixados sob demanda com `nltk.download(...)` — na Issue 4 serão usados `stopwords`, `rslp` e `punkt_tab`. O **spacy** foi descartado por depender de um modelo de pipeline completo (`pt_core_news_sm`) baixado separadamente — mais pesado e mais um artefato a manter. Observação: os dados do nltk não são travados pelo `requirements.txt`; a versão utilizada deve ser registrada na documentação de decisões (Issue 11). Caso a lematização se mostre necessária, a decisão poderá ser revista na Issue 4.

## Organização dos dados

Os datasets originais devem ser armazenados em `data/raw/` e preservados sem modificações. Em `data/processed/` devem ser salvos somente os dados resultantes de limpeza, transformação ou preparação.

Datasets e artefatos gerados podem ser grandes ou conter informações que não devem ser publicadas. Por isso, o `.gitignore` impede o versionamento desses arquivos por padrão. A forma de obtenção dos dados e eventuais decisões de preparação devem ser documentadas em `data/README.md`.

## Fluxo de desenvolvimento

O fluxo esperado é:

```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```

Regras do projeto:

1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade deve ser desenvolvida em uma branch própria.
3. Não devem ser realizados commits diretamente na `main`.
4. Toda alteração deve entrar na `main` por meio de Pull Request.
5. O Pull Request deve estar associado à Issue correspondente. Quando o merge concluir a atividade, use `Closes #numero_da_issue` na descrição do PR.
6. O autor do Pull Request não pode aprovar o próprio PR.
7. Todo PR precisa de pelo menos uma aprovação de outro integrante da equipe antes do merge.
8. Cada integrante deve utilizar sua própria conta GitHub e realizar seus próprios commits.
9. O histórico do GitHub será utilizado para acompanhar a participação individual dos integrantes.

### Exemplo de início de uma atividade

Depois de criar ou escolher uma Issue, atualize a `main` local e crie uma branch com um nome descritivo:

```bash
git switch main
git pull
git switch -c issue-12-descricao-curta
```

Faça commits pequenos e claros. Ao concluir a atividade, envie a branch e abra um Pull Request utilizando o modelo do repositório.

## Referências

### Artigo principal

Marreira, E.; Oliveira, M.; Melo, T. *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features*. Artigo disponibilizado neste repositório em [`RatingPredictionInBrazilianPortugueseReviews.pdf`](RatingPredictionInBrazilianPortugueseReviews.pdf).

### Repositório original

Marreira, E. *Rating Prediction with Textual Features*. Repositório utilizado como referência para a implementação e os experimentos: [github.com/emanuellemarreira/rating-prediction-with-textual-features](https://github.com/emanuellemarreira/rating-prediction-with-textual-features).
