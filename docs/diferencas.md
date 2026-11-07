# Diferenças em relação ao artigo e decisões da réplica

Este documento registra onde a réplica se afasta do artigo *Rating Prediction in Brazilian Portuguese Reviews* e por quê. Cada decisão deve apontar para uma evidência em `results/`. Novas diferenças devem ser acrescentadas à seção 1 conforme o projeto avança.

## 1. Diferenças em relação ao artigo

| Tema | Artigo | Réplica (estado atual) | Observação |
| --- | --- | --- | --- |
| Representação do texto | 58 features manuais (POS, sintáticas, léxicas, concretude, estruturais, Twitter/X, miscelânea) | Baseline com **TF-IDF** (1–2-gramas) + Regressão Logística; as 58 features ainda não entram | O artigo cita TF-IDF e Bag of Words apenas como *trabalho futuro* (Seção 5). O baseline é, portanto, um ponto de partida próprio da réplica, não uma reprodução. |
| Normalização do texto | Não descreve nenhuma etapa de normalização | `normalize()` aplicada antes do TF-IDF | Avaliada na seção 2; decisão na seção 2.5. |
| Métricas | MAE, RMSE e AUC | Acurácia, F1 macro, MAE e RMSE no baseline TF-IDF; MAE, RMSE, F1 macro, acurácia e AUC no protocolo do artigo | A AUC está implementada no harness do protocolo do artigo (seção 3), não no baseline LR. |
| Representação no SVM (RQ1) | 58 features manuais | SVM com as 58 features (seção 3) e, como comparação complementar, SVM sobre TF-IDF | O SVM com as 58 features é a reprodução do artigo; o SVM com TF-IDF é um experimento próprio. |
| IC de 95% | IC calculado com a linha da média incluída (6 valores, 5 g.l.) | IC calculado só entre os 5 folds (5 valores, 4 g.l.) | O IC publicado é cerca de 24% mais estreito. Seção 3.3. |
| Duplicatas treino/teste | Não discutidas no artigo | 280 combinações (`categoria`, `text`, `rating`) aparecem nos dois splits | Herdado do split oficial; pode inflar as métricas. Ver `data/README.md`. |

## 2. Normalização de texto: com e sem

### 2.1 Pergunta

A etapa de normalização (`src/preprocessing/normalization.py`) melhora o baseline TF-IDF + Regressão Logística? E, se houver efeito, quais de suas etapas contribuem?

### 2.2 Método

`scripts/compare_normalization.py` treina o mesmo modelo (TF-IDF com os parâmetros de `DEFAULT_PARAMS`, Regressão Logística com `C=1.0`, `seed=42`) em sete variantes que diferem **apenas** no pré-processamento do texto:

| Variante | O que faz |
| --- | --- |
| `sem_normalizacao` | Texto bruto. Só o tratamento padrão do `TfidfVectorizer` (`lowercase=True`, `strip_accents="unicode"`). **É a referência.** |
| `normalizacao_completa` | `normalize()` com os padrões do projeto (a usada no baseline atual). |
| `apenas_acentos` | Só remoção de acentos. |
| `apenas_repeticoes` | Só colapso de caracteres repetidos (`ameiii` → `ameii`). |
| `apenas_abreviacoes` | Só expansão de abreviações (`vc` → `você`). |
| `completa_sem_pontuacao` | Completa + pontuação trocada por espaço. |
| `completa_numero_token` | Completa + números trocados pelo token `NUM`. |

Cuidados para a comparação ser justa:

- **Mesmas linhas em todas as variantes.** Textos que ficam vazios após a normalização completa são removidos em todas, inclusive na bruta (41.171 linhas de treino e 10.292 de teste; 1 removida em cada, como no baseline).
- **Dois protocolos.** *Validação*: 80% do treino → 20% de validação (o split da Issue 7, que não toca no teste). *Teste*: treino completo → teste oficial. A decisão se apoia na validação; o teste só confirma.
- **Bootstrap pareado** (1.000 reamostragens, `seed=42`) no conjunto de avaliação: o mesmo sorteio de linhas é aplicado a todas as variantes, e o IC 95% da diferença em relação à referência indica se ela se distingue de zero. A coluna *Sig.* é "sim" quando o IC não contém zero. Os ICs completos estão nos arquivos `.json`.

### 2.3 Resultados

Δ = variante − `sem_normalizacao`. Para acurácia e F1, maior é melhor; para MAE e RMSE, menor é melhor.

**Validação (treino 80% = 32.936 → validação = 8.235)**: base da decisão.

| Variante | Acurácia | F1 macro | MAE | RMSE | Vocab. | Δ acurácia | Δ MAE | Sig. acc | Sig. MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `sem_normalizacao` | 0,4865 | 0,4830 | 0,7022 | 1,0914 | 23.330 | +0,0000 | +0,0000 | ref | ref |
| `normalizacao_completa` | 0,4866 | 0,4833 | 0,7001 | 1,0883 | 23.332 | +0,0001 | -0,0022 | não | não |
| `apenas_acentos` | 0,4865 | 0,4830 | 0,7022 | 1,0914 | 23.330 | +0,0000 | +0,0000 | não | não |
| `apenas_repeticoes` | 0,4860 | 0,4827 | 0,7013 | 1,0895 | 23.327 | -0,0005 | -0,0010 | não | não |
| `apenas_abreviacoes` | 0,4866 | 0,4832 | 0,6993 | 1,0865 | 23.335 | +0,0001 | -0,0029 | não | não |
| `completa_sem_pontuacao` | 0,4865 | 0,4831 | 0,7007 | 1,0891 | 23.332 | +0,0000 | -0,0016 | não | não |
| `completa_numero_token` | 0,4839 | 0,4804 | 0,7052 | 1,0948 | 23.304 | -0,0026 | +0,0029 | não | não |

**Teste oficial (treino = 41.171 → teste = 10.292)**: confirmação.

| Variante | Acurácia | F1 macro | MAE | RMSE | Vocab. | Δ acurácia | Δ MAE | Sig. acc | Sig. MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `sem_normalizacao` | 0,4903 | 0,4866 | 0,6824 | 1,0633 | 28.488 | +0,0000 | +0,0000 | ref | ref |
| `normalizacao_completa` | 0,4888 | 0,4851 | 0,6836 | 1,0639 | 28.500 | -0,0015 | +0,0013 | não | não |
| `apenas_acentos` | 0,4903 | 0,4866 | 0,6824 | 1,0633 | 28.488 | +0,0000 | +0,0000 | não | não |
| `apenas_repeticoes` | 0,4912 | 0,4875 | 0,6798 | 1,0602 | 28.487 | +0,0009 | -0,0025 | não | sim |
| `apenas_abreviacoes` | 0,4889 | 0,4854 | 0,6831 | 1,0627 | 28.501 | -0,0014 | +0,0007 | não | não |
| `completa_sem_pontuacao` | 0,4894 | 0,4857 | 0,6823 | 1,0623 | 28.500 | -0,0009 | -0,0001 | não | não |
| `completa_numero_token` | 0,4896 | 0,4857 | 0,6840 | 1,0661 | 28.450 | -0,0007 | +0,0017 | não | não |

Arquivos: `results/metrics/normalization_comparison_validation.{csv,json,md}` e `results/metrics/normalization_comparison.{csv,json,md}`.

### 2.4 Discussão

**A normalização não tem efeito mensurável neste baseline.** A maior diferença de acurácia em módulo é 0,0026 (validação) e 0,0015 (teste), e o IC 95% da diferença tem meia-largura de ≈0,002 a 0,003. Nenhuma diferença de acurácia ou F1 se distingue de zero em qualquer variante, em qualquer protocolo.

**O sinal da normalização completa muda de lado entre os protocolos** (Δ acurácia de +0,0001 na validação e −0,0015 no teste). Isso é o comportamento esperado de ruído, não de um efeito real.

**Variações desse tamanho também aparecem entre execuções do mesmo código.** O JSON versionado do baseline (`baseline_logreg.json`) registra acurácia de 0,4898 para a configuração equivalente à `normalizacao_completa`; reexecutada neste ambiente, a mesma configuração deu 0,4888. A causa não foi investigada (hipótese: diferenças numéricas do solver entre plataformas). O importante é que essa diferença, de ≈0,001, é do mesmo tamanho dos efeitos medidos acima. Por isso todas as variantes devem ser comparadas **na mesma execução**, como o script faz, e não contra números versionados em outra máquina.

**`apenas_acentos` é idêntico à referência**, em todas as métricas e no vocabulário (23.330 e 28.488 termos). A etapa altera 78,7% dos textos, mas o `TfidfVectorizer` já remove acentos internamente (`strip_accents="unicode"`), então ela é redundante.

**`completa_sem_pontuacao` não muda o vocabulário em relação à `normalizacao_completa`** (28.500 termos nos dois casos), apesar de alterar 91,9% dos textos. O `token_pattern` padrão do TF-IDF só reconhece sequências de 2+ caracteres alfanuméricos, então a pontuação já é ignorada. Além disso, a docstring de `normalize()` avisa que trocar pontuação por espaço desfaz emoticons.

**O único resultado "significativo" não se sustenta.** No teste, `apenas_repeticoes` reduz o MAE em 0,0025 com IC que exclui zero (e sem efeito significativo na acurácia). Na validação, a mesma variante dá ΔMAE = −0,0010 com IC que contém zero, e a acurácia cai 0,0005. Foram 12 comparações (6 variantes × acurácia e MAE) sem correção para múltiplos testes, o que torna esperado ≈0,6 falso positivo ao nível de 5%. Além disso, essa variante só se destacou ao olhar o teste, e a decisão não deve ser tomada nele. Tratamos como acaso.

**Por que o efeito é pequeno.** Provavelmente porque as etapas de fato tocam poucos textos: o colapso de repetições altera 7,8% dos textos de treino e a expansão de abreviações 3,9%; e o vocabulário muda menos de 0,1% (28.488 → 28.500 termos). Essa explicação é uma interpretação: não foi testada isoladamente.

**`completa_numero_token`** é a única variante com sinal consistentemente desfavorável: pior acurácia na validação (−0,0026, IC [−0,0058; +0,0007]) e sem ganho no teste. O IC contém zero, então também não é conclusivo, mas não há motivo para adotá-la.

### 2.5 Decisão

1. **Manter a normalização atual no baseline** (minúsculas, colapso de repetições, expansão de abreviações, espaços), sem alterar o código. Não há evidência de benefício nem de prejuízo; manter evita invalidar os resultados, notebooks e testes já versionados, e as etapas têm justificativa linguística. A escolha é por custo de mudança, **não por ganho medido**.
2. **Não ativar `strip_punctuation`**: não tem efeito sobre o TF-IDF e desfaz emoticons.
3. **Não ativar `number_token`**: sem benefício e com sinal desfavorável na validação.
4. **Considerar `strip_accent` redundante** com o TF-IDF. Pode permanecer no código (resultado idêntico), mas não deve ser citado como contribuição da normalização.
5. **Não investir mais tempo ajustando a normalização** para o TF-IDF: ela não é uma alavanca de desempenho neste modelo.
6. **Para as 58 features do artigo, extrair a partir do texto bruto**, nunca do normalizado. Várias dependem de informação que `normalize()` remove ou altera: letras maiúsculas (F48–F51), palavras alongadas (F52), emojis e emoticons (F55–F56), marcadores de pontuação (F28, F33) e palavras escritas corretamente (F58). Hoje as features vêm pré-calculadas nos CSVs, então não há risco imediato; o cuidado vale se a equipe as recalcular. Esta recomendação vem da leitura das definições das features na Tabela 2 do artigo e **não foi testada empiricamente**.

A alternativa de adotar `sem_normalizacao` como baseline (mais simples) é igualmente defensável pelos dados; a equipe deve confirmar a escolha na revisão do Pull Request.

### 2.6 Limitações

- Um único modelo (Regressão Logística) e uma única configuração de TF-IDF. O efeito pode ser diferente com outros classificadores ou representações, e as conclusões **não se estendem** ao pipeline com as 58 features.
- Uma única `seed` e um único split. O bootstrap mede a incerteza da amostra de avaliação, não a do treino.
- Duplicatas entre treino e teste (seção 1) podem atenuar diferenças entre variantes no teste.
- Sem correção para múltiplas comparações; os ICs são individuais.
- A coluna `tempo_s` dos CSVs muda a cada execução; as demais colunas são determinísticas dado o ambiente.

### 2.7 Como reproduzir

```bash
python scripts/compare_normalization.py --protocol validation   # decisão (não usa o teste)
python scripts/compare_normalization.py                         # confirmação no teste
python scripts/compare_normalization.py --sample-size 2000      # execução rápida
```

Cada execução completa leva poucos minutos. Requer `data/raw/train.csv` e `data/raw/test.csv` (ver `data/README.md`).

## 3. SVM com as 58 features (RQ1)

### 3.1 O que foi replicado

`scripts/train_article_model.py --model svm` aplica o protocolo de `code/train_model.py` e `code/RQ1/scripts/svm.py` do repositório original (detalhes no `README.md`):

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados (51.465 linhas) | Igual: **o teste participa da validação cruzada**, o que impede ler o resultado como desempenho em dados nunca vistos. |
| Features | 58 (`str_`, `pos_`, `synt_`, `lex_`, `conc_`, `twt_`, `sbj_`) | Igual; `categoria`, `text` e `rating` ficam fora. Uma coluna fora desse conjunto interrompe a execução. |
| Folds | `StratifiedKFold(5, shuffle=False)` por `categoria_rating` | Igual. Como não há embaralhamento, a ordem das linhas (treino, depois teste) define os folds. `--shuffle` existe, mas fica registrado no JSON. |
| Busca | `RandomizedSearchCV`, `n_iter=300`, RMSE, `random_state=42` | Igual. |
| Grid | `C` ∈ {0,01; 0,1}, `kernel` ∈ {linear, rbf}, `gamma` ∈ {scale, auto, 0,01, 0,1} | Igual. |
| Métricas | MAE, RMSE, AUC (classes 4–5 positivas, sobre predições binárias) | Igual, mais F1 macro, acurácia e uma AUC de diagnóstico por `decision_function`. |
| Linhas com `text` vazio | Mantidas | **Mantidas** (2 linhas, uma por split), porque as features já estão calculadas. O baseline TF-IDF as remove (41.171 e 10.292 linhas); por isso os dois experimentos não usam exatamente as mesmas linhas. |

### 3.2 Diferenças em relação ao artigo

1. **`n_iter=300` avalia 16 candidatos.** O grid tem 2 × 2 × 4 = 16 combinações; o scikit-learn limita a busca a elas e avisa. Vale para o artigo e para a réplica; o JSON registra `n_iter_requested` e `n_candidates_evaluated`. Além disso, `gamma` não afeta o kernel linear: os 8 candidatos lineares (2 valores de `C` × 4 de `gamma`) correspondem a apenas 2 ajustes distintos, e os outros 6 repetem trabalho. Mantivemos a busca como está, por fidelidade.
2. **Sem padronização das features, como no artigo.** As escalas diferem muito (por exemplo, `str_qtcharacters` chega a 4.930, enquanto outras ficam entre 0 e 1). Isso torna o SVM linear lento e provavelmente prejudica o kernel RBF. Não corrigimos para não nos afastarmos do protocolo; padronizar seria uma extensão a testar à parte.
3. **scikit-learn puro.** O código original chama `patch_sklearn()` do `scikit-learn-intelex` (Intel oneDAL), que acelera o `SVC` e pode produzir diferenças numéricas pequenas. A réplica usa o scikit-learn do `requirements.txt` (1.9.1).
4. **Rótulos 0..4.** O projeto usa `encode_rating` (0..4); o artigo usa 1..5 no SVM. O erro (MAE, RMSE) é calculado na escala original de 1 a 5 estrelas e a escolha de rótulos não altera o SVC, que só depende da ordem das classes.
5. **`refit=False` na busca.** O código original reajusta o melhor modelo em todos os dados ao final da busca, mas esse modelo nunca é usado (cada fold é treinado de novo). Omitimos esse treino, sem efeito nos resultados.
6. **Sem arredondamento por fold.** O artigo arredonda as métricas de cada fold a 4 casas antes de agregar; a réplica agrega com precisão total (diferença da ordem de 1e-5).

### 3.3 O IC de 95% do artigo é mais estreito que o IC correto

No código original, a linha da média é anexada ao DataFrame **antes** de calcular o erro-padrão e o intervalo. Assim, `stats.sem` e os graus de liberdade (`len(...) - 1`) passam a considerar 6 valores (os 5 folds mais a própria média), e não 5. Isso foi verificado contra `code/RQ1/results/SVC_results.csv`:

| Métrica | Média ± desvio (artigo) | IC publicado | IC entre os 5 folds (réplica) |
| --- | --- | --- | --- |
| MAE | 1,0431 ± 0,0070 | [1,0366; 1,0496] | [1,0345; 1,0517] |
| RMSE | 1,5085 ± 0,0118 | [1,4974; 1,5196] | [1,4939; 1,5232] |
| AUC | 0,7304 ± 0,0046 | [0,7261; 0,7347] | [0,7248; 0,7361] |

O IC publicado só é reproduzido incluindo a média como sexto valor com 5 graus de liberdade; sua meia-largura é cerca de 24% menor que a correta. **A média e o desvio-padrão não são afetados.** A réplica usa o IC correto (`t` com `n_folds − 1` graus de liberdade); a comparação com o artigo deve ser feita pela média e pelo desvio, não pelo IC. O teste `test_ic_reproduz_valores_do_artigo_por_fold` fixa esses valores.

### 3.4 Comparação complementar: SVM vs. Regressão Logística, ambos com TF-IDF

`scripts/train_svm_tfidf.py` treina o SVM sobre a matriz TF-IDF da Issue 6 (mesmo vetorizador, mesmo split oficial treino → teste do baseline). Os hiperparâmetros são **fixos e sem busca** (`kernel="linear"`, `C=1.0`), não ajustados no teste. Resultado no teste oficial (10.292 linhas); Δ = SVM − LR:

| Métrica | LR + TF-IDF (`baseline_logreg.json`) | SVM + TF-IDF (`svm_tfidf.json`) | Δ (SVM − LR) |
| --- | ---: | ---: | ---: |
| Acurácia | 0,4898 | 0,4884 | −0,0014 |
| F1 macro | 0,4862 | 0,4849 | −0,0013 |
| MAE (estrelas) | 0,6819 | 0,6830 | +0,0011 |
| RMSE (estrelas) | 1,0619 | 1,0596 | −0,0023 |

O SVM (kernel linear, `C=1.0`) ficou **praticamente empatado** com a Regressão Logística: a maior diferença em módulo é 0,0023, e o sinal muda conforme a métrica (a LR é um pouco melhor em acurácia, F1 e MAE; o SVM, em RMSE). Diferenças desse tamanho são da mesma ordem da variação de ≈0,001 entre execuções do mesmo código em ambientes diferentes (seção 2.4), e não foi feito bootstrap, então **não há evidência de que um dos classificadores seja melhor** sobre este TF-IDF. O treino do SVM levou ≈10 min em 1 CPU. A AUC do SVM (predições binárias) foi 0,8261; o baseline LR não a registra.

Limitações: um único split e uma única configuração do SVM (sem busca de `C`); diferenças pequenas devem ser lidas com a ressalva da seção 2.4, de que a mesma configuração varia na ordem de 0,001 entre ambientes. Esta comparação **não** mede o efeito das 58 features, e sim de trocar o classificador mantendo o TF-IDF; para 58 features contra TF-IDF, veja a seção 3.5.

### 3.5 Resultado do protocolo completo com as 58 features

Execução completa: 51.465 linhas (41.172 de `train.csv` + 10.293 de `test.csv`), 5 folds, 16 candidatos avaliados (`n_iter=300` pedido), scikit-learn 1.9.1, `--n-jobs -1`. A busca levou ≈ 7,6 h e a avaliação dos 5 folds e do holdout ≈ 16 min (o tempo depende do número de núcleos). Os melhores hiperparâmetros foram `kernel=linear`, `gamma=scale`, `C=0.1`, **os mesmos do artigo**.

**Comparação com a Tabela 3 do artigo** (média ± desvio-padrão entre os 5 folds):

| Métrica | Artigo | Réplica | Δ da média | IC de 95% da réplica |
| --- | --- | --- | ---: | --- |
| MAE | 1,0431 ± 0,0070 | 1,0430 ± 0,0062 | −0,0001 | [1,0353; 1,0506] |
| RMSE | 1,5085 ± 0,0118 | 1,5085 ± 0,0112 | +0,00003 | [1,4947; 1,5224] |
| AUC | 0,7304 ± 0,0046 | 0,7303 ± 0,0047 | −0,0001 | [0,7246; 0,7361] |

Outras métricas da réplica (não constam na Tabela 3): F1 macro 0,3550 ± 0,0023, IC [0,3521; 0,3578]; acurácia 0,3796 ± 0,0020, IC [0,3771; 0,3821]; AUC por `decision_function` (diagnóstico) 0,8000 ± 0,0046, IC [0,7943; 0,8057].

Leitura:

- **A réplica reproduz o artigo.** As médias diferem em no máximo 0,00014 (no MAE), mais de 40 vezes menos que o desvio entre folds (≈ 0,006). O MAE de cada fold difere do publicado em no máximo 0,0014, e o fold 4 é o melhor nas duas execuções, o que indica folds construídos da mesma forma (`shuffle=False`).
- O desvio-padrão da réplica é um pouco menor que o do artigo (MAE 0,0062 contra 0,0070). A causa não foi isolada: a seção 3.2 (item 3) cita o `scikit-learn-intelex` como possível fonte de diferença numérica, e o artigo arredonda as métricas por fold antes de agregar.
- O IC da réplica é mais largo que o publicado (por exemplo, MAE [1,0353; 1,0506] contra [1,0366; 1,0496]) pelo motivo da seção 3.3: o artigo inclui a média no cálculo. Compare pela média e pelo desvio.
- A AUC do artigo (0,730) é a de predições binárias, um único ponto operacional; a AUC por scores contínuos é bem maior (0,800).
- `n_train=41172` e `n_test=10293` em cada fold de `svm_rq1_folds.csv` coincidem com os tamanhos de `train.csv` e `test.csv` porque 51.465 × 0,8 = 41.172. Não é o split oficial; o holdout abaixo é.

**Holdout** (treino em `train.csv`, avaliação em `test.csv`, mesmos hiperparâmetros): MAE 1,0476, RMSE 1,5160, F1 macro 0,3539, acurácia 0,3791, AUC 0,7295 (por scores: 0,7999). Fica dentro da faixa dos 5 folds em todas as métricas, o que sugere que o fato de o teste participar da validação cruzada (limitação da seção 3.1) pouco altera a estimativa.

**58 features vs. TF-IDF, no mesmo split oficial treino → teste:**

| Métrica | SVM, 58 features (holdout) | SVM + TF-IDF | LR + TF-IDF |
| --- | ---: | ---: | ---: |
| Acurácia | 0,3791 | 0,4884 | 0,4898 |
| F1 macro | 0,3539 | 0,4849 | 0,4862 |
| MAE (estrelas) | 1,0476 | 0,6830 | 0,6819 |
| RMSE (estrelas) | 1,5160 | 1,0596 | 1,0619 |
| AUC (predições binárias) | 0,7295 | 0,8261 | não registrada |

Com o mesmo classificador, o TF-IDF fica cerca de 0,36 estrela abaixo no MAE e cerca de 0,11 acima na acurácia. Ressalvas: o TF-IDF remove 1 linha de `text` vazio por split, o SVM com 58 features usa hiperparâmetros buscados e o com TF-IDF usa hiperparâmetros fixos (seção 3.4). Nenhuma delas deve explicar uma diferença dessa magnitude, mas a comparação não foi testada estatisticamente.

### 3.6 Como reproduzir

```bash
python scripts/train_article_model.py --model svm --sample-n 2000   # rápido, ~2 min em 1 CPU
python scripts/train_article_model.py --model svm                   # completo (≈ 8 h na execução de referência)
python scripts/train_svm_tfidf.py                                   # SVM + TF-IDF vs. LR
```

## 4. XGBoost com as 58 features (RQ1)

O XGBoost foi o melhor modelo do artigo nas três métricas da Tabela 3 (MAE 0,9499 ± 0,0047; RMSE 1,3721 ± 0,0055; AUC 0,7390 ± 0,0046) e o escolhido para as RQs seguintes; por isso a TAREFA 14 o replica. Parte do texto abaixo refere-se à execução completa que é rodada no escopo da TAREFA 14 e cujos números finais preenchem as subseções 4.3 e 4.4.

### 4.1 O que foi replicado

`scripts/train_article_model.py --model xgboost` aplica o mesmo protocolo de `code/train_model.py` do repositório original (detalhes no `README.md`) ao `XGBClassifier`:

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados (51.465 linhas) | Igual: **o teste participa da validação cruzada**. |
| Features | 58 (`str_`, `pos_`, `synt_`, `lex_`, `conc_`, `twt_`, `sbj_`) | Igual; `categoria`, `text` e `rating` ficam fora; linhas com `text` vazio mantidas. |
| Folds | `StratifiedKFold(5, shuffle=False)` por `categoria_rating` | Igual. |
| Busca | `RandomizedSearchCV`, `n_iter=300`, RMSE, `random_state=42` | Igual. |
| Grid | 9 dimensões (TAREFA 14), 8.748 combinações | Igual (exatamente os valores da TAREFA; teste fixa o grid). |
| Métricas | MAE, RMSE, AUC (classes 4–5 positivas, sobre predições binárias) | Igual, mais F1 macro, acurácia e AUC de diagnóstico por `predict_proba`. |
| Holdout | treino em `train.csv`, avaliação em `test.csv` | Igual, com os mesmos hiperparâmetros. |

### 4.2 Diferenças em relação ao artigo

1. **Rótulos 0..4 vs. 1..5.** O projeto usa `encode_rating` (labels 0..4) e converte para a escala de 1 a 5 estrelas somente no cálculo de MAE/RMSE (`compute_metrics`). O código original do XGB também treina com ratings 0..4 e converte as predições para 1..5 na agregação; a semântica do erro é a mesma (nota registrada na TAREFA 14). A diferença não altera o modelo: `XGBClassifier` ordena as classes e a numeração é apenas nominal.
2. **Versão da biblioteca.** A réplica fixa `xgboost==3.4.1` (`requirements.txt`); a versão usada pelo código original não é informada. Pequenas diferenças numéricas são possíveis (mesma ressalva do `scikit-learn-intelex`, seção 3.2).
3. **Paralelismo.** `random_state=42` fixo no estimador e `n_jobs=-1` no nível externo (CLI), com o estimador em `n_jobs=1` para que a busca e o refit por fold controlem o paralelismo — decisão igual à do Random Forest aprovado (reprodutibilidade por seed; o `n_jobs` não afeta o resultado).
4. **Sem padronização das features, como no artigo** (mesma justificativa da seção 3.2, item 2).
5. **`refit=False` na busca** e **IC de 95% com `n_folds − 1` graus de liberdade**, pelos mesmos motivos das seções 3.2 (item 5) e 3.3 — compare-se com o artigo pela média e pelo desvio.
6. **Melhores hiperparâmetros do artigo não informados** (`ARTICLE_BEST_PARAMS` vazio), igual ao caso do Random Forest.

### 4.3 Comparação complementar: XGBoost + TF-IDF vs. Regressão Logística

*Preenchido após a execução de `scripts/train_xgboost_tfidf.py` na base completa (Task 5 do plano da TAREFA 14).*

### 4.4 Resultado do protocolo completo

*Preenchido após a execução completa de `--model xgboost` (Task 5): comparação com a Tabela 3, tabelas de métricas, holdout e "como reproduzir" com os tempos observados.*
