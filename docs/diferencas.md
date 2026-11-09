# Diferenças em relação ao artigo e decisões da réplica

Este documento registra onde a réplica se afasta do artigo *Rating Prediction in Brazilian Portuguese Reviews* e por quê. Cada decisão deve apontar para uma evidência em `results/`. Novas diferenças devem ser acrescentadas à seção 1 conforme o projeto avança. A visão consolidada do projeto, com dados, features, arquitetura, resultados e reprodução, está em [`docs/documentacao.md`](documentacao.md).

## 1. Diferenças em relação ao artigo

| Tema | Artigo | Réplica (estado atual) | Observação |
| --- | --- | --- | --- |
| Representação do texto | 58 features manuais (POS, sintáticas, léxicas, concretude, estruturais, Twitter/X, miscelânea) | 58 features no protocolo da RQ1; **TF-IDF** (1–2-gramas) + Regressão Logística como baseline e comparação complementar | O artigo cita TF-IDF e Bag of Words apenas como *trabalho futuro* (Seção 5). O TF-IDF é um experimento próprio da réplica, não uma reprodução. |
| Normalização do texto | Não descreve nenhuma etapa de normalização | `normalize()` aplicada antes do TF-IDF | Avaliada na seção 2; decisão na seção 2.5. Não afeta o protocolo com as 58 features. |
| Métricas | MAE, RMSE e AUC | Acurácia, F1 macro, MAE e RMSE no baseline TF-IDF; MAE, RMSE, F1 macro, acurácia e AUC no protocolo do artigo | A AUC está implementada no harness do protocolo do artigo, não no baseline LR. |
| Representação no SVM (RQ1) | 58 features manuais | SVM com as 58 features (seção 3) e, como comparação complementar, SVM sobre TF-IDF | O SVM com as 58 features é a reprodução do artigo; o SVM com TF-IDF é um experimento próprio. |
| Random Forest (RQ1) | 58 features, 5 folds, `n_iter=300`, `random_state=None` | Modelo e grid implementados; execução versionada é uma amostra de 5.000 linhas com 1 candidato e `random_state=42` | A execução completa do grid de 19.440 combinações **não** foi feita; os números não são comparáveis à Tabela 3. Seção 6. |
| Gradient Boosting (RQ1) | Grid de 2160 combinações, `n_iter=300` | Mesmo grid e protocolo; nesta etapa só o **modo rápido** foi executado (`--sample-n 2000`, `n_iter=5`) | Execução completa pendente; números não comparáveis. Seções 5.2 e 5.3. |
| XGBoost (RQ1) | 58 features, 5 folds, `n_iter=300` | Apenas modo rápido (`--sample-n 2000`, `n_iter=20`) | Execução completa interrompida por hardware; números não comparáveis. Seção 4. |
| Regressão Logística (RQ1) | LR com 58 features (linha da Tabela 3) | LR + **TF-IDF** no baseline; LR com 58 features em **modo rápido** (amostra de 2.000, seção 7) | A comparação com a Tabela 3 exige a execução completa, pendente. A LR + TF-IDF da issue #7 é adicional. Seções 7 e 8.5. |
| TF-IDF × 58 features | Não existe no artigo (TF-IDF é trabalho futuro) | Comparação para SVM (completa), RF (amostra), XGB, GBT e LR (modo rápido/amostral) | Experimento próprio; resultados indicativos, sem teste estatístico. Seção 8.3. |
| IC de 95% | IC calculado com a linha da média incluída (6 valores, 5 g.l.) | IC calculado só entre os 5 folds (5 valores, 4 g.l.) | O IC publicado é cerca de 24% mais estreito. Seção 3.3. |
| Duplicatas treino/teste | Não discutidas no artigo | 280 combinações (`categoria`, `text`, `rating`) aparecem nos dois splits (351 ocorrências) | Herdado do split oficial; mantido por fidelidade e pode inflar as métricas. Ver `data/README.md` e seção 8.4. |
| Status geral da RQ1 | Cinco modelos completos | SVM no protocolo completo; LR em modo rápido (amostra de 2.000) | RF, XGB e GBT são parciais (amostra/modo rápido); a RQ1 da réplica tem **1 linha comparável** (SVM) à Tabela 3. Seções 7 e 8. |

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

O XGBoost foi o melhor modelo do artigo nas três métricas da Tabela 3 (MAE 0,9499 ± 0,0047; RMSE 1,3721 ± 0,0055; AUC 0,7390 ± 0,0046) e o escolhido para as RQs seguintes; por isso a TAREFA 14 o replica. Nesta etapa a execução usou o **modo rápido documentado pela própria issue** (restrição de hardware da máquina responsável); o protocolo completo não rodou até o fim — contexto, números e comando de reprodução estão nas subseções 4.4 e 4.5.

### 4.1 O que foi replicado

`scripts/train_article_model.py --model xgboost` aplica o mesmo protocolo de `code/train_model.py` do repositório original (detalhes no `README.md`) ao `XGBClassifier`:

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados (51.465 linhas) | Igual: **o teste participa da validação cruzada**. |
| Features | 58 (`str_`, `pos_`, `synt_`, `lex_`, `conc_`, `twt_`, `sbj_`) | Igual; `categoria`, `text` e `rating` ficam fora; linhas com `text` vazio mantidas. |
| Folds | `StratifiedKFold(5, shuffle=False)` por `categoria_rating` | Igual. |
| Busca | `RandomizedSearchCV`, `n_iter=300`, RMSE, `random_state=42` | Igual. |
| Grid | 9 dimensões (TAREFA 14), 26.244 combinações | Igual (exatamente os valores da TAREFA; teste fixa o grid). |
| Métricas | MAE, RMSE, AUC (classes 4–5 positivas, sobre predições binárias) | Igual, mais F1 macro, acurácia e AUC de diagnóstico por `predict_proba`. |
| Holdout | treino em `train.csv`, avaliação em `test.csv` | Igual, com os mesmos hiperparâmetros. |

### 4.2 Diferenças em relação ao artigo

1. **Rótulos 0..4 vs. 1..5.** O projeto usa `encode_rating` (labels 0..4) e converte para a escala de 1 a 5 estrelas somente no cálculo de MAE/RMSE (`compute_metrics`). O código original do XGB também treina com ratings 0..4 e converte as predições para 1..5 na agregação; a semântica do erro é a mesma (nota registrada na TAREFA 14). A diferença não altera o modelo: `XGBClassifier` ordena as classes e a numeração é apenas nominal.
2. **Versão da biblioteca.** A réplica fixa `xgboost==3.4.1` (`requirements.txt`); a versão usada pelo código original não é informada. Pequenas diferenças numéricas são possíveis (mesma ressalva do `scikit-learn-intelex`, seção 3.2).
3. **Paralelismo.** `random_state=42` fixo no estimador e `n_jobs=-1` no nível externo (CLI), com o estimador em `n_jobs=1` para que a busca e o refit por fold controlem o paralelismo — decisão igual à do Random Forest aprovado (reprodutibilidade por seed; o `n_jobs` não afeta o resultado).
4. **Sem padronização das features, como no artigo** (mesma justificativa da seção 3.2, item 2).
5. **`refit=False` na busca** e **IC de 95% com `n_folds − 1` graus de liberdade**, pelos mesmos motivos das seções 3.2 (item 5) e 3.3 — compare-se com o artigo pela média e pelo desvio.
6. **Melhores hiperparâmetros do artigo não informados** (`ARTICLE_BEST_PARAMS` vazio), igual ao caso do Random Forest.

### 4.3 Comparação complementar: representações sobre TF-IDF (modo rápido)

Nesta entrega também a comparação TF-IDF rodou em **modo rápido** (`--sample-n 2000`; 6:22 em 1 núcleo), e por isso **não há comparação com o baseline LR**: em `run_xgboost_tfidf` a comparação com `baseline_logreg.json` só é gravada na base completa (memória de `results/sample/` não é versionada).

O que é possível comparar no cenário rápido é o XGBoost nas duas representações — **ambas medidas no holdout** (treino no `train.csv` da amostra, avaliação no `test.csv` da amostra; treino ligeiramente diferente: 1.587 linhas no protocolo de 58 features contra 2.000 no TF-IDF):

| Métrica | XGBoost, 58 features (holdout: treino 1.587 → teste 413) | XGBoost + TF-IDF (holdout: 2.000 → 500) |
| --- | ---: | ---: |
| Acurácia | 0,3075 | 0,3740 |
| F1 macro | 0,3005 | 0,3734 |
| MAE (estrelas) | 1,0508 | 0,9420 |
| RMSE (estrelas) | 1,4159 | 1,3409 |
| AUC (predições binárias) | 0,7518 | 0,7267 |

**Leitura, com ressalvas:** no modo rápido, TF-IDF vence em MAE (−0,11 estrela), RMSE, acurácia e F1; a AUC binária vai para as 58 features. A direção é a mesma que a comparação completa do SVM (seção 3.5), em que TF-IDF trouxe ganhos claros — mas aqui os tamanhos de treino divergem e nada foi testado estatisticamente, então o resultado é **indicativo**, não conclusivo. Na base completa vale rodar `python scripts/train_xgboost_tfidf.py` (a comparação com a LR é gravada só nesse caso).

### 4.4 Resultado do protocolo em modo rápido (execução deste repositório)

**A execução completa não foi concluída nesta etapa.** Foi iniciada no notebook do responsável e interrompida ~40 min depois, por restrição de hardware (notebook portátil em uso, ventilação insuficiente para 9–10 núcleos sustentados por horas; ver seção 4.5). Nenhum artefato parcial foi gravado (a busca só grava ao fim). A entrega desta tarefa usa, portanto, o **modo rápido documentado pela própria issue**: `--sample-n 2000` com `n_iter` reduzido — o critério de conclusão da issue "execução completa com os 5 folds e o `n_iter` documentado" **não é atendido** e fica registrado aqui para a TAREFA 15 e para quem retomar.

Melhores hiperparâmetros encontrados no modo rápido (`RandomizedSearchCV`, 20 candidatos × 5 folds, ≈1.600 linhas de treino por fold e 1.587/413 no holdout; `seed=42`): `n_estimators=300`, `max_depth=10`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.4`, `gamma=0.0`, `reg_alpha=0.1`, `reg_lambda=1`, `min_child_weight=1`. Para efeito de escala: no modo rápido a busca amostra só 20 dos 26.244 candidatos do grid (≈0,08% do espaço), os folds cortam 2.000 linhas e existe uma única semente de amostragem (`seed=42`); a execução completa avaliará 300 candidatos sobre a base inteira e pode ordenar os hiperparâmetros de forma diferente.

**Comparação com a Tabela 3 do artigo** — os números não devem ser lidos como reprodução:

| Métrica | Artigo (Tabela 3) | Réplica *modo rápido* | Δ da média |
| --- | --- | --- | ---: |
| MAE | 0,9499 ± 0,0047 | 1,0215 ± 0,0327 | +0,0716 |
| RMSE | 1,3721 ± 0,0055 | 1,4337 ± 0,0490 | +0,0616 |
| AUC | 0,7390 ± 0,0046 | 0,7204 ± 0,0306 | −0,0186 |

Outras métricas do modo rápido (não constam na Tabela 3): F1 macro 0,3485 ± 0,0178; acurácia 0,3535 ± 0,0153. **Holdout do modo rápido** (treino 1.587 → teste 413, com os hiperparâmetros da busca): MAE 1,0508; RMSE 1,4159; AUC 0,7518.

**Leitura:**

- **O desvio-padrão entre folds no modo rápido (≈0,03–0,05) é 7–9× o do artigo (≈0,005)** — é o esperado com 1/26 dos dados por treino (1.587 contra ≈41.172). Toda diferença em relação à Tabela 3 está dentro do intervalo de confiança do modo rápido; isto não replica nem refuta o artigo.
- MAE e RMSE pioraram de modo consistente (≈ +0,06–0,07 estrelas), direção coerente com menos dados; a AUC caiu apenas −0,019.
- A AUC acima é a do artigo (predições binárias); a AUC de diagnóstico por `predict_proba` e as métricas de cada fold estão nos JSON do modo rápido.

Artefatos do modo rápido (não versionados): `results/sample/metrics/xgboost_rq1.json`, `xgboost_rq1_folds.csv`, `xgboost_rq1_holdout.json`, `xgboost_tfidf.json` e `../figures/confusion_matrix_xgboost.png`.

### 4.5 Como reproduzir

```bash
python scripts/train_article_model.py --model xgboost --sample-n 2000 --n-iter 20 --output-dir results/sample
# modo rápido usado nesta etapa: 33 s em ~9 núcleos
python scripts/train_xgboost_tfidf.py --sample-n 2000 --output-dir results/sample
# TF-IDF em modo rápido: 6:22 em 1 núcleo

# Execução completa (protocolo do artigo; NÃO executada integralmente nesta etapa):
python scripts/train_article_model.py --model xgboost        # ≈ 3 h de busca (estimada) + ~2 min de avaliação
python scripts/train_xgboost_tfidf.py                        # comparação com a reg logística, só na base completa
```

Observação: a estimativa de ≈3 h da execução completa vem do ritmo observado nos ~40 min em que ela aqui rodou (≈19% dos ≈1.500 fits totais da busca — 300 candidatos × 5 folds — em 9–10 núcleos), não de 1.500 fits concluídos. Para a execução futura até o fim: as linhas de whitelist do `.gitignore` (acrescentadas neste branch) já permitem versionar `results/metrics/xgboost_*`; o modo rápido grava em `results/sample/`, então o comando completo (sem `--sample-n`) sobrescreve apenas os artefatos válidos de `results/metrics/`.

## 5. Gradient Boosting com as 58 features (RQ1)

O Gradient Boosting (GBT) ficou entre os melhores modelos do artigo, com desempenho estatisticamente próximo do XGBoost (Tabela 3: MAE 0,9612 ± 0,0061; RMSE 1,3727 ± 0,0051; AUC 0,7318 ± 0,0058); por isso a TAREFA 13 o replica com o mesmo harness do SVM, do Random Forest e do XGBoost. Nesta etapa a execução usou o **modo rápido documentado** (seção 5.3), como na seção 4 do XGBoost; a execução completa não foi realizada.

### 5.1 O que foi replicado

`scripts/train_article_model.py --model gradient_boosting` aplica o protocolo da seção 3 ao `GradientBoostingClassifier`:

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados (51.465 linhas) | Igual: **o teste participa da validação cruzada**. |
| Features | 58 (`str_`, `pos_`, `synt_`, `lex_`, `conc_`, `twt_`, `sbj_`) | Igual; `categoria`, `text` e `rating` ficam fora; linhas com `text` vazio mantidas. |
| Folds | `StratifiedKFold(5, shuffle=False)` por `categoria_rating` | Igual. |
| Busca | `RandomizedSearchCV`, `n_iter=300`, RMSE, `random_state=42` | Igual no padrão; `--n-iter` é configurável (ver 5.2, item 1). |
| Grid | `n_estimators` ∈ {100, 200, 300, 500, 1000}, `learning_rate` ∈ {0,001; 0,01; 0,05; 0,1}, `max_depth` ∈ {3, 5, 10}, `min_samples_split` ∈ {5, 10, 15}, `min_samples_leaf` ∈ {1, 2, 4}, `max_features` ∈ {sqrt, log2, 0,5, None} | Igual: 5 × 4 × 3 × 3 × 3 × 4 = **2160 combinações** (o teste fixa o grid). |
| Métricas | MAE, RMSE, AUC (classes 4–5 positivas, sobre predições binárias) | Igual, mais F1 macro, acurácia e AUC de diagnóstico por probabilidades. |
| Holdout | treino em `train.csv`, avaliação em `test.csv` | Igual, com os mesmos hiperparâmetros. |

### 5.2 Diferenças em relação ao artigo

1. **`n_iter` menor.** O grid tem 2160 combinações e o artigo sorteou 300. Cada candidato exige 5 ajustes (um por fold), e o GBT cresce uma árvore por classe a cada iteração de boosting. Em uma medição exploratória (1 núcleo, 8 mil linhas, não versionada), uma combinação leve (`max_depth=3`, `max_features="sqrt"`, 100 árvores) levou ≈ 3 s e uma pesada (`max_depth=10`, `max_features=None`, 100 árvores) ≈ 61 s; extrapolando para 41 mil linhas e 1000 árvores, uma única combinação pesada passa de 40 minutos. Por isso `--n-iter` é configurável (padrão 300, como no artigo) e a execução usa um valor menor. O valor usado fica registrado na seção 5.3 e nos campos `n_iter_requested` e `n_candidates_evaluated` do JSON. Com poucos candidatos, a busca pode não alcançar a região do grid encontrada pelo artigo, e as médias podem ficar um pouco piores que as publicadas.
2. **Modo rápido com `n_iter=5`.** Com `--sample-n`, o padrão do GBT é 5 candidatos (o Random Forest usa 10; o SVM mantém 300 porque seu grid só tem 16 combinações). Os números do modo rápido não são comparáveis ao artigo.
3. **AUC de diagnóstico por probabilidades.** O harness prefere `decision_function` a `predict_proba`, e o GBT multiclasse tem as duas: a primeira devolve log-odds por classe. A TAREFA pede a AUC com probabilidades (P(4 estrelas) + P(5 estrelas)). Para isso o estimador é `GradientBoostingProbaScores`, uma subclasse que declara `diagnostic_from_proba = True`, e `diagnostic_scores` (`src/models/article_protocol.py`) passou a respeitar esse atributo. As previsões são idênticas às do `GradientBoostingClassifier` padrão, então MAE, RMSE, F1, acurácia e a AUC do artigo (predições binárias) não mudam. SVM, Random Forest e XGBoost não declaram o atributo e se comportam como antes (coberto por `tests/test_gradient_boosting.py`).
4. **Sem padronização das features, como no artigo** (mesma justificativa da seção 3.2, item 2).
5. **`refit=False` na busca** e **IC de 95% com `n_folds − 1` graus de liberdade**, pelos mesmos motivos das seções 3.2 (item 5) e 3.3: compare-se com o artigo pela média e pelo desvio.
6. **Melhores hiperparâmetros do artigo não informados** (`ARTICLE_BEST_PARAMS` vazio), igual ao Random Forest e ao XGBoost.

### 5.3 Resultado do protocolo em modo rápido (execução deste repositório)

**A execução completa não foi realizada nesta etapa.** Os resultados abaixo vêm do **modo rápido** do protocolo (`--sample-n 2000`, `n_iter=5`), o mesmo critério adotado para o XGBoost (seção 4.4). O motivo é o custo de tempo: cada candidato do grid completo exige 5 ajustes em ≈ 41 mil linhas, e uma única combinação pesada leva dezenas de minutos (seção 5.2, item 1). Os números **não devem ser lidos como reprodução da Tabela 3**.

Melhores hiperparâmetros encontrados no modo rápido (`RandomizedSearchCV`, 5 candidatos × 5 folds, ≈ 1.600 linhas de treino por fold e 1.587/413 no holdout; `seed=42`): `n_estimators=100`, `max_depth=5`, `learning_rate=0.01`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features=None`. Uma taxa de aprendizado de 0,01 com apenas 100 árvores é uma configuração de pouco ajuste: reflete a amostra minúscula e o sorteio de poucos candidatos, e **não é uma conclusão sobre os melhores hiperparâmetros**.

**Comparação com a Tabela 3 do artigo** (média ± desvio-padrão entre os 5 folds):

| Métrica | Artigo (Tabela 3) | Réplica *modo rápido* | Δ da média |
| --- | --- | --- | ---: |
| MAE | 0,9612 ± 0,0061 | 1,0400 ± 0,0549 | +0,0788 |
| RMSE | 1,3727 ± 0,0051 | 1,4527 ± 0,0584 | +0,0800 |
| AUC | 0,7318 ± 0,0058 | 0,7072 ± 0,0133 | −0,0246 |

IC de 95% do modo rápido: MAE [0,9718; 1,1082], RMSE [1,3801; 1,5252], AUC [0,6907; 0,7237]. **Holdout do modo rápido** (treino 1.587 → teste 413, com os hiperparâmetros da busca): MAE 1,0726; RMSE 1,4901; AUC 0,7422. A acurácia, o F1 macro e a AUC de diagnóstico por probabilidades (`auc_scores`) de cada fold estão nos JSON do modo rápido.

**Leitura:**

- **O desvio-padrão entre folds no modo rápido é muito maior que o do artigo**: 9 vezes no MAE, 11 vezes no RMSE e 2,3 vezes na AUC. É o esperado com cerca de 1/26 dos dados por treino.
- **As médias do artigo ficam um pouco fora do IC de 95% do modo rápido nas três métricas** (por exemplo, MAE 0,9612 contra um limite inferior de 0,9718). Isso é diferente do XGBoost, em que as diferenças couberam no IC. A direção é coerente com treinar em muito menos dados e com apenas 5 candidatos (a busca pode não ter chegado à região do grid encontrada pelo artigo); não indica erro no protocolo, mas também não permite afirmar equivalência com o artigo.
- O holdout tem AUC (0,7422) acima da média dos folds (0,7072), mas é uma única divisão com 413 linhas de teste; não deve ser comparado com a média dos folds.

Artefatos esperados do modo rápido (não versionados; gerados na máquina do responsável): `results/sample/metrics/gradient_boosting_rq1.json`, `gradient_boosting_rq1_folds.csv`, `gradient_boosting_rq1_holdout.json`, `gradient_boosting_tfidf.json` e `results/sample/figures/confusion_matrix_gradient_boosting.png`.

### 5.4 Comparação complementar: GBT com TF-IDF (modo rápido)

`scripts/train_gradient_boosting_tfidf.py` treina o GBT sobre a matriz TF-IDF da Issue 6 (mesmo vetorizador e mesmo split oficial treino → teste do baseline). Os hiperparâmetros vêm de uma **busca curta feita só no `train.csv`** (`RandomizedSearchCV`, 6 candidatos, 3 folds, RMSE, grid reduzido em `TFIDF_GBT_GRID`); o teste só é usado na avaliação final. O grid é menor que o do artigo porque o TF-IDF tem ≈ 28 mil colunas esparsas, e `max_features=None` com árvores profundas seria impraticável. A opção `--no-search` usa parâmetros fixos (100 árvores, profundidade 3, `max_features="sqrt"`); em um teste exploratório, com validação interna de 20% do treino (não versionado), essa configuração ficou claramente subajustada (RMSE ≈ 1,41, contra ≈ 1,18 com 300 árvores de profundidade 5), e por isso a busca é o padrão.

Esta entrega rodou em **modo rápido** (`--sample-n 2000`, 36,0 s com `--n-jobs -1`; 2.000 linhas de treino, 500 de teste, 28.500 colunas) e, por isso, **não há comparação com o baseline LR**: a comparação com `baseline_logreg.json` só é gravada na base completa. Melhores hiperparâmetros da busca curta: `n_estimators=200`, `max_depth=8`, `learning_rate=0.1`, `max_features=0.05`. Métricas no teste da amostra: acurácia 0,3860; F1 macro 0,3839; MAE 0,9260; RMSE 1,3364; AUC (artigo) 0,7333; AUC por probabilidades 0,8316.

O que é possível comparar no cenário rápido é o GBT nas duas representações, **ambas medidas no holdout** (treinos de tamanhos ligeiramente diferentes: 1.587 linhas nas 58 features contra 2.000 no TF-IDF):

| Métrica | GBT, 58 features (treino 1.587 → teste 413) | GBT + TF-IDF (treino 2.000 → teste 500) |
| --- | ---: | ---: |
| MAE (estrelas) | 1,0726 | 0,9260 |
| RMSE (estrelas) | 1,4901 | 1,3364 |
| AUC (predições binárias) | 0,7422 | 0,7333 |

**Leitura, com ressalvas:** no modo rápido, o TF-IDF vence em MAE e RMSE (cerca de −0,15 estrela em ambos) e a AUC binária vai para as 58 features (+0,009). É a mesma direção observada para o XGBoost no modo rápido (seção 4.3). Mas são uma única divisão por representação, com 413 e 500 linhas de teste, treinos de tamanhos diferentes e sem teste estatístico; trate como indício, não como conclusão. A AUC por probabilidades (0,8316) é um diagnóstico diferente da AUC do artigo e não se compara com a Tabela 3.

**A comparação com a Regressão Logística (TAREFA 15) não foi realizada nesta etapa**; ela exige a execução na base completa (seção 5.5). Ressalvas a registrar junto dos números dessa execução: o TF-IDF remove 1 linha de `text` vazio por split, enquanto o GBT com 58 features usa todas; a busca do GBT é curta; e não há teste estatístico da diferença.

### 5.5 Como reproduzir

```bash
python scripts/train_article_model.py --model gradient_boosting --sample-n 2000   # modo rápido (n_iter=5), usado nesta etapa
python scripts/train_gradient_boosting_tfidf.py --sample-n 2000                   # TF-IDF em modo rápido: 36 s

# Execução completa (protocolo do artigo; NÃO executada nesta etapa):
python scripts/train_article_model.py --model gradient_boosting --n-iter 20       # n_iter reduzido
python scripts/train_gradient_boosting_tfidf.py                                   # comparação com a regressão logística, só na base completa
```

Observação: a duração da execução completa depende do `--n-iter` e dos núcleos. Uma estimativa grosseira, por extrapolação de medições em 1.600 e 8 mil linhas (não de uma execução completa), é da ordem de 20 minutos por candidato em 1 núcleo; divida pelo número de núcleos e some o reajuste por fold e o holdout.

## 6. Random Forest com as 58 features (RQ1)

### 6.1 O que foi implementado

`src/models/random_forest.py` define `RANDOM_FOREST_SPEC` com `RandomForestClassifier` e o grid da TAREFA 12:

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados | Igual; textos vazios mantidos |
| Folds | 5 folds estratificados por `categoria_rating`, `shuffle=False` | Igual |
| Busca | `RandomizedSearchCV`, 300 candidatos, RMSE | Igual no código; a execução versionada usou 1 candidato |
| Grid | 8 dimensões (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `max_leaf_nodes`, `min_impurity_decrease`) | **19.440 combinações** |
| `random_state` | `None` no script original | `42`, para reprodutibilidade |
| Melhores parâmetros do artigo | Não informados | `ARTICLE_BEST_PARAMS` vazio |
| Métricas | MAE, RMSE, AUC | Igual + F1 macro, acurácia e AUC por scores |
| Holdout | — | Treino em `train.csv`, avaliação em `test.csv` |

### 6.2 Estado da execução

O artefato versionado `results/metrics/random_forest_rq1.json` **não é a execução completa do protocolo**. Ele registra:

- `sample_n=5000`, `n_rows=5000`, `n_train_csv_rows=4034`, `n_test_csv_rows=966`;
- `n_iter_requested=1`, `n_candidates_evaluated=1`;
- um `param_grid` com **um único valor por hiperparâmetro**, que não corresponde ao grid de 19.440 combinações do código;
- scikit-learn **1.8.0**.

O CSV de folds (`random_forest_rq1_folds.csv`) mostra `n_train=4000` e `n_test=1000` por fold, consistente com uma validação cruzada de 5 folds sobre as 5.000 linhas da amostra. O comando exato que gerou o artefato não está registrado. **A execução completa do Random Forest continua pendente.**

### 6.3 Números da amostra (não comparáveis à Tabela 3)

| Métrica | Média ± desvio | IC 95% |
| --- | --- | --- |
| MAE | 1,0784 ± 0,0200 | [1,0536; 1,1032] |
| RMSE | 1,5165 ± 0,0296 | [1,4797; 1,5532] |
| AUC | 0,7168 ± 0,0110 | [0,7032; 0,7304] |
| Acurácia | 0,3532 ± 0,0122 | [0,3381; 0,3683] |
| F1 macro | 0,3300 ± 0,0157 | [0,3105; 0,3495] |
| AUC por scores | 0,7925 ± 0,0101 | [0,7800; 0,8050] |

**Holdout da amostra** (4.034 → 966): MAE 1,1087; RMSE 1,5407; F1 macro 0,3131; acurácia 0,3375; AUC 0,7150; AUC por scores 0,7867.

**Tempos da amostra:** busca 0,22 s; avaliação 0,28 s. Não representam o custo do protocolo completo.

**RF + TF-IDF (amostra de 5.000):** `results/metrics/random_forest_tfidf.json` registra MAE 0,8088; RMSE 1,2487; F1 macro 0,4458; acurácia 0,4568; AUC 0,7867; AUC por scores 0,8647; 28.500 features; `n_iter=1`; 8,96 s; scikit-learn 1.8.0; `comparison_with_baseline=null` porque o modo rápido não grava a comparação.

### 6.4 Como reproduzir

```bash
python scripts/train_article_model.py --model random_forest                      # protocolo completo (grid de 19.440; não executado nesta etapa)
python scripts/train_article_model.py --model random_forest --sample-n 2000      # modo rápido (n_iter padrão 10)
python scripts/train_random_forest_tfidf.py                                      # TF-IDF na base completa
python scripts/train_random_forest_tfidf.py --sample-n 2000                      # TF-IDF rápido
```

### 6.5 Limitações

- Só a amostra de 5.000 foi versionada; o grid completo e os 300 candidatos não foram executados.
- O artefato foi gerado com scikit-learn 1.8.0, diferente dos demais (1.9.1).
- O `param_grid` do JSON não corresponde ao grid do código; trate os números como smoke test, não como protocolo.
- O TF-IDF remove 1 linha de `text` vazio por split; o RF com 58 features mantém todas.

## 7. Regressão Logística com as 58 features (RQ1)

A LR completa a lista de modelos da RQ1. Nesta etapa foi executada no **modo rápido** (amostra de 2.000), o mesmo critério do XGBoost (seção 4) e do Gradient Boosting (seção 5); a execução completa — a que é comparável à Tabela 3 — **continua pendente**. Quando acontecer, ela grava nos mesmos caminhos de `results/metrics/` (substituindo os amostrais) e as subseções 7.3, 8.1 e 8.5 serão atualizadas.

### 7.1 O que foi implementado

`src/models/logistic_regression.py` define `LOGISTIC_REGRESSION_SPEC` com o grid exato de `code/RQ1/scripts/logistic_regression.py` do repositório original:

| Item | Artigo | Réplica |
| --- | --- | --- |
| Dados | `train.csv` + `test.csv` concatenados (51.465 linhas) | Igual no protocolo; nesta etapa, amostra de 2.000 (o teste participa da validação cruzada) |
| Features | 58 (`str_`, `pos_`, `synt_`, `lex_`, `conc_`, `twt_`, `sbj_`) | Igual; `categoria`, `text` e `rating` ficam fora; textos vazios mantidos |
| Folds | `StratifiedKFold(5, shuffle=False)` por `categoria_rating` | Igual |
| Busca | `RandomizedSearchCV`, `n_iter=300`, RMSE, `random_state=42` | Igual; grid de 50 combinações → **exaustiva nas 50** (aviso do sklearn registrado no JSON) |
| Grid | `C` ∈ {0,01; 0,1; 1; 10; 100} × `solver` ∈ {newton-cg, lbfgs, liblinear, sag, saga} × `class_weight` ∈ {None, balanced}, com `max_iter=10000` e `random_state=None` | Igual: **5 × 5 × 2 = 50 combinações** (teste fixa o grid) |
| Métricas | MAE, RMSE, AUC (classes 4–5 positivas, predições binárias) | Igual, mais F1 macro, acurácia e AUC de diagnóstico por `decision_function` |
| Holdout | — | Treino no `train.csv` da amostra, avaliação no `test.csv` da amostra |
| Melhores do artigo | `solver=lbfgs, C=100, class_weight=None` (`LogisticRegression_best_params.txt`) |Registrados em `ARTICLE_BEST_PARAMS` |

### 7.2 Diferenças em relação ao artigo

1. **`liblinear` não resolve multiclasse no scikit-learn atual.** O esquema *one-vs-rest* era exclusivo do `liblinear` e foi descontinuado (deprecado na 1.5 e removido nas versões seguintes): no scikit-learn 1.9.1 o `LogisticRegression` levanta erro para ≥ 3 classes. Dos 50 candidatos do grid, os **10 `liblinear`** (5 `C` × 2 `class_weight`) falham em cada fold com resultado NaN (`FitFailedWarning`); a busca avalia as 50 (o JSON registra `n_candidates_evaluated=50`), mas o ranking considera só os 40 válidos. No código original (sklearn antigo), os 10 avaliavam como OvR. Os melhores hiperparâmetros do artigo (`lbfgs`, `C=100`) continuam alcançáveis; coberto pelo teste `test_liblinear_multiclasse_falha_no_sklearn_atual`.
2. **Versões não pinadas no Colab.** A execução no **Colab** usou scikit-learn **1.6.1** (numpy/pandas do Colab, prática dos notebooks da equipe, seção de reprodução); a reexecução local usou **1.9.1** (`requirements.txt`). As métricas coincidiram ao 4.º decimal — o JSON registra a versão de cada execução (mesma prática do Random Forest com 1.8.0, seção 6.2).
3. **Sem padronização das features,** como no artigo (mesma justificativa da seção 3.2, item 2); com `max_iter=10000` do grid. Avisos de convergência de `lbfgs`/`sag`/`saga` com `C` pequeno são esperados e aparecem no log.
4. **`refit=False` na busca e IC de 95% com `n_folds − 1` g.l.** — mesmos motivos das seções 3.2 (item 5) e 3.3.
5. **Execução distribuída (Colab + local).** O modo rápido rodou no Colab free (2 vCPU) e foi reproduzido localmente (10 núcleos, 91 s de busca contra ≈ 1.730 s do Colab); os resultados são idênticos — ver 7.3.

### 7.3 Resultado do modo rápido (execuções Colab e local)

Execução no Colab: amostra de 2.000, 50 candidatos × 5 folds, seed 42, busca em ≈ 28,8 min. A reexecução local reproduziu os mesmos agregados ao 4.º decimal (controle de reprodutibilidade). Melhores hiperparâmetros da amostra: `solver=newton-cg, C=1, class_weight=None` — **não comparáveis** aos do artigo; com 1.600 linhas por fold e 40 candidatos válidos, o ranking não é conclusivo.

Comparação com a Tabela 3 — **os números não devem ser lidos como reprodução**:

| Métrica | Artigo (Tabela 3) | Réplica *modo rápido* | Δ da média | IC 95% da amostra |
| --- | --- | --- | ---: | --- |
| MAE | 1,0149 ± 0,0043 | 1,0230 ± 0,0515 | +0,0081 | [0,9591; 1,0869] |
| RMSE | 1,4691 ± 0,0042 | 1,4548 ± 0,0579 | −0,0143 | [1,3828; 1,5267] |
| AUC | 0,7328 ± 0,0064 | 0,7266 ± 0,0338 | −0,0062 | [0,6847; 0,7686] |

Outras métricas (não constam na Tabela 3): acurácia 0,3630 ± 0,0233, IC [0,3341; 0,3919]; F1 macro 0,3477 ± 0,0257, IC [0,3158; 0,3796]. **Holdout da amostra** (treino 1.600 → teste 400): MAE 1,0048; RMSE 1,4082; AUC 0,7598; acurácia 0,3656; F1 0,3513.

Métricas por fold da amostra (`logistic_regression_rq1_folds.csv`):

| Fold | n_train | n_test | MAE | RMSE | F1 macro | Acurácia | AUC | AUC por scores |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.600 | 400 | 1,0000 | 1,4577 | 0,3703 | 0,3850 | 0,7294 | 0,8143 |
| 2 | 1.600 | 400 | 1,1075 | 1,5387 | 0,3064 | 0,3250 | 0,6719 | 0,7586 |
| 3 | 1.600 | 400 | 0,9800 | 1,4124 | 0,3683 | 0,3775 | 0,7438 | 0,8083 |
| 4 | 1.600 | 400 | 1,0350 | 1,4748 | 0,3475 | 0,3600 | 0,7259 | 0,7973 |
| 5 | 1.600 | 400 | 0,9925 | 1,3901 | 0,3458 | 0,3675 | 0,7622 | 0,8259 |

**Leitura:**

- O desvio-padrão entre folds na amostra (≈0,05) é ~8× o do artigo (≈0,004): o esperado com 1/26 do treino por fold; o fold 2 é o pior (MAE 1,1075) e explica boa parte do desvio.
- As médias do artigo ficam **dentro** do IC da amostra nas três métricas: a execução amostral não replica nem refuta a Tabela 3.
- Diagonal da matriz de confusão fora-da-dobra: 726/2.000 = 36,3%, igual à acurácia agregada.
- **Controle de reprodutibilidade:** Colab (sklearn 1.6.1, 2 vCPU) e local (sklearn 1.9.1, 10 núcleos) coincidem ao 4.º decimal em todas as métricas e fold a fold, com os mesmos melhores hiperparâmetros — a amostragem e os folds são determinísticos (seed 42).

Artefatos **versionados** (amostra; serão substituídos pela execução completa): `results/metrics/logistic_regression_rq1.json` (com `sample_n=2000`, `n_candidates_evaluated=50` e o bloco `article.comparison`), `logistic_regression_rq1_folds.csv`, `logistic_regression_rq1_holdout.json` e `results/figures/confusion_matrix_logistic_regression.png`.

### 7.4 Comparação amostral com XGBoost e Gradient Boosting (referência fraca)

Nos modos rápidos já registrados (mesma amostra de 2.000; **número de candidatos diferente por modelo**): XGB MAE 1,0215 ± 0,0327 (seção 4.4); GBT 1,0400 ± 0,0549 (seção 5.3); LR 1,0230 ± 0,0515 (7.3). A LR fica no mesmo patamar dos dois — indício de viabilidade, **não** um ranking: candidatos, hiperparâmetros e Amostragem diferem por execução, e não há teste estatístico.

### 7.5 Como reproduzir

```bash
python scripts/train_article_model.py --model logistic_regression --sample-n 2000   # modo rápido (≈2–3 min local/10 núcleos; ≈29 min no Colab free)
python scripts/train_article_model.py --model logistic_regression                   # COMPLETA (pendente; grava nos mesmos caminhos)
python scripts/monitor_regressao_logistica.py --log results/logistic_regression_run.log  # com --verbose 11: fases, %, ETA
```

Notebook do Colab: `notebooks/logistic_regression_rq1_colab.ipynb` — clona esta branch, baixa os splits do repositório original (ou aceita upload manual), executa modo rápido e a execução completa (~2–2,5 h nas 2 vCPU) com progresso por fit no output e cópia dos artefatos para o Drive; compara com a Tabela 3 ao final. Comando de lançamento independente do terminal (usado na execução amostral local): `PYTHONUNBUFFERED=1` recomendado para o log fluir em tempo real (os workers do joblib bufferiam as linhas `[CV]`).

## 8. Consolidação da RQ1, TF-IDF × 58 features e pendências

### 8.1 Tabela consolidada da RQ1

Média ± desvio-padrão entre os 5 folds. **Apenas o SVM é comparável à Tabela 3.**

| Modelo | Execução | MAE | RMSE | AUC | Acurácia | F1 macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SVM | Completa (51.465 linhas) | 1,0430 ± 0,0062 | 1,5085 ± 0,0112 | 0,7303 ± 0,0047 | 0,3796 ± 0,0020 | 0,3550 ± 0,0023 |
| Random Forest | Amostra 5.000, 1 candidato | 1,0784 ± 0,0200 | 1,5165 ± 0,0296 | 0,7168 ± 0,0110 | 0,3532 ± 0,0122 | 0,3300 ± 0,0157 |
| XGBoost | Modo rápido 2.000, 20 candidatos | 1,0215 ± 0,0327 | 1,4337 ± 0,0490 | 0,7204 ± 0,0306 | 0,3535 ± 0,0153 | 0,3485 ± 0,0178 |
| Gradient Boosting | Modo rápido 2.000, 5 candidatos | 1,0400 ± 0,0549 | 1,4527 ± 0,0584 | 0,7072 ± 0,0133 | n/d | n/d |
| Regressão Logística | Modo rápido 2.000, 50 candidatos (10 `liblinear` falham) | 1,0230 ± 0,0515 | 1,4548 ± 0,0579 | 0,7266 ± 0,0338 | 0,3630 ± 0,0233 | 0,3477 ± 0,0257 |

Referência do artigo (Tabela 3): GBT 0,9612 / 1,3727 / 0,7318; LR 1,0149 / 1,4691 / 0,7328; RF 0,9717 / 1,4001 / 0,7367; XGB 0,9499 / 1,3721 / 0,7390; SVM 1,0431 / 1,5085 / 0,7304.

**Conclusão:** o SVM reproduz o artigo; RF, XGB, GBT e LR estão parciais (amostra/modo rápido). A RQ1 da réplica tem **1 linha comparável à Tabela 3** (SVM; LR completa pendente) e ainda não permite ranking estatístico dos cinco algoritmos.

### 8.2 Holdout complementar

| Modelo | Representação | Treino → teste | MAE | RMSE | AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| LR | TF-IDF | 41.171 → 10.292 | 0,6819 | 1,0619 | n/d |
| SVM | 58 features | 41.172 → 10.293 | 1,0476 | 1,5160 | 0,7295 |
| SVM | TF-IDF | 41.171 → 10.292 | 0,6830 | 1,0596 | 0,8261 |
| Random Forest | 58 features | 4.034 → 966 | 1,1087 | 1,5407 | 0,7150 |
| Random Forest | TF-IDF | 5.000 → 1.250 | 0,8088 | 1,2487 | 0,7867 |
| XGBoost | 58 features | 1.587 → 413 | 1,0508 | 1,4159 | 0,7518 |
| XGBoost | TF-IDF | 2.000 → 500 | 0,9420 | 1,3409 | 0,7267 |
| Gradient Boosting | 58 features | 1.587 → 413 | 1,0726 | 1,4901 | 0,7422 |
| Gradient Boosting | TF-IDF | 2.000 → 500 | 0,9260 | 1,3364 | 0,7333 |
| Regressão Logística | 58 features | 1.600 → 400 (amostra) | 1,0048 | 1,4082 | 0,7598 |

Os holdouts de RF/XGB/GBT/LR são recortes das amostras; apenas LR + TF-IDF e SVM usam a base completa. A AUC do baseline LR não foi registrada.

### 8.3 TF-IDF × 58 features

| Modelo | MAE 58 features | MAE TF-IDF | Δ (TF-IDF − 58) | Status |
| --- | ---: | ---: | ---: | --- |
| SVM | 1,0476 | 0,6830 | −0,3646 | Completo, mesmo split oficial |
| Random Forest | 1,1087 | 0,8088 | −0,2999 | Amostral, treinos diferentes |
| XGBoost | 1,0508 | 0,9420 | −0,1088 | Modo rápido, treinos diferentes |
| Gradient Boosting | 1,0726 | 0,9260 | −0,1466 | Modo rápido, treinos diferentes |
| Regressão Logística | 1,0048 (amostra, holdout) | 0,6819 (completo) | −0,3229 | Modo rápido × baseline completo |

Em todas as comparações disponíveis, o TF-IDF teve MAE menor. A diferença é grande no SVM completo e nos outros modos rápidos. Ressalvas: o TF-IDF remove 1 linha vazia por split; o SVM com 58 features usa hiperparâmetros buscados e o com TF-IDF usa parâmetros fixos; os modos rápidos usam amostras e treinos de tamanhos diferentes (a LR da amostra avalia holdout de 400 linhas contra o teste oficial de 10.292 do baseline); não há teste estatístico. A comparação é **indicativa**, não conclusiva, e não mede o efeito das 58 features no protocolo completo dos demais modelos.

A **LR + TF-IDF** do baseline (MAE 0,6819; RMSE 1,0619; acurácia 0,4898; F1 macro 0,4862 no teste oficial) mantém a contraparte com 58 features **amostral** (holdout 1,0048; acurácia 0,3656; F1 0,3513 — holdout de 400 linhas). É a mesma direção dos demais modelos (TF-IDF abaixo no MAE), mas o par 58 features × TF-IDF da LR só fica comparável quando a execução completa estiver registrada (seção 7). Os números amostrais não devem ser comparados à linha Logistic Regression da Tabela 3.

### 8.4 Decisão sobre as duplicatas

As 280 combinações distintas (351 ocorrências) de (`categoria`, `text`, `rating`) entre treino e teste foram **mantidas**. Remover duplicatas mudaria a base do artigo e invalidaria a comparação direta; o artigo não discute o problema. A limitação fica registrada: o modelo pode ter visto no treino exemplos do teste, o que tende a **inflar** as métricas. O mesmo vale para as duplicatas dentro de cada split (949 linhas repetidas no treino e 104 no teste, além da primeira ocorrência), que não foram tratadas.

### 8.5 LR do artigo × LR da réplica

A linha da Tabela 3 é **Logistic Regression com as 58 features** (MAE 1,0149 ± 0,0043; RMSE 1,4691 ± 0,0042; AUC 0,7328 ± 0,0064). A LR da réplica usa **TF-IDF** (baseline da issue #7: MAE 0,6819, RMSE 1,0619, acurácia 0,4898, F1 macro 0,4862 no teste oficial) e tem, desde a seção 7, a contraparte **amostral** com as 58 features (modo rápido: MAE 1,0230 ± 0,0515). São experimentos diferentes: a LR + TF-IDF **não** reproduz a linha da Tabela 3; e a LR com 58 features só a reproduz **na execução completa**, ainda pendente — os números amostrais não fecham essa comparação.

### 8.6 Pendências para completar a RQ1

1. Executar Random Forest no protocolo completo (grid de 19.440, `n_iter=300`).
2. Executar XGBoost no protocolo completo (grid de 26.244, `n_iter=300`).
3. Executar Gradient Boosting no protocolo completo (grid de 2.160, `n_iter` documentado).
4. Executar a LR com as 58 features **no protocolo completo** (o modo rápido da seção 7 já está registrado; a completa substitui os artefatos amostrais).
5. Executar as comparações TF-IDF + RF/XGB/GBT na base completa.
6. Rodar Friedman/Nemenyi para a PP1.
7. Executar PP2 (grupos), PP3 (ablation/RFE) e PP4 (por categoria).

Enquanto esses itens não forem concluídos, os números de modo rápido/amostrais devem ser lidos apenas como diagnóstico interno, nunca como reprodução do artigo.
