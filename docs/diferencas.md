# Diferenças em relação ao artigo e decisões da réplica

Este documento registra onde a réplica se afasta do artigo *Rating Prediction in Brazilian Portuguese Reviews* e por quê. Cada decisão deve apontar para uma evidência em `results/`. Novas diferenças devem ser acrescentadas à seção 1 conforme o projeto avança.

## 1. Diferenças em relação ao artigo

| Tema | Artigo | Réplica (estado atual) | Observação |
| --- | --- | --- | --- |
| Representação do texto | 58 features manuais (POS, sintáticas, léxicas, concretude, estruturais, Twitter/X, miscelânea) | Baseline com **TF-IDF** (1–2-gramas) + Regressão Logística; as 58 features ainda não entram | O artigo cita TF-IDF e Bag of Words apenas como *trabalho futuro* (Seção 5). O baseline é, portanto, um ponto de partida próprio da réplica, não uma reprodução. |
| Normalização do texto | Não descreve nenhuma etapa de normalização | `normalize()` aplicada antes do TF-IDF | Avaliada na seção 2; decisão na seção 2.5. |
| Métricas | MAE, RMSE e AUC | Acurácia, F1 macro, MAE e RMSE | O AUC ainda não foi implementado no baseline. |
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
