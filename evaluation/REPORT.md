# Relatório de avaliação offline

**Projeto:** `zC4sTr0/semantic-paper-router`  
**Data da execução:** 2026-09-06 17:28:53 -03:00  
**Referências:** `data/references.json`  
**Avaliação:** `data/evaluation.json`  
**Harness:** `scripts/evaluate.py`

## Escopo

Esta avaliação mede o baseline local `BagOfWordsEmbedder` +
`SemanticClassifier`. Ela não chama AWS, não usa Bedrock e não representa a
qualidade de um modelo semântico real.

O conjunto contém 15 artigos:

- 3 exemplos de Computer Science;
- 3 exemplos de Biology;
- 3 exemplos de Economics;
- 6 artigos fora das três categorias.

Os 9 primeiros são paráfrases e não cópias literais das referências do corpus.

## Execução reproduzível

```bash
python scripts/evaluate.py \
  --references data/references.json \
  --examples data/evaluation.json \
  --format json \
  --min-score 0 \
  --min-margin 0 \
  --output evaluation/results-baseline.json
```

## Resultado sem rejeição

| Métrica | Resultado |
|---|---:|
| Artigos avaliados | 15 |
| Acertos | 7 |
| Erros | 8 |
| Accuracy | 46,7% |
| Fora do escopo esperados | 6 |
| Fora do escopo rejeitados | 0 |

### Precision e recall por classe

| Classe | Precision | Recall |
|---|---:|---:|
| Biology | 50,0% | 66,7% |
| Computer Science | 37,5% | 100,0% |
| Economics | 66,7% | 66,7% |
| Out of scope | 0,0% | 0,0% |
| Unknown | 0,0% | 0,0% |

### Matriz de confusão

Linhas são os rótulos esperados; colunas são os rótulos previstos.

| Esperado / Previsto | Biology | Computer Science | Economics | Unknown |
|---|---:|---:|---:|---:|
| Biology | 2 | 0 | 1 | 0 |
| Computer Science | 0 | 3 | 0 | 0 |
| Economics | 0 | 1 | 2 | 0 |
| Out of scope | 2 | 4 | 0 | 0 |

## Resultado com `min_score=0.30`

Comando:

```bash
python scripts/evaluate.py \
  --references data/references.json \
  --examples data/evaluation.json \
  --format json \
  --min-score 0.30 \
  --min-margin 0 \
  --output evaluation/results-score-030.json
```

| Métrica | Resultado |
|---|---:|
| Artigos avaliados | 15 |
| Acertos | 7 |
| Erros | 8 |
| Accuracy | 46,7% |
| Fora do escopo esperados | 6 |
| Rejeitados como `unknown` | 2 |
| Artigos válidos rejeitados | 1 |

O limiar rejeitou um artigo válido de Economics (`econ-eval-003`) e um artigo
fora do escopo (`oos-eval-001`). Os outros cinco artigos fora do escopo ainda
foram aceitos como Biology ou Computer Science.

## Casos classificados incorretamente sem rejeição

| ID | Esperado | Previsto | Score | Margem |
|---|---|---|---:|---:|
| `bio-eval-003` | Biology | Economics | 0,333 | 0,013 |
| `econ-eval-002` | Economics | Computer Science | 0,387 | 0,000 |
| `oos-eval-001` | Out of scope | Computer Science | 0,289 | 0,011 |
| `oos-eval-002` | Out of scope | Biology | 0,416 | 0,127 |
| `oos-eval-003` | Out of scope | Biology | 0,302 | 0,013 |
| `oos-eval-004` | Out of scope | Computer Science | 0,333 | 0,013 |
| `oos-eval-005` | Out of scope | Computer Science | 0,333 | 0,000 |
| `oos-eval-006` | Out of scope | Computer Science | 0,471 | 0,132 |

## Conclusão

O pipeline funciona tecnicamente, mas o baseline ainda não é um classificador
confiável de artigos gerais. O corpus é pequeno, a representação é lexical e
não há evidência suficiente para escolher um threshold de rejeição seguro.

O adapter `BedrockTitanEmbedder` está implementado e testado offline, mas não
foi chamado nesta avaliação porque isso exigiria autorização para custo AWS.

## Próximos passos recomendados

1. Avaliar o mesmo conjunto usando Titan Embeddings com autorização de custo.
2. Aumentar o corpus e separar claramente calibração de teste.
3. Medir accuracy seletiva, coverage, risco de rejeição e F1 macro.
4. Calibrar `min_score` e `min_margin` somente em um split de calibração.
5. Adicionar mais categorias e exemplos fora do domínio.
