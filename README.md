# Classificação Fine-Grained de Raças de Cães

Trabalho final da disciplina **Processamento e Análise de Imagens**, da
pós-graduação em LLM e IA Generativa.

**Integrantes:** Mariana Zanon e Victor Macaúbas

**Notebook de entrega:** [`notebooks/99_final.ipynb`](notebooks/99_final.ipynb)

## Objetivo

O projeto compara estratégias para classificar imagens do Stanford Dogs em 120
raças. A pergunta central é quanto de representação visual precisa ser aprendida
do zero e quanto pode ser transferida de um modelo pré-treinado quando há dados
limitados em uma tarefa de classificação fine-grained.

## Base de dados e ambiente

Usamos o dataset [Stanford Dogs](https://huggingface.co/datasets/maurice-fp/stanford-dogs),
em sua versão parquet no Hugging Face. Ele contém 20.580 imagens de 120 raças,
com split oficial de 12.000 imagens de treino e 8.580 de teste. Reservamos 15%
do treino para validação, com semente fixa 42, resultando em 10.200 imagens de
treino e 1.800 de validação. O treino oficial é perfeitamente balanceado, com
100 imagens por raça.

Os experimentos que exigem treinamento foram executados no Google Colab com GPU
T4. O notebook de entrega, porém, usa `RETRAIN = False`: ele lê as predições e
figuras versionadas em `reports/`, sem baixar a base, embeddings ou checkpoints.

## Experimentos e resultados

| Experimento | Estratégia | Split | Top-1 | Top-5 | F1 macro |
|---|---|---:|---:|---:|---:|
| Experimento 01 | CNN pequena treinada do zero | validação | 10,39% | 29,06% | 7,98% |
| Experimento 02 | Linear probe sobre embeddings ResNet50 | validação | 88,39% | 98,00% | 88,05% |
| Experimento 02 — ablação | Linear probe sem BatchNorm | validação | **90,89%** | **99,22%** | **90,56%** |
| Experimento 03 | Fine-tuning parcial da ResNet50 | validação | 85,33% | 98,61% | 83,97% |
| Experimento 03 | Fine-tuning parcial da ResNet50 | teste | 85,17% | 98,58% | 84,42% |

O principal achado é o salto de 10,39% para 88,39% ao transferir a
representação visual da ResNet50. Nesta execução, a ablação sem BatchNorm foi o
melhor resultado de validação, superando inclusive o fine-tuning parcial.

## Reprodução

Para abrir a entrega no Colab, use o botão abaixo:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victormacaubas/project-image-processing/blob/main/notebooks/99_final.ipynb)

Em um runtime limpo, basta executar todas as células. O notebook de entrega usa
os artefatos versionados e não exige GPU; a T4 foi necessária apenas na etapa de
treinamento que gerou os resultados apresentados.

## Estrutura

```text
src/dogs/
  config.py       caminhos, configurações e ambiente
  data.py         dataset, split determinístico e transformações
  eda.py          figuras da análise exploratória
  features.py     embeddings da ResNet50
  models.py       arquiteturas dos três experimentos
  train.py        treinamento e checkpoints
  evaluate.py     métricas, predições e análise de erros
  viz.py          visualizações

notebooks/
  99_final.ipynb  notebook de entrega

reports/
  results.csv     métricas consolidadas
  predictions/    logits e rótulos versionados
  figures/        EDA, comparação e análise de erros

docs/
  plan.md         decisões e planejamento do projeto
```
