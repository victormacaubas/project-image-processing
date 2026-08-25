# Classificação Fine-Grained de Raças de Cães

Trabalho final de Visão Computacional — pós-graduação em LLM e IA Generativa.

**Dataset:** Stanford Dogs (120 raças, 20.580 imagens)
**Entrega:** sábado, 29/08/2026
**Notebook final:** `notebooks/99_final.ipynb`

- **[`docs/plan.md`](docs/plan.md)** — as decisões: problema, dataset, escopo, divisão
- **[`docs/roteiro.md`](docs/roteiro.md)** — os passos, dia a dia. É por aqui que se trabalha.

> Os módulos em `src/dogs/` são **esqueletos**: assinaturas, docstrings com o que
> implementar e as armadilhas conhecidas. Só `config.py` vem preenchido, por ser
> configuração. O resto é trabalho de vocês.

## Setup

### Colab (recomendado)

Abra `notebooks/99_final.ipynb` e rode a primeira célula. Ela clona o repositório
e instala as dependências sozinha — nada de montar Drive ou ajustar caminho.

Montar o Drive é **opcional**, e só serve para não perder checkpoints quando o
Colab desconectar:

```python
from google.colab import drive
drive.mount('/content/drive')   # opcional — só como cache de artefatos
```

> O código nunca *depende* do Drive. Quem clonar o repo numa conta qualquer
> consegue rodar tudo. Ver `drive_mounted()` em `config.py`.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install     # obrigatório: evita conflito em notebook
```

## Fluxo

```bash
# 1. Gerar embeddings — roda UMA vez, destrava todo o resto
python -m dogs.features --backbone resnet50

# 2. Experimentos: ver notebooks/
```

## Estrutura

```
src/dogs/
  config.py     # caminhos e hiperparâmetros — nada hardcoded fora daqui
  data.py       # dataset, splits, transforms
  features.py   # extração de embeddings (rodar primeiro)
  models.py     # SmallCNN (E1), LinearProbe (E2), fine-tuning (E3)
  train.py      # loop de treino único, compartilhado
  evaluate.py   # métricas, results.csv, análise de erros

notebooks/
  01_eda.ipynb        # Mari
  02_baseline.ipynb   # Mari — E1 e E2
  03_finetune.ipynb   # Victor — E3
  04_analysis.ipynb   # Victor — E4
  99_final.ipynb      # entrega — montado na sexta

reports/results.csv   # toda métrica de todo experimento
```

## Experimentos

| ID | O quê | Dono | Status |
|---|---|---|---|
| E1 | CNN do zero | Mari | ⬜ |
| E2 | Linear probe (backbone congelado) | Mari | ⬜ |
| E3 | Fine-tuning parcial | Victor | ⬜ |
| E4 | Análise de erros | Victor | ⬜ |
| E5 | Segundo backbone *(bônus)* | — | ⬜ |
| E6 | CLIP zero-shot *(bônus)* | — | ⬜ |

## Regras

- Notebook nunca é commitado com output (`pre-commit install` cuida disso)
- Lógica vai em `src/`, notebook só importa e chama
- Todo resultado é registrado via `log_result()` em `reports/results.csv`
- O split de teste é avaliado **uma vez**, no fim. Até lá, só validação.
