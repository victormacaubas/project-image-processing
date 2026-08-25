# Roteiro

**Hoje:** terça, 25/08 · **Entrega:** sábado, 29/08 · Restam **4 dias**.

Repositório: <https://github.com/victormacaubas/project-image-processing>

🔵 Victor · 🟢 Mari · 🔴 bloqueia o outro

Cada passo tem um **Pronto quando** — o critério objetivo. Marque o checkbox só
quando ele for atendido.

| Onde | O quê |
|---|---|
| [`plan.md`](plan.md) | as decisões e o porquê de cada uma |
| [`pytorch-basico.md`](pytorch-basico.md) | PyTorch em 20 min — **leia antes dos Passos 5-7** |
| docstrings em `src/dogs/` | como cada função funciona |

---

## Visão geral

| Dia | Victor 🔵 | Mari 🟢 |
|---|---|---|
| ~~Seg~~ | ~~repo + módulos~~ **feito** | — |
| **Ter** | 1 🔴 embeddings · 8 fine-tuning | 2 ambiente · 4 EDA |
| **Qua** | 8 terminar · 9 análise de erros | 5 E1 · 6 E2 |
| **Qui** | 10 montar notebook | 7 texto (1-3) |
| **Sex** | 11 testar entrega | 7 texto (4-7) |
| **Sáb** | 12 revisão e entrega | 12 |

**O gargalo é o Passo 1.** Enquanto os embeddings não subirem, a Mari não faz o
E2 e depende de GPU pra tudo.

### Já implementado

`data.py`, `features.py`, `train.py` e `evaluate.py` estão prontos no repositório.
Só `models.py` está em aberto: `SmallCNN` e `LinearProbe` são da Mari,
`build_finetune_model` é do Victor.

> Esses módulos foram escritos sem que fosse possível executar PyTorch, então a
> integração real ainda não foi testada. O Passo 1 é o primeiro contato com
> dados de verdade.

---

# Terça

## Passo 1 · Gerar os embeddings 🔴🔵 · ~25 min

Extrair os vetores de 2048 dimensões que a ResNet50 produz para cada imagem.
Depois disso, a Mari treina classificadores em segundos, na CPU, sem depender
de você nem de GPU.

**No Colab:**

1. `Ambiente de execução` → `Alterar o tipo` → **T4 GPU**
2. Montar o Drive: `drive.mount('/content/drive')`
   *Não é opcional: `/content` é efêmero e o dataset tem 776 MB. Com o Drive
   montado, o download acontece uma vez só.*
3. `!git clone <repo>` → `%cd project-image-processing` →
   `!pip install -q -r requirements-colab.txt`
4. `sys.path.insert(0, 'src')` e conferir com `describe_environment()` —
   precisa mostrar GPU e Drive montado
5. **Validar o `data.py`:** `load_data()` e puxar o primeiro lote.
   Esperado: `[64, 3, 224, 224]`, `[64]`, 120 classes, nomes como
   `n02085620-Chihuahua`. O download acontece aqui (~5-10 min).
6. `!python -m dogs.features --backbone resnet50` (~8 min)
7. Copiar `data/processed/features/*.npy` para
   `/content/drive/MyDrive/project-image-processing/features/`
8. Avisar a Mari

**Pronto quando:** os seis `.npy` estão no Drive e a Mari confirmou que carregou.

---

## Passo 2 · Ambiente 🟢 · 45 min

- [ ] Clonar o repo e abrir no Colab
- [ ] `pip install -r requirements-colab.txt`
- [ ] `pre-commit install` — **não pule.** Sem isso, dois commits no mesmo
      notebook viram conflito impossível de resolver
- [ ] Ler [`pytorch-basico.md`](pytorch-basico.md) inteiro (~20 min)

**Pronto quando:** `from dogs.config import PROJECT_ROOT` funciona e você leu o
guia de PyTorch.

---

## Passo 3 · Fine-tuning 🔵 · 1h de código + 2h de treino

`notebooks/03_finetune.ipynb`. **Comece hoje** — são 2h que rodam sozinhas em
background. Empurrar para quarta come a folga da sexta.

- [ ] Implementar `build_finetune_model()` em `models.py`
- [ ] Treinar com `unfreeze_last_n_blocks=2`, `learning_rate=1e-4`
- [ ] `log_result()` **e** `save_predictions()` — commite o `.npz`

**Pronto quando:** E3 está no `results.csv` e o `.npz` foi commitado.

---

## Passo 4 · EDA 🟢 · 3h

`notebooks/01_eda.ipynb`. Duas coisas saem daqui e voltam no relatório: saber se
o dataset é balanceado (decide se F1 macro importa) e uma figura de raças
parecidas, que na quinta será comparada com os erros reais do modelo.

- [ ] **Distribuição por classe** — `Counter(labels)`, barras ordenadas.
      Reportar min, max e mediana
- [ ] **Grid de amostras** de 8-10 raças, 4 imagens cada
- [ ] **Resoluções** — amostrar ~500 imagens, scatter largura × altura
- [ ] **Pares de raças parecidas** ← a figura mais importante. Escolha 3-4
      pares (terriers entre si, Husky vs. Malamute) e monte lado a lado
- [ ] Salvar tudo em `reports/figures/` com `dpi=150`

**Atalho importante:** para contar classes, **não itere sobre o DataLoader** —
ele aplica transforms em cada imagem e leva minutos à toa. Vá direto ao dataset
do HuggingFace, onde a coluna `label` sai como lista de inteiros na hora.
*DataLoader para ver imagens, dataset cru para contar.*

**Armadilha:** imagem vinda do DataLoader já passou por `Normalize` e aparece
com cores lavadas no `imshow`. Para visualizar, use a imagem crua do
HuggingFace.

**Pronto quando:** as figuras estão salvas e você consegue responder sem olhar o
código: *o dataset é balanceado?* e *quais pares eu apostaria que o modelo vai
confundir?*

---

# Quarta

## Passo 5 · E1, CNN do zero 🟢 · 3h

`notebooks/02_baseline.ipynb`. A tarefa mais didática do trabalho: construir uma
rede do zero e ver na prática por que ela não dá conta.

Quatro blocos iguais — **Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)** — dobrando
os canais e cortando a resolução pela metade a cada bloco:

| Bloco | Canais | Resolução |
|---|---|---|
| entrada | 3 | 224² |
| 1-4 | 32 → 64 → 128 → 256 | 112² → 56² → 28² → 14² |
| pool final | 256 | 1×1 |

Depois: `Flatten` → `Dropout` → `Linear(256, 120)`.
A docstring de `SmallCNN` tem os detalhes e o porquê de cada escolha.

- [ ] Implementar `SmallCNN`
- [ ] **Teste de sanidade:** passar um tensor `(2, 3, 224, 224)` e conferir
      saída `(2, 120)` e ~1 milhão de parâmetros
- [ ] **Rodar 1 epoch antes dos 15.** São 40 min em jogo; um erro de shape
      aparece em 3 min em vez de 40
- [ ] Treinar 15 epochs, `log_result()` + `save_predictions()`

| Acurácia top-1 | Leitura |
|---|---|
| 15-25% | esperado — é o resultado correto |
| < 5% | quebrado (acaso puro é 0,8%) |
| > 40% | desconfie: peso pré-treinado entrou sem querer |

**Acurácia baixa aqui não é fracasso, é o achado.** Ela é o piso contra o qual
E2 e E3 vão brilhar.

**Se a loss travar em ~4,8:** `ln(120) ≈ 4,79` é a loss de quem chuta. O
gradiente não está fluindo — ver a seção 4 do [guia de PyTorch](pytorch-basico.md).

**Pronto quando:** `E1_scratch` está no `results.csv` e você explica em duas
frases por que a acurácia ficou baixa.

---

## Passo 6 · E2, linear probe 🟢 · 1h30

Treina em **segundos**, na CPU. Dá para iterar à vontade.

O Victor já passou as 20 mil imagens por uma ResNet50 pré-treinada e salvou o
que sai da penúltima camada: um vetor de 2048 números por imagem. A pergunta
aqui: **essa descrição, feita por uma rede que nunca viu o Stanford Dogs, já
separa as raças?** Para responder, treinamos o classificador mais simples
possível em cima dela.

- [ ] Carregar os embeddings com `load_features()` de `dogs.features`
- [ ] Montar os loaders com `TensorDataset` (dados já em memória, dispensa
      escrever um `Dataset`). `batch_size=256` é confortável
- [ ] Implementar `LinearProbe` — só `BatchNorm1d` + `Linear`
- [ ] Treinar com `num_epochs=30`, `learning_rate=1e-3`
- [ ] **Ablação:** rodar sem o `BatchNorm1d`, registrar como
      `E2_linear_probe_sem_bn`. Custa 2 min e vira um parágrafo
- [ ] Comparar com E1 e escrever 3-4 frases

Espere um salto grande sobre E1 — **esse contraste é o coração do trabalho.** Se
a diferença for pequena, algo está errado.

**Armadilhas:** acurácia perto de 100% significa que treino e validação se
misturaram; `BatchNorm1d` quebra com batch de tamanho 1 (use `drop_last=True`).

**Pronto quando:** E2 e a ablação estão no CSV, e você explica por que uma única
camada linear ganhou de uma CNN inteira.

---

## Passo 7 · Análise de erros 🔵 · 3h

`notebooks/04_analysis.ipynb`. É o que separa "rodamos uns modelos" de um
trabalho de verdade.

- [ ] Top-20 de pares confundidos (`most_confused_pairs`)
- [ ] Grid das imagens desses pares — comparar com a figura do EDA
- [ ] Recorte da matriz de confusão para um grupo parecido
- [ ] As 10 imagens com maior perda (`hardest_examples`) — olhar uma a uma
- [ ] Gráfico de barras comparando top-1 dos experimentos

**Pronto quando:** você aponta um par confundido e explica por quê.

### 🚩 Quarta, 22h — ponto de decisão

Olhar o `results.csv`. **E1, E2 e E3 estão lá?** Se sim, pode haver um bônus na
quinta (E5 ou E6, um só). Se falta algum, corta todo o bônus.

Com um dia a menos, a régua é dura: **se faltar E3, entregue com E1 + E2 +
análise de erros.** Continua cobrindo os cinco itens obrigatórios.

---

# Quinta

## Passo 8 · Texto, seções 1-3 🟢 · 3h

Direto no `notebooks/99_final.ipynb`, nas células marcadas
`<!-- OBRIGATÓRIO -->`.

- [ ] Descrição do problema
- [ ] Descrição da base de dados
- [ ] Metodologia

Escreva para alguém que conhece ML mas não conhece este trabalho. Sem enrolação
— o professor vai ler vários.

---

## Passo 9 · Montar o notebook final 🔵 · 3h

- [ ] Trazer resultados e figuras dos notebooks de trabalho
- [ ] Tabela consolidada a partir do `results.csv`
- [ ] **Avaliar no split de teste — a única vez.** Só o melhor modelo
- [ ] Tornar a pasta do Drive pública (Compartilhar → Qualquer pessoa com o
      link → Leitor)

**Pronto quando:** o Passo 11 passa.

---

# Sexta

## Passo 10 · Texto, seções 4-7 🟢 · 3h

- [ ] Relato dos experimentos
- [ ] Resultados (comentar a tabela, não só colar)
- [ ] Análise — **incluindo a contaminação Stanford Dogs / ImageNet**
- [ ] Conclusões, respondendo à pergunta da seção 1

A seção sobre contaminação é o diferencial. O dataset saiu do ImageNet, então o
backbone pré-treinado já viu essas imagens; os números de transfer learning são
otimistas. Quase ninguém menciona isso.

---

## Passo 11 · Teste da entrega 🔵 · 45 min

Reproduzir exatamente o que a professora vai fazer. Testar na sua máquina não
prova nada — você tem o repo, o Drive e as libs.

- [ ] Abrir uma **janela anônima** (conta diferente, sem seu Drive)
- [ ] Subir o `.ipynb` num Colab novo, **sem** montar Drive nem clonar nada
- [ ] `Ambiente de execução → Reiniciar e executar tudo`
- [ ] Cronometrar: deve levar ~2 min. Se passar de 15, algo está retreinando

Checar: o `git clone` funcionou (**repo está público?**), a célula de
verificação imprimiu "Ambiente OK", todas as figuras renderizaram, os números
batem com o `results.csv`, nenhum traceback.

Se falhar, corrija, dê push e **repita o teste inteiro.** Teste parcial não vale.

- [ ] Exportar o PDF a partir desta execução

---

# Sábado · entrega

- [ ] Última execução limpa
- [ ] Commit e push · confirmar que o repositório está público
- [ ] Subir no Classroom

| Arquivo | Por quê |
|---|---|
| `99_final.ipynb` | a entrega |
| `99_final.pdf` | o Classroom não renderiza `.ipynb` |
| Link do repo (comentário) | dá acesso ao `src/` e ao histórico |

> A descrição diz que a entrega pode ser "exclusivamente via notebook", então o
> notebook basta. Suba os três mesmo assim — cobre qualquer forma de correção.

---

## Se algo der errado

| Problema | O que fazer |
|---|---|
| Colab desconecta no treino | `batch_size=32`, `image_size=160`. O checkpoint por epoch já está previsto |
| Fine-tuning pior que o linear probe | Learning rate alto. Tentar 1e-5 no backbone, 1e-3 na cabeça |
| Erro que você não entende | Última linha do traceback primeiro. É shape? Imprima `.shape` antes e depois |
| Travada há 30 min | Chame o Victor. Com 4 dias, ficar travada calada é o maior risco |
| Atrasou dois dias | Corta E3. E1 + E2 + análise ainda é um trabalho completo |
