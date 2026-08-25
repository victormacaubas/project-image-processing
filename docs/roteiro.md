# Roteiro — do zero à entrega

**Hoje:** segunda, 24/08. **Entrega:** sábado, 29/08.

Cada passo tem dono, tempo estimado e um critério de pronto verificável. Marque o
checkbox quando o critério for atendido — não antes.

As decisões de escopo (por que Stanford Dogs, quais experimentos, quais métricas)
estão em [`plan.md`](plan.md). Este documento é só a execução.

**Legenda:** 🔵 Victor · 🟢 Mari · 🔴 bloqueia o outro, faça primeiro

## Visão geral

| Dia | Victor 🔵 | Mari 🟢 |
|---|---|---|
| **Seg** | Passos 1, 2, 3 — repo, ambiente, `data.py` 🔴 | Passo 2 — ambiente |
| **Ter** | Passos 4, 5 — embeddings 🔴, `train.py` | Passo 6 — EDA |
| **Qua** | Passo 9 — fine-tuning (E3) | Passos 7, 8 — E1 e E2 |
| **Qui** | Passo 10 — análise de erros (E4) | Passo 11 — texto, seções 1-3 |
| **Sex** | Passos 13, 13-B — notebook final e teste | Passo 14 — texto, seções 4-7 |
| **Sáb** | Passo 15 — revisão cruzada e entrega | Passo 15 |

Os dois passos 🔴 são os que bloqueiam a Mari. Enquanto `data.py` (Passo 3) e
os embeddings (Passo 4) não existirem, ela só consegue avançar no Passo 2 e na
leitura do Apêndice A.

Se a Mari nunca usou PyTorch, o **Apêndice A** é pré-requisito dos Passos 6-8.

---

## Segunda 24/08 — fundação

### Passo 1 · Combinar com a Mari 🔵🟢 · 30 min

- [x] Criar o repo no GitHub
- [ ] Dar acesso de escrita à Mari (Settings → Collaborators)
- [ ] Criar a pasta compartilhada no Drive (onde os embeddings vão morar)
- [ ] Mandar a mensagem abaixo para ela
- [ ] Marcar o call diário de 15 min (sugestão: 19h)

#### Mensagem para a Mari — copiar e colar

> Fechei a estrutura do nosso trabalho de Visão Computacional. Prazo: **sábado**,
> então o escopo está enxuto de propósito.
>
> **O que vamos fazer:** classificar 120 raças de cães (Stanford Dogs). Três
> experimentos em progressão — CNN treinada do zero, depois transfer learning com
> backbone congelado, depois fine-tuning — e uma análise de erros no fim.
>
> **Repo:** `<COLE A URL AQUI>`
>
> Comece por dois documentos:
>
> - `docs/plan.md` — as decisões e o porquê de cada uma (leitura de 10 min)
> - `docs/roteiro.md` — os passos dia a dia. **Suas tarefas são as marcadas 🟢**:
>   Passos 2, 6, 7, 8, 11 e 14.
>
> **Antes de qualquer código**, leia o **Apêndice A** no fim do roteiro. São 20
> minutos de PyTorch essencial: tensor, Dataset/DataLoader, nn.Module, o loop de
> treino e os erros clássicos. Escrevi pensando em quem programa em Python mas
> nunca mexeu em PyTorch. Vai economizar horas.
>
> **Hoje (seg):** só o Passo 2 — deixar o ambiente rodando (~45 min). O
> importante é o `pre-commit install`; sem ele, nossos commits em notebook viram
> conflito impossível.
>
> **Amanhã (ter):** Passo 6, a análise exploratória. Independente do meu trabalho,
> pode tocar sozinha.
>
> Cada tarefa no roteiro tem "Pronto quando" — é o critério objetivo, não precisa
> adivinhar se terminou. E se travar mais de 30 minutos em qualquer coisa, me
> chama. Com cinco dias, ficar travado calado é o maior risco que a gente tem.
>
> Call rápido hoje à noite pra alinhar?

**Pronto quando:** ela confirmou acesso ao repo e leu o Apêndice A.

---

### Passo 2 · Ambiente 🔵🟢 · 45 min

Cada um no seu.

- [ ] Clonar o repo
- [ ] `pip install -r requirements.txt` (local; no Colab é `requirements-colab.txt`)
- [ ] `pip install -e .` (permite `from dogs.config import ...` de qualquer lugar)
- [ ] `pre-commit install` ← **não pule.** Sem isso, dois commits no mesmo
      notebook viram conflito irresolvível, e vocês não têm tempo pra isso.
- [ ] Testar no Colab: montar Drive, `pip install -r requirements.txt`,
      `sys.path.insert(0, 'src')`, confirmar `torch.cuda.is_available() == True`

**Pronto quando:** `python -c "from dogs.config import PROJECT_ROOT; print(PROJECT_ROOT)"`
funciona nos dois ambientes.

> `src/dogs/config.py` já vem preenchido — é configuração, não tem valor pedagógico.
> Ajustem os caminhos se quiserem outra organização.

---

### Passo 3 · `data.py` 🔴🔵 · 2h

**Bloqueia todo o resto. É a prioridade da noite.**

Implementar, nesta ordem:

- [ ] `build_transforms()` — treino com augmentation, eval determinístico
- [ ] `HFImageDataset` — `__len__` e `__getitem__`
- [ ] `load_data()` — download, split semeado, três DataLoaders

Armadilhas conhecidas:

- Algumas imagens são grayscale. Sem `.convert("RGB")` o batch quebra com erro de
  shape, e a mensagem não é óbvia.
- Split de validação sai do **treino**. O teste não se toca até sexta.
- Semear o gerador do `randperm`. Se o split mudar entre execuções, os
  experimentos deixam de ser comparáveis e vocês só vão descobrir isso tarde.

**Pronto quando:** chamar `load_data()` com uma `TrainConfig` qualquer e puxar o
primeiro lote do `train_loader` devolve imagens de shape `(64, 3, 224, 224)`,
labels de shape `(64,)` e `num_classes == 120`.

---

## Terça 25/08 — embeddings e EDA

### Passo 4 · `features.py` 🔴🔵 · 2h + 8 min de execução

**A tarefa mais importante da semana.** Enquanto os embeddings não existirem, a
Mari está bloqueada em E2 e depende de GPU pra tudo.

- [ ] `get_device()`, `build_backbone()`, `extract()`, `main()`
- [ ] Rodar: `python -m dogs.features --backbone resnet50`
- [ ] Conferir os shapes: train ≈ (10200, 2048), val ≈ (1800, 2048), test ≈ (8580, 2048)
- [ ] **Subir `data/processed/features/` no Drive compartilhado**
- [ ] Avisar a Mari no mesmo momento

Armadilha: rodar com `use_augmentation=True`. O embedding de cada imagem mudaria a
cada execução e o cache perderia a razão de existir.

**Pronto quando:** os seis `.npy` estão no Drive e a Mari confirmou que carregou.

---

### Passo 5 · `train.py` + `evaluate.py` 🔵 · 2h30

A Mari precisa disso pra rodar E1. Terminar hoje.

- [ ] `Metrics`, `predict()`, `metrics_from_logits()`, `evaluate()`
- [ ] `log_result()` — o CSV que vai virar a tabela do relatório
- [ ] `train_model()` — loop com early stopping

Armadilha: passar todos os parâmetros para o AdamW quando parte está congelada.
Filtrar por `requires_grad`. O bug é silencioso — treina, só não do jeito que você acha.

**Pronto quando:** `train_model()` roda 2 epochs de um modelo qualquer sem erro e
grava linha em `reports/results.csv`.

---

### Passo 6 · EDA 🟢 · 3h

`notebooks/01_eda.ipynb`. Independente de tudo — pode começar assim que o Passo 3 fechar.

> **Antes de começar:** leia o [Apêndice A](#apêndice-a--pytorch-essencial), seções
> A.1 e A.2. São 15 minutos e evitam travar no primeiro `for` sobre o DataLoader.

#### Por que este passo existe

Não é enfeite. Duas coisas concretas saem daqui e voltam no relatório:

1. **Saber se o dataset é balanceado** decide se F1 macro importa ou não.
2. **A figura de raças parecidas** vai ser comparada, na quinta, com os erros
   reais do modelo. Se o modelo confundir os mesmos pares que você separou aqui,
   isso vira um parágrafo forte na análise.

#### Carregando os dados

No topo do notebook, acrescente `../src` ao `sys.path` para conseguir importar o
pacote `dogs`. A partir daí, `load_data()` de `dogs.data` devolve os loaders, e
`dogs.config` tem as constantes (`FIGURES_DIR`, `HF_DATASET`, `RAW_DIR`).

Para estatísticas de classe, porém, **não itere sobre o DataLoader** — ele aplica
os transforms em cada imagem e leva minutos à toa. Vá direto ao dataset do
HuggingFace (`load_dataset` com `cache_dir=RAW_DIR`): a coluna `label` sai como
lista de inteiros na hora, e `features["label"].names` dá os nomes das raças.

Regra geral do passo: **DataLoader para ver imagens, dataset cru para contar.**

#### Tarefas

- [ ] **Distribuição por classe.** `collections.Counter(labels)`, barras
      ordenadas. Reportar min, max e mediana de imagens por classe.
      → *Pergunta a responder: a classe menor tem menos da metade da maior?*
- [ ] **Grid de amostras** de 8-10 raças, 4 imagens cada.
- [ ] **Resoluções.** Amostrar ~500 imagens (não as 20 mil), coletar
      `img.size`, e plotar largura × altura em scatter.
      → *Pergunta: 224×224 corta muita coisa?*
- [ ] **Pares de raças parecidas** — a figura mais importante. Escolha 3-4 pares
      (sugestões: os vários terriers, Husky vs. Malamute, Whippet vs. Italian
      Greyhound) e monte um grid lado a lado, com nome embaixo.
- [ ] Salvar tudo em `FIGURES_DIR` com `dpi=150` e `bbox_inches="tight"`.

#### Armadilhas

- **Imagem normalizada fica com cor esquisita ao plotar.** Se você pegar do
  DataLoader, o tensor já passou por `Normalize` e o `imshow` mostra cores
  lavadas. Para visualizar, use a imagem crua do HuggingFace (`raw[i]["image"]`),
  não o tensor.
- Tensor do PyTorch é `(C, H, W)`; o matplotlib quer `(H, W, C)`. Se precisar
  plotar um tensor, `img.permute(1, 2, 0)`.

#### Pronto quando

As figuras estão em `reports/figures/` e você consegue responder, sem olhar o
código: *o dataset é balanceado?* e *quais pares de raças eu apostaria que o
modelo vai confundir?*

---

## Quarta 26/08 — os modelos

### Passo 7 · E1, CNN do zero 🟢 · 3h

`notebooks/02_baseline.ipynb`. É a tarefa mais didática do trabalho: construir uma
rede do zero e ver, na prática, por que ela não dá conta.

> **Antes de começar:** [Apêndice A](#apêndice-a--pytorch-essencial), seções A.3 e A.4.

#### O que você vai construir

Uma CNN pequena, empilhando quatro blocos iguais. Cada bloco faz, nesta ordem:
**Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)**.

- **Conv2d** — o filtro que detecta padrões (bordas, texturas). `kernel_size=3,
  padding=1` mantém altura e largura; quem reduz é o pooling.
- **BatchNorm2d** — normaliza as ativações. Sem ele, redes profundas treinam mal
  ou nem treinam.
- **ReLU** — não-linearidade. Sem ela, empilhar camadas é inútil: a composição de
  funções lineares continua linear.
- **MaxPool2d(2)** — corta altura e largura pela metade. É o que dá "visão mais
  ampla" às camadas seguintes.

A cada bloco você **dobra os canais e reduz a resolução pela metade**. É o padrão
de praticamente toda CNN clássica:

| Bloco | Canais | Resolução |
|---|---|---|
| entrada | 3 | 224×224 |
| 1 | 32 | 112×112 |
| 2 | 64 | 56×56 |
| 3 | 128 | 28×28 |
| 4 | 256 | 14×14 |
| pool final | 256 | 1×1 |

#### Como montar

A classe `SmallCNN` está em `src/dogs/models.py`, com a docstring detalhando o
que implementar. A estrutura:

- Uma função auxiliar `block(in_ch, out_ch)` que devolve um `nn.Sequential` com
  as quatro camadas acima. Escrever uma vez e chamar quatro vezes é bem melhor
  que repetir o bloco.
- `self.features` — os quatro blocos em sequência, terminando com
  `AdaptiveAvgPool2d(1)`, que reduz `(B, 256, 14, 14)` para `(B, 256, 1, 1)`.
- `self.classifier` — `Flatten` (vira `(B, 256)`), `Dropout`, e `Linear` para
  120 saídas.
- `forward` — uma linha: passa por `features`, depois por `classifier`.

Duas escolhas que valem citar no relatório:

- **`bias=False` na Conv** quando vem BatchNorm logo depois. O parâmetro `beta`
  do BN já cumpre o papel do bias, então o bias da Conv seria redundante.
- **`AdaptiveAvgPool2d(1)` em vez de achatar direto.** Achatar 256×14×14 daria
  50.176 features e uma camada final com ~6 milhões de parâmetros, que overfita
  na hora. Com o pooling, são 256.

#### Treinar

Monte uma `TrainConfig` com `experiment_name="E1_scratch"`, carregue os dados com
`load_data()`, instancie a `SmallCNN` e passe tudo para `train_model()`. No fim,
registre com `log_result()` — se não registrar, o resultado não entra na tabela
do relatório.

**Antes de rodar os 15 epochs, rode 1.** São 40 minutos em jogo; um erro de shape
aparece em 3 minutos em vez de 40.

#### Checklist

- [ ] `SmallCNN` implementada
- [ ] Teste de sanidade passou (ver abaixo)
- [ ] 1 epoch sem erro
- [ ] 15 epochs completos
- [ ] `log_result` gravou a linha no CSV

**Teste de sanidade**, antes de qualquer treino: instancie o modelo, passe um
tensor aleatório de shape `(2, 3, 224, 224)` e confirme que a saída tem shape
`(2, 120)`. Aproveite para somar `p.numel()` de todos os parâmetros — a ordem de
grandeza esperada é ~1 milhão. Se der 50 milhões, o pooling final ficou de fora.

#### Interpretando o resultado

| Acurácia top-1 | Leitura |
|---|---|
| 15-25% | Esperado. É o resultado correto. |
| < 5% | Algo quebrado. ~0,8% é o acaso puro (1/120). Ver armadilhas. |
| > 40% | Desconfie. Provavelmente entrou peso pré-treinado sem querer. |

**Acurácia baixa aqui não é fracasso, é o achado.** Ela é o piso contra o qual E2
e E3 vão brilhar. Sem esse número, o trabalho não tem com o que comparar.

#### Armadilhas

- **Loss não desce (fica em ~4,8).** `ln(120) ≈ 4,79` é a loss de um modelo que
  chuta uniformemente. Se travou aí, o gradiente não está fluindo: confira se
  esqueceu o `optimizer.zero_grad()` ou se o learning rate está absurdo.
- **CUDA out of memory.** Baixe `batch_size` para 32.
- **Erro de shape no `Linear`.** O número que entra no `Linear` tem que bater com
  os canais do último bloco. Se mudou a arquitetura, mude o `256` também.

#### Pronto quando

A linha `E1_scratch` está no `results.csv` e você consegue explicar em duas frases
por que a acurácia ficou baixa.

---

### Passo 8 · E2, linear probe 🟢 · 1h30

Mesmo notebook. Treina em **segundos**, na CPU — dá pra iterar à vontade.

#### A ideia

O Victor já passou as 20 mil imagens por uma ResNet50 pré-treinada em ImageNet e
salvou o que sai da penúltima camada: um vetor de 2048 números por imagem. Esse
vetor é a "descrição" que a rede faz da imagem.

A pergunta deste experimento: **essa descrição, feita por uma rede que nunca viu
o Stanford Dogs, já separa as raças?** Para responder, treinamos o classificador
mais simples possível em cima dela — uma única camada linear.

Se funcionar bem, a conclusão é forte: o trabalho pesado já estava feito pelo
pré-treino, e bastava aprender a fatiar o espaço de features.

Como as imagens já viraram vetores, não há convolução, não há GPU, não há
DataLoader de imagem. Só álgebra linear.

#### Carregando os embeddings

Os arquivos estão em `FEATURES_DIR`, no padrão `resnet50_{split}_X.npy` e
`resnet50_{split}_y.npy`. Carregue com `np.load` e converta para tensor:
`X` em `float`, `y` em `long` — a `CrossEntropyLoss` exige inteiro nos rótulos.

`X_train` deve sair com shape aproximado `(10200, 2048)`.

Para montar os loaders, use **`TensorDataset`**: é o atalho de quando os dados já
estão em memória como tensor, e dispensa escrever uma classe `Dataset`. Como não
há imagem para decodificar, dá para usar `batch_size` bem maior — 256 é
confortável.

#### Implementar

`LinearProbe` em `models.py` é bem mais simples que a SmallCNN: um
`nn.Sequential` com duas camadas apenas — `BatchNorm1d(input_dim)` seguido de
`Linear(input_dim, num_classes)`.

É `BatchNorm1d`, não `2d`: aqui cada amostra é um vetor, não uma imagem. E ele
está aí porque a escala das 2048 dimensões varia bastante entre si, o que
atrapalha o otimizador.

#### Treinar

Mesmo `train_model()` do E1 — é essa reutilização que torna a comparação honesta.
Como o treino é rápido, use mais epochs (30) e learning rate maior (1e-3) do que
no E1. O `input_dim` sai de `X_train.shape[1]`.

#### Checklist

- [ ] `LinearProbe` implementada
- [ ] Treinada e registrada no CSV
- [ ] **Ablação:** rodar sem o `BatchNorm1d`, registrar como
      `E2_linear_probe_sem_bn`. Custa 2 minutos e vira um parágrafo do relatório.
- [ ] Comparar com E1 e escrever 3-4 frases sobre a diferença

#### O que esperar

Bem acima de E1. Se a diferença for pequena, algo está errado — provavelmente os
embeddings foram gerados com augmentation ligado, ou `X` e `y` estão
desalinhados.

Guarde a comparação: **este contraste é o coração do trabalho.**

#### Armadilhas

- **`BatchNorm1d` quebra com batch de tamanho 1.** Se o último batch tiver uma
  amostra só, dá erro. Use `drop_last=True` no DataLoader de treino.
- **Acurácia perto de 100% na validação.** Bom demais para ser verdade: os
  embeddings de treino e validação provavelmente se misturaram.
- **`X` e `y` desalinhados.** Só acontece se os `.npy` foram gerados em execuções
  diferentes. Confira que `len(X) == len(y)` nos três splits.

#### Pronto quando

`E2_linear_probe` está no CSV, a ablação também, e você consegue explicar por que
uma única camada linear ganhou de uma CNN inteira treinada do zero.

---

### Passo 9 · E3, fine-tuning 🔵 · 1h de código + 2h de treino

`notebooks/03_finetune.ipynb`. Começar cedo — o treino é longo e não dá pra apressar.

- [ ] Implementar `build_finetune_model()`
- [ ] Treinar com `unfreeze_last_n_blocks=2`, lr menor (1e-4)
- [ ] `log_result("E3_finetune_last2", "val", metrics)`
- [ ] Se sobrar tempo de GPU: repetir com `unfreeze_last_n_blocks=1` para comparar

Colab desconecta. Salvar checkpoint a cada melhora (o `train_model` já faz) e
manter a aba viva.

**Pronto quando:** E3 está no CSV e o checkpoint está salvo no Drive.

---

### 🚩 Quarta, 22h — ponto de decisão

Olhar o `results.csv`:

- **E1, E2 e E3 estão lá?** → quinta pode ter bônus (E5 ou E6, um só)
- **Falta algum?** → corta todo o bônus. Quinta é para fechar o obrigatório.

Decidam isso explicitamente no call. Um trabalho completo e modesto vale mais que
um ambicioso pela metade — e é sexta que essa diferença aparece.

---

## Quinta 27/08 — análise e texto

### Passo 10 · E4, análise de erros 🔵 · 3h

`notebooks/04_analysis.ipynb`. É o que separa "rodamos uns modelos" de um trabalho
de verdade. Não trate como sobra de tempo.

- [ ] `most_confused_pairs()` em `evaluate.py`
- [ ] Top-20 de pares confundidos do melhor modelo
- [ ] **Grid de imagens dos pares mais confundidos** — comparar com a figura do EDA
- [ ] Recorte da matriz de confusão para um grupo parecido (ex.: só os terriers)
- [ ] As 10 imagens com maior perda — olhar uma a uma. Costuma aparecer rótulo
      errado, imagem com dois cães, cão pequeno demais no quadro.
- [ ] Gráfico de barras comparando top-1 dos experimentos
- [ ] Salvar tudo em `reports/figures/`

**Pronto quando:** você consegue apontar um par confundido e explicar por quê.

---

### Passo 11 · Primeira versão do texto 🟢 · 3h

Direto no `notebooks/99_final.ipynb`, nas seções marcadas com `<!-- OBRIGATÓRIO -->`.

- [ ] Seção 1 — descrição do problema
- [ ] Seção 2 — descrição da base de dados
- [ ] Seção 3 — metodologia

Escrever para alguém que conhece ML mas não conhece este trabalho. Sem enrolação:
o professor vai ler cinco desses.

**Pronto quando:** as três seções têm texto de verdade, não bullet points soltos.

---

### Passo 12 · Bônus, só se liberado na quarta 🔵🟢 · 2h

Escolher **um**:

- **E5** — segundo backbone (ViT-B/16) sobre os embeddings. Mais barato, mais seguro.
- **E6** — CLIP zero-shot. Mais interessante e mais alinhado com a pós: um modelo
  que nunca viu um exemplo rotulado do dataset contra o de vocês, fine-tuned.

**Pronto quando:** está no CSV. Se passar do tempo, abandona sem dó.

---

## Sexta 28/08 — fechamento

### Passo 13 · Montar o notebook final 🔵 · 3h

- [ ] Trazer os resultados dos notebooks de trabalho para o `99_final.ipynb`
- [ ] Colocar os treinos longos atrás de `RETRAIN = False`, carregando checkpoint
- [ ] Tabela consolidada a partir do `results.csv`
- [ ] Inserir as figuras
- [ ] **Avaliar no split de teste — a única vez.** Só o melhor modelo.
      Registrar com `split="test"`.

Preparar a entrega autossuficiente:

- [ ] Deixar o repositório **público** no GitHub (ou dar acesso ao professor)
- [ ] Preencher `REPO_URL` na célula de bootstrap
- [ ] Subir checkpoints e embeddings numa pasta pública do Drive; colar o ID em
      `DRIVE_FOLDER_ID` e implementar o download
- [ ] Colar o link do repositório no cabeçalho do notebook

**Pronto quando:** o Passo 13-B passa.

---

### Passo 13-B · Teste da entrega 🔵 · 45 min

O objetivo é um só: **reproduzir exatamente o que a professora vai fazer.** Não é
verificar se roda pra você — isso não prova nada, sua máquina tem o repo, o Drive
montado e as libs instaladas.

#### O que já está protegido no código

Quatro armadilhas conhecidas já foram resolvidas, mas vale saber quais são para
não reintroduzi-las:

| Armadilha | Proteção |
|---|---|
| `config.py` apontando para o *seu* Drive | `PROJECT_ROOT` vem de `Path(__file__).parents[2]`. O Drive só é usado se `drive_mounted()` for verdadeiro. |
| Reinstalar torch no Colab dispara "restart runtime" e aborta o run | `requirements-colab.txt` traz só o que o Colab não tem |
| `pip -q` engolindo o erro, que reaparece 20 células depois como `ImportError` | `_run()` imprime stdout/stderr e levanta exceção na hora |
| Rodar a célula de bootstrap duas vezes e clonar aninhado | `_find_repo_root()` é idempotente |

**Nunca** escreva um caminho absoluto no notebook. Nem `/content/drive/...`, nem
`/Users/victor/...`. Todo caminho sai de `dogs.config`.

#### O teste

- [ ] Abrir uma **janela anônima** (garante conta diferente, sem seu Drive)
- [ ] Subir o `.ipynb` num Colab novo — **sem** montar Drive, **sem** clonar nada
      manualmente
- [ ] `Ambiente de execução → Reiniciar e executar tudo`
- [ ] Cronometrar. Se passar de ~15 min com `RETRAIN=False`, algo está
      retreinando sem querer.

Checar item a item:

- [ ] O `git clone` funcionou → **o repositório está público?** Privado falha
      pedindo credencial que ela não tem.
- [ ] A célula de verificação imprimiu "Ambiente OK"
- [ ] O download dos artefatos funcionou → **a pasta do Drive está pública,
      como "qualquer pessoa com o link"?** É o erro mais comum: funciona pra
      você porque é sua.
- [ ] Todas as figuras renderizaram
- [ ] Os números da tabela batem com o `results.csv`
- [ ] Nenhum traceback, em nenhuma célula

#### Se falhar

Não conserte só o sintoma — o mesmo erro costuma estar em três células. Corrija,
dê push e **repita o teste inteiro do zero.** Um teste parcial depois de um
conserto não vale.

- [ ] Passou de ponta a ponta na janela anônima
- [ ] Exportar o PDF **a partir desta execução** (`Arquivo → Imprimir → PDF`)

> Se sobrar tempo: peça para a Mari rodar o mesmo teste na máquina dela.
> Duas contas diferentes pegam coisas que uma só não pega.

---

### Passo 14 · Terminar o texto 🟢 · 3h

- [ ] Seção 4 — relato dos experimentos
- [ ] Seção 5 — resultados (comentar a tabela, não só colar)
- [ ] Seção 6 — análise, **incluindo 6.3, contaminação Stanford Dogs / ImageNet**
- [ ] Seção 7 — conclusões, respondendo à pergunta da seção 1
- [ ] Referências

A seção 6.3 é o diferencial. Stanford Dogs saiu do ImageNet, então o backbone
pré-treinado já viu essas imagens; os números de transfer learning são otimistas.
Quase ninguém menciona isso. Vocês mencionando, com honestidade sobre o que
invalida e o que não invalida, mostra maturidade que a maioria dos trabalhos não tem.

---

### Passo 15 · Revisão cruzada 🔵🟢 · 1h30

Cada um revisa o que o **outro** escreveu.

- [ ] Os cinco itens obrigatórios estão todos cobertos?
- [ ] Todo número no texto bate com o `results.csv`?
- [ ] Toda figura tem título, eixos rotulados e é citada no texto?
- [ ] Alguma afirmação sem respaldo nos experimentos?
- [ ] Nomes dos autores no topo?

---

## Sábado 29/08 — entrega

- [ ] Última execução limpa de ponta a ponta
- [ ] Commit final e push
- [ ] Confirmar que o repositório está público
- [ ] Subir no Classroom (ver abaixo)
- [ ] Comemorar

### O que subir no Google Classroom

A descrição do trabalho não especifica o formato de submissão, mas diz que a
entrega pode ser feita "exclusivamente via notebook" desde que tenha as células
de texto. Ou seja: **o notebook basta.** O repositório não é exigido.

Ainda assim, suba os três — custa nada e cobre qualquer forma de correção:

| Arquivo | Por quê |
|---|---|
| `99_final.ipynb` | A entrega em si |
| `99_final.pdf` | O Classroom não renderiza `.ipynb`. Se o professor corrigir pelo navegador, sem baixar, é o PDF que ele vê. |
| Link do repo no comentário | Dá acesso ao `src/`, ao histórico e mostra o trabalho de engenharia |

Para o PDF: `Arquivo → Imprimir → Salvar como PDF` no Colab, com o notebook já
executado. Conferir que os gráficos aparecem e que nada ficou cortado.

> **Confirme com o professor** se ele quer o repositório também. A descrição não
> menciona, mas cada disciplina tem seu costume — e é uma pergunta de trinta
> segundos que evita retrabalho no sábado.

---

## Se algo der errado

| Problema | O que fazer |
|---|---|
| Colab desconecta no meio do treino | Reduzir epochs, batch 32, `image_size=160`. Checkpoint por epoch já está previsto. |
| Download do dataset falhando | Baixar uma vez, deixar no Drive, apontar `RAW_DIR` para lá. |
| Fine-tuning pior que o linear probe | Learning rate alto demais. Tentar 1e-5 no backbone e 1e-3 na cabeça. |
| Mari travada numa tarefa | Ela passa a tarefa e assume redação. Descobrir isso na sexta é fatal — por isso o call diário. |
| Atrasou dois dias | Corta E3. E1 + E2 + análise de erros bem-feita ainda é um trabalho completo. |

---

# Apêndice A · PyTorch essencial

Para quem programa em Python mas nunca usou PyTorch. Leitura de ~20 minutos.
Não é um tutorial completo — é o mínimo para não travar nos Passos 6, 7 e 8.

## A.1 · Tensor

Um tensor é um array N-dimensional, como o do NumPy, com duas diferenças: roda na
GPU e registra as operações para calcular gradientes automaticamente.

Operações que você vai usar o tempo todo:

| Operação | O que faz |
|---|---|
| `.shape` | as dimensões do tensor |
| `.to("cuda")` / `.to(device)` | move para a GPU |
| `.cpu().numpy()` | volta para NumPy — precisa sair da GPU antes |
| `.permute(...)` | reordena as dimensões |
| `.argmax(dim=1)` | índice do maior valor por linha |

**A convenção de shape que você vai ver o tempo todo:**

| Notação | Significa |
|---|---|
| `(B, C, H, W)` | lote de imagens: Batch, Canais, Altura, Largura |
| `(B, D)` | lote de vetores: Batch, Dimensão |

Um batch de 64 fotos coloridas 224×224 tem shape `(64, 3, 224, 224)`.

> **Atenção:** PyTorch usa `(C, H, W)`; matplotlib e PIL usam `(H, W, C)`. Para
> plotar um tensor de imagem: `tensor.permute(1, 2, 0)`.

**90% dos erros de iniciante em PyTorch são erro de shape.** Quando algo quebrar,
o primeiro reflexo é imprimir `.shape` antes e depois da operação.

## A.2 · Dataset e DataLoader

**Dataset** sabe devolver *um* item. Herda de `torch.utils.data.Dataset` e
implementa dois métodos: `__len__`, que diz quantos itens existem, e
`__getitem__(i)`, que devolve o item `i` já pronto — normalmente a tupla
`(tensor_da_imagem, label_inteiro)`.

**DataLoader** recebe um Dataset e entrega *lotes*, embaralhando se você pedir
(`shuffle=True`). Iterar sobre ele num `for` devolve pares
`(imagens, labels)` — com `batch_size=64`, shapes `(64, 3, 224, 224)` e `(64,)`.

Neste projeto, `load_data()` já devolve os três loaders prontos. Você raramente
precisa criar um do zero — a exceção é o Passo 8, onde `TensorDataset` resolve.

Parâmetros do DataLoader que aparecem por aqui:

| Parâmetro | Para quê |
|---|---|
| `batch_size` | quantas amostras por lote |
| `shuffle=True` | embaralha — só no treino, nunca na avaliação |
| `num_workers` | processos paralelos carregando dados |
| `drop_last=True` | descarta o último lote incompleto; evita erro no BatchNorm |

## A.3 · Módulo

Todo modelo herda de `nn.Module`, define as camadas no `__init__` e descreve o
fluxo de dados no `forward`. Duas regras:

- **`super().__init__()` como primeira linha do `__init__`.** Esquecer isso gera
  um erro confuso sobre atributos não inicializados.
- **Chame o modelo diretamente** (`modelo(x)`), não `modelo.forward(x)`. A chamada
  direta dispara os hooks internos do PyTorch; a outra pula etapas.

`nn.Sequential` encadeia camadas quando o fluxo é uma linha reta, sem desvios —
você passa as camadas na ordem e ele cuida de aplicar uma após a outra. É o que
usamos nos dois modelos deste projeto.

**Camadas que aparecem neste projeto:**

| Camada | O que faz |
|---|---|
| `nn.Conv2d(in, out, kernel_size=3, padding=1)` | detecta padrões locais; `padding=1` preserva H e W |
| `nn.BatchNorm2d(n)` / `nn.BatchNorm1d(n)` | normaliza ativações; `2d` para imagem, `1d` para vetor |
| `nn.ReLU()` | zera os negativos — a não-linearidade |
| `nn.MaxPool2d(2)` | reduz H e W pela metade |
| `nn.AdaptiveAvgPool2d(1)` | reduz H e W a 1×1, seja qual for a entrada |
| `nn.Flatten()` | `(B, C, 1, 1)` → `(B, C)` |
| `nn.Dropout(p)` | desliga p% das ativações no treino; regularização |
| `nn.Linear(in, out)` | camada densa |

## A.4 · O loop de treino

Já está implementado em `train.py` — você não precisa escrevê-lo. Mas precisa
entender, porque é onde os erros aparecem.

A cada **epoch**, o loop faz duas passagens.

**Fase de treino.** Coloca o modelo em `modelo.train()` (ativa dropout, e o
BatchNorm passa a atualizar suas estatísticas) e itera sobre o `train_loader`.
Para cada lote, cinco operações na ordem:

| Ordem | Operação | Para quê |
|---|---|---|
| 1 | `.to(device)` nos tensores | levar dados para onde o modelo está |
| 2 | `optimizer.zero_grad()` | zerar os gradientes da iteração anterior |
| 3 | `modelo(x)` | forward — produz os logits |
| 4 | `criterion(saida, y)` e `.backward()` | mede o erro e calcula os gradientes |
| 5 | `optimizer.step()` | aplica os gradientes, atualizando os pesos |

**Fase de validação.** Coloca em `modelo.eval()` (desliga dropout, congela as
estatísticas do BatchNorm) e roda dentro de `torch.no_grad()`, que evita
construir o grafo de gradientes — mais rápido e economiza memória.

**Vocabulário:**

- **epoch** — uma passada completa por todos os dados de treino
- **batch** — um lote (aqui, 64 imagens processadas juntas)
- **loss** — número que mede o erro; treinar é minimizá-lo
- **logits** — a saída crua do modelo, antes de virar probabilidade. Shape
  `(B, 120)`. A classe prevista é `logits.argmax(dim=1)`.
- **gradiente** — a direção que reduz a loss; `backward()` calcula, `step()` aplica

**Os quatro erros clássicos:**

1. Esquecer `optimizer.zero_grad()` → gradientes acumulam e o treino enlouquece
2. Esquecer `modelo.eval()` na avaliação → dropout ativo, métrica inconsistente
3. Esquecer `.to(device)` → `Expected all tensors to be on the same device`
4. Avaliar sem `torch.no_grad()` → lento e come memória à toa

## A.5 · Como interpretar a loss

Com 120 classes, um modelo que chuta uniformemente tem loss `ln(120) ≈ 4,79`.

| Comportamento da loss | Diagnóstico |
|---|---|
| Travada em ~4,8 | Não está aprendendo nada. Ver os erros de A.4. |
| Descendo devagar | Normal nos primeiros epochs |
| Desce e a validação sobe | Overfitting — é para isso que existe o early stopping |
| `nan` | Learning rate alto demais. Divida por 10. |

## A.6 · Quando travar

Nesta ordem, antes de chamar o Victor:

1. **Leia a última linha do traceback**, não a primeira. É onde está o erro real.
2. **É erro de shape?** Imprima `.shape` antes e depois da operação que quebrou.
3. **É `device`?** Algum tensor ficou na CPU enquanto o modelo está na GPU.
4. **É OOM (out of memory)?** `batch_size=32`, e `Ambiente de execução →
   Reiniciar` para limpar a VRAM.
5. **Continua travado depois de 30 minutos?** Chame. Trinta minutos é o limite —
   com cinco dias de prazo, orgulho sai caro. Mande o traceback inteiro e o
   trecho de código, não só "deu erro".
