# PyTorch essencial

Para quem programa em Python mas nunca usou PyTorch. Leitura de ~20 minutos.
Não é um tutorial completo — é o mínimo para não travar nos Passos 6, 7 e 8
do [roteiro](roteiro.md).

Vale ler inteiro uma vez antes de começar, e voltar aqui como consulta.

## 1 · Tensor

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

## 2 · Dataset e DataLoader

**Dataset** sabe devolver *um* item. Herda de `torch.utils.data.Dataset` e
implementa dois métodos: `__len__`, que diz quantos itens existem, e
`__getitem__(i)`, que devolve o item `i` já pronto — normalmente a tupla
`(tensor_da_imagem, label_inteiro)`.

**DataLoader** recebe um Dataset e entrega *lotes*, embaralhando se você pedir
(`shuffle=True`). Iterar sobre ele num `for` devolve pares
`(imagens, labels)` — com `batch_size=64`, shapes `(64, 3, 224, 224)` e `(64,)`.

Neste projeto, `load_data()` já devolve os três loaders prontos. Você raramente
precisa criar um do zero — a exceção é o Passo 6, onde `TensorDataset` resolve.

Parâmetros do DataLoader que aparecem por aqui:

| Parâmetro | Para quê |
|---|---|
| `batch_size` | quantas amostras por lote |
| `shuffle=True` | embaralha — só no treino, nunca na avaliação |
| `num_workers` | processos paralelos carregando dados |
| `drop_last=True` | descarta o último lote incompleto; evita erro no BatchNorm |

## 3 · Módulo

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

## 4 · O loop de treino

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

## 5 · Como interpretar a loss

Com 120 classes, um modelo que chuta uniformemente tem loss `ln(120) ≈ 4,79`.

| Comportamento da loss | Diagnóstico |
|---|---|
| Travada em ~4,8 | Não está aprendendo nada. Ver os erros da seção 4. |
| Descendo devagar | Normal nos primeiros epochs |
| Desce e a validação sobe | Overfitting — é para isso que existe o early stopping |
| `nan` | Learning rate alto demais. Divida por 10. |

## 6 · Quando travar

Nesta ordem, antes de chamar o Victor:

1. **Leia a última linha do traceback**, não a primeira. É onde está o erro real.
2. **É erro de shape?** Imprima `.shape` antes e depois da operação que quebrou.
3. **É `device`?** Algum tensor ficou na CPU enquanto o modelo está na GPU.
4. **É OOM (out of memory)?** `batch_size=32`, e `Ambiente de execução →
   Reiniciar` para limpar a VRAM.
5. **Continua travado depois de 30 minutos?** Chame. Trinta minutos é o limite —
   com cinco dias de prazo, orgulho sai caro. Mande o traceback inteiro e o
   trecho de código, não só "deu erro".
