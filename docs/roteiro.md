# Roteiro — do zero à entrega

**Hoje:** segunda, 24/08. **Entrega:** sábado, 29/08.

Cada passo tem dono, tempo estimado e um critério de pronto verificável. Marque o
checkbox quando o critério for atendido — não antes.

As decisões de escopo (por que Stanford Dogs, quais experimentos, quais métricas)
estão em [`plan.md`](plan.md). Este documento é só a execução.

**Legenda:** 🔵 Victor · 🟢 Parceiro · 🔴 bloqueia o outro, faça primeiro

---

## Segunda 24/08 — fundação

### Passo 1 · Combinar com o parceiro 🔵🟢 · 30 min

- [ ] Mandar `plan.md` e este roteiro para ele
- [ ] Alinhar: dataset, escopo obrigatório, quem faz o quê
- [ ] Marcar o call diário de 15 min (sugestão: 19h)
- [ ] Criar o repo no GitHub e dar acesso a ele
- [ ] Criar a pasta compartilhada no Google Drive (é onde os embeddings vão morar)

**Pronto quando:** os dois têm acesso ao repo e ao Drive, e concordam com o escopo.

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

**Pronto quando:** este trecho roda e mostra imagens de verdade —

```python
from dogs.config import TrainConfig
from dogs.data import load_data

data = load_data(TrainConfig(experiment_name="smoke"))
images, labels = next(iter(data.train_loader))
print(images.shape, labels.shape, data.num_classes)   # (64,3,224,224) (64,) 120
```

---

## Terça 25/08 — embeddings e EDA

### Passo 4 · `features.py` 🔴🔵 · 2h + 8 min de execução

**A tarefa mais importante da semana.** Enquanto os embeddings não existirem, o
parceiro está bloqueado em E2 e depende de GPU pra tudo.

- [ ] `get_device()`, `build_backbone()`, `extract()`, `main()`
- [ ] Rodar: `python -m dogs.features --backbone resnet50`
- [ ] Conferir os shapes: train ≈ (10200, 2048), val ≈ (1800, 2048), test ≈ (8580, 2048)
- [ ] **Subir `data/processed/features/` no Drive compartilhado**
- [ ] Avisar o parceiro no mesmo momento

Armadilha: rodar com `use_augmentation=True`. O embedding de cada imagem mudaria a
cada execução e o cache perderia a razão de existir.

**Pronto quando:** os seis `.npy` estão no Drive e o parceiro confirmou que carregou.

---

### Passo 5 · `train.py` + `evaluate.py` 🔵 · 2h30

O parceiro precisa disso pra rodar E1. Terminar hoje.

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

- [ ] Distribuição de imagens por classe (é balanceado?)
- [ ] Grid de amostras de 8-10 raças
- [ ] Distribuição de resoluções e proporções
- [ ] **Encontrar 3-4 pares de raças visualmente parecidas e montar um grid
      lado a lado.** Essa figura vai voltar na análise de erros — a graça é
      comparar com o que o modelo realmente confundiu.
- [ ] Salvar os gráficos em `reports/figures/`

**Pronto quando:** as figuras estão salvas e ele consegue explicar em duas frases
por que esse problema é difícil.

---

## Quarta 26/08 — os modelos

### Passo 7 · E1, CNN do zero 🟢 · 3h

`notebooks/02_baseline.ipynb`

- [ ] Implementar `SmallCNN` em `models.py`
- [ ] Treinar (15 epochs, ~40 min na GPU do Colab)
- [ ] `log_result("E1_scratch", "val", metrics)`

Acurácia baixa é o resultado esperado, não um erro. Se der acima de 40%, desconfie:
provavelmente o backbone pré-treinado entrou sem querer.

**Pronto quando:** a linha E1 está no `results.csv`.

---

### Passo 8 · E2, linear probe 🟢 · 1h30

Mesmo notebook. Treina em segundos — dá pra iterar à vontade.

- [ ] Implementar `LinearProbe`
- [ ] Treinar sobre os embeddings do Drive
- [ ] `log_result("E2_linear_probe", "val", metrics)`
- [ ] Ablação barata: rodar sem BatchNorm e registrar também

Esperado: salto grande sobre E1. Esse contraste é o coração do trabalho.

**Pronto quando:** E2 está no CSV e ele consegue explicar por que ganhou de E1.

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

> Se sobrar tempo: peça para o seu parceiro rodar o mesmo teste na máquina dele.
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
| Parceiro travado numa tarefa | Ele passa a tarefa e assume redação. Descobrir isso na sexta é fatal — por isso o call diário. |
| Atrasou dois dias | Corta E3. E1 + E2 + análise de erros bem-feita ainda é um trabalho completo. |
