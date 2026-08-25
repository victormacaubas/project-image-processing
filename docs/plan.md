# Plano de Trabalho — Classificação Fine-Grained de Raças de Cães

**Entrega: sábado, 29/08/2026.** Hoje é segunda, 24/08. Cinco dias.

---

## 1. O problema

Classificar a raça de um cão a partir de uma foto, entre 120 raças (Stanford Dogs).

Isso é *fine-grained classification*: todas as classes pertencem à mesma categoria geral
(cão), então as diferenças entre classes são sutis — formato de focinho, textura da
pelagem, proporção das orelhas. Ao mesmo tempo, a variação **dentro** de cada classe é
grande: pose, iluminação, fundo, oclusão, cães filhotes vs. adultos. É comum que duas
imagens da mesma raça sejam visualmente mais distantes entre si do que duas imagens de
raças diferentes. É isso que torna o problema difícil e interessante.

### A pergunta que guia o trabalho

> Quanto de representação visual é preciso *aprender* versus *transferir* para resolver
> classificação fine-grained com dados limitados?

Cada experimento responde uma parte dela. Isso dá ao relatório um fio condutor, em vez
de uma lista solta de modelos testados.

### ⚠️ Ponto crítico: Stanford Dogs vem do ImageNet

As imagens do Stanford Dogs foram extraídas do ImageNet. Backbones pré-treinados em
ImageNet-1k **já viram essas imagens durante o pré-treino**. Isso significa que a
acurácia do transfer learning aqui é otimista e não representa o que aconteceria num
domínio novo.

Não é um problema — é o **achado mais interessante do trabalho**, desde que tratemos com
honestidade. Ganha uma seção própria na análise. A maioria dos trabalhos com esse
dataset ignora isso; mencionar demonstra maturidade.

---

## 2. Dataset

**Stanford Dogs** — 20.580 imagens, 120 raças, ~150-250 imagens por classe.

- Fonte: `https://huggingface.co/datasets/Voxel51/StanfordDogs`
- Split oficial: 12.000 treino / 8.580 teste
- Vamos separar 15% do treino como validação (o teste só é tocado uma vez, no fim)

Dataset pequeno por classe — exatamente o cenário onde transfer learning importa.

---

## 3. Escopo — o que entra e o que fica de fora

Cinco dias. O escopo abaixo é o compromisso; qualquer coisa além disso é bônus.

### Obrigatório (o trabalho existe sem isso? não)

| # | Experimento | Por que está aqui |
|---|---|---|
| E1 | CNN pequena treinada do zero | Baseline. Vai ter acurácia baixa (~15-25%). Prova que o problema é difícil e dá referência para o resto. |
| E2 | Linear probe sobre backbone congelado (ResNet50) | Extração de features pura. Mostra o salto do transfer learning sem treinar o backbone. |
| E3 | Fine-tuning parcial (descongelar últimos blocos) | Adaptar a representação ao domínio. Melhor resultado esperado. |
| E4 | Análise de erros | Matriz de confusão, pares de raças mais confundidos, imagens com maior perda. É aqui que o trabalho deixa de ser "rodei uns modelos". |

### Se sobrar tempo (nesta ordem)

- **E5** — Segundo backbone para comparação (ViT-B/16 ou EfficientNet) sobre os mesmos embeddings pré-computados. Custo baixo.
- **E6** — CLIP zero-shot. Sem treino nenhum, só prompts. Conecta com o tema da pós e dá um contraste forte: "modelo de fundação sem ver um único exemplo rotulado vs. nosso modelo fine-tuned".
- **E7** — Ablação de data augmentation.

### Fora de escopo (declarar no relatório como trabalho futuro)

Detecção/localização, atenção por partes (métodos fine-grained especializados),
ensembles, busca de hiperparâmetros extensiva, deploy.

---

## 4. A decisão técnica que salva o prazo

**Pré-computar os embeddings uma vez e reusar.**

Passar as 20.580 imagens por um backbone congelado leva ~5-8 min na GPU do Colab. Salvos
em disco (`.npy`, ~200 MB), qualquer classificador linear em cima deles treina em
**segundos**, na CPU.

Consequências:

1. E2, E5 e boa parte da análise viram experimentos instantâneos — dá pra iterar dezenas
   de vezes numa tarde.
2. **A Mari consegue trabalhar sem GPU e sem depender de você.** Você gera os
   embeddings na terça de manhã, sobe no Drive compartilhado, e ela fica autônoma.

O único treino caro é E3 (fine-tuning), que roda uma ou duas vezes.

Isso é o que torna 5 dias viável.

---

## 5. Cronograma

| Dia | Victor | Mari | Marco |
|---|---|---|---|
| **Seg 24** | Repo, `data.py`, download do dataset, split | Ambiente Colab funcionando, roda o notebook de EDA | Dataset carregando |
| **Ter 25** | `features.py` → **gerar e subir embeddings no Drive** (prioridade máxima da semana) | EDA: distribuição de classes, grid de imagens, tamanhos, exemplos de classes parecidas | **Embeddings prontos até o fim do dia** |
| **Qua 26** | E3 fine-tuning (treino longo, deixar rodando) | E1 CNN do zero + E2 linear probe | Todos os resultados obrigatórios existem |
| **Qui 27** | E4 análise de erros, gráficos comparativos | Redação: problema, dataset, metodologia | Bônus (E5/E6) só se tudo acima estiver fechado |
| **Sex 28** | Montar notebook final, revisar números | Redação: experimentos, resultados, conclusão | **Notebook roda de ponta a ponta sem erro** |
| **Sáb 29** | Revisão cruzada, entrega | Revisão cruzada | Entregue |

### Regra de corte

**Quarta à noite é o ponto de decisão.** Se E1-E4 não estiverem prontos até lá, corta
todo o bônus sem discussão e usa quinta pra fechar o obrigatório. Um trabalho completo
e modesto vale mais que um ambicioso pela metade.

---

## 6. Divisão do trabalho

A Mari tem menos experiência, então a divisão dá a ela tarefas **bem delimitadas
com critério de pronto claro**, e a você tudo que é infraestrutura ou pode travar o
outro.

### Victor — infraestrutura e modelagem pesada

- `src/dogs/data.py` — download, splits, transforms, DataLoaders
- `src/dogs/features.py` — extração de embeddings (**bloqueia a Mari, faz primeiro**)
- `src/dogs/train.py` — loop de treino, early stopping, checkpoints
- E3 fine-tuning
- E4 análise de erros
- Montagem do notebook final e revisão dos números

### Mari — experimentos guiados e narrativa

- EDA e visualizações (`notebooks/01_eda.ipynb`)
- E1 — CNN do zero (tarefa didática: ela constrói a arquitetura)
- E2 — linear probe sobre os embeddings que você gerou
- Tabela consolidada de resultados
- Primeira versão do texto de todas as seções

### Regras de convivência no git

- **Ninguém commita `.ipynb` com output.** Rodar `nbstripout` antes do commit (já
  configurado no `.pre-commit-config.yaml`) — notebook com output gera conflito
  irresolvível.
- Cada um trabalha no seu próprio notebook. Só o `99_final.ipynb` é compartilhado, e
  só na sexta.
- Lógica de verdade vai em `src/`, não no notebook. Notebook importa e chama.
- Resultados numéricos vão para `reports/results.csv` (append), não ficam só na cabeça
  de quem rodou.

### Sincronização

Call rápido de 15 min no fim de cada dia. O que rodou, o que quebrou, o que muda amanhã.
Com 5 dias, descobrir na sexta que algo não funciona é fatal.

---

## 7. Formato de entrega

**Repo + notebook final** (`notebooks/99_final.ipynb`).

O notebook é o documento de entrega e precisa cobrir os cinco itens exigidos, cada um
com célula de texto explicativa:

1. Descrição do problema
2. Descrição da base de dados
3. Metodologia
4. Relato dos experimentos
5. Análise dos resultados e conclusões

O notebook importa de `src/` em vez de definir tudo inline. Isso mantém ele legível e
permitiu que vocês trabalhassem em paralelo a semana toda.

**Requisito não negociável:** o notebook precisa rodar de ponta a ponta, do zero, sem
erro. Testar isso na sexta, em runtime limpo do Colab. Carregar checkpoints salvos em
vez de retreinar — deixar o código de treino visível mas atrás de um flag
`RETRAIN = False`.

---

## 8. Métricas

- **Acurácia top-1** — métrica principal
- **Acurácia top-5** — relevante em fine-grained com 120 classes; mostra se o modelo
  está "quase certo"
- **F1 macro** — o dataset é levemente desbalanceado; média macro não deixa classes
  pequenas sumirem
- **Matriz de confusão** — 120x120 não é legível inteira. Mostrar o top-20 de pares mais
  confundidos e um recorte de grupos parecidos (ex.: terriers entre si)

Todo experimento reporta as três primeiras na mesma tabela, com o mesmo split. Sem isso,
comparação não vale nada.
