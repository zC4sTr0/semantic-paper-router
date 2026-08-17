# LEARNING.md

Notas pessoais de estudo. Este arquivo é para **mim**, não para recrutador.
Escrito em português, no nível mais simples possível.

---

## O que vamos construir?

Um sistema que recebe o **abstract** (resumo) de um artigo científico e responde:
*"Este artigo é de Computer Science, Biology ou Economics?"*

A resposta vem da *semântica* — do significado do texto — e não de palavras-chave soltas.

Exemplos concretos de textos que o sistema deverá entender:

- **Computer Science**
  *"We propose a new neural network architecture that reduces inference latency by 40% while keeping accuracy unchanged on image classification benchmarks."*
  → Deveria ser classificado como Computer Science.

- **Biology**
  *"We investigated the effect of a point mutation on protein folding efficiency in Escherichia coli under heat stress."*
  → Deveria ser classificado como Biology.

- **Economics**
  *"This paper estimates the impact of central bank interest rate changes on household consumption and savings behavior."*
  → Deveria ser classificado como Economics.

A máquina não lê como a gente lê. Ela só entende números. A estratégia é
**transformar textos em números** e comparar os números entre si.

---

## O que eu preciso entender antes da próxima fase?

Sem matemática avançada — só as ideias:

### O que é um vetor?

Um vetor é uma **lista ordenada de números**. Exemplo: `[0.5, -1.2, 0.3]`.
Cada posição da lista é uma "dimensão". Um vetor com 3 números vive em um
espaço de 3 dimensões. Um vetor com 300 números vive em um espaço de 300
dimensões (não dá para visualizar, mas as contas são as mesmas).

### O que é um embedding?

Embedding é o **vetor que representa um texto**. Um modelo de embeddings foi
treinado com uma quantidade enorme de textos para aprender a transformar cada
texto em um vetor. A parte mágica: o modelo aprende a colocar **textos com
significado parecido em regiões próximas** do espaço vetorial.

### Por que textos parecidos ficam próximos?

Porque o modelo foi treinado para isso. Durante o treinamento, ele aprendeu
padrões como: palavras que aparecem em contextos parecidos ("neural",
"network", "training") costumam andar juntas. O resultado é que o *significado*
vira posição: dois textos sobre redes neurais ficam perto, um texto sobre
economia fica longe deles.

### O que será cosine similarity?

Cosine similarity é uma **fórmula que mede o quanto dois vetores apontam na
mesma direção**. Em vez de comparar os números um a um, ela compara o "ângulo"
entre os vetores. Resultado entre -1 e 1: perto de 1 significa "muito
parecidos", perto de 0 significa "sem relação". É a forma padrão de comparar
embeddings.

### Qual será o papel do Amazon Bedrock?

Bedrock é um serviço da AWS que entrega modelos de embeddings **prontos**
(como o Titan Embeddings). Em vez de treinar nosso próprio modelo — que exigiria
dados e computação gigantes — a gente manda o texto para a API do Bedrock e ela
nos devolve o vetor. Usaremos isso na **Fase 2**, via `boto3` (biblioteca
oficial da AWS para Python).

---

## O que EU vou implementar na próxima fase?

Antes de avançarmos para a Fase 1, faça esta tarefa manual — eu não vou
implementar por você:

1. **Crie dois vetores simples em Python.** Sugestão: represente duas frases
   curtas usando contagem de palavras. Ex.: `"hello world"` e
   `"hello world again"` podem virar vetores onde cada posição conta quantas
   vezes uma palavra aparece. Os vetores terão poucos números — perfeito para
   começar.

2. **Rode e observe.** Imprima os vetores no terminal. Tente explicar, com suas
   próprias palavras, o que cada número significa e por que os dois vetores são
   parecidos (ou diferentes).

3. **Implemente uma função de cosine similarity, passo a passo.** Orientação
   (sem entregar a solução pronta — tente antes de consultar qualquer coisa):
   - O numerador é a **soma dos produtos** das posições correspondentes dos
     dois vetores (isso se chama produto escalar / dot product).
   - O denominador é o **produto das magnitudes** (comprimentos) dos vetores.
     A magnitude de um vetor é a raiz quadrada da soma dos quadrados dos seus
     números.
   - Divida um pelo outro e confira: dois vetores idênticos devem dar 1.

   Escreva os testes mentais antes de rodar: qual valor você espera comparando
   um vetor com ele mesmo? E com um vetor cheio de zeros?

Dica geral: comece com vetores de 2 ou 3 dimensões e confira os resultados na
mão (papel e caneta) antes de deixar o Python resolver.
