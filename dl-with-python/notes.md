[Sintaxe markdown] (https://docs.github.com/pt/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

# Deep Learning with Python

Second Edition

Francois Chollet

## Ch1. What is Deep Learning?

- É preciso ser capaz de identificar a verdade em meios aos ruídos.

- Até onde a aprendizagem profunda conseguiu chegar?
- Qual é a sua importância?
- Para onde devemos seguir?
- Devemos acreditar no hype?

## Artificial intelligence, machine learining, and deep learning

### Artificial intelligence

- Resumidamente, IA pode ser descrita como **esforço para automatizar tarefas intelectuais normalmente executadas por humanos**.
- Desde da origem da ideia de IA em 1950 até 1980, o conceito de aprendizado (learning) não era comum. Os programas eram desenvolvidos através de regras explicitas a partir de dados explicitos, essa técnica ficou conhecida como IA simbólica.

### Machine learning

- Em 1843, Ada Lovelace comentou a invenção do Analytical Engine (Charles Babbage, 1830), criado a partir do Pascaline (Blaise Pascal, 1642): "O Analytical Engine não tem qualquer pretensão de criar nada. Pode fazer tudo o que sabemos ordenar-lhe que faça. . . . A sua função é ajudar-nos a tornar disponível aquilo que já conhecemos.

-  A sua observação foi mais tarde citada pelo pioneiro da IA Alan Turing como "Lady Lovelace's objection" no seu artigo de referência de 1950 "Computing Machinery and Intelligence "1 , que introduziu o teste de Turing, bem como conceitos-chave que viriam a moldar a IA.2 Turing era da opinião - altamente provocadora na altura - de que os computadores poderiam, em princípio, ser capazes de emular todos os aspectos da inteligência humana.

- Um sistema de aprendizagem automática é treinado em vez de ser explicitamente programado.

- A aprendizagem automática está relacionada com a estatística matemática, mas difere da estatística em vários aspectos importantes, no mesmo sentido em que a medicina está relacionada com a química, mas não pode ser reduzida a esta, uma vez que a medicina lida com sistemas distintos com propriedades distintas.

- A aprendizagem automática, apresenta pouca teoria matemáticae é fundamentalmente uma disciplina de engenharia. Ao contrário da física teórica ou da matemática, a aprendizagem automática é um domínio muito prático, impulsionado por descobertas empíricas e profundamente dependente dos avanços no software e no hardware.

### Learning rules and representations from data

- Aprendizagem profundo x aprendizagem automática
- Necessário três coisas para o aprendizado automático
	- Dados de entrada;
	- Exemplos do resultado esperado;
	- Uma forma de medir o desempenho do algoritmo.

- O ajuste do algoritmo para dimimunir a distância do resultado atual do resultado esperado e chamada de aprendizado.

- O problema central da aprendizagem automática é aprender representações úteis a partir dos dados de entrada.

- Procura automática de transformações dos dados para representar de forma adequada para a tarefa em questão.

- Não são criativos. Apenas procuram dentro de um espaço de hipóteses

- A aprendizagem automática procura represtações e regras úteis sobre alguns dados de entrada, dentro de um espaço de possibilidades predefinido, utilizando a orientação de um sinal de feedback. 

### The "deep" in "deep learning"

- Deep são camadas sucessivas de representações. (dezenas ou centenas)

- O conjunto de camadas são chamadas Redes Neurais Profundas / Aprendizado Profundo. Apesar de inspirados na nossa compreensão do cérebro, um rede neural artificial não são modelos do cérebro. 

- Estrutura matemática para represetações de dados.

- Aprendizagem profunda é uma forma de aprender representações de dados em várias fases. Uma ideia simples que quando escalonadas acaba por parecer mágia.

### Understanding how deep learning works, in three figures

- Aprender significa encontrar um conjunto de valores para os pesos de todas as camadas de uma rede, de forma que a rede mapeie corretamente as entradas de exemplo para os seus alvos associados. 

- Para o ajuste da rede para aproximar o resultado atual do resultado esperado é calculado a perda da rede através da função de perda (função objetivo / função de custo)

- A pontuação gerada pela função de perda é utilizada para realizar um pequeno ajuste nos pesos numa  direção que diminua a pontuação de perda.

- O ajuste na direção correta é realizada pelo otimizador, que implementa o algoritmo de retropropagação.

- Inicialmente os pesos são definidos de forma aleatória. Os ajustes são realizados algumas dezenas de vezes de forma que o erro seja minimizado.

- Uma rede treinada é a aproximação das saídas aos valores de entrada com uma pontuação de perda menor possível.

- 




