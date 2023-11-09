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

### Artificial intelligence, machine learining, and deep learning

#### Artificial intelligence

- Resumidamente, IA pode ser descrita como **esforço para automatizar tarefas intelectuais normalmente executadas por humanos**.
- Desde da origem da ideia de IA em 1950 até 1980, o conceito de aprendizado (learning) não era comum. Os programas eram desenvolvidos através de regras explicitas a partir de dados explicitos, essa técnica ficou conhecida como IA simbólica.

#### Machine learning

- Em 1843, Ada Lovelace comentou a invenção do Analytical Engine (Charles Babbage, 1830), criado a partir do Pascaline (Blaise Pascal, 1642): "O Analytical Engine não tem qualquer pretensão de criar nada. Pode fazer tudo o que sabemos ordenar-lhe que faça. . . . A sua função é ajudar-nos a tornar disponível aquilo que já conhecemos.

-  A sua observação foi mais tarde citada pelo pioneiro da IA Alan Turing como "Lady Lovelace's objection" no seu artigo de referência de 1950 "Computing Machinery and Intelligence "1 , que introduziu o teste de Turing, bem como conceitos-chave que viriam a moldar a IA.2 Turing era da opinião - altamente provocadora na altura - de que os computadores poderiam, em princípio, ser capazes de emular todos os aspectos da inteligência humana.

- Um sistema de aprendizagem automática é treinado em vez de ser explicitamente programado.

- A aprendizagem automática está relacionada com a estatística matemática, mas difere da estatística em vários aspectos importantes, no mesmo sentido em que a medicina está relacionada com a química, mas não pode ser reduzida a esta, uma vez que a medicina lida com sistemas distintos com propriedades distintas.

- A aprendizagem automática, apresenta pouca teoria matemáticae é fundamentalmente uma disciplina de engenharia. Ao contrário da física teórica ou da matemática, a aprendizagem automática é um domínio muito prático, impulsionado por descobertas empíricas e profundamente dependente dos avanços no software e no hardware.

#### Learning rules and representations from data

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

#### The "deep" in "deep learning"

- Deep são camadas sucessivas de representações. (dezenas ou centenas)

- O conjunto de camadas são chamadas Redes Neurais Profundas / Aprendizado Profundo. Apesar de inspirados na nossa compreensão do cérebro, um rede neural artificial não são modelos do cérebro. 

- Estrutura matemática para represetações de dados.

- Aprendizagem profunda é uma forma de aprender representações de dados em várias fases. Uma ideia simples que quando escalonadas acaba por parecer mágia.

#### Understanding how deep learning works, in three figures

- Aprender significa encontrar um conjunto de valores para os pesos de todas as camadas de uma rede, de forma que a rede mapeie corretamente as entradas de exemplo para os seus alvos associados. 

- Para o ajuste da rede para aproximar o resultado atual do resultado esperado é calculado a perda da rede através da função de perda (função objetivo / função de custo)

- A pontuação gerada pela função de perda é utilizada para realizar um pequeno ajuste nos pesos numa  direção que diminua a pontuação de perda.

- O ajuste na direção correta é realizada pelo otimizador, que implementa o algoritmo de retropropagação.

- Inicialmente os pesos são definidos de forma aleatória. Os ajustes são realizados algumas dezenas de vezes de forma que o erro seja minimizado.

- Uma rede treinada é a aproximação das saídas aos valores desejados com uma pontuação de perda menor possível.

#### What deep learning has achieved so far

- Desde 2010 a aprendizagem profunda tem se destacada com resultados notáveis em tarefas intuitivas do ser-humano consideradas difíceis para as máquinas.

- Ainda que esteja na fase de exploração da capacidade da aprendizagem profunda, problemas antes ditos impossíveis de serem executados por máquinas, atualmente são realidades nas mais diversas áreas, como: ciência, medicina, indústria, energia, transportes, desenvolvimento de software, agricultura e até criação artística.

#### Don't believe the short-term hype

- Não se deve levar a sério a ideia de uma inteligência geral a nível humano. Apesar do avanço da IA estar mudando vários aspectos da sociedade, as espectativas geradas são demasiadamente superiores ao que é provável que a IA entregue no curto. 

- A IA já passou por duas fases de euforia, a primeira em 1970 com a IA simbólica e em 1980 com a IA de sistemas especialistas. Ambas levaram a acreditar que a criação de uma máquina com a inteligência de uma ser-humano médio estaria ao virar a esquina. 

- Como consequência a IA não conrrespondeu aos grandes investimentos e causaram o "IA winter" após esses dois momentos de euforia.

- É possível que estajamos vivenviando o terceiro ciclo.

#### The promisse of AI

- Embora as espectativas da IA não sejam realistas a curto-prazo, o mesmo não se pode afirmar quando olhamos a longo-prazo, dado principalmente as possíveis aplicações que ainda pode se desenvolver para auxílio à sociedade.

- A IA não só servirá apenas como acessório na nossa vida cotidiana, mas passará a ser central na nossa forma como trabalhamos, pensamos e vivemos.

- A IA será parte da vida das pessoas, assim como a Internet se tornou essencial.

- "Não acredite na propaganda de curto-prazo, mas acredite ms visão a longo prazo. Pode haver contratempos até que a IA seja utilizada de acordo com seu potencial, mas a IA está a chegar e vai transformar o nosso mundo de uma forma fantástica."

### Before deep learning: A brief history of machine learning

- A maioria dos algoritmos de aprendizados automáticos utilizados na indústria não são profundos.

- Nem sempre o aprendizagem profundo é a ferramenta certa.

- Importante estar familiarizado com outras abordagens de aprendizado automático.

- "Se tudo o que tem é o martelo do aprendizado profundo, todos os problemas de aprendizagem automática passam a parecer um prego".

#### Probabilistic modeling

- Aplicação dos princípios da estatística à análise de dados
- Um dos algoritmos mais conhecidos desta categoria: Naive Bayes
- O teorema de bayes assume que todas as variáveis de entrada são independentes
- Primeira implementação informática em na década de 50. Mas surgio décadas antes disso. (Século XVIII)
- Outro algoritmo é a Regressão Logística (Hello World da apendizagem automática)
- Frequentemente usados nos primeiro testes pelos cientistas de dados

#### Early neural networks

- O conceito de redes neurais suirgiu em meados de 1950, mas só se tornou "útil" depois do algorítmo de retropopagação na década de 1980, utilzando o gradiente descendente como forma de otimização. 
- A primeira implementação de uma rede neural bem sucedida foi a LeNet por Yann Lecun, quando combinou as ideias das redes convolucionais e retropagação para classificação de dígitos manuscritos na década de 90. A rede foi utilizada no serviço postal dos USA.

#### Kernel methods

- Os métodos de kernel surgiram na década de 90 e contribuíram para que as ANN caíssem no esquecimento.
- São métodos de classificação sendo o destaque entre eles o SVM (Máquina de vetor de suporte) proposta por Vladimir Vapnik e Corinna Cortes em 1995.
- SVM
	- Encontra limites de decisão que separam duas classses
	- Os dados são transformados para uma nova representação de alta dimensão de forma que seja possível separar as classes através de hiperplanos
	- Maximização de margem: Hiperplano calculado de forma que maximize a distância dos prontos mais próximos entre as duas classes
	- A função de kernel calcula a distância entre os pontos, dispensando a necessidade do cálculo das coordenadas para encontrar os hiperplanos
	- A distância entre os pontos se mantém entre a representação inicial dos dados e a representação destino
	- Método bem aceito por estar apoiado por uma extensa teoria matemática bem compreensível e explicável
	- Dificuldade me lidar com grande quantidade de dados
	- Não tem bom desempenho para problemas de percepção 

#### Decision Trees, random forests, and gradient boosting machines

- As árvores de decisão criam fluxogramas em forma de árvores que tentam represnetam as regreas a aprendidas a partir dos dados de entrada.
- Em 2010 já eram preferidas aos métodos de kernel
- O Random Forest cria várias árvores especializadas e geralmente alcançam bons resultados em problemas superficiais.
- A partir de 2014 as gradient boosting machines suepraram as random forests e ganharam notoriedade. Elas combinam modelos fracos (normalmente árvores de decisão) e melhoram seus pontos fracos de forma iterativa.
- Uma das técnicas mais utilizadas atualmente para problemas não perceptuais.

#### Back to neural networks

- Até 2010 poucas pessoas/labs ainda trabalhavam com redes neurais;
- Em 2011, Dan Ciresan da IDSIA começou a ganhar concursos acadêmicos para classificação de imagens com ANNs profundas;
- Em 2012, Geoffrey Hinton, da Univ. Toronto, liderou uma equipe no ILSVRC (ImageNet) que elevou a taxa de acerto de 74,3% para 83,6%. Desde então, as redes convolucionais dominaram a competição e tingiu 96,4% em 2015.
- Desde então, 2012, as convnets dominam tarefas de visão computacional e ainda outras tarefas de percepção.

#### What makes deep learning different

- Uma das principais vantagens do aprendizado profundo é eliminar a necessidade da engenheria de características;
- Diferentes dos métodos superficiais, as DL possuem diversas camadas que realizam várias transformações nos dados, antes realizada manualmente.
- Mesmo que sejam empilhados várias camadas superficiais, os ajustes são realizados de forma independente. Já na DL, todas as fases de transformação, ou camadas são ajustadas de forma conjunta e harmonioza.
- Duas características essenciais DL: a forma incremental, camada a camada, como são desenvolvidas representações cada vez mais complexas, e o facto de estas representações incrementais intermédias serem aprendidas em conjunto, Em conjunto, estas duas propriedades tornaram a aprendizagem profunda muito mais bem sucedida do que as abordagens anteriores à aprendizagem automática.

#### The modern machine learning landscape

- A platfaorma kaggle é referência para verificar as bibliotecas mais utilizadas no mercado;
- De 2016 a 2020, DL e árvores dorminam entre os algoritmos mais utilizados;
- DL para problemas perceptivos e árvores para problemas estruturados.
- Para árvores são utilizados Scikit-learn, XGBoost ou LightGBM. E DL é o keras em conjunto do tensor-flow
- "Em termos técnicos, isto significa que terá de estar familiarizado com Scikit-learn, XGBoost e Keras. as três bibliotecas que atualmente dominam as competições Kaggle."

### Why deep learning? Why now?

 - Em geral, três forças técnicas estão a impulsionar os avanços na aprendizagem automática: Hardware, Conjuntos de dados benchmarks e Avanços algorítmicos
 - Uma vez que o campo é orientado por descobertas experimentais e não pela teoria, os avanços algorítmicos só se tornam possíveis quando estão disponíveis dados e hardware adequados para experimentar novas ideias (ou para ampliar ideias antigas, como é frequentemente o caso). A aprendizagem automática não é matemática ou física, onde os grandes avanços podem ser feitos com uma caneta e um pedaço de papel. É uma ciência da engenharia.













