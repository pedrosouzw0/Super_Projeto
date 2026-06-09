# 💳 Análise de Risco de Crédito e Classificação Preditiva

**Disciplina:** Estatística e Probabilidade  
**Instituição:** Centro Universitário do Estado do Pará (CESUPA)  
**Professor:** Pedro Girotto  
**Aluno:** Pedro Andrade Gonçalves de Souza  
**Ano:** 2026

---

## 🎯 Objetivo do Projeto
Este projeto aplica conceitos práticos de Estatística, Probabilidade e Machine Learning no contexto do mercado financeiro (Credit Scoring). O objetivo principal é analisar o perfil de clientes e calcular a probabilidade de inadimplência grave (atrasos superiores a 90 dias) utilizando dados reais. 

O trabalho integra análise exploratória, limpeza de dados, aplicação matemática do Teorema de Bayes e treinamento de classificadores preditivos, culminando em um Dashboard Interativo para simulação de risco.

## 📂 Fonte de Dados
O conjunto de dados utilizado é o **Give Me Some Credit**, uma base pública disponibilizada pela plataforma Kaggle.
* **Variável Alvo:** `SeriousDlqin2yrs` (0 = Adimplente | 1 = Inadimplente).
* **Desbalanceamento:** A base apresenta um forte desbalanceamento, com aproximadamente 93,5% de adimplentes e 6,5% de inadimplentes.

## 🗂️ Estrutura do Repositório
O projeto foi desenvolvido de forma modular, dividido nos seguintes arquivos:

* **`Tratamento.ipynb`**: Focado na limpeza e preparação dos dados. Inclui imputação de valores ausentes pela mediana, tratamento de *outliers* na variável idade e renda, além da engenharia de atributos (criação da variável `total_atrasos`).
* **`EDA.ipynb`**: Análise Exploratória de Dados. Explora as distribuições das variáveis quantitativas e qualitativas, analisa correlações e identifica padrões de risco entre diferentes faixas etárias e de renda.
* **`Bayes.ipynb`**: Núcleo probabilístico do projeto. Implementa o cálculo do **Teorema de Bayes** para encontrar a probabilidade a posteriori de inadimplência, cruzando as verossimilhanças de variáveis categóricas (Idade, Renda e Histórico de Atrasos).
* **`Algoritmos.ipynb`**: Treinamento e avaliação de três modelos de Machine Learning: **Árvore de Decisão**, **K-Nearest Neighbors (KNN)** e **Naive Bayes**. Compara métricas como Acurácia, Precisão, Recall, F1-Score e plota a Curva ROC e a Matriz de Confusão.
* **`app.py`**: O Dashboard Interativo construído com Streamlit. Ele carrega a base limpa, exibe os principais gráficos da análise exploratória e fornece uma interface onde o usuário pode inserir os dados de um novo cliente e receber as predições de risco em tempo real pelos 3 modelos.

## 🛠️ Tecnologias e Bibliotecas
* Python 3
* Pandas & NumPy (Manipulação de Dados)
* Matplotlib & Seaborn (Visualização Gráfica)
* Scikit-Learn (Machine Learning e Métricas)
* Streamlit (Criação do Dashboard Web)

## 🚀 Como Executar o Projeto Localmente

**Passo 1: Clonar o repositório**
```bash
git clone [https://github.com/pedrosouzw0/SEU_REPOSITORIO.git](https://github.com/pedrosouzw0/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
Passo 2: Instalar as dependências

Bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
Passo 3: Executar o Dashboard Interativo
Certifique-se de que os arquivos dataset_limpo.csv e app.py estão no mesmo diretório e rode o comando:

Bash
streamlit run app.py
O Dashboard será aberto automaticamente no seu navegador padrão.
```
📈 Principais Conclusões
Histórico é o melhor preditor: A variável criada total_atrasos e a taxa de utilização do limite de crédito são os fatores de maior peso na determinação do risco.

Desempenho dos Modelos: Devido ao forte desbalanceamento das classes, a métrica de Recall foi priorizada para minimizar os Falsos Negativos (classificar um mau pagador como bom). A Árvore de Decisão se mostrou superior nesse aspecto.

Abordagem Bayesiana: A aplicação matemática do Teorema de Bayes forneceu uma baseline sólida e interpretável para validar as saídas dos modelos mais complexos de Machine Learning.
