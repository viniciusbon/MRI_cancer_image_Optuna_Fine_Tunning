Brain Tumor Classification with Fine-Tuning using Optuna

Este projeto implementa um sistema de classificação de tumores cerebrais utilizando técnicas de fine-tuning e otimização de hiperparâmetros com Optuna. O modelo é treinado para classificar imagens médicas do cérebro em diferentes categorias de tumores.

📊 Resultados do Treinamento
O modelo foi treinado por 20 épocas, alcançando os seguintes resultados finais:

Loss de Treino: 0.2323

Acurácia de Treino: 85.26%

AUC de Treino: 0.9115

Loss de Validação: 0.3330

Acurácia de Validação: 77.28%

AUC de Validação: 0.8157

Evolução do Treinamento
O modelo mostrou melhoria consistente nas métricas de treino ao longo das épocas, com a AUC de treino aumentando de 0.8185 para 0.9115. A validação mantém desempenho estável, indicando boa generalização.

🚀 Funcionalidades
Fine-tuning de modelos pré-treinados para classificação de imagens médicas

Otimização com Optuna para encontrar os melhores hiperparâmetros

Métricas abrangentes: Loss, Acurácia e AUC

Validação cruzada para garantir robustez do modelo

🛠️ Tecnologias Utilizadas
Python

PyTorch / TensorFlow

Optuna

Scikit-learn

OpenCV/PIL para processamento de imagens

📁 Estrutura do Projeto
text
brain-tumor-classification/
├── src/
│   ├── train.py
│   ├── optuna_optimization.py
│   ├── models/
│   └── utils/
├── data/
├── results/
├── models/
└── README.md
🔧 Instalação e Uso
Clone o repositório:

bash
git clone https://github.com/seu-usuario/brain-tumor-classification.git
cd brain-tumor-classification
Instale as dependências:

bash
pip install -r requirements.txt
Execute o fine-tuning com Optuna:

bash
python src/optuna_optimization.py
Treine o modelo com os melhores parâmetros:

bash
python src/train.py
⚙️ Otimização com Optuna
O Optuna é utilizado para otimizar:

Taxa de aprendizado

Tamanho do batch

Arquitetura do modelo

Parâmetros de data augmentation

Hiperparâmetros do otimizador

📈 Métricas Monitoradas
Loss: Função de perda durante treino e validação

Acurácia: Porcentagem de classificações corretas

AUC: Area Under the Curve ROC, medida de capacidade discriminativa

🤝 Créditos
Este projeto é baseado e inspirado no trabalho desenvolvido por:

https://github.com/Fff4ntinh0/Brain-Tumor-Detect-IA/

Agradecimentos especiais ao autor do repositório original pelo trabalho fundamental na detecção de tumores cerebrais usando IA.

Nota: Este projeto é para fins educacionais e de pesquisa.