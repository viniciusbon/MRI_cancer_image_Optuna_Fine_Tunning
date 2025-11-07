#%%
# ------------------------------------------------------------------
# 1. IMPORTAÇÕES
# ------------------------------------------------------------------
# Bibliotecas padrão e de processamento
import os
import numpy as np
from PIL import Image
import random # Necessário para set_seeds

# Bibliotecas do PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Bibliotecas de Otimização e Suporte
import optuna
from sklearn.model_selection import train_test_split

# Remover importações não utilizadas para limpar o script
# import matplotlib.pyplot as plt
# import optunahub
# import torch.nn.functional as F
# from torchvision import transforms

#%%
# ------------------------------------------------------------------
# 2. CONFIGURAÇÃO DE SEMENTES (REPRODUTIBILIDADE)
# ------------------------------------------------------------------
# Função para garantir que os resultados sejam os mesmos em todas as execuções
def set_seeds(seed_value=42):
    """Define as sementes para reprodutibilidade."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # Garante a reprodutibilidade em operações CUDA (pode afetar performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Chamamos a função imediatamente
set_seeds(42)

#%%
# ------------------------------------------------------------------
# 3. CONFIGURAÇÕES GLOBAIS (CAMINHOS E DISPOSITIVO)
# ------------------------------------------------------------------
# Seus caminhos originais
image_path_yes = r"C:\Users\ADM\Desktop\Optuna fine tunning\Brain-Tumor-Detect-IA\brain_tumor_dataset\yes"
image_path_no  = r"C:\Users\ADM\Desktop\Optuna fine tunning\Brain-Tumor-Detect-IA\brain_tumor_dataset\no"

# Define o dispositivo de treinamento (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Usando dispositivo: {device}')

# Definir o número de épocas para cada "trial" do Optuna
# Um valor baixo (ex: 10-20) é bom para otimizações rápidas de HP
N_EPOCHS_PER_TRIAL = 15 
# Número de "trials" (tentativas) que o Optuna fará
N_TRIALS = 50 

#%%
# ------------------------------------------------------------------
# 4. FUNÇÃO DE CARREGAMENTO DE DADOS (Seu código)
# ------------------------------------------------------------------
# Pequena modificação: garantir dtype=np.float32 para PyTorch
def carregar_imagens(pasta, rotulo, tamanho=(128, 128)):
    """Carrega, redimensiona, converte para escala de cinza e normaliza imagens."""
    imagens = [] 
    labels = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(caminho).convert('L') # 'L' = escala de cinza
                img = img.resize(tamanho)
                # Converte para float32 e normaliza para [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0 
                imagens.append(img_array)
                labels.append(rotulo)
            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {e}")
    return np.array(imagens), np.array(labels)

#%%
# ------------------------------------------------------------------
# 5. PROCESSAMENTO E DIVISÃO DOS DADOS
# ------------------------------------------------------------------
# Carregar dados "sim" (1) e "não" (0)
X_sim, y_sim = carregar_imagens(image_path_yes, 1) 
X_nao, y_nao = carregar_imagens(image_path_no, 0) 

# Juntar tudo em arrays únicos
X = np.concatenate((X_sim, X_nao), axis=0)
y = np.concatenate((y_sim, y_nao), axis=0)

# [CORREÇÃO CRÍTICA] Adicionar dimensão do canal
# CNNs do PyTorch esperam (Batch, Canais, Altura, Largura)
# Nossos dados estão como (N, 128, 128). Precisam ser (N, 1, 128, 128)
X = np.expand_dims(X, axis=1) # Adiciona o canal '1'

print(f"Shape final do X (com canal): {X.shape}")
print(f"Shape final do y: {y.shape}")

# Dividir os dados em conjuntos de TREINO e VALIDAÇÃO
# O Optuna usará o set de VALIDAÇÃO para avaliar o "trial"
# Usamos 20% dos dados para validação
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, # Semente para reprodutibilidade da divisão
    stratify=y       # Garante proporção igual de classes em treino/val
)

print(f"Tamanho Treino: {len(X_train)}, Tamanho Validação: {len(X_val)}")

# Converter os arrays de numpy para Tensores do PyTorch
# .float() é essencial para dados de imagem e labels de BCEWithLogitsLoss
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).float()
X_val_tensor = torch.tensor(X_val).float()
y_val_tensor = torch.tensor(y_val).float()

# Criar Datasets do PyTorch (agrupa dados e rótulos)
# Os DataLoaders serão criados DENTRO da função objective,
# pois o 'batch_size' será um hiperparâmetro otimizado.
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

#%%
# ------------------------------------------------------------------
# 6. DEFINIÇÃO DO MODELO (CNN)
# ------------------------------------------------------------------
# Esta classe define nossa arquitetura de CNN.
# O 'trial' do Optuna será passado aqui para sugerir
# o número de filtros, neurônios e taxas de dropout.

class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        
        # --- Bloco Convolucional 1 ---
        # Sugere o número de filtros (canais de saída) para a primeira camada
        # Usamos 'categorical' para forçar potências de 2, que são eficientes
        out_channels_1 = trial.suggest_categorical('out_channels_1', [16, 32, 64])
        self.conv1 = nn.Conv2d(
            in_channels=1,            # 1 canal de entrada (escala de cinza)
            out_channels=out_channels_1,
            kernel_size=5,            # Kernel 5x5
            stride=1,
            padding=2                 # Padding 'same' (mantém 128x128)
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Imagem -> 64x64
        
        # --- Bloco Convolucional 2 ---
        out_channels_2 = trial.suggest_categorical('out_channels_2', [32, 64, 128])
        self.conv2 = nn.Conv2d(
            in_channels=out_channels_1, # Entrada é a saída da camada anterior
            out_channels=out_channels_2,
            kernel_size=3,            # Kernel 3x3
            stride=1,
            padding=1                 # Padding 'same' (mantém 64x64)
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Imagem -> 32x32

        # --- Camadas Totalmente Conectadas (Classificador) ---
        
        # Calculamos o tamanho de entrada para a camada linear
        # Após 2x MaxPool (128 -> 64 -> 32), a imagem é 32x32
        # O número de features é (canais_saida_2 * 32 * 32)
        self.fc1_in_features = out_channels_2 * 32 * 32
        
        # Sugere o número de neurônios na camada oculta
        fc1_units = trial.suggest_int('fc1_units', 50, 500, log=True)
        self.fc1 = nn.Linear(self.fc1_in_features, fc1_units)
        self.relu3 = nn.ReLU()
        
        # Sugere a taxa de dropout
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Camada de saída final: 1 neurônio (classificação binária)
        self.fc2 = nn.Linear(fc1_units, 1)

    def forward(self, x):
        # Passa pelos blocos convolucionais
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # "Achata" o tensor 4D (Batch, C, H, W) para 2D (Batch, Features)
        x = x.view(-1, self.fc1_in_features)
        
        # Passa pelo classificador
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x) # Saída é um "logit" bruto
        return x

#%%
# ------------------------------------------------------------------
# 7. FUNÇÃO "OBJECTIVE" (O Coração do Optuna)
# ------------------------------------------------------------------
# Esta função é chamada pelo Optuna para cada "trial" (tentativa).
# Ela define os HPs, treina o modelo e retorna uma métrica (acurácia).

def objective(trial):
    
    # Garante reprodutibilidade para este trial específico
    set_seeds(42)

    # --- 1. Definir Hiperparâmetros do Trial ---
    
    # Instanciar o modelo, passando 'trial' para que ele sugira a arquitetura
    model = Net(trial).to(device)
    
    # Sugerir o otimizador
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # Sugerir a taxa de aprendizado (learning rate)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Criar o otimizador com os HPs sugeridos
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Sugerir o tamanho do batch
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Criar os DataLoaders com o batch_size sugerido
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False # Não precisa embaralhar na validação
    )
    
    # Função de perda (Loss)
    # BCEWithLogitsLoss é ideal para classificação binária com 1 saída
    # Ela já aplica a função Sigmoid internamente (mais estável)
    criterion = nn.BCEWithLogitsLoss()

    # --- 2. Loop de Treinamento e Validação ---
    for epoch in range(N_EPOCHS_PER_TRIAL):
        
        # --- Treinamento ---
        model.train() # Coloca o modelo em modo de treino
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ajustar shape do label para [Batch, 1] (exigido pela loss)
            labels = labels.unsqueeze(1)
            
            optimizer.zero_grad()    # Zerar gradientes
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels) # Calcular perda
            loss.backward()          # Backward pass (calcular gradientes)
            optimizer.step()         # Atualizar pesos
            
        # --- Validação ---
        model.eval() # Coloca o modelo em modo de avaliação (desliga dropout)
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Desliga o cálculo de gradientes
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                
                outputs = model(inputs) # Obter logits
                
                # Converter logits para probabilidades (Sigmoid) e depois para classes (0 ou 1)
                predicted = torch.sigmoid(outputs) > 0.5
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calcular acurácia da validação
        val_accuracy = val_correct / val_total
        
        # --- Relatório para o Optuna (Pruning) ---
        # Reporta a acurácia da época atual para o 'trial'
        trial.report(val_accuracy, epoch)
        
        # Verifica se este 'trial' deve ser interrompido (poda)
        # Se o 'trial' está performando muito mal comparado a outros, 
        # o Optuna o cancela para economizar tempo.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Retorna a métrica final (acurácia da última época)
    # O Optuna tentará MAXIMIZAR este valor.
    return val_accuracy

#%%
# ------------------------------------------------------------------
# 8. EXECUÇÃO DO ESTUDO OPTUNA
# ------------------------------------------------------------------

print("Iniciando estudo de otimização do Optuna...")

# Criar um "Pruner": um objeto que monitora e "poda" (corta) trials ruins
pruner = optuna.pruners.MedianPruner()

# Criar o estudo
# direction="maximize": queremos maximizar a acurácia
study = optuna.create_study(direction="maximize", pruner=pruner)

# Iniciar a otimização
# O 'objective' será chamado N_TRIALS vezes
try:
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        timeout=600 # Opcional: tempo máximo (em seg) para o estudo
    )
except KeyboardInterrupt:
    print("Estudo interrompido manualmente.")

# --- 9. Resultados ---
print("\n" + "="*30)
print(" ESTUDO CONCLUÍDO ")
print("="*30)

print(f"Número de trials concluídos: {len(study.trials)}")

print("\nMelhor Trial:")
trial = study.best_trial

print(f"  🏆 Valor (Acurácia): {trial.value:.4f}")

print("  📋 Melhores Hiperparâmetros:")
for key, value in trial.params.items():
    print(f"    - {key}: {value}")
# %%
