import numpy as np

class Neuronio:
    def __init__(self, num_entradas):
        self.pesos = np.random.rand(num_entradas)
        self.bias = np.random.rand()
        self.taxa_aprendizado = 0.1

    def funcao_ativacao(self, soma):
        return 1 if soma >= 0 else 0

    def prever(self, entradas):
        soma_ponderada = np.dot(entradas, self.pesos) + self.bias
        return self.funcao_ativacao(soma_ponderada)

    def treinar(self, entradas, alvo):
        # Calcula a previsão atual
        previsao = self.prever(entradas)
        
        # Calcula o erro
        erro = alvo - previsao
        
        # Ajusta os pesos e o bias
        self.pesos += self.taxa_aprendizado * erro * entradas
        self.bias += self.taxa_aprendizado * erro

# Exemplo de uso
np.random.seed(42)
neuronio = Neuronio(num_entradas=2)

dados_treino = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
rotulos = np.array([0, 0, 0, 1])

# Treinamento por 10 épocas
for epoca in range(10):
    for i in range(len(dados_treino)):
        neuronio.treinar(dados_treino[i], rotulos[i])

# Teste do neurônio treinado
print("Previsão [0, 1]:", neuronio.prever([0, 1]))  
print("Previsão [1, 1]:", neuronio.prever([1, 1]))  
