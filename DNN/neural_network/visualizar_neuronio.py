import pygame
import sys
import math

# Inicialização do Pygame
pygame.init()

# Configurações da Janela
LARGURA, ALTURA = 800, 600
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Visualização de Neurônio Artificial")
clock = pygame.time.Clock()

# Cores Modernas (Paleta Dark Mode)
COR_BG = (18, 18, 24)        # Fundo escuro
COR_NEURONIO = (58, 134, 255) # Azul brilhante
COR_ENTRADA = (255, 0, 127)   # Rosa choque
COR_TEXTO = (240, 240, 245)   # Branco suave
COR_LINHA = (80, 80, 100)     # Cinza para conexões
COR_PULSO = (0, 255, 180)     # Verde neon para o fluxo de dados

# Fontes
fonte_principal = pygame.font.SysFont("Arial", 22, bold=True)
fonte_sub = pygame.font.SysFont("Arial", 16)

# Posições dos Elementos (X, Y)
pos_saida = (550, ALTURA // 2)
pos_entradas = [
    (250, ALTURA // 2 - 120),  # Entrada X1
    (250, ALTURA // 2),        # Entrada X2
    (250, ALTURA // 2 + 120)   # Entrada X3 / Bias
]

# Variáveis da Animação do Fluxo
tempo = 0.0

def desenhar_texto(texto, fonte, cor, x, y, centralizado=True):
    """Função utilitária para renderizar textos na tela."""
    superficie = fonte.render(texto, True, cor)
    retangulo = superficie.get_rect()
    if centralizado:
        retangulo.center = (x, y)
    else:
        retangulo.topleft = (x, y)
    tela.blit(superficie, retangulo)

# Loop Principal da Janela
while True:
    tela.fill(COR_BG)
    tempo += 0.05  # Controla a velocidade da animação dos pulsos

    # Captura de eventos para fechar o programa
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 1. DESENHAR AS CONEXÕES (SINAPSES) E ANIMAÇÃO DE FLUXO
    for i, pos_in in enumerate(pos_entradas):
        # Linha base da conexão
        pygame.draw.line(tela, COR_LINHA, pos_in, pos_saida, 3)
        
        # Efeito de pulso de dados (bolinha correndo pelos fios)
        # O uso do operador modulo (%) faz a bolinha reiniciar o caminho continuamente
        distancia_atual = (tempo + i * 0.3) % 1.0 
        pulso_x = pos_in[0] + (pos_saida[0] - pos_in[0]) * distancia_atual
        pulso_y = pos_in[1] + (pos_saida[1] - pos_in[1]) * distancia_atual
        pygame.draw.circle(tela, COR_PULSO, (int(pulso_x), int(pulso_y)), 6)

        # Rótulos dos Pesos (w) no meio das linhas
        meio_x = (pos_in[0] + pos_saida[0]) // 2
        meio_y = (pos_in[1] + pos_saida[1]) // 2 - 15
        desenhar_texto(f"w{i+1}", fonte_sub, COR_TEXTO, meio_x, meio_y)

    # 2. DESENHAR OS NÓS DE ENTRADA (INPUTS)
    rotulos_entradas = ["Entrada X1", "Entrada X2", "Viés (Bias)"]
    for i, pos_in in enumerate(pos_entradas):
        # Sombra/Brilho externo do nó
        pygame.draw.circle(tela, (COR_ENTRADA[0]//3, COR_ENTRADA[1]//3, COR_ENTRADA[2]//3), pos_in, 32)
        # Nó principal
        pygame.draw.circle(tela, COR_ENTRADA, pos_in, 25)
        # Texto descritivo ao lado do nó
        desenhar_texto(rotulos_entradas[i], fonte_sub, COR_TEXTO, pos_in[0] - 80, pos_in[1], centralizado=False)

    # 3. DESENHAR O NEURÔNIO PRINCIPAL (CÉLULA DE PROCESSAMENTO)
    # Sombra/Brilho externo do neurônio
    pygame.draw.circle(tela, (COR_NEURONIO[0]//4, COR_NEURONIO[1]//4, COR_NEURONIO[2]//4), pos_saida, 75)
    # Corpo celular
    pygame.draw.circle(tela, COR_NEURONIO, pos_saida, 60)
    # Borda interna estilizada
    pygame.draw.circle(tela, COR_TEXTO, pos_saida, 60, 2)
    
    # Textos dentro do neurônio indicando a soma matemática e a ativação
    desenhar_texto("Σ (Soma)", fonte_principal, COR_TEXTO, pos_saida[0], pos_saida[1] - 15)
    desenhar_texto("f(x) Ativação", fonte_principal, COR_TEXTO, pos_saida[0], pos_saida[1] + 15)

    # 4. DESENHAR A SAÍDA (OUTPUT)
    # Linha saindo do neurônio para a direita
    pos_fim_saida = (pos_saida[0] + 120, pos_saida[1])
    pygame.draw.line(tela, COR_NEURONIO, pos_saida, pos_fim_saida, 4)
    desenhar_texto("Saída (Y)", fonte_principal, COR_PULSO, pos_fim_saida[0] + 50, pos_fim_saida[1])

    # Atualiza o frame da tela
    pygame.display.flip()
    clock.tick(60) # Mantém o programa rodando estavelmente a 60 FPS
