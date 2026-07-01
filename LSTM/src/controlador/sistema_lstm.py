from dados.carregador_dados import CarregadorDados
from dados.normalizador import Normalizador
from dados.gerador_sequencias import GeradorSequencias
from modelo.rede_lstm import RedeLSTM
from modelo.treinador import Treinador
from modelo.previsor import Previsor
from metricas.avaliador import Avaliador
from visualizacao.graficos import Graficos


class SistemaLSTM:

    def __init__(self):

        self.carregador = CarregadorDados(
            "dados/passageiros.csv"
        )
        self.normalizador = Normalizador()
        self.gerador = GeradorSequencias(
            tamanho_janela=12,
            percentual_treino=0.80
        )
        self.rede = RedeLSTM()
        self.treinador = Treinador()
        self.previsor = Previsor()
        self.avaliador = Avaliador()
        self.graficos = Graficos()

    def executar(self):

        print("=" * 60)
        print("PREVISÃO DE SÉRIES TEMPORAIS UTILIZANDO LSTM")
        print("=" * 60)

        dados = self.carregador.carregar()

        serie = self.carregador.obter_serie_temporal()

        serie_normalizada = self.normalizador.normalizar(serie)

        (
            x_treino,
            y_treino,
            x_teste,
            y_teste
        ) = self.gerador.gerar(serie_normalizada)

        modelo = self.rede.criar(
            entrada=(x_treino.shape[1], 1)
        )

        historico = self.treinador.treinar(
            modelo,
            x_treino,
            y_treino
        )

        previsoes = self.previsor.prever(
            modelo,
            x_teste
        )

        y_real = self.normalizador.desnormalizar(y_teste)
        y_previsto = self.normalizador.desnormalizar(previsoes)
        resultado = self.avaliador.avaliar(y_real,y_previsto)

        self.avaliador.exibir(resultado)
        self.graficos.loss(historico)
        self.graficos.previsao(y_real,y_previsto)

        self.graficos.serie_temporal(
            dados["Passengers"].values
        )

        modelo.save("resultados/modelo.keras")

        print()
        print("Modelo salvo em resultados/modelo.keras")
        print("Gráficos salvos em resultados/")
        print()
        print("Execução finalizada.")