from dataset import DatasetCifar10
from modelos import ModelosCNN
from treinamento import TreinadorCNN
from avaliacao import AvaliadorCNN
from utilitarios import GeradorGraficos


class AplicacaoCNN:

    def __init__(self) -> None:

        self._dataset = DatasetCifar10()

        (
            self._imagens_treinamento,
            self._rotulos_treinamento,
            self._imagens_teste,
            self._rotulos_teste
        ) = self._dataset.carregar()

        self._modelos = ModelosCNN(
            formato_entrada=self._dataset.formato_entrada(),
            quantidade_classes=self._dataset.quantidade_classes()
        )

        self._gerador_graficos = GeradorGraficos()

        self._nomes_classes = self._dataset.obter_nomes_classes()

    def executar(self) -> None:

        modelos = {
            "modelo_1": self._modelos.criar_modelo_1(),
            "modelo_2": self._modelos.criar_modelo_2(),
            "modelo_3": self._modelos.criar_modelo_3()
        }

        resultados = {}

        for nome_modelo, modelo in modelos.items():

            print("=" * 60)
            print(f"Treinando {nome_modelo}")
            print("=" * 60)

            treinador = TreinadorCNN(modelo)

            historico = treinador.treinar(
                imagens_treinamento=self._imagens_treinamento,
                rotulos_treinamento=self._rotulos_treinamento,
                nome_modelo=nome_modelo
            )

            avaliador = AvaliadorCNN(
                modelo=modelo,
                nomes_classes=self._nomes_classes
            )

            resultado = avaliador.avaliar(
                imagens_teste=self._imagens_teste,
                rotulos_teste=self._rotulos_teste
            )

            self._gerador_graficos.gerar_graficos_treinamento(
                historico,
                nome_modelo
            )

            self._gerador_graficos.gerar_matriz_confusao(
                resultado["matriz_confusao"],
                self._nomes_classes,
                nome_modelo
            )

            resultados[nome_modelo] = resultado["acuracia"]

            print(f"Acurácia: {resultado['acuracia']:.4f}")
            print(f"Perda: {resultado['perda']:.4f}")
            print()

        self._gerador_graficos.gerar_grafico_comparacao(
            resultados
        )

        self._mostrar_resultados(resultados)

    @staticmethod
    def _mostrar_resultados(
        resultados: dict[str, float]
    ) -> None:

        print("=" * 60)
        print("RESULTADOS")
        print("=" * 60)

        for nome_modelo, acuracia in resultados.items():

            print(
                f"{nome_modelo:<15} -> {acuracia:.4f}"
            )

        melhor_modelo = max(
            resultados,
            key=resultados.get
        )

        print()

        print(
            f"Melhor modelo: {melhor_modelo}"
        )

        print(
            f"Acurácia: {resultados[melhor_modelo]:.4f}"
        )


def main() -> None:

    aplicacao = AplicacaoCNN()

    aplicacao.executar()


if __name__ == "__main__":
    main()