# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# Método que representa gráficamente el coste acumulado en cada entrenamiento y
# la tasa de acierto que la red obtiene al ejecutar el conjunto de test
def costHitsGraphica(cost, hits):
    plt.figure()
    plt.subplot(211)
    plt.plot(cost)
    plt.ylabel("Cost")
    plt.title("Learning")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(hits, 'g')
    plt.xlabel("Iterations")
    plt.ylabel("Percentage of hits")
    plt.grid(True)

    plt.show()