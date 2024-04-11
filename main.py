################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

from QL import QL
from DE import DE
import cec2017
from cec2017.functions import f4,f6,f10,f12,f20

if __name__ == "__main__":
    # Uruchomienie algorytmu QL z DE

    algorithm = DE()
    ql = QL(algorithm, f4, 10000)
    ql.run()

    # Uruchomienie algorytmu DE z przykładowymi parametrami

    # de = DE(iterations=100)
    # print(de.io.getAction(185))
    # de.run(185, f4, 10000)
