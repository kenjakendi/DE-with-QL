################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

from abc import ABC, abstractmethod
from Exceptions import NotImplementedError


class Algorithm(ABC):

    # Wykonanie algorytmu, zwraca następny stan i nagrodę
    @abstractmethod
    def run(self):
        raise NotImplementedError
        # return next_state, reward, budget_left

    # Metoda zwracająca ilość możliwych stanów
    @abstractmethod
    def numberOfStates(self):
        raise NotImplementedError

    # Metoda zwracająca ilość możliwych akcji
    @abstractmethod
    def numberOfActions(self):
        raise NotImplementedError

    # Metoda pozwalająca na wybranie losowej z możliwych akcji
    @abstractmethod
    def chooseRandomAction(self):
        raise NotImplementedError

    # Metoda zwracająca stan początkowy algorytmu
    @abstractmethod
    def getFirstState(self):
        raise NotImplementedError
