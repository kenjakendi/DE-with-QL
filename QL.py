################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

from InOut import InOut
import numpy as np
import time
import os


class QL:
    def __init__(self, algorithm, function, budget=25, eps=0.15, gamma=0.1, beta=0.8):
        # Wybór algorytmu
        self.algorithm = algorithm
        self.function = function

        # Wczytanie klasy wejścia-wyjścia
        self.io = InOut()

        # Parametry Q-Learning
        self.budget = budget
        self.eps = eps
        self.gamma = gamma
        self.beta = beta

        # Wczytanie stanu i tabeli Q
        self.state = self.getState()
        self.qtable = self.getQTable()

    def run(self):
        # Główna pętla działania Q-Learning
        i = 1
        while(self.budget > 0):
            os.system('clear')
            print(f"QL {i} iteration:")

            # Wybór akcji
            action = self.calculateAction()
            print(f"Selected action {action} in  state {self.state}")

            # Wykonanie akcji oraz wyznaczenie nagrody
            next_state, reward, self.budget = self.algorithm.run(action, self.function, self.budget)
            print(f"Received reward: {reward}")

            # Aktualizacja tabeli Q
            self.updateQTable(next_state, reward, action)

            # Aktualizacja stanu
            self.state = next_state
            i += 1

            time.sleep(0.1)

        # Zapis tabeli Q i ostatniego stanu do pliku

        # self.io.saveState(self.state)
        # self.io.saveQTable(self.qtable)

    def getQTable(self):
        # Wczytanie tabeli Q z pliku
        qtable = self.io.readQTable(self.algorithm)

        # Jeżeli nie ma zapisanej tabeli Q następuje jej inicjalizacja
        if qtable is None:
            actions = self.algorithm.numberOfActions()
            states = self.algorithm.numberOfStates()
            qtable = np.zeros((states, actions), np.float16)
        return qtable

    def getState(self):
        # Wczytanie stanu z pliku
        state = self.io.readState()

        # Jeżeli nie ma zapisanego stanu następuje jego inicjalizacja
        if state is None:
            state = self.algorithm.getFirstState()
        return state

    def calculateAction(self):
        # Strategia epsilon zachłanna
        # Z prawdopodobieństwem epsilon wybierana jest losowa akcja
        if np.random.uniform() < self.eps:
            action = self.algorithm.chooseRandomAction()

        # Z prawdopodobieństwem 1-epsilon wybierana jest akcja zachłanna
        else:
            max = np.amax(self.qtable[self.state])
            action = np.where(self.qtable[self.state] == max)
            action = np.random.choice(action[0])
        self.eps = self.eps*0.9
        return action

    def updateQTable(self, next_state, reward, action):
        # Aktualizacja tabeli Q zgodnie ze wzorem, beta oznacza rozmiar kroku
        self.qtable[self.state][action] = (1-self.beta)*self.qtable[self.state][action]+self.beta*(reward + self.gamma * max(self.qtable[next_state]))
