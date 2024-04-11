################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

from random import choices, randint, random
import itertools
import random
import math
import numpy as np


class DEInOut():
    def __init__(self, length=10):
        # Inicjalizacja
        self.genBounds = [100, 0]
        self.actionCombo = self.actionCombinations()
        self.actions = self.defineActions()
        self.length = length
        self.stateCombo = self.stateCombinations()
        self.states = self.defineStates()

    def getUpperBound(self):
        # Zwraca górną granicę dla genu
        return self.genBounds[0]

    def getLowerBound(self):
        # Zwraca dolną granicę dla genu
        return self.genBounds[1]

    def getLength(self):
        # Zwraca długość wektora / wymiar problemu
        return self.length

    def savePopulation(self, population):
        # Zapis populacji do pliku
        self.clearFile("files_txt/population.txt")
        with open(r'files_txt/population.txt', 'w') as handle:
            for genome in population:
                for gene in genome:
                    handle.write("%.24f " % gene)
                handle.write("\n")

    def readPopulation(self):
        # Odczytanie populacji z pliku
        population = []
        try:
            with open(r'files_txt/population.txt', 'r') as handle:
                for line in handle:
                    genome = line.split(" ")
                    genome = [float(gene) for gene in genome if gene.strip()]
                    population.append(genome)
                return population
        # Jeżeli nie ma pliku zwrana None
        except FileNotFoundError:
            return None

    def clearFile(self, file):
        # Czyszczenie pliku
        try:
            with open(file, 'r+') as file:
                file.truncate(0)
        except FileNotFoundError:
            return 0

    def actionCombinations(self):
        # Zwraca listę list wszystkich możliwych kombionacji parametrów
        probability = np.arange(0, 1.1, 0.1).tolist()

        prob_scale = [[p, q] for p, q in itertools.permutations(probability, 2)]
        prob_scale += [[p, p] for p in probability]
        cros_prob_scale = []
        for i in range(2):
            for [p, q] in prob_scale:
                cros_prob_scale.append([i, p, q])

        pop_cros_prob_scale = []
        for k in range(100, 201):
            for [c, p, q] in cros_prob_scale:
                pop_cros_prob_scale.append([k, i, round(p, 2), round(q, 2)])

        return pop_cros_prob_scale

    def defineActions(self):
        # Zwraca słownik możliwych kombinacji parametrów i ich indeksów akcji
        actions = {}
        for index in range(0, len(self.actionCombo)):
            actions[index] = self.actionCombo[index]
        return actions

    def numberOfActions(self):
        # Zwraca ilość wszystkich możliwych akcji
        return len(self.actions)

    def stateCombinations(self):
        # Zwraca listę list wszystkich możliwych stanów
        dist_succ = []
        diff = (self.getUpperBound() - self.getLowerBound()) * (self.getUpperBound() - self.getLowerBound())
        distance = np.arange(0, math.sqrt(self.length * diff) + 1, 1).tolist()
        success = np.arange(0, 101, 1).tolist()
        for i in distance:
            for j in success:
                dist_succ.append(str(round(i, 0)) + str(" ") + str(round(j, 0)))
        return dist_succ

    def defineStates(self):
        # Zwraca słownik możliwych stanów i ich indeksów
        index = 0
        states = {}
        for state in self.stateCombo:
            states[state] = index
            index += 1
        return states

    def numberOfStates(self):
        # Zwraca ilość wszystkich możliwych stanów
        return len(self.stateCombo)

    def getState(self, state):
        # Zwraca stan odpowiedni do podanego indeksu
        key = str(round(state[0], 0)) + str(" ") + str(int(round(state[1], 0)))
        return self.states[key]

    def getAction(self, index):
        # Odczytanie indexu akcji i zwrócenie konkretnych parametrów algorytmu
        return self.actions[index]

    def saveState(self, state):
        # Zapis stanu do pliku
        self.clearFile("files_txt/DEstate.txt")
        with open(r'files_txt/DEstate.txt', 'w') as handle:
            handle.write("%s " % state[0])
            handle.write("%i " % state[1])

    def readState(self):
        # Odczytanie stanu z pliku
        try:
            with open(r'files_txt/DEstate.txt', 'r') as handle:
                for line in handle:
                    state = line.split()
                    state = [float(st) for st in state if st.strip()]
                return state
        # Jeżeli nie ma pliku zwraca None
        except FileNotFoundError:
            return None

    def saveFloatData(self, data, file_name):
        # Dopis danych w postaci Float na koniec pliku
        # Jeżeli nie ma pliku, tworzony jest nowy
        with open(rf'files_txt/{file_name}.txt', 'a') as handle:
            handle.write("%.2f\n" % data)
