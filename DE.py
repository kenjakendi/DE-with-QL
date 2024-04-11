################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

from random import choices, randint, random
from typing import List, Callable, Tuple
from Algorithm import Algorithm
import time, random, math
import itertools
import numpy as np
from DEInOut import DEInOut
import cec2017
from cec2017.functions import f1

Genome = List[float]
Population = List[Genome]
FitnessFunc = Callable[[Genome], float]


class DE(Algorithm):
    def __init__(self, iterations=1, mut_type="best"):
        # Inicjalizacja potrzebnych parametrów
        self.io = DEInOut()
        self.upBound = self.io.getUpperBound()
        self.downBound = self.io.getLowerBound()
        self.length = self.io.getLength()
        self.iterations = iterations
        self.mut_type = mut_type
        self.success = 0

    def chooseRandomAction(self):
        # Wybór losowej akcji
        return randint(0, self.io.numberOfActions()-1)

    def numberOfStates(self) -> int:
        # Ilość wszystkich możliwych stanów
        return self.io.numberOfStates()

    def getFirstState(self):
        # Początkowy stan
        return self.io.getState([0.0, 0])

    def numberOfActions(self) -> int:
        # Ilość wszystkich możliwych akcji
        return self.io.numberOfActions()

    def readAction(self, index: int):
        # Ustawienie parametrów algorytmu na podstawie indeksu akcji
        action = self.io.getAction(index)
        self.size = action[0]
        self.mode = "bin" if action[1] == 1 else "exp"
        self.Pcr = action[2]
        self.F = action[3]

    def generateGenome(self) -> Genome:
        # Generacja pojedynczego losowego genomu
        return [random.uniform(self.downBound, self.upBound) for _ in range(self.length)]

    def generatePopulation(self, size: int) -> Population:
        # Generacja populacji losowych genomów o określonym rozmiarze
        # albo wczytanie określonej populacji jeżeli istnieje zapisany plik
        population = self.io.readPopulation()
        if population is not None:
            diff = len(population) - size
            if diff > 0:
                for i in range(diff):
                    population.pop()
                return population
            elif diff < 0:
                to_add = random.choices(population, k=abs(diff))
                population += to_add
                return population
            else:
                return population
        return [self.generateGenome() for _ in range(size)]

    def selectRand(self, population: Population, number_of_units: int = 1) -> Population:
        # Wybór określonej liczby losowych genomów z populacji
        return random.sample(population, number_of_units)

    def selectBest(self, population: Population, fitness_func: FitnessFunc, number_of_units: int = 1) -> Population:
        # Wybór określonej liczby najlepszych genomów z populacji bazując na funkcji celu
        popul = sorted(population, key=lambda genome: fitness_func(genome), reverse=False)
        return popul[0:number_of_units]

    def selectLast(self, population: Population, fitness_func: FitnessFunc, number_of_units: int = 1) -> Population:
        # Wybór określonej liczby najgorszych genomów z populacji bazując na funkcji celu
        length = len(population)
        popul = sorted(population, key=lambda genome: fitness_func(genome), reverse=False)
        return popul[(length-number_of_units):length]

    def crossoverEXP(self, original: Genome, mutant: Genome, Pcr: float) -> Genome:
        # Krzyżowanie wykładnicze
        if len(original) != len(mutant):
            raise ValueError("different lengths")
        length = len(original)
        if length < 2:
            return original, mutant
        offspring = []
        for i in range(length):
            if random.uniform(0, 1) < Pcr:
                offspring.append(mutant[i])
            else:
                offspring.append(original[i])
        return offspring

    def crossoverBIN(self, original: Genome, mutant: Genome, Pcr: float) -> Genome:
        # Krzyżowanie dwumianowe
        length = len(original)
        index = randint(0, length-1)
        cloned = 0
        offspring = original.copy()
        while cloned < length:
            if random.uniform(0, 1) < Pcr:
                offspring[index] = mutant[index]
                cloned += 1
                index += 1
                if index == length:
                    index = 0
            else:
                return offspring
        return offspring

    def mutation(self, original: Genome, genome1: Genome, genome2: Genome, F: float) -> Genome:
        # Generowanie mutanta
        length = len(genome1)
        mutant = []
        for i in range(length):
            mutant.append(original[i] + F * (genome1[i] - genome2[i]))
        return mutant

    def mutationLocalToBest(self, local: Genome, best: Genome, genome1: Genome, genome2: Genome, F: float) -> Genome:
        # Mutacja lokalnego osobnika na podstawie najlepszego
        length = len(genome1)
        mutant = []
        for i in range(length):
            mutant.append(local[i] + F * (best[i] - local[i]) + F * (genome1[i] - genome2[i]))
        return mutant

    def checkBounds(self, mutant: Genome) -> Genome:
        # Sprawdzenie ograniczeń wartości genów
        for i in range(len(mutant)):
            if self.upBound > mutant[i] > self.downBound:
                pass
            elif mutant[i] > self.upBound:
                mutant[i] = self.upBound
            else:
                mutant[i] = self.downBound
        return mutant

    def calculateDistance(self, population: Population) -> float:
        # Obliczenie średniego dystansu między osobnikami populacji
        total_dist = 0
        pairs = 0
        for p, q in itertools.combinations(population, 2):
            total_dist += math.dist(p, q)
            pairs += 1
        dist = total_dist/pairs
        return dist

    def calculateReward(self, state: List[int]) -> int:
        # Obliczenie nagrody dla algorytmu QL
        reward = 0
        prev = self.io.readState()
        if prev is None:
            reward = 2
        else:
            if state[0] < prev[0]:
                reward += 0
            if state[1] > prev[1]:
                reward += 1
        return reward

    def checkQuality(self, genome: Genome, fitness_func: FitnessFunc) -> float:
        # Sprawdzenie wartości funkcji celu dla genomu
        return fitness_func(genome)

    def successRate(self, population: Population) -> int:
        # Obliczenie procentu sukcesów mutacji
        rate = self.success / (int(len(population))*self.iterations) * 100
        self.success = 0
        return rate

    def run(self, index: int, fitness_func: FitnessFunc = f1, budget=100) -> Tuple[int, int]:
        # Uruchomienie kolejnych etapów algorytmu DE
        self.readAction(index)
        population = self.generatePopulation(self.size)
        cross_func = self.crossoverEXP if self.mode == "exp" else self.crossoverBIN

        for _ in range(self.iterations):
            for _ in range(len(population)):
                budget -= 1
                parent = population[0]
                best = self.selectBest(population, fitness_func)
                rand = self.selectRand(population, 2)
                mutant = self.checkBounds(self.mutationLocalToBest(parent, best[0], rand[0], rand[1], self.F))
                offspring = cross_func(parent, mutant, self.Pcr)
                if self.checkQuality(parent, fitness_func) > self.checkQuality(offspring, fitness_func):
                    population.append(offspring)
                    population.remove(parent)
                    self.success += 1
                else:
                    population.remove(parent)
                    population.append(parent)

            # Posortowanie populacji
            population = sorted(population, key=lambda genome: fitness_func(genome), reverse=False)

            # Zapis wyników:
            # Zapis średniego dystansu
            distance = self.calculateDistance(population)
            print(f"Average distance: {distance}")
            self.io.saveFloatData(distance, 'distance')

            # Zapis procentu sukcesów
            success = self.successRate(population)
            print(f"Success rate: {success}")
            self.io.saveFloatData(success, 'success')

            # Zapis najlepszej otrzymanej wartości funkcji
            best = fitness_func(population[0])
            print(f"Best benchmark function value: {best}")
            self.io.saveFloatData(best, 'best')

            # Zapis najgorszej otrzymanej wartości funkcji
            worst = fitness_func(population[-1])
            print(f"Worst benchmark function value: {worst}")
            self.io.saveFloatData(worst, 'worst')

        # Przygotowanie wartości dla algorytmu QL
        state = [distance, success]
        reward = self.calculateReward(state)
        self.io.saveState(state)
        self.io.savePopulation(population)
        return self.io.getState(state), reward, budget
