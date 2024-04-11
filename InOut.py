################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

import numpy as np


class InOut():
    # Zapis tablicy Q do pliku
    def saveQTable(self, qtable):
        print("Saving QTable...")
        with open(r'files_txt/qtable.txt', 'w') as handle:
            for row in qtable:
                for item in row:
                    handle.write("%.1f " % item)
                handle.write("\n")

    # Odczyt tablicy Q z pliku
    def readQTable(self, structure):
        print("Reading QTable...")
        cols = structure.numberOfActions()
        rows = structure.numberOfStates()
        qtable = np.zeros((rows, cols), np.float16)
        i = 0
        try:
            with open(r'files_txt/qtable.txt', 'r') as handle:
                for line in handle:
                    row = line.split()
                    row = [float(item) for item in row]
                    row = np.array(row)
                    qtable[i] = row
                    i += 1
            return qtable
        # Jeżeli nie ma pliku zwracany jest None
        except FileNotFoundError:
            return None

    # Zapis stanu do pliku
    def saveState(self, state):
        with open(r'files_txt/state.txt', 'w') as handle:
            handle.write("%i" % state)

    # Odczyt stanu z pliku
    def readState(self):
        try:
            with open(r'files_txt/state.txt', 'r') as handle:
                for line in handle:
                    state = int(line)
            return state
        # Jeżeli nie ma pliku zwracany jest None
        except FileNotFoundError:
            return None
