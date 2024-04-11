################################################
# Autorzy: Julita Wasilewska, Michał Laskowski #
################################################

# Wyjątek wywoływany gdy nie zostanie zostanie zaimplementowana wymagana metoda algorytmu
class NotImplementedError(Exception):
    def __str__(self):
        return "Required method in algorithm is not implemented"
