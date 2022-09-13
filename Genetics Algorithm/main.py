from typing import List
import random
import math
import string


class Individual:
    def __init__(self, chromosome: List[str], score: int) -> None:
        self.chromosome = chromosome
        self.score = score



def initial_population(chromosome: List[str], 
                       population: List[str], 
                       number_of_population: int, 
                       target: str) -> None:
    
    for i in range(number_of_population):
        
        for j in range(len(list(target))):
            chromosome.append(random.choice(string.printable))

        individual = Individual(chromosome, 0) 
        population.append(individual)


if __name__ == "__main__":
    initial_chromosome = []
    population = []
    initial_population(
        chromosome=initial_chromosome,
        population=population,
        number_of_population=200,
        target='Aku sayang ibu'
    )

    for i in population:
        print(i.initial_chromosome)



