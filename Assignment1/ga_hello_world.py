import random
import math

# Individual class implementation


class Individual:
    def __init__(self, chromosome: str):
        self.chromosome = chromosome
        self.fitness = 0

    def __repr__(self):
        return "Individual(" + self.genoToPhenotype() + ", fitness=" + str(self.fitness) + ")"

    def getChromosome(self):
        return self.chromosome

    def setChromosome(self, chromosome):
        self.chromosome = chromosome

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def genoToPhenotype(self):
        return ("".join(self.chromosome))


"""
@(#)HeapSortAlgorithm.java   1.0 95/06/23 Jason Harrison

Copyright (c) 1995 University of British Columbia

Permission to use, copy, modify, and distribute this software
and its documentation for NON-COMMERCIAL purposes and without
fee is hereby granted provided that this copyright notice
appears in all copies. Please refer to the file "copyright.html"
for further important copyright and licensing information.

UBC MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF
THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE, OR NON-INFRINGEMENT. UBC SHALL NOT BE LIABLE FOR
ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR
DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

A heap sort demonstration algorithm
SortAlgorithm.java, Thu Oct 27 10:32:35 1994

Modified by Steven de Jong for Genetic Algorithms.

Modified by Jo Stevens for practical session.

Rewritten by Michal Pavlíček to Python (with help from GitHub Copilot).

@author Jason Harrison@cs.ubc.ca
@version 1.0, 23 Jun 1995

@author Steven de Jong
@version 1.1, 08 Oct 2004

@author Jo Stevens
@version 1.2, 14 Nov 2008

@author Michal Pavlíček
@version 1.3, 19 Nov 2022
"""


class HeapSort:
    def sort(self, i: list[Individual]):
        N = len(i)

        k = int(N / 2)
        while k > 0:
            self.downheap(i, k, N)
            k -= 1

        while N > 1:
            T = i[0]
            i[0] = i[N - 1]
            i[N - 1] = T

            N = N - 1
            self.downheap(i, 1, N)

    def downheap(self, i, k, N):
        T = i[k - 1]

        while k <= N / 2:
            j = k + k
            if j < N and i[j - 1].getFitness() > i[j].getFitness():
                j += 1

            if T.getFitness() <= i[j - 1].getFitness():
                break
            else:
                i[k - 1] = i[j - 1]
                k = j

        i[k - 1] = T

# Constant variables


TARGET = "HELLO WORLD"
POPULATION_SIZE = 100
alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
alphabet.append(' ')

# Helper functions


def create_initial_population(population_size) -> list[Individual]:
    # we initialize the population with random characters
    population = []
    for i in range(population_size):
        tempChromosome = []
        for j in range(len(TARGET)):
            tempChromosome.append(
                alphabet[random.randint(0, len(alphabet) - 1)])
        population.append(Individual(tempChromosome))

    return population


def print_population(population: list[Individual]):
    for i in population:
        print(i.genoToPhenotype())


def assign_fitness(population: list[Individual]):
    for individual in population:
        fitness = 0
        chromosome = individual.getChromosome()

        for index in range(len(chromosome)):
            if chromosome[index] == TARGET[index]:
                fitness += 1

        individual.setFitness(fitness)


def sort(population: list[Individual]):
    heapSort = HeapSort()
    heapSort.sort(population)

    return population

# Selection functions


def elitist_selection(population: list[Individual], percentage: int):
    if percentage > 100:
        percentage = 100
    # select the best *percentage* individuals
    return population[:int(len(population) / 100 * percentage)]


def roulette_selection(population: list[Individual], percentage: int):
    if percentage > 100:
        percentage = 100
    # Power is used to increase the probability of the fittest individuals
    power = 10
    new_individuals = []

    for _ in range(int(len(population) / 100 * percentage)):
        # Calculates the total fitness in population
        total_fitness = 0

        for individual in population:
            total_fitness += individual.getFitness() ** power

        # Select a random number from 0 to total_fitness
        rand = random.randrange(0, total_fitness)

        # The code below adds the fitness of each individual to the sum until it is greater than the random number
        # Once the sum is greater than the random number, the current individual is selected
        selected_individual = None
        temp_sum = 0
        for individual in population:
            if temp_sum >= rand:
                selected_individual = individual
                break
            temp_sum += individual.getFitness() ** power
        else:
            selected_individual = population[-1]

        new_individuals.append(selected_individual)
        population.remove(selected_individual)

    return new_individuals


def tournament_selection(population: list[Individual], percentage: int):
    if percentage > 100:
        percentage = 100
    # Generates tournament size (= k) between 2 and len(population) / 2
    tournament_size = random.randint(2, int(len(population) / 2))

    new_individuals = []
    for _ in range(int(len(population) / 100 * percentage)):
        # Selects k random individuals from population
        sample = random.sample(population, tournament_size)
        # Chooses the best individual from the sample
        best = max(sample, key=lambda x: x.getFitness())

        # Adds the best individual to the new population and removes it from the old one
        new_individuals.append(best)
        population.remove(best)

    return new_individuals


def middle_crossover(population: list[Individual], num_crossovers: int, population_size: int):
    new_population = []

    for individual1 in population:
        for individual2 in population:
            chromosome1 = []
            chromosome2 = []

            alternate = True
            step = math.ceil(len(TARGET) / (num_crossovers+1))

            for i in range(0, len(TARGET), step):
                added_chromosome_1 = individual1.getChromosome()[i:i+step]
                added_chromosome_2 = individual2.getChromosome()[i:i+step]

                if alternate:
                    chromosome1.append(added_chromosome_1)
                    chromosome2.append(added_chromosome_2)
                else:
                    chromosome1.append(added_chromosome_2)
                    chromosome2.append(added_chromosome_1)
                alternate = not alternate

            res_1 = chromosome1[0]
            for i in range(1, len(chromosome1)):
                res_1 += chromosome1[i]

            res_2 = chromosome2[0]
            for i in range(1, len(chromosome2)):
                res_2 += chromosome2[i]

            offspring1 = Individual(res_1)
            offspring2 = Individual(res_2)
            new_population.append(offspring1)
            new_population.append(offspring2)

    return random.sample(new_population, population_size)

# Mutation functions


def mutation(population: list[Individual], chance: int):
    if chance > 50:
        chance = 50

    for individual in population:
        for index in range(len(individual.getChromosome())):
            if random.randint(0, 100) <= chance:
                individual.getChromosome()[index] = alphabet[random.randint(
                    0, len(alphabet) - 1)]

    return population


def run_ga(population: list[Individual], selection_function, success_rate: int, num_crossovers: int, mutation_chance: int, population_size: int):
    generation = 0
    while True:
        # print("Generation: " + str(generation))

        assign_fitness(population)

        population = sort(population)

        # print(population[0])

        if population[0].getFitness() == len(TARGET):
            print(generation)
            return generation

        population = selection_function(population, success_rate)

        population = middle_crossover(
            population, num_crossovers, population_size)

        population = mutation(population, mutation_chance)

        generation += 1


population = create_initial_population(100)

run_ga(population, elitist_selection, success_rate=15,
       num_crossovers=1, mutation_chance=5, population_size=100)
