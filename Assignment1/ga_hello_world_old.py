import random


class Individual:
    chromosome = ""
    fitness = 0

    def __init__(self, chromosome):
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

    def clone(self):
        chromClone = ""
        for i in range(len(self.chromosome)):
            chromClone[i] = self.chromosome[i]

        return Individual(chromClone)


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

Rewritten by Michal Pavlíček to Python.

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
    def sort(self, i):
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


"""
Some very basic stuff to get you started. It shows basically how each
chromosome is built.

@ author Jo Stevens
@ version 1.0, 14 Nov 2008

@ author Alard Roebroeck
@ version 1.1, 12 Dec 2012

@ author Michal Pavlíček
@ version 1.2, 19 Nov 2022
"""

TARGET = "HELLO WORLD"
alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
alphabet.append(' ')

# What does your population look like?


def create_initial_population():
    # we initialize the population with random characters
    population = []
    for i in range(100):
        tempChromosome = []
        for j in range(len(TARGET)):
            tempChromosome.append(
                alphabet[random.randint(0, len(alphabet) - 1)])
        population.append(Individual(tempChromosome))

    return population


def print_population(population):
    for i in population:
        print(i.genoToPhenotype())


def assign_fitness(population):
    for individual in population:
        fitness = 0
        chromosome = individual.getChromosome()

        for index in range(len(chromosome)):
            if chromosome[index] == TARGET[index]:
                fitness += 1

        individual.setFitness(fitness)


def elitist_selection(population):
    # sort the population
    heapSort = HeapSort()
    heapSort.sort(population)

    # select the best 10 individuals
    print("   * " + str(population[0]))
    return population[:10]


def middle_crossover(population):
    crossover_point = 5
    new_population = []

    for individual1 in population:
        for individual2 in population:
            chromosome1 = individual1.getChromosome(
            )[:crossover_point] + individual2.getChromosome()[crossover_point:]
            chromosome2 = individual2.getChromosome(
            )[:crossover_point] + individual1.getChromosome()[crossover_point:]

            offspring1 = Individual(chromosome1)
            offspring2 = Individual(chromosome2)
            new_population.append(offspring1)
            new_population.append(offspring2)

    return new_population


def mutation(population):
    for individual in population:
        for index in range(len(individual.getChromosome())):
            if random.randint(0, 100) <= 5:
                individual.getChromosome()[index] = alphabet[random.randint(
                    0, len(alphabet) - 1)]

    return population


population = create_initial_population()

for i in range(50):
    print("Generation: " + str(i))

    assign_fitness(population)

    population = elitist_selection(population)

    population = middle_crossover(population)

    for i in population:
        if i.genoToPhenotype() == TARGET:
            print("Found the target!")
            print(i.genoToPhenotype())
            exit()

    mutation(population)
