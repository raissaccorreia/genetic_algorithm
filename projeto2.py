import numpy as np
import random, operator, pandas as pd
import matplotlib.pyplot as plt
import time 

#Gene: a city
#Individual: a route satisfying the conditions above
#Population: a collection of possible routes
#Parents: two routes that are combined to create a new route
#Mating pool: a collection of parents that are used to create our next population
#Fitness: a function that tells us how good each route is (in our case, how short the distance is)
#Mutation: a way to introduce variation in our population by randomly swapping two cities in a route
#Elitism: a way to carry the best individuals into the next generation

#The cities are simply our (x, y) coordinates. 
#Within the City class, we add a distance calculation (Pythagorean distance)
#and a cleaner way to output the cities as coordinates with "__repr__" 
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

#In our case, the fitness as the inverse of the route distance.
#The idea is to minimize route distance, so a larger fitness score is better. 
#We need to start and end at the same place, so this extra calculation is accounted in the distance calculation.
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#We now can make our initial population. 
#We create a function that produces routes that satisfy our conditions
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#Looping through the createRoute function until we have as many routes as we want for our population.
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#We make use of Fitness to rank each individual in the population.
#Our output will be an ordered list with the route IDs and each associated fitness score.
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()        
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#We’ll use the output from rankRoutes to determine which routes to select in our selection function.
#We set up the roulette wheel by calculating a relative fitness weight for each individual. 
#We compare a randomly drawn number to these weights to select our mating pool.
#We’ll also want to hold on to our best routes, so we introduce elitism in sequence.
#The selection function returns a list of route IDs, which we can use to create the mating pool
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Alternative function for selection: besides the elite, selects individuals randomly 
def selectionAlt(popRanked, eliteSize):
	selectionResults = []

	for i in range(0, eliteSize):
		selectionResults.append(popRanked[i][0])

	popIndexes = []
	for i in range(0, len(popRanked)):
		popIndexes.append(i)

	for i in range(0, len(popRanked) - eliteSize):
		j = random.choice(popIndexes)
		selectionResults.append(popRanked[j][0])

	return selectionResults

#Extracting the selected individuals from our population.
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Now we create the next generation in a process called crossover. 
#In TSP we need to include all locations exactly one time. So we used ordered crossover. 
#We randomly select a subset of the first parent string, and then fill the remainder of the route
#with the genes from the second parent in the order in which they appear, without duplicating any genes
#in the selected subset from the first parent.
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#Generalizing this to create our offspring population. 
#We use elitism to retain the best routes from the current population. 
#The breed function fills out the rest of the next generation.
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#alternative function for breeding using cycle crossover(CX)
#as described in https://arxiv.org/pdf/1203.3097.pdf 
def breedAlt(parent1, parent2):
    child = [None] * (len(parent1))
    child[0] = parent1[0]
    i = 0

    while(parent2[i] not in child):
        j = parent1.index(parent2[i])
        child[j] = parent1[j]
        i = j
    
    for i in range(len(child)):
        if child[i] == None:
            child[i] = parent2[i]
    
    return child 
    	
#applying to the whole population
def breedPopulationAlt(matingpool, eliteSize):
	children = []
	length = len(matingpool) - eliteSize
	pool = random.sample(matingpool, len(matingpool))

	for i in range(0, eliteSize):
		children.append(matingpool[i])

	for i in range(0, length):
		child = breedAlt(pool[i], pool[len(matingpool)-i-1])
		children.append(child)
	return children 

#TSP has a special consideration when it comes to mutation. 
#We can’t drop cities. Instead, we use swap mutation, with the specified probability, defined by the user
#two cities will swap places in our route. We’ll do this for one individual in our mutate function
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#Next, we can extend the mutate function to run through the new population.
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#alternative function for mutating instead of swapping cities, 
#we swap sequences of cities by dividing an individual in half
def mutateAlt(individual, mutationRate):
	i = 1
	j = len(individual)//2
	mutIndividual = individual
	if(random.random() < mutationRate):
		while(i != len(individual)//2):
			mutIndividual[i] = individual[j]
			mutIndividual[j] = individual[i]
			i+=1
			j+=1
	return mutIndividual 

def mutatePopulationAlt(population, mutationRate):
	mutatedPop = []

	for ind in range(0, len(population)):
		mutatedInd = mutateAlt(population[ind], mutationRate)
		mutatedPop.append(mutatedInd)
	return mutatedPop 

#To create the next generation. We rank the routes in the current generation using rankRoutes. 
#We then determine our potential parents by running the selection function
#Which allows us to create the mating pool using the matingPool function. 
#We then create our new generation using the breedPopulation function and then applying
#mutation using the mutatePopulation function.
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration
#Alternative version
def nextGenerationAlt(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selectionAlt(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulationAlt(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#No plot version
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations, alternative_choice):
    #getting inicial population
    pop = initialPopulation(popSize, population)

    #showing zero generation results
    print("Initial best distance: " + str(1 / rankRoutes(pop)[0][1]))
    print("Initial worst distance: " + str(1 / rankRoutes(pop)[popSize-1][1]))
    soma = 0
    for i in range (popSize-1):
        soma += (1 / rankRoutes(pop)[i][1])
    soma = soma/popSize-1       
    print("Initial average distance: " + str(soma))

    #initial values
    progressBest = []
    progressWorst = []    
    progressBest.append(1 / rankRoutes(pop)[0][1])
    progressWorst.append(1 / rankRoutes(pop)[popSize-1][1])    
    t_start = time.time()

    #alternative choice
    if alternative_choice == 'y':
        #iterating for all generations
        for i in range(0, generations):
            #getting next population
            pop = nextGenerationAlt(pop, eliteSize, mutationRate)
            #append the best and the worst
            progressBest.append(1 / rankRoutes(pop)[0][1])
            progressWorst.append(1 / rankRoutes(pop)[popSize-1][1])
    #the original version
    else:
        for i in range(0, generations):
            pop = nextGeneration(pop, eliteSize, mutationRate)
            progressBest.append(1 / rankRoutes(pop)[0][1])
            progressWorst.append(1 / rankRoutes(pop)[popSize-1][1])

    #final results
    print("Final best distance: " + str(1 / rankRoutes(pop)[0][1]))
    print("Final worst distance: " + str(1 / rankRoutes(pop)[popSize-1][1]))    
    soma = 0
    for i in range (popSize-1):
       soma += (1 / rankRoutes(pop)[i][1])      
    soma = soma/popSize  
    print("Final average distance: " + str(soma))

    #calculating execution time
    t_end = time.time()
    t = t_end - t_start 
    print("Time elapsed in seconds: ", t) 

    #plotting
    plt.plot(progressBest)
    plt.plot(progressWorst)    
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show() 

#now the execution!!!

print("Number of cities (at most 50):")
numCities = int(input())
if numCities > 50:
    numCities = 50

print("Population size (at most 150):")
numPopulation = int(input())
if numPopulation > 150:
    numPopulation = 150

#elite size can't be larger than population size 
print("Elite size (at most 35):")
eliteNumber = int(input())
if eliteNumber > numPopulation:
    eliteNumber = numPopulation

print("Mutation rate (from 0.001 to 0.1):")
rateMutation = float(input())
if rateMutation > 0.1:
    rateMutation = 0.1
if rateMutation < 0.001:
    rateMutation = 0.001

print("Number of Generations(at most 1000):")
numGenerations = int(input())
if numGenerations > 1000:
    numGenerations = 1000

print("Do you want the alternative version, modified by us? y or n")
alternative_choice = input()

print("Executing, this will take at most 2 minutes...")
print("If it takes longer Ctrl+C to interrupt!")

#Generating the random (x,y) pairs that represent a city for each in numCities
cityList = []
for i in range(0,numCities):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

#running the version with plot
geneticAlgorithmPlot(population=cityList, popSize=numPopulation, eliteSize=eliteNumber, mutationRate=rateMutation,
 generations=numGenerations, alternative_choice=alternative_choice)
