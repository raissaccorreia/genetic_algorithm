# Genetic Algorithm Project

Project for "Introduction to AI, UNICAMP discipline, MC906
Profª Esther

Will take around 20 seconds to run:
python 3 projeto2.py

requirements:
pip3 install numpy
pip3 install matplotlib
pip3 install pandas

## Applying Concepts

Gene: a city, represented as (x, y) coordinates
Individual: the “chromosome”, a single route satisfying the conditions above
Population: a collection of possible routes, a collection of individuals
Parents: two routes that are combined to create a new route
Mating Pool: a collection of parents that are used to create our next population
Fitness: a function that tells us how good each route is, how short the distance is, in this case.
Mutation: a way to introduce variation in our population by randomly swapping two cities in a route
Elitism: a way to carry the best individuals into the next 

## The City Class

We first create a City class that allow us to create and handle our cities. These are simply their (x, y) coordinates. Within the City class, we add a distance calculation, the pythagorean theorem distance, and a cleaner way to output the cities as coordinates with “__repr__”

## The “Fitness” Class 

In this case, we’ll treat the fitness as the inverse of the route distance. The idea is to minimize route distance, so a larger fitness score is better. And also obeying the rule, that we need to start and end at the same place.
Creating the Route
Now we can make our initial population, the first generation. To do so, we need a way to create a function that produces routes that satisfy our conditions.

## Initial Population Function

This one produces one individual, to create a full population, we’ll do that in our next function. This is as simple as looping through the createRoute function until we have as many routes as we want for our population.
“Rank Routes” Survival of the Fittest
To simulate our natural selection, we can make use of Fitness to rank each individual in the population. Our output will be an ordered list with the route IDs and each associated fitness score. All this done by the rankRoutes function.

## The Selection

First, we’ll use the output from rankRoutes to determine which routes to select in our selection function. Then, we set up the roulette wheel by calculating a relative fitness weight for each individual. After that, we compare a randomly drawn number to these weights to select our mating pool.
We’ll also want to hold on to our best routes, so we introduce elitism. Ultimately, the selection function returns a list of route IDs, which we can use to create the mating pool, in the matingPool function of course.

## Alternative Selection

Alternative function for selection: besides the elite, selects individuals randomly.

## Mating Pool

Now that we have the IDs of the routes that will make up our mating pool from the selection function, we can create the mating pool. We’re simply extracting the selected individuals from our population.]

## Breed
With our mating pool created, we can create the next generation in a process called crossover, the breeding. 
In the TSP, we need to include all locations exactly one time. To abide by this rule, we can use a special breeding function called ordered crossover. In ordered crossover, we randomly select a subset of the first parent string and then fill the remainder of the route with the genes from the second parent in the order in which they appear, without duplicating any genes in the selected subset from the first parent.

## Create Breed Population
Next, we’ll generalize this to create our offspring population. We use elitism to retain the best routes from the current population. Then, we use the breed function to fill out the rest of the next generation.

## Alternative Breeding

 This alternative function for breeding is using cycle crossover as described in this article: https://arxiv.org/pdf/1203.3097.pdf 
 
## Mutate

The TSP has a special consideration when it comes to mutation. We need to abide by our rules, we can’t drop cities. Instead, we’ll use swap mutation. This means that, with specified low probability, two cities will swap places in our route. We’ll do this for one individual in our mutate function.
To Mutate the whole population, we just  extend the mutate function to run through the new population, in the “mutatePopulation” function.

## Alternative Mutation

Instead of swapping cities, we swap sequences of cities by dividing an individual in half, and apply to all population again.
Creating the Next Generation
Pulling these pieces together to create a function that produces a new generation. First, we rank the routes in the current generation using rankRoutes. We then determine our potential parents by running the selection function, which allows us to create the mating pool using the matingPool function. Finally, we then create our new generation using the breedPopulation function and then applying mutation using the mutatePopulation function.

## The Genetic Algorithm Function

All we need to do is create the initial population, and then we can loop through as many generations as we desire. Of course we also want to see the best route and how much we’ve improved, so we capture the initial distance(the inverse of the fitness), the final distance and the best route.

## Parameters

Number of Cities: at most 50
Population Size: at most 150
Elite Population Size: at most 30 or Population Size
Mutation Rate: beetwen 0.1% to 10%
Number of Generations: at most 1000
Our Alternative Version: 'y' or 'n'

## Generating Random City List

We generate a numCities size list, where a city is defined by its X and Y position, we randomize the X and Y floating number position in the range [0;1) and multiply by 200, to give a random list of X and Y positions.

## Executing and Plotting

At first the use inputs the number of cities, limited to 50, because of computation cost viability. Then the population size that we limited to 150. Followed by elite size with a maximum of 35. Mutation rate between 0.1 to 10%, and the last two options the number of generations limited to a thousand and if you want to run the alternative version of mutation, breed and selection that we made.
In execution we print in terminal the minimum, average and maximum distance at the first and last generation, after the execution we plot using matplotlib the minimum and maximum distance in every generation.
