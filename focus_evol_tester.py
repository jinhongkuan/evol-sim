import numpy as np 
import matplotlib.pyplot as plt 
from copy import copy 
population = []

# Additive 

# Initialize
pop_size = 10 
state_size = 10
for i in range(pop_size):
    population += [np.zeros(state_size)]

# Iterated evolution
iterations = 500
fitness = lambda x : 1/abs(2-x**2)
express = lambda pop : sum([x * 10 ** -i for i,x in enumerate(pop)])
mutation_rate = 0.02
record = []

for i in range(iterations):
    scores = np.zeros(pop_size)
    for p in range(pop_size):
        pheno = express(population[p])
        scores[p] = fitness(pheno)
    
    scores /= sum(scores)
    
    indices = np.random.choice(a=pop_size, size=pop_size, p=scores, replace=True)
    offsprings = []
    for j in range(pop_size):
        child = copy(population[indices[j]])

        # Additive Mutation
        for k in range(state_size):
            if np.random.uniform(0,1) < mutation_rate:
                if np.random.uniform(0,1) <= 0.5:
                    child[k] += 1 
                else:
                    child[k] -= 1 
        val = express(child) 
        if val < 0:
            child = np.zeros((state_size))
        else:
            for k in range(state_size):
                child[k] = int(val * 10 ** k) % 10 
        offsprings += [child]
    population = offsprings
    average = sum([express(x) for x in population])/pop_size 
    record += [average]

for i in range(pop_size):
    print(express(population[i]))

plt.plot(record)
plt.show()