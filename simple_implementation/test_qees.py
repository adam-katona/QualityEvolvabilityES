

import torch
import numpy as np
import matplotlib.pyplot as plt

import qees

# In this file we demonstrate the usage of the simple implementation of Quality Evolvability ES on a simple task

# Simple task:
# The fitness is that the sum of the parameters need to be close to 10
# Behaviour is described by the sin of the parameters
# In order to maximize evolvability, the algorithm needs to select parameters that have maximum bahaviour variance when perturbed.
# This is tha case when the values are close to where sin have the largest slope (n*PI)
def evaluate_fun(params):
    result = {
        "fitness" : -1*np.abs(np.sum(params)-10), # the parameters need to sum to 10
        "bc" : list(np.sin(params))  # to maximize evolvability, the algorithm need select values where sin have large slope (n*PI)
    }
    return result


config = {
    "ES_popsize" : 50,
    "ES_sigma" : 0.01,
    "ES_lr" : 0.1,
}
NUM_GENERATIONS = 500

initial_theta = np.random.randn(5)

fitnesses = []
evolvabilities = []
all_params = []

algo = qees.QualityEvolvabilityES(config,initial_theta)
for gen in range(NUM_GENERATIONS):
    
    # take a step with the algorithm
    pop = algo.ask()
    results = [evaluate_fun(individual) for individual in pop]
    algo.tell(results)
    
    # calculate values for logging
    fitnesses.append(np.mean(np.array([res["fitness"] for res in results])))
    behaviour_characterizations = np.array([res["bc"] for res in results])
    evolvabilities.append(np.sum((behaviour_characterizations - behaviour_characterizations.mean(axis=0)) ** 2))
    all_params.append(algo.theta.clone().detach().cpu().numpy())

    
    
solution = algo.theta
print("Solution found: ",solution.detach().cpu().numpy())
all_params = np.array(all_params)

plt.plot(fitnesses)
plt.xlabel("generations")
plt.ylabel("finess")
plt.savefig("fitnesses.png")

plt.clf()
plt.plot(evolvabilities)
plt.xlabel("generations")
plt.ylabel("evolvability")
plt.savefig("evolvabilities.png")

plt.clf()
for i in range(5):
    plt.plot(all_params[:,i])
plt.xlabel("generations")
plt.ylabel("parameters")
plt.savefig("solutions.png")
