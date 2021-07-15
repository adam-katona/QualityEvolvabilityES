import torch
import numpy as np
import functools

# Simple implementation of Quality Evolvability ES
#
# usage:
# algo = QualityEvolvabilityES(config)
# for gen in range(NUM_GENERATIONS):
#     pop = algo.ask()
#     results = [evaluate_fun(individual) for individual in pop]
#     algo.tell(results)
# 
# solution = algo.theta



class QualityEvolvabilityES():

    def __init__(self,config,initial_theta):

        self.config = config
        
        self.required_config_fields = [
            "ES_popsize", # should be divisible by 2, so we can do mirrored sampling
            "ES_sigma",
            "ES_lr",
        ]

        self.last_perturbations = None
        self.current_generation = 0
        self.HALF_POP_SIZE = config["ES_popsize"] // 2

        initial_theta = torch.from_numpy(initial_theta).float()
        self.num_params = initial_theta.shape[0]
        self.theta = torch.nn.Parameter(initial_theta) 
        self.optimizer = torch.optim.Adam([self.theta],lr=self.config["ES_lr"])


    def ask(self):
        if self.last_perturbations is not None:
            raise "Error, are you calling ask() twice without calling tell()?"

        noise = torch.randn(self.HALF_POP_SIZE,self.num_params)
        noise = torch.cat([noise,-noise],dim=0) # mirrored sampling
        self.last_perturbations = noise

        population = [(self.theta + noise[i] * self.config["ES_sigma"]).detach().numpy() for i in range(noise.shape[0])]
        return population


    def tell(self,results):
        if self.last_perturbations is None:
            raise "Error, are you calling tell() without calling ask() first?"

        # For quality evolvability we simultaneously optimize for both excpected fitness and evolvability (variance of behaviour)
        # Fitness is the reward score on whatever task we are trying to solve
        fitnesses = np.array([res["fitness"] for res in results])

        # We define evolvability is the variance of the behaviour of the offspring. (see the paper for why this is, and alternative definitions)
        # In case of ES, the population is a sample of the offspring of a single individual (perturbed copy of the center theta)
        # This is why the variance of the population is the same as the evolvability of the center theta.        
        behaviour_characterizations = np.array([res["bc"] for res in results])
        evolvability = np.sum((behaviour_characterizations - behaviour_characterizations.mean(axis=0)) ** 2,axis=1)

        # do nondominated sorting between the fitness and evolvability objectives
        multi_objective_fitnesses = np.concatenate([fitnesses.reshape(-1,1),evolvability.reshape(-1,1)],axis=1)

        fronts = calculate_pareto_fronts(multi_objective_fitnesses)
        nondomination_rank_dict = fronts_to_nondomination_rank(fronts)
        crowding = calculate_crowding_metrics(multi_objective_fitnesses,fronts)
        non_domiated_sorted_indicies = nondominated_sort(nondomination_rank_dict,crowding)

        # calculate normalized ranks
        # sort is ascending (best is last), we want the gradient to point towards the high fitness and high evolvability, the first index gets the lowest score
        all_ranks = np.linspace(-0.5,0.5,len(non_domiated_sorted_indicies)) 
        perturbation_ranks = np.zeros(len(non_domiated_sorted_indicies))
        perturbation_ranks[non_domiated_sorted_indicies] = all_ranks
        perturbation_ranks = torch.from_numpy(perturbation_ranks).float()

        grad = torch.matmul(perturbation_ranks,self.last_perturbations)  # ES update, calculate the weighted sum of the perturbations
        grad = grad / self.config["ES_popsize"] / self.config["ES_sigma"]

        self.theta.grad = -grad # we are maximizing, but torch optimizer steps in the opposite direction of the gradient, multiply by -1 so we can maximize.
        self.optimizer.step()

        self.current_generation += 1
        self.last_perturbations = None





########################################
## Functions for nondominated sorting ## 
########################################

def dominates(fitnesses_1,fitnesses_2):
    # fitnesses_1 is a array of objectives of solution 1 [objective1, objective2 ...]
    larger_or_equal = fitnesses_1 >= fitnesses_2
    larger = fitnesses_1 > fitnesses_2
    if np.all(larger_or_equal) and np.any(larger):
        return True
    return False


# returns a matrix with shape (pop_size,pop_size) to answer the question does individial i dominates individual j -> domination_matrix[i,j]
# much faster then calling dominates() in 2 for loops
def calculate_domination_matrix(fitnesses):    
    
    pop_size = fitnesses.shape[0]
    num_objectives = fitnesses.shape[1]
    
    # numpy meshgrid does not work if original array is 2d, so we have to build the mesh grid manually
    fitness_grid_x = np.zeros([pop_size,pop_size,num_objectives])
    fitness_grid_y = np.zeros([pop_size,pop_size,num_objectives])
    
    for i in range(pop_size):
        fitness_grid_x[i,:,:] = fitnesses[i]
        fitness_grid_y[:,i,:] = fitnesses[i]
    
    larger_or_equal = fitness_grid_x >= fitness_grid_y
    larger = fitness_grid_x > fitness_grid_y
    
    return np.logical_and(np.all(larger_or_equal,axis=2),np.any(larger,axis=2))




def calculate_pareto_fronts(fitnesses):
    
    # Calculate dominated set for each individual
    domination_sets = []
    domination_counts = []
    
    domination_matrix = calculate_domination_matrix(fitnesses)
    pop_size = fitnesses.shape[0]
    
    for i in range(pop_size):
        current_dimination_set = set()
        domination_counts.append(0)
        for j in range(pop_size):
            if domination_matrix[i,j]:
                current_dimination_set.add(j)
            elif domination_matrix[j,i]:
                domination_counts[-1] += 1
                
        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts==0)[0]
        if len(current_front) == 0:
            #print("Done")
            break
        #print("Front: ",current_front)
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[individual] = -1 # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1
            
    return fronts



def calculate_crowding_metrics(fitnesses,fronts):
    
    num_objectives = fitnesses.shape[1]
    num_individuals = fitnesses.shape[0]
    
    # Normalise each objectives, so they are in the range [0,1]
    # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
    normalized_fitnesses = np.zeros_like(fitnesses)
    for objective_i in range(num_objectives):
        min_val = np.min(fitnesses[:,objective_i])
        max_val = np.max(fitnesses[:,objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:,objective_i] = (fitnesses[:,objective_i] - min_val) / val_range
    
    fitnesses = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for front in fronts:
        for objective_i in range(num_objectives):
            
            sorted_front = sorted(front,key = lambda x : fitnesses[x,objective_i])
            
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf
            if len(sorted_front) > 2:
                for i in range(1,len(sorted_front)-1):
                    crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i+1],objective_i] - fitnesses[sorted_front[i-1],objective_i]

    return  crowding_metrics



def fronts_to_nondomination_rank(fronts):
    nondomination_rank_dict = {}
    for i,front in enumerate(fronts):
        for x in front:   
            nondomination_rank_dict[x] = i
    return nondomination_rank_dict
        

def nondominated_sort(nondomination_rank_dict,crowding):
    
    num_individuals = len(crowding)
    indicies = list(range(num_individuals))

    def nondominated_compare(a,b):
        # returns 1 if a dominates b, or if they equal, but a is less crowded
        # return -1 if b dominates a, or if they equal, but b is less crowded
        # returns 0 if they are equal in every sense
        
        
        if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
            return -1
        elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
            return 1
        else:
            if crowding[a] < crowding[b]:   # crowding metrics, larger better
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0

    non_domiated_sorted_indicies = sorted(indicies,key = functools.cmp_to_key(nondominated_compare),reverse=False) # ascending order, the best is the last
    return non_domiated_sorted_indicies








