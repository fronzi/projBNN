
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from deap import base, creator
from deap import tools
import warnings
warnings.filterwarnings('ignore')


# In[2]:
Number_Monolayers = 773
path = './LASSO_BR2_5/'
dataset = pd.read_csv(str(path)+'PLMF.csv').values
X, y = dataset[:,1:-1], dataset[:,-1]


monolayer_descriptors = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=0) # read file with monolayers names and descriptors 
titles = pd.read_csv("1l_atomicPLMF_"+str(Number_Monolayers)+"structures.csv",header=None)
numMonolayerColumns = monolayer_descriptors.shape[1] 
numMonolayerRecords = monolayer_descriptors.shape[0] 

print('numMonolayerColumns',numMonolayerColumns)
print('numMonolayerRecords',numMonolayerRecords)


BilayerProperty = pd.read_csv("Bilayer_Energy.csv",header=0) # read file with bilayers names and target values

print(BilayerProperty)

numBilayerRecords = BilayerProperty.shape[0]
print('numBilayerRecords',numBilayerRecords)
bilayers = BilayerProperty.iloc[:,0]
print('bilayers',bilayers)
monolayers = monolayer_descriptors.iloc[:,0]
print('monolayers',monolayers)




# In[5]:


def _create(self):
    creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FeatureSelect)
    return creator

def create_toolbox(self):
   

    self._init_toolbox()
    return toolbox

def register_toolbox(self,toolbox):
    
    toolbox.register("evaluate", self.evaluate)
    self.toolbox = toolbox


def _init_toolbox(self):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, self.n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def _default_toolbox(self):
    toolbox = self._init_toolbox()
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", self.evaluate)
    return toolbox

def get_final_scores(self,pop,fits):
    self.final_fitness = list(zip(pop,fits))



def generate(self,n_pop,cxpb = 0.5,mutxpb = 0.2,ngen=5,set_toolbox = False):


    if self.verbose==1:
        print("Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(n_pop,cxpb,mutxpb,ngen))

    if not set_toolbox:
        self.toolbox = self._default_toolbox()
    else:
        raise Exception("Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox")
    pop = self.toolbox.population(n_pop)
    CXPB, MUTPB, NGEN = cxpb,mutxpb,ngen

    # Evaluate the entire population
    print("EVOLVING.......")
    fitnesses = list(map(self.toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        print("-- GENERATION {} --".format(g+1))
        offspring = self.toolbox.select(pop, len(pop))
        self.fitness_in_generation[str(g+1)] = max([ind.fitness.values[0] for ind in pop])
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        weak_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, weak_ind))
        for ind, fit in zip(weak_ind, fitnesses):
            ind.fitness.values = fit
        print("Evaluated %i individuals" % len(weak_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

                # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    if self.verbose==1:
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- Only the fittest survives --")

    self.best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))
    self.get_final_scores(pop,fits)

    return pop

#==============================================================================
# Class performing feature selection with genetic algorithm
#==============================================================================
class GeneticSelector():
    def __init__(self, estimator, n_gen, size, n_best, n_rand, 
                 n_children, mutation_rate):
        # Estimator 
        self.estimator = estimator
        # Number of generations
        self.n_gen = n_gen
        # Number of chromosomes in population
        self.size = size
        # Number of best chromosomes to select
        self.n_best = n_best
        # Number of random chromosomes to select
        self.n_rand = n_rand
        # Number of children created during crossover
        self.n_children = n_children
        # Probablity of chromosome mutation
        self.mutation_rate = mutation_rate
        
        if int((self.n_best + self.n_rand) / 2) * self.n_children != self.size:
            raise ValueError("The population size is not stable.")  
            
    def initilize(self):
        population = []
        for i in range(self.size):
            chromosome = np.ones(self.n_features, dtype=np.bool)
            mask = np.random.rand(len(chromosome)) < 0.3
            chromosome[mask] = False
            population.append(chromosome)
        return population

    def fitness(self, population):
        X, y = self.dataset
        scores = []
        for chromosome in population:
            score = -1.0 * np.mean(cross_val_score(self.estimator, X[:,chromosome], y, 
                                                       cv=5, 
                                                       scoring="neg_mean_squared_error"))
            scores.append(score)
        scores, population = np.array(scores), np.array(population) 
        inds = np.argsort(scores)
        return list(scores[inds]), list(population[inds,:])

    def select(self, population_sorted):
        population_next = []
        for i in range(self.n_best):
            population_next.append(population_sorted[i])
        for i in range(self.n_rand):
            population_next.append(random.choice(population_sorted))
        random.shuffle(population_next)
        return population_next

    def crossover(self, population):
        population_next = []
        for i in range(int(len(population)/2)):
            for j in range(self.n_children):
                chromosome1, chromosome2 = population[i], population[len(population)-1-i]
                child = chromosome1
                mask = np.random.rand(len(child)) > 0.5
                child[mask] = chromosome2[mask]
                population_next.append(child)
        return population_next

    def mutate(self, population):
        population_next = []
        for i in range(len(population)):
            chromosome = population[i]
            if random.random() < self.mutation_rate:
                mask = np.random.rand(len(chromosome)) < 0.05
                chromosome[mask] = False
            population_next.append(chromosome)
        return population_next

    def generate(self, population):
        # Selection, crossover and mutation
        scores_sorted, population_sorted = self.fitness(population)
        population = self.select(population_sorted)
        population = self.crossover(population)
        population = self.mutate(population)
        # History
        self.chromosomes_best.append(population_sorted[0])
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
        
        return population

    def fit(self, X, y):
 
        self.chromosomes_best = []
        self.scores_best, self.scores_avg  = [], []
        
        self.dataset = X, y
        self.n_features = X.shape[1]
        
        population = self.initilize()
        for i in range(self.n_gen):
            population = self.generate(population)
            
        return self 
    
    @property
    def support_(self):
        return self.chromosomes_best[-1]

    def plot_scores(self):
        plt.plot(self.scores_best, label='Best')
        plt.plot(self.scores_avg, label='Average')
        plt.legend()
        plt.ylabel('Scores')
        plt.xlabel('Generation')
        plt.show()


# In[6]:



#sel = GeneticSelector(estimator=Lasso(), 
#                      n_gen=7, size=200, n_best=40, n_rand=40, 
#                      n_children=5, mutation_rate=0.05)

sel = GeneticSelector(estimator=Lasso(), 
                      n_gen=20, size=20, n_best=4, n_rand=4, 
                      n_children=5, mutation_rate=0.05)

sel.fit(X, y)
sel.plot_scores()
est= Lasso(alpha=1e-6)

score = -1.0 * cross_val_score(est, X[:,sel.support_], y, cv=5, scoring="neg_mean_squared_error")
print("CV MSE after feature selection: {:.2f}".format(np.mean(score)))


# In[7]:


print ('Indexes of useful features', np.where(sel.support_))
print ('Indexes of useful features', sel.support_)

#fields = pd.DataFrame(np.where(sel.support_)).to_csv(str(path)+"GA_lasso_coefficients.csv", header=None)
GA_lasso_fields = pd.DataFrame(np.where(sel.support_))



# In[8]:


print ('Number of useful features:',np.sum(sel.support_))

###############################################################################


thefile = open(str(path)+'GA_lasso_fields.csv', 'w')
new_training_set = open(str(path)+'BR2_GA_training-test_set.csv', 'w')

#new_training_set.write("%s," % GA_lasso_fields.T) 
for j in range(0,numBilayerRecords):
    new_training_set.write("%s," % j)
    for i in range(0,numMonolayerColumns-1):
        if i in GA_lasso_fields:  
            new_training_set.write("%s," % X[j][i])
#            new_training_set.write("%s," % str(X.iloc[j,i]))
    new_training_set.write("%s\n" % y[j]) 
#    new_training_set.write("%s\n," % str(y.iloc[j,0]))     
new_training_set.close()
    


for i in range(0,titles.shape[1]-1):
        if i in GA_lasso_fields:
            print(titles.iloc[0,i+1])
    
    
lasso_monolayer_data= open(str(path)+'GA_lasso_monolayer_data.csv', 'w')
for j in range(0,Number_Monolayers-1):                                                #####NUMBER OF MONOLAYERS+1 ######
    lasso_monolayer_data.write("%s," % titles.iloc[j,0])
    for i in range(0,numMonolayerColumns-2):
        if i in GA_lasso_fields:
            lasso_monolayer_data.write("%s," % titles.iloc[j,i+1])
    lasso_monolayer_data.write("\n")
lasso_monolayer_data.close()

GA_lasso_fields.T.to_csv(str(path)+"GA_lasso_fields.csv", index = None, header=None)

