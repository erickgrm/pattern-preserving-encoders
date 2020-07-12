""" 
    Implementation of several Pattern Preserving Encoders. For details see:
    "On the encoding of  categorical variables for Machine Learning applica-
    tions", Chapter 3.

    author github.com/erickgrm
"""
# Required libraries
import pandas as pd
import numpy as np
import random
import itertools
from sklearn.preprocessing import MinMaxScaler
import collections

# Clerical
from .utilities import *
from .encoder import Encoder

# Libraries providing estimators
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.svm import SVR
from .polynomial_regression import PolynomialRegression, OddDegPolynomialRegression
from sklearn.neural_network import MLPRegressor

dict_estimators = {}
dict_estimators['LinearRegression'] = LinearRegression()
dict_estimators['SGDRegressor'] = SGDRegressor(loss='squared_loss')
dict_estimators['SVR'] = SVR()
dict_estimators['PolynomialRegression'] = PolynomialRegression(max_degree=3)
dict_estimators['Perceptron'] = MLPRegressor(max_iter=150, hidden_layer_sizes=(10,5)) 
dict_estimators['CESAMORegression'] = OddDegPolynomialRegression(max_degree=11)

from multiprocessing import Pool, Process, cpu_count

class SimplePPEncoder(Encoder):
    """ Samples randomly 600 sets of codes (can be changed with self.sampling_size), 
        encodes with best found 
    """

    def __init__(self, estimator_name='PolynomialRegression', num_predictors=2, 
                 sample_size=600, n_thread=4):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
        """
        super(SimplePPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.sample_size = sample_size
        self.best_soc = None
        self.codes = {}
        self.categorical_var_list =[]

        self.df = None
        self.history = []
        self.categories = {} 
        self.n_thread = min(cpu_count()-1, n_thread)

    def fit(self, df, target=None, cat_cols=[]):
        #Restart in case the same instance is called
        self.codes = {} 

        # Set which variables will be encoded
        self.categorical_var_list = var_types(df, cat_cols)

        # Scale and transform vars in cat_cols to be categorical if needed
        self.df = set_categories(scale_df(df.copy()), self.categorical_var_list)
        self.categories = categorical_instances(self.df)

        # Parallel creation of self.sample_size sets of codes
        pool = Pool(self.n_thread)
        self.history = pool.map(self.new_soc, range(self.sample_size))
        pool.close()

        # Try to free up memory
        del self.df
        
        # Pick best set of codes
        self.best_soc = min(self.history, key=lambda x: x.fitness)
        self.codes = self.best_soc.codes

    def new_soc(self, i):
        np.random.seed()
        soc = SetOfCodes()
        for x in self.categories:
            xcodes= np.random.uniform(0, 1, len(self.categories[x]))
            soc.codes[x] = dict(zip(self.categories[x], xcodes)) 
            
        soc.fitness = evaluate(soc.codes, self.df, self.estimator, self.num_predictors)

        return soc

    def plot_history(self):
        plot(self.history)


class AgingPPEncoder(Encoder):
    """ Samples sets of codes accorgding to a simplified genetic algorithm, called Aging Evolution
        and that only allows mutation and deletes oldest individual at each iteration.

        Completes 800 iterations (can be changed with self.cycles)
    """

    def __init__(self, estimator_name='PolynomialRegression', num_predictors=2, 
                 cycles=800, n_thread=4):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
            cycles should be a multiple of 200
        """
        super(AgingPPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.cycles = cycles # How many codes will be sampled, instead of generations
        self.prob_mutation = 0.20
        self.size_population = 25
        self.subsample_size = int(self.size_population/5)
        self.codes = {}
        self.best_soc = None
        self.categorical_var_list =[]

        self.df = None
        self.history = []
        self.categories = {}
        self.n_thread = min(cpu_count(), n_thread)
        self.job_size = int(min(self.cycles, 200))
        self.jobs = int(self.cycles/self.job_size)
        

    def fit(self, df, target=None, cat_cols=[]):
        #Restart in case the same instance is called
        self.codes = {} 

        # Set which variables will be encoded
        self.categorical_var_list = var_types(df, cat_cols)

        # Scale and transform vars in cat_cols to be categorical if needed
        self.df = set_categories(scale_df(df.copy()), self.categorical_var_list)
        self.categories = categorical_instances(self.df)

        # Evolve the population
        pool = Pool(self.n_thread)
        parallel_outputs = pool.map(self.aging_algorithm, range(self.jobs))
        pool.close()

        # Try to free up memory
        del self.df

        # Flatten outputs
        self.history = list(itertools.chain.from_iterable(parallel_outputs))

        # pick the best codes
        self.best_soc = min(self.history, key=lambda x: x.fitness)
        self.codes = self.best_soc.codes

        
    def aging_algorithm(self, i):
        np.random.seed()

        population = collections.deque()
        partial_history = []

        # Initialise population with random individuals
        while len(population) < self.size_population: 
            soc = SetOfCodes()
            for x in self.categories:
                xcodes = np.random.uniform(0,1, len(self.categories[x])) 
                soc.codes[x] = dict(zip(self.categories[x], xcodes))
            soc.fitness = evaluate(soc.codes, self.df, self.estimator, self.num_predictors)
            population.append(soc)
            partial_history.append(soc) 

        while len(partial_history) < self.job_size:
            sample_inds = np.random.randint(0, self.size_population, self.subsample_size)
            sample = [population[i] for i in sample_inds]
            
            parent = min(sample, key=lambda x: x.fitness)

            child = SetOfCodes()
            child.codes = self.mutate(parent.codes)
            child.fitness = evaluate(child.codes, self.df, self.estimator, self.num_predictors)
            population.append(child)
            partial_history.append(child)
            
            population.popleft()

        return partial_history

            
    def mutate(self, codes):
        for x in codes:
            for category in codes[x]:
                if np.random.uniform(1) < self.prob_mutation:
                    codes[x][category] = np.random.uniform()
        return codes

    def plot_history(self):
        plot(self.history)

class GeneticPPEncoder(Encoder):
    """ Samples sets of codes according to the Eclectic Genetic Algorithm.  
        Completes 80 generations of a population of size 
    """

    def __init__(self, estimator_name='PolynomialRegression', num_predictors=2, 
                 generations=60, n_thread=4):
        """ estimator_name -> any of the dict_estimators keys
            num_predictors -> how many predictors to use for each regression
        """
        super(GeneticPPEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = num_predictors
        self.best_soc = None
        self.codes = {}
        self.categories = {}
        self.categorical_var_list = []

        self.df = None
        self.generations = generations # How many generations the GA will run for
        self.size_population = 25
        self.rate_mutation = 0.10
        self.prob_mutation = 20
        self.population = []
        self.history = []
        self.n_thread = min(cpu_count(), n_thread)

    def fit(self, df, target=None, cat_cols=[]):
        #Restart in case the same instance is called
        self.codes = {} 

        # Set which variables will be encoded
        self.categorical_var_list = var_types(df, cat_cols)

        # Scale and transform vars in cat_cols to be categorical if needed
        self.df = set_categories(scale_df(df.copy()), self.categorical_var_list)
        self.categories = categorical_instances(self.df)

        # Evolve the population
        self.EGA()

        # Try to free up memory
        del self.df

        # Pick the best set of codes
        self.best_soc = min(self.population, key=lambda x: x.fitness)
        self.codes = self.best_soc.codes
        
    def EGA(self):
        """ Implementation of the Eclectic Genetic Algorithm
        """
        # Parallel initialisation with random individuals  
        pool = Pool(self.n_thread)
        self.history = pool.map(self.new_soc, range(self.size_population))
        pool.close()
        for soc in self.history:
            ordered_insert(self.population, soc)

        # Evolution of the population
        G = 0
        while G  < self.generations: 
            self.crossover_population()
            self.mutate_population()
            G += 1

    def new_soc(self, i):
        np.random.seed()
        soc = SetOfCodes()
        for x in self.categories:
            xcodes= np.random.uniform(0, 1, len(self.categories[x]))
            soc.codes[x] = dict(zip(self.categories[x], xcodes)) 

        soc.fitness = evaluate(soc.codes, self.df, self.estimator, self.num_predictors)
        
        return soc

            
    def crossover_population(self):
        """ Routine for the crossover of the population, by pairing individuals
        """

        pool = Pool(self.n_thread)
        children = pool.map(self.crossover, range(int(self.size_population/2)))
        pool.close()
        # Flatten
        children = list(itertools.chain.from_iterable(children))
        
        for soc in children: 
            self.history.append(soc)
            ordered_insert(self.population, soc)

        self.population = self.population[:self.size_population]


    def mutate_population(self):
        """ Routine to perform mutation on the population
        """
        # Choose how many and which will be mutated
        how_many = int(self.rate_mutation*self.size_population)
        chosen = np.random.randint(0, self.size_population, how_many)

        # Mutate
        pool = Pool(self.n_thread)
        children = pool.map(self.mutate, chosen)
        pool.close()

        for soc in children: 
            self.history.append(soc)
            ordered_insert(self.population, soc)

        self.population = self.population[:self.size_population]


    def crossover(self, i):
        """ Routine for the (anular) crossover of two individuals
        """
        codes1 = self.population[i].codes
        codes2 = self.population[self.size_population-i-1].codes

        indexes = list(self.categories.keys())
        while True:
            x = random.choice(indexes)
            y = random.choice(indexes)
            if x < y:
                break
        new_codes1 = {}
        new_codes2 = {}

        for v in [v for v in indexes if v <= x]:
            new_codes1[v] = codes1[v]
            new_codes2[v] = codes2[v]

        xcategories = list(codes1[x].keys())
        for c in xcategories:
            if np.random.uniform() < 0.5:
                new_codes1[x][c] = codes2[x][c]
                new_codes2[x][c] = codes1[x][c]
            
        for v in [v for v in indexes if x < v and v < y]:
            new_codes1[v] = codes2[v]
            new_codes2[v] = codes1[v]

        for v in [v for v in indexes if y <= v]:
            new_codes1[v] = codes1[v]
            new_codes2[v] = codes2[v]

        ycategories = list(codes1[y].keys())
        for c in ycategories:
            if np.random.uniform() < 0.5:
                new_codes1[y][c] = codes2[y][c]
                new_codes2[y][c] = codes1[y][c]

        child1 = SetOfCodes()
        child1.codes = new_codes1
        child1.fitness = evaluate(child1.codes, self.df, self.estimator, self.num_predictors)

        child2 = SetOfCodes()
        child2.codes = new_codes2
        child2.fitness = evaluate(child2.codes, self.df, self.estimator, self.num_predictors)

        return [child1, child2]


    def mutate(self, i):
        """ Routine to perform mutation on an individual
        """
        codes = self.population[i].codes
        for x in codes:
            for category in codes[x]:
                if np.random.uniform(1) < self.prob_mutation:
                    codes[x][category] = np.random.uniform()

        child = SetOfCodes()
        child.codes = codes
        child.fitness = evaluate(child.codes, self.df, self.estimator, self.num_predictors)

        return child

    def plot_history(self):
        plot(self.history)
