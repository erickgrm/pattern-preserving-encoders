""" Auxiliary functions for the Pattern Preserving Encoders
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SetOfCodes(object):
    """ Object for an abstract set of codes. Consists of the codes themselves and 
        the attribute "fitness" 

    """
    def __init__(self):
        self.codes = {} # {col_num:{category:code}}
        self.fitness = None

    def get_fitness(self):
        return self.fitness

    def get_codes(self):
        return self.codes

def regression(df, target, estimator, num_predictors):
    """" Picks num_predictors predictor variables at random but distinct from target and fits 
        the chosen estimator. Numerical variables must be already scaled to [0,1]
        
         Returns mean squared error
         
         If a predictor is categorical, it is first encoded with random numbers
    """
    # Relabel columns from 0 
    df.columns = range(len(df.columns)) 

    # Choose random indices for the predictor variables
    chosen_cols = []
    while len(chosen_cols) < num_predictors:
        chosen_cols = np.random.randint(0, len(df.columns), num_predictors)
        chosen_cols = np.unique(chosen_cols)

    # Pick the predictor variables
    predictors = df[chosen_cols]
    predictors = random_encoding_of_categories(predictors)

    # Fit the estimator
    estimator.fit(predictors, target)

    return mean_squared_error(target, estimator.predict(predictors))

def evaluate(codes, df, estimator, num_predictors):
    try:
        X = replace_in_df(df, codes)
    except:
        print("Incorrect number of codes")

    max_error = 0
    for x in X.columns:
        error = regression(X.drop([x], axis =1), X[x], estimator, num_predictors)
        if max_error < error:
            max_error = error

    return max_error

def ordered_insert(population, soc):
    i = 0
    while i < len(population) and population[i].fitness < soc.fitness:
        i += 1
    population.insert(i,soc)
    return population

def plot(setsOfCodes):
    """ Plots the fitness of the sets of codes 
    """
    plt.figure(figsize=(16,8))
    plt.scatter(range(len(setsOfCodes)), [x.fitness for x in setsOfCodes], marker='X')

def replace_in_df(df, mapping): 
    """ Replaces categories by numbers according to the mapping
        If a category is not in mapping, it gets a random code
        mapping: dictionary from categories to codes
    """
    # Ensure df has the right type
    if not(isinstance(df,(pd.DataFrame))):
        try: 
            df = pd.DataFrame(df)
        except:
            raise Exception('Cannot convert to pandas.DataFrame')
            
    cat_cols = categorical_cols(df)

    # Updates the mapping with random codes for categories not 
    # previously in the mapping
    for x in cat_cols:
        values = np.unique(df[x])
        for v in values:
            if not(v in mapping[x]):
                mapping[x][v] = np.random.uniform(0,1)

    return df.replace(mapping)


def scale_df(df):
    """ Scale all numerical variables to [0,1]
    """
    numerical_cols = [x for x in list(df.columns) if x not in categorical_cols(df)]
    sc = MinMaxScaler()

    for x in numerical_cols:
        if min(df[x].values) < 0.0 or 1.0 < max(df[x].values):
            df.loc[0:, x] = sc.fit_transform(df[x].values.reshape(-1,1))

    return df


def is_categorical(array):
    """ Tests if the column is categorical
    """
    return array.dtype.name == 'category' or array.dtype.name == 'object'

def categorical_cols(df): 
    """ Return the column numbers of the categorical variables in df
    """
    cols = []
    # Rename columns as numbers
    df.columns = range(len(df.columns))
    
    for x in df.columns: 
        if is_categorical(df[x]):
            cols.append(x)
    return cols


def categorical_instances(df):
    """ Returns an array with all the categorical instances in df, 
        column by column
    """
    instances = {}
    cols = categorical_cols(df)
    for x in cols:
        instances[x] = list(np.unique(df[x]))

    return instances


def num_categorical_instances(df):
    """ Returns the total number of categorical instances in df
    """
    instances = categorical_instances(df)

    return np.sum([len(instances[x]) for x in instances])


def random_encoding_of_categories(df):
    """ Encodes the categorical variables with random numbers in [0,1]
    """
    cols = categorical_cols(df)
    mapping = {}
    for x in cols:
        np.random.seed()
        codes = np.random.uniform(0,1,len(np.unique(df[x])))
        mapping[x] = dict(zip(np.unique(df[x]), codes))

    return df.replace(mapping)


def var_types(df, cat_cols):
    categorical_var_list = []
    for x in df.columns:
        if is_categorical(df[x]) or x in cat_cols:
            categorical_var_list.append(x)

    return categorical_var_list

def set_categories(df, cat_cols=[]):
    already_categorical = categorical_cols(df)
    cols = [x for x in cat_cols if x not in already_categorical]
    categories = {}
    for x in cols:
        unique = np.unique(df[x])
        xcats = {}
        for v in unique:
            xcats[v]= 'X'+str(x)+'_'+str(v)

        categories[x] = xcats
    return df.replace(categories)

