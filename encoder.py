""" Super class for all encoders used in
    "On the encoding of categorical variables for machine learning applications"
"""
from utilities import *

class Encoder():

    def __init__(self):
        self.codes = {}

    def transform(self, X):
        return replace_in_df(X, self.codes)

    def fit_transform(self, X,y):
        self.fit(X,y)
        return self.transform(X)

    def get_codes(self):
        if self.codes == {}:
            print("First call the fit or the fit_transform method")
        else:
            return self.codes
            

