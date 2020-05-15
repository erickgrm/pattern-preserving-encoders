# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRegression(LinearRegression):

    def __init__(self, max_degree=1, interaction=False):
        super().__init__()
        self.max_degree = max_degree
        self.interaction = interaction
        self.poly = PolynomialFeatures(self.max_degree, interaction_only=self.interaction)

    def fit(self, X, y):
        return super(PolynomialRegression, self).fit(self.poly.fit_transform(X),y)

    def predict(self, X):
            return super(PolynomialRegression, self).predict(self.poly.fit_transform(X))

         
