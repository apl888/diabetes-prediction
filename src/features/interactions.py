import pandas as pd  
from sklearn.base import BaseEstimator, TransformerMixin

class AddInteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):           
        X = X.copy()
        X['GlucoseXAge'] = X['Glucose']*X['Age']
        X['GlucoseXBMI'] = X['Glucose']*X['BMI']
        X['InsulinXBMI'] = X['Insulin']*X['BMI']
        X['GlucoseInsulinRatio'] = X['Glucose']/(X['Insulin']+1e-6)
        X['BMISqrd'] = X['BMI']**2
        X['GlucoseSqrd'] = X['Glucose']**2
        X['AgeSqrd'] = X['Age']**2
        return X