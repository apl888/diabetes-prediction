import pandas as pd  
from sklearn.base import BaseEstimator, TransformerMixin    

class BinFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''initialize with clinical bin definitions''' 
        self.glc_bins = [0,139,199,float('inf')] # useing inf for open-ended
        self.glc_labels = ['Normal','Prediabetes','Diabetes']

        self.bmi_bins = [0,18.5,24.9,29.9,34.9,39.9,float('inf')]
        self.bmi_labels = ['BMIunderweight','BMIhealthy','BMIoverweight','BMIobese1','BMIobese2','BMIobese3']

        self.ins_bins = [0,16,166,float('inf')]
        self.ins_labels = ['InuslinLow','InsulinNormal','InsulinHigh']

    def transform(self, X):
        '''apply clinical binning with edge handling'''
        X = X.copy()
        
        # glucose binning 
        X['GlucoseBins'] = pd.cut(
            X['Glucose'],
            bins=self.glc_bins,
            labels=self.glc_labels,
            right=False,
            include_lowest=True
        )
        
        # BMI binning
        X['BMIbins'] = pd.cut(
            X['BMI'],
            bins=self.bmi_bins,
            labels=self.bmi_labels,
            right=False,
            include_lowest=True
        )
        
        # insulin binning
        X['InsulinBins'] = pd.cut(
            X['Insulin'],
            bins=self.ins_bins,
            labels=self.ins_labels,
            right=False,
            include_lowest=True
        )
        
        # validate no NA bins
        if X[['GlucoseBins', 'BMIbins', 'InsulinBins']].isna().any().any():
            raise ValueError("Some values couldn't be binned. Check binning ranges.")
                             
        return X
    
    def fit(self, X, y=None):
        '''no fitting needed for fixed clinical bins'''
        return self