from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler        
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from .interactions import AddInteractionFeatures # ../src/features/interactions.py
from .binning import BinFeatures # ../src/features/binning.py

def create_preprocessor_pipeline():
    """
    Returns a configured preprocessing pipeline with:
    1. Median imputation
    2. Interaction features
    3. Clinical binning
    4. One-hot encoding
    5. Log transformation
    6. Standard scaling
    """
    # Define feature lists 
    features_to_impute = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    interaction_features = ['GlucoseXAge','GlucoseXBMI','InsulinXBMI','GlucoseInsulinRatio','BMISqrd',
                           'GlucoseSqrd','AgeSqrd']
    features_to_logtransform = features_to_impute + interaction_features + ['Pregnancies','DiabetesPedigreeFunction','Age']
    features_to_encode = ['GlucoseBins','BMIbins','InsulinBins']
    
    # step 1: Impute missing values
    preprocess_impute = ColumnTransformer(
        transformers=[('impute', SimpleImputer(missing_values=np.nan,strategy='median'), features_to_impute)],
        remainder='passthrough',
        verbose_feature_names_out=False,  # Disable renaming
        force_int_remainder_cols=False # avoid future column name issues
    ).set_output(transform='pandas')

    # step 2: AddInteractionFeatures() class
    # step 3: BinFeatures() class

    # step 4: encode binned features
    preprocess_encode = ColumnTransformer(
        transformers=[('encode', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False), features_to_encode)],
        remainder='passthrough',
        verbose_feature_names_out=False,  # Disable renaming
        force_int_remainder_cols=False # avoid future column name issues
    ).set_output(transform='pandas')

    # step 5: log-transform
    preprocess_logtransform = ColumnTransformer(
        transformers=[('log', FunctionTransformer(np.log1p, validate=False), features_to_logtransform)],
        remainder='passthrough',
        verbose_feature_names_out=False,  # Disable renaming
        force_int_remainder_cols=False # avoid future column name issues
    ).set_output(transform='pandas')

    # step 6: scale
    preprocess_scale = ColumnTransformer(
        transformers=[('scale', StandardScaler(), features_to_logtransform)],
        remainder='passthrough',
        verbose_feature_names_out=False,  # Disable renaming
        force_int_remainder_cols=False # avoid future column name issues
    ).set_output(transform='pandas')

    return Pipeline([
        ('impute', preprocess_impute),
        ('add_interaction', AddInteractionFeatures()),
        ('bin_features', BinFeatures()),
        ('encode', preprocess_encode),
        ('log_transform', preprocess_logtransform),
        ('scale', preprocess_scale)
    ])