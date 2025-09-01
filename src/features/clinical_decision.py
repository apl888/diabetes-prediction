import pandas as pd

def assess_patient_risk(
    patient_features, 
    important_features, 
    preprocessor, 
    calibrated_model,
    threshold = 0.330
    ):
    '''
    Assess diabetes risk for a patient and provide clinical recommendations.
    
    Parameters:
    -----------
    patient_features : DataFrame
        Raw patient features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    important_features : list
        List of feature names to keep after importance filtering
    preprocessor : sklearn Pipeline
        Fitted preprocessing pipeline
    calibrated_model : fitted model
        Calibrated classification model
    threshold : Clinical decision threshold (default: 0.330 for balanced approach)
    
    Returns:
    --------
    dict: Contains probability, risk level, and recommendation
    '''
    
    # validate threshold
    if not (0 <= threshold <= 1):
        raise ValueError('Threshold must be between 0 and 1')
    
    # preprocess and predict
    processed_data = preprocessor.transform(patient_features)
    important_data = processed_data[important_features]
    probability = calibrated_model.predict_proba(important_data)[:,1][0]

    # define risk bands relative to the selected threshold
    high_risk_band = threshold
    moderate_risk_band = max(0, threshold - 0.150) # ensure that it is not < 0
    
    # clinical recommendation
    if probability >= high_risk_band:
        recommendation = 'Refer for diabetes testing and counseling'
        risk_level = 'High'
    elif probability >= moderate_risk_band:
        recommendation = 'Monitor with lifestyle counseling'
        risk_level = 'Moderate'
    else:
        recommendation = 'Routine screening recommended'
        risk_level = 'Low'

    return {
        'probability': f'{probability:.1%}',
        'risk_level': risk_level,
        'recommendation': recommendation,
        'threshold_used': threshold
    }