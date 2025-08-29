import pandas as pd

def assess_patient_risk(patient_features, important_features, preprocessor, calibrated_model):
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
    threshold : float, default=0.46
        Risk threshold for clinical decisions
    
    Returns:
    --------
    dict: Contains probability, risk level, and recommendation
    '''
    
    # preprocess and predict
    processed_data = preprocessor.transform(patient_features)
    important_features = processed_data[important_features]
    probability = calibrated_model.predict_proba(important_features)[:,1][0]

    # clinical recommendation
    if probability >= 0.46:
        recommendation = 'Refer for diabetes testing and counseling'
        risk_level = 'High'
    elif probability >= 0.32:
        recommendation = 'Monitor with lifestyle counseling'
        risk_level = 'Moderate'
    else:
        recommendation = 'Routine screening recommended'
        risk_level = 'Low'

    return {
        'probability': f'{probability:.1%}',
        'risk_level': risk_level,
        'recommendation': recommendation
    }