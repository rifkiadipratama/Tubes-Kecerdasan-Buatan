from joblib import load

scaler = load(r'saved_models\classifier_20250518_155733.joblib')
print(type(scaler))
