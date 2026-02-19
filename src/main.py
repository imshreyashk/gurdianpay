from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.xgboost
from feast import FeatureStore
import pandas as pd
import os
import pathlib

app = FastAPI(title="GuardianPay Fraud Detection API")

# 1. PATH FIX: Use pathlib for cross-platform compatibility (Windows/Linux)
# This finds the directory where main.py sits, regardless of where the command is run from.
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MLRUNS_DIR = BASE_DIR / "mlruns"

# Setup MLflow Tracking
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

# Connect to Feast
store = FeatureStore(repo_path=str(BASE_DIR / "feature_repo"))

def load_best_model():
    try:
        experiment_name = "GuardianPay_Fraud_Detection"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise Exception("Experiment not found!")

        # 🔍 BROAD SEARCH: Look inside the entire Experiment ID folder
        # This will find 'model.ubj' regardless of how many 'models/m-xxx' folders are in the way
        exp_folder = BASE_DIR / "mlruns" / experiment.experiment_id
        
        print(f"🔎 Searching for model.ubj in: {exp_folder}")
        model_files = list(exp_folder.rglob("model.ubj"))
        
        if not model_files:
            raise Exception(f"Could not find model.ubj anywhere in experiment {experiment.experiment_id}")
            
        # Use the first one found (which will be from the best run if sorted, 
        # but since you only have one major model file, this is safe)
        model_load_path = model_files[0].parent
        print(f"✅ Found model at: {model_load_path}")
        
        # We still need a Run ID for the API response
        # We can extract it from the path string
        run_id = model_files[0].parts[-5] # Extracts the run_id from the path
        
        return mlflow.xgboost.load_model(str(model_load_path)), run_id
        
    except Exception as e:
        print(f"❌ Load failed: {e}")
        raise e
model, best_run_id = load_best_model()

@app.get("/")
def home():
    return {"message": "GuardianPay Fraud API is Online", "model_run_id": best_run_id}

@app.get("/predict/{user_id}")
def predict(user_id: int):
    try:
        # Get live features from Feast
        feature_vector = store.get_online_features(
            features=[
                "user_transaction_stats:V1",
                "user_transaction_stats:V2",
                "user_transaction_stats:V3",
                "user_transaction_stats:Amount",
            ],
            entity_rows=[{"user_id": user_id}],
        ).to_dict()

        features_df = pd.DataFrame.from_dict(feature_vector)
        
        # Check if features exist
        if features_df['V1'].isnull().values.any():
            raise Exception(f"User {user_id} not found in Feature Store")

        X = features_df[['V1', 'V2', 'V3', 'Amount']]

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        return {
            "user_id": user_id,
            "is_fraud": bool(prediction),
            "risk_score": f"{probability:.2%}",
            "status": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ SAFE"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)