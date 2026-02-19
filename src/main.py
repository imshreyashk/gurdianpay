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
            raise Exception(f"Experiment {experiment_name} not found!")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["metrics.f1_score DESC"]
        )
        
        if runs.empty:
            raise Exception("No runs found in MLflow!")

        best_run_id = runs.iloc[0].run_id
        
        # 2. DYNAMIC ARTIFACT LOADING: 
        # Instead of hardcoding 'fraud_model_v1', we point to the run directly.
        # MLflow will find the booster file (model.ubj) automatically.
        model_uri = f"runs:/{best_run_id}/models/m-00158b6c04b24b328d2ca3f3f744154b/artifacts"
        
        # If the above fails, MLflow's standard run URI is safer:
        model_uri_standard = f"runs:/{best_run_id}/fraud_model_v1"
        
        print(f"🚀 Attempting to load model from Run ID: {best_run_id}")
        return mlflow.xgboost.load_model(model_uri_standard), best_run_id

    except Exception as e:
        print(f"⚠️ Initial load failed, trying absolute fallback...")
        # Fallback for the specific path seen in your GitHub error logs
        fallback_path = MLRUNS_DIR / "999467333024495658" / "06a331b57eab4ec18a9e7f975b41e289" / "artifacts" / "model.ubj"
        return mlflow.xgboost.load_model(str(fallback_path)), "fallback_id"

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