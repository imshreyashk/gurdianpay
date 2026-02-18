from fastapi import FastAPI, HTTPException
import mlflow
from feast import FeatureStore
import pandas as pd
import os

app = FastAPI(title="GuardianPay Fraud Detection API")

current_dir = os.getcwd()
# Setup MLflow Tracking
abspath = os.path.abspath(os.path.dirname(__file__))
mlflow.set_tracking_uri(f"file://{os.path.join(os.getcwd(), 'mlruns')}")
# Connect to Feast
store = FeatureStore(repo_path="feature_repo")

# Load the Best Model once when the server starts
experiment_name = "GuardianPay_Fraud_Detection"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                          order_by=["metrics.f1_score DESC"])
best_run_id = runs.iloc[0].run_id
model_path = os.path.join(os.getcwd(), "mlruns", experiment.experiment_id, best_run_id, "artifacts", "fraud_model_v1")

print(f"🚀 Loading model from: {model_path}")
model = mlflow.xgboost.load_model(model_path)


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

        # Align features
        features_df = pd.DataFrame.from_dict(feature_vector)
        X = features_df[['V1', 'V2', 'V3', 'Amount']]

        # Predict
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