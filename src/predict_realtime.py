import mlflow
from feast import FeatureStore
import pandas as pd
import os

# 1. Setup MLflow Tracking Path (so it knows where to look)
abspath = os.path.abspath(os.path.dirname(__file__))
mlflow.set_tracking_uri(f"file:///{abspath}/mlruns")

# Replace your current Step 2 with this:
experiment_name = "GuardianPay_Fraud_Detection"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Find the run with the best F1 score automatically
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                          order_by=["metrics.f1_score DESC"])

latest_run_id = runs.iloc[0].run_id
print(f"Using Best Model from Run ID: {latest_run_id}")

model_uri = f"runs:/{latest_run_id}/fraud_model_v1"
model = mlflow.xgboost.load_model(model_uri)

# 3. Connect to Feast
store = FeatureStore(repo_path="feature_repo")

def predict_fraud(user_id):
    # 4. Get LIVE features
    feature_vector = store.get_online_features(
        features=[
            "user_transaction_stats:V1",
            "user_transaction_stats:V2",
            "user_transaction_stats:V3",
            "user_transaction_stats:Amount",
        ],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()

    # 5. Format and RE-ORDER columns to match the model
    features_df = pd.DataFrame.from_dict(feature_vector)
    
    # Define the EXACT order the model expects
    correct_order = ['V1', 'V2', 'V3', 'Amount']
    
    # Re-order the dataframe (and drop user_id at the same time)
    X = features_df[correct_order]

    # 6. Predict!
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0][1]
    
    status = "🚨 FRAUD DETECTED" if prediction[0] == 1 else "✅ TRANSACTION SAFE"
    
    print(f"\n--- GuardianPay Security Check ---")
    print(f"User: {user_id}")
    print(f"Result: {status}")
    print(f"Risk Score: {probability:.2%}")
    print(f"----------------------------------")


    
# Test it out for specific users
predict_fraud(1001)
predict_fraud(1005)