import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os 

# 1. Connect to your Feast Feature Store
store = FeatureStore(repo_path="feature_repo")

# 2. Get your Entity Data (GuardianPay Users)
# Changed "driver_id" to "user_id" to match your actual data
entity_df = pd.read_parquet("../data/processed/fraud_data_clean.parquet")[
    ["user_id", "event_timestamp", "Class"]].head(10000)  # Use a subset for faster training during development

# Add this right after creating entity_df
entity_df = entity_df.drop_duplicates(subset=['user_id', 'event_timestamp'])

# 3. Pull historical features from Feast
print("--- Fetching GuardianPay features from Feast ---")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_transaction_stats:V1",
        "user_transaction_stats:V2",
        "user_transaction_stats:V3",
        "user_transaction_stats:Amount",
    ],
).to_df()

# 4. Prepare for training
# Ensure we drop user_id (the ID) and keep the features
X = training_df.drop(columns=["Class", "event_timestamp", "user_id"])
y = training_df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking to the local 'src' folder
abspath = os.path.abspath(os.path.dirname(__file__))
mlflow.set_tracking_uri(f"file:///{abspath}/mlruns")

# 5. Start an MLflow Experiment
mlflow.set_experiment("GuardianPay_Fraud_Detection")

with mlflow.start_run():
    params = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.xgboost.log_model(model, "fraud_model_v1")
    
    print(f"✅ Model Trained! Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("Check your results at localhost:5000")