from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# 1. Define the Source 
# We point to the CLEANED Parquet file from our prepare_data.py script
fraud_source = FileSource(
    path="E:/gurdianpay/data/processed/fraud_data_clean.parquet",
    event_timestamp_column="event_timestamp",
)

# 2. Define the Entity (The "Who")
# In fraud detection, our entity is the user_id, not a driver
user = Entity(
    name="user", 
    join_keys=["user_id"],
    value_type=ValueType.INT64
)

# 3. Define the Feature View (The "What")
# We use the actual columns from the Kaggle Credit Card dataset
user_transaction_stats = FeatureView(
    name="user_transaction_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="V1", dtype=Float32),
        Field(name="V2", dtype=Float32),
        Field(name="V3", dtype=Float32),
        Field(name="Amount", dtype=Float32),
    ],
    online=True,
    source=fraud_source,
)