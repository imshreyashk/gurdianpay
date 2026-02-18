import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 1. Load the raw data from your specific path
input_path = "data/creditcard.csv" 
print(f"Reading {input_path}...")
df = pd.read_csv(input_path)

# 2. Add 'user_id' (Feast needs an Entity)
np.random.seed(42)
df['user_id'] = np.random.randint(1000, 2000, size=len(df))

# 3. Add 'event_timestamp' (Feast needs time-series logic)
# We set these to "now" so Feast considers them fresh
end_date = datetime.now()
df['event_timestamp'] = [end_date - timedelta(minutes=np.random.randint(0, 1440)) for _ in range(len(df))]

# 4. Create the 'processed' directory
os.makedirs("data/processed", exist_ok=True)

# 5. Save as PARQUET (This is what Feast actually wants)
output_path = "data/processed/fraud_data_clean.parquet"
df.to_parquet(output_path, index=False)

print(f"✅ SUCCESS! Parquet file created at: {output_path}")