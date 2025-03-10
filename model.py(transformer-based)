import numpy as np
from sklearn.ensemble import IsolationForest

# Load millions of login records from DB (Mock data for now)
data = np.random.rand(1000000, 3)  # 1M login records

# Train an advanced anomaly detection model
model = IsolationForest(contamination=0.02)
model.fit(data)

# Save the trained model
import joblib
joblib.dump(model, "ai_anomaly_model.pkl")
