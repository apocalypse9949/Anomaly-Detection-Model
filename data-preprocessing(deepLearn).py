import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Loading millions of login records (CSV, database, or real-time streams) recommded kaggle dataset
df = pd.read_csv("login_data.csv")  

# One-hot encode categorical data (device type, country)
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['Device', 'Country']]).toarray()

# now we have to Scale numerical features (Login Hour, Failed Attempts)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Login_Hour', 'Failed_Attempts']])

# now Combining processed features
X = np.hstack((scaled_features, encoded_features))
y = df['Label'].values  # 0 = Normal, 1 = Anomaly

#  Save processed data for training 
np.save("X_train.npy", X)
np.save("y_train.npy", y)
