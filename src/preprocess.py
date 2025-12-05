import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
data = pd.read_csv(file_path)
return data




def preprocess_data(data):
X = data.drop('CLASS', axis=1)
y = data['CLASS']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
return X_scaled, y




def split_data(X, y, test_size=0.2, random_state=42):
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=test_size, random_state=random_state)
return X_train, X_test, y_train, y_test
