import numpy as np
import pandas as pd
from imblearn import over_sampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/BankChurners.csv", delimiter=';')
df.drop('CLIENTNUM', axis=1, inplace=True)  # Column not needed.

# Attrition and gender are binary features, thus we use label encoder to encode them.
enc = LabelEncoder()
df["Attrition_Flag"] = enc.fit_transform(df["Attrition_Flag"])
df["Gender"] = enc.fit_transform(df["Gender"])

# One-Hot Encoding is used for the remaining categorical features.
onehot_cols = ['Education_Level', 'Income_Category', 'Card_Category', 'Marital_Status']
onehot_cols_new = pd.get_dummies(df[onehot_cols], drop_first=False)

# Concatenate df with new columns.
df = pd.concat([df, onehot_cols_new], axis=1)
df.drop(onehot_cols, axis=1, inplace=True)

# Convert into numpy array.
all = np.array(df)

# Standardize features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X = scaler.fit_transform(all[:, 1:27])  # Input features.
y = all[:, 0]  # Attrition flag - Target.

# First, split into train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Oversample the minority class for dataset balancing.
ros = over_sampling.RandomOverSampler(sampling_strategy='auto')
X_train, y_train = ros.fit_resample(X_train, y_train)
