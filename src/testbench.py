import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("../../data/BankChurners.csv", delimiter=';')
df.drop('CLIENTNUM', axis = 1,inplace=True) #column not needed

# Attrition and gender are binary features, thus we use label encoder to encode them.
enc = LabelEncoder()
df["Attrition_Flag"] = enc.fit_transform(df["Attrition_Flag"])
df["Gender"] = enc.fit_transform(df["Gender"])

# One-Hot Encoding is used for the remaining categorical features.
onehot_cols = ['Education_Level', 'Income_Category','Card_Category','Marital_Status']
onehot_cols_new = pd.get_dummies(df[onehot_cols], drop_first=False) #

# Concatenate df with new columns.
df = pd.concat([df,onehot_cols_new], axis=1)
df.drop(onehot_cols, axis = 1,inplace=True)

# Scale values to avoid outliers.
#scaler = StandardScaler()
#all = scaler.fit_transform(np.array(df))
all = np.array(df)


X = all[:,1:27]
y = all[:,0] # Attrition flag

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # First, split into train and test.
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)  # 0.5 x 0.3 = 0.15 split for validation set.

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=6, random_state=40)
rf.fit(X_train,y_train)