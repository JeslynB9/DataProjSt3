import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
data = pd.read_csv("SDG_goal3_clean.csv")
X = data[["Health worker density, by type of occupation (per 10,000 population)::NURSEMIDWIFE"]]
y = data["Neonatal mortality rate (deaths per 1,000 live births)"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the SVR model with a linear kernel
model = SVR(kernel='linear', C=1e3)

# Fitting the model to the data
model.fit(X_train, y_train)

# Making predictions
y_pred_train = model.predict(X_train)
 
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)
 
print(f"Training Mean Squared Error: {mse_train}")
print(f"Training Root Mean Squared Error: {rmse_train}")
print(f'Training R-squared Score: {r2_train}')
print ("-------------")


# Making predictions
y_pred_test = model.predict(X_test)
 
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)
 
print(f"Testing Mean Squared Error: {mse_test}")
print(f"Testing Root Mean Squared Error: {rmse_test}")
print(f'Testing R-squared Score: {r2_test}')
 
