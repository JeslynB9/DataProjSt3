import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
data = pd.read_csv("SDG_goal3_clean.csv")
X = data[["Health worker density, by type of occupation (per 10,000 population)::NURSEMIDWIFE"]]
y = data["Neonatal mortality rate (deaths per 1,000 live births)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the SVR model with a linear kernel
model = SVR(kernel='linear', C=1e3)

# Fitting the model to the data
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_train)
 
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)
 
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f'R-squared Score: {r2}')
 
