import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("water_quality_data.csv")

# Split data into training and testing sets
train_data = df[:int(len(df) * 0.8)]
test_data = df[int(len(df) * 0.8):]

X_train = train_data.drop(["BOD"], axis=1)
y_train = train_data["BOD"]

X_test = test_data.drop(["BOD"], axis=1)
y_test = test_data["BOD"]

# Initialize models
models = [LinearRegression(), SVR(), DecisionTreeRegressor()]

start_time = time.time()

# Evaluate each model
errors = []
for model in tqdm(models, desc="Evaluating models"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

# Choose the model with the lowest mean squared error
best_model = models[np.argmin(errors)]

end_time = time.time()

print("Time spent on analysis: {:.2f} seconds".format(end_time - start_time))
