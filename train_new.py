import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from sklearn.metrics import meansqe
from sklearn.metrics import r2score
from sklearn.model_selection import splittraintest
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


file = 'Dataset.csv'
dataf = pd.read_csv(file)
dataf.columns = dataf.columns.str.strip()

# Preprocessing Step 1: Remove the "opponents" columns
dataf = dataf.loc[:, ~dataf.columns.str.contains('Opponent')]

# Preprocessing Step 2: Remove rows where 'Angle' has scientific notation values like -4.11E-04
dataf = dataf[~dataf['Angle'].astype(str).str.contains('E')]

# Preprocessing Step 3: Define input features (X) and output features (y)
X = dataf.drop(columns=['CurrentLapTime', 'Damage', 'DistanceFromStart',
       'DistanceCovered', 'FuelLevel', 'Gear', 'LastLapTime', 'RacePosition', 
        'TrackPosition', 'WheelSpinVelocity_1',
       'WheelSpinVelocity_2', 'WheelSpinVelocity_3', 'WheelSpinVelocity_4',
       'Z', 'Acceleration', 'Braking', 'Clutch', 'Gear_Output', 'Steering'])  # Input features to drop

y = dataf[['Acceleration', 'Braking', 'Clutch', 'Steering', 'Gear_Output']]  # Output features

# Preprocessing Step 4: Use MinMaxScaler instead of StandardScaler
scaledval = MinMaxScaler()
X_scaledval = scaledval.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = splittraintest(X_scaledval, y, tsize=0.2, random_state=42)

# Step 5: Define and train the MLP model with more layers and neurons
model = MLPRegressor(random_state=42)

# Simplified hyperparameter grid to speed up search
grid = {
    'hidden layer sizes': [(256, 128)],  # Reduced architectures
    'alpha': [0.0001],  # Regularization parameter
    'learning_rate': ['constant'],  # Removed 'adaptive' for simplicity
    'max iterations': [500]  # Reduced iterations
}

# Using GridSearchCV with the reduced hyperparameters
gsearch = GridSearchCV(model, grid, cv=5, jobs=-1, verbose=10)
gsearch.fit(X_train, y_train)

# Train the model with the best parameters
modelbest = gsearch.best_estimator_
# Best parameters
print("Best parameters: ", gsearch.best_params_)


# Step 6: Evaluate the model
y_pred = modelbest.predict(X_test)
mse = meansqe(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Evaluate performance for each output
for feat in y.columns:
    predfeaturey = modelbest.predict(X_test)[:, y.columns.get_loc(feat)]
    truefeaturey = y_test[feat]
    s = r2score(truefeaturey, predfeaturey)
    print(f"Performance for {feat}:")
    print(f"  R2 score: {s}")

# Save the best model and scaler for later use
joblib.dump(modelbest, 'best_mlp_model.pkl')
joblib.dump(scaledval, 'minmax_scaler.pkl')

