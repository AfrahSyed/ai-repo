import pandas as pd
from sklearn.metrics import r2score
import joblib
import numpy as np
from sklearn.model_selection import splittraintest
from sklearn.metrics import meansqe
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


file = 'Dataset.csv'
dataf = pd.read_csv(file)
dataf.columns = dataf.columns.str.strip()


# Preprocessing Step 1: Remove the "opponents" columns
dataf = dataf.loc[:, ~dataf.columns.str.contains('Opponent')]

# Preprocessing Step 2: Remove rows where 'Angle' has scientific notation values like -4.11E-04
dataf = dataf[~dataf['Angle'].astype(str).str.contains('E')]

# Preprocessing Step 3: Define input features (X) and output features (y)
# All the track columns and TrackPosition will be kept as input features
X = dataf.drop(columns=['CurrentLapTime', 'Damage', 'DistanceFromStart',
       'DistanceCovered', 'FuelLevel', 'LastLapTime', 'RacePosition', 
        'TrackPosition', 'Acceleration', 'Braking', 'Clutch', 'Gear_Output', 'Steering'])  # Input features to drop
# print(X.columns)
y = dataf[['Acceleration', 'Braking', 'Clutch', 'Steering', 'Gear_Output']]  # Output features

# Preprocessing Step 4: Normalize the features (important for neural networks)
scalervals = StandardScaler()
X_scaledval = scalervals.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = splittraintest(X_scaledval, y, tsize=0.2, random_state=42)

# Step 5: Define and train the MLP model
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predy = model.predict(X_test)
mse = meansqe(y_test, predy)
print(f"Mean Squared Error: {mse}")

# Optional: Check the performance for each output
# Evaluate the model for each output feature
for feat in y.columns:
    # Make predictions for the specific feature
    predfeaturey = model.predict(X_test)[:, y.columns.get_loc(feat)]  # Extract predictions for that feature
    truefeaturey = y_test[feat]  # Get the true values for that feature
    
    # Calculate R² score manually
    s = r2score(truefeaturey, predfeaturey)
    
    print(f"Performance for {feat}:")
    print(f"  R2 score: {s}")


# Save the trained model and scaler for later use
joblib.dump(model, 'mlp_model.pkl')  # Save the model
joblib.dump(scalervals, 'scaler.pkl')  # Save the scaler

