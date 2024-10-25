import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt

# Enhanced data preparation function with additional features
def prepare_data():
    # Create a sample DataFrame (Replace with your actual data loading method)
    np.random.seed(0)
    dates = pd.date_range(start="2023-01-01", periods=365, freq='D')
    data = {
        'date': dates,
        'temperature': np.random.normal(15, 10, 365),
        'solar_radiation': np.random.normal(200, 50, 365),
        'wind_speed': np.random.normal(10, 2, 365),
        'humidity': np.random.normal(70, 10, 365),
        'precipitation': np.random.normal(5, 2, 365),
        'energy_consumption': np.random.normal(2000, 500, 365)
    }
    df = pd.DataFrame(data)
    
    # Extract time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Adding lag features for energy consumption
    df['lag_1'] = df['energy_consumption'].shift(1)
    df['lag_7'] = df['energy_consumption'].shift(7)
    df = df.dropna()  # Drop rows with NaN values due to shifting
    
    return df

# Load and prepare data
df = prepare_data()

# Define features and target variable
X = df[['temperature', 'solar_radiation', 'wind_speed', 'humidity', 'precipitation',
        'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_7']]
y = df['energy_consumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Updated hyperparameter tuning with 'max_features' only allowing 'sqrt' or 'log2'
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Removed 'auto' to avoid errors
}

model = RandomForestRegressor(random_state=0)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=0, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# Best model from the search
best_model = random_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Parameters:", random_search.best_params_)
print("Mean Absolute Error after tuning:", mae)
print("Root Mean Squared Error after tuning:", rmse)

# Feature Importance Visualization
feature_importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.title("Feature Importances")
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Energy Consumption", color="blue")
plt.plot(y_pred, label="Predicted Energy Consumption", color="red", alpha=0.7)
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Test Samples")
plt.ylabel("Energy Consumption")
plt.legend()
plt.show()
