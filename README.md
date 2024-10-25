# **Energy Consumption Prediction Model**

This project is a machine learning model built in Python to predict energy consumption based on weather conditions, historical energy usage, and time-based features. The model is aimed at aiding sustainable energy and industry management by providing insights into future energy demands.

## **Table of Contents**
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results and Evaluation](#results-and-evaluation)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

## **Features**
- Predicts daily energy consumption using historical data and weather conditions.
- Supports feature engineering with time-based features and lagged data to improve model accuracy.
- Performs hyperparameter tuning using `RandomizedSearchCV` to optimize model performance.
- Generates visualizations for feature importance and compares actual vs. predicted energy usage.

## **Technologies Used**
- **Python 3.x**
- **Libraries**:
  - **pandas**: Data manipulation and analysis
  - **numpy**: Numerical computing
  - **scikit-learn**: Machine learning model and evaluation metrics
  - **matplotlib**: Data visualization

## **Getting Started**

### **Prerequisites**
- Python 3.x installed on your machine.
- Basic knowledge of machine learning and data science concepts.

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-consumption-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd energy-consumption-prediction
   ```
3. Install the required Python packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

## **Usage**
1. Prepare the dataset:
   - The code currently generates a sample dataset with weather and energy usage data. Replace this with actual data if available.

2. Run the model:
   ```bash
   python energy_consumption_model.py
   ```

3. Follow the output to view:
   - Best hyperparameters after tuning
   - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the model
   - Feature importance plot and predictions vs. actual energy consumption plot

## **Model Details**
- **Model Type**: Random Forest Regressor
- **Feature Engineering**:
  - Weather-based features: `temperature`, `solar_radiation`, `wind_speed`, `humidity`, `precipitation`
  - Time-based features: `day_of_week`, `month`, `day_of_year`, `is_weekend`
  - Lag features: `lag_1`, `lag_7` (previous day and week energy consumption)
- **Hyperparameter Tuning**:
  - Uses `RandomizedSearchCV` to optimize parameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.

## **Results and Evaluation**
- The model is evaluated using:
  - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values.
  - **Root Mean Squared Error (RMSE)**: Square root of the average squared differences between predicted and actual values.
  
- **Visualization**:
  - Feature Importance: Highlights the most influential features for energy consumption.
  - Actual vs. Predicted: Compares the model’s predictions with real values to assess accuracy visually.

## **Future Enhancements**
- **Additional Features**: Include more detailed weather variables or other external factors influencing energy usage.
- **Different Models**: Experiment with other models like Gradient Boosting, XGBoost, or Neural Networks.
- **Real-Time Data Integration**: Connect to real-time data sources for weather and energy usage.
- **Advanced Time-Series Analysis**: Implement methods like ARIMA or LSTM for better time-based predictions.

## **Contact**
For questions or suggestions, please feel free to contact:
- **Email**: karamimohammadamin754@gmail.com
- **GitHub**: [Eventhorizon32](github.com/Eventhorizon32)

