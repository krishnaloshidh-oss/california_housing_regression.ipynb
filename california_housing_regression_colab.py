# California Housing Regression Assignment
# You can copy this code into Google Colab cell by cell, or upload this .py file.

# 1. Import required libraries
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 2. Load the California Housing dataset
# data_home stores the downloaded dataset in the current working folder.
housing = fetch_california_housing(as_frame=True, data_home="./sklearn_data")

X = housing.data
y = housing.target


# 3. Convert to pandas DataFrame
df = X.copy()
df["MedHouseVal"] = y

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset information:")
print(df.info())


# 4. Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


# 5. Basic statistical summary
print("\nStatistical summary:")
print(df.describe())


# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 7. Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(kernel="rbf")
}


# 8. Train and evaluate each model
results = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2
    })


# 9. Display comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModel Performance Comparison:")
print(results_df)


# 10. Identify best and worst models
best_model = results_df.iloc[0]
worst_model = results_df.iloc[-1]

print("\nBest-performing model:")
print(best_model)

print("\nWorst-performing model:")
print(worst_model)


# 11. Short conclusion
print("\nConclusion:")
print(
    "The best-performing model is the model with the highest R2 score and lowest "
    "MSE/MAE. In this run, Random Forest usually performs best because it captures "
    "non-linear relationships and feature interactions. Linear Regression often "
    "performs worst because it assumes a simple linear relationship."
)
