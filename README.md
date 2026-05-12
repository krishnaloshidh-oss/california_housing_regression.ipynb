Preprocessing Explanation

The California Housing dataset was loaded using fetch_california_housing() from sklearn.datasets. The dataset was then converted into a pandas DataFrame because DataFrames make it easier to view the data, check column names, inspect missing values, and understand statistical summaries.

Missing values were checked using isnull().sum(). The dataset did not contain missing values, but SimpleImputer was still used in the model pipeline with the median strategy. This makes the workflow safer and more reliable in case missing values appear later.

Feature scaling was performed using StandardScaler. Standardization converts features so that they have a mean of 0 and standard deviation of 1. This is important because the dataset contains features with different ranges, such as income, population, rooms, latitude, and longitude. Scaling is especially important for SVR because it is sensitive to feature magnitude.

The dataset was split into training and testing sets using an 80:20 ratio. The training data was used to train the models, while the testing data was used to evaluate model performance on unseen data.

Algorithm Explanations

Linear Regression:
Linear Regression finds the best-fitting straight-line relationship between the input features and the target value. It is simple, fast, and useful as a baseline model. It is suitable for this dataset as a starting point, but it may not perform best because housing prices often depend on non-linear relationships.

Decision Tree Regressor:
Decision Tree Regressor splits the data into smaller groups based on feature values and makes predictions using the average target value in each group. It can handle non-linear patterns and feature interactions, which makes it suitable for housing data. However, a single decision tree can overfit the training data.

Random Forest Regressor:
Random Forest builds multiple decision trees and combines their predictions by averaging. This reduces overfitting and usually improves accuracy compared to a single decision tree. It is suitable for this dataset because housing prices depend on many interacting features such as income, location, house age, and occupancy.

Gradient Boosting Regressor:
Gradient Boosting builds models one after another, where each new model tries to correct the errors made by the previous models. It performs well on structured datasets and can capture complex non-linear relationships. This makes it suitable for the California Housing dataset.

Support Vector Regressor (SVR):
SVR tries to find a function that predicts values within a certain error margin. With the RBF kernel, it can model non-linear relationships. SVR is suitable for this dataset because it can capture complex patterns, but it requires scaled features and may take more time on larger datasets.

Model Evaluation Explanation

The models were evaluated using three metrics:

Mean Squared Error (MSE):
MSE calculates the average squared difference between actual and predicted values. A lower MSE means better performance. It penalizes large errors more strongly.

Mean Absolute Error (MAE):
MAE calculates the average absolute difference between actual and predicted values. A lower MAE means the predictions are closer to the actual values.

R-squared Score (R²):
R² shows how much variation in the target variable is explained by the model. A higher R² score means better performance.

Comparison and Conclusion

Based on the results, the Random Forest Regressor was the best-performing model. It had the lowest MSE and MAE and the highest R² score. This means it predicted house prices more accurately than the other models. Random Forest performed well because it can capture non-linear relationships and interactions between different housing features.

The Linear Regression model was the worst-performing model. Although it is simple and useful as a baseline, it assumes a linear relationship between features and house prices. The California Housing dataset contains more complex patterns, so Linear Regression could not capture them as effectively as the tree-based ensemble models.

Therefore, the best model for this dataset is Random Forest Regressor, while the weakest model is Linear Regression.
