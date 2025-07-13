# Guide to Building a House Price Predictor

# Step 1: Import necessary libraries
# Import pandas for data manipulation and scikit-learn for building the model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
# Use pandas to read the 'train.csv' file into a DataFrame.
df = pd.read_csv('train.csv')

# Step 3: Prepare the data
# Separate the features (the inputs) from the target variable (the output).
# Handle categorical features using one-hot encoding.

# Select numerical features
numerical_features = df[['LotArea', 'TotRmsAbvGrd']]

# Select categorical features and apply one-hot encoding
categorical_features = pd.get_dummies(df['Neighborhood'], prefix='Neighborhood')

# Combine numerical and one-hot encoded categorical features
X = pd.concat([numerical_features, categorical_features], axis=1)

# Our target is 'SalePrice'. Let's call this variable 'y'.
y = df['SalePrice']

# Step 4: Split the data into training and testing sets
# Use scikit-learn's train_test_split function to divide the data.
# This helps us evaluate the model on data it has never seen before.
# A common split is 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Step 5: Create and train the model
# Create an instance of the Linear Regression model from scikit-learn.
# Train the model using the training data (X_train, y_train) with the .fit() method.
reg = LinearRegression()
reg.fit(X_train, y_train)

# Step 6: Make predictions on the test set
# Use the trained model to make predictions on the test features (X_test).
y_pred = reg.predict(X_test)

# Step 7: Evaluate the model
# Compare the model's predictions with the actual prices of the test set (y_test).
# You can use metrics like Mean Squared Error (MSE) or R-squared to see how well the model performed.
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5 # Calculate RMSE
r2 = r2_score(y_test, y_pred) # Calculate R-squared

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 8: Make a prediction on new data (Optional)
# Create a new data point (e.g., a house with a specific area and number of rooms).
# Use the trained model to predict its price.
# Remember that the model expects the data in the same format as the training data.

# Example new data point
new_house_data = {
    'LotArea': [8500], # Example LotArea
    'TotRmsAbvGrd': [7], # Example TotRmsAbvGrd
    'Neighborhood': ['CollgCr'] # Example Neighborhood
}

new_house_df = pd.DataFrame(new_house_data)

# Apply one-hot encoding to the new data's Neighborhood
new_house_categorical = pd.get_dummies(new_house_df['Neighborhood'], prefix='Neighborhood')

# Combine numerical features with one-hot encoded categorical features for the new data
new_house_features = pd.concat([
    new_house_df[['LotArea', 'TotRmsAbvGrd']],
    new_house_categorical
], axis=1)

# Ensure the new data has all the same columns as the training data (X)
# Fill missing columns (neighborhoods not in new_house_data) with 0
new_house_features = new_house_features.reindex(columns=X.columns, fill_value=0)

# Make the prediction
predicted_price = reg.predict(new_house_features)

print(f"\nPredicted price for the new house: ${predicted_price[0]:,.2f}")
