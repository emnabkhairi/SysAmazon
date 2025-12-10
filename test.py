# test.py
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the saved Keras model
model = tf.keras.models.load_model("artifacts/testmodel.h5")
print("Model loaded successfully!")
model.summary()

# Step 2: Load test data
test_df = pd.read_csv("artifacts/test.csv")
# Assuming first 2 columns are user_id and product_id
X_test = test_df.iloc[:, :2].values
y_test = test_df.iloc[:, -1].values  # last column = actual ratings

# Step 3: Make predictions
predictions = model.predict([X_test[:, 0], X_test[:, 1]])
print("\nFirst 10 predictions:")
print(predictions[:10])

# Step 4: Evaluate the model (optional)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nEvaluation Metrics:\nMSE: {mse}\nMAE: {mae}\nR2 Score: {r2}")
