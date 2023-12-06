import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the data
df = pd.read_csv('training_data/data_from_20-24.csv')

# Function to convert MPG values like 20/30 to 30(average)
def convert_mpg(value):
    try:
        return float(value)
    except ValueError:
        numbers = list(map(float, value.split('/')))
        return sum(numbers) / len(numbers)

# Convert MPG fields
df['City MPG'] = df['City MPG'].apply(convert_mpg)
df['Hwy MPG'] = df['Hwy MPG'].apply(convert_mpg)
df['Cmb MPG'] = df['Cmb MPG'].apply(convert_mpg)

y = df[['City MPG', 'Hwy MPG', 'Cmb MPG']].values

# DNN
# Categorical
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(df[['Model', 'Trans', 'Drive', 'Fuel', 'Cert Region', 'Stnd', 'Veh Class', 'SmartWay']]).toarray()

# Numerical variables
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[['Displ', 'Cyl', 'Air Pollution Score', 'Greenhouse Gas Score']])

# Combine numerical and categorical features
X = np.hstack([scaled_numerical, encoded_categorical])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the DNN
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3))
model.compile(optimizer=Adam(), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

model.save('dnn_model')
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder.categories_, f)
with open('encoder.pkl', 'rb') as f:
    categories = pickle.load(f)

# Save the scaler
np.save('scaler.npy', [scaler.mean_, scaler.scale_])

# Generate predictions
predictions = model.predict(X_test)

# Save prediction data to CSV
predictions_df = pd.DataFrame(predictions, columns=['Predicted City MPG', 'Predicted Hwy MPG', 'Predicted Cmb MPG'])
actuals_df = pd.DataFrame(y_test, columns=['Actual City MPG', 'Actual Hwy MPG', 'Actual Cmb MPG'])
comparison_df = pd.concat([actuals_df, predictions_df], axis=1)
comparison_df.to_csv('predictions_comparison.csv', index=False)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

# Plotting for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plotting for MSE, RMSE, MAE
plt.figure(figsize=(10, 6))
plt.bar(['MSE', 'RMSE', 'MAE'], [mse, rmse, mae], color=['blue', 'green', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
for i, value in enumerate([mse, rmse, mae]):
    plt.text(i, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
plt.show()

# Plotting for MPG comparison
df = pd.read_csv('predictions_comparison.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
mpg_types = ['City', 'Hwy', 'Cmb']
for i, mpg in enumerate(mpg_types):
    ax[i].scatter(df[f'Actual {mpg} MPG'], df[f'Predicted {mpg} MPG'], alpha=0.5)
    ax[i].plot([df[f'Actual {mpg} MPG'].min(), df[f'Actual {mpg} MPG'].max()], [df[f'Actual {mpg} MPG'].min(), df[f'Actual {mpg} MPG'].max()], 'r--')
    ax[i].set_xlabel(f'Actual {mpg} MPG')
    ax[i].set_ylabel(f'Predicted {mpg} MPG')
    ax[i].set_title(f'{mpg} MPG Comparison')
plt.tight_layout()
plt.show()
