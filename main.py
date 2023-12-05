import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Assuming the data is loaded into a pandas DataFrame `df` with the correct columns
df = pd.read_csv('training_data/all_alpha_23.csv')


def convert_mpg(value):
    try:
        return float(value)
    except ValueError:  # if value is not just a float, it's expected to be a string like '25/60'
        numbers = list(map(float, value.split('/')))
        return sum(numbers) / len(numbers)  # return the average of the numbers


# Apply this conversion to the MPG columns
df['City MPG'] = df['City MPG'].apply(convert_mpg)
df['Hwy MPG'] = df['Hwy MPG'].apply(convert_mpg)
df['Cmb MPG'] = df['Cmb MPG'].apply(convert_mpg)

# Now you can safely convert to float
y = df[['City MPG', 'Hwy MPG', 'Cmb MPG']].values

# Preprocess the data
# Convert categorical columns using one-hot encoding
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(
    df[['Model', 'Trans', 'Drive', 'Fuel', 'Cert Region', 'Stnd', 'Veh Class', 'SmartWay']]).toarray()

# Scale numerical features
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[['Displ', 'Cyl', 'Air Pollution Score', 'Greenhouse Gas Score']])

# Combine numerical and categorical features
X = np.hstack([scaled_numerical, encoded_categorical])

# Target variables
y = df[['City MPG', 'Hwy MPG', 'Cmb MPG']].astype(float).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for the CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the CNN model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3))  # Output layer for 'City MPG', 'Hwy MPG', 'Cmb MPG'

# Compile the model
model.compile(optimizer=Adam(), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Save the model and the scalers
model.save('cnn_model')
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder.categories_, f)
with open('encoder.pkl', 'rb') as f:
    categories = pickle.load(f)
np.save('scaler.npy', [scaler.mean_, scaler.scale_])

# Generate predictions
predictions = model.predict(X_test)

# Convert the predictions and actual values to DataFrames for easier comparison and export
predictions_df = pd.DataFrame(predictions, columns=['Predicted City MPG', 'Predicted Hwy MPG', 'Predicted Cmb MPG'])
actuals_df = pd.DataFrame(y_test, columns=['Actual City MPG', 'Actual Hwy MPG', 'Actual Cmb MPG'])
comparison_df = pd.concat([actuals_df, predictions_df], axis=1)

# Save the comparison to a CSV file
comparison_df.to_csv('predictions_comparison.csv', index=False)

# Plot training & validation loss values

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()