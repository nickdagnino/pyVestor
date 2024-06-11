import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the training data
train_df = pd.read_csv('ClassificationTest\\training_data.csv')
train_x = train_df[['sepal_length', 'sepal_width', 'petal_length', 'pedal_width']]
train_y = to_categorical(train_df['class'])

# Load the test data
test_df = pd.read_csv('ClassificationTest\\test_data.csv')
test_x = test_df[['sepal_length', 'sepal_width', 'petal_length', 'pedal_width']]
test_y = to_categorical(test_df['class'])

# Create a Sequential model
model = Sequential()

# Add an input layer 
model.add(Dense(12, input_shape=(4,), activation='relu'))

# Add four hidden layer 
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=15, batch_size=3, verbose=1)

# Evaluate the model
scores = model.evaluate(test_x, test_y, verbose=1)
print(f"Accuracy: {scores[1]*100}")
