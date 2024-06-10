import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load the training data
train_df = pd.read_csv('training_data.csv')
train_x = train_df['sepal length', 'sepal width', 'petal length', 'pedal width']
train_y = to_categorical(train_df['class'])

# Load the test data
test_df = pd.read_csv('test_data.csv')
test_x = test_df['sepal length', 'sepal width', 'petal length', 'pedal width']
test_y = to_categorical(test_df['class'])

# Create a Sequential model
model = Sequential()

# Add an input layer 
model.add(Dense(12, input_dim=8, activation='relu'))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=150, batch_size=10, verbose=0)

# Evaluate the model
scores = model.evaluate(test_x, test_y, verbose=1)
print(f"Accuracy: {scores[1]*100}")
