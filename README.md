# IMDB-Dataset-Review-Classification-Using-LSTM-GRU-and-Standard-RNN

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, Dropout, SimpleRNN
import matplotlib.pyplot as plt

# Load and preprocess the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Get final validation loss and accuracy
lloss = history.history['loss'][-1]
lacc = history.history['accuracy'][-1]

# Evaluate the model on test data and print final test loss and accuracy
ltest_loss, ltest_acc = model.evaluate(x_test, y_test)

# Print final validation and test loss/accuracy
print(f'Final Training Loss: {lloss:.4f}')
print(f'Final Training Accuracy: {lacc:.4f}')
print(f'Test Loss: {ltest_loss:.4f}')
print(f'Test Accuracy: {ltest_acc:.4f}')

# Predict for a sample test sequence
test_sequence = np.reshape(x_test[7], (1, -1))
predictions = model.predict(test_sequence)[0]
print('Positive Review' if int(predictions[0]) == 1 else 'Negative Review')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU, Dropout, SimpleRNN
import matplotlib.pyplot as plt

# Load and preprocess the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(GRU(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Get final training loss and accuracy
gloss = history.history['loss'][-1]
gacc = history.history['accuracy'][-1]

# Evaluate the model on test data and print final test loss and accuracy
gtest_loss, gtest_acc = model.evaluate(x_test, y_test)

# Print final validation and test loss/accuracy
print(f'Final Training Loss: {gloss:.4f}')
print(f'Final Training Accuracy: {gacc:.4f}')
print(f'Test Loss: {gtest_loss:.4f}')
print(f'Test Accuracy: {gtest_acc:.4f}')

# Predict for a sample test sequence
test_sequence = np.reshape(x_test[7], (1, -1))
predictions = model.predict(test_sequence)[0]
print('Positive Review' if int(predictions[0]) == 1 else 'Negative Review')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()



# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(SimpleRNN(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Get final validation loss and accuracy
sloss = history.history['loss'][-1]
sacc = history.history['accuracy'][-1]

# Evaluate the model on test data and print final test loss and accuracy
stest_loss, stest_acc = model.evaluate(x_test, y_test)

# Print final validation and test loss/accuracy
print(f'Final Training Loss: {sloss:.4f}')
print(f'Final Training Accuracy: {sacc:.4f}')
print(f'Test Loss: {stest_loss:.4f}')
print(f'Test Accuracy: {stest_acc:.4f}')

# Predict for a sample test sequence
test_sequence = np.reshape(x_test[7], (1, -1))
predictions = model.predict(test_sequence)[0]
print('Positive Review' if int(predictions[0]) == 1 else 'Negative Review')


# Create subplots for training and testing loss and accuracy
plt.figure(figsize=(16, 12))

# 1. Training Accuracy Comparison
plt.subplot(2, 2, 1)
plt.bar(['LSTM', 'GRU', 'Simple RNN'], [lacc, gacc, sacc], color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title('Training Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')

# 2. Training Loss Comparison
plt.subplot(2, 2, 2)
plt.bar(['LSTM', 'GRU', 'Simple RNN'], [lloss, gloss, sloss], color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title('Training Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Models')

# 3. Testing Accuracy Comparison
plt.subplot(2, 2, 3)
plt.bar(['LSTM', 'GRU', 'Simple RNN'], [ltest_acc, gtest_acc, stest_acc], color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title('Testing Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')

# 4. Testing Loss Comparison
plt.subplot(2, 2, 4)
plt.bar(['LSTM', 'GRU', 'Simple RNN'], [ltest_loss, gtest_loss, stest_loss], color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title('Testing Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Models')

plt.tight_layout()
plt.show()
