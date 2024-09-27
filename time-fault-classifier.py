import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed

# Define parameters
input_shape = (T, height, width, channels)  # e.g. (10, 64, 64, 1)
num_classes = Neu  # Number of classes for classification

# Build the hybrid CNN + LSTM model
model = Sequential()

# TimeDistributed CNN to extract features from each frame
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))  # Flatten each CNN output before feeding into LSTM

# LSTM layer to capture temporal dependencies between frames
model.add(LSTM(128, return_sequences=False))  # You can adjust LSTM units and set return_sequences=True if needed

# Dense layers
model.add(Dense(128, activation='relu'))  # Fully connected layer with ReLU
model.add(Dense(num_classes, activation='softmax'))  # Output layer with Softmax for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
