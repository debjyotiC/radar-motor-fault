import numpy as np
import seaborn as sns
from os import listdir
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print(tf.__version__)

range_doppler_features = np.load("data/npz_files/radar-balanced-motor.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

classes_values = listdir("data/radar-motor")
classes = len(classes_values)

print(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

# Splitting data into train, validation, and test sets
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))

# Add channel dimension for CNN
x_train = tf.expand_dims(x_train, axis=-1)
x_val = tf.expand_dims(x_val, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

BATCH = 50

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH)

# Define CNN-GRU Model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((16, 128, 1), input_shape=x_train.shape[1:]),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    # tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Reshape((-1, 64)),  # Adjust based on feature map size
    tf.keras.layers.GRU(30, return_sequences=False),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(classes, activation='softmax')
])

model.summary()

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved-model/best-radar-motor-fault.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping, checkpoint]
)

# Load the best model
model = tf.keras.models.load_model("saved-model/best-radar-motor-fault.keras")

# Evaluate on test data
predicted_labels = model.predict(x_test)
actual_labels = y_test

label_predicted = np.argmax(predicted_labels, axis=1)
label_actual = np.argmax(actual_labels, axis=1)

# Compute confusion matrix
results = confusion_matrix(label_actual, label_predicted)

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.max(acc), 3)}")
print(f"Validation Accuracy: {round(np.max(val_acc), 3)}")

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot loss
axs[0].plot(epochs, loss, '-', label='Training Loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')

# Plot accuracy
axs[1].plot(epochs, acc, '-', label='Training Accuracy')
axs[1].plot(epochs, val_acc, 'b', label='Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')

plt.show()

# Confusion Matrix
ax = plt.subplot()
sns.heatmap(results, annot=True, annot_kws={"size": 20}, ax=ax, fmt='g', cmap='Blues')

# Labels, title, and ticks
ax.set_xlabel('Predicted Labels', fontsize=12)
ax.set_ylabel('True Labels', fontsize=12)
ax.set_title(f'Confusion Matrix for RADAR Data (Best Accuracy: {round(np.max(acc), 3)})')
ax.xaxis.set_ticklabels(classes_values, fontsize=15)
ax.yaxis.set_ticklabels(classes_values, fontsize=15)
plt.tight_layout()
plt.savefig("data/images/cm.png", dpi=600)
plt.show()
