import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir

# Load data
range_doppler_features = np.load("data/npz_files/radar-balanced-motor.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

# Get class count and names (sorted)
classes_values = sorted(listdir("data/radar-motor"))
num_classes = len(classes_values)
print("Classes:", classes_values)

# One-hot encode labels (assuming labels start from 1)
y_data = tf.keras.utils.to_categorical(y_data - 1, num_classes)

# Prepare sequences of 4 frames
sequence_length = 4
x_sequences = []
y_sequences = []

for i in range(len(x_data) - sequence_length + 1):
    x_seq = x_data[i:i + sequence_length]
    y_seq = y_data[i + sequence_length - 1]  # Label from last frame
    x_sequences.append(x_seq)
    y_sequences.append(y_seq)

x_sequences = np.array(x_sequences)  # (samples, 4, 64, 64)
y_sequences = np.array(y_sequences)
x_sequences = np.expand_dims(x_sequences, axis=-1) # Add channel dim: (samples, 4, 64, 64, 1)

# Train/Val/Test split
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

x_train, x_temp, y_train, y_temp = train_test_split(x_sequences, y_sequences, test_size=1 - train_ratio, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio / (validation_ratio + test_ratio), shuffle=True)

# TF datasets
BATCH_SIZE = 60
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# --- Squeeze-and-Excitation Block ---
def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(channels // reduction, activation='relu')(se)
    se = tf.keras.layers.Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

# --- Shared CNN Feature Extractor ---
def create_shared_cnn():
    input_layer = tf.keras.Input(shape=(16, 128, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    return tf.keras.models.Model(input_layer, x, name="shared_cnn")

# --- Temporal Attention Layer ---
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.q_dense = tf.keras.layers.Dense(feature_dim)
        self.k_dense = tf.keras.layers.Dense(feature_dim)
        self.v_dense = tf.keras.layers.Dense(feature_dim)

    def call(self, x):
        q = self.q_dense(x)
        k = self.k_dense(x)
        v = self.v_dense(x)

        score = tf.matmul(q, k, transpose_b=True)
        score /= tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        weights = tf.nn.softmax(score, axis=-1)
        attention_output = tf.matmul(weights, v)
        return attention_output

# --- Model Assembly ---
def build_temporal_attention_model():
    shared_cnn = create_shared_cnn()

    input_sequence = tf.keras.Input(shape=(4, 16, 128, 1))  # 4 time steps
    features = tf.keras.layers.TimeDistributed(shared_cnn)(input_sequence)  # (batch, 4, features)

    attention_features = TemporalAttention()(features)  # (batch, 4, features)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_features)

    x = tf.keras.layers.Dense(128, activation='relu')(pooled)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.models.Model(inputs=input_sequence, outputs=output)

# Build and compile model
model = build_temporal_attention_model()
model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved-model/best-radar-motor-fault.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping, checkpoint]
)

# Load best model
model = tf.keras.models.load_model("saved-model/best-radar-motor-fault.keras", custom_objects={'TemporalAttention': TemporalAttention})

# Evaluate
predicted_labels = model.predict(x_test)
label_predicted = np.argmax(predicted_labels, axis=1)
label_actual = np.argmax(y_test, axis=1)

# Confusion matrix & accuracy
results = confusion_matrix(label_actual, label_predicted)
acc_score = accuracy_score(label_actual, label_predicted)
print("Test Accuracy:", round(acc_score, 3))

# Training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot history
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(epochs, loss, '-', label='Training Loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
axs[0].set_title("Loss")
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, acc, '-', label='Training Accuracy')
axs[1].plot(epochs, val_acc, 'b', label='Validation Accuracy')
axs[1].set_title("Accuracy")
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix Plot
ax = plt.subplot()
sns.heatmap(results, annot=True, fmt='g', cmap='Blues', ax=ax, annot_kws={"size": 14})

ax.set_xlabel('Predicted Labels', fontsize=12)
ax.set_ylabel('True Labels', fontsize=12)
ax.set_xticklabels(classes_values, fontsize=12)
ax.set_yticklabels(classes_values, fontsize=12)
plt.tight_layout()
plt.savefig("data/images/cm.png", dpi=600)
plt.show()
