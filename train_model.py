import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths & constants
DATASET_DIR = "dataset"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Create data generators (train/test split)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 80% train, 20% test
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

test_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, epochs=5, validation_data=test_data)

# Evaluate on test set
loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
model.save("model.h5")
print("✅ Trained model saved as model.h5")
