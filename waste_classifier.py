# waste_classifier.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)
val_data = train_datagen.flow_from_directory(
    'dataset',
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# Model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("waste_classifier_model.h5")
print("✅ Model trained and saved successfully!")
