import os
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
# Visualization of sample predictions
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Image parameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Evaluation
loss, acc = model.evaluate(test_gen)
print(f'Test accuracy: {acc:.4f}')



# Get predictions
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_gen.classes

# Get file paths for test images
file_paths = [test_gen.filepaths[i] for i in range(len(y_true))]

# Show 8 random test images with predictions
num_images = 8
indices = np.random.choice(len(y_true), num_images, replace=False)
plt.figure(figsize=(16, 8))
for i, idx in enumerate(indices):
    img = plt.imread(file_paths[idx])
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    true_label = 'PNEUMONIA' if y_true[idx] == 1 else 'NORMAL'
    pred_label = 'PNEUMONIA' if y_pred[idx] == 1 else 'NORMAL'
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()