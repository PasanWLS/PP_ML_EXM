import tensorflow as tf
from tensorflow.keras import datasets, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Ensure that PIL.Image is imported correctly
from PIL import Image 

# Setting key parameters
validation_split = 0.2  # Example validation split
img_size = 128  # Example image size, adjust as needed
m = "best_model.h5"  # Model file path
d = "./dataset_directory"  # Dataset directory
nb_epochs = 10  # Number of epochs, adjust as needed

# Data augmentation and splitting into training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    d,
    target_size=(img_size, img_size),
    class_mode='binary',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    d,
    target_size=(img_size, img_size),
    class_mode='binary',
    subset='validation'
)

# Building the CNN model
model = models.Sequential()

# Convolutional layer and max-pooling layer 1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D((2, 2)))

# Convolutional layer and max-pooling layer 2
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening the result to feed into the dense layer
model.add(Flatten())

# Hidden layer with 32 neurons
model.add(Dense(32, activation='relu'))

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Checkpoint to save the best model based on validation accuracy
checkpoint_callback = ModelCheckpoint(
    filepath=m,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback],
    epochs=nb_epochs
)

# Save the final model
model.save("best.h5")

# Plotting the training accuracy and validation accuracy
plt.plot(history.history['accuracy'], color='teal', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='red', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Load the saved model for prediction
model = tf.keras.models.load_model(m)

# Labels of the classes
labels = train_generator.class_indices
print(labels)

# Path to test images
path = './test/' 
k = []  # Array for images
names = []  # Array for image names

# Loading test images
for filename in os.listdir(path):
    p = os.path.join(path, filename)
    if 'jpg' in p:
        c = cv2.imread(p)
        c = cv2.resize(c, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        k.append(c)
        names.append(filename)

k = np.array(k)
print(k.shape)
print(np.size(names))

# Class names (0: Negative, 1: Positive)
class_names = [0, 1]

# Predict the class labels for the test set
predicted_labels = (model.predict(k) > 0.5).astype("int32")
predicted_labels = predicted_labels.flatten()

# Map the predicted labels to class names
predicted_labels = [class_names[i] for i in predicted_labels]
print(predicted_labels)

# Extract true labels from file names
test_arr = []
for i in names:
    if i[0] == 'n': 
        test_arr.append(0)
    elif i[0] == 'p': 
        test_arr.append(1)

test_arr = np.array(test_arr)

# Evaluate the model on test images
if len(test_arr) == len(k):
    score = model.evaluate(k, test_arr)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
else:
    print("Mismatch between test images and labels.")
