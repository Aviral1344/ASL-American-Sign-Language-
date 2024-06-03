# ASL (American Sign Language)

This repository contains a Convolutional Neural Network (CNN) model designed to recognize letters in the American Sign Language (ASL) alphabet from images.

## Repository Structure

- `asl-cnn-model.ipynb`: The Jupyter Notebook where all the model development, training, and evaluation are performed.
- `model/`: This folder contains the trained model file `model.keras`.

## Model Description

The model is a CNN built using TensorFlow and Keras. It takes an image as input and predicts the corresponding letter in the ASL alphabet.

### Model Architecture

The model consists of several layers designed to extract features from the input images and classify them into one of the 29 classes. Here is a detailed breakdown of the model architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.losses import CategoricalCrossentropy

num_classes = 29  # Number of classes in the dataset

model = Sequential([
    Input(shape=(64, 64, 1), name='input_layer'),
    
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    
    GlobalAveragePooling2D(),
    
    BatchNormalization(),
    
    Dense(512, activation='relu'),
    
    Dropout(0.3),
    
    Dense(512, activation='relu'),
    
    Dense(num_classes, activation='softmax', name='output_layer')
])

model.compile(optimizer=Adamax(learning_rate=1e-3), loss=CategoricalCrossentropy(), metrics=['accuracy'])
```

### Explanation of Each Layer

1. **Input Layer**: 
   - `Input(shape=(64, 64, 1), name='input_layer')`: Specifies the input shape of the images (64x64 pixels, grayscale).

2. **Convolutional Layers**:
   - `Conv2D(16, (3, 3), activation='relu')`: First convolutional layer with 16 filters of size 3x3, using ReLU activation.
   - `MaxPooling2D(pool_size=(3, 3))`: First max-pooling layer with a pool size of 3x3 to reduce spatial dimensions.
   - `Conv2D(32, (3, 3), activation='relu')`: Second convolutional layer with 32 filters.
   - `MaxPooling2D(pool_size=(3, 3))`: Second max-pooling layer.
   - `Conv2D(64, (3, 3), activation='relu')`: Third convolutional layer with 64 filters.
   - `MaxPooling2D(pool_size=(3, 3))`: Third max-pooling layer.

3. **Global Average Pooling**:
   - `GlobalAveragePooling2D()`: Reduces each feature map to a single value, which helps in flattening the output while retaining the spatial information.

4. **Batch Normalization**:
   - `BatchNormalization()`: Normalizes the output of the previous layers to improve training stability and performance.

5. **Fully Connected Layers**:
   - `Dense(512, activation='relu')`: First fully connected layer with 512 neurons and ReLU activation.
   - `Dropout(0.3)`: Dropout layer with a dropout rate of 30% to prevent overfitting.
   - `Dense(512, activation='relu')`: Second fully connected layer with 512 neurons.

6. **Output Layer**:
   - `Dense(num_classes, activation='softmax', name='output_layer')`: Output layer with a softmax activation function to classify the input into one of the 29 classes.

### Compilation

The model is compiled with the Adamax optimizer, categorical crossentropy loss, and accuracy as a metric:

```python
model.compile(optimizer=Adamax(learning_rate=1e-3), loss=CategoricalCrossentropy(), metrics=['accuracy'])
```

### Dataset

The dataset used for training consists of over 90,000 images spanning 29 classes: the alphabets A-Z, space, nothing, and backspace.

## Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- Jupyter Notebook (for running the `.ipynb` file)

### Loading the Model

You can load the pre-trained model using TensorFlow's `load_model` function and make predictions using the `predict` function.

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model/model.keras')

# Function to predict the ASL letter from an image
def predict_asl_letter(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch axis
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1)
    return predicted_class.numpy()[0]

# Example usage
image_path = 'path_to_your_image.jpg'
predicted_letter = predict_asl_letter(image_path)
print(f'The predicted ASL letter is: {predicted_letter}')
```

### Running the Notebook

To understand the model development process and experiment with the code, open the `asl-cnn-model.ipynb` file in Jupyter Notebook.

```bash
jupyter notebook asl-cnn-model.ipynb
```
