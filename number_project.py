import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

#load the data 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#normalize the data between 0 and 1 
train_images = train_images / 255.0
test_images = test_images / 255.0 

#reshape the data (processing) 
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#setting up the neural network's layers 
model = models.Sequential([
	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
	layers.MaxPooling2D((2, 2)),
	layers.Conv2D(32, (3, 3), activation='relu'),
	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(10, activation='softmax')
])

#show a summary of the model
model.summary() 
model.compile(	optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

predictions = model.predict(test_images)
print(f"Prediction for the first test image: {np.argmax(predictions[0])}") 

model.save('digit_recognition_model.keras') 
print("Model saved to 'digit_recognition_model.keras'")

