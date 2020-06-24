import tensorflow as tf
#keras is an hihglevel API for tensorflow, to write less code, no def tensors
from tensorflow import keras
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0
print(len(train_images))
# for i in range(10):
#     print(train_images[i])
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.show()

#build a NN with a sequence of layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")#sum of all = 1, i.e probability
])

#configure model of training
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#epoch is the no. of times model is going to see the information(images)
#the reason we do this, is coz order in which the images come in inflence how paratemetrs are tweaked
#thus we send images in arbitrary order each epoch
model.fit(train_images, train_labels, epochs=10)

#test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc:", test_acc)