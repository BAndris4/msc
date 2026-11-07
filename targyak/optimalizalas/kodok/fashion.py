import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras.callbacks import EarlyStopping, LearningRateScheduler


num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


# Scale the images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# a learning rate scheduler
def scheduler1(epoch, lr):
    if epoch % 10 == 0 and epoch < 60:
        lr = lr * 0.92
    return lr


# the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


# parameters + training
batch_size = 1024
epochs = 100
lr = 0.001
opt = keras.optimizers.RMSprop(learning_rate=lr)
es = EarlyStopping(monitor='val_loss', verbose=0, patience=6)
lrsch = LearningRateScheduler(scheduler1)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,callbacks=[es])
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'bs= {batch_size}, lr= {lr}')
plt.legend(['train loss', 'validation loss'], loc='upper right')
plt.show()
