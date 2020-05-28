#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.utils import to_categorical
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.datasets import mnist
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import random,time
#%%
input_size=(28,28)
EPOCHS = 45
INIT_LR = 1e-3
BS = 32
#%%model
def feed_forward_model():
    model=Sequential()
    model.add(Dense(258, input_dim=512, init="uniform",activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(128, init="uniform",activation="relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    return model

#%%
encoder=load_model('./encoder_512_v3.h5')
num_class=10

(x_train, trainY), (x_test, testY) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# %%

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
c_model=feed_forward_model()
c_model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# %%

# (trainX, testX, trainY, testY) = train_test_split(features,labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, num_classes=num_class)
testY = to_categorical(testY, num_classes=num_class)

#%%
trainX=encoder.predict(x_train_noisy)#train_featutes
testX =encoder.predict(x_test_noisy)#test_feature
# %%
H = c_model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=50, verbose=1 )

# %%
p=c_model.predict(testX)
y_pred=[]
y_test=[]
for i,v in enumerate(p):
    y_pred.append(np.argmax(v))
    y_test.append(np.argmax(testY[i]))


# %%
from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test, y_pred)

# %%
print(classification_report(y_test,y_pred))

# %%
input_shape = (28, 28, 1)
from keras.models import Model
from keras.layers import Input
inputs = Input(shape=input_shape, name='encoder_input')
classifier = Model(inputs, c_model(encoder(inputs)), name='classifier')
classifier.layers[1].trainable=False
classifier.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# %%

# %%
(loss, accuracy) = classifier.evaluate(x_test_noisy, testY,
	batch_size=128, verbose=1)

# %%
classifier.save('classification_model.h5')

# %%
