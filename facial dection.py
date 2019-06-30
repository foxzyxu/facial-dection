# %% cell 0
import pandas as pd
import numpy as np

train = pd.read_csv('training.csv')
#print(train.info())
#print(train.isnull().any().value_counts())
train.fillna(method='ffill', inplace=True)
#print(train.isnull().any().value_counts())
img = []
for i in range(len(train)):
  temp = train['Image'][i].split(' ')
  temp = ['0' if x == '' else x for x in temp]
  img.append(temp)

# %% cell 1
import matplotlib.pyplot as plt

image = np.array(img,dtype = 'float')
X = image.reshape(-1,96,96,1)
plt.imshow(X[5].reshape(96,96),cmap='gray')
plt.show()

# %% cell 2
facial_pts_data = train.drop(['Image'], axis=1)
Y = []
for i in range(len(facial_pts_data)):
    Y.append(facial_pts_data.iloc[i])
Y = np.array(Y, dtype='float')

# %% cell 3
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import  Convolution2D, BatchNormalization, Flatten, Dense, Dropout, MaxPool2D

model = Sequential()
model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
#model.summary()

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(X, Y, epochs = 3, batch_size = 256, validation_split = 0.1)

# %% cell 4
model.save('facial dection.h5')