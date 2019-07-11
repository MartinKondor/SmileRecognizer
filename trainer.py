import gc
import os
from PIL import Image

import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import loadimg, DATA_DIR, IMG_SHAPE


NUMBER_OF_IMG_TO_LOAD = 1200

if not os.path.isdir(DATA_DIR):
    fetch_lfw_people(resize=.7)
    

# Filling these lists
NON_SMILE_IMAGES, SMILE_IMAGES = [], []

print('Load non smile list ...')
with open(os.path.join('data', 'NON-SMILE_list.txt'), 'r') as nonsmile_file:
    NON_SMILE_IMAGES = [line for line in nonsmile_file.read().splitlines() if line]

print('Load smile list ...')
with open(os.path.join('data', 'SMILE_list.txt'), 'r') as smile_file:
    SMILE_IMAGES = [line for line in smile_file.read().splitlines() if line]


print('Load & prepare images ...')
print('Number of NON_SMILE_IMAGES', len(NON_SMILE_IMAGES))
print('Number of SMILE_IMAGES', len(SMILE_IMAGES))

X = []
y = []

# load non smiling images
for i in tqdm(range(len(NON_SMILE_IMAGES))):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(NON_SMILE_IMAGES, i))
        y.append(0)
    except FileNotFoundError:
        continue


# load smiling images
for i in tqdm(range(len(SMILE_IMAGES))):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(SMILE_IMAGES, i))
        y.append(1)
    except FileNotFoundError:
        continue


# convert them to numpy.ndarray
X = np.array(X) / 255
y = np.array(y)  # 1 = Smile, 0 = Non-smile

# preprocessing
print('Preprocessing ...')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.08, random_state=1, shuffle=True)
del X, y, NON_SMILE_IMAGES, SMILE_IMAGES, DATA_DIR, NUMBER_OF_IMG_TO_LOAD, i

print(len(X_train), 'train images')
print(len(X_val), 'test images')

print('Training model ...')

# Release some memory
gc.collect()

# Building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

model.summary()
print()
print(len(X_train), 'train samples')
print(len(X_val), 'validation samples')
print()

# Training
N_OF_EPOCHS = 18
history = model.fit(x=X_train, y=y_train, epochs=N_OF_EPOCHS, batch_size=64, validation_data=(X_val, y_val,), verbose=1)
del X_train, X_val, y_train, y_val

# Visualizing errors and accuracy
sns.set()
x = range(N_OF_EPOCHS)

plt.subplot(121)
plt.title('Accuracy')
plt.plot(x, history.history['acc'], color='purple', label='Training accuracy')
plt.plot(x, history.history['val_acc'], color='red', label='Validation accuracy')
plt.legend()

plt.subplot(122)
plt.title('Loss')
plt.plot(x, history.history['loss'], color='purple', label='Training loss')
plt.plot(x, history.history['val_loss'], color='red', label='Validation loss')
plt.legend()

plt.show()

print('Saving model ...')
model.save('trained/cnn.h5')
