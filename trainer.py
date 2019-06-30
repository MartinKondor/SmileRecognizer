import gc
import os
from PIL import Image

import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from matplotlib import pyplot as plt

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
for i in range(len(NON_SMILE_IMAGES)):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(NON_SMILE_IMAGES, i))
        y.append(0)
    except FileNotFoundError:
        continue


# load smiling images
for i in range(len(SMILE_IMAGES)):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(SMILE_IMAGES, i))
        y.append(1)
    except FileNotFoundError:
        continue


# convert them to numpy.ndarray
X = np.array(X)
y = np.array(y)  # 1 = Smile, 0 = Non-smile
y = to_categorical(y)

# preprocessing
print('Preprocessing ...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1, shuffle=True)
del X, y, NON_SMILE_IMAGES, SMILE_IMAGES, DATA_DIR, NUMBER_OF_IMG_TO_LOAD, i

X_train = X_train / 255
X_test = X_test / 255

print(len(X_train), 'train images')
print(len(X_test), 'test images')

print('Training model ...')
# Release some memory
gc.collect()

# Building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

# Training
N_OF_EPOCHS = 10
history = model.fit(x=X_train, y=y_train, epochs=N_OF_EPOCHS, batch_size=64, validation_data=(X_test, y_test,))

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
"""
The model's summary:
```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 78, 88, 32)        896
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 76, 86, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 38, 43, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 38, 43, 32)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 52288)             0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               6692992
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 258
=================================================================
Total params: 6,703,394
Trainable params: 6,703,394
Non-trainable params: 0
_________________________________________________________________
```

Accuracy on train set: **99.9**
Accuracy on test set: **88.99**
"""
