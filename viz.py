"""
Visualize the learned features/windows of the convolutional layers.
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img
from sklearn.datasets import get_data_home

from utils import IMG_SHAPE


sns.set()

# Loading the trained model
try:
    model = load_model('./trained/cnn.h5')
except FileNotFoundError:
    print('Before usage you must download the trained models from github or train the model yourself')
    exit(0)

model.summary()
print()

# Loading and showing test image
img = np.array(load_img(get_data_home() + '/lfw_home/lfw_funneled/Donald_Trump/Donald_Trump_0001.jpg', target_size=IMG_SHAPE)) / 255
img = img.reshape((1, img.shape[1], img.shape[0], 3))
print('Image shape:', img.shape)
print()

layer_outputs = [layer.output for layer in model.layers if type(layer) is Conv2D or type(layer) is MaxPooling2D]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img)

print('First layer of activations shape:', activations[0].shape)

n_maps = 5
plt.figure(figsize=(10, 5))

for i in range(5):
    plt.subplot(151 + i)
    plt.imshow(activations[0][0, :, :, i], cmap='viridis')
    plt.xticks(())
    plt.yticks(())
    
plt.show()
