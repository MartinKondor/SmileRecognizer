"""
Example usage:

$ python issmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Arnold_Schwarzenegger\Arnold_Schwarzenegger_0006.jpg
$ python issmile.py %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Yoko_Ono\Yoko_Ono_0003.jpg
"""
import argparse

import numpy as np
from keras.models import load_model

from utils import vectorizeimg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='path to the image you want to classify')
    parser.add_argument('--show', help='if present, show the image', action='store_true')
    parser.add_argument('--train', help='if present, every other parameter is ignored and training starts', action='store_true')
    args = parser.parse_args()

    if args.train:
        import trainer
        print('Training finished.')
        exit(0)

    try:
        img = vectorizeimg(args.image_path)
    except FileNotFoundError:
        print('The given image "%s" is not found' % args.image_path)		
        exit(1)

    if args.show:
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    try:
        model = load_model('./trained/cnn.h5')
    except FileNotFoundError:
        print('Before usage you must download the trained models from github or train the model yourself')
        exit(0)

    input_data = np.array([img])
    prediction = model.predict(input_data)
    prediction = np.argmax(prediction, axis=1)
    print('This is NOT a smile.' if prediction == 0 else 'This is a smile.')
