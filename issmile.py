"""
Example usage:

$ python issmile.py --show %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Arnold_Schwarzenegger\Arnold_Schwarzenegger_0006.jpg
$ python issmile.py %userprofile%\scikit_learn_data\lfw_home\lfw_funneled\Yoko_Ono\Yoko_Ono_0003.jpg
"""
import argparse

import numpy as np
from PIL import Image
from sklearn.externals import joblib


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
        img = Image.open(args.image_path)
    except FileNotFoundError:
        print('The given image "%s" is not found' % args.image_path)		
        exit(1)

    if args.show:
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    img = np.array(img.convert('L').resize((90, 80))).reshape(7200)

    encoder = joblib.load('trained/class_encoder.pkl')
    scaler = joblib.load('trained/data_normalizer.pkl')
    pca = joblib.load('trained/pca.pkl')
    model = joblib.load('trained/mlp_model.pkl')

    result = encoder.inverse_transform(model.predict(pca.transform(scaler.transform((img,)))))
    print('Smile' if result[0][0] else 'Not smile')
