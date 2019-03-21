import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, show 
from argparse import ArgumentParser
from sklearn.externals.joblib import load as load_model


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('image_path', help='path to the image you want to classify')
	parser.add_argument('--show', help='if present, show the image', action='store_true')
	args = parser.parse_args()
	
	try:
		img = Image.open(args.image_path)
	except FileNotFoundError:
		print('The given image "%s" is not found' % args.image_path)		
		exit()
	
	if args.show:
		imshow(img)	
		show()
	
	img = img.convert('L')  # convert image to black and white
	img = img.resize((300, 168), Image.ANTIALIAS)  # resize to the acceptable size
	data = np.array(img.getdata())  # extract data from img

	model = load_model('trained_models/best_model.pkl') 
	print( 'It\'s a real smile.' if model.predict([data])[0] == 1 else 'It\'s a fake smile' )

