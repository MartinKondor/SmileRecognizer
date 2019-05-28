import numpy as np
from os import listdir
from PIL import Image


def decode_img_folder(folder: str, save_to: str):
    # decode images to matrices and save them
    # and return the biggest sized img
    biggest_img_size = [0, 0,]
    for filename in listdir(folder):
        img = Image.open(folder + '/' + filename)

        with open(save_to + '/' + filename.split('.')[0] + '.txt', 'w+') as file:
            # get the pixels from the img
            for x in range(img.size[1]):
                row = []
                for y in range(img.size[0]):
                    row.append(str(sum(img.getpixel((y, x))) // 3))
                file.write('\t'.join(row) + '\n')

        if img.size[0] > biggest_img_size[0]:
            biggest_img_size[0] = img.size[0]
        if img.size[1] > biggest_img_size[1]:
            biggest_img_size[1] = img.size[1]

    return biggest_img_size


def read_out_img(file: str):
    """
    Reading out img matrix from file.
    """
    data = []
    with open(file, 'r') as f:
        for line in f.read().splitlines():
            row = []
            for col in line.split('\t'):
                row.append(float(col))
            data.append(np.array(row))
    return np.array(data)
                

def load_all_img_data(folder: str):
    """
    Reading out all images data from a folder. 
    """
    data = []
    for file in listdir(folder):
        if file == 'names.txt' or file == 'target.txt':
            continue
        data.append(read_out_img(folder + '/' + file))
    return np.array(data)


def load_targets(folder):
    lines = []
    with open('target.txt') as f:
        for line in f.read().splitlines():
            lines.append(float(line.strip()))
    return lines


if __name__ == "__main__":
    print(decode_img_folder('raw_data', 'data'))

