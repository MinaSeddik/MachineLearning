import errno
import logging
import os
from random import random, seed
from shutil import copyfile

import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from matplotlib.image import imread
from numpy import asarray, save, load

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_all_images():
    images, labels = list(), list()

    # build the data dir where the images are exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats', 'images')

    logger.debug('the images dir: %s', images_dir)
    if not os.path.exists(images_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), images_dir)

    for file in os.listdir(images_dir):
        # logger.debug('loading file: %s', file)
        image_file = os.path.join(images_dir, file)
        image = load_img(image_file, target_size=(200, 200))

        # convert the image to numpy array
        # logger.debug('Convert the image to numpy array ...')
        image = img_to_array(image)

        label = 1 if file.startswith('cat') else 0

        logger.debug('Adding image name = %s, label = %d', file, label)
        images.append(image)
        labels.append(label)

    # convert to a numpy arrays
    images = asarray(images)
    labels = asarray(labels)

    logger.debug('Images shape = %s, Labels shape = %s', images.shape, labels.shape)

    # save the reshaped photos
    images_file = os.path.join(root_dir, 'data', 'dogs-vs-cats', 'dogs_vs_cats_images.npy')
    labels_file = os.path.join(root_dir, 'data', 'dogs-vs-cats', 'dogs_vs_cats_labels.npy')

    save(images_file, images)
    save(labels_file, labels)

    # load and confirm the shape
    images = load(images_file)
    labels = load(labels_file)

    return images, labels


def plot_cats_and_dogs():
    # build the data dir where the images are exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats', 'images')

    for i in range(1, 25):
        file_path = os.path.join(images_dir, 'cat.' + str(i - 1) + '.jpg')

        # load image pixels
        image = imread(file_path)

        ax = plt.subplot(6, 4, i)
        ax.xaxis.set_label_position('top')
        ax.legend().set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        plt.imshow(image)

    plt.show()

    for i in range(1, 25):
        file_path = os.path.join(images_dir, 'dog.' + str(i - 1) + '.jpg')

        # load image pixels
        image = imread(file_path)

        ax = plt.subplot(6, 4, i)
        ax.xaxis.set_label_position('top')
        ax.legend().set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        plt.imshow(image)

    plt.show()


def build_dir_structure_4_keras():
    # build the data dir where the images are exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats')

    images_dir = os.path.join(root_dir, 'images')

    logger.debug('the images dir: %s', images_dir)
    if not os.path.exists(images_dir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), images_dir)

    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')

    # create the sub directory structure as Keras requires
    sub_dirs = [train_dir, test_dir]
    for sub_dir in sub_dirs:
        # create label subdirectories
        label_dirs = ['dogs', 'cats']
        for label_dir in label_dirs:
            new_dir = os.path.join(sub_dir, label_dir)
            os.makedirs(new_dir, exist_ok=True)

    # seed random number generator, seed is to be constant so the data is reproducible
    seed(1)

    # define ratio of pictures to use for test
    test_ratio = 0.25

    # copy the files into the new structure
    for file in os.listdir(images_dir):
        logger.debug('loading file: %s', file)
        source_path = os.path.join(images_dir, file)

        dest_path = test_dir if random() < test_ratio else train_dir
        dest_path = os.path.join(dest_path, 'cats', file) if file.startswith('cat') else os.path.join(dest_path, 'dogs',
                                                                                                      file)

        logger.debug('source file: %s & dest file: %s', source_path, dest_path)
        copyfile(source_path, dest_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # load_all_images()
    # plot_cats_and_dogs()
    build_dir_structure_4_keras()
