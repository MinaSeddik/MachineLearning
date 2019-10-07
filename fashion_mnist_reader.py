import errno
import gzip
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_dataset_file(file_name):
    # The images are 28 pixels in each dimension.
    img_size = 28

    # Fahsion MNIST data directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    fahsion_mnist_dir = os.path.join(root_dir, 'data', 'fashion-mnist')

    fahsion_mnist_file = os.path.join(fahsion_mnist_dir, file_name)

    if not os.path.exists(fahsion_mnist_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fahsion_mnist_file)

    return fahsion_mnist_file


def _load_images_dataset(file):
    # The images are 28 pixels in each dimension.
    img_size = 28

    mnist_file = _load_dataset_file(file)

    # un-gzip the file and read it as numpy array
    with gzip.open(mnist_file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # Reshape to 2-dim array with shape (num_images, img_size_flat).
    images_flat = data.reshape(-1, img_size * img_size)

    return images_flat


def _load_labels_dataset(file):
    mnist_file = _load_dataset_file(file)

    # un-gzip the file and read it as numpy array
    with gzip.open(mnist_file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data


def load_fashion_mnist_dataset():
    x_train = _load_images_dataset('train-images-idx3-ubyte.gz')
    y_train = _load_labels_dataset('train-labels-idx1-ubyte.gz')
    x_test = _load_images_dataset('t10k-images-idx3-ubyte.gz')
    y_test = _load_labels_dataset('t10k-labels-idx1-ubyte.gz')

    return x_train, y_train, x_test, y_test
