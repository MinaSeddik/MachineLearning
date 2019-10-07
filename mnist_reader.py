import gzip
import logging
import os
from urllib.parse import urlparse

import numpy as np

from download import download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_dataset(url):
    # MNIST data directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = os.path.join(root_dir, 'data', 'MNIST')

    url_path = urlparse(url)
    file_name = os.path.basename(url_path.path)
    mnist_file = os.path.join(mnist_dir, file_name)

    # download MNIST file if not exist locally
    if not os.path.exists(mnist_file):
        mnist_file = download(url, mnist_file)

    return mnist_file


def _load_images_dataset(url):
    # The images are 28 pixels in each dimension.
    img_size = 28

    mnist_file = _load_dataset(url)

    # un-gzip the file and read it as numpy array
    with gzip.open(mnist_file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # Reshape to 2-dim array with shape (num_images, img_size_flat).
    images_flat = data.reshape(-1, img_size * img_size)

    return images_flat


def _load_labels_dataset(url):
    mnist_file = _load_dataset(url)

    # un-gzip the file and read it as numpy array
    with gzip.open(mnist_file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data


def load_mnist_dataset():
    x_train = _load_images_dataset('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    y_train = _load_labels_dataset('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    x_test = _load_images_dataset('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    y_test = _load_labels_dataset('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

    return x_train, y_train, x_test, y_test
