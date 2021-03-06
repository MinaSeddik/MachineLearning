import matplotlib.pyplot as plt

from mnist_reader import load_mnist_dataset


def test_load_mnist_dataset():
    # load MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # assert array sizes
    img_size = 28

    assert x_train.shape[0] == 60000
    assert x_train.shape[1] == img_size * img_size

    assert y_train.shape[0] == 60000

    assert x_test.shape[0] == 10000
    assert x_test.shape[1] == img_size * img_size

    assert y_test.shape[0] == 10000


def test_display_first_9():
    img_size = 28

    # load MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # get the first 9 images to display
    x_display = x_train[0:9].reshape((-1, img_size, img_size))
    y_display = y_train[0:9]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Sample MNIST Data Images', fontsize=20)

    for i in range(1, 10):
        ax = plt.subplot(3, 3, i)
        ax.xaxis.set_label_position('top')
        ax.legend().set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xlabel(y_display[i - 1], fontdict={'fontsize': 18, 'fontweight': 'bold'})
        plt.imshow(x_display[i - 1], cmap='gray')

    plt.show()

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     test_load_mnist_dataset()
