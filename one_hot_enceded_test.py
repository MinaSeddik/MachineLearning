from numpy import argmax, array
from keras.utils import to_categorical


if __name__ == '__main__':
    # define example
    # data = [1, 3, 2, 0, 3, 20, 2, 1, 0, 1]
    data = ['cat', 'dog', 'horse', 'cat', 'dog', 'cat']
    data = array(data)
    print(data)
    # one hot encode
    encoded = to_categorical(data)
    print(encoded)
    # invert encoding
    inverted = argmax(encoded[0])
    print(inverted)


