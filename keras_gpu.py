from tensorflow.python.client import device_lib
from keras import backend as K


# https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu

if __name__ == '__main__':
    print(device_lib.list_local_devices())


    K.tensorflow_backend._get_available_gpus()