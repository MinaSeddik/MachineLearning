import logging
import os

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model):
    # build the data dir where the images are exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats')

    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')

    # create data generators
    train_data_gen = ImageDataGenerator(rescale=1.0 / 255.0,
                                        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    # prepare iterators
    train_itr = train_data_gen.flow_from_directory(train_dir, class_mode='binary', batch_size=64,
                                                   target_size=(224, 224))
    test_itr = test_data_gen.flow_from_directory(test_dir, class_mode='binary', batch_size=64, target_size=(224, 224))

    # fit model
    history = model.fit_generator(train_itr, steps_per_epoch=len(train_itr),
                                  validation_data=test_itr, validation_steps=len(test_itr), epochs=20, verbose=2)

    # evaluate model
    test_loss, test_acc = model.evaluate_generator(test_itr, steps=len(test_itr), verbose=2)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))

    return test_acc


def predict_model(test_itr, model):
    # Predict on the test images.
    file_names = test_itr.filenames
    samples_count = len(file_names)
    predictions = model.predict_generator(test_itr, steps=samples_count)
    # HINT: the prediction is ratio of all classes per test row

    logger.info('predictions shape = %s', predictions.shape)

    # get the max class for each
    y_predict = np.argmax(predictions, axis=1)

    logger.info("Display the First 10 test Image's predictions:")
    logger.info('\tLabel\t\tPredicted Label:')
    for i in range(0, 10):
        logger.info('\t%s\t\t%d', test_itr[i], y_predict[i])


def define_one_block_VGG_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    learning_rate = 0.001
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def define_two_block_VGG_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    learning_rate = 0.001
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def define_three_block_VGG_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    learning_rate = 0.001
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def define_VGG16_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    # compile model
    learning_rate = 0.001
    opt = SGD(lr=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def report_model_specs_results(model, model_name, accuracy):
    logger.info('----------------  %s  ----------------', model_name)
    model.summary(print_fn=lambda line: logger.info(line))
    logger.info("Test accuracy: %.2f%%" % (accuracy * 100))


def predict_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    return img


if __name__ == '__main__':
    # create model
    model1 = define_one_block_VGG_model()
    model2 = define_two_block_VGG_model()
    model3 = define_three_block_VGG_model()
    vgg16_model = define_VGG16_model()

    accuracy1 = train_model(model1)
    accuracy2 = train_model(model2)
    accuracy3 = train_model(model3)
    accuracy4 = train_model(vgg16_model)

    logger.info('================  Summary  ================\n')
    report_model_specs_results(model1, 'One Block VGG Model', accuracy1)
    report_model_specs_results(model2, 'Two Block VGG Model', accuracy2)
    report_model_specs_results(model3, 'Three Block VGG Model', accuracy3)
    report_model_specs_results(vgg16_model, 'Keras VGG-16 Model', accuracy4)

    # # predict_model
    # # ==============
    #
    # # build the data dir where the images are exist
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # test_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats', 'test')
    #
    # # create test data generators
    # test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    #
    # # prepare iterators
    # test_itr = test_data_gen.flow_from_directory(test_dir, class_mode='binary', batch_size=64, target_size=(224, 224))
    #
    # predict_model(test_itr, vgg16_model)

    # # predict single image
    # # ====================
    #
    #
    # file_name = 'path of the image'
    # image = predict_image(file_name)
    # result = vgg16_model.predict(image)
