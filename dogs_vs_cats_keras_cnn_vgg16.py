import logging
import os

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    model_file_name = 'dogs_vs_cats_keras_cnn_final.h5'

    # create model
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

    # build the data dir where the images are exist
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data', 'dogs-vs-cats')

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

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
                                  validation_data=test_itr, validation_steps=len(test_itr), epochs=33, verbose=2)

    # evaluate model
    test_loss, test_acc = model.evaluate_generator(test_itr, steps=len(test_itr), verbose=2)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))

    # Save the model to disk.
    model_file_path = os.path.join(root_dir, 'saved_models', 'dogs-vs-cats', model_file_name)

    logger.info('Save model to the disk [%s]', model_file_path)
    model.save(model_file_path)

    # Returns a compiled model identical to the previous one
    # from keras.models import load_model
    # model = load_model(model_file_path)
