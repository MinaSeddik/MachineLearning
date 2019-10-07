import logging
import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    model_file_name = 'dogs_vs_cats_keras_cnn_final.h5'

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(root_dir, 'saved_models', 'dogs-vs-cats', model_file_name)

    model = load_model(model_file_path)

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

    # checkpoint
    check_point_file_name = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
    check_point_file_path = os.path.join(root_dir, 'saved_models', 'dogs-vs-cats', check_point_file_name)
    checkpoint = ModelCheckpoint(check_point_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # fit model
    history = model.fit_generator(train_itr, steps_per_epoch=len(train_itr),
                                  validation_data=test_itr, validation_steps=len(test_itr), epochs=33,
                                  callbacks=callbacks_list, verbose=2)

    # evaluate model
    test_loss, test_acc = model.evaluate_generator(test_itr, steps=len(test_itr), verbose=2)
    logger.info("Test accuracy: %.2f%%" % (test_acc * 100))

    logger.info('Save model to the disk [%s]', model_file_path)
    model.save(model_file_path)

    logger.info(history.history.keys())
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
