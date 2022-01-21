import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from keras.models import model_from_json

PROCESSED_TRAIN_DIR = 'data/processed_train'  # directory with preprocessed images


def keras_preprocessing(directory: str, labels: list):
    """Perform keras preprocessing on the image data to split into training and validation sets"""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=labels, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='training')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=labels, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='validation')
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    train_images = np.concatenate([x for x, y in train_ds], axis=0).squeeze()

    val_labels = np.concatenate([y for x, y in val_ds], axis=0)
    val_images = np.concatenate([x for x, y in val_ds], axis=0).squeeze()
    assert not any(x for x in np.isnan(train_labels).tolist())
    assert not any(x for x in np.isnan(train_images).tolist())
    assert not any(x for x in np.isnan(val_labels).tolist())
    assert not any(x for x in np.isnan(val_images).tolist())
    return train_labels, train_images, val_labels, val_images


def build_and_compile_model() -> models.Sequential:
    """Builds and compiles the model and returns the result"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.build()
    compile_model(model)
    return model


def compile_model(model) -> None:
    """Compiles the given model using the given optimizer, loss, and metric"""
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError()])


def plot_history(history) -> None:
    """Plots the history for the model to view model accuracy by epoch"""
    plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
    plt.plot(history.history['val_root_mean_squared_error'], label='val_root_mean_squared_error')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


# saving model
def save_model(model: models.Model):
    """Saves the given model to a json file and its weights to model.h5"""
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('model.h5')


# loading model
def load_model(model_path: str, weights_path: str) -> models.Model:
    """Loads a model from a Json file at the given path with weights at the other given path"""
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    print('# loaded model from disk')
    return loaded_model


if __name__ == '__main__':
    pawpularity = list(loadtxt('data/processed_train/pawpularity.txt', delimiter='\n', unpack=False))
    train_labels, train_images, val_labels, val_images = keras_preprocessing(PROCESSED_TRAIN_DIR, pawpularity)
    model = build_and_compile_model()
    history = model.fit(train_images, train_labels, epochs=1,
                        validation_data=(val_images, val_labels))
    plot_history(history)
    test_loss, test_acc = model.evaluate(val_images, val_labels, verbose=2)
    save_model(model)

    model = load_model('model.json', 'model.h5')
    compile_model(model)
    score = model.evaluate(val_images, val_labels, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
