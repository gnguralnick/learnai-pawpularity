import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
import os
from keras.models import model_from_json

pawpularity = list(loadtxt('data/processed_train/pawpularity.txt', delimiter='\n', unpack=False))

directory = 'data/processed_train'  # directory with coloured images

# train_ds, val_ds with coloured images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='training')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='validation')


# print(np.array(list(train_ds.unbatch())).shape)

train_labels = np.concatenate([y for x, y in train_ds], axis=0)
train_images = np.concatenate([x for x, y in train_ds], axis=0).squeeze()

val_labels = np.concatenate([y for x, y in val_ds], axis=0)
val_images = np.concatenate([x for x, y in val_ds], axis=0).squeeze()

# print('labels: ', train_labels, train_labels.shape)
# print('images: ', train_images, train_images.shape)

# print('train labels: ', any(x for x in np.isnan(train_labels).tolist()))
# print('train images', any(x for x in np.isnan(train_images).tolist()))
#
# print('val labels: ', any(x for x in np.isnan(val_labels).tolist()))
# print('val images', any(x for x in np.isnan(val_images).tolist()))


# viewing first 10 images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
#         plt.title(int(labels[i]))
#         plt.axis("off")
# plt.show()

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


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(train_images, train_labels, epochs=1,
                    validation_data=(val_images, val_labels))


plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
plt.plot(history.history['val_root_mean_squared_error'], label='val_root_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(val_images, val_labels, verbose=2)

# saving model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('model.h5')
print("# SAVED")

print(test_acc)


# loading model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('# loaded model from disk')

loaded_model.compile(optimizer='adam',
                     loss='mse',
                     metrics=[tf.keras.metrics.RootMeanSquaredError()])
score = loaded_model.evaluate(val_images, val_labels, verbose=2)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
