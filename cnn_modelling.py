import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

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


# standardize data
# rgb values are in [0, 255]
# NOTE: this works but tbh i have no clue what's really going on
#       other option is to include the layer inside model definition
# normalization_layer = tf.keras.layers.Rescaling(1./255)
# # normalizing training ds
# normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# train_image_batch, train_labels_batch = next(iter(normalized_train_ds))
# print('len train batch: ', len(train_image_batch)) # only 32?
# first_image_train = train_image_batch[0]
# print(np.min(first_image_train), np.max(first_image_train), np.mean(first_image_train))
#
# # normalizing validating ds
# normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# val_image_batch, val_labels_batch = next(iter(normalized_val_ds))
# print('len val batch: ', len(val_image_batch)) # only 32?
# first_image_val = val_image_batch[0]
# print(np.min(first_image_val), np.max(first_image_val), np.mean(first_image_val))



# umm, configure dataset for performance? (saw this in a tutorial)
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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

print(test_acc)
