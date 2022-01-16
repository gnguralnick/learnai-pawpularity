import os

from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pathlib
import numpy as np

train_images = glob.glob('data/train/*.jpg')
# image = Image.open(train_images[5])
#
# i, (im1) = plt.subplots(1)
# i.set_figwidth(15)
# im1.imshow(image)
# plt.show()


total_portrait = 0
total_landscape = 0
sum_portrait_height_width = 0
sum_landscape_height_width = 0
sum_height_portrait = 0
sum_width_portrait = 0
portraits = []

# RESIZING
# print('image path: ', train_images[0].split('train_all')[1])

df = pd.read_csv('data/train.csv')
# print("--------df after dropping: \n", df)

# dropping all images that are collages or have no faces
df.drop(df[(df['Collage'] > 0) & (df['Face'] == 0)].index, inplace=True)

# print("--------df after dropping: \n", df)
print(df.columns)
# print(df.index[df['Id'] == '0007de18844b0dbbb5e1f607da0606e0'][0])
# print(df.at[df.index[df['Id'] == '0007de18844b0dbbb5e1f607da0606e0'][0], 'Pawpularity'])

processed_ids = df['Id'].tolist()

# array for pawpularity score
pawpularity = []



pawpularity_text = open('data/processed_train/pawpularity.txt', 'w')

for img_path in train_images:
    img_id = img_path.split("\\")[1][:-4]
    # only look into resizing if its not collage and has face
    if img_id in processed_ids:
        image = Image.open(img_path)

        if image.height > image.width:
            portraits += [img_path]
            # get pawpularity score

            pawpularity_text.write(str(df.at[df.index[df['Id'] == img_id][0], 'Pawpularity']) + '\n')
            # pawpularity += [df.at[df.index[df['Id'] == img_id][0], 'Pawpularity']]


pawpularity_text.close()

# # save processed images
# for img_path in portraits:
#     image = Image.open(img_path)
#     # NOTE: grayscale <- can do this here or just in preprocessing
#     # image = image.convert('L')
#
#     # resize image
#     # image = image.resize((650, 920))
#     image = image.resize((180, 256))
#
#
#     # image.save(img_path.replace('train', 'processed_train/all'))

# print('\n\nkeras image dataset from directory:\n')


# keras data preprocessing
# directory = 'data/train_bw' # directory with grayscaled images
# directory = 'data/processed_train'  # directory with coloured images
#
#
# print(directory)
#
# print('directory is valid??', os.path.isdir(directory))
# print(len(os.listdir('data/processed_train_elizabeth')))


# # train_ds, val_ds with coloured images
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='training')
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='validation')
#
#
# # print(np.array(list(train_ds.unbatch())).shape)
#
# train_labels = np.concatenate([y for x, y in train_ds], axis=0)
# train_images = np.concatenate([x for x, y in train_ds], axis=0).squeeze()
#
# val_labels = np.concatenate([y for x, y in val_ds], axis=0)
# val_images = np.concatenate([x for x, y in val_ds], axis=0).squeeze()
#
# # print('labels: ', train_labels, train_labels.shape)
# # print('images: ', train_images, train_images.shape)
#
#
#
# # viewing first 10 images
# # plt.figure(figsize=(10, 10))
# # for images, labels in train_ds.take(1):
# #     for i in range(9):
# #         ax = plt.subplot(3, 3, i + 1)
# #         plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
# #         plt.title(int(labels[i]))
# #         plt.axis("off")
# # plt.show()
#
#
# # standardize data
# # rgb values are in [0, 255]
# # NOTE: this works but tbh i have no clue what's really going on
# #       other option is to include the layer inside model definition
# # normalization_layer = tf.keras.layers.Rescaling(1./255)
# # # normalizing training ds
# # normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# # train_image_batch, train_labels_batch = next(iter(normalized_train_ds))
# # print('len train batch: ', len(train_image_batch)) # only 32?
# # first_image_train = train_image_batch[0]
# # print(np.min(first_image_train), np.max(first_image_train), np.mean(first_image_train))
# #
# # # normalizing validating ds
# # normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# # val_image_batch, val_labels_batch = next(iter(normalized_val_ds))
# # print('len val batch: ', len(val_image_batch)) # only 32?
# # first_image_val = val_image_batch[0]
# # print(np.min(first_image_val), np.max(first_image_val), np.mean(first_image_val))
#
#
#
# # umm, configure dataset for performance? (saw this in a tutorial)
# # AUTOTUNE = tf.data.AUTOTUNE
# #
# # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
# model.summary()
#
# model.compile(
#     optimizer='sgd',
#     loss='mse',
#     metrics=[tf.keras.metrics.RootMeanSquaredError()])
#
# history = model.fit(train_images, train_labels, epochs=10,
#                     validation_data=(test_images, test_labels))
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)
