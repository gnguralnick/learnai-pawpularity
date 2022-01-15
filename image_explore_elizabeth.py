import os

from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pathlib

train_images = glob.glob('data/train_all/*.jpg')
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
print('image path: ', train_images[0].split('train_all')[1])

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


for img_path in train_images:
    img_id = img_path.split("\\")[1][:-4]
    # only look into resizing if its not collage and has face
    if img_id in processed_ids:
        image = Image.open(img_path)

        if image.height > image.width:
            total_portrait += 1
            sum_portrait_height_width += image.height / image.width
            sum_height_portrait += image.height
            sum_width_portrait += image.width
            portraits += [img_path]
            # get pawpularity score
            pawpularity += [df.at[df.index[df['Id'] == img_id][0], 'Pawpularity']]
        else:
            total_landscape += 1
            sum_landscape_height_width += image.height / image.width

print(total_portrait, total_landscape)
print(sum_portrait_height_width / total_portrait, sum_landscape_height_width / total_landscape)
avg_portrait_height = sum_height_portrait / total_portrait
avg_portrait_width = sum_width_portrait / total_portrait
print(sum_height_portrait / total_portrait, sum_width_portrait / total_portrait)


# save processed images
for img_path in portraits:
    # grayscale
    image = Image.open(img_path).convert('L')
    # resize image
    image = image.resize((650, 920))
    image = image.resize((180, 256))
    # image.save(img_path.replace('train_all', 'train'))


print('\n\nkeras image dataset from directory:\n')


# keras data preprocessing
directory = 'data/train'
print(directory)
# directory = '/data/processed_train_elizabeth/'

print('directory is valid??', os.path.isdir(directory))
# print(len(os.listdir('data/processed_train_elizabeth')))


train_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='training')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels=pawpularity, color_mode='grayscale', image_size=(180, 256), seed=1000, validation_split=0.2, subset='validation')


