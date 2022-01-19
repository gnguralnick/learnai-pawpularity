"""This file contains most of the image preprocessing we perform before constructing our model.
The images and text file saved by these functions are then used in our model construction code."""
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_DIR_PATH = 'data/train/'


def show_image_in_order(train_images: list, img_num: int):
    """Show a specific image from the images folder
    The number given corresponds to the order of the images as saved on the local disk"""
    image = Image.open(train_images[img_num])
    i, (im1) = plt.subplots(1)
    i.set_figwidth(15)
    im1.imshow(image)
    plt.show()


def load_and_drop_train_data(file_path: str):
    """Loads the training data from the given file and drops the Collage items
    and items with no Face"""
    df = pd.read_csv(file_path)
    # dropping all images that are collages or have no faces
    df.drop(df[(df['Collage'] > 0) & (df['Face'] == 0)].index, inplace=True)
    return df


def process_images_and_scores(df: pd.DataFrame, portraits_only: bool):
    """Resize, convert to grayscale, and save either all the portrait or landscape photos.
    For each photo, find its pawpularity in df and write that to a file"""
    pawpularity_text = open('data/processed_train/pawpularity.txt', 'w')
    processed_ids = df['Id'].tolist()
    for img_path in train_images:
        # print(img_path)
        img_id = img_path.split("/")[2][:-4]
        # print(img_id)
        # only look into resizing if its not collage and has face
        if img_id in processed_ids:
            image = Image.open(img_path)

            if (portraits_only and image.height >= image.width) or (not portraits_only and image.height <= image.width):
                image = image.convert('L')
                # TODO: decide on resizing val for landscape?
                image = image.resize((180, 256)) if portraits_only else image.resize((0, 0))
                image.save(img_path.replace('train', 'processed_train/all'))
                # save pawpularity score
                pawpularity_text.write(str(df.at[df.index[df['Id'] == img_id][0], 'Pawpularity']) + '\n')
    pawpularity_text.close()


if __name__ == '__main__':
    train_images = glob.glob(TRAIN_DIR_PATH + '*.jpg')
    show_image_in_order(train_images, 5)
    df = load_and_drop_train_data('data/train.csv')
    # print("--------df after dropping: \n", df)
    process_images_and_scores(df, True)
