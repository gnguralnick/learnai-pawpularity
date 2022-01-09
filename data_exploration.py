"""Exploring the data to see if there's any parts of it we should filter out"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
from skimage import io
import numpy

train_data_path = "data/train.csv"
train_img_path = "data/train"

df = pd.read_csv('data/train.csv')

print(df)
print(df.info())

print(df.cumsum().drop('Id', axis=1).drop('Pawpularity', axis=1))

def plot_attribute(attribute: str) -> None:
    sorted_df = df.groupby('Pawpularity')[attribute].apply(list)
    yvalues = []
    xvalues = range(0, 100)
    for lst in sorted_df:
        yvalues.append(sum(lst) / len(lst))
    plt.scatter(xvalues, yvalues)
    plt.xlabel("Pawpularity")
    plt.ylabel(attribute)


plot_attribute("Human")

train_images = glob.glob('data/train/*.jpg')
image = io.imread(train_images[0])

i, (im1) = plt.subplots(1)
i.set_figwidth(15)
im1.imshow(image)
plt.show()
