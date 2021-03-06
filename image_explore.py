from PIL import Image
import glob
import matplotlib.pyplot as plt

train_images = glob.glob('data/train/*.jpg')
image = Image.open(train_images[5])

i, (im1) = plt.subplots(1)
i.set_figwidth(15)
im1.imshow(image)
plt.show()


total_portrait = 0
total_landscape = 0
sum_portrait_height_width = 0
sum_landscape_height_width = 0
sum_height_portrait = 0
sum_width_portrait = 0
portraits = []

for img_path in train_images:
    image = Image.open(img_path)
    if image.height > image.width:
        total_portrait += 1
        sum_portrait_height_width += image.height / image.width
        sum_height_portrait += image.height
        sum_width_portrait += image.width
        portraits += [img_path]
    else:
        total_landscape += 1
        sum_landscape_height_width += image.height / image.width
print(total_portrait, total_landscape)
print(sum_portrait_height_width / total_portrait, sum_landscape_height_width / total_landscape)
avg_portrait_height = sum_height_portrait / total_portrait
avg_portrait_width = sum_width_portrait / total_portrait
print(sum_height_portrait / total_portrait, sum_width_portrait / total_portrait)

for img_path in portraits:
    image = Image.open(img_path)
    image = image.resize((650, 920))
    image = image.resize((180, 256))
    image.save(img_path.replace('train', 'processed_train'))
