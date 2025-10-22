import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math
import os


dataset, info = tfds.load("oxford_iiit_pet:4.*.*", with_info=True)

def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (128, 128), method="nearest")
    input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
    return input_image, input_mask

def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_image_test(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

BATCH_SIZE=64
BUFFER_SIZE=1000

train_batches = (train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE))

validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input image", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def show_many(images, mask, num_cols=5):
    num_images = len(images)
    num_rows = math.ceil(num_images / num_cols)
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(tf.keras.utils.array_to_img(images[i]))
        plt.imshow(tf.keras.utils.array_to_img(mask[i]), alpha=0.4)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

test_sampels = list(test_dataset.as_numpy_iterator())
last_400_sampels = test_sampels[-400:]

test_images = [img for img, _ in last_400_sampels]
test_masks = [mask for _, mask in last_400_sampels]

for i in range(0, 100, 20):
    show_many(test_images[i:i+20], test_masks[i:i+20],  num_cols=5)

os.makedirs("output_images/originals", exist_ok=True)
os.makedirs("output_images/masks", exist_ok=True)
os.makedirs("output_images/overlays", exist_ok=True)


def save_image(array, path):
    plt.imsave(path, tf.keras.utils.array_to_img(array))


for i, (image, mask) in enumerate(zip(test_images, test_masks)):
    img = tf.keras.utils.array_to_img(image)
    mask_img = tf.keras.utils.array_to_img(mask)

    img.save(f"output_images/originals/image_{i+1:03d}.png")
    mask_img.save(f"output_images/masks/mask_{i+1:03d}.png")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask_img, alpha=0.4)
    ax.axis("off")
    fig.savefig(f"output_images/overlays/overlay_{i+1:03d}.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)