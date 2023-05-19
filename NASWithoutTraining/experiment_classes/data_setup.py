import numpy as np
import tensorflow.keras as keras # type: ignore


def per_pixel_normalization(images: np.ndarray):
  images = np.asarray(images, np.float32)
  per_pixel_std = np.std(images, axis=0)
  per_pixel_mean = np.mean(images, axis=0)
  images -= per_pixel_mean
  images /= (per_pixel_std + 1e-14)
  return images


def setup_dataset():
  (train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
  train_images = per_pixel_normalization(train_images)
  return train_images
