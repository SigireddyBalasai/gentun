#!/usr/bin/env python
"""
Test the GeneticCnnModel using the MNIST dataset
"""

import os
import sys
# oxfors iit pet dataset from tfds
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random
import cv2



# Add parent directory to path
train = tfds.load('oxford_iiit_pet:3.2.0', split='train[:80%]', shuffle_files=True,as_supervised=True)
test = tfds.load('oxford_iiit_pet:3.2.0', split='test[:20%]', shuffle_files=True,as_supervised=True)
train = train.prefetch(tf.data.experimental.AUTOTUNE)
test = test.prefetch(tf.data.experimental.AUTOTUNE)
train_images = []
train_labels = []
test_images = []
test_labels = []
for image, label in train:
    train_images.append(cv2.resize(image.numpy(),(28,28)))
    train_labels.append(label.numpy())
for image, label in test:
    test_images.append(cv2.resize(image.numpy(),(28,28)))
    test_labels.append(label.numpy())
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GeneticCnnModel

    # load oxford_iiit_pet dataset
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(37))

    selection = random.sample(range(n), 500)
    y_train = lb.transform(train_labels[selection])
    x_train = train_images.reshape(n, 28, 28, 3)[selection]
    x_train = x_train / 255  # Normalize train data

    model = GeneticCnnModel(
        x_train, y_train,
        {'S_1': '0000000100'},  # Genes to test
        (5,),  # Number of nodes per DAG (corresponds to gene bytes)
        (28, 28, 3),  # Shape of input data
        (50,),  # Number of kernels per layer
        ((3,3) , (5, 5), (5, 5)),  # Sizes of kernels per layer
        500,  # Number of units in Dense layer
        0.7,  # Dropout probability
        37,  # Number of classes to predict
        kfold=5,
        epochs=(25, 20, 1),
        learning_rate=(1e-3, 1e-4, 1e-5),
        batch_size=128
    )
    print(model.cross_validate())
