import tensorflow as tf
import numpy as np
import os

NUM_CLASSES = 10
INIT_LRN_RATE = 1e-2
MIN_LRN_RATE = 1e-4
WEIGHT_DECAY_RATE = 1e-4
RELU_LEAKINESS = 0.1
NUM_TRAIN_IMAGES = 50000

HEIGHT = 32
WIDTH = 32
DEPTH = 3

NEW_HEIGHT = 32
NEW_WIDTH = 32


def get_filename(data_dir, train_mode):
    """Returns a list of filenames based on 'mode'."""
    if train_mode:
        return os.path.join(data_dir, 'train.bin')
    else:
        return os.path.join(data_dir, 'test.bin')

def dataset_parser(value):
    label_bytes = 1
    image_bytes = HEIGHT * WIDTH * DEPTH
    record_bytes = label_bytes + image_bytes

    raw_record = tf.decode_raw(value, tf.uint8)
    label = tf.cast(raw_record[0], tf.int32)

    depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
                           [DEPTH, HEIGHT, WIDTH])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
    return image, label
    # return image, tf.one_hot(label, NUM_CLASSES)

cifar100_mean = [129.304, 124.070, 112.434]
cifar100_std = [68.170, 65.392, 70.418]

def train_preprocess_fn(image, label):
    image = tf.image.resize_image_with_crop_or_pad(image, NEW_HEIGHT+4, NEW_WIDTH+4)
    image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, 3])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - cifar100_mean) / cifar100_std
    return image, label

def test_preprocess_fn(image, label):
    # image = tf.image.resize_images(image, [NEW_HEIGHT+4, NEW_WIDTH+4])
    # image = tf.random_crop(image, [NEW_HEIGHT, NEW_WIDTH, 3])
    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - cifar100_mean) / cifar100_std
    return image, label

def read_bin_file(bin_fpath):
    """ Read CIFAR-10 .bin file returns images and labels """
    with open(bin_fpath, 'rb') as fd:
        bstr = fd.read()

    coarse_label_byte = 1
    label_byte = 1
    image_byte = HEIGHT * WIDTH * DEPTH

    array = np.frombuffer(bstr, dtype=np.uint8).reshape((-1, coarse_label_byte + label_byte + image_byte))
    coarse_labels = array[:,:(coarse_label_byte)].flatten().astype(np.int32)
    labels = array[:,coarse_label_byte:(coarse_label_byte+label_byte)].flatten().astype(np.int32)
    images = array[:,(coarse_label_byte+label_byte):].reshape((-1, DEPTH, HEIGHT, WIDTH)).transpose((0, 2, 3, 1))

    return images, labels

def input_fn(data_dir, batch_size, train_mode, num_threads=8):
    # Read CIFAR-100 dataset
    images_arr, labels_arr = read_bin_file(get_filename(data_dir, train_mode))
    dataset = tf.data.Dataset.from_tensor_slices((images_arr, labels_arr))

    if train_mode:
        buffer_size = int(50000 * 0.4) + 3 * batch_size
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(train_preprocess_fn, batch_size, num_threads))
    else:
        dataset = dataset.repeat()
        dataset = dataset.apply(tf.contrib.data.map_and_batch(test_preprocess_fn, batch_size, num_threads))

    # check TF version >= 1.8
    ver = tf.__version__
    if float(ver[:ver.rfind('.')]) >= 1.8:
        dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/GPU:0'))
    else:
        dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    images.set_shape((batch_size, NEW_WIDTH, NEW_HEIGHT, DEPTH))
    labels.set_shape((batch_size, ))

    return images, labels
