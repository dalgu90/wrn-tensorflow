#!/usr/bin/env python

import sys
import os
from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import cifar100 as data_input
import resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('data_dir', './cifar-100-binary', """Path to the CIFAR-100 binary data.""")
tf.app.flags.DEFINE_integer('num_classes', 100, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual block per group.
                                                Total number of conv layers will be 6n+4""")
tf.app.flags.DEFINE_integer('k', 2, """Network width multiplier""")

# Testing Configuration
tf.app.flags.DEFINE_string('ckpt_path', '', """Path to the checkpoint or dir.""")
tf.app.flags.DEFINE_bool('train_data', False, """Whether to test over training set.""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_string('output', '', """Path to the output txt.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# Other Configuration(not needed for testing, but required fields in
# build_model())
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tCIFAR-100 dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    print('\tNetwork width multiplier: %d' % FLAGS.k)

    print('[Testing Configuration]')
    print('\tCheckpoint path: %s' % FLAGS.ckpt_path)
    print('\tDataset: %s' % ('Training' if FLAGS.train_data else 'Test'))
    print('\tNumber of testing iterations: %d' % FLAGS.test_iter)
    print('\tOutput path: %s' % FLAGS.output)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        # The CIFAR-100 dataset
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=FLAGS.train_data, num_threads=1)

        # The class labels
        with open(os.path.join(FLAGS.data_dir, 'fine_label_names.txt')) as fd:
            classes = [temp.strip() for temp in fd.readlines()]

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, data_input.HEIGHT, data_input.WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # Build model
        decay_step = FLAGS.lr_step_epoch * FLAGS.num_train_instance / FLAGS.batch_size
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_classes=FLAGS.num_classes,
                            num_residual_units=FLAGS.num_residual_units,
                            k=FLAGS.k,
                            weight_decay=FLAGS.l2_weight,
                            initial_lr=FLAGS.initial_lr,
                            decay_step=decay_step,
                            lr_decay=FLAGS.lr_decay,
                            momentum=FLAGS.momentum)
        network = resnet.ResNet(hp, images, labels, None)
        network.build_model()
        # network.build_train_op()  # NO training op

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if os.path.isdir(FLAGS.ckpt_path):
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            # Restores from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
               print('\tRestore from %s' % ckpt.model_checkpoint_path)
               saver.restore(sess, ckpt.model_checkpoint_path)
            else:
               print('No checkpoint file found in the dir [%s]' % FLAGS.ckpt_path)
               sys.exit(1)
        elif os.path.isfile(FLAGS.ckpt_path):
            print('\tRestore from %s' % FLAGS.ckpt_path)
            saver.restore(sess, FLAGS.ckpt_path)
        else:
            print('No checkpoint file found in the path [%s]' % FLAGS.ckpt_path)
            sys.exit(1)

        # Start queue runners
        tf.train.start_queue_runners(sess=sess)

        # Testing!
        result_ll = [[0, 0] for _ in range(FLAGS.num_classes)] # Correct/wrong counts for each class
        test_loss = 0.0, 0.0
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            preds_val, loss_value, acc_value = sess.run([network.preds, network.loss, network.acc],
                        feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
            test_loss += loss_value
            for j in range(FLAGS.batch_size):
                correct = 0 if test_labels_val[j] == preds_val[j] else 1
                result_ll[test_labels_val[j] % FLAGS.num_classes][correct] += 1
        test_loss /= FLAGS.test_iter

        # Summary display & output
        acc_list = [float(r[0])/float(r[0]+r[1]) for r in result_ll]
        result_total = np.sum(np.array(result_ll), axis=0)
        acc_total = float(result_total[0])/np.sum(result_total)

        print('Class    \t\t\tT\tF\tAcc.')
        format_str = '%-31s %7d %7d %.5f'
        for i in range(FLAGS.num_classes):
            print(format_str % (classes[i], result_ll[i][0], result_ll[i][1], acc_list[i]))
        print(format_str % ('(Total)', result_total[0], result_total[1], acc_total))

        # Output to file(if specified)
        if FLAGS.output.strip():
            with open(FLAGS.output, 'w') as fd:
                fd.write('Class    \t\t\tT\tF\tAcc.\n')
                format_str = '%-31s %7d %7d %.5f'
                for i in range(FLAGS.num_classes):
                    t, f = result_ll[i]
                    format_str = '%-31s %7d %7d %.5f\n'
                    fd.write(format_str % (classes[i].replace(' ', '-'), t, f, acc_list[i]))
                fd.write(format_str % ('(Total)', result_total[0], result_total[1], acc_total))


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
