#!/usr/bin/env python

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
tf.app.flags.DEFINE_integer('num_train_instance', 50000, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 10000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_residual_units', 2, """Number of residual block per group.
                                                Total number of conv layers will be 6n+4""")
tf.app.flags.DEFINE_integer('k', 2, """Network width multiplier""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_step_epoch', 100.0, """Epochs after which learing rate decays""")
tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
# tf.app.flags.DEFINE_boolean('basenet_train', True, """Flag whether the model will train the base network""")
# tf.app.flags.DEFINE_float('basenet_lr_ratio', 0.1, """Learning rate ratio of basenet to bypass net""")
# tf.app.flags.DEFINE_boolean('finetune', False,
                            # """Flag whether the L1 connection weights will be only made at
                            # the position where the original bypass network has nonzero
                            # L1 connection weights""")
# tf.app.flags.DEFINE_string('pretrained_dir', './pretrain', """Directory where to load pretrained model.(Only for --finetune True""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('test_interval', 1000, """Number of iterations to run a test""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 10000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tCIFAR-100 dir: %s' % FLAGS.data_dir)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tResidual blocks per group: %d' % FLAGS.num_residual_units)
    print('\tNetwork width multiplier: %d' % FLAGS.k)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tEpochs per lr step: %f' % FLAGS.lr_step_epoch)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per testing: %d' % FLAGS.test_interval)
    print('\tSteps during testing: %d' % FLAGS.test_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of CIFAR-100
        with tf.variable_scope('train_image'):
            train_images, train_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=True)
        with tf.variable_scope('test_image'):
            test_images, test_labels = data_input.input_fn(FLAGS.data_dir, FLAGS.batch_size, train_mode=False)

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
        network = resnet.ResNet(hp, images, labels, global_step)
        network.build_model()
        network.build_train_op()

        # Summaries(training)
        train_summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
           print('\tRestore from %s' % ckpt.model_checkpoint_path)
           # Restores from checkpoint
           saver.restore(sess, ckpt.model_checkpoint_path)
           init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
           print('No checkpoint file found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)
        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Training!
        test_best_acc = 0.0
        for step in range(init_step, FLAGS.max_steps):
            # Test
            if step % FLAGS.test_interval == 0:
                test_loss, test_acc = 0.0, 0.0
                for i in range(FLAGS.test_iter):
                    test_images_val, test_labels_val = sess.run([test_images, test_labels])
                    loss_value, acc_value = sess.run([network.loss, network.acc],
                                feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
                    test_loss += loss_value
                    test_acc += acc_value
                test_loss /= FLAGS.test_iter
                test_acc /= FLAGS.test_iter
                test_best_acc = max(test_best_acc, test_acc)
                format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
                print(format_str % (datetime.now(), step, test_loss, test_acc))

                test_summary = tf.Summary()
                test_summary.value.add(tag='test/loss', simple_value=test_loss)
                test_summary.value.add(tag='test/acc', simple_value=test_acc)
                test_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
                summary_writer.add_summary(test_summary, step)
                # test_loss_summary = tf.Summary()
                # test_loss_summary.value.add(tag='test/loss', simple_value=test_loss)
                # summary_writer.add_summary(test_loss_summary, step)
                # test_acc_summary = tf.Summary()
                # test_acc_summary.value.add(tag='test/acc', simple_value=test_acc)
                # summary_writer.add_summary(test_acc_summary, step)
                # test_best_acc_summary = tf.Summary()
                # test_best_acc_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
                # summary_writer.add_summary(test_best_acc_summary, step)
                summary_writer.flush()

            # Train
            start_time = time.time()
            train_images_val, train_labels_val = sess.run([train_images, train_labels])
            _, lr_value, loss_value, acc_value, train_summary_str = \
                    sess.run([network.train_op, network.lr, network.loss, network.acc, train_summary_op],
                        feed_dict={network.is_train:True, images:train_images_val, labels:train_labels_val})
            duration = time.time() - start_time

            assert not np.isnan(loss_value)

            # Display & Summary(training)
            if step % FLAGS.display == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, acc_value, lr_value,
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
