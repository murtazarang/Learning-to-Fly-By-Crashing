
import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

# Path to the textfiles
#train_file = '/home/tagomago/tum_simulator_ws/src/alexnet_ws/train.txt'
#val_file = '/home/tagomago/tum_simulator_ws/src/alexnet_ws/val.txt'

#arc paths
train_file = '/home/nnaresh/mur/alexnet_ws/train.txt'
val_file = '/home/nnaresh/mur/alexnet_ws/val.txt'

# Learning params
learn_rate = 0.01
global_step = tf.Variable(0, trainable=False)
num_epochs = 60
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7', 'fc6']

prev_test_acc = 0

display_step = 20

filewriter_path = "/home/nnaresh/mur/alexnet_ws/finetune_alexnet_2/tensorboard"
checkpoint_path = "/home/nnaresh/mur/alexnet_ws/finetune_alexnet_2/checkpoints"

"""
Main Part of the finetuning Script.
"""

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)


x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

#Initialize AlexNet
model = AlexNet(x, keep_prob, num_classes, train_layers)

#Model Score
score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

with tf.name_scope("train"):

    l_r = tf.train.exponential_decay(learn_rate, global_step, 1, decay_rate=0.1, staircase=True)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)
    optimizer = tf.train.AdamOptimizer(learning_rate=l_r, epsilon=0.1).minimize(loss, var_list=var_list)

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter(filewriter_path)

saver = tf.train.Saver()

# Number of training/validation/epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)

    model.load_initial_weights(sess)
    #saver.restore(sess, "/home/tagomago/finetune_alexnet/checkpoints")

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            sess.run(optimizer, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            if step % display_step == 0:
                s, l = sess.run([merged_summary, loss], feed_dict={x: img_batch,
                                                    y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                print("Step: {}/{} \t Loss: {}".format(step, train_batches_per_epoch, l))
        #Validation
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1

        test_acc /= test_count

        if test_acc - prev_test_acc <= 0.001:
            global_step += 1
        prev_test_acc = test_acc

        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
