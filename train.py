import tensorlayer as tl
import tensorflow as tf
from sklearn import svm,linear_model
from sklearn.externals import joblib
# from sklearn.model_selection import learning_cure
# from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from models import c3d_model
import  numpy as np
import os
import threading
import input_data
import random
import queue
import time

batch_size = 1
seq_len = 16
crop_size = 112
channel = 3
learning_rate = 0.001
decay_rate=0.1
weight_decay= 0.004
n_classes = 101
n_epoch = 40
decay_steps=20000
start_step = 0
print_seq = 20

models="./"
model_dir = "models"
train_path = "list/train.list"
test_path = "list/test.list"
iteration = len(open(train_path))/batch_size

# define the queue for Multithreading
data = queue.Queue(maxsize=20)
labels = queue.Queue(maxsize=20)

def placeholder_input(batch_size):
    # define the input placeholder
    images = tf.placeholder(tf.float32,shape=[batch_size,seq_len,crop_size,crop_size,channel],name="x")
    labels = tf.placeholder(tf.int32,shape=[batch_size,],name="y")
    return images,labels

def read_data(filepath,shuffle):
    # get the data and the label
    images, labels, _, _, _ = input_data.read_clip_and_label(
        filename=filepath,
        batch_size=batch_size,
        num_frames_per_clip=seq_len,
        crop_size=crop_size,
        shuffle=shuffle
    )
    return images , labels

def get_data():
    for epoch in range(n_epoch):
        for i in range(iteration):
            images_data, image_labels, _, _, _ = input_data.read_clip_and_label(
                filename=train_path,
                batch_size=batch_size,
                num_frames_per_clip=seq_len,
                crop_size=crop_size,
                shuffle=True)
            data.put(images_data)
            labels.put(image_labels)
            time.sleep(random.randrange(1))

def train_c3d():
    sess = tf.InteractiveSession()
    x , y = placeholder_input(batch_size)
    networks = c3d_model(x,n_classes)
    networks.print_params()
    # get the output
    y_ = networks.outputs
    y_op = tf.argmax(tf.nn.softmax(y_),1)
    # cross entropy cost
    cross_entropy = tl.cost.cross_entropy(y_,y,name="cross entropy")
    tf.summary.scalar("cross_entropy",cross_entropy)
    # accuracy
    correct_pred = tf.equal(tf.cast(y_op,tf.int32),y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy",accuracy)

    # l2 loss
    l2_cost = tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[0]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[2]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[4]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[5]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[7]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[8]) + \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[11])+ \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[13])+ \
              tf.contrib.layers.l2_regularizer(weight_decay)(networks.all_params[14])
    tf.summary.scalar("l2 loss", l2_cost)
    # total loss
    total_loss = l2_cost + cross_entropy
    tf.summary.scalar("total loss",cross_entropy)

    # load the pre-trained model
    if  os.path.exists(models):
        load_params = tl.files.load_npz(path='models/', name='model-1000.npz')
        tl.files.assign_params(sess, load_params, networks)
    # show the networks' information
    networks.print_layers()

    global_steps = tf.Variable(0,trainable=False)

    lr = tf.train.exponential_decay(learning_rate,
                                    global_steps,
                                    decay_steps,
                                    decay_rate,
                                    staircase=True)
    tf.summary.scalar("learning rate",lr)
    # get the network parameters'
    train_parms = networks.all_params
    # optimizer
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(total_loss,
                                                              var_list=train_parms,
                                                              global_step=global_steps)
    # init the parmeters
    sess.run(tf.global_variables_initializer())
    # mergerd the summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs/train_log",sess.graph,max_queue=5)
    test_writer = tf.summary.FileWriter("logs/test_log",sess.graph,max_queue=5)
    # train
    duration = 0
    for epoch in range(n_epoch):
        for i in range(iteration):
            # get the data from the queue
            x_data = data.get()
            y_label = labels.get()
            # open the dropout layer with training
            feed_dict = {x : x_data , y : y_label}
            feed_dict.update(networks.all_drop)
            # start time
            start_time = time.time()
            accu ,summary ,all = sess.run([accuracy,merged,train_op],feed_dict=feed_dict)
            duration += time.time() - start_time

            # Save a checkpoint and evaluate the model periodically.
            if (global_steps) % print_seq == 0 :
                tl.files.save_npz(networks.all_params,"ufc-101_model_%d.npz",sess)
                print('Training Data Eval:')
                train_images,train_labels = read_data(train_path,True)
                feed_dict = {x: train_images, y: train_labels}
                feed_dict.update(networks.all_drop)
                acc, summary ,op= sess.run([accuracy,merged,train_op],feed_dict=feed_dict)
                print("accuracy %.5f".format(acc))
                train_writer.add_summary(summary, global_steps)
                # test the model on validation dataset
                print('Validation Data Eval:')
                val_images,val_labels = read_data(test_path,True)
                # close the dropout layer
                dp_dict = tl.utils.dict_to_one(networks.all_drop)
                feed_dict = {x: val_images, y_: val_labels}
                feed_dict.update(dp_dict)
                acc, summary, op= sess.run([accuracy, merged,train_op],feed_dict=feed_dict)
                print("accuracy: " + "{:.5f}".format(acc))
                test_writer.add_summary(summary, global_steps)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_size=1):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def train_svm():
    '''
      when we train the svm,the softmax layer should be removed
      in the C3D model
    '''
    sess = tf.InteractiveSession()
    # get the c3d Temporal-spatial features
    x, y = placeholder_input(batch_size)
    networks = c3d_model(x, n_classes)
    y_ = networks.outputs

    #init the parameters 
    sess.run(tf.global_variables_initializer())
    # load the pre-trained c3d model
    saver = tf.train.Saver()
    saver.restore(sess,models)
    X = []
    Y = []
    for i in range(iteration):
        data,labels = read_data(train_path,False)
        fc_out = sess.run(y_,feed_dict={x:data,y:labels})
        X.append(fc_out)
        Y.append(labels)
    # train the linearSVM
    clf = linear_model.SGDClassifier()
    #split the dataset
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # plot_learning_curve(clf,"Learning Curves(SVM)",X,Y,ylim=[0.7,1.01],cv=cv,train_size=[0.2,0.4,0.6,0.8])
    joblib.dump(clf,"svm/model.pkl")


def main():
    t1 = threading.Thread(target=get_data,name="get_data")
    t2 = threading.Thread(target=train_c3d,name="train_c3d")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':
    main()

