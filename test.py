import tensorlayer as tl
import tensorflow as tf
from sklearn import linear_model
from sklearn.externals import joblib
from models import c3d_model
import input_data
import time

model_name = "./sports1m_finetuning_ucf101.model"
svm_model = "svm/model.pkl"
batch_size = 10
length = 16
crop_size = 112
channel = 3
n_classes=101

def placeholder_input():
    # define the input placeholder
    images = tf.placeholder(tf.float32,shape=[batch_size,length,crop_size,crop_size,channel],name="x")
    labels = tf.placeholder(tf.int32,shape=[batch_size,],name="y")
    return images,labels

def c3d_softmax():
    files = open("list/test.list")
    videonum = len(list(files))
    print("The number of test video={}".format(videonum))
    x ,y = placeholder_input()

    sess = tf.InteractiveSession()
    # get the output of the network
    network = c3d_model(x,n_classes)
    y_ = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y_),1)
    correct_pred = tf.equal(tf.cast(y_op, tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # init the parameters
    sess.run(tf.global_variables_initializer())
    # load the model
    saver = tf.train.Saver()
    saver.restore(sess, model_name)
    next_start_pos = 0
    total_acc=0
    iteration = int(videonum / batch_size)
    for i in range(iteration):
        duration = 0
        start_time = time.time()
        test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                "list/test.list",
                batch_size,
                start_pos=next_start_pos
            )

        feed_dict = {x:test_images,y:test_labels}
        acc= sess.run(accuracy,feed_dict=feed_dict)
        duration = time.time() - start_time
        print("iteration %d has been finished in %d secends".format(i,duration))
        total_acc+=acc
    print("Done")
    average_acc = total_acc/iteration
    print("The test average accuracy is %.6f".format(average_acc))

def c3d_svm():
    files = open("list/test.list")
    videonum = len(list(files))
    print("The number of test video={}".format(videonum))
    x, y = placeholder_input()
    sess = tf.InteractiveSession()
    # get the output of the network
    network = c3d_model(x, n_classes)

    sess.run(tf.global_variables_initializer())
    iteration = int(videonum/batch_size)
    next_start_pos=0
    X_ = []
    Y_ = []
    for i in range(iteration):
        test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                "list/test.list",
                batch_size,
                start_pos=next_start_pos
            )
        feed_dict = {x:test_images,y:test_labels}
        y_ = sess.run(y_,feed_dict=feed_dict)
        X_.append(y_)
        Y_.append(test_labels)

    clf = joblib.load(svm_model)
    clf.score(X_,Y_)

def main():
    c3d_softmax()

if __name__ == '__main__':
    main()