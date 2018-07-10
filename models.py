import tensorflow as tf
import tensorlayer as tl

def c3d_model(input,n_class):

    with tf.name_scope("C3D"):

        # Input Layer
        Input = tl.layers.InputLayer(input,name="input_layer")

        # convluation layer 1
        Conv1a = tl.layers.Conv3dLayer(prev_layer=input,
                                       act=tf.nn.relu,
                                       shape=[3, 3, 3, 3, 64],
                                       strides=[1, 1, 1, 1, 1],
                                       padding='SAME',
                                       name='Conv1a')
        Pool1 = tl.layers.PoolLayer(prev_layer=Conv1a,
                                    ksize=[1, 1, 2, 2, 1],
                                    strides=[1, 1, 2, 2, 1],
                                    padding='SAME',
                                    pool=tf.nn.max_pool3d,
                                    name='Pool1')
        # Convluation Layer 2
        Conv2a = tl.layers.Conv3dLayer(prev_layer=Pool1,
                                       act=tf.nn.relu,
                                       shape=[3, 3, 3, 64, 128],
                                       strides=[1, 1, 1, 1, 1],
                                       padding='SAME',
                                       name='Conv2a')
        Pool2 = tl.layers.PoolLayer(prev_layer=Conv2a,
                                    ksize=[1, 2, 2, 2, 1],
                                    strides=[1, 2, 2, 2, 1],
                                    padding='SAME',
                                    pool=tf.nn.max_pool3d,
                                    name='Pool2')

        # Convluation Layer  3
        Conv3a = tl.layers.Conv3dLayer(prev_layer=Pool2,
                                       act=tf.nn.relu,
                                       shape=[3, 3, 3, 128, 256],
                                       strides=[1, 1, 1, 1, 1],
                                       padding='SAME',
                                       name='Conv3a')
        Conv3b = tl.layers.Conv3dLayer(prev_layer=Conv3a,
                                       act=tf.nn.relu,
                                       shape=[3, 3, 3, 256, 256],
                                       strides=[1, 1, 1, 1, 1],
                                       padding='SAME',
                                       name='Conv3b')

        Pool3 = tl.layers.PoolLayer(prev_layer=Conv3b,
                                    ksize=[1, 2, 2, 2, 1],
                                    strides=[1, 2, 2, 2, 1],
                                    padding='SAME',
                                    pool=tf.nn.max_pool3d,
                                    name='Pool3')

        # Convluation layer 4
        Conv4a = tl.layers.Conv3dLayer(prev_layer=Pool3,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,256,512],
                                       strides=[1,1,1,1,1],
                                       padding="SAME",
                                       name="Conv4a")
        Conv4b = tl.layers.Conv3dLayer(prev_layer=Conv4a,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,512,512],
                                       strides=[1,1,1,1,1],
                                       padding="SAME",
                                       name="Conv4b")
        Pool4 = tl.layers.PoolLayer(prev_layer=Conv4b,
                                    ksize=[1,2,2,2,1],
                                    strides=[1,1,1,1,1],
                                    padding="SAME",
                                    pool=tf.nn.max_pool3d,
                                    name="Pool4")

        # Convluation Layer 5
        Conv5a  = tl.layers.Conv3dLayer(prev_layer=Pool4,
                                        act=tf.nn.relu,
                                        shape=[3,3,3,512,512],
                                        strides=[1,1,1,1,1],
                                        padding="SAME",
                                        name="Conv5a")
        Conv5b = tl.layers.Conv3dLayer(prev_layer=Conv5a,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,512,512],
                                       strides=[1,1,1,1,1],
                                       padding="SAME",
                                       name="Conv5b")

        Pool5 = tl.layers.PoolLayer(prev_layer=Conv5b,
                                    ksize=[1,2,2,2,1],
                                    strides=[1,2,2,2,1],
                                    padding="SAME",
                                    pool=tf.nn.max_pool3d,
                                    name="Pool5")

        flatten = tl.layers.FlattenLayer(prev_layer=Pool5,name="flatten_layer")

        # fully connected layer.

        fc6 = tl.layers.DenseLayer(prev_layer=flatten,
                                   act=tf.nn.relu,
                                   n_units=n_class,
                                   name="fc6")

        drop1 = tl.layers.DropoutLayer(prev_layer=fc6,
                                       keep=0.5,
                                       name="dropout_layer1")

        fc7 = tl.layers.DenseLayer(prev_layer=drop1,
                                   act=tf.nn.relu,
                                   n_units=4096,
                                   name="fc7")

        drop2 = tl.layers.DropoutLayer(prev_layer=fc7,
                                       keep=0.5,
                                       name="dropout_layer2")
        fc8 = tl.layers.DenseLayer(prev_layer=drop2,
                                     act=tf.identity,
                                     n_units=n_class,
                                     name="fc8")

        return fc8


def c3d_clstm(inputs, num_classes):

    # """Builds the Conv3D ConvLSTM Networks."""
    with tf.name_scope("C3D_LSTM"):
        if inputs.get_shape().ndims!=5:
            raise Exception("The input dimension of 3DCNN must be rank 5")

        # Input Layer
        network_input = tl.layers.InputLayer(inputs, name='input_layer')

        # convluation layer 1
        Conv1a = tl.layers.Conv3dLayer(prev_layer=network_input,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,3,64],
                                       strides=[1,1,1,1,1],
                                       padding='SAME',
                                       name='Conv1a')
        Pool1 = tl.layers.PoolLayer(prev_layer=Conv1a,
                                    ksize=[1,1,2,2,1],
                                    strides=[1,1,2,2,1],
                                    padding='SAME',
                                    pool = tf.nn.max_pool3d,
                                    name='Pool1')

        # Convluation Layer 2
        Conv2a = tl.layers.Conv3dLayer(prev_layer=Pool1,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,64,128],
                                       strides=[1,1,1,1,1],
                                       padding='SAME',
                                       name='Conv2a')
        Pool2 = tl.layers.PoolLayer(prev_layer=Conv2a,
                                    ksize=[1,2,2,2,1],
                                    strides=[1,2,2,2,1],
                                    padding='SAME',
                                    pool = tf.nn.max_pool3d,
                                    name='Pool2')

        # Convluation Layer  3
        Conv3a = tl.layers.Conv3dLayer(prev_layer=Pool2,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,128,256],
                                       strides=[1,1,1,1,1],
                                       padding='SAME',
                                       name='Conv3a')
        Conv3b = tl.layers.Conv3dLayer(prev_layer=Conv3a,
                                       act=tf.nn.relu,
                                       shape=[3,3,3,256,256],
                                       strides=[1,1,1,1,1],
                                       padding='SAME',
                                       name='Conv3b')

        Pool3 = tl.layers.PoolLayer(prev_layer=Conv3b,
                                    ksize=[1,2,2,2,1],
                                    strides=[1,2,2,2,1],
                                    padding='SAME',
                                    pool=tf.nn.max_pool3d,
                                    name='Pool3')
        # Pool3 = tf.transpose(Pool3,perm=[0,1,4,2,3])

        # ConvLstm Layer
        shape3d = Pool3.outputs.get_shape().as_list()
        num_steps = shape3d[1]
        convlstm1=tl.layers.ConvLSTMLayer(prev_layer=Pool3,
                                          cell_shape=[14,14],
                                          filter_size=[3,3],
                                          feature_map=256,
                                          initializer=tf.random_uniform_initializer(-0.1,0.1),
                                          n_steps=num_steps,
                                          return_last=False,
                                          return_seq_2d=False,
                                          name='clstm_layer_1')

        convlstm2 = tl.layers.ConvLSTMLayer(prev_layer=convlstm1,
                                            cell_shape=[14,14],
                                            filter_size=[3,3],
                                            feature_map=384,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                            n_steps=num_steps,
                                            return_last=True,
                                            return_seq_2d=False,
                                            name='clstm_layer_2')
        # FC Layer 1
        convlstm2 = tl.layers.FlattenLayer(convlstm2,name='flatten')


        fc = tl.layers.DenseLayer(prev_layer=convlstm2,
                                   n_units=num_classes,
                                   act=tf.identity,
                                   name='fc2')
        drop = tl.layers.DropoutLayer(prev_layer=fc,
                                    keep=0.5,
                                    name="droplayer")

        return drop




