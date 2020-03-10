#Bottle Level Indicator Kaggle Challenge

import tensorflow as tf
import numpy as np
import time
import csv
import glob
import numpy as np
import scipy
import warnings
import tensorflow as tf
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, stride=1, index=0):
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # stride in tf.nn.conv2d is in the format: [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, stride, stride, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x, is_training):
        """
        :param input_x: The input that needed for normalization.
        :param is_training: To control the training or inference phase
        """
        with tf.variable_scope('batch_norm'):
            batch_mean, batch_variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def True_fn():
                ema_op = ema.apply([batch_mean, batch_variance])
                with tf.control_dependencies([ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            def False_fn():
                return ema.average(batch_mean), ema.average(batch_variance)

            mean, variance = tf.cond(is_training, True_fn, False_fn)

            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0):
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

#Kaggle Net
def kaggle_Net(input_x, input_y, is_training,
          img_len=28, channel_num=1, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    #raise NotImplementedError
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    # flatten
    pool_shape = pooling_layer_0.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_0.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)
    # output
    out = fc_layer_1.output()
    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return out, loss

def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output))

    return ce

def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('kaggle_net_error_num', error_num)
    return error_num

def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step
    

def kaggle_training(X_train, y_train, X_val, y_val,
                conv_featmap=(6, 16),
                fc_units=(120, 84),
                conv_kernel_size=(5, 5),
                pooling_size=(2, 2),
                l2_norm=0.01,
                seed=235,
                learning_rate=1e-3,
                epoch=20,
                batch_size=295,
                verbose=False,
                pre_trained_model=None):

    print("Building kaggle Net. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    img_len = X_train.shape[1]
    channel_num = X_train.shape[-1]
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, img_len, img_len, channel_num], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        is_training = tf.placeholder(tf.bool, name='is_training')

    output, loss = kaggle_Net(xs, ys, is_training,
                         img_len=img_len,
                         channel_num=channel_num,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss, learning_rate)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'KaggleModel'
    #cur_model_name = 'best_LeNet_model'

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                ys: training_batch_y,
                                                                is_training: True})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val,
                                                                                ys: y_val,
                                                                                is_training: False})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)
                    print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))

def predict(output):
  with tf.name_scope('predict'):
      pred = tf.argmax(output, axis=1)
  return pred



def kaggle_testing(X_test,pre_trained_model,batch_size):
# define the variables and parameter needed during testing
  N = X_test.shape[0]
  with tf.name_scope('inputs'):
      xs = tf.placeholder(shape=[None,128, 128, 3], dtype=tf.float32)
      ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
      is_training = tf.placeholder(tf.bool, name='is_training')

  output, _ = kaggle_Net(xs, ys, is_training,
                       img_len=128,
                       channel_num=3,
                       output_size=5,
                       conv_featmap=[64],
                       fc_units=[100],
                       conv_kernel_size=[32],
                       pooling_size=[16],
                       l2_norm=0.01,
                       seed=235)

  pred = predict(output)
  y_test = []
  saver = tf.train.Saver()
  iters = int(X_test.shape[0] / batch_size)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'model/{}'.format(pre_trained_model))
    
    for itr in range(iters):
      test_batch_x = X_test[itr * batch_size: (1 + itr) * batch_size]
      temp = sess.run([pred], feed_dict={xs: test_batch_x,
                                         ys: np.ones(test_batch_x.shape[0],),
                                         is_training: False})
      for j in range(len(temp[0])):
        y_test.append(temp[0][j])

  y_test = np.array(y_test)
  return y_test

#Creating X_train and y_train
image_list = []
for filename in glob.glob('./train_128/0/*.png'):
    im = scipy.ndimage.imread(filename)
    image_list.append(im)
for filename in glob.glob('./train_128/1/*.png'):
    im = scipy.ndimage.imread(filename)
    image_list.append(im)
for filename in glob.glob('./train_128/2/*.png'):
    im = scipy.ndimage.imread(filename)
    image_list.append(im)
for filename in glob.glob('./train_128/3/*.png'):
    im = scipy.ndimage.imread(filename)
    image_list.append(im)
for filename in glob.glob('./train_128/4/*.png'):
    im = scipy.ndimage.imread(filename)
    image_list.append(im)

X_train_raw = np.asarray(image_list)
y_train_raw = np.concatenate((np.full(3000, 0),np.full(3000, 1),np.full(3000, 2),\
                         np.full(3000, 3),np.full(3000, 4)))
print(X_train_raw.shape, y_train_raw.shape)

#Splitting into training and validation sets
permutation = np.random.permutation(X_train_raw.shape[0])
X_train = X_train_raw[permutation]
y_train = y_train_raw[permutation]

X_train = X_train.reshape(-1,128*128*3)
num_train = 12000
num_val = 3000

permutation = np.random.permutation(X_train_raw.shape[0])
X_train = X_train_raw[permutation]
y_train = y_train_raw[permutation]

X_train = X_train.reshape(-1,128*128*3)

X_val = X_train[num_train:]
y_val = y_train[num_train:]

X_train = X_train[:num_train]
y_train = y_train[:num_train]

mean_image = np.mean(X_train, axis=0)
X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
X_val = X_val.astype(np.float32) - mean_image

X_train = X_train.reshape([-1,128,128,3])/255
X_val = X_val.reshape([-1,128,128,3])/255

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

#Training 
tf.reset_default_graph()
kaggle_training(X_train, y_train, X_val, y_val, 
         conv_featmap=[6, 16],
         fc_units=[84],
         conv_kernel_size=[5, 5],
         pooling_size=[2, 2],
         l2_norm=0.01,
         seed=235,
         learning_rate=1e-3,
         epoch=20,
         batch_size=200,
         verbose=False,
         pre_trained_model=None)

#Testing with test imageset and saving the best model
image_list = []
extension = '.png'
path_to_image_folder = './test_128/'
num_test_samples = 3500
img_names = [path_to_image_folder+str(idx)+extension for idx in range(num_test_samples)]

for filename in img_names:
    im = plt.imread(filename)
    image_list.append(im)
    
X_test = np.asarray(image_list)

tf.reset_default_graph()
with tf.Session() as sess: 
        saver = tf.train.import_meta_graph('model/KaggleModel.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        tf_input = graph.get_operations()[0].name+':0'
        x = graph.get_tensor_by_name(tf_input)
        tf_output = 'fc_layer_1/Add:0'
        y = graph.get_tensor_by_name(tf_output)
        y_out = sess.run(y, feed_dict={x: X_test})
        y_pred = np.argmax(y_out,axis=1)
        print(y_pred)

#Generating csv for Kaggle
with open('predicted.csv','w') as csvfile:
    fieldnames = ['Id','label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()    
    for index,l in enumerate(y_pred):
        filename = str(index)+'.png'
        label = str(l)
        writer.writerow({'Id': filename, 'label': label})
