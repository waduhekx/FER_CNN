def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_s(x,W, s):
    return tf.nn.conv2d(x,W, strides = [1, s, s, 1], padding = 'SAME')

def conv2d_dw(x,W):
    return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def conv2d_group(x, W, group, s):
    splits_x = tf.split(x, num_or_size_splits=group, axis=3)
    splits_w = tf.split(W, num_or_size_splits=group, axis=3)
    convs_group = []
    for (split_x, split_w) in zip(splits_x, splits_w):
        convs_group.append(tf.nn.conv2d(split_x, split_w, strides = [1, s, s, 1], padding = 'SAME'))
    convs = tf.concat(convs_group, axis = -1)
    return convs

def channel_shuffle(x, group):
    n, h, w, c = x.shape.as_list()
    assert c % group == 0
    x_reshaped = tf.reshape(x, [-1, h, w, group, c // group])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    return output


def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def global_avg_pool(x, w):
    return tf.nn.avg_pool(x, ksize=w, strides=w, padding='SAME')

def avgpool(x):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def maxpool_1(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
