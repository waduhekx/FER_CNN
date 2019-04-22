def inception_v3(x, is_training):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    # inception_v3   maybe mini
    # conv_base
    w_conv_base = weight_variables([5, 5, 1, 128])
    b_conv_base = bias_variable([128])
    h_conv_base = tf.nn.relu(conv2d(x_image, w_conv_base) + b_conv_base)
    h_norm_base = tf.nn.lrn(h_conv_base, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    h_pool_base = maxpool(h_norm_base)

    # inception_1
    # conv1
    w_icp_1_conv_1 = weight_variables([1, 1, 128, 64])
    b_icp_1_conv_1 = bias_variable([64])
    h_icp_1_conv_1 = tf.nn.relu(((conv2d(h_pool_base, w_icp_1_conv_1) + b_icp_1_conv_1)))
    # conv3
    w_icp_1_conv_3_1 = weight_variables([1, 1, 128, 64])
    w_icp_1_conv_3_3 = weight_variables([3, 3, 64, 64])
    b_icp_1_conv_3_1 = bias_variable([64])
    b_icp_1_conv_3_3 = bias_variable([64])
    h_icp_1_conv_3_1 = tf.nn.relu(
        ((conv2d(h_pool_base, w_icp_1_conv_3_1) + b_icp_1_conv_3_1)))
    h_icp_1_conv_3_3 = tf.nn.relu(
        ((conv2d(h_icp_1_conv_3_1, w_icp_1_conv_3_3) + b_icp_1_conv_3_3)))
    # conv5
    w_icp_1_conv_5_1 = weight_variables([1, 1, 128, 64])
    w_icp_1_conv_5_5 = weight_variables([5, 5, 64, 64])
    b_icp_1_conv_5_1 = bias_variable([64])
    b_icp_1_conv_5_5 = bias_variable([64])
    h_icp_1_conv_5_1 = tf.nn.relu(
        ((conv2d(h_pool_base, w_icp_1_conv_5_1) + b_icp_1_conv_5_1)))
    h_icp_1_conv_5_5 = tf.nn.relu(
        ((conv2d(h_icp_1_conv_5_1, w_icp_1_conv_5_5) + b_icp_1_conv_5_5)))
    # pool
    w_icp_1_pool_conv = weight_variables([1, 1, 128, 64])
    b_icp_1_pool_conv = bias_variable([64])
    h_icp_1_pool_conv = tf.nn.relu(
        ((conv2d(h_pool_base, w_icp_1_pool_conv) + b_icp_1_pool_conv)))
    h_icp_1_pool = maxpool_1(h_icp_1_pool_conv)
    # concat1
    h_concat1 = tf.concat([h_icp_1_conv_1, h_icp_1_conv_3_3, h_icp_1_conv_5_5, h_icp_1_pool], 3)

    # maxpool
    h_maxpool1 = maxpool(h_concat1)

    # inception_2
    # conv1
    w_icp_2_conv_1 = weight_variables([1, 1, 256, 64])
    b_icp_2_conv_1 = bias_variable([64])
    h_icp_2_conv_1 = tf.nn.relu(((conv2d(h_maxpool1, w_icp_2_conv_1) + b_icp_2_conv_1)))
    # conv3
    w_icp_2_conv_3_1 = weight_variables([1, 1, 256, 64])
    w_icp_2_conv_3_3_1 = weight_variables([3, 1, 64, 64])
    w_icp_2_conv_3_1_3 = weight_variables([1, 3, 64, 64])
    b_icp_2_conv_3_1 = bias_variable([64])
    b_icp_2_conv_3_3_1 = bias_variable([64])
    b_icp_2_conv_3_1_3 = bias_variable([64])
    h_icp_2_conv_3_1 = tf.nn.relu(
        ((conv2d(h_maxpool1, w_icp_2_conv_3_1) + b_icp_2_conv_3_1)))
    h_icp_2_conv_3_3_1 = tf.nn.relu(
        ((conv2d(h_icp_2_conv_3_1, w_icp_2_conv_3_3_1) + b_icp_2_conv_3_3_1)))
    h_icp_2_conv_3_1_3 = tf.nn.relu(
        ((conv2d(h_icp_2_conv_3_1, w_icp_2_conv_3_1_3) + b_icp_2_conv_3_1_3)))
    # conv5
    w_icp_2_conv_5_1 = weight_variables([1, 1, 256, 64])
    w_icp_2_conv_5_3_3 = weight_variables([3, 3, 64, 64])
    w_icp_2_conv_5_3_1 = weight_variables([3, 1, 64, 64])
    w_icp_2_conv_5_1_3 = weight_variables([1, 3, 64, 64])
    b_icp_2_conv_5_1 = bias_variable([64])
    b_icp_2_conv_5_3_3 = bias_variable([64])
    b_icp_2_conv_5_3_1 = bias_variable([64])
    b_icp_2_conv_5_1_3 = bias_variable([64])
    h_icp_2_conv_5_1 = tf.nn.relu(
        ((conv2d(h_maxpool1, w_icp_2_conv_5_1) + b_icp_2_conv_5_1)))
    h_icp_2_conv_5_3_3 = tf.nn.relu(
        ((conv2d(h_icp_2_conv_5_1, w_icp_2_conv_5_3_3) + b_icp_2_conv_5_3_3)))
    h_icp_2_conv_5_3_1 = tf.nn.relu(
        ((conv2d(h_icp_2_conv_5_3_3, w_icp_2_conv_5_3_1) + b_icp_2_conv_5_3_1)))
    h_icp_2_conv_5_1_3 = tf.nn.relu(
        ((conv2d(h_icp_2_conv_5_3_3, w_icp_2_conv_5_1_3) + b_icp_2_conv_5_1_3)))
    # pool
    w_icp_2_pool_conv = weight_variables([1, 1, 256, 64])
    b_icp_2_pool_conv = bias_variable([64])
    h_icp_2_pool_conv = tf.nn.relu(
        (conv2d(h_maxpool1, w_icp_2_pool_conv) + b_icp_2_pool_conv))
    h_icp_2_pool = maxpool_1(h_icp_2_pool_conv)
    # concat
    h_concat2 = tf.concat(
        [h_icp_2_conv_1, h_icp_2_conv_3_3_1, h_icp_2_conv_3_1_3, h_icp_2_conv_5_3_1, h_icp_2_conv_5_1_3, h_icp_2_pool],
        3)
    # global_avg_pool
    h_avg_2 = global_avg_pool(h_concat2, [1, 6, 6, 1])
    #h_pool = maxpool(h_concat2)
    h_flat_2 = tf.reshape(h_avg_2, [-1, 384 * 2 * 2])
    w_fc2 = weight_variables([384 * 2 * 2, 7])
    b_fc2 = bias_variable([7])
    # fc
    return tf.nn.relu(((tf.matmul(h_flat_2, w_fc2) + b_fc2)))
