def mobilenet_original(x):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    # conv1 3*3
    w_conv1 = weight_variables([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    # conv_dw1
    w_convdw_1 = weight_variables([3, 3, 64, 1])
    b_convdw_1 = bias_variable([64])
    h_convdw_1 = tf.nn.relu(conv2d_dw(h_conv1, w_convdw_1) + b_convdw_1)
    # conv_pw1
    w_convpw_1 = weight_variables([1, 1, 64, 64])
    b_convpw_1 = bias_variable([64])
    h_convpw_1 = tf.nn.relu(conv2d(h_convdw_1, w_convpw_1) + b_convpw_1)

    # conv_dw_2
    w_convdw_2 = weight_variables([3, 3, 64, 1])
    b_convdw_2 = bias_variable([64])
    h_convdw_2 = tf.nn.relu(conv2d_dw(h_convpw_1, w_convdw_2) + b_convdw_2)
    # conv_pw_2
    w_convpw_2 = weight_variables([3, 3, 64, 64])
    b_convpw_2 = bias_variable([64])
    h_convpw_2 = tf.nn.relu(conv2d_s(h_convdw_2, w_convpw_2, 2) + b_convpw_2)

    # conv_dw_3
    w_convdw_3 = weight_variables([3, 3, 64, 1])
    b_convdw_3 = bias_variable([64])
    h_convdw_3 = tf.nn.relu(conv2d_dw(h_convpw_2, w_convdw_3) + b_convdw_3)
    # conv_pw_3, s=2
    w_convpw_3 = weight_variables([1, 1, 64, 128])
    b_convpw_3 = bias_variable([128])
    h_convpw_3 = tf.nn.relu(conv2d(h_convdw_3, w_convpw_3) + b_convpw_3)

    # conv_dw_4
    w_convdw_4 = weight_variables([3, 3, 128, 1])
    b_convdw_4 = bias_variable([128])
    h_convdw_4 = tf.nn.relu(conv2d_dw(h_convpw_3, w_convdw_4) + b_convdw_4)
    # conv_pw_4
    w_convpw_4 = weight_variables([1, 1, 128, 128])
    b_convpw_4 = bias_variable([128])
    h_convpw_4 = tf.nn.relu(conv2d_s(h_convdw_4, w_convpw_4, 2) + b_convpw_4)

    # conv_dw_5
    w_convdw_5 = weight_variables([3, 3, 128, 1])
    b_convdw_5 = bias_variable([128])
    h_convdw_5 = tf.nn.relu(conv2d_dw(h_convpw_4, w_convdw_5) + b_convdw_5)
    # conv_pw_5
    w_convpw_5 = weight_variables([1, 1, 128, 128])
    b_convpw_5 = bias_variable([128])
    h_convpw_5 = tf.nn.relu(conv2d(h_convdw_5, w_convpw_5) + b_convpw_5)

    # conv_dw_6
    w_convdw_6 = weight_variables([3, 3, 128, 1])
    b_convdw_6 = bias_variable([128])
    h_convdw_6 = tf.nn.relu(conv2d_dw(h_convpw_5, w_convdw_6) + b_convdw_6)
    # conv_pw_6, s=2
    w_convpw_6 = weight_variables([1, 1, 128, 256])
    b_convpw_6 = bias_variable([256])
    h_convpw_6 = tf.nn.relu(conv2d_s(h_convdw_6, w_convpw_6, 2) + b_convpw_6)

    # conv_dw_7
    w_convdw_7 = weight_variables([3, 3, 256, 1])
    b_convdw_7 = bias_variable([256])
    h_convdw_7 = tf.nn.relu(conv2d_dw(h_convpw_6, w_convdw_7) + b_convdw_7)
    # conv_pw_7
    w_convpw_7 = weight_variables([1, 1, 256, 256])
    b_convpw_7 = bias_variable([256])
    h_convpw_7 = tf.nn.relu(conv2d_s(h_convdw_7, w_convpw_7, 1) + b_convpw_7)

    # conv_dw_8
    w_convdw_8 = weight_variables([3, 3, 256, 1])
    b_convdw_8 = bias_variable([256])
    h_convdw_8 = tf.nn.relu(conv2d_dw(h_convpw_7, w_convdw_8) + b_convdw_8)
    # conv_pw_8
    w_convpw_8 = weight_variables([1, 1, 256, 256])
    b_convpw_8 = bias_variable([256])
    h_convpw_8 = tf.nn.relu(conv2d_s(h_convdw_8, w_convpw_8, 1) + b_convpw_8)

    # conv_dw_9
    w_convdw_9 = weight_variables([3, 3, 256, 1])
    b_convdw_9 = bias_variable([256])
    h_convdw_9 = tf.nn.relu(conv2d_dw(h_convpw_8, w_convdw_9) + b_convdw_9)
    # conv_pw_9
    w_convpw_9 = weight_variables([1, 1, 256, 256])
    b_convpw_9 = bias_variable([256])
    h_convpw_9 = tf.nn.relu(conv2d_s(h_convdw_9, w_convpw_9, 1) + b_convpw_9)

    # conv_dw_10
    w_convdw_10 = weight_variables([3, 3, 256, 1])
    b_convdw_10 = bias_variable([256])
    h_convdw_10 = tf.nn.relu(conv2d_dw(h_convpw_9, w_convdw_10) + b_convdw_10)
    # conv_pw_10
    w_convpw_10 = weight_variables([1, 1, 256, 256])
    b_convpw_10 = bias_variable([256])
    h_convpw_10 = tf.nn.relu(conv2d_s(h_convdw_10, w_convpw_10, 1) + b_convpw_10)

    # conv_dw_11
    w_convdw_11 = weight_variables([3, 3, 256, 1])
    b_convdw_11 = bias_variable([256])
    h_convdw_11 = tf.nn.relu(conv2d_dw(h_convpw_10, w_convdw_11) + b_convdw_11)
    # conv_pw_11
    w_convpw_11 = weight_variables([1, 1, 256, 256])
    b_convpw_11 = bias_variable([256])
    h_convpw_11 = tf.nn.relu(conv2d_s(h_convdw_11, w_convpw_11, 1) + b_convpw_11)

    # conv_dw_12
    w_convdw_12 = weight_variables([3, 3, 256, 1])
    b_convdw_12 = bias_variable([256])
    h_convdw_12 = tf.nn.relu(conv2d_dw(h_convpw_11, w_convdw_12) + b_convdw_12)
    # conv_pw_12, s=2
    w_convpw_12 = weight_variables([1, 1, 256, 512])
    b_convpw_12 = bias_variable([512])
    h_convpw_12 = tf.nn.relu(conv2d_s(h_convdw_12, w_convpw_12, 12) + b_convpw_12)

    # conv_dw_13
    w_convdw_13 = weight_variables([3, 3, 512, 1])
    b_convdw_13 = bias_variable([512])
    h_convdw_13 = tf.nn.relu(conv2d_dw(h_convpw_12, w_convdw_13) + b_convdw_13)
    # conv_pw_13
    w_convpw_13 = weight_variables([1, 1, 512, 512])
    b_convpw_13 = bias_variable([512])
    h_convpw_13 = tf.nn.relu(conv2d_s(h_convdw_13, w_convpw_13, 1) + b_convpw_13)

    # conv_dw_14
    w_convdw_14 = weight_variables([3, 3, 512, 1])
    b_convdw_14 = bias_variable([512])
    h_convdw_14 = tf.nn.relu(conv2d_dw(h_convpw_13, w_convdw_14) + b_convdw_14)
    # conv_pw_14, s=2
    w_convpw_14 = weight_variables([1, 1, 512, 1024])
    b_convpw_14 = bias_variable([1024])
    h_convpw_14 = tf.nn.relu(conv2d_s(h_convdw_14, w_convpw_14, 2) + b_convpw_14)

    # conv_dw_15
    w_convdw_15 = weight_variables([3, 3, 1024, 1])
    b_convdw_15 = bias_variable([1024])
    h_convdw_15 = tf.nn.relu(conv2d_dw(h_convpw_14, w_convdw_15) + b_convdw_15)
    # conv_pw_15
    w_convpw_15 = weight_variables([1, 1, 1024, 1024])
    b_convpw_15 = bias_variable([1024])
    h_convpw_15 = tf.nn.relu(conv2d_s(h_convdw_15, w_convpw_15, 1) + b_convpw_15)
    # global_avg_pool
    #h_gap = global_avg_pool(h_convpw_6, [1, 6, 6, 1])
    # fc
    h_flat = tf.reshape(h_convpw_8, [-1, 1024 * 3 * 3])
    w_fc = weight_variables([1024 * 3 * 3, 7])
    b_fc = bias_variable([7])
    y_conv = (tf.matmul(h_flat, w_fc) + b_fc)

    return y_conv
