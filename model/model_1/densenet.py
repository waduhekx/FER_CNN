def densenet(x):
    x_image = tf.reshape(x, [-1, 48, 48, 1])

    #base
    w_base_1 = weight_variables([5, 5, 1, 128])
    w_base_2 = weight_variables([1, 1, 128, 64])
    b_base_1 = bias_variable([128])
    b_base_2 = bias_variable([64])
    h_base_1 = tf.nn.relu(conv2d(x_image, w_base_1) + b_base_1)
    h_base_2 = tf.nn.relu(conv2d(h_base_1, w_base_2) + b_base_2)

    #ds_block_1
    #ds_block_1_1
    w_ds_1_1_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_1_1_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_1_1_conv1 = bias_variable([32])
    b_ds_1_1_conv3 = bias_variable([32])
    h_ds_1_1_conv1 = tf.nn.relu(conv2d(h_base_2, w_ds_1_1_conv1) + b_ds_1_1_conv1)
    h_ds_1_1_conv3 = tf.nn.nn.relu(conv2d(h_ds_1_1_conv1, w_ds_1_1_conv3) + b_ds_1_1_conv3)
    #ds_block_1_2
    w_ds_1_2_conv1 = weight_variables([1, 1, 32, 32])
    w_ds_1_2_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_1_2_conv1 = bias_variable([32])
    b_ds_1_2_conv3 = bias_variable([32])
    h_ds_1_2_conv1 = tf.nn.relu(conv2d(h_ds_1_1_conv3, w_ds_1_2_conv1) + b_ds_1_2_conv1)
    h_ds_1_2_conv3 = tf.nn.relu(conv2d(h_ds_1_2_conv1, w_ds_1_2_conv3) + b_ds_1_2_conv3)
    #ds_block_1_3
    h_ds_1_concat12 = tf.concat(h_ds_1_1_conv3, h_ds_1_2_conv3)
    w_ds_1_3_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_1_3_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_1_3_conv1 = bias_variable([32])
    b_ds_1_3_conv3 = bias_variable([32])
    h_ds_1_3_conv1 = tf.nn.relu(conv2d(h_ds_1_concat12, w_ds_1_3_conv1) + b_ds_1_3_conv1)
    h_ds_1_3_conv3 = tf.nn.relu(conv2d(h_ds_1_3_conv1, w_ds_1_3_conv3) + b_ds_1_3_conv3)
    #transition layer
    h_ds_1_concat123 = tf.concat(h_ds_1_1_conv3, h_ds_1_2_conv3, h_ds_1_3_conv3)
    w_ds_1_tran_conv1 = weight_variables([1, 1, 96, 64])
    b_ds_1_tran_conv1 = bias_variable([64])
    h_ds_1_tran_conv1 = tf.nn.relu(conv2d(h_ds_1_concat123, w_ds_1_tran_conv1) + b_ds_1_tran_conv1)
    h_ds_1_tran_pool = avgpool(h_ds_1_tran_conv1)

    #ds_block_2
    #ds_block_2_1
    w_ds_2_1_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_2_1_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_2_1_conv1 = bias_variable([32])
    b_ds_2_1_conv3 = bias_variable([32])
    h_ds_2_1_conv1 = tf.nn.relu(conv2d(h_ds_1_tran_pool, w_ds_2_1_conv1) + b_ds_2_1_conv1)
    h_ds_2_1_conv3 = tf.nn.nn.relu(conv2d(h_ds_2_1_conv1, w_ds_2_1_conv3) + b_ds_2_1_conv3)
    #ds_block_1_2
    w_ds_2_2_conv1 = weight_variables([1, 1, 32, 32])
    w_ds_2_2_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_2_2_conv1 = bias_variable([32])
    b_ds_2_2_conv3 = bias_variable([32])
    h_ds_2_2_conv1 = tf.nn.relu(conv2d(h_ds_2_1_conv3, w_ds_2_2_conv1) + b_ds_2_2_conv1)
    h_ds_2_2_conv3 = tf.nn.relu(conv2d(h_ds_2_2_conv1, w_ds_2_2_conv3) + b_ds_2_2_conv3)
    #ds_block_1_3
    h_ds_2_concat12 = tf.concat(h_ds_2_1_conv3, h_ds_2_2_conv3)
    w_ds_2_3_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_2_3_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_2_3_conv1 = bias_variable([32])
    b_ds_2_3_conv3 = bias_variable([32])
    h_ds_2_3_conv1 = tf.nn.relu(conv2d(h_ds_2_concat12, w_ds_2_3_conv1) + b_ds_2_3_conv1)
    h_ds_2_3_conv3 = tf.nn.relu(conv2d(h_ds_2_3_conv1, w_ds_2_3_conv3) + b_ds_2_3_conv3)
    #ds_block_1_4
    h_ds_2_concat123 = tf.concat(h_ds_2_1_conv3, h_ds_2_2_conv3, h_ds_2_3_conv3)
    w_ds_2_4_conv1 = weight_variables([1, 1, 96, 32])
    w_ds_2_4_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_2_4_conv1 = bias_variable([32])
    b_ds_2_4_conv3 = bias_variable([32])
    h_ds_2_4_conv1 = tf.nn.relu(conv2d(h_ds_2_concat123, w_ds_2_4_conv1) + b_ds_2_4_conv1)
    h_ds_2_4_conv3 = tf.nn.relu(conv2d(h_ds_2_4_conv1, w_ds_2_4_conv3) + b_ds_2_4_conv3)
    #transition layer
    h_ds_2_concat1234 = tf.concat(h_ds_2_1_conv3, h_ds_2_2_conv3, h_ds_2_3_conv3, h_ds_2_4_conv3)
    w_ds_2_tran_conv1 = weight_variables([1, 1, 128, 64])
    b_ds_2_tran_conv1 = bias_variable([64])
    h_ds_2_tran_conv1 = tf.nn.relu(conv2d(h_ds_2_concat1234, w_ds_2_tran_conv1) + b_ds_2_tran_conv1)
    h_ds_2_tran_pool = avgpool(h_ds_2_tran_conv1)

    #ds_block_3
    #ds_block_3_1
    w_ds_3_1_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_3_1_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_3_1_conv1 = bias_variable([32])
    b_ds_3_1_conv3 = bias_variable([32])
    h_ds_3_1_conv1 = tf.nn.relu(conv2d(h_ds_2_tran_pool, w_ds_3_1_conv1) + b_ds_3_1_conv1)
    h_ds_3_1_conv3 = tf.nn.nn.relu(conv2d(h_ds_3_1_conv1, w_ds_3_1_conv3) + b_ds_3_1_conv3)
    #ds_block_1_2
    w_ds_3_2_conv1 = weight_variables([1, 1, 32, 32])
    w_ds_3_2_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_3_2_conv1 = bias_variable([32])
    b_ds_3_2_conv3 = bias_variable([32])
    h_ds_3_2_conv1 = tf.nn.relu(conv2d(h_ds_3_1_conv3, w_ds_3_2_conv1) + b_ds_3_2_conv1)
    h_ds_3_2_conv3 = tf.nn.relu(conv2d(h_ds_3_2_conv1, w_ds_3_2_conv3) + b_ds_3_2_conv3)
    #ds_block_1_3
    h_ds_3_concat12 = tf.concat(h_ds_3_1_conv3, h_ds_3_2_conv3)
    w_ds_3_3_conv1 = weight_variables([1, 1, 64, 32])
    w_ds_3_3_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_3_3_conv1 = bias_variable([32])
    b_ds_3_3_conv3 = bias_variable([32])
    h_ds_3_3_conv1 = tf.nn.relu(conv2d(h_ds_3_concat12, w_ds_3_3_conv1) + b_ds_3_3_conv1)
    h_ds_3_3_conv3 = tf.nn.relu(conv2d(h_ds_3_3_conv1, w_ds_3_3_conv3) + b_ds_3_3_conv3)
    #ds_block_1_4
    h_ds_3_concat123 = tf.concat(h_ds_3_1_conv3, h_ds_3_2_conv3, h_ds_3_3_conv3)
    w_ds_3_4_conv1 = weight_variables([1, 1, 96, 32])
    w_ds_3_4_conv3 = weight_variables([3, 3, 32, 32])
    b_ds_3_4_conv1 = bias_variable([32])
    b_ds_3_4_conv3 = bias_variable([32])
    h_ds_3_4_conv1 = tf.nn.relu(conv2d(h_ds_3_concat123, w_ds_3_4_conv1) + b_ds_3_4_conv1)
    h_ds_3_4_conv3 = tf.nn.relu(conv2d(h_ds_3_4_conv1, w_ds_3_4_conv3) + b_ds_3_4_conv3)
    #transition layer
    h_ds_3_concat1234 = tf.concat(h_ds_3_1_conv3, h_ds_3_2_conv3, h_ds_3_3_conv3, h_ds_3_4_conv3)
    w_ds_3_tran_conv1 = weight_variables([1, 1, 128, 64])
    b_ds_3_tran_conv1 = bias_variable([64])
    h_ds_3_tran_conv1 = tf.nn.relu(conv2d(h_ds_3_concat1234, w_ds_3_tran_conv1) + b_ds_3_tran_conv1)
    h_ds_3_tran_pool = avgpool(h_ds_3_tran_conv1)

    #global avg pool
    h_gap = global_avg_pool(h_ds_3_tran_pool, [1, 6, 6, 1])
    h_flat = tf.reshape(h_gap, [-1, 64])
    w_fc = weight_variables([64, 7])
    b_fc = bias_variable([7])
    y_conv = tf.matmul(h_flat, w_fc) + b_fc
    return y_conv
