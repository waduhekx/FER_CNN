def shuffle_v2_original(x, group):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    #base
    w_base = weight_variables([5, 5, 1, 32])
    b_base = bias_variable([32])
    h_base = tf.nn.relu(conv2d(x_image, w_base) + b_base)

    #stage1
    #stage1_1
    #left channel
    w_s1_1_conv3_l = weight_variables([3, 3, int(32/group), 32])
    w_s1_1_conv1_l = weight_variables([1, 1, 32, 32])
    b_s1_1_conv3_l = bias_variable([32])
    b_s1_1_conv1_l = bias_variable([32])
    h_s1_1_conv3_l = tf.nn.relu(conv2d_group(h_base, w_s1_1_conv3_l, group, 2) + b_s1_1_conv3_l)
    h_s1_1_conv1_l = tf.nn.relu(conv2d(h_s1_1_conv3_l, w_s1_1_conv1_l) + b_s1_1_conv1_l)
    #right channel
    w_s1_1_conv1_1_r = weight_variables([1, 1, 32, 32])
    w_s1_1_dwconv3_r = weight_variables([3, 3, int(32/group), 32])
    w_s1_1_conv1_2_r = weight_variables([1, 1, 32, 32])
    b_s1_1_conv1_1_r = bias_variable([32])
    b_s1_1_dwconv3_r = bias_variable([32])
    b_s1_1_conv1_2_r = bias_variable([32])
    h_s1_1_conv1_1_r = tf.nn.relu(conv2d(h_base, w_s1_1_conv1_1_r) + b_s1_1_conv1_1_r)
    h_s1_1_dwconv3_r = tf.nn.relu(conv2d_group(h_s1_1_conv1_1_r, w_s1_1_dwconv3_r, group, 2) + b_s1_1_dwconv3_r)
    h_s1_1_conv1_2_r = tf.nn.relu(conv2d(h_s1_1_dwconv3_r, w_s1_1_conv1_2_r) + b_s1_1_conv1_2_r)
    #concat
    h_s1_1_concat = tf.concat([h_s1_1_conv1_l, h_s1_1_conv1_2_r], 3)
    #shuffle
    h_s1_1_shuffle = channel_shuffle(h_s1_1_concat, 4)

    #stage1_2
    [split_s1_2_l, split_s1_2_r] = tf.split(h_s1_1_shuffle, 2, 3)
    #right channel
    w_s1_2_conv1_1_r = weight_variables([1, 1, 32, 32])
    w_s1_2_dwconv3_r = weight_variables([3, 3, int(32/group), 32])
    w_s1_2_conv1_2_r = weight_variables([1, 1, 32, 32])
    b_s1_2_conv1_1_r = bias_variable([32])
    b_s1_2_dwconv3_r = bias_variable([32])
    b_s1_2_conv1_2_r = bias_variable([32])
    h_s1_2_conv1_1_r = tf.nn.relu(conv2d(split_s1_2_r, w_s1_2_conv1_1_r) + b_s1_2_conv1_1_r)
    h_s1_2_dwconv3_r = tf.nn.relu(conv2d_group(h_s1_2_conv1_1_r, w_s1_2_dwconv3_r, group, 1) + b_s1_2_dwconv3_r)
    h_s1_2_conv1_2_r = tf.nn.relu(conv2d(h_s1_2_dwconv3_r, w_s1_2_conv1_2_r) + b_s1_2_conv1_2_r)
    #concat
    h_s1_2_concat = tf.concat([split_s1_2_l, h_s1_2_conv1_2_r], 3)
    #shuffle
    h_s1_2_shuffle = channel_shuffle(h_s1_2_concat, 4)

    #stage1_3
    [split_s1_3_l, split_s1_3_r] = tf.split(h_s1_2_shuffle, 2, 3)
    #right channel
    w_s1_3_conv1_1_r = weight_variables([1, 1, 32, 32])
    w_s1_3_dwconv3_r = weight_variables([3, 3, int(32/group), 32])
    w_s1_3_conv1_2_r = weight_variables([1, 1, 32, 32])
    b_s1_3_conv1_1_r = bias_variable([32])
    b_s1_3_dwconv3_r = bias_variable([32])
    b_s1_3_conv1_2_r = bias_variable([32])
    h_s1_3_conv1_1_r = tf.nn.relu(conv2d(split_s1_3_r, w_s1_3_conv1_1_r) + b_s1_3_conv1_1_r)
    h_s1_3_dwconv3_r = tf.nn.relu(conv2d_group(h_s1_3_conv1_1_r, w_s1_3_dwconv3_r, group, 1) + b_s1_3_dwconv3_r)
    h_s1_3_conv1_2_r = tf.nn.relu(conv2d(h_s1_3_dwconv3_r, w_s1_3_conv1_2_r) + b_s1_3_conv1_2_r)
    #concat
    h_s1_3_concat = tf.concat([split_s1_3_l, h_s1_3_conv1_2_r], 3)
    #shuffle
    h_s1_3_shuffle = channel_shuffle(h_s1_3_concat, 4)

    #stage1_4
    [split_s1_4_l, split_s1_4_r] = tf.split(h_s1_3_shuffle, 2, 3)
    #right channel
    w_s1_4_conv1_1_r = weight_variables([1, 1, 32, 32])
    w_s1_4_dwconv3_r = weight_variables([3, 3, int(32/group), 32])
    w_s1_4_conv1_2_r = weight_variables([1, 1, 32, 32])
    b_s1_4_conv1_1_r = bias_variable([32])
    b_s1_4_dwconv3_r = bias_variable([32])
    b_s1_4_conv1_2_r = bias_variable([32])
    h_s1_4_conv1_1_r = tf.nn.relu(conv2d(split_s1_4_r, w_s1_4_conv1_1_r) + b_s1_4_conv1_1_r)
    h_s1_4_dwconv3_r = tf.nn.relu(conv2d_group(h_s1_4_conv1_1_r, w_s1_4_dwconv3_r, group, 1) + b_s1_4_dwconv3_r)
    h_s1_4_conv1_2_r = tf.nn.relu(conv2d(h_s1_4_dwconv3_r, w_s1_4_conv1_2_r) + b_s1_4_conv1_2_r)
    #concat
    h_s1_4_concat = tf.concat([split_s1_4_l, h_s1_4_conv1_2_r], 3)
    #shuffle
    h_s1_4_shuffle = channel_shuffle(h_s1_4_concat, 4)

    #stage2
    #stage_2_1
    #left channel
    w_s2_1_conv3_l = weight_variables([3, 3, int(64/group), 64])
    w_s2_1_conv1_l = weight_variables([1, 1, 64, 64])
    b_s2_1_conv3_l = bias_variable([64])
    b_s2_1_conv1_l = bias_variable([64])
    h_s2_1_conv3_l = tf.nn.relu(conv2d_group(h_s1_4_shuffle, w_s2_1_conv3_l, group, 2) + b_s2_1_conv3_l)
    h_s2_1_conv1_l = tf.nn.relu(conv2d(h_s2_1_conv3_l, w_s2_1_conv1_l) + b_s2_1_conv1_l)
    #right channel
    w_s2_1_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_1_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_1_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_1_conv1_1_r = bias_variable([64])
    b_s2_1_dwconv3_r = bias_variable([64])
    b_s2_1_conv1_2_r = bias_variable([64])
    h_s2_1_conv1_1_r = tf.nn.relu(conv2d(h_s1_4_shuffle, w_s2_1_conv1_1_r) + b_s2_1_conv1_1_r)
    h_s2_1_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_1_conv1_1_r, w_s2_1_dwconv3_r, group, 2) + b_s2_1_dwconv3_r)
    h_s2_1_conv1_2_r = tf.nn.relu(conv2d(h_s2_1_dwconv3_r, w_s2_1_conv1_2_r) + b_s2_1_conv1_2_r)
    #concat
    h_s2_1_concat = tf.concat([h_s2_1_conv1_l, h_s2_1_conv1_2_r], 3)
    #shuffle
    h_s2_1_shuffle = channel_shuffle(h_s2_1_concat, 4)

    #stage_2_2
    [split_s2_2_l, split_s2_2_r] = tf.split(h_s2_1_shuffle, 2, 3)
    #right channel
    w_s2_2_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_2_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_2_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_2_conv1_1_r = bias_variable([64])
    b_s2_2_dwconv3_r = bias_variable([64])
    b_s2_2_conv1_2_r = bias_variable([64])
    h_s2_2_conv1_1_r = tf.nn.relu(conv2d(split_s2_2_r, w_s2_2_conv1_1_r) + b_s2_2_conv1_1_r)
    h_s2_2_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_2_conv1_1_r, w_s2_2_dwconv3_r, group, 1) + b_s2_2_dwconv3_r)
    h_s2_2_conv1_2_r = tf.nn.relu(conv2d(h_s2_2_dwconv3_r, w_s2_2_conv1_2_r) + b_s2_2_conv1_2_r)
    #concat
    h_s2_2_concat = tf.concat([split_s2_2_l, h_s2_2_conv1_2_r], 3)
    #shuffle
    h_s2_2_shuffle = channel_shuffle(h_s2_2_concat, 4)

    #stage_2_3
    [split_s2_3_l, split_s2_3_r] = tf.split(h_s2_2_shuffle, 2, 3)
    #right channel
    w_s2_3_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_3_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_3_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_3_conv1_1_r = bias_variable([64])
    b_s2_3_dwconv3_r = bias_variable([64])
    b_s2_3_conv1_2_r = bias_variable([64])
    h_s2_3_conv1_1_r = tf.nn.relu(conv2d(split_s2_3_r, w_s2_3_conv1_1_r) + b_s2_3_conv1_1_r)
    h_s2_3_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_3_conv1_1_r, w_s2_3_dwconv3_r, group, 1) + b_s2_3_dwconv3_r)
    h_s2_3_conv1_2_r = tf.nn.relu(conv2d(h_s2_3_dwconv3_r, w_s2_3_conv1_2_r) + b_s2_3_conv1_2_r)
    #concat
    h_s2_3_concat = tf.concat([split_s2_3_l, h_s2_3_conv1_2_r], 3)
    #shuffle
    h_s2_3_shuffle = channel_shuffle(h_s2_3_concat, 4)

    #stage_2_4
    [split_s2_4_l, split_s2_4_r] = tf.split(h_s2_3_shuffle, 2, 3)
    #right channel
    w_s2_4_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_4_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_4_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_4_conv1_1_r = bias_variable([64])
    b_s2_4_dwconv3_r = bias_variable([64])
    b_s2_4_conv1_2_r = bias_variable([64])
    h_s2_4_conv1_1_r = tf.nn.relu(conv2d(split_s2_4_r, w_s2_4_conv1_1_r) + b_s2_4_conv1_1_r)
    h_s2_4_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_4_conv1_1_r, w_s2_4_dwconv3_r, group, 1) + b_s2_4_dwconv3_r)
    h_s2_4_conv1_2_r = tf.nn.relu(conv2d(h_s2_4_dwconv3_r, w_s2_4_conv1_2_r) + b_s2_4_conv1_2_r)
    #concat
    h_s2_4_concat = tf.concat([split_s2_4_l, h_s2_4_conv1_2_r], 3)
    #shuffle
    h_s2_4_shuffle = channel_shuffle(h_s2_4_concat, 4)

    #stage_2_5
    [split_s2_5_l, split_s2_5_r] = tf.split(h_s2_4_shuffle, 2, 3)
    #right channel
    w_s2_5_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_5_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_5_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_5_conv1_1_r = bias_variable([64])
    b_s2_5_dwconv3_r = bias_variable([64])
    b_s2_5_conv1_2_r = bias_variable([64])
    h_s2_5_conv1_1_r = tf.nn.relu(conv2d(split_s2_5_r, w_s2_5_conv1_1_r) + b_s2_5_conv1_1_r)
    h_s2_5_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_5_conv1_1_r, w_s2_5_dwconv3_r, group, 1) + b_s2_5_dwconv3_r)
    h_s2_5_conv1_2_r = tf.nn.relu(conv2d(h_s2_5_dwconv3_r, w_s2_5_conv1_2_r) + b_s2_5_conv1_2_r)
    #concat
    h_s2_5_concat = tf.concat([split_s2_5_l, h_s2_5_conv1_2_r], 3)
    #shuffle
    h_s2_5_shuffle = channel_shuffle(h_s2_5_concat, 4)

    #stage_2_6
    [split_s2_6_l, split_s2_6_r] = tf.split(h_s2_5_shuffle, 2, 3)
    #right channel
    w_s2_6_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_6_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_6_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_6_conv1_1_r = bias_variable([64])
    b_s2_6_dwconv3_r = bias_variable([64])
    b_s2_6_conv1_2_r = bias_variable([64])
    h_s2_6_conv1_1_r = tf.nn.relu(conv2d(split_s2_6_r, w_s2_6_conv1_1_r) + b_s2_6_conv1_1_r)
    h_s2_6_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_6_conv1_1_r, w_s2_6_dwconv3_r, group, 1) + b_s2_6_dwconv3_r)
    h_s2_6_conv1_2_r = tf.nn.relu(conv2d(h_s2_6_dwconv3_r, w_s2_6_conv1_2_r) + b_s2_6_conv1_2_r)
    #concat
    h_s2_6_concat = tf.concat([split_s2_6_l, h_s2_6_conv1_2_r], 3)
    #shuffle
    h_s2_6_shuffle = channel_shuffle(h_s2_6_concat, 4)

    #stage_2_7
    [split_s2_7_l, split_s2_7_r] = tf.split(h_s2_6_shuffle, 2, 3)
    #right channel
    w_s2_7_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_7_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_7_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_7_conv1_1_r = bias_variable([64])
    b_s2_7_dwconv3_r = bias_variable([64])
    b_s2_7_conv1_2_r = bias_variable([64])
    h_s2_7_conv1_1_r = tf.nn.relu(conv2d(split_s2_7_r, w_s2_7_conv1_1_r) + b_s2_7_conv1_1_r)
    h_s2_7_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_7_conv1_1_r, w_s2_7_dwconv3_r, group, 1) + b_s2_7_dwconv3_r)
    h_s2_7_conv1_2_r = tf.nn.relu(conv2d(h_s2_7_dwconv3_r, w_s2_7_conv1_2_r) + b_s2_7_conv1_2_r)
    #concat
    h_s2_7_concat = tf.concat([split_s2_7_l, h_s2_7_conv1_2_r], 3)
    #shuffle
    h_s2_7_shuffle = channel_shuffle(h_s2_7_concat, 4)

    #stage_2_8
    [split_s2_8_l, split_s2_8_r] = tf.split(h_s2_7_shuffle, 2, 3)
    #right channel
    w_s2_8_conv1_1_r = weight_variables([1, 1, 64, 64])
    w_s2_8_dwconv3_r = weight_variables([3, 3, int(64/group), 64])
    w_s2_8_conv1_2_r = weight_variables([1, 1, 64, 64])
    b_s2_8_conv1_1_r = bias_variable([64])
    b_s2_8_dwconv3_r = bias_variable([64])
    b_s2_8_conv1_2_r = bias_variable([64])
    h_s2_8_conv1_1_r = tf.nn.relu(conv2d(split_s2_8_r, w_s2_8_conv1_1_r) + b_s2_8_conv1_1_r)
    h_s2_8_dwconv3_r = tf.nn.relu(conv2d_group(h_s2_8_conv1_1_r, w_s2_8_dwconv3_r, group, 1) + b_s2_8_dwconv3_r)
    h_s2_8_conv1_2_r = tf.nn.relu(conv2d(h_s2_8_dwconv3_r, w_s2_8_conv1_2_r) + b_s2_8_conv1_2_r)
    #concat
    h_s2_8_concat = tf.concat([split_s2_8_l, h_s2_8_conv1_2_r], 3)
    #shuffle
    h_s2_8_shuffle = channel_shuffle(h_s2_8_concat, 4)

    #stage3
    #stage_3_1
    #left channel
    w_s3_1_conv3_l = weight_variables([3, 3, int(128/group), 128])
    w_s3_1_conv1_l = weight_variables([1, 1, 128, 128])
    b_s3_1_conv3_l = bias_variable([128])
    b_s3_1_conv1_l = bias_variable([128])
    h_s3_1_conv3_l = tf.nn.relu(conv2d_group(h_s2_8_shuffle, w_s3_1_conv3_l, group, 2) + b_s3_1_conv3_l)
    h_s3_1_conv1_l = tf.nn.relu(conv2d(h_s3_1_conv3_l, w_s3_1_conv1_l) + b_s3_1_conv1_l)
    #right channel
    w_s3_1_conv1_1_r = weight_variables([1, 1, 128, 128])
    w_s3_1_dwconv3_r = weight_variables([3, 3, int(128/group), 128])
    w_s3_1_conv1_2_r = weight_variables([1, 1, 128, 128])
    b_s3_1_conv1_1_r = bias_variable([128])
    b_s3_1_dwconv3_r = bias_variable([128])
    b_s3_1_conv1_2_r = bias_variable([128])
    h_s3_1_conv1_1_r = tf.nn.relu(conv2d(h_s2_8_shuffle, w_s3_1_conv1_1_r) + b_s3_1_conv1_1_r)
    h_s3_1_dwconv3_r = tf.nn.relu(conv2d_group(h_s3_1_conv1_1_r, w_s3_1_dwconv3_r, group, 2) + b_s3_1_dwconv3_r)
    h_s3_1_conv1_2_r = tf.nn.relu(conv2d(h_s3_1_dwconv3_r, w_s3_1_conv1_2_r) + b_s3_1_conv1_2_r)
    #concat
    h_s3_1_concat = tf.concat([h_s3_1_conv1_l, h_s3_1_conv1_2_r], 3)
    #shuffle
    h_s3_1_shuffle = channel_shuffle(h_s3_1_concat, 4)

    #stage_3_2
    [split_s3_2_l, split_s3_2_r] = tf.split(h_s3_1_shuffle, 2, 3)
    #right channel
    w_s3_2_conv1_1_r = weight_variables([1, 1, 128, 128])
    w_s3_2_dwconv3_r = weight_variables([3, 3, int(128/group), 128])
    w_s3_2_conv1_2_r = weight_variables([1, 1, 128, 128])
    b_s3_2_conv1_1_r = bias_variable([128])
    b_s3_2_dwconv3_r = bias_variable([128])
    b_s3_2_conv1_2_r = bias_variable([128])
    h_s3_2_conv1_1_r = tf.nn.relu(conv2d(split_s3_2_r, w_s3_2_conv1_1_r) + b_s3_2_conv1_1_r)
    h_s3_2_dwconv3_r = tf.nn.relu(conv2d_group(h_s3_2_conv1_1_r, w_s3_2_dwconv3_r, group, 1) + b_s3_2_dwconv3_r)
    h_s3_2_conv1_2_r = tf.nn.relu(conv2d(h_s3_2_dwconv3_r, w_s3_2_conv1_2_r) + b_s3_2_conv1_2_r)
    #concat
    h_s3_2_concat = tf.concat([split_s3_2_l, h_s3_2_conv1_2_r], 3)
    #shuffle
    h_s3_2_shuffle = channel_shuffle(h_s3_2_concat, 4)

    #stage_3_3
    [split_s3_3_l, split_s3_3_r] = tf.split(h_s3_2_shuffle, 2, 3)
    #right channel
    w_s3_3_conv1_1_r = weight_variables([1, 1, 128, 128])
    w_s3_3_dwconv3_r = weight_variables([3, 3, int(128/group), 128])
    w_s3_3_conv1_2_r = weight_variables([1, 1, 128, 128])
    b_s3_3_conv1_1_r = bias_variable([128])
    b_s3_3_dwconv3_r = bias_variable([128])
    b_s3_3_conv1_2_r = bias_variable([128])
    h_s3_3_conv1_1_r = tf.nn.relu(conv2d(split_s3_3_r, w_s3_3_conv1_1_r) + b_s3_3_conv1_1_r)
    h_s3_3_dwconv3_r = tf.nn.relu(conv2d_group(h_s3_3_conv1_1_r, w_s3_3_dwconv3_r, group, 1) + b_s3_3_dwconv3_r)
    h_s3_3_conv1_2_r = tf.nn.relu(conv2d(h_s3_3_dwconv3_r, w_s3_3_conv1_2_r) + b_s3_3_conv1_2_r)
    #concat
    h_s3_3_concat = tf.concat([split_s3_3_l, h_s3_3_conv1_2_r], 3)
    #shuffle
    h_s3_3_shuffle = channel_shuffle(h_s3_3_concat, 4)

    #stage_3_4
    [split_s3_4_l, split_s3_4_r] = tf.split(h_s3_3_shuffle, 2, 3)
    #right channel
    w_s3_4_conv1_1_r = weight_variables([1, 1, 128, 128])
    w_s3_4_dwconv3_r = weight_variables([3, 3, int(128/group), 128])
    w_s3_4_conv1_2_r = weight_variables([1, 1, 128, 128])
    b_s3_4_conv1_1_r = bias_variable([128])
    b_s3_4_dwconv3_r = bias_variable([128])
    b_s3_4_conv1_2_r = bias_variable([128])
    h_s3_4_conv1_1_r = tf.nn.relu(conv2d(split_s3_4_r, w_s3_4_conv1_1_r) + b_s3_4_conv1_1_r)
    h_s3_4_dwconv3_r = tf.nn.relu(conv2d_group(h_s3_4_conv1_1_r, w_s3_4_dwconv3_r, group, 1) + b_s3_4_dwconv3_r)
    h_s3_4_conv1_2_r = tf.nn.relu(conv2d(h_s3_4_dwconv3_r, w_s3_4_conv1_2_r) + b_s3_4_conv1_2_r)
    #concat
    h_s3_4_concat = tf.concat([split_s3_4_l, h_s3_4_conv1_2_r], 3)
    #shuffle
    h_s3_4_shuffle = channel_shuffle(h_s3_4_concat, 4)

    #stage4 conv1
    w_s4_conv1 = weight_variables([1, 1, 256, 256])
    b_s4_conv1 = bias_variable([256])
    h_s4_conv1 = tf.nn.relu(conv2d(h_s3_4_shuffle, w_s4_conv1) + b_s4_conv1)
    #global_avg_pool
    h_gap = global_avg_pool(h_s4_conv1, [1, 6, 6, 1])
    #flat
    h_flat = tf.reshape(h_gap, [-1, 256])
    #fc
    w_fc = weight_variables([256, 7])
    b_fc = bias_variable([7])
    h_fc = tf.matmul(h_flat, w_fc) + b_fc
    return h_fc
