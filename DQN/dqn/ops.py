import tensorflow as tf

def clipped_error(x):
    """
        # Huber loss (delta = 1)
        L(a) = if abs(a) < delta -> 0.5*a*a
               else -> delta*(abs(a) - 0.5*delta)
    """
    # function where return the coordination of the values which is meet for the condition
    # where(condition, x, y) 根据condition返回x或y中的元素
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    """
        conv2d layer with self-defined params
    """
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            # Notice the third params is the channel number that is -> C
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim],tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format=data_format)

        # TODO: 这个return放在with外面还是里面有区别吗？源码编写方式有点奇怪
        if activation_fn != None:
            out = activation_fn(out)

        return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    """
        Dense layer with self-defined params
        # TODO: We can add initializer function params in the input params
    """
    # shape is batch_size * all_params_list
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size], tf.float32,
                initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            out = activation_fn(out)

        return out, w, b
