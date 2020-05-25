import tensorflow as tf
from baselines.common.models import register

@register("mydense")
def mydense_builder(num_layers=1, num_units=30, **dense_kwargs):
    def my_network(X):
        #print('mydense', X)
        out = tf.cast(X, tf.float32)
        out = tf.contrib.layers.flatten(out)
        for num_l in range(num_layers):
            out = tf.contrib.layers.fully_connected(
                inputs=out,
                num_outputs=num_units,
                activation_fn=tf.nn.relu
            )
        return out
    return my_network

@register("myconv1d")
def myconv1d_builder(convs=[(32, 3, 1)], **conv1d_kwargs):
    def network_fn(X):
        out = tf.cast(X, tf.float32)
        for num_outputs, kernel_size, stride in convs:
            out = tf.contrib.layers.conv1d(
                inputs=out,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME',
                data_format='NWC',
                activation_fn=tf.nn.relu,  
            )
        return out
    return network_fn

@register("myconv2d")
def myconv2d_builder(convs=[(32, 3, 1)], **conv2d_kwargs):
    # convs = [(filter_number, filter_size, stride)]
    def network_fn(X):
        out = tf.cast(X, tf.float32)
        for num_outputs, kernel_size, stride in convs:
            out = tf.contrib.layers.convolution2d(
                inputs=out,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME',
                data_format='NCHW',
                activation_fn=tf.nn.relu,
            )
        return out
    return network_fn

@register("mynet")
def mynet_builder(convs=[(32, 3, 1)], **conv2d_kwargs):
    # convs = [(filter_number, filter_size, stride)]
    def network_fn(X):
        out = tf.cast(X, tf.float32)
        for num_outputs, kernel_size, stride in convs:
            out = tf.contrib.layers.convolution2d(
                inputs=out,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding='VALID',
                data_format='NCHW',
                activation_fn=tf.nn.relu,
            )
        out = tf.contrib.layers.flatten(out)
        num_units = 32
        out = tf.contrib.layers.fully_connected(
            inputs=out,
            num_outputs=num_units,
            activation_fn=tf.nn.relu
        )
        return out
    return network_fn
