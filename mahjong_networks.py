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

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    from tensorflow.keras import layers
    filters1, filters2 = filters
    # if backend.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, 3,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # x = layers.Conv2D(filters3, (1, 1),
    #                   padding='same',
    #                   kernel_initializer='he_normal',
    #                   name=conv_name_base + '2c')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


# /Users/shimizutapishi/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/keras_applications/
# /Users/shimizutaishi/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages/keras/applications/
@register("myresnet")
def myresnet_builder(num_blocks=10, **resnet_kwargs):
    from tensorflow.keras import layers

    def network_fn(X):
        # if backend.image_data_format() == 'channels_last':
        #     bn_axis = 3
        # else:
        #     bn_axis = 1
        bn_axis = 1
        
        # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(X)
        x = layers.Conv2D(128, 3,
                          padding='same',
                          kernel_initializer='he_normal',
                          name='conv1')(X)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        # x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        # x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        
        #x = conv_block(X, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        for i in range(num_blocks):
            x = identity_block(x, 3, [128, 128], stage=i, block=str(i))
        
        #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        # x = identity_block(x, 3, [128, 128], stage=3, block='b')
        # x = identity_block(x, 3, [128, 128], stage=3, block='c')
        # x = identity_block(x, 3, [128, 128], stage=3, block='d')
        
        #x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        # x = identity_block(x, 3, [256, 256], stage=4, block='b')
        # x = identity_block(x, 3, [256, 256], stage=4, block='c')
        # x = identity_block(x, 3, [256, 256], stage=4, block='d')
        # x = identity_block(x, 3, [256, 256], stage=4, block='e')
        # x = identity_block(x, 3, [256, 256], stage=4, block='f')
        
        #x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
        return x
    
    return network_fn


def resnet_layer(
        input_tensor,
        num_filters=64,
        kernel_size=3,
        strides=1,
        activation='relu',
        batch_normalization=True
    ):
    from tensorflow.keras import layers
    x = layers.Conv1D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal',)(input_tensor)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
        #x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x


@register("myresnet1")
def myresnet_builder(num_blocks=5, num_filters=64, **resnet_kwargs):
    print('num_blocks', num_blocks)
    from tensorflow.keras import layers
    def network_fn(X):
        kernel_size = 3
        x = resnet_layer(X, num_filters, kernel_size)
        for i in range(num_blocks):
            y = resnet_layer(x, num_filters, kernel_size)
            y = resnet_layer(y, num_filters, activation=None)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        #x = resnet_layer(x, num_filters=2)
        x = resnet_layer(x, num_filters=num_filters)
        conv1d = layers.Conv1D(2, kernel_size=1, padding='same', kernel_initializer='he_normal')
        x = conv1d(x)
        return x
    return network_fn

# /home/t_shimizu/baselines/baselines/common
def resnet_layer_v2(
        input_tensor,
        num_filters=64,
        kernel_size=3,
        strides=1,
        activation=True,
        batch_normalization=True
    ):
    x = tf.layers.conv1d(
        input_tensor,
        num_filters,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal'
    )
    if batch_normalization:
        x = tf.layers.batch_normalization(x)
    if activation:
        x = tf.nn.relu(x)
    return x

@register("myresnet1_v2")
def myresnet1_v2_builder(num_blocks=5, num_filters=64, **resnet_kwargs):
    def network_fn(X):
        x = resnet_layer_v2(X, num_filters=num_filters)
        for i in range(num_blocks):
            y = resnet_layer_v2(x, num_filters=num_filters)
            y = resnet_layer_v2(y, num_filters=num_filters, activation=False)
            x = tf.math.add(x, y)
            x = tf.nn.relu(x)
        x = resnet_layer_v2(x, num_filters=num_filters)
        x = tf.layers.conv1d(x, filters=2, kernel_size=1)
        return x
    return network_fn
