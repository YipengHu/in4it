#utility functions
import tensorflow as tf
from tensorflow.keras import layers


class ResUNet(tf.keras.Model):

    def __init__(self, init_ch=32, num_levels=3, out_ch=1):
        super(ResUNet, self).__init__()
        self.encoder = [self._resnet_block(2**i*init_ch, type='down') for i in range(num_levels)]
        self.encoder += [self._resnet_block(2**num_levels*init_ch, type='none')]
        self.decoder = [self._resnet_block(2**i*init_ch, type='up') for i in range(num_levels,0,-1)] 
        self.decoder += [self._resnet_block(init_ch, type='none')]
        self.out_layer = layers.Conv2D(
            filters=out_ch,
            kernel_size=(3,3),
            strides=(1,1), 
            padding='same', 
            use_bias=True, 
            activation='sigmoid')

    def call(self, inputs):
        x = inputs
        skips = []
        for down in self.encoder[:-1]:
            x = down(x)
            skips += [x]
        x = self.encoder[-1](x)
        for up, skip in zip(self.decoder[:-1], reversed(skips)):
            x = self._resize_to(x,skip) + skip
            x = up(x)
        x = self._resize_to(x,inputs)
        x = self.decoder[-1](x)
        return self.out_layer(x)

    def build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))

    def _resnet_block(self, ch, type, bn=True):

        def _call(_input):

            x = layers.Conv2D(
                filters=ch,
                kernel_size=(3,3),
                strides=(1,1), 
                padding='same', 
                use_bias=False, 
                activation='relu')(_input)
            x = layers.BatchNormalization()(x) if bn else x
            x = layers.Conv2D(
                filters=ch,
                kernel_size=(3,3),
                strides=(1,1), 
                padding='same', 
                use_bias=False, 
                activation='relu')(x)
            x = layers.BatchNormalization()(x) if bn else x

            x += _input

            if type == "down":
                x = layers.Conv2D(
                    filters=ch*2,
                    kernel_size=(3,3),
                    strides=(1,1), 
                    padding='same', 
                    use_bias=False, 
                    activation='relu')(x)
                x = layers.MaxPool2D(
                    pool_size=(2,2),
                    strides=(2,2), 
                    padding='same')(x)
                y = layers.BatchNormalization()(x) if bn else x
            elif type == "up":
                x = layers.Conv2DTranspose(
                    filters=ch/2,
                    kernel_size=(3,3),
                    strides=(2,2), 
                    padding='same',
                    output_padding=1,
                    use_bias=False, 
                    activation='relu')(x)
                y = layers.BatchNormalization()(x) if bn else x
            else:  #none
                y = x
            
            return y

        return _call
    
    def _resize_to(self, x, y):
        if x.shape[1:3]==y.shape[1:3]:
            return x
        elif  any([abs(x.shape[i]-y.shape[i])>1 for i in [1,2]]):
            raise Warning('padding/cropping more than 1.')
        else:
            return tf.image.resize_with_crop_or_pad(x, y.shape[1], y.shape[2])
