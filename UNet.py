import tensorflow as tf
from tensorflow import keras
from keras import layers

class UNet_Named_Layers():

    def __init__(self, model_name, encoder_names=True, decoder_names=True):
        """ Encoder and decoder names utilised to help with the generation and application of transfer learning weights """
        self.encoder_names = encoder_names
        self.decoder_names = decoder_names
        self.model_name = model_name

    def double_conv_block(self, x, n_filters, layer_num=None):
        """Constructs a network block of two 
        convolution layers with ReLU activation"""
        if layer_num is not None:
            x = layers.Conv2D(
                n_filters, 3, padding="same",
                activation="relu", 
                kernel_initializer="he_normal",
                name=f'conv_{layer_num}_1'
            )(x)
            
            x = layers.Conv2D(
                n_filters, 3, padding="same",
                activation="relu", 
                kernel_initializer="he_normal",
                name=f'conv_{layer_num}_2'
            )(x)
        else:
            x = layers.Conv2D(
                n_filters, 3, padding="same",
                activation="relu", 
                kernel_initializer="he_normal"
            )(x)
            
            x = layers.Conv2D(
                n_filters, 3, padding="same",
                activation="relu", 
                kernel_initializer="he_normal",
            )(x)
        
        return x
    
    def downsample_block(self, x, n_filters, layer_num):
        """Construct a downsampling block with 
        convolution, pooling and dropout"""
        if self.encoder_names:
            f = self.double_conv_block(x, n_filters, f'{layer_num}d')
            p = layers.MaxPool2D(2, name=f'pool_{layer_num}')(f)
            p = layers.Dropout(0.3, name=f'pool_dropout_{layer_num}')(p)
        else:
            print("No encoder names")
            f = self.double_conv_block(x, n_filters)
            p = layers.MaxPool2D(2)(f)
            p = layers.Dropout(0.3)(p)
        
        return f, p
    
    def upsample_block(self, x, conv_features, n_filters, layer_num):
        """Construct an upsampling block with 
        dropout and the convolution"""
        if self.decoder_names:
            x = layers.Conv2DTranspose(
                n_filters, 3, 2, padding="same", name=f'upsample_{layer_num}'
            )(x)
        
            x = layers.concatenate([x, conv_features])
            x = layers.Dropout(0.3, name=f'upsample_dropout_{layer_num}')(x)
            x = self.double_conv_block(x, n_filters, f'{layer_num}u')
        else:
            x = layers.Conv2DTranspose(
                n_filters, 3, 2, padding="same"
            )(x)
        
            x = layers.concatenate([x, conv_features])
            x = layers.Dropout(0.3)(x)
            x = self.double_conv_block(x, n_filters)
        
        return x
    
    def create_model(self, input_shape, num_classes, bottleneck_name=True):
        """Constructs the general U-Net 
        shape using the above functions"""
        inputs = layers.Input(shape=input_shape)
        
        f1, p1 = self.downsample_block(inputs, 64, 1)
        f2, p2 = self.downsample_block(p1, 128, 2)
        f3, p3 = self.downsample_block(p2, 256, 3)
        f4, p4 = self.downsample_block(p3, 512, 4)

        if bottleneck_name:
            bottleneck = self.double_conv_block(p4, 1024, "bottleneck")
        else:
            bottleneck = self.double_conv_block(p4, 1024)
        
        u6 = self.upsample_block(bottleneck, f4, 512, 4)
        u7 = self.upsample_block(u6, f3, 256, 3)
        u8 = self.upsample_block(u7, f2, 128, 2)
        u9 = self.upsample_block(u8, f1, 64, 1)
        
        outputs = layers.Conv2D(
            num_classes, 1, padding="same", 
            activation="softmax",
            name=f"{self.model_name}_outputs"
        )(u9)
        
        unet_model = keras.Model(
            inputs, outputs, name="U-Net"
        )
        
        return unet_model