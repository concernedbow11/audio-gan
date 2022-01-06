from keras.layers import Conv1D, BatchNormalization, Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

#func for building discriminator layers
def build_disc(start_filters, spatial_dim, filter_size):
    #func for building a CNN block for downsampling the image   
    def add_discriminator_block(x, filters, filter_size):
        x = Conv1D(filters, filter_size, strides=2, padding='valid')(x)
        x = LeakyReLU(0.2)(x)
        return x
    
    x = Dense(2048, activation='LeakyReLU')(x)
    x = Dense(1, activation='Sigmoid')(x)
