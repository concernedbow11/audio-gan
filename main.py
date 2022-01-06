from keras.layers import Conv1D, BatchNormalization, Input, GlobalAveragePooling2D, Dense, merge
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Deconvolution2D, Reshape

#func for building discriminator layers
def build_disc(start_filters, spatial_dim, filter_size):
    #func for building a CNN block for downsampling the image   
    def add_disc_block(x, filters, filter_size):
        x = Conv1D(filters, filter_size, strides=2, padding='valid')(x)
        x = LeakyReLU(0.2)(x)
        return x
    
    inp = Input(shape=(spatial_dim,spatial_dim,2))
    x = add_disc_block(inp, start_filters, filter_size)
    x = add_disc_block(x, start_filters * 4, 3)
    x = add_disc_block(x, start_filters * 8, 5)
    x = add_disc_block(x, start_filters * 16, 7)

    x = Dense(2048, activation='LeakyReLU')
    x = Dense(1, activation='Sigmoid')(x)

    return Model(inputs=inp, outputs=x)

def build_generator(start_filters, filter_size, latent_dim):
    def add_generator_block_1(x, filters, filter_size):
      x = Conv1D(filters, filter_size, strides=2, padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.2)(x)
      return x
    
    
    g1 = add_generator_block_1(5, start_filters * 4, filter_size)
    g2 = add_generator_block_1(x, start_filters * 2, filter_size)
    g3 = add_generator_block_1(x, start_filters, filter_size)
    g4 = add_generator_block_1(x, start_filters, filter_size)  


