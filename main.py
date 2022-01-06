from keras.layers import Conv1D, BatchNormalization, Input, GlobalAveragePooling2D, Dense, Concatenate, Add
from keras.layers.merge import concatenate
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
    
    def add_generator_block_2(x, filters, filter_size):
        x = Conv1D(filters, filter_size, strides=2, padding='same')(x)
        #x = pixshuf(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x  

    # def add_generator_block_2(x, filters, filter_size):
    #     x = Conv1D(filters, filter_size, strides=2, padding='same')(x)
    #     #y = pixshuf(x)
    #     #x = Add(x,y)
    #     x = BatchNormalization()(x)
    #     x = LeakyReLU(0.2)(x)
    #     return x 

    inp = Input(shape=(latent_dim,))

    # projection of the noise vector into a tensor with 
    # same shape as last conv layer in discriminator
    x = Dense(4 * 4 * (start_filters * 16), input_dim=latent_dim)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(4, 4, start_filters * 8))(x)

    #following code upsamples the image
    g1 = add_generator_block_1(x, start_filters * 16, 3)
    g2 = add_generator_block_1(g1, start_filters * 8, 3)
    g3 = add_generator_block_1(g2, start_filters * 4, 5)
    g4 = add_generator_block_1(g3, start_filters * 2, 7)  
    
    g5 = add_generator_block_2(g4, start_filters * 16, 3)
    concatenate(g5,g3)

    g6 = add_generator_block_2(g5, start_filters * 8, 3)
    concatenate(g6,g2)

    g7 = add_generator_block_2(g6, start_filters * 4, 5)
    concatenate(g7,g1)

    g8 = add_generator_block_2(g7, start_filters * 2, 7) 

    x = Conv1D(3, kernel_size=5, padding='same', activation='LeakyReLU')(g8)

    return Model(inputs=inp, outputs=x)



