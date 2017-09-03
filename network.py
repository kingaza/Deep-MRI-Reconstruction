import numpy as np

import keras
import keras.backend as K
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Conv2D, Lambda, PReLU
from keras.regularizers import l2
from keras.models import Model
from keras.utils.vis_utils import plot_model

def complex2real(x):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        2d/3d/4d complex valued tensor (n, nx, ny) or (n, nx, ny, nt)

    Returns
    -------
    y: 4d tensor (n, 2, nx, ny)
    '''
    x_real = np.real(x)
    x_imag = np.imag(x)
    y = np.array([x_real, x_imag]).astype(np.float32)
    # re-order in convenient order
    if x.ndim >= 3:
        y = y.swapaxes(0, 1)
    return y


def real2complex(x):
    '''
    Converts from array of the form ([n, ]2, nx, ny[, nt]) to ([n, ]nx, ny[, nt])
    '''
    x = np.asarray(x)
    
        
    if x.shape[0] == 2 and x.shape[1] != 2:  # Hacky check
        return x[0] + x[1] * 1j
    elif x.shape[1] == 2:
        y = x[:, 0] + x[:, 1] * 1j
        return y
    else:
        raise ValueError('Invalid dimension')


def mask_c2r(m):
    return complex2real(m * (1+1j))


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def complex_to_network(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        raise ValueError('3D is not supported')

    if mask:  # Hacky solution
        x = x*(1+1j)

    x = complex2real(x)
    
    if K.image_data_format() == 'channels_last':
        x = x.transpose(0, 2, 3, 1)

    return x


def complex_from_network(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """
    if x.ndim == 5:  # n 3D inputs. reorder axes
        raise ValueError('3D is not supported')

    if mask:
        x = mask_r2c(x)
    else:
        x = real2complex(x)

    return x


     
def build_simple_model(shape, block_num=3):
        
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    
    for i in range(block_num):
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Conv2D(64, (3, 3), padding='same'))
    
    model.add(Conv2D(2, (1, 1), padding='same'))
    
    return model

    
def FFT(x):
    image = K.tf.complex(x[:,:,:,0], x[:,:,:,1])
    raw =  K.tf.fft2d(image)
    real = K.tf.real(raw)
    imag = K.tf.imag(raw)
    return K.tf.stack([real, imag], axis=3)


def IFFT(x):
    image = K.tf.complex(x[:,:,:,0], x[:,:,:,1])
    raw =  K.tf.ifft2d(image)
    real = K.tf.real(raw)
    imag = K.tf.imag(raw)
    return K.tf.stack([real, imag], axis=3)    


def build_res_model(shape, l2_reg):
    """
    TODO: Create network with more flexibility
    TODO: densely net in future
    """    
    
    input_image = Input(shape=shape, dtype=np.float32)
    input_raw = Input(shape=shape, dtype=np.float32)
    lost_mask = Input(shape=shape, dtype=np.float32)
    
    # Convolutional Block
    x = Conv2D(64, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(input_image)
    x = PReLU('zero')(x)
    x = Conv2D(64, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(x)
    x = PReLU('zero')(x)
    x = Conv2D(2, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(x)
    x = PReLU('zero')(x)
    x = layers.add([x, input_image])
    
    # Data Consistancy Block
    x = Lambda(FFT)(x)
    x = layers.multiply([x,lost_mask])
    x = layers.add([x, input_raw])
    input2 = Lambda(IFFT)(x)
    
    # Convolutional Block
    x = Conv2D(64, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(input2)
    x = PReLU('zero')(x)
    x = Conv2D(64, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(x)
    x = PReLU('zero')(x)
    x = Conv2D(2, (3, 3), 
               padding='same', 
               kernel_initializer='he_normal',
               kernel_regularizer=l2(l2_reg))(x)
    x = PReLU('zero')(x)
    x = layers.add([x, input2])
    
    # Data Consistancy Block
    x = Lambda(FFT)(x)
    x = layers.multiply([x,lost_mask])
    x = layers.add([x, input_raw])
    x = Lambda(IFFT)(x)
    
    # Residual connection
    x = layers.add([x, input_image])
    
    model = Model([input_image, input_raw, lost_mask], x)
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['accuracy'])    
    
    model.summary()
    plot_model(model, to_file='model.png',show_shapes=True)    
    
    return model
    

