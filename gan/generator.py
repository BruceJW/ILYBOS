from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

class Generator:

  def __init__(self):
    '''
    '''
    self.N = None

    self._noise_size = None

  def setup(self, noise_size=(100,)):
    '''
    '''

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_shape=noise_size))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    # model.summary()

    noise = Input(shape=noise_size)
    image = model(noise)

    self._noise_size = noise_size
    self.N = Model(noise, image)


