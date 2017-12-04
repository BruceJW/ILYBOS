import keras
import keras.backend as K
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator

class WGAN:
  '''
  Wasserstein GAN

  Heavily inspired from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
  '''

  @staticmethod
  def loss(y_true, y_pred):
      return K.mean(y_true * y_pred)


  def __init__(self):
    '''
    '''
    self._image_size = None

    # Following parameter and optimizer set as recommended in paper
    self._n_critic = 5
    self._clip_value = 0.01
    self._optimizer = keras.optimizers.RMSprop(lr=0.00005)

    self._batch_size = 32
    self._epochs = 4000
    self._save_image_interval = 50


    self._discriminator = None
    self._generator = None

    self.N = None


  def setup(self, image_size=(28,28,1), ):
    '''
    '''

    # setup discriminator
    discriminator = Discriminator()
    discriminator.setup(image_size)
    discriminator.N.compile(loss=WGAN.loss, optimizer=self._optimizer, metrics=['accuracy'])
    discriminator.N.trainable = False
    self._discriminator = discriminator

    # setup generator
    generator = Generator()
    generator.setup()
    generator.N.compile(loss=WGAN.loss, optimizer=self._optimizer)
    self._generator = generator

    # setup GAN
    noise = keras.layers.Input(shape=generator._noise_size)
    generated_images = self._generator.N(noise)
    validated_images = self._discriminator.N(generated_images)

    self.N = keras.models.Model(noise, validated_images)
    self.N.compile(loss=WGAN.loss, optimizer=self._optimizer, metrics=['accuracy'])

    # self.N.summary()


  def train(self, X_train):
    '''
    '''

    half_batch = int(self._batch_size / 2)

    for epoch in range(self._epochs):

      for _ in range(self._n_critic):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, self._generator._noise_size[0]))

        # Generate a half batch of new images
        gen_imgs = self._generator.N.predict(noise)

        # Train the discriminator
        d_loss_real = self._discriminator.N.train_on_batch(imgs, -np.ones((half_batch, 1)))
        d_loss_fake = self._discriminator.N.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Clip discriminator weights
        for l in self._discriminator.N.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -self._clip_value, self._clip_value) for w in weights]
            l.set_weights(weights)


      # ---------------------
      #  Train Generator
      # ---------------------

      noise = np.random.normal(0, 1, (self._batch_size, self._generator._noise_size[0]))

      # Train the generator
      g_loss = self.N.train_on_batch(noise, -np.ones((self._batch_size, 1)))

      # Plot the progress
      print "%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0])

      if epoch % self._save_image_interval == 0:
          self.store_generated_images(epoch)


  def store_generated_images(self, epoch):

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = self._generator.N.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 1

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("/tmp/mnist_%d.png" % epoch)
    plt.close()
