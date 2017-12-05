import flickr_api
import imageio
import numpy as np
import scipy.misc
import urllib2

class Data:

  def __init__(self, what, size=(240, 240), batch_size=100, flickr_size='Small', grayscale=False, normalize=False, expand=False, downsample=False):
    '''
    '''

    self._what = what
    self._flickr_size = flickr_size
    self._size = size

    self._flickr_max = 4000
    self._page = 1
    self._batch_size = batch_size

    self._grayscale = grayscale
    self._normalize = normalize
    self._expand = expand
    self._downsample = downsample

    self._cache = {}


  def next(self, page=None):
    '''
    '''
    increase_page = True
    if page:
      increase_page = False
    else:
      page = self._page

    #
    # check if we reached the flickr max
    #
    if self._batch_size * page > self._flickr_max:
      page -= self._flickr_max

    print 'using page', page

    photos = flickr_api.Photo.search(tags=self._what, \
                                     sort='relevance', \
                                     content_type=1, \
                                     machine_tag_mode = 'all',
                                     per_page=self._batch_size, \
                                     page=page)

    current_batch = []

    for p in photos:

      current_id = p['id']

      if current_id in self._cache:

        current_batch.append(self._cache[current_id])

      else:

        try:
          url = p.getSizes()[self._flickr_size]['source']
        except:
          # reached the bottom, restart now
          self._page = 1
          return current_batch



        img = urllib2.urlopen(url).read()
        img = imageio.imread(img)

        pad_height = (self._size[0]+2) - img.shape[0]
        pad_width = (self._size[1]+2) - img.shape[1]
        
        img = np.pad(img, ((pad_height/2,pad_height/2), (0,0), (0,0)), mode='constant')
        img = np.pad(img, ((0,0), (pad_width/2,pad_width/2), (0,0)), mode='constant')
        
        img = img[:self._size[0],:self._size[1]]
        
        # normalize

        # grayscale
        if self._grayscale:
          img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

          if self._normalize:
            img = (img.astype(np.float32) - 127.5) / 127.5

          if self._downsample:
            img = scipy.misc.imresize(img, (28,28))

          if self._expand:
            img = np.expand_dims(img, axis=2)

        self._cache[current_id] = img

        current_batch.append(img)

    if increase_page:
      self._page += 1

    return current_batch
