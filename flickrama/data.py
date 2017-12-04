import flickr_api
import imageio
import numpy as np
import urllib2

class Data:

  def __init__(self, what, size=(240, 240), batch_size=100, flickr_size='Small'):
    '''
    '''

    self._what = what
    self._flickr_size = flickr_size
    self._size = size

    self._flickr_max = 4000
    self._page = 1
    self._batch_size = batch_size

    self._cache = {}


  def next_batch(self, page=None):
    '''
    '''
    increase_page = True
    if not page:
      increase_page = False
      page = self._page
    print 'using page', page

    #
    # check if we reached the flickr max
    #
    if self._batch_size * page > self._flickr_max:

      print 'reset queue'
      page = 1


    at_least_one_cached = False


    photos = flickr_api.Photo.search(tags=self._what, \
                                     sort='relevance', \
                                     content_type=1, \
                                     machine_tag_mode = 'all',
                                     per_page=self._batch_size, \
                                     page=page)

    current_batch = []

    for p in photos:

      current_id = photos[0]['id']

      if current_id in self._cache:

        if not at_least_one_cached:
          print 'use cache at least once'

        at_least_one_cached = True

        current_batch.append(self._cache[current_id])

      else:

        url = p.getSizes()[self._flickr_size]['source']
        img = urllib2.urlopen(url).read()
        img = imageio.imread(img)

        pad_height = (self._size[0]+2) - img.shape[0]
        pad_width = (self._size[1]+2) - img.shape[1]
        
        img = np.pad(img, ((pad_height/2,pad_height/2), (0,0), (0,0)), mode='constant')
        img = np.pad(img, ((0,0), (pad_width/2,pad_width/2), (0,0)), mode='constant')
        
        img = img[:self._size[0],:self._size[1]]
        
        self._cache[current_id] = img

        current_batch.append(img)

    if increase_page:
      self._page += 1

    return current_batch
