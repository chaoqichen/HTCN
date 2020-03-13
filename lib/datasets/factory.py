# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.sim10k import sim10k
from datasets.clipart import clipart
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.kitti_car import kitti_car

import numpy as np


###########################################
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval','trainval_cg']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007']:
  for split in ['train', 'val', 'train_cg']:
    name = 'clipart_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split : clipart(split,year))

for year in ['2007']:
  for split in ['train', 'val', 'train_combine_fg', 'train_cg_fg']:
    name = 'cs_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: cityscape(split, year))

for year in ['2007']:
  for split in ['train', 'val', 'train_combine','train_cg']:
    name = 'cs_fg_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: foggy_cityscape(split, year))

for year in ['2012']:
  for split in ['trainval', 'trainval_combine']:
    name = 'sim10k_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: sim10k(split, '2012'))

for year in ['2007']:
  for split in ['train', 'val', 'train_combine','train_combine_kt']:
    name = 'cs_car_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: cityscape_car(split, year))

for year in ['2007']:
  for split in ['trainval', 'trainval_combine']:
    name = 'kitti_car_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: kitti_car(split, year))

###########################################


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
