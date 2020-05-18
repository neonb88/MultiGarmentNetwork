"""
Shows MGN results superimposed on original images,   just like Kanazawa did in hmr/demo.py

Editor /Author : Nathan X Bendich (nxb)
date : May 18, 2020
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from hmr.src.util import renderer as vis_util
from hmr.src.util import image as img_util
from hmr.src.util import openpose as op_util
import hmr.src.config
from hmr.src.RunModel import RunModel

from hmr.demo import visualize

import imageio as ii

if __name__=="__main__":
  config = flags.FLAGS
  renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
  img_path='/home/nathanbendich/intermediate_steps_ClothX/images/img0.jpg' # TODO:  I think we'll need to change this

  #TODO: this is the sketchiest, shittiest way to get this done.  But at least we should have SOMETHING to mess with
  visualize( 
    np.asarray(ii.imread(img_path)), # "img"
    {'end_pt': np.array([336, 336]) # "proc_param"
      'img_size': 225,
      'scale': 1.0,
      'start_pt': np.array([112,112]),
    },
    np.array([
       [  86.60865021, 163.28405762],
       [ 128.131073  , 128.39300537],
       [  75.79335785, 126.55383301],
       [  88.04506683, 120.98227692],
       [ 137.06164551, 114.05370331],
       [  99.83338165, 155.52114868],
       [  98.47254181, 153.68869019],
       [  85.55800629, 117.11668396],
       [ 106.13191223,  81.75570679],
       [ 120.21483612,  71.11662292],
       [ 136.10830688,  85.37127686],
       [ 162.79808044,  94.84520721],
       [ 118.37543488,  70.18653107],
       [ 134.57743835,  44.83231354],
       [ 145.21495056,  62.38152695],
       [ 141.02410889,  54.53079224],
       [ 144.30183411,  57.21850586],
       [ 126.66687775,  51.3151474 ],
       [ 133.51797485,  57.24398041],
    ]),# "joints[0]"
    model['vertices_naked'], # from MGN.  # the real one is in "make_SMPL_mesh.py_____dev.py"
    np.array([ 1.23492968, -0.18135118,  0.35370526]) # cams[0]  (prob "camera parameters")
  )
