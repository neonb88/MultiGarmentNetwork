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

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import imageio as ii

from cx import parse_obj_file

flags.DEFINE_string('img_path', 'hmr/data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()

if __name__=="__main__":
  config = flags.FLAGS
  config(sys.argv)
  # Using pre-trained model, change this to use your own.
  config.load_path = src.config.PRETRAINED_MODEL
  config.batch_size = 1

  renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
  #img_path='/home/nathanbendich/CIHP_PGN/datasets/CIHP/images/img0.jpg'
  img_path='/home/nathanbendich/CIHP_PGN/datasets/CIHP/images_May_19_2020_____11:40_P.M._EDT/img0.jpg'

  #TODO: NOTE this is the sketchiest, shittiest way to get this done.  But at least we should have SOMETHING to mess with
  #TODO: change!
  #TODO: change!
  #TODO: change!
  #TODO: change!
  #TODO: change!
  #TODO: NOTE this is the sketchiest, shittiest way to get this done.  But at least we should have SOMETHING to mess with
  tmp_obj_fname='assets/MGN_obj__2020_May_19____11:39_AM__/cust.obj'
  vs_naked, _ = parse_obj_file(tmp_obj_fname)
  visualize( 
    np.asarray(ii.imread(img_path)), # "img"
    {'end_pt': np.array([336, 336]), # "proc_param"
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
    #model['vertices_naked'], # from MGN.  # the real one is in "make_SMPL_mesh.py_____dev.py"
    vs_naked,
    np.array([ 1.23492968, -0.18135118,  0.35370526]) # cams[0]  (prob "camera parameters")
  )

























































