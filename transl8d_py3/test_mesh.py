'''
Code to dress SMPL with registered garments.
Set the "path" variable in this code to the downloaded Multi-Garment Dataset

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
Shout out to Chaitanya for intersection removal code
'''

from psbody.mesh import Mesh, MeshViewers
from utils.smpl_paths import SmplPaths
from lib.ch_smpl import Smpl
from utils.interpenetration_ind import remove_interpenetration_fast
import numpy as np

import pickle as pkl
from os.path import join, split
from glob import glob
from time import sleep
from pprint import pprint as p
import datetime

if __name__ == '__main__':
    person_fname='/home/nathanbendich/MultiGarmentNetwork/assets/cust_body_SMPL_b4_fine_tuning___2020_03_08____11:25_AM__.obj'
    person= Mesh(filename=person_fname)
    pants_fname='/home/nathanbendich/MultiGarmentNetwork/transl8d_py3/clothes_objs/125611512607128/Pants.obj'
    pants = Mesh(filename= pants_fname)
    pants_tex_fname='/home/nathanbendich/MultiGarmentNetwork/transl8d_py3/clothes_objs/125611512607128/multi_tex.jpg'
    pants.set_texture_image(pants_tex_fname) # This line ("`set_texture_image()`") worked too!   -nxb, Fri Mar  6 19:29:00 EST 2020

    mvs = MeshViewers((1, 1), keepalive=True)
    mvs[0][0].set_static_meshes([person, pants])
    #mvs[0][0].set_static_meshes([pants])
    timestamp=datetime.datetime.now().strftime('__%Y_%m_%d____%H:%M_%p__')
    out_pants_fname='person_w_pants{}.png'.format(timestamp)
    # TODO:  repose / color before saving render.  This can go after the MVP, though.  They'll probably want fit before they want to see their own color?
    # person.set_texture_image(person_tex_fname)  # we have to get this from the 1-8 images.

    #dressed = dress(jonah, pants) # TODO FIXME   change this line to something that works. -nxb, March 6, 2020
    mvs[0][0].save_snapshot(out_pants_fname)

    sleep(10)
    print('Done')









