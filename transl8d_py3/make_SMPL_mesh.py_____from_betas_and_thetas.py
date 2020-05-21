'''
Code to dress SMPL with registered garments.
Set the "path" variable in this code to the downloaded Multi-Garment Dataset

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
Shout out to Chaitanya for intersection removal code


  NOTE: More similar to SMPL_sandbox.py rather than to make_SMPL_mesh.py -nxb, Wed May 20 22:26:57 EDT 2020
'''

from psbody.mesh import Mesh, MeshViewers

from utils.smpl_paths import SmplPaths
from lib.ch_smpl import Smpl
from utils.interpenetration_ind import remove_interpenetration_fast

import numpy as np
import cv2

import pickle as pkl
from os.path import join, split
from glob import glob
import datetime

def load_smpl_from_file(file):
    dat = pkl.load(open(file, 'rb'), encoding='latin1')
    dp = SmplPaths(gender=dat['gender'])
    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    smpl_h.pose[:] = dat['pose']
    smpl_h.betas[:] = dat['betas']
    smpl_h.trans[:] = dat['trans']

    return smpl_h

# TODO:
# NOTE: Documentation for SMPL high resolution:
"""

#  There might be a reference for this (what each m.pose value means) somewhere.  But I doubt it.

#smpl_h.pose[1]
#m.pose[1] = 0.34 # rotates roughly around the axis from bellybutton to spinal cord.
#m.pose[2] = 0.33 # rotates rougly around the axis from foot to head
#=======================================
#m.pose[3] = -0.9  # left leg back (positive)
#m.pose[4] =  0.9 # left foot out (~marching band splayed out feet pose, positive)
#m.pose[5] =  0.9 # left foot away 
#=======================================
#m.pose[6] =  0.9  # values [6] thru [8] are identical, but for right leg, foot, etc.
#============ right leg ================
#m.pose[9]  =  0.9 # rotates torso down (bend over to touch toes)
#m.pose[10] = 0.9 # rotating to crack back
#m.pose[11] = 0.9 # torso again
#============ left leg ================
#m.pose[12]=0.9 # left knee
#m.pose[13]=0.9 # left leg rotates a diff way
#m.pose[14]=0.9 # left leg again
#============ right leg ================
#m.pose[15]=0.9 # right knee (tibia up into air)
#m.pose[18]=0.9 # bend down

#m.pose[27]=1.9 # head down
#m.pose[31]=3.9 # toes
#m.pose[36]=1.9 # also head down.  Why??  But it doesn't look like ALL of the poses repeat cyclically.  Just a few...  Or maybe I can't recognize the subtle differences between, say, m.pose[36] and m.pose[27].
#m.pose[39]=1.1 # a hand movement!  Good news, Chris Columbus; there IS land on the other side of the ocean.
#m.pose[42]=1.4   # right arm pitch? yaw? roll?

## Rotates (like a backflip)
m.pose[0] =  pi  # NOTE # I'm p sure these rotations (I think it was pose[:3] that are the rotations) are taken care of differently in HMR.
#m.pose[0] = 2.4
#m.pose[41]= #-pi/4 #-1.1  # m.pose[41] is shoulder-level rotation.
#m.pose[44]=   pi/4 # 1.1  # m.pose[44] is shoulder-level rotation.

#m.pose[41]= -0.9
#m.pose[44]=  0.9     # 0.9 is good arm pose to fit in the gucci.obj shirt/sweater/top that  Simo sent us on Slack.
"""

def pose_garment(garment, vert_indices, smpl_params):
    '''
    :param smpl_params: dict with pose, betas, v_template, trans, gender
    '''
    dp = SmplPaths(gender=smpl_params['gender'])
    smpl = Smpl(dp.get_hres_smpl_model_data() )
    smpl.pose[:] = 0
    smpl.betas[:] = smpl_params['betas']
    # smpl.v_template[:] = smpl_params['v_template']

    offsets = np.zeros_like(smpl.r)
    offsets[vert_indices] = garment.v - smpl.r[vert_indices]
    smpl.v_personal[:] = offsets
    smpl.pose[:] = smpl_params['pose']
    smpl.trans[:] = smpl_params['trans']

    mesh = Mesh(smpl.r, smpl.f).keep_vertices(vert_indices)
    return mesh

def retarget(garment_mesh, src, tgt):
    '''
    For each vertex finds the closest point and
    :return:
    '''
    from psbody.mesh import Mesh
    verts, _ = src.closest_vertices(garment_mesh.v)
    verts = np.array(verts)
    tgt_garment = garment_mesh.v - src.v[verts] + tgt.v[verts]
    return Mesh(tgt_garment, garment_mesh.f)

def dress(smpl_tgt, body_src, garment, vert_inds, garment_tex = None):
    '''
    :param smpl: SMPL in the output pose
    :param garment: garment mesh in t-pose
    :param body_src: garment body in t-pose
    :param garment_tex: texture file
    :param vert_inds: vertex association b/w smpl and garment
    :return:
    To use texture files, garments must have vt, ft
    '''
    tgt_params = {'pose': np.array(smpl_tgt.pose.r), 'trans': np.array(smpl_tgt.trans.r), 'betas': np.array(smpl_tgt.betas.r), 'gender': 'neutral'}
    smpl_tgt.pose[:] = 0
    body_tgt = Mesh(smpl_tgt.r, smpl_tgt.f)

    ## Re-target
    ret = retarget(garment, body_src, body_tgt)

    ## Re-pose
    ret_posed = pose_garment(ret, vert_inds, tgt_params)
    body_tgt_posed = pose_garment(body_tgt, list(range(len(body_tgt.v))), tgt_params)

    ## Remove intersections
    ret_posed_interp = remove_interpenetration_fast(ret_posed, body_tgt_posed)
    ret_posed_interp.vt = garment.vt
    ret_posed_interp.ft = garment.ft
    ret_posed_interp.set_texture_image(garment_tex)

    return ret_posed_interp
#===========================================================







#===========================================================
def read_pose(fname):
  # TODO TODO TODO:    TODO TODO TODO:
  thetas_full_3x3 = np.load(fname)
  thetas_72, _    = cv2.rodrigues(thetas_full_3x3) # vectorize this operation / reshape the numpy arrays / reshape the tensorflow tensors / convert tensorflow to numpy 
  return thetas_72
#===========================================================











#===========================================================
if __name__ == '__main__':
    #===========================================================
    path = '/home/nathanbendich/MultiGarmentNetwork/assets/clothes_and_scans/Multi-Garment_dataset/125611512607128/' # mgn-3 GCloud VM, April 20, 2020 -nxb
    dp = SmplPaths()
    vt, ft = dp.get_vt_ft_hres()
    smpl = Smpl(dp.get_hres_smpl_model_data())

    body = load_smpl_from_file(join(path, 'registration.pkl'))
    body.pose[:] = 0

    # angles are in radians:
    body.pose[5] =  0.5 # left  leg
    body.pose[8] = -0.5 # right leg
    body.pose     =read_pose('/home/nathanbendich/MultiGarmentNetwork/assets/MGN_obj_[...insert_date_here...]/thetas.txt')

    # NOTE: you can reshape the body here:
    #body.betas[:] = np.random.randn(10) *0.01   # 10 shape parameters
    body.betas[1] = -2.0
    body.trans[:] = 0
    body = Mesh(body.v, body.f)
    body.write_obj('smpl_{}.obj'.format(datetime.datetime.now().strftime("_%B_%d_%Y____%H:%M_%p__"))  )

    #tex = join(path, 'multi_tex.jpg')

    ## Generate random SMPL body (Feel free to set up ur own smpl) as target subject
    """
    smpl.pose[:] = np.random.randn(72) *0.05
    smpl.betas[:] = np.random.randn(10) *0.01
    smpl.trans[:] = 0
    tgt_body = Mesh(smpl.r, smpl.f)
    """

    #body.set_texture_image(tex)

    """
    mvs = MeshViewers((1, 2))
    mvs[0][0].set_static_meshes([body])
    #mvs[0][1].set_static_meshes([new_garment, tgt_body])
    input('Hit [enter] to exit')
    """

#===========================================================
