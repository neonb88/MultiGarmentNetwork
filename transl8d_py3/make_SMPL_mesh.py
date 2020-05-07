'''
This code runs pre-trained MGN.

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
'''
print('='*99)
print('Process started. ')
print('Please wait for pkl file. ')
print('='*99)
print('\n'*9)
print('='*99)

import tensorflow as tf
import numpy as np
import sys
import datetime
import shutil
import os
import glob

from time import sleep

#from cx import backup, clear_dir, 

# TODO:
# loop_cond=test_data.pkl_file_not_present
# while(loop_cond):
#   loop_cond=os.path.exists(test_data.pkl_file)

# main()

#============================================= FIXME: remove ==========================================
if not sys.version_info[0] == 3:
  raise Exception("Please use python3.  Various edits I (nxb)'ve made make it only work with python3.")
#======================================================================================================



# python2:
if sys.version_info[0] == 2:
  import cPickle as pkl          #import cPickle fine in python2; python3 lacks cPickle
# python3:
else:
  import pickle as pkl

from network.base_network import PoseShapeOffsetModel
from config_ver1 import config, NUM, IMG_SIZE, FACE

def pca2offsets(pca_layers, scatter_layers, pca_coeffs, naked_verts, vertexlabel, return_all = False):
    disps = []
    for l, s, p in zip(pca_layers, scatter_layers, pca_coeffs):
        temp = l(p)
        temp = s(temp)
        disps.append(temp)
    temp = tf.stack(disps, axis=-1)
    temp = tf.concat([tf.keras.backend.expand_dims(naked_verts, -1), temp], axis=-1)
    temp2 = tf.transpose(temp, perm=[0, 1, 3, 2])
    temp = tf.batch_gather(temp2, tf.cast(vertexlabel, tf.int32))
    temp = tf.squeeze(tf.transpose(temp, perm=[0, 1, 3, 2]))
    if return_all:
        return temp, temp2
    return temp

def split_garments(pca, mesh, vertex_label, gar):
    '''
      Since garments are layered we do net get high frequency parts for invisible garment vertices.
      Hence we generate the base garment from pca predictions and add the hf term whenever available.
      :param pred_mesh:
      :param garments:
      :return:
    '''
    vertex_label = vertex_label.reshape(-1,)
    base = pca_verts[config.garmentKeys[gar]].inverse_transform(pca).reshape(-1, 3)
    ind = np.where(TEMPLATE[config.garmentKeys[gar]][1])[0]
    gar_mesh = Mesh(mesh.v, mesh.f)

    gar_mesh.v[ind] = base
    gar_mesh.v[vertex_label] = mesh.v[vertex_label]
    gar_mesh.keep_vertices(ind)
    return gar_mesh

def get_results(m, inp, with_pose = False):
    images = [inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
    J_2d = [inp['J_2d_{}'.format(i)].astype('float32') for i in range(NUM)]
    vertex_label = inp['vertexlabel'].astype('int64')

    out = m([images, vertex_label, J_2d])

    with open('assets/hresMapping.pkl', 'rb') as f:
        if sys.version_info[0]  ==  3:
          _, faces = pkl.load(f, encoding='latin1')
        else:
          _, faces = pkl.load(f)

    pca_layers = [l.PCA_ for l in m.garmentModels]
    scatter_layers = m.scatters
    pca_coeffs = np.transpose(out['pca_verts'], [1, 0, 2])
    naked_verts = out['vertices_naked']
    temp = pca2offsets(pca_layers, scatter_layers, pca_coeffs, naked_verts.numpy().astype('float32'), vertex_label)

    pred_mesh = Mesh(out['vertices_tposed'][0].numpy(), faces)
    pred_naked = Mesh(naked_verts[0].numpy(), faces)
    pred_pca = Mesh(temp[0].numpy(), faces)

    gar_meshes= []
    for gar in np.unique(inp['vertexlabel'][0]): #np.where(inp['garments'][0])[0]:
        if gar == 0:
            continue
        gar_meshes.append(split_garments(out['pca_verts'][0][gar-1], pred_mesh, vertex_label[0] == gar, gar -1))

    return {'garment_meshes': gar_meshes, 'body': pred_naked, 'pca_mesh': pred_pca}

def load_model(model_dir):
    m = PoseShapeOffsetModel(config, latent_code_garms_sz=int(config.latent_code_garms_sz / 2))

    # Create the models and optimizers.
    model_objects = {
        'network': m,
        'optimizer': m.optimizer,
        'step': tf.Variable(0),
    }
    latest_cpkt = tf.train.latest_checkpoint(model_dir)
    if latest_cpkt:
        print(('Using latest checkpoint at ' + latest_cpkt))
    else:
        print('No pre-trained model found')
    checkpoint = tf.train.Checkpoint(**model_objects)

    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(latest_cpkt)

    return m

def fine_tune(m, inp, out, display = False):
    ## Need to do a forward pass to get trainable variables
    images = [inp['image_{}'.format(i)].astype('float32') for i in range(NUM)]
    vertex_label = inp['vertexlabel'].astype('int64')
    J_2d = [inp['J_2d_{}'.format(i)].astype('float32') for i in range(NUM)]

    _ = m([images, vertex_label, J_2d])

    ## First optimize pose then other stuff
    vars = []
    losses_2d = {}
    epochs = 50
    vars = ['pose_trans']
    losses_2d['rendered'] = 5 * 10. ** 3
    losses_2d['laplacian'] = 5 * 10 ** 5.
    for i in range(NUM):
        losses_2d['J_2d_{}'.format(i)] = 10**3.
    vars2opt = []
    for v in vars:
        for vv in m.trainable_variables:
            if v in vv.name:
                vars2opt.append(vv.name)

    for ep in range(epochs):
        lo = m.train(inp, out, loss_dict=losses_2d, vars2opt=vars2opt)
        J_2d = 0
        stri = ''
        for k in losses_2d:
            if 'J_2d' in k:
                J_2d += abs(lo[k])
                continue
            stri = stri + k + ' :{}, '.format(lo[k])
        stri = stri + 'J_2d' + ' :{}'.format(J_2d / NUM)
        print(('Ep: {}, {}'.format(ep, stri)))

    vars.extend(['pca_comp', 'betas', 'latent_code_offset_ShapeMerged', 'byPass'])
    losses_2d['laplacian'] = 5* 10 ** 5.
    losses_2d['rendered'] =  5 * 10. ** 5
    for i in range(NUM):
        losses_2d['J_2d_{}'.format(i)] = 10.

    vars2opt = []
    for v in vars:
        for vv in m.trainable_variables:
            if v in vv.name:
                vars2opt.append(vv.name)

    for ep in range(epochs):
        lo = m.train(inp, out, loss_dict=losses_2d, vars2opt=vars2opt)
        J_2d = 0
        stri = ''
        for k in losses_2d:
            if 'J_2d' in k:
                J_2d += abs(lo[k])
                continue
            stri = stri + k + ' :{}, '.format(lo[k]) 
            stri = stri + 'J_2d' + ' :{}'.format(J_2d/NUM)
        print(('Ep: {}, {}'.format(ep, stri)))

    return m

if __name__ == "__main__":
    import os
    from os.path import exists, join, split
    from psbody.mesh import Mesh, MeshViewer, MeshViewers

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  # original.  - nxb, Mon Feb 10 18:58:09 EST 2020
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=conf)

    with open('assets/hresMapping.pkl', 'rb') as f:
        if sys.version_info[0]  ==  3:
          _, faces = pkl.load(f, encoding='latin1')
        else:
          _, faces = pkl.load(f)

    with open('assets/allTemplate_withBoundaries_symm.pkl', 'rb') as f:
        if sys.version_info[0]  ==  3:
          TEMPLATE = pkl.load(f, encoding='latin1')
        else:
          TEMPLATE = pkl.load(f)
    pca_verts = {}
    for garment in config.garmentKeys:
        with open(os.path.join('assets/garment_basis_35_temp20', garment + '_param_{}_corrected.pkl'.format(config.PCA_)), 'rb') as f:
            if sys.version_info[0]  ==  3:
              pca_verts[garment] = pkl.load(f, encoding='latin1')
            else:
              pca_verts[garment] = pkl.load(f)

    model_dir = 'saved_model/'
    ## Load model
    m = load_model(model_dir)

    ## Load test data
    #dat = pkl.load(open('/home/nathanbendich/MultiGarmentNetwork/assets/test_data_nxb_modified__2020_02_19____05:00_AM__.pkl', 'rb'), encoding='latin1')

    pkl_path='assets/test_data.pkl'
    i=0
    # wait for test_data.pkl
    while not glob.glob(pkl_path):
      sleep(1)
      if i%60 ==0:
        print("{} minutes have passed since we started waiting for the file at {}.".format(i/60, pkl_path) )
      i+=1

    dat = pkl.load(open(pkl_path, 'rb'), encoding='latin1')
    # Remove "standard path" pkl file:
    #   (but test_data.pkl is still backed up in assets/MGN_pkl__2020_MM_DD____HH:mm_PM__)
    os.remove(pkl_path)

    ## Get results before optimization
    pred = get_results(m, dat)

  #============================================================================================================
    #   Temporary mesh-saving:   nxb, Mon Feb 24 20:08:17 EST 2020
    timestamp   = datetime.datetime.now().strftime('__%Y_%B_%d____%H:%M_%p__')
    # TODO:  repose / color before saving
    # TODO:  test

    obj_path='./assets/cust.obj'
    pred['body'].write_obj(obj_path) # NOTE: overwrites old cust.obj
    # back up cust.obj w/ timestamp
    dated_obj_dir='/home/nathanbendich/MultiGarmentNetwork/assets/MGN_obj{}/'.format(timestamp)
    os.makedirs(dated_obj_dir)
    shutil.copy2(obj_path, dated_obj_dir + 'cust.obj')
    print('Done')
  #============================================================================================================



























































