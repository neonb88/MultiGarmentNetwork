# NOTE: this was meant for python2
import sys
import glob
from time import sleep

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config_ver1 import config, NUM, IMG_SIZE

if sys.version_info[0]==2:
  import cPickle as pkl
elif sys.version_info[0]==3:
  import pickle as pkl


def resize(ims, IMG_SIZE=IMG_SIZE):
  '''
    Specialized resize() func just for test_data.pkl from https://github.com/bharat-b7/MultiGarmentNetwork/
  '''
  def resize_single(im, IMG_SIZE=IMG_SIZE):
    # 255 to scale from [0,1] to [0,255].
    SCALE=255
    # Convert to Image just to use the resize() method:
    '''
    print("np.min(im):" )
    print(np.min(im) )
    print("np.max(im):" )
    print(np.max(im) )
    #print(im)
    print(im[ 
      int(round(im.shape[0]/2) )-100 ,      # above (shirt color)
      int(round(im.shape[1]/2) )
    ])
    print(im[ 
      int(round(im.shape[0]/2) ) ,
      int(round(im.shape[1]/2) )
    ])
    print("="*99)
    plt.imshow(im); plt.show()

    print("np.min(im*255):" )
    print(np.min(im*SCALE) )
    print("np.max(im*255):" )
    print(np.max(im*SCALE) )
    print("="*99)
    print(im[ 
      int(round(im.shape[0]/2) )-100 ,      # above (shirt color)
      int(round(im.shape[1]/2) )
    ]*SCALE)
    print(im[ 
      int(round(im.shape[0]/2) ) ,
      int(round(im.shape[1]/2) )
    ]*SCALE)
    print(im.dtype)
    plt.imshow(im* SCALE); plt.show()
    '''

    return np.array(
      Image.fromarray(    
        np.uint8( 
          im* SCALE)
        )  .resize((IMG_SIZE,IMG_SIZE))  # probably 720==IMG_SIZE  May 5, 2020
      ).astype('float32') /SCALE
 #end function def of resize_single(im, IMG_SIZE=IMG_SIZE):========================================

  reshaped0 = resize_single(ims[0] , IMG_SIZE=IMG_SIZE)
  reshaped1 = resize_single(ims[1] , IMG_SIZE=IMG_SIZE)
  # in test_data.pkl, BLB's images are shaped like (2, 720, 720, 3), where height == width ==720, there are 2 images, and 3
  reshaped0 = reshaped0.reshape( (1,)+reshaped0.shape)
  return np.concatenate( 
    (reshaped0, 
     reshaped1.reshape( (1,)+reshaped1.shape)),
    axis=0)





def tell_next_proc_CIHP_is_reshaped(flag_fname='/home/nathanbendich/MultiGarmentNetwork/assets/MGN_reshaping_done.txt'):
  open(flag_fname, 'w').write('')
  return flag_fname

def wait_4_pkl_file(pkl_path='assets/test_data.pkl'):
  i=0
  # Wait for  CIHP_PGN to finish segmenting
  #       and OpenPose to finish pose-detection
  #     (to make test_data.pkl) :
  while not glob.glob(pkl_path):
    sleep(1)
    if i%60 ==0:
      print("{} minutes have passed since we started waiting for the file at {}.".format(i/60, pkl_path) )
    i+=1

#==========================================================================================================
def wait_4_pkl_file_2_fully_write(pkl_path='/home/nathanbendich/MultiGarmentNetwork/assets/test_data.pkl', n_secs=30):
  # NOTE: CIHP_PGN takes much much longer than this, so 30 seconds isn't THAT big a deal.
  print(' waiting for file  {} to be fully written!'.format(pkl_path))
  time.sleep(n_secs) 
#==========================================================================================================


#==========================================================================================================
def main():
  fname='/home/nathanbendich/MultiGarmentNetwork/assets/test_data.pkl'

  # NOTE: CIHP_PGN takes much much longer than this, so 30 seconds isn't THAT big a deal.
  #   nonetheless, we want to refactor everything so we don't have to wait those 30 seconds.
  #   NOTE: I got this "30 seconds" from multiple runs of resaving the original size images in  "test_data.pkl" .  -nxb, Sat May 30 17:02:35 EDT 2020

  #==============================================================================================
  # NOTE: don't change this!    Don't break things!
  # NOTE: don't change this!    Don't break things!
  EMPIRICALLY_DERIVED_PKL_SAVE_TIME=30  # NOTE: don't change this!    Don't break things!
  # NOTE: don't change this!    Don't break things!
  # NOTE: don't change this!    Don't break things!
  #==============================================================================================
  wait_4_pkl_file( pkl_path=fname)
  print(' file  {} arrived!'.format(fname))
  wait_4_pkl_file_2_fully_write(pkl_path=fname, nsecs=EMPIRICALLY_DERIVED_PKL_SAVE_TIME)
  if sys.version_info[0]==2:
    # python2  doesn't require encoding='latin1'
    test_data = pkl.load(    open(fname, 'rb')    )
  elif sys.version_info[0]==3:
    test_data = pkl.load(    open(fname, 'rb')    , encoding='latin1')
  rendered =()

  for i in range(NUM):
    # Because BLB built this to run on two sets of images at once:
    print("resizing image {}".format(i))
    img = resize(
      test_data['image_{}'.format(i)],
      IMG_SIZE=IMG_SIZE)
    #plt.imshow(img[0]);   plt.show()
    #plt.imshow(img[1]);   plt.show()
    test_data['image_{}'.format(i)]=img
    rendered += (   img.reshape( img.shape+(1,)  ),   )
  # TODO: double-check  the     value for key "rendered' (np.concatenate((..., ..., ..., )) )


  test_data['rendered']=np.concatenate( rendered, axis=4)
  out_fname=fname # overwrite old file "test_data.pkl".  May 6, 2020.      #'assets/test_data.pkl_test'
  pkl.dump(  test_data ,   open(out_fname, 'wb') ) # TODO:  make sure this is the right protocol (I think it wawas 2?)
  tell_next_proc_CIHP_is_reshaped('/home/nathanbendich/MultiGarmentNetwork/assets/MGN_reshaping_done.txt')
  print('\n'*3)
  print('done.')
  print('='*99)
  print('\n'*3)
#==========================================================================================================
  



if __name__=="__main__":
  main()









