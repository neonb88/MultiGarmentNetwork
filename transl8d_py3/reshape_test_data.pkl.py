# NOTE: this was meant for python2
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# TODO: mv to ~/MultiGarmentNetwork/
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








def main():
  #fname='assets/test_data.pkl'
  fname='/home/n/test_data.pkl'# TODO: change
  test_data = pkl.load(    open(fname, 'rb')    ) # python2  doesn't require encoding='latin1'
  rendered =()

  for i in range(NUM):
    # Because BLB built this to run on two sets of images at once:
    img = resize(
      test_data['image_{}'.format(i)],
      IMG_SIZE=IMG_SIZE)
    #plt.imshow(img[0]);   plt.show()
    #plt.imshow(img[1]);   plt.show()
    test_data['image_{}'.format(i)]=img
    rendered += (   img.reshape( img.shape+(1,)  ),   )
  # TODO: double-check  the     value for key "rendered' (np.concatenate((..., ..., ..., )) )


  test_data['rendered']=np.concatenate( rendered, axis=4)
  out_fname='assets/test_data.pkl_test' # TODO: change
  pkl.dump(  test_data ,   open(out_fname, 'wb') )
  



if __name__=="__main__":
  main()
