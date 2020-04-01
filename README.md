# MultiGarmentNetwork
Nathan X. Bendich's (neonb88@github.com, nxb) edits to Bharat L. Bhatnagar's
Repo for **"Multi-Garment Net: Learning to Dress 3D People from Images, ICCV'19"**
because I only want to pay for 1 Tesla K80 GPU on GCloud Compute Engine.

Link to paper: https://arxiv.org/abs/1908.06903

## Quickstart (as of February 13, 2020, at 14:47:47 EST) -nxb

###### Start 
-4.  start the instance (`gcloud compute instances start mgn-3` or in the GCloud Compute UI)
#### NOTE: in each step after this point, when I say "do xyz in shell n," I mean "Open a new cloud shell in a new tab from the GCloud menu and then do xyz."  (and by "open a new cloud shell," I mean "copy and paste a url similar to `https://ssh.cloud.google.com/projects/helpful-valve-195602/zones/us-east1-c/instances/mgn-3?authuser=0&hl=en_US&projectNumber=962799681872` into a new browser tab")
###### Start Docker container in shell 1 (AKA OpenPose shell)
-3. `sudo docker start 77f72f9855ad`

###### Upload customer video:
0.  Open shell 2 and `cd /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/UIUX_FrontEnd_nodejs_____interaction/ && ~/UI_launch.sh`
###### Make 8 frames from video:
1.  Open shell 3 and `conda activate mgn1 && python2 ~/w8_4_vid_upload_____then_cut_vid__.py`
2.  Open shell 4 and `conda activate mgn1 && python2 ~/cut_up_sync.py`
###### Rotate each of the 8 images s.t. they're "face up" (OpenPose)
3.  Go to shell 1 and run `sudo docker attach 77f72f9855ad`, then once that loads, `conda activate utils1 && cd /openpose/ && python3 w8_4_orientation_img_upload_____then_run_OPose__.py`
4.  Open shell 5 and run `conda activate mgn1 && cd /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/cut_up_vid_____interaction/ && python2 w8_4_angle_upload_then_rotate_all_frames_.py`
5.  Open shell 6 and run `conda activate mgn1 && python2 ~/sync_gsutil_angle_bucket_____1_hr.py`
6.  Open shell 7 and run `conda activate mgn1 && python2 /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/cut_up_vid_____interaction/sync_gsutil_openpose_4_cropping_bucket_____1_hr.py`


step 6 includes cropping.

###### Segment  (segments each piece of clothing independently):
7.  Open shell 8 and run `conda activate mgn1 && cd /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/CIHP_PGN_____interaction/ && python2 ~/w8_4_img_upload_____then_check_resolution__.py`
  (1st decrease resolution b/c CIHP-PGN won't run on a Tesla K80 GPU at smartphone resolution)
8.  Open shell 9 and run `conda activate mgn1 && cd /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/CIHP_PGN_____interaction/ && python2 w8_4_img_upload_____then_run_PGN__.py`
##### 2-D Pose estimation (OpenPose) (in parallel with the clothing segmentation, not dependent on it)
##### TODO: write code to copy all the images to the OpenPose docker container and have openpose run only when the last image is "`docker cp`"ed
9.  Go back to shell 1 and `cd /openpose && python3 w8_4_img_upload_____then_run_OPose__known_fname.py`
(/root/x/p/vr_mall____fresh___Dec_12_2018/smplx/OpenPose_____interaction/w8_4_img_upload_____then_run_OPose__known_fname.py)
10. Open shell 10 and run `conda activate mgn1 && cd /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/MGN_____interaction/ && python2 prep_MGN_inputs___OpenPose_and___PGN_seg.py
11. Open shell 11 and run `conda activate mgn1 && python2 /home/nathanbendich/x/p/vr_mall____fresh___Dec_12_2018/smplx/MGN_____interaction/sync_gsutil_json_bucket_____1_hr.py`

I think we'll have to write in a bit of code to make "prep_MGN...py" wait for sync_gsutil_openpose_json.py. 

12. Open shell 12 and run `conda activate mgn_py37 && cd /home/nathanbendich/MultiGarmentNetwork/transl8d_py3/ && python3 make_SMPL_mesh.py`
13. [.../di's_part/...]   (TODO: modify.  NOTE: this should be taken care of by the Hilaga et al. Reeb_graph code / whatever Di finds works well)
14. ~/[...]/[...]/render_png.py (TODO)
15.
16.
17.
18.


## Dress SMPL body model with our Digital Wardrobe

1. Download digital wardrobe: https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip
This dataset contains scans, SMPL registration, texture_maps, segmentation_maps and multi-mesh registered garments.
2. visualize_scan.py: Load scan and visualize texture and segmentation
3. visualize_garments.py: Visualize random garment and coresponding SMPL model
4. dress_SMPL.py: Load random garment and dress desired SMPL body with it


## Pre-requisites for running MGN
The code has been tested in python 2.7, Tensorflow 1.13

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the `assets` folder.
```
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download and install DIRT: https://github.com/pmh47/dirt.

Download and install mesh packages for visualization: https://github.com/MPI-IS/mesh

This repo contains code to run pretrained MGN model.
Download saved weights from : https://1drv.ms/u/s!AohQYySSg0mRmju7Of80mQ09wR5-?e=IbbHQ1

## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Crop your images to 720x720. In our testing setup we used roughly centerd subjects at a distance of around 2m from the camer.
2. Run semantic segmentation on images. We used [PGN semantic segmentation](https://github.com/Engineering-Course/CIHP_PGN) and manual correction. Segment garments, Pants (65, 0, 65), Short-Pants (0, 65, 65), Shirt (145, 65, 0), T-Shirt (145, 0, 65) and Coat (0, 145, 65).
3. Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) body_25 for 2D joints.

Semantic segmentation and OpenPose keypoints form the input to MGN. See `assets/test_data.pkl` folder for sample data.

## Texture

The following code may be used to stitch a texture for the reconstruction: https://github.com/thmoa/semantic_human_texture_stitching

Cite us:
```
@inproceedings{bhatnagar2019mgn,
    title = {Multi-Garment Net: Learning to Dress 3D People from Images},
    author = {Bhatnagar, Bharat Lal and Tiwari, Garvita and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
    month = {Oct},
    organization = {{IEEE}},
    year = {2019},
}
```

## License

Copyright (c) 2019 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Multi-Garment Net: Learning to Dress 3D People from Images** paper in documents and papers that report on research using this Software.


### Shoutouts

Chaitanya Patel: code for interpenetration removal, Thiemo Alldieck: code for texture/segmentation
stitching and Verica Lazova: code for data anonymization.



































































































