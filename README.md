# MultiGarmentNetwork
Nathan X. Bendich's (neonb88@github.com, nxb) edits to Bharat L. Bhatnagar's
Repo for **"Multi-Garment Net: Learning to Dress 3D People from Images, ICCV'19"**
because I only want to pay for 1 Tesla K80 GPU on GCloud Compute Engine.

Link to paper: https://arxiv.org/abs/1908.06903

## Quickstart (as of February 13, 2020, at 14:47:47 EST) -nxb

###### Start 
0.  start the instance (`gcloud compute instances start mgn-3` or in the GCloud Compute UI)
#### NOTE: in each step after this point, when I say "do xyz in shell n," I mean "Open a new cloud shell in a new tab from the GCloud menu and then do xyz."  (and by "open a new cloud shell," I mean "copy and paste a url similar to `https://ssh.cloud.google.com/projects/helpful-valve-195602/zones/us-east1-c/instances/mgn-3?authuser=0&hl=en_US&projectNumber=962799681872` into a new browser tab")
###### Start Docker containers in shell 1
1.  Clean out files from the previous customer's video and start dockers:
```
./clean.sh &&\
s &&\
sudo docker start 5b1c347bf448
``` 

###### Upload customer video:
2.  Open shell 2 and 
```
cd x/p/vr_mall____fresh___Dec_12_2018/smplx/UIUX_FrontEnd_nodejs_____interaction/ &&\
./launch.sh
```
###### Make 8 frames from video:
3.  Open shell 3 and 
```
2 &&\
p2 w8_4_vid_upload_____then_cut_vid__.py
```
4.  Open shell 4 and
```
2 &&\
p2 cut_up_sync.py
```
###### Rotate each of the 8 images s.t. they're "face up" (OpenPose)
5.  Go to shell 1 and run 
```
a
```
then, once that loads,
```
3 &&\
p3 w8_4_orientation_img_upload_____then_run_OPose__.py
```
6.  Open shell 5 and run 
```
2 &&\
c x/p/vr_mall____fresh___Dec_12_2018/smplx/cut_up_vid_____interaction/ &&\
p2 w8_4_angle_upload_then_rotate_all_frames_.py       # includes cropping.  should rename
```
step 6 includes cropping.

7.  Open shell 6 and run 
```
2 &&\
p2 ~/sync_gsutil_angle_bucket_____1_hr.py
```
8.  Open shell 7 and run 
```
2 &&\
p2 x/p/vr_mall____fresh___Dec_12_2018/smplx/cut_up_vid_____interaction/sync_gsutil_openpose_4_cropping_bucket_____1_hr.py
```

###### Segment  (segments each piece of clothing independently):
9.  Open shell 8 and run 
```
2 &&\
c x/p/vr_mall____fresh___Dec_12_2018/smplx/CIHP_PGN_____interaction/ &&\
p2 w8_4_img_upload_____then_check_resolution__.py
```
We decreased resolution before running CIHP-PGN b/c a Tesla K80 GPU is too shit for high-res NN calculations

10.  Open shell 9 and run 
```
2 &&\
c CIHP_PGN &&\
p2 w8_4_img_upload_____then_run_PGN__.py
```
##### 2-D Pose estimation (OpenPose) (in parallel with the clothing segmentation, not dependent on it)
11.  Open shell 13 and 
```
sudo docker attach 5b1c347bf448
```
then, once that has loaded
```
3 &&\
c /openpose &&\
python3 w8_4_img_upload_____then_run_OPose__known_fname.py      
# full script path is 
#   /root/x/p/vr_mall____fresh___Dec_12_2018/smplx/OpenPose_____interaction/w8_4_img_upload_____then_run_OPose__known_fname.py 
```

12. Open shell 10 and run 
```
2 &&\
c x/p/vr_mall____fresh___Dec_12_2018/smplx/MGN_____interaction/ &&\
p2 prep_MGN_inputs___OpenPose_and___PGN_seg.py
```
13. Open shell 11 and run 
```
2 &&\
p2 x/p/vr_mall____fresh___Dec_12_2018/smplx/MGN_____interaction/sync_gsutil_json_bucket_____1_hr.py
```
14. Open shell 12 and run 
```
3 &&\
c MultiGarmentNetwork/transl8d_py3/ &&\
p3 make_SMPL_mesh.py
```
15. [.../di's_part/...]   (TODO: modify.  NOTE: this should be taken care of by the Hilaga et al. Reeb_graph code / whatever Di finds works well)
16. ~/[...]/[...]/render_png.py (TODO)
17. TURN OFF THE VM: 
```
gcloud compute instances stop mgn-3
```
Otherwise, Nathan will be billed.
18.   Shopping
19.
20.


## File locs:

If I've listed a directory, please empty it completely with `~/clean.sh`.  Otherwise, only delete the file I listed: please delete none of the others in that directory.

VM "mgn-3" :
1.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/last_custs_angle.txt
2.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/videos/
3.  /home/nathanbendich/intermediate_steps_ClothX/images/
4.  /home/nathanbendich/intermediate_steps_ClothX/edges/
5.  /home/nathanbendich/intermediate_steps_ClothX/labels/
6.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/high_res.py    (the contents of this file "high_res.py" should be "high_res=True")
7.  /home/nathanbendich/intermediate_steps_ClothX/OpenPose_json_4_cropping/
8.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/images/
9.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/labels/
10.  /home/nathanbendich/CIHP_PGN/datasets/CIHP/edges/
11.  /home/nathanbendich/CIHP_PGN/output/cihp_parsing_maps/
12.  /home/nathanbendich/CIHP_PGN/output/cihp_edge_maps/
13.  /home/nathanbendich/intermediate_steps_ClothX/OpenPose_json/
14.  /home/nathanbendich/MultiGarmentNetwork/assets/test_data.pkl
15.  /home/nathanbendich/intermediate_steps_ClothX/cihp_parsing_maps/
16.  /home/nathanbendich/MultiGarmentNetwork/assets/cust.obj
17.
18.

You can `ls` all these at once by doing `~/ls_all.sh`

OpenPose Docker container:
1.  /root/orientation_imgs/
2.  /root/cust_imgs
3.  /openpose/angles/last_custs_angle.txt
4.  /openpose/output/keypoints_json/
5.  

GCloud storage buckets:
1. gs://cust_vids/
2. gs://openpose_json2/
3. gs://openpose_json_4_cropping/
4. gs://vid_angles/

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











































