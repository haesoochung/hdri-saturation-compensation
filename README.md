# High Dynamic Range Imaging of Dynamic Scenes with Saturation Compensation but without Explicit Motion Compensation 

This repository contains the official Tensorflow implementation of the following paper:

> **High Dynamic Range Imaging of Dynamic Scenes with Saturation Compensation but without Explicit Motion Compensation**<br>
> Haesoo Chung and Nam Ik Cho<br>
> https://arxiv.org/abs
>
> **Abstract:** *High dynamic range (HDR) imaging is a highly challenging task since a large amount of information is lost due to the limitations of camera sensors. For HDR imaging, some methods capture multiple low dynamic range (LDR) images with altering exposures to aggregate more information. However, these approaches introduce ghosting artifacts when significant inter-frame motions are present. Moreover, although multi-exposure images are given, we have little information in severely over-exposed areas. Most existing methods focus on motion compensation, i.e., alignment of multiple LDR shots to reduce the ghosting artifacts, but they still produce unsatisfying results. These methods also rather overlook the need to restore the saturated areas. In this paper, we generate well-aligned multi-exposure features by reformulating a motion alignment problem into a
simple brightness adjustment problem. In addition, we propose a coarse-to-fine merging strategy with explicit saturation compensation. The saturated areas are reconstructed with similar well-exposed content using adaptive contextual attention. We demonstrate that our method outperforms the state-of-the-art methods regarding qualitative and quantitative evaluations.*

## Dependencies
* Python
* Tesorflow 
* OpenCV

## Training
1. Download Kalantari dataset from https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/.
2. Make tfrecords
```
python utils/make_tfrecords.py
```
3. Train
```
python main.py --mode train
```
## Test
```
python main.py --mode test 
```

## Citation
```
@inproceedings{chung2022high,
  title={High Dynamic Range Imaging of Dynamic Scenes With Saturation Compensation but Without Explicit Motion Compensation},
  author={Chung, Haesoo and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2951--2961},
  year={2022}
}
```
