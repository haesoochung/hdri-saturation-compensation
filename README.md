# High Dynamic Range Imaging of Dynamic Scenes with Saturation Compensation but without Explicit Motion Compensation 

Official Tensorflow implementation of our paper: High Dynamic Range Imaging of Dynamic Scenes with Satu
ration Compensation but without Explicit Motion Compensation (WACV 2022)

## Dependencies
* Python
* Tesorflow 
* OpenCV

## Training
1. Download Kalantari dataset 
```
cd dataset
sh download_dataset.sh
cd ..
```
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
