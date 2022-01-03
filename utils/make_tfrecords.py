import numpy as np
import os
import tensorflow as tf
import cv2
from random import shuffle
import glob
from scipy import ndimage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PATCH_SIZE = 128
IMG_DEPTH = 3
STRIDE = 64
gamma = 2.2

is_shuffle = True
files_per_tfrecord = 1000

scene_list = sorted(glob.glob("/dataset/train/*"))
tfrecord_path = "./tfrecords"
if not os.path.exists(tfrecord_path):
    os.makedirs(tfrecord_path)

nscene = 74
nrow = 12
ncol = 20
nrot = 4
nflip = 2
npatch = nscene*nrow*ncol*nrot*nflip
print ('Train data: %d patches' % (npatch))

n = np.arange(npatch)
np.random.shuffle(n)

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

i = 0
j = 0
while i < npatch :
	n_tfrecord = i // files_per_tfrecord
	train_filename = "%s/train_%03d.tfrecords" % (tfrecord_path, n_tfrecord)
	 
	if not i % files_per_tfrecord:
		print ('Train data: %d/%d. %s is generated!' % (i, len(n), train_filename))

	with tf.python_io.TFRecordWriter(train_filename) as writer :
		j = 0
		while i < npatch and j < files_per_tfrecord :
			s = n[i]//(nrow*ncol*nrot*nflip)
			h = (n[i]-nrow*ncol*nrot*nflip*s)//(ncol*nrot*nflip)
			w = (n[i]-nrow*ncol*nrot*nflip*s-ncol*nrot*nflip*h)//(nrot*nflip)
			r = (n[i]-nrow*ncol*nrot*nflip*s-ncol*nrot*nflip*h-nrot*nflip*w)//nflip
			f = n[i]-nrow*ncol*nrot*nflip*s-ncol*nrot*nflip*h-nrot*nflip*w-nflip*r
			
			scene_path = scene_list[s]
			image_pathss = '%s/*.tif' % scene_path
			image_paths = sorted(glob.glob(image_pathss))
			info_path = '%s/input_exp.txt' % scene_path
			hdr_path = '%s/ref_hdr.hdr' % scene_path

			input_LDR_low = cv2.imread(image_paths[0]).astype(np.float32)/255.
			input_LDR_mid = cv2.imread(image_paths[2]).astype(np.float32)/255.
			input_LDR_high = cv2.imread(image_paths[4]).astype(np.float32)/255.
			gt_HDR = cv2.imread(hdr_path,-1).astype(np.float32)

			height, width, channel = input_LDR_low.shape

			expo = np.zeros(3)
			file = open(info_path, 'r')
			expo[0]= np.power(2.0, float(file.readline()))
			expo[1]= np.power(2.0, float(file.readline()))
			expo[2]= np.power(2.0, float(file.readline()))
			file.close()
			input_LDR_low_patch = input_LDR_low[h*STRIDE:h*STRIDE+PATCH_SIZE, w*STRIDE:w*STRIDE+PATCH_SIZE,::-1]
			input_LDR_mid_patch = input_LDR_mid[h*STRIDE:h*STRIDE+PATCH_SIZE, w*STRIDE:w*STRIDE+PATCH_SIZE,::-1]
			input_LDR_high_patch = input_LDR_high[h*STRIDE:h*STRIDE+PATCH_SIZE, w*STRIDE:w*STRIDE+PATCH_SIZE,::-1]
			gt_HDR_patch = gt_HDR[h*STRIDE:h*STRIDE+PATCH_SIZE, w*STRIDE:w*STRIDE+PATCH_SIZE,::-1]
			
			input_HDR_low_patch = np.power(input_LDR_low_patch, gamma)/expo[0]
			input_HDR_mid_patch = np.power(input_LDR_mid_patch, gamma)/expo[1]
			input_HDR_high_patch = np.power(input_LDR_high_patch, gamma)/expo[2]

			gt_LDR_low_patch = np.clip(gt_HDR_patch*expo[0], 0.0, 1.0)
			gt_LDR_mid_patch = np.clip(gt_HDR_patch*expo[1], 0.0, 1.0)
			gt_LDR_high_patch = np.clip(gt_HDR_patch*expo[2], 0.0, 1.0)
			gt_LDR_low_patch = np.power(gt_LDR_low_patch, 1./gamma)
			gt_LDR_mid_patch = np.power(gt_LDR_mid_patch, 1./gamma)
			gt_LDR_high_patch = np.power(gt_LDR_high_patch, 1./gamma)
	
			gt_HDR_low_patch = np.power(gt_LDR_low_patch, gamma)/expo[0]
			gt_HDR_mid_patch = np.power(gt_LDR_mid_patch, gamma)/expo[1]
			gt_HDR_high_patch = np.power(gt_LDR_high_patch, gamma)/expo[2]

			input_LDR_low_patch = (255.*input_LDR_low_patch).astype(np.uint8)
			input_LDR_mid_patch = (255.*input_LDR_mid_patch).astype(np.uint8)
			input_LDR_high_patch = (255.*input_LDR_high_patch).astype(np.uint8)
			gt_LDR_low_patch = (255.*gt_LDR_low_patch).astype(np.uint8)
			gt_LDR_mid_patch = (255.*gt_LDR_mid_patch).astype(np.uint8)
			gt_LDR_high_patch= (255.*gt_LDR_high_patch).astype(np.uint8)

			input_LDR_low_patch = cv2.resize(input_LDR_low_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			input_LDR_mid_patch = cv2.resize(input_LDR_mid_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			input_LDR_high_patch = cv2.resize(input_LDR_high_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			input_HDR_low_patch = cv2.resize(input_HDR_low_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			input_HDR_mid_patch = cv2.resize(input_HDR_mid_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			input_HDR_high_patch = cv2.resize(input_HDR_high_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_LDR_low_patch = cv2.resize(gt_LDR_low_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_LDR_mid_patch = cv2.resize(gt_LDR_mid_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_LDR_high_patch = cv2.resize(gt_LDR_high_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_HDR_low_patch = cv2.resize(gt_HDR_low_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_HDR_mid_patch = cv2.resize(gt_HDR_mid_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_HDR_high_patch = cv2.resize(gt_HDR_high_patch, (PATCH_SIZE-3,PATCH_SIZE-3))
			gt_HDR_patch = cv2.resize(gt_HDR_patch, (PATCH_SIZE-3,PATCH_SIZE-3))

			input_LDR_low_patch = cv2.resize(input_LDR_low_patch, (PATCH_SIZE,PATCH_SIZE))
			input_LDR_mid_patch = cv2.resize(input_LDR_mid_patch, (PATCH_SIZE,PATCH_SIZE))
			input_LDR_high_patch = cv2.resize(input_LDR_high_patch, (PATCH_SIZE,PATCH_SIZE))
			input_HDR_low_patch = cv2.resize(input_HDR_low_patch, (PATCH_SIZE,PATCH_SIZE))
			input_HDR_mid_patch = cv2.resize(input_HDR_mid_patch, (PATCH_SIZE,PATCH_SIZE))
			input_HDR_high_patch = cv2.resize(input_HDR_high_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_LDR_low_patch = cv2.resize(gt_LDR_low_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_LDR_mid_patch = cv2.resize(gt_LDR_mid_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_LDR_high_patch = cv2.resize(gt_LDR_high_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_HDR_low_patch = cv2.resize(gt_HDR_low_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_HDR_mid_patch = cv2.resize(gt_HDR_mid_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_HDR_high_patch = cv2.resize(gt_HDR_high_patch, (PATCH_SIZE,PATCH_SIZE))
			gt_HDR_patch = cv2.resize(gt_HDR_patch, (PATCH_SIZE,PATCH_SIZE))

			theta = np.random.uniform(-10,10)
			gt_LDR_low_patch = ndimage.rotate(gt_LDR_low_patch, theta, reshape=False,mode='mirror')
			gt_HDR_low_patch = ndimage.rotate(gt_HDR_low_patch, theta, reshape=False,mode='mirror')
			theta = np.random.uniform(-10,10)
			gt_LDR_high_patch = ndimage.rotate(gt_LDR_high_patch, theta, reshape=False,mode='mirror')
			gt_HDR_high_patch = ndimage.rotate(gt_HDR_high_patch, theta, reshape=False,mode='mirror')

			if r == 1 :
				input_LDR_low_patch = cv2.rotate(input_LDR_low_patch, cv2.ROTATE_90_CLOCKWISE)
				input_LDR_mid_patch = cv2.rotate(input_LDR_mid_patch, cv2.ROTATE_90_CLOCKWISE)
				input_LDR_high_patch = cv2.rotate(input_LDR_high_patch, cv2.ROTATE_90_CLOCKWISE)
				input_HDR_low_patch = cv2.rotate(input_HDR_low_patch, cv2.ROTATE_90_CLOCKWISE)
				input_HDR_mid_patch = cv2.rotate(input_HDR_mid_patch, cv2.ROTATE_90_CLOCKWISE)
				input_HDR_high_patch = cv2.rotate(input_HDR_high_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_LDR_low_patch = cv2.rotate(gt_LDR_low_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_LDR_mid_patch = cv2.rotate(gt_LDR_mid_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_LDR_high_patch = cv2.rotate(gt_LDR_high_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_HDR_low_patch = cv2.rotate(gt_HDR_low_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_HDR_mid_patch = cv2.rotate(gt_HDR_mid_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_HDR_high_patch = cv2.rotate(gt_HDR_high_patch, cv2.ROTATE_90_CLOCKWISE)
				gt_HDR_patch = cv2.rotate(gt_HDR_patch, cv2.ROTATE_90_CLOCKWISE)
			elif r == 2 :				
				input_LDR_low_patch = cv2.rotate(input_LDR_low_patch, cv2.ROTATE_180)
				input_LDR_mid_patch = cv2.rotate(input_LDR_mid_patch, cv2.ROTATE_180)
				input_LDR_high_patch = cv2.rotate(input_LDR_high_patch, cv2.ROTATE_180)
				input_HDR_low_patch = cv2.rotate(input_HDR_low_patch, cv2.ROTATE_180)
				input_HDR_mid_patch = cv2.rotate(input_HDR_mid_patch, cv2.ROTATE_180)
				input_HDR_high_patch = cv2.rotate(input_HDR_high_patch, cv2.ROTATE_180)
				gt_LDR_low_patch = cv2.rotate(gt_LDR_low_patch, cv2.ROTATE_180)
				gt_LDR_mid_patch = cv2.rotate(gt_LDR_mid_patch, cv2.ROTATE_180)
				gt_LDR_high_patch = cv2.rotate(gt_LDR_high_patch, cv2.ROTATE_180)
				gt_HDR_low_patch = cv2.rotate(gt_HDR_low_patch, cv2.ROTATE_180)
				gt_HDR_mid_patch = cv2.rotate(gt_HDR_mid_patch, cv2.ROTATE_180)
				gt_HDR_high_patch = cv2.rotate(gt_HDR_high_patch, cv2.ROTATE_180)
				gt_HDR_patch = cv2.rotate(gt_HDR_patch, cv2.ROTATE_180)
			elif r == 3 :				
				input_LDR_low_patch = cv2.rotate(input_LDR_low_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				input_LDR_mid_patch = cv2.rotate(input_LDR_mid_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				input_LDR_high_patch = cv2.rotate(input_LDR_high_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				input_HDR_low_patch = cv2.rotate(input_HDR_low_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				input_HDR_mid_patch = cv2.rotate(input_HDR_mid_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				input_HDR_high_patch = cv2.rotate(input_HDR_high_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_LDR_low_patch = cv2.rotate(gt_LDR_low_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_LDR_mid_patch = cv2.rotate(gt_LDR_mid_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_LDR_high_patch = cv2.rotate(gt_LDR_high_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_HDR_low_patch = cv2.rotate(gt_HDR_low_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_HDR_mid_patch = cv2.rotate(gt_HDR_mid_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_HDR_high_patch = cv2.rotate(gt_HDR_high_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
				gt_HDR_patch = cv2.rotate(gt_HDR_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

			if f > 0 :
				input_LDR_low_patch = input_LDR_low_patch[:,::-1,:]
				input_LDR_mid_patch = input_LDR_mid_patch[:,::-1,:]
				input_LDR_high_patch = input_LDR_high_patch[:,::-1,:]
				input_HDR_low_patch = input_HDR_low_patch[:,::-1,:]
				input_HDR_mid_patch = input_HDR_mid_patch[:,::-1,:]
				input_HDR_high_patch = input_HDR_high_patch[:,::-1,:]
				gt_LDR_low_patch = gt_LDR_low_patch[:,::-1,:]
				gt_LDR_mid_patch = gt_LDR_mid_patch[:,::-1,:]
				gt_LDR_high_patch = gt_LDR_high_patch[:,::-1,:]
				gt_HDR_low_patch = gt_HDR_low_patch[:,::-1,:]
				gt_HDR_mid_patch = gt_HDR_mid_patch[:,::-1,:]
				gt_HDR_high_patch = gt_HDR_high_patch[:,::-1,:]
				gt_HDR_patch = gt_HDR_patch[:,::-1,:]

			feature = {'train/input_LDR_low'	: _bytes_feature(input_LDR_low_patch.tostring()),
								'train/input_LDR_mid'	: _bytes_feature(input_LDR_mid_patch.tostring()),
								'train/input_LDR_high'	: _bytes_feature(input_LDR_high_patch.tostring()),
								'train/input_HDR_low'	: _bytes_feature(input_HDR_low_patch.tostring()),
								'train/input_HDR_mid'	: _bytes_feature(input_HDR_mid_patch.tostring()),
								'train/input_HDR_high'	: _bytes_feature(input_HDR_high_patch.tostring()),
	    					'train/gt_LDR_low'		: _bytes_feature(gt_LDR_low_patch.tostring()),
	    					'train/gt_LDR_mid'		: _bytes_feature(gt_LDR_mid_patch.tostring()),
	    					'train/gt_LDR_high'		: _bytes_feature(gt_LDR_high_patch.tostring()),
	    					'train/gt_HDR_low'		: _bytes_feature(gt_HDR_low_patch.tostring()),
	    					'train/gt_HDR_mid'		: _bytes_feature(gt_HDR_mid_patch.tostring()),
	    					'train/gt_HDR_high'		: _bytes_feature(gt_HDR_high_patch.tostring()),
	    					'train/gt_HDR'			: _bytes_feature(gt_HDR_patch.tostring())}

			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

			print("%d / %d patches are converted!" % (i, npatch))

			i += 1
			j += 1
