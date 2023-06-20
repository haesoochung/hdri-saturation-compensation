import numpy as np
import tensorflow as tf
import os
import cv2

import math
# from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
import time
import csv
import re

from colorama import init, Fore
init(autoreset=True)

def make_batch_0(data_path, height, width, batch_size):

  IMG_HEIGHT = height
  IMG_WIDTH = width
  IMG_DEPTH = 3

  filename_queue = tf.train.string_input_producer(data_path)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  feature = {'train/input_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR': tf.FixedLenFeature([], tf.string)}

  features = tf.parse_single_example(serialized_example, features=feature)

  inLl = tf.decode_raw(features['train/input_LDR_low'], tf.uint8)
  inLm = tf.decode_raw(features['train/input_LDR_mid'], tf.uint8)
  inLh = tf.decode_raw(features['train/input_LDR_high'], tf.uint8)
  inHl = tf.decode_raw(features['train/input_HDR_low'], tf.float32)
  inHm = tf.decode_raw(features['train/input_HDR_mid'], tf.float32)
  inHh = tf.decode_raw(features['train/input_HDR_high'], tf.float32)
  gtLl = tf.decode_raw(features['train/gt_LDR_low'], tf.uint8)
  gtLm = tf.decode_raw(features['train/gt_LDR_mid'], tf.uint8)
  gtLh = tf.decode_raw(features['train/gt_LDR_high'], tf.uint8)
  gtHl = tf.decode_raw(features['train/gt_HDR_low'], tf.float32)
  gtHm = tf.decode_raw(features['train/gt_HDR_mid'], tf.float32)
  gtHh = tf.decode_raw(features['train/gt_HDR_high'], tf.float32)
  gt = tf.decode_raw(features['train/gt_HDR'], tf.float32)

  inLl = tf.reshape(inLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLm = tf.reshape(inLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLh = tf.reshape(inLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHl = tf.reshape(inHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHm = tf.reshape(inHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHh = tf.reshape(inHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLl = tf.reshape(gtLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLm = tf.reshape(gtLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLh = tf.reshape(gtLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHl = tf.reshape(gtHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHm = tf.reshape(gtHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHh = tf.reshape(gtHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gt = tf.reshape(gt, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

  inLl = tf.cast(inLl, tf.float32)/255.
  inLm = tf.cast(inLm, tf.float32)/255.
  inLh = tf.cast(inLh, tf.float32)/255.
  inHl = tf.cast(inHl, tf.float32)
  inHm = tf.cast(inHm, tf.float32)
  inHh = tf.cast(inHh, tf.float32)
  gtLl = tf.cast(gtLl, tf.float32)/255.
  gtLm = tf.cast(gtLm, tf.float32)/255.
  gtLh = tf.cast(gtLh, tf.float32)/255.
  gtHl = tf.cast(gtHl, tf.float32)
  gtHm = tf.cast(gtHm, tf.float32)
  gtHh = tf.cast(gtHh, tf.float32)
  gt = tf.cast(gt, tf.float32)
  
  inLls, inLms, inLhs, inHls, inHms, inHhs, gts = tf.train.shuffle_batch([inLl, inLm, inLh, inHl, inHm, inHh, gt], batch_size=batch_size, capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)

  return inLls, inLms, inLhs, inHls, inHms, inHhs, gts

def make_batch_12(data_path, height, width, batch_size):

  IMG_HEIGHT = height
  IMG_WIDTH = width
  IMG_DEPTH = 3

  filename_queue = tf.train.string_input_producer(data_path)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  feature = {'train/input_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR': tf.FixedLenFeature([], tf.string)}

  features = tf.parse_single_example(serialized_example, features=feature)

  inLl = tf.decode_raw(features['train/input_LDR_low'], tf.uint8)
  inLm = tf.decode_raw(features['train/input_LDR_mid'], tf.uint8)
  inLh = tf.decode_raw(features['train/input_LDR_high'], tf.uint8)
  inHl = tf.decode_raw(features['train/input_HDR_low'], tf.float32)
  inHm = tf.decode_raw(features['train/input_HDR_mid'], tf.float32)
  inHh = tf.decode_raw(features['train/input_HDR_high'], tf.float32)
  gtLl = tf.decode_raw(features['train/gt_LDR_low'], tf.uint8)
  gtLm = tf.decode_raw(features['train/gt_LDR_mid'], tf.uint8)
  gtLh = tf.decode_raw(features['train/gt_LDR_high'], tf.uint8)
  gtHl = tf.decode_raw(features['train/gt_HDR_low'], tf.float32)
  gtHm = tf.decode_raw(features['train/gt_HDR_mid'], tf.float32)
  gtHh = tf.decode_raw(features['train/gt_HDR_high'], tf.float32)
  gt = tf.decode_raw(features['train/gt_HDR'], tf.float32)

  inLl = tf.reshape(inLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLm = tf.reshape(inLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLh = tf.reshape(inLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHl = tf.reshape(inHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHm = tf.reshape(inHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHh = tf.reshape(inHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLl = tf.reshape(gtLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLm = tf.reshape(gtLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLh = tf.reshape(gtLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHl = tf.reshape(gtHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHm = tf.reshape(gtHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHh = tf.reshape(gtHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gt = tf.reshape(gt, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

  inLl = tf.cast(inLl, tf.float32)/255.
  inLm = tf.cast(inLm, tf.float32)/255.
  inLh = tf.cast(inLh, tf.float32)/255.
  inHl = tf.cast(inHl, tf.float32)
  inHm = tf.cast(inHm, tf.float32)
  inHh = tf.cast(inHh, tf.float32)
  gtLl = tf.cast(gtLl, tf.float32)/255.
  gtLm = tf.cast(gtLm, tf.float32)/255.
  gtLh = tf.cast(gtLh, tf.float32)/255.
  gtHl = tf.cast(gtHl, tf.float32)
  gtHm = tf.cast(gtHm, tf.float32)
  gtHh = tf.cast(gtHh, tf.float32)
  gt = tf.cast(gt, tf.float32)
  
  inLls, inLms, inLhs, inHls, inHms, inHhs, Dgts = tf.train.shuffle_batch([inLl, inLm, inLh, inHl, inHm, inHh, gt], batch_size=int(batch_size*0.25), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)
  gtLls, gtLms, gtLhs, gtHls, gtHms, gtHhs, Sgts = tf.train.shuffle_batch([gtLl, gtLm, gtLh, gtHl, gtHm, gtHh, gt], batch_size=int(batch_size*0.75), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)

  inLls = tf.concat([gtLls,inLls],0)
  inLms = tf.concat([gtLms,inLms],0)
  inLhs = tf.concat([gtLhs,inLhs],0)
  inHls = tf.concat([gtHls,inHls],0)
  inHms = tf.concat([gtHms,inHms],0)
  inHhs = tf.concat([gtHhs,inHhs],0)
  gts = tf.concat([Sgts,Dgts],0)

  return inLls, inLms, inLhs, inHls, inHms, inHhs, gts

def make_batch_8(data_path, height, width, batch_size):

  IMG_HEIGHT = height
  IMG_WIDTH = width
  IMG_DEPTH = 3

  filename_queue = tf.train.string_input_producer(data_path)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  feature = {'train/input_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR': tf.FixedLenFeature([], tf.string)}

  features = tf.parse_single_example(serialized_example, features=feature)

  inLl = tf.decode_raw(features['train/input_LDR_low'], tf.uint8)
  inLm = tf.decode_raw(features['train/input_LDR_mid'], tf.uint8)
  inLh = tf.decode_raw(features['train/input_LDR_high'], tf.uint8)
  inHl = tf.decode_raw(features['train/input_HDR_low'], tf.float32)
  inHm = tf.decode_raw(features['train/input_HDR_mid'], tf.float32)
  inHh = tf.decode_raw(features['train/input_HDR_high'], tf.float32)
  gtLl = tf.decode_raw(features['train/gt_LDR_low'], tf.uint8)
  gtLm = tf.decode_raw(features['train/gt_LDR_mid'], tf.uint8)
  gtLh = tf.decode_raw(features['train/gt_LDR_high'], tf.uint8)
  gtHl = tf.decode_raw(features['train/gt_HDR_low'], tf.float32)
  gtHm = tf.decode_raw(features['train/gt_HDR_mid'], tf.float32)
  gtHh = tf.decode_raw(features['train/gt_HDR_high'], tf.float32)
  gt = tf.decode_raw(features['train/gt_HDR'], tf.float32)

  inLl = tf.reshape(inLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLm = tf.reshape(inLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLh = tf.reshape(inLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHl = tf.reshape(inHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHm = tf.reshape(inHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHh = tf.reshape(inHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLl = tf.reshape(gtLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLm = tf.reshape(gtLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLh = tf.reshape(gtLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHl = tf.reshape(gtHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHm = tf.reshape(gtHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHh = tf.reshape(gtHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gt = tf.reshape(gt, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

  inLl = tf.cast(inLl, tf.float32)/255.
  inLm = tf.cast(inLm, tf.float32)/255.
  inLh = tf.cast(inLh, tf.float32)/255.
  inHl = tf.cast(inHl, tf.float32)
  inHm = tf.cast(inHm, tf.float32)
  inHh = tf.cast(inHh, tf.float32)
  gtLl = tf.cast(gtLl, tf.float32)/255.
  gtLm = tf.cast(gtLm, tf.float32)/255.
  gtLh = tf.cast(gtLh, tf.float32)/255.
  gtHl = tf.cast(gtHl, tf.float32)
  gtHm = tf.cast(gtHm, tf.float32)
  gtHh = tf.cast(gtHh, tf.float32)
  gt = tf.cast(gt, tf.float32)

  inLls, inLms, inLhs, inHls, inHms, inHhs, Dgts = tf.train.shuffle_batch([inLl, inLm, inLh, inHl, inHm, inHh, gt], batch_size=int(batch_size*0.5), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)
  gtLls, gtLms, gtLhs, gtHls, gtHms, gtHhs, Sgts = tf.train.shuffle_batch([gtLl, gtLm, gtLh, gtHl, gtHm, gtHh, gt], batch_size=int(batch_size*0.5), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)
  
  inLls = tf.concat([gtLls,inLls],0)
  inLms = tf.concat([gtLms,inLms],0)
  inLhs = tf.concat([gtLhs,inLhs],0)
  inHls = tf.concat([gtHls,inHls],0)
  inHms = tf.concat([gtHms,inHms],0)
  inHhs = tf.concat([gtHhs,inHhs],0)
  gts = tf.concat([Sgts,Dgts],0)

  return inLls, inLms, inLhs, inHls, inHms, inHhs, gts

def make_batch_4(data_path, height, width, batch_size):

  IMG_HEIGHT = height
  IMG_WIDTH = width
  IMG_DEPTH = 3

  filename_queue = tf.train.string_input_producer(data_path)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  feature = {'train/input_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/input_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_LDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_low': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_mid': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR_high': tf.FixedLenFeature([], tf.string),
           'train/gt_HDR': tf.FixedLenFeature([], tf.string)}

  features = tf.parse_single_example(serialized_example, features=feature)

  inLl = tf.decode_raw(features['train/input_LDR_low'], tf.uint8)
  inLm = tf.decode_raw(features['train/input_LDR_mid'], tf.uint8)
  inLh = tf.decode_raw(features['train/input_LDR_high'], tf.uint8)
  inHl = tf.decode_raw(features['train/input_HDR_low'], tf.float32)
  inHm = tf.decode_raw(features['train/input_HDR_mid'], tf.float32)
  inHh = tf.decode_raw(features['train/input_HDR_high'], tf.float32)
  gtLl = tf.decode_raw(features['train/gt_LDR_low'], tf.uint8)
  gtLm = tf.decode_raw(features['train/gt_LDR_mid'], tf.uint8)
  gtLh = tf.decode_raw(features['train/gt_LDR_high'], tf.uint8)
  gtHl = tf.decode_raw(features['train/gt_HDR_low'], tf.float32)
  gtHm = tf.decode_raw(features['train/gt_HDR_mid'], tf.float32)
  gtHh = tf.decode_raw(features['train/gt_HDR_high'], tf.float32)
  gt = tf.decode_raw(features['train/gt_HDR'], tf.float32)

  inLl = tf.reshape(inLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLm = tf.reshape(inLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inLh = tf.reshape(inLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHl = tf.reshape(inHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHm = tf.reshape(inHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  inHh = tf.reshape(inHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLl = tf.reshape(gtLl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLm = tf.reshape(gtLm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtLh = tf.reshape(gtLh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHl = tf.reshape(gtHl, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHm = tf.reshape(gtHm, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gtHh = tf.reshape(gtHh, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
  gt = tf.reshape(gt, [IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

  inLl = tf.cast(inLl, tf.float32)/255.
  inLm = tf.cast(inLm, tf.float32)/255.
  inLh = tf.cast(inLh, tf.float32)/255.
  inHl = tf.cast(inHl, tf.float32)
  inHm = tf.cast(inHm, tf.float32)
  inHh = tf.cast(inHh, tf.float32)
  gtLl = tf.cast(gtLl, tf.float32)/255.
  gtLm = tf.cast(gtLm, tf.float32)/255.
  gtLh = tf.cast(gtLh, tf.float32)/255.
  gtHl = tf.cast(gtHl, tf.float32)
  gtHm = tf.cast(gtHm, tf.float32)
  gtHh = tf.cast(gtHh, tf.float32)
  gt = tf.cast(gt, tf.float32)
  
  inLls, inLms, inLhs, inHls, inHms, inHhs, Dgts = tf.train.shuffle_batch([inLl, inLm, inLh, inHl, inHm, inHh, gt], batch_size=int(batch_size*0.75), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)
  gtLls, gtLms, gtLhs, gtHls, gtHms, gtHhs, Sgts = tf.train.shuffle_batch([gtLl, gtLm, gtLh, gtHl, gtHm, gtHh, gt], batch_size=int(batch_size*0.25), capacity=1000+3*batch_size, num_threads=1, min_after_dequeue=1000)

  inLls = tf.concat([gtLls,inLls],0)
  inLms = tf.concat([gtLms,inLms],0)
  inLhs = tf.concat([gtLhs,inLhs],0)
  inHls = tf.concat([gtHls,inHls],0)
  inHms = tf.concat([gtHms,inHms],0)
  inHhs = tf.concat([gtHhs,inHhs],0)
  gts = tf.concat([Sgts,Dgts],0)

  return inLls, inLms, inLhs, inHls, inHms, inHhs, gts

def calc_param(scope=None):
    N = np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print(Fore.YELLOW + '# Model Params: %d K' % (N/1000))

def load_ckpt(saver, sess, checkpoint_dir, saved_iter):
    # print('==================== Loading Checkpoints ====================')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if saved_iter != 0:
        saved_ckpt = checkpoint_dir + '/iter_%06d' % saved_iter
        saver.restore(sess, saved_ckpt)
        print(Fore.MAGENTA + '=================== iter_{} is loaded ===================='.format(saved_iter))        
        return True, int(saved_iter)
    elif ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(Fore.MAGENTA + "=================== {} is loaded ====================".format(ckpt_name))
        return True, step
    else:
        print(Fore.MAGENTA + "=================== No checkpoint ====================")
        return False, 0

def save_ckpt(saver, sess, checkpoint_dir, step):
    saver.save(sess, '%s/iter_%06d' % (checkpoint_dir, step))
    print(Fore.MAGENTA + "=================== iter_{} is saved ====================".format(step))

def make_test_batch(image_paths, info_path):

  IMG_HEIGHT = 1000
  IMG_WIDTH = 1500
  IMG_DEPTH = 3

  gamma = 2.2

  input_LDR_low_ = cv2.imread(image_paths[0]).astype(np.float32)/255.
  input_LDR_mid_ = cv2.imread(image_paths[2]).astype(np.float32)/255.
  input_LDR_high_ = cv2.imread(image_paths[4]).astype(np.float32)/255.

  input_LDR_low_ = input_LDR_low_[:,:,::-1]
  input_LDR_mid_ = input_LDR_mid_[:,:,::-1]
  input_LDR_high_ = input_LDR_high_[:,:,::-1]

  height, width, channel = input_LDR_mid_.shape

  expo = np.zeros(3)
  file = open(info_path, 'r')
  expo[0]= np.power(2.0, float(file.readline()))
  expo[1]= np.power(2.0, float(file.readline()))
  expo[2]= np.power(2.0, float(file.readline()))
  file.close()

  input_HDR_low_ = np.power(input_LDR_low_, gamma)/expo[0]
  input_HDR_mid_ = np.power(input_LDR_mid_, gamma)/expo[1]
  input_HDR_high_ = np.power(input_LDR_high_, gamma)/expo[2]

  input_LDR_low_ = (255.*input_LDR_low_).astype(np.uint8)
  input_LDR_mid_ = (255.*input_LDR_mid_).astype(np.uint8)
  input_LDR_high_ = (255.*input_LDR_high_).astype(np.uint8)

  batch_input_LDR_low = []
  batch_input_LDR_mid = []
  batch_input_LDR_high = []
  batch_input_HDR_low = []
  batch_input_HDR_mid = []
  batch_input_HDR_high = []

  input_LDR_low_ = input_LDR_low_.astype(np.float32)/255.
  input_LDR_mid_ = input_LDR_mid_.astype(np.float32)/255.
  input_LDR_high_ = input_LDR_high_.astype(np.float32)/255.

  batch_input_LDR_low.append(input_LDR_low_)
  batch_input_LDR_mid.append(input_LDR_mid_)
  batch_input_LDR_high.append(input_LDR_high_)
  batch_input_HDR_low.append(input_HDR_low_)
  batch_input_HDR_mid.append(input_HDR_mid_)
  batch_input_HDR_high.append(input_HDR_high_)

  batch_input_LDR_low_tensor = np.stack(batch_input_LDR_low, axis=0)
  batch_input_LDR_mid_tensor = np.stack(batch_input_LDR_mid, axis=0)
  batch_input_LDR_high_tensor = np.stack(batch_input_LDR_high, axis=0)
  batch_input_HDR_low_tensor = np.stack(batch_input_HDR_low, axis=0)
  batch_input_HDR_mid_tensor = np.stack(batch_input_HDR_mid, axis=0)
  batch_input_HDR_high_tensor = np.stack(batch_input_HDR_high, axis=0)

  return batch_input_LDR_low_tensor, batch_input_LDR_mid_tensor, batch_input_LDR_high_tensor, batch_input_HDR_low_tensor, batch_input_HDR_mid_tensor, batch_input_HDR_high_tensor

def evaluate(hdr, gt, hdr_tm, gt_tm):
    mseT = np.mean( (hdr_tm - gt_tm) ** 2 )
    if mseT == 0:
        return 100
    PIXEL_MAX = 1.0 
    psnrT = 20 * math.log10(PIXEL_MAX / math.sqrt(mseT))

    mseL = np.mean((gt-hdr)**2)
    psnrL = -10.*math.log10(mseL)

    ssimT = ssim(gt_tm, hdr_tm, multichannel=True, data_range=gt_tm.max() - gt_tm.min())
    ssimL = ssim(gt, hdr, multichannel=True, data_range=gt.max() - gt.min())

    return psnrT, psnrL, ssimT, ssimL

def make_list(iteration, PSNRs, avg_psnrT, avg_psnrL, avg_ssimT, avg_ssimL):
  list = []
  list.append(iteration)
  list.extend(PSNRs)
  list.extend([avg_psnrT, avg_psnrL, avg_ssimT, avg_ssimL])
  list = tuple(list)
  return list

def write_csv(csv_path, final_list):
  if os.path.isfile(csv_path) == False:
    col_0 = []
    col_0.append('iter')
    col_0.extend(range(1,16))
    col_0.extend(['psnrT','psnrL','ssimT','ssimL'])
    col_0 = tuple(col_0)
    final_list.insert(0, col_0)
  zipped = list(zip(*final_list))  
  with open(csv_path,'a',newline='') as f:
      wr = csv.writer(f)
      for row in zipped:
        wr.writerow([*row])

def tonemap(img):
  return tf.log(1 + 5000.*(img)) / tf.log(1+5000.)

def ud(x, h, w, c, N, K):
  return tf.reshape(tf.reverse(tf.reverse(x, [1]), [5]), [-1, h, w, c, N, K*K])

def lr(x, h, w, c, N, K):
  return tf.reshape(tf.reverse(tf.reverse(x, [2]), [6]), [-1, h, w, c, N, K*K])

def tr(x, h, w, c, N, K):
  return tf.reshape(tf.transpose(x, perm=[0,2,1,3,4,6,5]), [-1, h, w, c, N, K*K])

def udtr(x, h, w, c, N, K):
  y = tf.reverse(tf.reverse(x, [1]), [5])
  return tf.reshape(tf.transpose(y, perm=[0,2,1,3,4,6,5]), [-1, h, w, c, N, K*K])

def lrtr(x, h, w, c, N, K):
  y = tf.reverse(tf.reverse(x, [2]), [6])
  return tf.reshape(tf.transpose(y, perm=[0,2,1,3,4,6,5]), [-1, h, w, c, N, K*K]) 

def udlr(x, h, w, c, N, K):
  y = tf.reverse(tf.reverse(x, [1]), [5])
  return tf.reshape(tf.reverse(tf.reverse(y, [2]), [6]), [-1, h, w, c, N, K*K])

def udlrtr(x, h, w, c, N, K):
  y = tf.reverse(tf.reverse(x, [1]), [5])
  y = tf.reverse(tf.reverse(y, [2]), [6])
  return tf.reshape(tf.transpose(y, perm=[0,2,1,3,4,6,5]), [-1, h, w, c, N, K*K]) 
