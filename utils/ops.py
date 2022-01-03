import tensorflow as tf
from utils.custom_vgg16 import *

def content_loss(loss_func, output, gt):

    if loss_func == 'l1' :
        loss = tf.reduce_mean(tf.abs(tf.subtract(output, gt)))

    if loss_func == 'l2' :
        loss = tf.reduce_mean(tf.square(tf.subtract(output, gt)))
        
    elif loss_func == 'vgg16_1-3' :
        data_dict = loadWeightsData('./utils/vgg16.npy')
        vgg_output = custom_Vgg16(output, data_dict=data_dict)
        vgg_gt = custom_Vgg16(gt, data_dict=data_dict)
        # loss = tf.reduce_mean(tf.square(tf.subtract(vgg_output, vgg_gt)))
        loss = tf.reduce_mean(tf.abs((vgg_output.pool1 - vgg_gt.pool1)))
        loss += tf.reduce_mean(tf.abs((vgg_output.pool2 - vgg_gt.pool2)))
        loss += tf.reduce_mean(tf.abs((vgg_output.pool3 - vgg_gt.pool3)))       
        # loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)       

    elif loss_func == 'tv' :
        y_final_gamma_pad_x = tf.pad(output, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
        y_final_gamma_pad_y = tf.pad(output, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
        tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
        tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
        loss = tv_loss_x + tv_loss_y

    elif loss_func == 'cosine' :
        mag = tf.norm(output,2,-1) * tf.norm(gt,2,-1) # B H W
        loss = 1 - tf.reduce_mean(tf.reduce_sum(output*gt, axis=-1) / mag)

    return loss 