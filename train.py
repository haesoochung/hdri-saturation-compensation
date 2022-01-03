from utils.utils import *
from utils.ops import *

import model
from math import e
from colorama import init, Fore
init(autoreset=True)

class Train(object):
    def __init__(self, affix, saved_iter, size, batch_size, learning_rate, max_epoch, tfrecord_path, checkpoint_dir, step_disp, w_vgg, w_tv, w_cos):
        print('[*] Initialize Training')
        self.affix=affix
        self.saved_iter=saved_iter
        self.HEIGHT=size[0]
        self.WIDTH=size[1]
        self.CHANNEL=size[2]
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.max_epoch=max_epoch
        self.tfrecord_path=tfrecord_path
        self.checkpoint_dir=checkpoint_dir
        self.num_of_data=num_of_data
        self.step_disp=step_disp
        self.w_vgg=w_vgg
        self.w_tv=w_tv
        self.w_cos=w_cos

        self.input_LDR_low = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.input_LDR_mid = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.input_LDR_high = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.input_HDR_low = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.input_HDR_mid = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.input_HDR_high = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.gt_HDR = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.gt_LDR_low = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.gt_LDR_mid = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.gt_LDR_high = tf.placeholder(tf.float32, shape=[self.batch_size, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.input_low = tf.concat([self.input_LDR_low, self.input_HDR_low], 3)
        self.input_mid = tf.concat([self.input_LDR_mid, self.input_HDR_mid], 3)
        self.input_high = tf.concat([self.input_LDR_high, self.input_HDR_high], 3)

        self.MODEL = model.Model(self.input_low, self.input_mid, self.input_high, self.gt_HDR, self.gt_LDR_low, self.gt_LDR_high, self.is_train, 'Model_%s' % self.affix)   
                                                
    def define_loss(self):      
        self.output_tm = tonemap(self.MODEL.G)
        self.gt_HDR_tm = tonemap(self.gt_HDR)

        self.output_fine_tm = tonemap(self.MODEL.output_fine)
        self.output_coarse_tm = tonemap(self.MODEL.output_coarse)
        self.tm_l2_loss = content_loss('l1', self.output_coarse_tm, self.gt_HDR_tm) \
                            + content_loss('l1', self.output_fine_tm, self.gt_HDR_tm) \
                            + content_loss('l1', self.output_tm, self.gt_HDR_tm)
        self.vgg_loss = content_loss('vgg16_1-3', self.output_coarse_tm, self.gt_HDR_tm)\
                        + content_loss('vgg16_1-3', self.output_fine_tm, self.gt_HDR_tm)
        self.tv_loss = content_loss('tv', self.output_coarse_tm, self.gt_HDR_tm)\
                     + content_loss('tv', self.output_fine_tm, self.gt_HDR_tm) 
        self.cosine_loss = content_loss('cosine', self.output_coarse_tm, self.gt_HDR_tm)\
                        + content_loss('cosine', self.output_fine_tm, self.gt_HDR_tm)             

        self.g_loss = self.tm_l2_loss + self.w_vgg*self.vgg_loss + self.w_tv*self.tv_loss + self.w_cos*self.cosine_loss

    def train(self):
        is_training = True

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
        self.global_step=tf.Variable(self.saved_iter, name='global_step', trainable=False)
        self.learning_rate=tf.maximum(self.learning_rate, 1e-5)
        self.define_loss()
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)

        self.saver=tf.train.Saver(max_to_keep=100000)

        calc_param(scope='G') 
       
        # Change this part according to the training stage
        inLls, inLms, inLhs, inHls, inHms, inHhs, gts = make_batch_4(self.tfrecord_path, self.HEIGHT, self.WIDTH, self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ckpt_exists, saved_step = load_ckpt(self.saver, sess, self.checkpoint_dir, self.saved_iter)
            
            if ckpt_exists:
                self.saved_iter = saved_step
                sess.run(self.global_step.assign(self.saved_iter))
                print('Current iteration:', self.saved_iter)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            step = self.saved_iter
            num_of_batch = self.num_of_data // self.batch_size
            curr_epoch = (step*self.batch_size) // self.num_of_data
            
            epoch = curr_epoch

            t2 = time.time()
            while True:
                try:
                    bt_inLl, bt_inLm, bt_inLh, bt_inHl, bt_inHm, bt_inHh, bt_gt = sess.run([inLls, inLms, inLhs, inHls, inHms, inHhs, gts])

                    sess.run(self.g_optimizer, feed_dict={
                                                               self.input_LDR_low    : bt_inLl,
                                                               self.input_LDR_mid    : bt_inLm,
                                                               self.input_LDR_high   : bt_inLh,
                                                               self.input_HDR_low    : bt_inHl,
                                                               self.input_HDR_mid    : bt_inHm,
                                                               self.input_HDR_high   : bt_inHh,
                                                               self.gt_HDR           : bt_gt,
                                                               self.is_train         : is_training}) 
                    step += 1

                    if step % self.step_disp == 0:
                        G, g_loss, tm_l2_loss, lr = sess.run([self.MODEL.G, self.g_loss, self.tm_l2_loss, self.learning_rate] \
                                                                    , feed_dict={
                                                                       self.input_LDR_low    : bt_inLl,
                                                                       self.input_LDR_mid    : bt_inLm,
                                                                       self.input_LDR_high   : bt_inLh,
                                                                       self.input_HDR_low    : bt_inHl,
                                                                       self.input_HDR_mid    : bt_inHm,
                                                                       self.input_HDR_high   : bt_inHh,
                                                                       self.gt_HDR           : bt_gt,
                                                                       self.is_train         : is_training}) 
                        print('epoch {0:2d} / lr {1:.1e}'.format(epoch, lr))      
                        print('[step {0:6d}]             g_loss : {1:.7f}'.format(step, g_loss))     

                    if step % num_of_batch == 0:
                        print(Fore.YELLOW + 'Epoch %d Done' % epoch)
                        epoch += 1
                        save_ckpt(self.saver, sess, self.checkpoint_dir, step)

                        if epoch == self.max_epoch:
                            break

                        print(Fore.YELLOW + 'Epoch %d starts / Total iteration %d' % (epoch, step))

                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print(Fore.YELLOW + 'Epoch %d / Iteration %d' % (epoch, step))
                    save_ckpt(self.saver, sess, self.checkpoint_dir, step)
                    break

            coord.request_stop()
            coord.join(threads)
            sess.close()