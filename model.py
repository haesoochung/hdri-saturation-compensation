from utils.layer_utils import *
from utils.utils import tonemap
from colorama import init, Fore
init(autoreset=True)

class Model(object):    
    def __init__(self, input_low, input_mid, input_high, gt_HDR, gt_LDR_low, gt_LDR_high, is_train, name):
        
        self.input_low = input_low
        self.input_mid = input_mid
        self.input_high = input_high
        self.gt_HDR = gt_HDR
        self.gt_LDR_low = gt_LDR_low
        self.gt_LDR_high = gt_LDR_high
        self.is_train = is_train

        self.name = name
        self.kernel_size=3
        self.num_of_inputs=3
        self.kernel_dim=3

        self.block_ch = 16

        self.build_model()

    def build_model(self):
        print('Building {}'.format(self.name))

        print(Fore.BLUE+'*************************************************************************************************')
        self.G = self.Generator('G')
        print(Fore.BLUE+'*************************************************************************************************')

    def Generator(self, scope):      
      with tf.variable_scope(scope) :
        self.aligned_low = self.Align(self.input_low, self.input_mid, self.is_train, 'Align_low')
        self.aligned_mid = self.Align(self.input_mid, self.input_mid, self.is_train, 'Align_mid')
        self.aligned_high = self.Align(self.input_high, self.input_mid, self.is_train, 'Align_high')
        self.output = self.Merge(self.aligned_low, self.aligned_mid, self.aligned_high, self.is_train, 'Merge')
        return self.output

    def Align(self, input_to_align, ref, is_train, scope):
        with tf.variable_scope(scope) :
            scope = "ref"
            ch = self.block_ch            

            ref_0 = self.conv_layer(ref, scope, name="layer0_conv1", kshape=[3,3,6,ch], strides=[1,1,1,1])

            ref_1 = self.conv_layer(ref_0, scope, name="layer1_conv1", kshape=[3,3,ch,ch], strides=[1,2,2,1])
            ref_1 = self.resden_block(ref_1, ch, is_train, scope, "resden1")
            
            ref_2 = self.conv_layer(ref_1, scope, name="layer2_conv1", kshape=[3,3,ch,ch], strides=[1,2,2,1])
            ref_2 = self.resden_block(ref_2, ch, is_train, scope, "resden2")

            scope = "sup"
       
            sup_0 = self.conv_layer(input_to_align, scope, name="layer0_conv1", kshape=[3,3,6,ch], strides=[1,1,1,1])

            sup_1 = self.conv_layer(sup_0, scope, name="layer1_conv1", kshape=[3,3,ch,ch], strides=[1,2,2,1])
            sup_1 = self.resden_block(sup_1, ch, is_train, scope, "resden1")

            sup_2 = tf.concat([ref_1, sup_1], 3)
            sup_2 = self.conv_layer(sup_2, scope, name="layer2_conv1", kshape=[1,1,ch*2,ch], strides=[1,1,1,1])
            self.feat_2 = self.lrelu(self.conv_layer(sup_2, scope, name="feat_2", kshape=[3,3,ch,ch], strides=[1,1,1,1]), scope, "feat2_lrelu")
            sup_2 = self.conv_layer(sup_2, scope, name="layer2_conv2", kshape=[3,3,ch,ch], strides=[1,2,2,1])
            sup_2 = self.resden_block(sup_2, ch, is_train, scope, "resden2")
            
            sup_3 = tf.concat([ref_2, sup_2], 3)
            sup_3 = self.conv_layer(sup_3, scope, name="layer3_conv1", kshape=[1,1,ch*2,ch], strides=[1,1,1,1])
            sup_3 = self.conv_layer(sup_3, scope, name="layer3_conv2", kshape=[3,3,ch,ch], strides=[1,1,1,1])
            self.feat = self.lrelu(sup_3, scope, "layer3_lrelu")

            h=tf.shape(self.feat)[1]
            w=tf.shape(self.feat)[2]
            final_ch = ch*self.kernel_size*self.kernel_size
            k_conv1 = tf.layers.conv2d(self.feat, final_ch, 3, padding='same', activation=tf.nn.relu)
            kernels = tf.reshape(k_conv1, [-1, h, w, ch, self.kernel_size*self.kernel_size]) 

            o_conv1 = tf.layers.conv2d(self.feat, ch, 3, padding='same', activation=tf.nn.relu)
            offsets = tf.layers.conv2d(o_conv1, 2*self.kernel_size*self.kernel_size, 3, padding='same', activation=None) 

            def_conv_1 = tf.reduce_sum(interpolation(ref_2, offsets, ch) * kernels, axis=-1) 
            feat_cat_1 = tf.concat([def_conv_1,ref_2],-1)
            feat_cat_1_conv = self.conv_layer(feat_cat_1, scope, name="feat_cat_1_conv", kshape=[3,3,ch*2,ch], strides=[1,1,1,1])
            feat_cat_1_deconv = tf.layers.conv2d_transpose(feat_cat_1_conv, ch, 3, strides=(2,2), padding='same', activation=None)
            
            self.feat_2 = tf.concat([self.feat_2,feat_cat_1_deconv],-1)
            k_conv2 = tf.layers.conv2d(self.feat_2, final_ch, 3, padding='same', activation=tf.nn.relu)
            kernels_x2 = tf.reshape(k_conv2, [-1, 2*h, 2*w, ch, self.kernel_size*self.kernel_size])  
            o_conv2 = tf.layers.conv2d(self.feat_2, ch, 3, padding='same', activation=tf.nn.relu)
            offsets_x2 = tf.layers.conv2d(o_conv2, 2*self.kernel_size*self.kernel_size, 3, padding='same', activation=None) 

            def_conv_2 = tf.reduce_sum(interpolation(ref_1, offsets_x2, ch) * kernels_x2, axis=-1) 
            feat_cat_2 = tf.concat([feat_cat_1_deconv, def_conv_2,ref_1],-1)
            feat_cat_2_conv = self.conv_layer(feat_cat_2, scope, name="feat_cat_2_conv", kshape=[3,3,ch*3,ch], strides=[1,1,1,1])
            bottom = tf.layers.conv2d_transpose(feat_cat_2_conv, ch, 3, strides=(2,2), padding='same', activation=None)

        return bottom

    def gen_conv(self, x, cnum, ksize, stride=1, rate=1, name='conv', padding='SAME', activation=tf.nn.elu, training=True):
        x = tf.layers.conv2d(x, cnum, ksize, stride, dilation_rate=rate, activation=None, padding=padding, name=name)
        if cnum == 3 or activation is None:        
            return x
        return activation(x)
    
    def gen_deconv(self, x, cnum, name='upsample', padding='SAME', training=True):
        x = self.upsample(x, name, factor=[2,2])
        x = self.gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding, training=training)
        return x
 
    def Merge(self, input_low, input_mid, input_high, is_train, scope):
        with tf.variable_scope(scope) :
            cnum = 48   
            ch=cnum
            cat = tf.concat([input_low,input_mid,input_high],3) #64*3
            
            inter = tf.layers.conv2d(cat, ch, 3, padding='same', activation=tf.nn.relu)
            for i in range(5) :
                inter = self.res_block(inter, ch, is_train, scope, "resb%d"%(i))
            conv = self.conv_layer(inter, scope, name="conv4", kshape=[3,3,ch,3], strides=[1,1,1,1])
            self.output_coarse = tf.nn.sigmoid(conv)      
            self.mask = self.conv_layer(inter, scope, name="mask_conv", kshape=[3,3,ch,1], strides=[1,1,1,1])
            self.mask = 1/(1+tf.exp(-3*self.mask))

            self.input_4 = tf.concat([self.output_coarse,self.mask],-1)

            x = self.gen_conv(self.input_4, cnum, 5, 1, name='xconv1')
            x = self.gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = self.gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = self.gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = self.gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = self.gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = self.gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = self.gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = self.gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = self.gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x

            x = self.gen_conv(self.input_4, cnum, 5, 1, name='pmconv1')
            x = self.gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = self.gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = self.gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = self.gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = self.gen_conv(x, 4*cnum, 3, 1, name='pmconv6', activation=tf.nn.relu)
            mask_s = self.upsample(self.mask, 'mask_s', to_shape=x.get_shape().as_list()[1:3])   
            x, self.flow = contextual_attention(x, x, mask=mask_s, ksize=3, stride=1, rate=2, soft_thr=0.5)
            x = self.gen_conv(x, 4*cnum, 3, 1, name='pmconv9') 
            x = self.gen_conv(x, 4*cnum, 3, 1, name='pmconv10') 
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = self.gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = self.gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = self.gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = self.gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = self.gen_deconv(x, cnum, name='allconv15_upsample')
            x = self.gen_conv(x, cnum//2, 3, 1, name='allconv16')
            x = self.gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            self.output_fine = tf.nn.sigmoid(x)
            output = self.output_fine*self.mask + self.output_coarse*(1.-self.mask)
            
        return output

    def resden_block(self, bottom, bsize, is_train, scope, name):
        y_0 = bottom

        y_1 = y_0 # 64
        y_1 = self.conv_layer(y_0, scope, name=name+"_conv1", kshape=[3,3,bsize+bsize*0,bsize], strides=[1,1,1,1])
        y_1 = self.lrelu(y_1, scope, "conv1_lrelu1")

        y_2 = tf.concat([y_0, y_1],3) #64+32*1
        y_2 = self.conv_layer(y_2, scope, name=name+"_conv2", kshape=[3,3,bsize+bsize*1,bsize], strides=[1,1,1,1])
        y_2 = self.lrelu(y_2, scope, "conv2_lrelu1")

        y_7 = tf.concat([y_0, y_1, y_2],3) #64+32*6
        y_7 = self.conv_layer(y_7, scope, name=name+"_conv7", kshape=[1,1,bsize+bsize*2,bsize], strides=[1,1,1,1])

        return y_0 + y_7*0.1

    def res_3d_block(self, bottom, kdepth, bsize, is_train, scope, name):
        y = self.conv_3d_layer(bottom, scope, name=name+"_conv1", kshape=[kdepth,3,3,bsize,bsize])
        y = self.batch_normalization(y, is_train, scope, name=name+"_batn1")
        y = tf.nn.relu(y)
        return bottom + y

    def res_block(self, bottom, bsize, is_train, scope, name):
        y = self.conv_layer(bottom, scope, name=name+"_conv1", kshape=[3,3,bsize,bsize], strides=[1,1,1,1])
        y = tf.nn.relu(y)
        y = self.conv_layer(y, scope, name=name+"_conv2", kshape=[3,3,bsize,bsize], strides=[1,1,1,1])
        return bottom + y

    def avg_pool(self, bottom, scope, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+"_"+name)

    def max_pool(self, bottom, scope, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+"_"+name)

    def upsample(self, x, name, factor=[2,2], to_shape=None, align_corners=True):
        with tf.name_scope(name):
            if to_shape is None:
                xs = x.get_shape().as_list()
                size = [int(xs[1]*factor[0]), int(xs[2]*factor[1])]
                x = tf.image.resize_nearest_neighbor(x, size=size, align_corners=align_corners, name=None)
            else:
                x = tf.image.resize_nearest_neighbor(x, size=[to_shape[0], to_shape[1]], align_corners=align_corners, name=None)
        return x

    def lrelu(self, bottom, scope, name, leak=0.2):
        with tf.variable_scope(scope+"_"+name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * bottom + f2 * abs(bottom)

    def conv_layer(self, bottom, scope, name, kshape, strides=[1, 1, 1, 1], is_train=True):
        with tf.variable_scope(scope+"_"+name):
            W = tf.get_variable(name=scope+"_"+name+"_weights",
                                shape=kshape,
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable(name=scope+"_"+name+"_biases",
                                shape=[kshape[3]],
                                initializer=tf.constant_initializer(0.0))
            out = tf.nn.conv2d(bottom,W,strides=strides, padding='SAME')
            out = tf.nn.bias_add(out, b)

            return out
