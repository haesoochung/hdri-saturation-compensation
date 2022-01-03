import tensorflow as tf
import numpy as np

def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                            fuse_k=3, softmax_scale=10., training=True, fuse=True, soft_thr=0.):
    """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    # print(raw_w)
    # print(raw_int_bs)
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    # raw_w = tf.reshape(raw_w, [-1, raw_int_bs[1]*raw_int_bs[2], kernel, kernel, raw_int_bs[3]]) ############# b hw k k c
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = upsample(f, 'f', to_shape=[int(np.ceil(raw_int_fs[1]/rate)), int(np.ceil(raw_int_fs[2]/rate))])  #########################  
    # f = upsample(f, 'f', factor=[1./rate,1./rate])  
    # b = self.upsample(f, 'b', factor=[1./rate,1./rate])  
    b = upsample(b, 'b', to_shape=[int(np.ceil(raw_int_bs[1]/rate)), int(np.ceil(raw_int_bs[2]/rate))])  
    # print(f.get_shape().as_list(),b.get_shape().as_list())
    if mask is not None:
        mask = upsample(mask, 'mask', factor=[1./rate,1./rate]) 
        mask = upsample(mask, 'mask',  to_shape=[int(np.ceil(raw_int_bs[1]/rate)), int(np.ceil(raw_int_bs[2]/rate))])  
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([int_bs[0], bs[1], bs[2], 1])
    
    #########
    int_ms = mask.get_shape().as_list()
    # print(int_ms)
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [int_ms[0], -1, ksize, ksize, int_ms[3]]) ################
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # print(m.get_shape)
    if soft_thr == 0.:
        mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[1,2,3], keep_dims=True), 0.), tf.float32) # hw / valid if mean=0
    else:
        # mm = tf.cast(tf.less(tf.reduce_max(m, axis=[1,2,3], keep_dims=True), soft_thr), tf.float32) # hw
        # mm = tf.cast(tf.less(tf.reduce_mean(m, axis=[1,2,3], keep_dims=True), 1-soft_thr), tf.float32) # mean
        temp = tf.reduce_mean(m, axis=[1,2,3], keep_dims=True) # b*1*1*1*hw
        mm = tf.ones_like(temp) - temp

    mm_groups = tf.split(mm, int_ms[0], axis=0)

    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1]) #b h/2 w/2 c   b*k*k*c*hw/4  b*k*k*c*hw  h/2*w/2
    for xi, wi, raw_wi, mm_ in zip(f_groups, w_groups, raw_w_groups, mm_groups):
        # conv for compare
        wi = wi[0]      #BG k*k*c*hw/4
        mm = mm_[0]     #mask h/2*w/2

        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4) #k*k*c*hw/4
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME") #h/2 w/2 c
        # correlation
        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])
        # print(f.get_shape())
        # print(w.get_shape())
        # print(b.get_shape())
        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3) ######attention
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        # print(raw_w.get_shape())
        # print(raw_wi.get_shape())

        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow =  upsample(flow, 'flow', factor=[rate,rate])
    return y, flow

def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))

def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        # img = img / 127.5 - 1.
        img = img / 255.
        return img

def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 255.
        # img = img / 127.5 - 1.
        return img

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

COLORWHEEL = make_color_wheel()

def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img

def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

def upsample(x, name, factor=[2,2], to_shape=None, align_corners=True):
    with tf.name_scope(name):
        if to_shape is None:
            xs = x.get_shape().as_list()
            size = [int(xs[1]*factor[0]), int(xs[2]*factor[1])]
            # size = [(tf.shape(x)[1] * factor[0]), (tf.shape(x)[2] * factor[1])]
            x = tf.image.resize_nearest_neighbor(x, size=size, align_corners=align_corners, name=None)
            # x =  tf.image.resize_images(x, size=size, method=tf.image.ResizeMethod.BILINEAR)

        else:
            print(to_shape)
            # x =  tf.image.resize_images(x, size=[to_shape[0], to_shape[1]], method=tf.image.ResizeMethod.BILINEAR)
            x = tf.image.resize_nearest_neighbor(x, size=[to_shape[0], to_shape[1]], align_corners=align_corners, name=None)
    return x

def interpolation(frame, offset, block_ch):
    """
    Interpolating the frames by bilinear, return the sampled pixels
    :param frames: shape [b, h, w, c], in which, b->batch, N->channel, c->color channel
    :param offset: shape [b, h, w, N*2], N is the number of sampled pixels
    :return: shape [b, h, w, c, N]
    """    
    b = tf.shape(frame)[0]
    h = tf.shape(frame)[1]
    w = tf.shape(frame)[2]
    c = tf.shape(frame)[3]
    N = tf.shape(offset)[3] // 2

    # offset = tf.stack([offset, offset, offset], axis=3) # b h w c N*2        
    off_stack = []
    for i in range(block_ch):
        off_stack.append(offset) 
    offset = tf.stack(off_stack, axis=3)       
    output = tf.zeros([b, h, w, c, N])


    w_pos, h_pos = tf.meshgrid(tf.range(0, w, 1.), tf.range(0, h, 1.)) #int32
    w_pos = tf.broadcast_to(tf.reshape(w_pos, [1,h,w,1,1]), [b, h, w, c, N])
    h_pos = tf.broadcast_to(tf.reshape(h_pos, [1,h,w,1,1]), [b, h, w, c, N])
    # w_pos, h_pos = tf.meshgrid(tf.range(0, w), tf.range(0, h)) #int32
    # w_pos = tf.cast(tf.broadcast_to(tf.reshape(w_pos, [1,h,w,1,1]), [b, h, w, c, N]),tf.float32)
    # h_pos = tf.cast(tf.broadcast_to(tf.reshape(h_pos, [1,h,w,1,1]), [b, h, w, c, N]),tf.float32)
    # N_pos

    # the wanted positions of sampled pixels
    ib = tf.reshape(tf.range(0,b,1.), [b,1,1,1,1])
    ic = tf.reshape(tf.range(0,c,1.), [1,1,1,c,1])

    ib = tf.broadcast_to(ib, [b, h, w, c, N])
    ic = tf.broadcast_to(ic, [b, h, w, c, N])

    # find the corners of a cubic
    # interpolation
    floor, ceil = tf.floor, lambda x: tf.floor(x)+1
    f_set = (
            (floor, floor),
            (floor, ceil),
            (ceil, floor),
            (ceil, ceil)
    )
    # x= tf.zeros([b, h, w, c, N], dtype=tf.int32)
    h = tf.cast(h,tf.float32)
    w = tf.cast(w,tf.float32)
    for fh, fw in f_set:
        # f_N_pos = fN(offset[:, 0::3, ...])
        f_h_pos = fh(offset[:, :, :, :, 0::2]) # b h w c N
        f_w_pos = fw(offset[:, :, :, :, 1::2])
        # x += tf.cast(f_h_pos, tf.int32) + h_pos
        output += _select_by_index(frame, ib, f_h_pos + h_pos, f_w_pos + w_pos, ic, h, w) \
                    * (1. - tf.abs(f_h_pos - offset[:, :, :, :, 0::2])) \
                    * (1. - tf.abs(f_w_pos - offset[:, :, :, :, 1::2]))   
    return output

def _select_by_index(tensor, ib, ih, iw, ic, h, w):
    """
    :param tensor: [b, h, w, c]
    :param iN: [b, h, w, c, N]
    :param ih: [b, h, w, c, N]
    :param iw: [b, h, w, c, N]
    :return: [b, h, w, c, N]
    """
    # h = tf.shape(tensor)[1]
    # w = tf.shape(tensor)[2]
    # if the position is outside the tensor, make the mask and set them to zero
    cond = tf.cast(tf.greater_equal(ih,0.), tf.float32) * tf.cast(tf.less(ih,h), tf.float32) * tf.cast(tf.greater_equal(iw,0.), tf.float32) * tf.cast(tf.less(iw,w), tf.float32)
    # cond = tf.logical_and(tf.logical_and(tf.logical_and(tf.greater_equal(ih,0.),tf.less(ih,h)),tf.greater_equal(iw,0.)),tf.less(iw,w))
    # cond = tf.cast(cond,tf.float32)
    ih = ih * cond
    iw = iw * cond
    ib = tf.cast(ib,tf.int32)
    ih = tf.cast(ih,tf.int32)
    iw = tf.cast(iw,tf.int32)
    ic = tf.cast(ic,tf.int32)
    # ih[mask_outside] = 0
    # iw[mask_outside] = 0
    indices = tf.stack([ib, ih, iw, ic], axis=-1)
    res = tf.gather_nd(tensor, indices) *  tf.cast(cond, tf.float32)
    # res = tensor[ib, ih, iw, ic]
    # res[mask_outside] = 0
    return res

def res_block(bottom, ch, is_train, scope, name):
    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None)
    # y = conv_layer(bottom, scope, name=name+"_conv1", kshape=[3,3,256,256], strides=[1,1,1,1])
    print(y)
    
    y = batch_normalization(y, is_train, scope, name=name+"_bn1")
    y = tf.nn.relu(y)
    print(y)

    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None)
    # y = conv_layer(y, scope, name=name+"_conv2", kshape=[3,3,256,256], strides=[1,1,1,1])
    y = batch_normalization(y, is_train, scope, name=name+"_bn2")
    print(y)

    y += bottom
    # y = tf.nn.relu(y)
    return y

def dilated_res_block(bottom, ch, is_train, scope, name):
    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None, dilation_rate=(2, 2))
    # y = conv_layer(bottom, scope, name=name+"_conv1", kshape=[3,3,256,256], strides=[1,1,1,1])
    print(y)
    
    y = batch_normalization(y, is_train, scope, name=name+"_bn1")
    y = tf.nn.relu(y)
    print(y)

    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None, dilation_rate=(2, 2))
    # y = conv_layer(y, scope, name=name+"_conv2", kshape=[3,3,256,256], strides=[1,1,1,1])
    y = batch_normalization(y, is_train, scope, name=name+"_bn2")
    print(y)

    y += bottom
    # y = tf.nn.relu(y)
    return y
    
def edsr_res_block(bottom, is_train, scope, name, ch=64):
    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None)
    y = tf.nn.relu(y)
    y = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=None)
    y = tf.add(y*0.1, bottom)
    # y = tf.nn.relu(y)
    return y

def avg_pool(bottom, scope, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope+"_"+name)

def batch_normalization(bottom, is_train, scope, name) : 
    return tf.contrib.layers.batch_norm(bottom, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_train, scope=scope+"_"+name)

def leaky_relu_1(x):
    return tf.nn.leaky_relu(x, alpha=0.01)

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

def conv_layer(bottom, scope, name, ch):
    for n in range(3):
        with tf.variable_scope(scope+"_"+name+"_%d" % n):
            bottom = tf.layers.conv2d(bottom, ch, 3, padding='same', activation=tf.nn.relu)
    return bottom