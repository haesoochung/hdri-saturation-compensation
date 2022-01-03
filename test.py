from utils.utils import *
import glob
import model

class Test(object):
    def __init__(self, affix, saved_iter, input_path, output_path, model_path):
        self.affix=affix
        self.saved_iter=saved_iter
        self.input_path=input_path
        self.output_path=output_path
        self.model_path=model_path
        self.IMG_HEIGHT = 1000
        self.IMG_WIDTH = 1500
        IMG_DEPTH = 3
        self.input_LDR_low = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.input_LDR_mid = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.input_LDR_high = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.input_HDR_low = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.input_HDR_mid = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.input_HDR_high = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.gt_HDR = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.gt_LDR_low = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.gt_LDR_high = tf.placeholder(tf.float32, shape=[1, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.input_low = tf.concat([self.input_LDR_low, self.input_HDR_low], 3)
        self.input_mid = tf.concat([self.input_LDR_mid, self.input_HDR_mid], 3)
        self.input_high = tf.concat([self.input_LDR_high, self.input_HDR_high], 3)
        self.MODEL = model.Model(self.input_low, self.input_mid, self.input_high, self.gt_HDR, self.gt_LDR_low, self.gt_LDR_high, self.is_train, 'Model_%s' % self.affix)   

    def test(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        final_list = []

        is_training = False

        output=self.MODEL.G

        saver=tf.train.Saver()

        with tf.Session() as sess:
            for iteration in [self.saved_iter] :
                ckpt_path = "%s/iter_%06d" % (self.model_path, iteration)
                if not tf.train.checkpoint_exists(ckpt_path):
                    print('%s does not exist' % ckpt_path)
                    break

                print(Fore.MAGENTA + 'step %d' % iteration)
                nscene = len(os.listdir(self.input_path))
                total_psnrT = 0
                total_psnrL = 0
                total_ssimT = 0
                total_ssimL = 0

                saver.restore(sess, ckpt_path)
                PSNRs = []
                scene_list = sorted(glob.glob("%s/*" % self.input_path))
                for n in range(1, nscene+1):
                    scene_path = scene_list[n-1]
                    image_path = sorted(glob.glob('%s/*.tif' % scene_path))
                    info_path = '%s/input_exp.txt' % scene_path
                    hdr_path = '%s/ref_hdr.hdr' % scene_path

                    gt_HDR_ = cv2.imread(hdr_path,-1).astype(np.float32)
                    gt_HDR_ = gt_HDR_[:,:,::-1]

                    batch_input_LDR_low_tensor, batch_input_LDR_mid_tensor, batch_input_LDR_high_tensor, batch_input_HDR_low_tensor, batch_input_HDR_mid_tensor, batch_input_HDR_high_tensor = make_test_batch(image_path, info_path)
                    
                    tic = time.clock()                     
                    output = sess.run([self.MODEL.G], 
                                        feed_dict={
                                        self.input_LDR_low    : batch_input_LDR_low_tensor,
                                        self.input_LDR_mid    : batch_input_LDR_mid_tensor,
                                        self.input_LDR_high    : batch_input_LDR_high_tensor,
                                        self.input_HDR_low    : batch_input_HDR_low_tensor,
                                        self.input_HDR_mid    : batch_input_HDR_mid_tensor,
                                        self.input_HDR_high    : batch_input_HDR_high_tensor,
                                        self.gt_HDR : batch_input_HDR_high_tensor,
                                        self.is_train         : is_training})
                    toc = time.clock()

                    out_hdr = output[0,:,:,::-1].astype(np.float32)
                    gt_HDR_ = gt_HDR_[:,:,::-1].astype(np.float32)
                    out_hdr = np.clip(out_hdr, 0., 1.)   

                    gt_tm = np.log(1.0+5000.0*gt_HDR_) / np.log(1.0+5000.0)
                    hdr_tm = np.log(1.0+5000.0*out_hdr) / np.log(1.0+5000.0)
                    psnrT, psnrL, ssimT, ssimL = evaluate(out_hdr, gt_HDR_, hdr_tm, gt_tm)

                    total_psnrT += psnrT
                    total_psnrL += psnrL
                    total_ssimT += ssimT
                    total_ssimL += ssimL

                    print('[scene #%i] PSNR_T : %.4f  / %fsec' % (n, psnrT, toc-tic))
                    PSNRs.append(psnrT)

                    hdr_tm_save = 255. * hdr_tm
                    gt_tm_save = 255. * gt_tm
                    
                    cv2.imwrite('%s/%03d_result_%06d.jpg' % (self.output_path, n, iteration), hdr_tm_save)
                    cv2.imwrite('%s/%03d_gt.jpg' % (self.output_path, n), gt_tm_save)
                avg_psnrT = total_psnrT/nscene  
                avg_psnrL = total_psnrL/nscene
                avg_ssimT = total_ssimT/nscene
                avg_ssimL = total_ssimL/nscene
                final_list.append(make_list(iteration, PSNRs, avg_psnrT, avg_psnrL, avg_ssimT, avg_ssimL))
                print('[iter %06d] Average_PSNR_T : %f' % (iteration, avg_psnrT))
                print('[iter %06d] Average_PSNR_L : %f' % (iteration, avg_psnrL))
                print('[iter %06d] Average_SSIM_T : %f' % (iteration, avg_ssimT))
                print('[iter %06d] Average_SSIM_L : %f\n' % (iteration, avg_ssimL))                    
            csv_path = "%s/result_%s.csv" % (self.output_path, self.affix)
            write_csv(csv_path, final_list)