import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import tensorflow as tf
import argparse
from glob import glob
import train
import test

parser = argparse.ArgumentParser(description='**HDRI**')
parser.add_argument('--mode', dest='mode', default='train', type=str)
parser.add_argument('--gpu', dest='gpu', default='1', type=str) 
parser.add_argument('--affix', dest='affix', default='test', type=str)
parser.add_argument('--saved_iter', type=int, dest='saved_iter', default=0)
parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
parser.add_argument('--kernel_size', dest='kernel_size', default=5, type=int)
parser.add_argument('--lr', dest='lr', default=1e-4, type=float)

parser.add_argument('--step_disp', dest='step_disp', default=5000, type=int)
parser.add_argument('--max_epoch', dest='max_epoch', default=50000, type=int)

parser.add_argument('--w_vgg', dest='w_vgg', default=0.001, type=float)
parser.add_argument('--w_tv', dest='w_tv', default=0.1, type=float)
parser.add_argument('--w_cos', dest='w_cos', default=1.0, type=float)

args = parser.parse_args()
print(args)

# Setting
HEIGHT = 128
WIDTH = 128
CHANNEL = 3
TFRECORD_PATH = glob('tfrecords/*.tfrecords')
CHECKPOINT_DIR = 'checkpoints/model_%s' % args.affix
NUM_OF_DATA = 142080

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def main():

    if args.mode == 'train':
        Train=train.Train(affix=args.affix, saved_iter=args.saved_iter, size=[HEIGHT, WIDTH, CHANNEL], batch_size=args.batch_size,
                          learning_rate=args.lr, max_epoch=args.max_epoch, tfrecord_path=TFRECORD_PATH, 
                          checkpoint_dir=CHECKPOINT_DIR, num_of_data=NUM_OF_DATA, step_disp = args.step_disp,
                          w_vgg=args.w_vgg, w_tv=args.w_tv, w_cos=args.w_cos)
        Train.train()
 
    elif args.mode == 'test':
        Test=test.Test(affix=args.affix,
                        saved_iter=args.saved_iter, 
                        input_path="dataset/test",
                        output_path="results/model_%s" % args.affix,
                        model_path=CHECKPOINT_DIR)
        Test.test()

    else:
        raise Exception('Mode should be train or test')

if __name__=='__main__':
    main()