import argparse
import os
import torch

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', type=str, default="MPL", help='the name of the experiment')
        self.parser.add_argument('--epoch_start', type=int, default=0, help='epoch to start training from')
        self.parser.add_argument('--epoch_num', type=int, default=200, help='number of epochs of training')
        self.parser.add_argument('--data_root', type=str, default='./dataset/train_data', help='dir of the dataset')
        self.parser.add_argument('--dataset_name', type=str, default="1204", help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=0.00001, help='adam: learning rate')
        self.parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
        self.parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
        self.parser.add_argument('--img_height', type=int, default=160, help='size of image height')
        self.parser.add_argument('--img_width', type=int, default=160, help='size of image width')
        self.parser.add_argument('--in_channels', type=int, default=3, help='number of input image channels')
        self.parser.add_argument('--out_channels', type=int, default=3, help='number of output image channels')
        self.parser.add_argument('--sample_interval', type=int, default=200, help='interval between sampling of images from generators')
        self.parser.add_argument('--n_D_layers', type=int, default=3, help='used to decision the patch_size in D-net, should less than 8')
        self.parser.add_argument('--img_result_dir', type=str, default='result_images', help=' where to save the result images')
        self.parser.add_argument('--model_result_dir', type=str, default='/saved_models', help=' where to save the checkpoints')

    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs('log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.img_result_dir), exist_ok=True)
        os.makedirs('log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.model_result_dir), exist_ok=True)

        print('------------ Options -------------')
        with open("./log/%s-%s/args.log" % (args.exp_name,  args.dataset_name) ,"w") as args_log:
            for k, v in sorted(vars(args).items()):
                print('%s: %s ' % (str(k), str(v)))
                args_log.write('%s: %s \n' % (str(k), str(v)))

        print('-------------- End ----------------')



        self.args = args
        return self.args