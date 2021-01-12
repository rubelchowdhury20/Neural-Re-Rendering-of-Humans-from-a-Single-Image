# Standard library imports
import os
import glob
import argparse

# Third party library imports
import torch
import torch.nn as nn
from torchvision import models

# Local imports
import config
import evaluate
import train

def main(args):
	# updating all the global variables based on the input arguments

	# updating batch size
	if(args.batch_size):
		config.PARAMS["batch_size"] = args.batch_size


	# taking care of gpu ids
	str_ids = args.gpu_ids.split(',')
	args.gpu_ids = []
	for str_id in str_ids:
		id = int(str_id)
		if id >= 0:
			args.gpu_ids.append(id)


	# updating command line arguments to the ARGS variable
	config.args = args

	# calling required functions based on the input arguments
	if args.is_train:
		train.train(config)
	else:
		evaluate.evaluate(config)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--is_train', action='store_false', help='continue training: load the latest model')

	# project parameters
	parser.add_argument('--name', type=str, default='render', help='name of the experiment. It decides where to store samples and models')        
	parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')


	# network parameters
	parser.add_argument("--batch_size", type=int, default=1, help="the batch_size for training as well as for inference")
	parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
	parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
	parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
	parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	parser.add_argument('--niter', type=int, default=5, help='# of iter at starting learning rate')
	parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')

	parser.add_argument("--data_directory", type=str, default="/media/rainier/rubel/projects/virtual-try-on/dataset/lip_dataset/", help="path to the directory having images for training.")
	# parser.add_argument("--data_directory", type=str, default="/media/tensor/EXTDRIVE/projects/virtual-try-on/dataset/zalando_final/", help="path to the directory having images for training.")	
	# parser.add_argument("--data_directory", type=str, default="/media/tensor/EXTDRIVE/projects/virtual-try-on/dataset/lip_mpv_dataset/", help="path to the directory having images for training.")
	parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
	parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
	parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')


	# for displays
	parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
	parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
	parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
	parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
	parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
	parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
	parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')


	# for feature net
	parser.add_argument('--feature_depth', type=int, default=2, help="# of downsampling blocks in the feature net architecture")
	parser.add_argument('--feature_output_nc', type=int, default=16, help="# of channels of feature net output")


	# for generator
	# parser.add_argument('--netG_input_nc', type=int, default=22, help="# of input channels to the generator")
	parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
	parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
	parser.add_argument('--n_downsample_global', type=int, default=2, help='number of downsampling layers in netG') 
	parser.add_argument('--n_blocks_global', type=int, default=3, help='number of residual blocks in the global generator network')
	parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
	parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
	parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')


	# for discriminators        
	parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
	parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
	parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
	# parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
	parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
	parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
	parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
	parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')


	# loss arguments
	parser.add_argument('--lambda_tex', type=float, default=1.0, help='lambda value for feature/texture loss in total loss')
	parser.add_argument('--lambda_adv', type=float, default=1.0, help='lambda value for adversarial loss in total loss')
	parser.add_argument('--lambda_vgg', type=float, default=10.0, help='lambda value for vgg loss in total loss')    

	main(parser.parse_args())