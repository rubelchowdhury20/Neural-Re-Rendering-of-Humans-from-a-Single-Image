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
	if(args.freeze_epochs):
		config.FREEZE_EPOCHS = args.freeze_epochs
	if(args.unfreeze_epochs):
		config.UNFREEZE_EPOCHS = args.unfreeze_epochs

	# updating batch size
	if(args.batch_size):
		config.PARAMS["batch_size"] = args.batch_size

	# updating command line arguments to the ARGS variable
	config.args = args

	# calling required functions based on the input arguments
	if args.mode == "train":
		train.train(config)
	else:
		evaluate.evaluate(config)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--mode",
		type=str,
		default="train",
		help="evaluation mode or training mode")

	# arguments for training
	parser.add_argument(
		"--batch_size",
		type=int,
		default=8,
		help="the batch_size for training as well as for inference")
	parser.add_argument(
		"--freeze_epochs",
		type=int,
		default=1,
		help="the total number of epochs for which the initial few layers will be frozen")
	parser.add_argument(
		"--unfreeze_epochs",
		type=int,
		default=200,
		help="the total number of epochs for which the full network will be unfrozen")
	parser.add_argument(
		"--resume",
		type=bool,
		default=True,
		help="Flag to resume the training from where it was stopped")
	parser.add_argument(
		"--checkpoint_name",
		type=str,
		default="checkpoint.pth",
		help="the name of the checkpoint file where the weights will be saved")
	parser.add_argument(
		"--data_directory",
		type=str,
		default="/media/tensor/EXTDRIVE/projects/virtual-try-on/dataset/zalando_final/",
		help="path to the directory having images for training.")

	main(parser.parse_args())