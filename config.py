 
# Third party imports
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data preprocessing details
data_transforms = {
	'train': transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

texture_transform = transforms.Compose([
		transforms.Resize((320, 480), interpolation=0),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])


dense_transform = transforms.Compose([
		transforms.ToTensor()
	])

# params info
PARAMS = {'batch_size': 1,
			'shuffle': True,
			'num_workers': 16}


# network parameters




# command line arguments
args = {}