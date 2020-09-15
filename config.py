 
# Third party imports
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# command line arguments
ARGS = {}