import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch

from PIL import Image as imp
from IPython.core.debugger import set_trace
import spectral
import tifffile as tif
import scipy.ndimage
import cv2

import torch.nn.functional as F

    
class ImageDataset_biopsies(Dataset):
    def __init__(self, root):
        # Update to handle the BSQfiles directory structure
        self.parent = root  # root should be the specific timestamped folder path
        self.files = os.listdir(self.parent)
        self.list_I = [f for f in self.files if f.endswith(('.bsq', '.hdr'))]

        if not self.list_I:
            raise ValueError("No BSQ/HDR files found in the specified directory")
        
    def __getitem__(self, index):
        # Get HDR file first, then find corresponding BSQ file
        if index >= len(self.list_I):
            index = index % len(self.list_I)
            
        hdr_file = next((f for f in self.list_I if f.endswith('.hdr')), None)
        if not hdr_file:
            raise ValueError("No HDR file found")
            
        bsq_file = hdr_file.replace(".hdr", ".bsq")
        if bsq_file not in self.list_I:
            raise ValueError("Corresponding BSQ file not found")
        
        # Open the image files
        name = self.parent
        image = spectral.envi.open(os.path.join(name, hdr_file), os.path.join(name, bsq_file))
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1))

        img = img[:,:,:].copy()
        img[img>4] = 4
        img[img<0] = 0
        img = img/1.5

        _,r, d = img.shape
        scale_factor = 2 / 0.69
        r, d = int(r * scale_factor), int(d * scale_factor)
        
        img_V = torch.zeros((1, 1, r, d))
        img = torch.from_numpy(img).float()

        return {'ir': img[:,:,:],'vis': img_V[0,:], 'name': os.path.basename(name)}

    def __len__(self):
        # Return 1 since we're only processing one set of files at a time
        return 1

