# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:24:24 2021

@author: kf4
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:05:11 2020

@author: kf4
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:30:47 2019

@author: kf4
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets_testcrop import ImageDataset_biopsies
from DRB import AttGenerator_ResNet50_SkipCon
import time
import cv2
import tifffile
import requests  # Add this import
#importing only what is required from files, added some timing.

def convert_coordinates(opt, ir_shape):
    """
    Direct BSQ coordinate calculation: 
    For 100,100: (100/1000 * 35495) = 3549, (100/1000 * 26252) = 2625
    """
    # Get BSQ dimensions
    _, _, bsq_height, bsq_width = ir_shape
    x1, y1, x2, y2 = opt.coordinates
    bsq_x1 = int((x1 / 1000.0) * bsq_width)
    bsq_y1 = int((y1 / 1000.0) * bsq_height)
    bsq_x2 = int((x2 / 1000.0) * bsq_width)
    bsq_y2 = int((y2 / 1000.0) * bsq_height)
    
    return (bsq_x1, bsq_y1, bsq_x2, bsq_y2)

def main(): 
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=8500, help='number of epochs of training')
    parser.add_argument('--dataset_name', type=str, default="facades", help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.599, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=1, help='epoch from which to start lr decay')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=3000, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing BSQ files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for TIFF files')
    parser.add_argument('--coordinates', type=float, nargs=4, required=True, help='Coordinates (x1,y1,x2,y2) for cropping')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp for output file')
    parser.add_argument('--folder_name', type=str, required=True, help='Original folder name')
    parser.add_argument('--image_width', type=float, required=False, help='Width of the image in success.html')
    parser.add_argument('--image_height', type=float, required=False, help='Height of the image in success.html')
    parser.add_argument('--server_url', type=str, required=True, help='URL of the server to send times')
    opt = parser.parse_args()
    cuda = torch.cuda.is_available()
    netG = AttGenerator_ResNet50_SkipCon()

    device = torch.device("cuda:0" if cuda else "cpu")
    if cuda:
        netG.load_state_dict(torch.load('src/saved_models/20_net_G.pth'))
        netG = nn.DataParallel(netG)
        netG.to(device)
        torch.set_grad_enabled(False)
        netG = netG.eval()

        print("Let's use", torch.cuda.device_count(), "GPUs!")

    Tensor2 = torch.FloatTensor
    count = 0
    BATCH_SIZE = 2
    dataloader = DataLoader(ImageDataset_biopsies(opt.input_dir),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    model_time = 0
    tif_write_time = 0
    png_write_time = 0

    model_start_time = time.time()
    for i, batch in enumerate(dataloader):
        torch.cuda.empty_cache()
        ir = ((batch['ir'].type(Tensor2)) - .5) / .5
        vis = ((batch['vis'].type(Tensor2)) - .5) / .5
        
        fact = 1472
        scaling_factor = (0.69 / 2)

        # Store original coordinates before conversion
        orig_x1, orig_y1, orig_x2, orig_y2 = opt.coordinates
        
        # Convert coordinates and print BSQ crop region
        bsq_x1, bsq_y1, bsq_x2, bsq_y2 = convert_coordinates(opt, ir.shape)

        # Crop the IR and VIS images before any processing
        ir = ir[:, :, bsq_y1:bsq_y2, bsq_x1:bsq_x2]
        vis = vis[:, :, int(bsq_y1*(1/scaling_factor)):int(bsq_y2*(1/scaling_factor)), int(bsq_x1*(1/scaling_factor)):int(bsq_x2*(1/scaling_factor))]
      
        # Process only the cropped region
        _, bands, _, _ = ir.shape
        b, _, r, c = vis.shape
        patch_size = fact
        rn = np.ceil(r / patch_size)
        cn = np.ceil(c / patch_size)
        DATA = torch.zeros(BATCH_SIZE, 10, patch_size, patch_size, device=device)
        
        
        # Create output tensor sized for cropped region only
        fhr = torch.zeros(b, 3, int(rn) * patch_size + 1, int(cn) * patch_size + 1).cpu().numpy()
        
        if count >= 0:
            for kk in range(b):
                patch_count = 0
                patches = []
                I = range(0, int(rn) * patch_size + 1, patch_size)
                J = range(0, int(cn) * patch_size + 1, patch_size)
                for ii in range(len(I) - 1):
                    for jj in range(len(J) - 1):
                        xx = ir[:, :, int(I[ii] * scaling_factor):int(I[ii + 1] * scaling_factor), int(J[jj] * scaling_factor):int(J[jj + 1] * scaling_factor)]
                        pad_rows = int(patch_size * scaling_factor) - xx.shape[2] if xx.shape[2] < int(patch_size * scaling_factor) else 0
                        pad_cols = int(patch_size * scaling_factor) - xx.shape[3] if xx.shape[3] < int(patch_size * scaling_factor) else 0

                        xx = F.pad(xx, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
                        xx = F.interpolate(xx, size=(patch_size, patch_size), mode='bicubic', align_corners=True)
                        vv = vis[:, :, int(I[ii]):int(I[ii + 1]), int(J[jj]):int(J[jj + 1])]
                        pad_rows = patch_size - vv.shape[2] if vv.shape[2] < patch_size else 0
                        pad_cols = patch_size - vv.shape[3] if vv.shape[3] < patch_size else 0

                        vv = F.pad(vv, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

                        DATA[patch_count, :bands, :, :] = xx
                        patches.append((ii, jj))
                        patch_count += 1

                        if patch_count == BATCH_SIZE:  # Process the batch
                            start_time = time.time()
                            
                            with autocast('cuda'):
                                output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5
                            output = output.cpu().numpy()
                            
                            for idx, (pi, pj) in enumerate(patches):
                                fhr[kk, :, int(I[pi]):int(I[pi + 1]), int(J[pj]):int(J[pj + 1])] = output[idx]

                            patch_count = 0
                            patches = []
                
                
                if patch_count > 0:  # Process remaining patches
                    with autocast('cuda'):
                        output = netG(DATA[:patch_count]).squeeze() * 0.5 + 0.5
                    output = output.cpu().numpy()
                    for idx, (pi, pj) in enumerate(patches):
                        fhr[kk, :, int(I[pi]):int(I[pi + 1]), int(J[pj]):int(J[pj + 1])] = output[idx]

        # Final output should now contain only the cropped region
        fhr = np.uint8(np.transpose(fhr, (0, 2, 3, 1)) * 256)
        fhr = fhr[0, :r, :c, :]  # Trim to exact crop size
        model_time = time.time() - model_start_time
        
        # Use cropped coordinates for naming
        output_base = f"{opt.folder_name}_{opt.timestamp}_{int(orig_x1)}_{int(orig_y1)}_{int(orig_x2)}_{int(orig_y2)}"
        
        # Create pyramid levels
        tif_start_time = time.time()
        pyramid_levels = [fhr]  # Start with original image
        
        # Only create pyramid levels if the image is large enough
        for i in range(1, 4):
            scale = 2 ** i

            resized = cv2.resize(fhr, (fhr.shape[1] // scale, fhr.shape[0] // scale))
            pyramid_levels.append(resized)

        # Write TIFF file
        aa = os.path.join(opt.output_dir, f"{output_base}.tif")
        with tifffile.TiffWriter(aa, bigtiff=True) as tiff:
            for level in pyramid_levels:
                tiff.write(level, photometric='rgb', compression='jpeg', tile=(256, 256), metadata={'axes': 'YXS'})
        tif_write_time = time.time() - tif_start_time

        # Create PNG version with updated naming convention
        png_start_time = time.time()
        png_output_dir = os.path.join(opt.output_dir, '../output-pngs')
        os.makedirs(png_output_dir, exist_ok=True)
        png_path = os.path.join(png_output_dir, f"{output_base}.png")
        cv2.imwrite(png_path, cv2.cvtColor(fhr, cv2.COLOR_RGB2BGR))
        png_write_time = time.time() - png_start_time
        
        # Send times to server
        times_data = {
            'model_time': model_time,
            'tiff_time': tif_write_time,
            'png_time': png_write_time
        }
        try:
            response = requests.post(f"{opt.server_url}/update-times", json=times_data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send times to server: {e}")
        
        print(f"MODEL:{model_time:.2f}")
        print(f"TIFF:{aa}")
        print(f"TIFF_TIME:{tif_write_time:.2f}")
        print(f"PNG:{png_path}")
        print(f"PNG_TIME:{png_write_time:.2f}")
if __name__ == '__main__':
    main()
