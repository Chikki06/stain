import os
import numpy as np
import spectral
from PIL import Image
import json
import cv2

def convert_bsq_to_png(bsq_dir, output_dir, timestamp):
    try:
        print(f"Starting conversion for directory: {bsq_dir}")

        files = os.listdir(bsq_dir)
        hdr_file = [f for f in files if f.endswith('.hdr')][0]
        bsq_file = hdr_file.replace(".hdr", ".bsq")

        image = spectral.envi.open(os.path.join(bsq_dir, hdr_file), os.path.join(bsq_dir, bsq_file))
        img = np.array(image.load())
        img = np.transpose(img, (2, 0, 1))

        img[img > 4] = 4
        img[img < 0] = 0
        img = (img / 4.0 * 255).astype(np.uint8)

        # Store original dimensions
        orig_dimensions = {
            'height': img.shape[1],
            'width': img.shape[2]
        }

        # Convert to RGB by selecting three bands
        img_rgb = img[:3, :, :].transpose(1, 2, 0)

        # Calculate dimensions for display (max 1200px width while maintaining aspect ratio)
        max_display_width = 1200
        bsq_width = img_rgb.shape[1]  # 35495
        bsq_height = img_rgb.shape[0]  # 26252
        
        scale_factor = max_display_width / bsq_width
        display_dimensions = {
            'height': int(bsq_height * scale_factor),
            'width': int(bsq_width * scale_factor)
        }

        # Create display version
        display_img = cv2.resize(img_rgb, (display_dimensions['width'], display_dimensions['height']))

        # Save metadata with exact dimensions and scaling
        metadata = {
            'original': {
                'width': bsq_width,
                'height': bsq_height
            },
            'display': display_dimensions,
            'scale_factor': scale_factor,
            'px_to_bsq_multiplier': 1/scale_factor  # Add this for coordinate conversion
        }
        
        metadata_filename = f"{os.path.basename(bsq_file)[:-4]}_{timestamp}_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Save display version as PNG
        output_filename = f"{os.path.basename(bsq_file)[:-4]}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)
        Image.fromarray(display_img).save(output_path, optimize=True, quality=85)

        print(f"METADATA:{metadata_path}")  # Add this line to output metadata path
        print(f"PNG:{output_path}")
        print("Conversion completed successfully.")
        return output_path

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    bsq_dir = sys.argv[1]
    output_dir = sys.argv[2]
    timestamp = sys.argv[3]
    output_path = convert_bsq_to_png(bsq_dir, output_dir, timestamp)
    if output_path:
        print(output_path)
