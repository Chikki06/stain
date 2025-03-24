# IR Image Processing Web Application

This project is a web application for processing infrared (IR) spectroscopy images. It allows users to upload BSQ/HDR file pairs, view them, select regions of interest, and generate stained versions of the selected regions.

## Project Structure

```
website/
├── files/                    # Upload and output directories
│   ├── BSQfiles/            # Uploaded BSQ/HDR file pairs
│   ├── pngs/                # Initial PNG conversions
│   ├── output-pngs/         # Processed PNG outputs
│   └── tiffs/              # Processed TIFF outputs
├── public/                  # Static web files
│   ├── index.html          # Upload interface
│   ├── success.html        # Image selection interface
│   ├── output.html         # Results display
│   ├── styles.css          # Styling
│   └── 404.html            # Error page
├── src/                    # Server and processing code
│   ├── server.js           # Express server
│   ├── convert_bsq_to_png.py  # BSQ to PNG converter
│   ├── testcrop.py         # Region processing script
│   ├── DRB.py             # Neural network model
│   └── saved_models/       # Trained model weights
└── README.md
```

## Prerequisites

- Node.js (v14 or higher)
- Python 3.6+ with the following packages:
  - numpy
  - torch
  - spectral
  - opencv-python
  - tifffile
  - Pillow

## Installation

1. Clone the repository
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Set up Python environment:
   ```bash
   conda env create -f environment.yml
   ```

## Usage

1. Start the server:
   ```bash
   npm start
   ```
2. Open http://localhost:3000 in your browser
3. Upload BSQ/HDR file pair
4. Select region of interest on the displayed image
5. Process the selection to generate stained output
6. Download results in PNG or TIFF format

## Features

- Large file upload support (up to 15GB)
- Interactive region selection
- Real-time processing status
- Multi-format output (PNG/TIFF)
- Processing time analytics
- Pyramid TIFF output for large images

## License

This project is licensed under the MIT License.