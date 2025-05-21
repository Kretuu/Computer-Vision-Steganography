# Image Tampering Detection with Digital Watermarking

This application allows you to embed invisible watermarks into images and later detect if the images have been tampered with. It uses SIFT (Scale-Invariant Feature Transform) keypoints and LSB (Least Significant Bit) steganography to embed watermarks in a way that's resistant to various image manipulations.

## Features

- Embed invisible digital watermarks into images
- Recover watermarks from images
- Detect image tampering and visualize tampered areas
- Side-by-side comparison of original and processed images
- Save watermarked images and recovered watermarks

## Prerequisites

### System Requirements

- Python 3.7 or higher
- OpenCV 4.5 or higher
- PyQt5
- NumPy

### Required Python Packages

```
opencv-python>=4.5.0
PyQt5>=5.15.0
numpy>=1.19.0
```


## Installation

1. Create and activate a virtual environment (recommended):
```shell script
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```


2. Install the required packages:
```shell script
pip install -r requirements.txt
```


## Running the Application

1. Make sure you're in the project's root directory
2. Run the main Python script:
```shell script
python main.py
```

   
3. If you encounter issues on Linux, make sure the XCB platform plugin is properly set:
```shell script
export QT_QPA_PLATFORM=xcb
python main.py
```


## Usage Guide

### Basic Workflow

1. **Import Image**: Click "Import Image" to load the image you want to work with.

2. **Import Watermark**: Click "Import Watermark" to load a watermark image. For best results, the watermark should be a simple binary image (black and white).

3. **Embed Watermark**: Click "Embed Watermark" to embed the watermark into the image. This process is invisible to the naked eye.

4. **Save Processed Image**: Click "Save Processed Image" to save the watermarked image to your computer.

### Tampering Detection

1. **Load a Watermarked Image**: Import an image that has been previously watermarked.

2. **Import the Original Watermark**: Import the same watermark that was used during embedding.

3. **Detect Tampering**: Click "Detect Tampering" to check if the image has been modified since the watermark was embedded.

4. If tampering is detected, the application will highlight the tampered areas with red circles.

### Watermark Recovery

1. **Load a Watermarked Image**: Import an image that has been previously watermarked.

2. **Recover Watermark**: Click "Recover Watermark" to extract the embedded watermark from the image.

3. **Save Watermark**: Click "Save Watermark" to save the recovered watermark to your computer.

## Troubleshooting

- **OpenCV Error**: If you encounter errors related to OpenCV, make sure you have installed the correct version (4.5+).
- **PyQt5 Display Issues**: If the GUI doesn't display properly, try setting the `QT_QPA_PLATFORM` environment variable as mentioned in the "Running the Application" section.
- **Watermark Embedding Failure**: While the theoretical maximum watermark size is 255×255 pixels, in practice, much smaller watermarks (around 48×48 pixels) work best due to the limited number of SIFT keypoints available in most images. Choose carrier images with rich textures and distinctive features to maximize available SIFT keypoints. High-contrast, simple watermark designs are more effectively embedded and recovered.
