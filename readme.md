# Object Tracking and Image Classification 

This project utilizes OpenCV to perform object tracking and image classification in real-time using a pre-trained DenseNet-121 model and the CSRT tracking algorithm.

![demo](https://github.com/chenyu-chang/object_traking_opencv/blob/master/video.gif)

## Features

- Real-time object tracking with CSRT (Channel and Spatial Reliability Tracker)
- Image classification using DenseNet-121, a convolutional neural network model
- Automatic extraction and classification of objects within a user-defined region of interest

## Prerequisites

Before you run this project, you must have the following installed:
- Python 3.6 or higher
- opencv-python
- opencv-contrib-python
- NumPy

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/chenyu-chang/object_traking_opencv
```

Ensure you have the required packages:

```bash
pip install numpy opencv-python opencv-contrib-python
```

## Usage

Run the script using Python:

```bash
python object_tracker.py
```

Upon running the script, a video window will open. Use the mouse to select a region of interest (ROI) for tracking and classification. The tracker will then initialize and begin tracking the object, with its predicted class displayed on the screen.

## Model Files

The DenseNet-121 model requires two files:
- `DenseNet_121.prototxt`: The architecture of the CNN.
- `DenseNet_121.caffemodel`: The pre-trained weights.

Make sure to download these files and place them in the `densenet_121` directory before running the script. The `synset_words.txt` file containing class labels should also be in this directory. You can download by link: [download](https://drive.google.com/file/d/1GsREyqXtwi_hhAf9ruivN19zjhsaLYBG/view?usp=drive_link)

## Additional Notes

- The ROI selection window can be closed by pressing the 'Esc' key.
- Tracking and classification continue in real-time; to exit, press the 'Esc' key in the main tracking window.

## Acknowledgements

- DenseNet-121 model creators and maintainers.
- OpenCV contributors and maintainers.