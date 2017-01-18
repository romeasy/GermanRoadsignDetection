# cnn
A convolutional Neural Network to detect german road signs in images of different sizes.
The networks layout is similar to the one described in 
The neural network features the following methods:
* Usage of an advanced optimizer (ADAM algorithm for gradient descent)
* Convolutions to speed up training

## Installation
Conducting training of your own using the convolutional neural network from our repository requires different software:
* Python
* Ipython notebook (for preprocessing and postprocessing scripts)
* Tensorflow (for training the cNN on graphics cards)
* Matplotlib (for plotting)

Installation can be done easily with pip by
```bash
pip install tensorflow matplotlib ipython
```
and ipython/jupyter notebook can be installed using the following guide http://jupyter.org/.

## Use the network
This repository includes several successfull training procedures which yield about 99% accuracy.
If you are interested in using them, have a look at the postprocessing script.

## Conduct training
If you want to conduct training yourself or change the existing code, you are welcome to do so. Preprocessing contains code to
read the images from the official GTSRB website. It also converts all images to a proper training and test set in which all images have
the same size, the same color layout (grayscale or YUV) and are rotated and scaled randomly to increase the training data size.
Several different training and test sets can be easily generated this way.
