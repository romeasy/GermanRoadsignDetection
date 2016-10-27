# cnn
A convolutional Neural Network to detect german road signs in images of different sizes.

## Use the network
This repository includes several successfull training procedures which yield about 92% accuracy.
If you are interested in using them, have a look at the postprocessing script.

## Conduct training
If you want to conduct training yourself or change the existing code, you are welcome to do so. Preprocessing contains code to
read the images from the official GTSRB. It also converts all images to a proper training and test set in which all images have
the same size, the same color layout (grayscale vs YUV) and are rotated and scaled randomly to increase the training data.
Several different training and test sets can be easily generated this way.
