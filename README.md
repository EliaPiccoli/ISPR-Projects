# ISPR-projects

Projects for the Intelligent Systems for Pattern Recognition course @ University of Pisa

## Midterms

### Midterm 1 - _Image processing assignments_
Implement the convolution of a set of edge detection filters with an image and apply it to one face image and one tree image of your choice from the dataset. Implement Roberts, Prewitt and Sobel filters (see [here](https://www.cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter5.pdf), Section 5.2, for a reference) and compare the results (it is sufficient to do it visually). You should not use the library functions for performing the convolution or to generate the Sobel filter. Implement your own and show the code!

##

### Midterm 2 - _Image processing assignments_
Implement from scratch an RBM and apply it to [MNIST](http://yann.lecun.com/exdb/mnist/). The RBM should be implemented fully by you (both CD-1 training and inference steps) but you are free to use library functions for the rest (e.g. image loading and management, etc.).
- Train an RBM with 100 hidden neurons (single layer) on the MNIST data (use the training set split provided by the website).
- Use the trained RBM to encode all the images using the corresponding activation of the hidden neurons.
- Train a simple classifier (e.g. any simple classifier in scikit) to recognize the MNIST digits using as inputs their encoding obtained at step 2. Use the standard training/test split. Show the resulting confusion matrices (training and test) in your presentation.