# A Simple Matlab Neural Network
Description: A very simple and basic Neural Network Toolbox for Matlab. Read the individual scripts, functions, and run NNdemo.m for an idea of how to use it.

I created this in order to understand neural networks better. It's very basic, in the sense that it has no optimisation at all and only has basic functions: ConstructNN, RunNN, TrainNN (split into TrainNNCPU, and TrainNNGPU).

######ConstructNN
Creates an object (which is the neural network) and has the matrices which keeps track of stuff.

######RunNN
Just runs the neural network on a single input (row vector).

######TrainNN
Does a backpropagation once through all the data you give it (so each time you run it is one epoch). TrainNNCPU uses the CPU, and TrainNNGPU uses the GPU.

The activation function used in all of these scripts is the leaky ReLU function, but the Sigmoid activation function is also included, and you can just replace all instances of the ReLU functions in the scripts with the Sigmoid one. It will run much slower though.

##Things to be added
- A full training script that for each epoch randomly pulls out a small batch of data (from the whole set) to train on, and then loop over to pull another random small batch of data.