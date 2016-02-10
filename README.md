# A Simple Matlab Neural Network
Description: A very simple and basic Neural Network Toolbox for Matlab. Read the individual scripts, functions, and run `NNdemo.m` for an idea of how to use it.

I created this in order to understand neural networks better. It's very basic, nothing fancy, with the functions: `ConstructNN.m`, `RunNN.m`, `TrainNN` (split into `TrainNNCPU.m`, and `TrainNNGPU.m`).

######`ConstructNN.m`
Creates an object (which is the neural network) and has the matrices which keeps track of stuff. Biases are initialised to 0, while weights are initialised to a Normal Gaussian Distribution ~G(0,1), after which they are then normalised again in the L2 norm. This creates a good starting point.

######`RunNN.m`
Just runs the neural network on a single input (row vector), gives the result (in the NN object).

######`TrainNN`
Implements stochastic gradient descent, also known as online training. It does backpropagation through the whole set of data you give it (so each time you run it is one epoch). `TrainNNCPU.m` uses the CPU, and `TrainNNGPU.m` uses the GPU. Regularisation is included. (mini batches have not been implemented)

######Activation Function
The activation function used in all of these scripts is the leaky ReLU function, but the Sigmoid activation function is also included, and you can just replace all instances of the ReLU functions in the scripts with the Sigmoid one. It will run much slower though. The pure ReLU function has a disadvantage in that it can never be "revived" when completely "turned off", hence leaky ReLU is used.

######Cost Function
The cost function is the standard quadratic cost function. I did not separate it out from TrainNN. This might be done so in the future. If you use the ReLU, though, you are unlikely to need the cross-entropy function.

##Things to be added
- A full training script that for each epoch randomly pulls out a small batch of data (from the whole set) to train on, and then loop over to pull another random small batch of data.

##Things that might be added
- Separating out the cost function to a separate file.
- Providing also the cross-entropy cost function.
- Altering `TrainNN` scripts to have an option of mini-batch (and hence batch) training by including a parameter of batch size.
- Merging both `TrainNN` scripts into one with a flag for CPU or GPU selection