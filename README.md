# A Simple Matlab Neural Network
Description: A very simple and basic (fully connected) Neural Network Toolbox for Matlab. Read the comments in the individual scripts, functions, and run `NNdemo.m` for an idea of how to use it. (NN stands for Neural Network)

I created this in order to understand neural networks better. It's very basic, nothing fancy, with the main functions: `ConstructNN.m`, `RunNN.m`, `TrainNN.m` (`TrainNNCPU` and `TrainNNGPU` are deprecated). The matrices in the NN are fully accessible for you to observe and perform surgery on (what I did for my project) so that you can train networks separately, splice, merge, etc.

##### `ConstructNN.m`
Creates an object (which is the neural network) and has the matrices which keeps track of stuff. Biases are initialised to 0, while weights are initialised to a Normal Gaussian Distribution, i.e. N(0,1), after which each series of weights entering a node is normalised in the L2 norm. This creates a good starting point.

##### `RunNN.m`
Just runs the neural network on a single input (row vector), gives the result (in the NN object under `NN.output`). (It will throw an error if you give it more than 1 row of input, i.e. more than 1 set of inputs.)

##### `TrainNN.m`
Implements stochastic gradient descent, also known as online training. It does backpropagation through the whole set of data you give it (so each time you run it is one epoch). It can train on the GPU if you set the flag and have the Parallel Computing Toolbox. L2 regularisation/weight-decay is included. (mini batches have not been implemented)

### Activation Functions
The Leaky ReLU function and the Sigmoid functions are provided. Leaky ReLU is recommended. Using the Sigmoid function will cause the network to run much slower. (The Leaky ReLU is used because the pure ReLU function has a disadvantage in that it can never be "revived" when "turned off" due to its zero gradient.)

### Cost Function
The commonly used Quadratic Cost Function is provided, but not the other commonly used cross-entropy function. If you use the ReLU, though, you are unlikely to need the cross-entropy function.
Note: The derivative of the cost function is meant to be added on during back-propagation, i.e. if your desired output is higher than the actual network output, the gradient should be positive, and vice versa. (Bear this in mind when you're creating your own cost functions)

## Dependencies
(Optional) Parallel Computing Toolbox - This is needed for training on the GPU. However, for neural networks up to around 200-300 nodes in maximum width, I've found that training on the CPU is still faster than the GPU.

## Things to be added
- A full training script that for each epoch randomly pulls out a small batch of data (from the whole set) to train on, and then loop over to pull another random small batch of data.

## Things that might be added
- Altering `TrainNN.m` to have an option of mini-batch (and hence batch) training by including a parameter of batch size.