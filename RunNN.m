function NN = RunNN(NN,I)

% Runs the input given through neural network.
% Access the output through NN.output

% Given a Neural Network object, NN, calculate its output based on an input
% vector I.

% this part pads the input vector with zeros if it doesn't match the matrix
% dimensions
if ~isrow(I)
    error('Error in RunNN. Argument I must be a row vector');
end

if size(I,2) ~= size(NN.x(:,:,1),2)
    I(size(I,2):size(NN.x(:,:,1),2)) = 0;
end

NN.x(:,:,1) = I;

% Compute the outputs of the Neural Network
for L = 1:NN.layers-1
    NN.x(:,:,L+1) = ReLU(NN.x(:,:,L),NN.w(:,:,L),NN.b(:,:,L));
end

NN.output = NN.x(1,1:NN.outputs,NN.layers);

end