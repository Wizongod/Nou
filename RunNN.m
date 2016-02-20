function NN = RunNN(NN,input)

% Runs the input given through neural network.
% Access the output through NN.output

% Given a Neural Network object, NN, calculate its output based on an input
% vector I.

if ~isrow(input)
    error('Error in RunNN. Argument I must be a row vector');
end

% this part pads the input vector with zeros if it doesn't match the matrix
% dimensions
if size(input,2) ~= size(NN.x(:,:,1),2)
    input(size(input,2):size(NN.x(:,:,1),2)) = 0;
end

NN.x(:,:,1) = input;

% Compute the outputs of the Neural Network
for L = 1:NN.layers-1
    NN.x(:,:,L+1) = NN.afunc(NN.x(:,:,L),NN.w(:,:,L),NN.b(:,:,L));
end

NN.output = NN.x(1,1:NN.outputs,NN.layers);

end